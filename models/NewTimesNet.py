import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


# -------------------------- 模块1：VMD时序分解（可微分实现） --------------------------
class VMD(nn.Module):
    """
    变分模态分解（Variational Mode Decomposition），用于将时序序列分解为趋势、周期、残差成分
    参考原始算法：https://doi.org/10.1109/TSP.2013.2288675
    """
    def __init__(self, k: int = 3, alpha: float = 2000.0, tau: float = 0.001, max_iter: int = 100):
        super().__init__()
        self.k = k  # 分解的模态数（默认3：趋势+周期+残差）
        self.alpha = alpha  # 惩罚参数（平衡重构误差和带宽约束）
        self.tau = tau  # 对偶上升步长
        self.max_iter = max_iter  # 最大迭代次数
        self.eps = 1e-7  # 数值稳定性参数

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        输入：x -> [B, L, D] （批次大小，时序长度，特征维度）
        输出：x_trend, x_period, x_res -> 均为[B, L, D]，分别对应趋势、周期、残差成分
        """
        B, L, D = x.shape
        device = x.device

        # 初始化变量（对每个样本+特征维度独立分解）
        u = torch.zeros(B, self.k, L, D, device=device)  # 各模态输出
        omega = torch.zeros(B, self.k, D, device=device)  # 各模态中心频率
        lambda_ = torch.zeros(B, L, D, device=device)     # 对偶变量

        # 傅里叶变换相关初始化（快速傅里叶变换，中心化处理）
        t = torch.arange(1, L+1, device=device).unsqueeze(0).unsqueeze(-1)  # [1, L, 1]
        f = torch.fft.fftfreq(L, d=1.0, device=device).unsqueeze(0).unsqueeze(-1)  # [1, L, 1]
        f_shifted = torch.fft.fftshift(f)  # 频率轴中心化
        f_shifted = f_shifted.expand(B, -1, D)  # [B, L, D]

        # VMD迭代过程（可微分）
        x_hat = torch.fft.fftshift(torch.fft.fft(x, dim=1), dim=1)  # 输入的傅里叶变换（中心化）
        for _ in range(self.max_iter):
            # 1. 更新各模态u（傅里叶域）
            for k_idx in range(self.k):
                # 计算当前模态的分子：(x_hat - 平均其他模态 + lambda_/2)
                sum_other = torch.sum(u[:, [k_ for k_ in range(self.k) if k_ != k_idx]], dim=1)  # [B, L, D]
                sum_other_hat = torch.fft.fftshift(torch.fft.fft(sum_other, dim=1), dim=1)  # 其他模态的傅里叶变换
                numerator = x_hat - sum_other_hat + lambda_ / 2  # [B, L, D]

                # 计算当前模态的分母：1 + 2*alpha*(f - omega[:,k_idx])^2
                omega_k = omega[:, k_idx].unsqueeze(1)  # [B, 1, D]
                denominator = 1 + 2 * self.alpha * (f_shifted - omega_k) ** 2  # [B, L, D]

                # 更新当前模态的傅里叶域表示
                u_hat_k = numerator / denominator  # [B, L, D]
                # 逆傅里叶变换回时域
                u[:, k_idx] = torch.fft.ifft(torch.fft.ifftshift(u_hat_k, dim=1), dim=1).real  # [B, L, D]

            # 2. 更新中心频率omega（按能量加权）
            for k_idx in range(self.k):
                u_k = u[:, k_idx]  # [B, L, D]
                u_k_hat = torch.fft.fftshift(torch.fft.fft(u_k, dim=1), dim=1)  # [B, L, D]
                # 能量加权计算中心频率（仅正频率部分，避免冗余）
                f_pos = f_shifted[f_shifted >= 0]  # 正频率点
                idx_pos = f_shifted >= 0  # 正频率掩码
                numerator_omega = torch.sum(f_shifted[idx_pos] * torch.abs(u_k_hat[idx_pos]) ** 2, dim=1)  # [B, D]
                denominator_omega = torch.sum(torch.abs(u_k_hat[idx_pos]) ** 2, dim=1) + self.eps  # [B, D]
                omega[:, k_idx] = numerator_omega / denominator_omega  # [B, D]

            # 3. 更新对偶变量lambda（梯度上升）
            sum_u = torch.sum(u, dim=1)  # [B, L, D]
            lambda_ = lambda_ + self.tau * (x - sum_u)  # [B, L, D]

        # 按频率排序：低频（趋势）→ 中频（周期）→ 高频（残差）
        omega_mean = omega.mean(dim=2).squeeze(-1)  # [B, k]，各模态平均频率
        sorted_indices = torch.argsort(omega_mean, dim=1)  # [B, k]，按频率升序的索引

        # 对每个样本按频率排序模态，确保输出顺序固定为：趋势（0）、周期（1）、残差（2）
        u_sorted = torch.zeros_like(u)  # [B, k, L, D]
        for b in range(B):
            u_sorted[b] = u[b, sorted_indices[b]]

        # 返回三个成分（取前3个模态，若k>3则截断，k=3时正好对应）
        x_trend, x_period, x_res = u_sorted[:, 0], u_sorted[:, 1], u_sorted[:, 2]
        return x_trend, x_period, x_res


# -------------------------- 模块2：CNN增强模块（深度可分离卷积） --------------------------
class CNNEnhance(nn.Module):
    """
    基于深度可分离卷积的Patch特征增强模块，补全Patch内部局部依赖
    输入：[B*nvars, patch_num, d_model] → 输出：[B*nvars, patch_num, d_model]
    """
    def __init__(self, d_model: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2  # 保持patch_num不变

        # 深度可分离卷积：Depthwise（逐通道卷积） + Pointwise（1x1卷积）
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=padding,
            groups=d_model,  # 逐通道卷积，每个通道单独处理
            bias=False
        )
        self.pointwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1,  # 1x1卷积，融合通道信息
            bias=False
        )

        # 激活与正则化
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)  # 对d_model维度归一化

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [B*nvars, patch_num, d_model] → 先转置为Conv1d所需的[B*nvars, d_model, patch_num]
        """
        residual = x  # 残差连接
        # 维度转换：[B*nvars, patch_num, d_model] → [B*nvars, d_model, patch_num]
        x = x.transpose(1, 2)

        # 深度可分离卷积
        x = self.depthwise_conv(x)  # [B*nvars, d_model, patch_num]
        x = self.gelu(x)
        x = self.pointwise_conv(x)  # [B*nvars, d_model, patch_num]
        x = self.gelu(x)

        # 维度还原：[B*nvars, d_model, patch_num] → [B*nvars, patch_num, d_model]
        x = x.transpose(1, 2)
        x = self.dropout(x)

        # 残差连接 + LayerNorm
        x = self.norm(residual + x)
        return x


# -------------------------- 模块3：稀疏注意力模块 --------------------------
class SparseAttention(nn.Module):
    """
    稀疏注意力模块：保留top-T%的注意力权重，过滤冗余依赖
    继承FullAttention的接口设计，仅修改注意力权重计算逻辑
    """
    def __init__(self, mask_flag: bool = True, factor: int = 5, scale: float = None, 
                 attention_dropout: float = 0.1, output_attention: bool = False,
                 sparse_rate: float = 0.3):  # 新增：稀疏率（保留top-30%权重）
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.sparse_rate = sparse_rate  # 稀疏率（0~1，值越小越稀疏）

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, 
                attn_mask: Tensor = None, tau: float = None, delta: float = None) -> tuple[Tensor, Tensor]:
        B, L, H, E = queries.shape  # [B, patch_num, H, d_k]
        _, S, _, D = values.shape    # [B, patch_num, H, d_v]
        scale = self.scale or 1. / math.sqrt(E)

        # 1. 计算原始注意力分数（同FullAttention）
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)  # [B, H, L, S]

        # 2. 掩码处理（同FullAttention）
        if self.mask_flag:
            if attn_mask is None:
                from utils.masking import TriangularCausalMask
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -math.inf)

        # 3. 稀疏化：保留每个查询的top-T%权重，其余置为-∞（避免梯度消失）
        k = max(1, int(S * self.sparse_rate))  # 每个查询保留的键数量（至少1个）
        # 对每个批次、每个注意力头、每个查询，取top-k的分数索引
        topk_values, topk_indices = torch.topk(scores, k=k, dim=-1)  # [B, H, L, k]
        # 创建稀疏掩码：仅top-k位置保留原始分数，其余置-∞
        sparse_mask = torch.full_like(scores, -math.inf, device=scores.device)
        sparse_mask = sparse_mask.scatter(-1, topk_indices, topk_values)  # 填充top-k分数
        scores = sparse_mask  # 替换为稀疏分数

        # 4. 权重归一化与值加权（同FullAttention）
        A = self.dropout(torch.softmax(scale * scores, dim=-1))  # [B, H, L, S]
        V = torch.einsum("bhls,bshd->blhd", A, values)  # [B, L, H, D]

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


# -------------------------- 辅助模块 --------------------------
class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    改进版Decomp-CNN-PatchTST：集成VMD分解、CNN增强、稀疏注意力
    """

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride
        d_model = configs.d_model

        # -------------------------- 1. 初始化新增模块 --------------------------
        # VMD时序分解模块（默认分解为3个成分：趋势+周期+残差）
        self.vmd = VMD(
            k=getattr(configs, 'vmd_k', 3),  # 从configs获取，默认3
            alpha=getattr(configs, 'vmd_alpha', 2000.0),
            max_iter=getattr(configs, 'vmd_max_iter', 100)
        )

        # CNN增强模块（默认核大小3， dropout同configs）
        self.cnn_enhance = CNNEnhance(
            d_model=d_model,
            kernel_size=getattr(configs, 'cnn_kernel', 3),
            dropout=configs.dropout
        )

        # 稀疏注意力模块（默认稀疏率0.3， 其他参数同FullAttention）
        self.sparse_attention = SparseAttention(
            mask_flag=False,  # PatchTST默认不使用因果掩码
            factor=configs.factor,
            attention_dropout=configs.dropout,
            output_attention=False,
            sparse_rate=getattr(configs, 'sparse_rate', 0.3)  # 从configs获取，默认0.3
        )

        # 多成分融合权重（可学习，softmax确保权重和为1）
        self.alpha = nn.Parameter(torch.ones(1, device=configs.device))  # 趋势权重
        self.beta = nn.Parameter(torch.ones(1, device=configs.device))   # 周期权重
        self.gamma = nn.Parameter(torch.ones(1, device=configs.device))  # 残差权重

        # -------------------------- 2. 原有模块 --------------------------
        # Patch Embedding
        self.patch_embedding = PatchEmbedding(
            d_model, patch_len, stride, padding, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    # 将FullAttention替换为SparseAttention
                    AttentionLayer(
                        self.sparse_attention,  # 新增的稀疏注意力
                        d_model, 
                        configs.n_heads
                    ),
                    d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        )

        # Prediction Head
        self.head_nf = d_model * int((configs.seq_len - patch_len) / stride + 2)
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)
        elif self.task_name in ['imputation', 'anomaly_detection']:
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in, configs.num_class)

    # -------------------------- 辅助函数：单成分处理（归一化+Patch+CNN+Encoder） --------------------------
    def _process_single_component(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        处理单个时序成分（趋势/周期/残差）：归一化 → Patch Embedding → CNN增强 → Encoder
        输入：x → [B, L, D]（单成分时序）
        输出：enc_out → [B, nvars, d_model, patch_num]（Encoder输出）, means, stdev（归一化参数）
        """
        B, L, D = x.shape
        nvars = D  # 特征维度即变量数

        # 1. 归一化（同原有逻辑，单成分独立归一化）
        means = x.mean(1, keepdim=True).detach()  # [B, 1, D]
        x_norm = x - means
        stdev = torch.sqrt(torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)  # [B, 1, D]
        x_norm /= stdev

        # 2. Patch Embedding（同原有逻辑）
        x_norm = x_norm.permute(0, 2, 1)  # [B, D, L] → 适配PatchEmbedding输入
        patch_out, _ = self.patch_embedding(x_norm)  # [B*nvars, patch_num, d_model]

        # 3. CNN增强（新增步骤）
        cnn_out = self.cnn_enhance(patch_out)  # [B*nvars, patch_num, d_model]

        # 4. Encoder（同原有逻辑，使用稀疏注意力）
        enc_out, _ = self.encoder(cnn_out)  # [B*nvars, patch_num, d_model]

        # 5. 维度重塑（同原有逻辑）
        enc_out = enc_out.reshape(-1, nvars, enc_out.shape[-2], enc_out.shape[-1])  # [B, nvars, patch_num, d_model]
        enc_out = enc_out.permute(0, 1, 3, 2)  # [B, nvars, d_model, patch_num] → 适配Head输入

        return enc_out, means, stdev

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 1. VMD时序分解（新增步骤：将原始输入分解为3个成分）
        x_trend, x_period, x_res = self.vmd(x_enc)  # 均为[B, L, D]

        # 2. 分别处理每个成分（归一化+Patch+CNN+Encoder）
        enc_trend, mean_trend, stdev_trend = self._process_single_component(x_trend)
        enc_period, mean_period, stdev_period = self._process_single_component(x_period)
        enc_res, mean_res, stdev_res = self._process_single_component(x_res)

        # 3. 多成分融合（可学习权重）
        weights = F.softmax(torch.cat([self.alpha, self.beta, self.gamma]), dim=0)  # [3]
        enc_total = weights[0] * enc_trend + weights[1] * enc_period + weights[2] * enc_res  # [B, nvars, d_model, patch_num]

        # 4. 预测头（同原有逻辑）
        dec_out = self.head(enc_total)  # [B, nvars, pred_len]
        dec_out = dec_out.permute(0, 2, 1)  # [B, pred_len, nvars]

        # 5. 反归一化（融合各成分的反归一化结果）
        dec_trend = dec_out * stdev_trend[:, 0, :].unsqueeze(1) + mean_trend[:, 0, :].unsqueeze(1)
        dec_period = dec_out * stdev_period[:, 0, :].unsqueeze(1) + mean_period[:, 0, :].unsqueeze(1)
        dec_res = dec_out * stdev_res[:, 0, :].unsqueeze(1) + mean_res[:, 0, :].unsqueeze(1)
        dec_out = weights[0] * dec_trend + weights[1] * dec_period + weights[2] * dec_res

        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # 1. VMD分解（忽略mask部分，用有效数据计算分解）
        x_enc_valid = x_enc * mask  # 仅保留有效数据
        x_trend, x_period, x_res = self.vmd(x_enc_valid)

        # 2. 单成分处理（归一化时仅用有效数据）
        def _impute_norm(x, mask):
            means = (x * mask).sum(1) / (mask.sum(1) + 1e-7)
            means = means.unsqueeze(1).detach()
            x_norm = (x - means) * mask
            stdev = torch.sqrt((x_norm ** 2).sum(1) / (mask.sum(1) + 1e-7) + 1e-5)
            stdev = stdev.unsqueeze(1).detach()
            x_norm /= stdev
            return x_norm, means, stdev

        # 处理每个成分
        x_trend_norm, mean_trend, stdev_trend = _impute_norm(x_trend, mask)
        x_period_norm, mean_period, stdev_period = _impute_norm(x_period, mask)
        x_res_norm, mean_res, stdev_res = _impute_norm(x_res, mask)

        # Patch+CNN+Encoder
        x_trend_norm = x_trend_norm.permute(0, 2, 1)
        x_period_norm = x_period_norm.permute(0, 2, 1)
        x_res_norm = x_res_norm.permute(0, 2, 1)

        patch_trend, _ = self.patch_embedding(x_trend_norm)
        patch_period, _ = self.patch_embedding(x_period_norm)
        patch_res, _ = self.patch_embedding(x_res_norm)

        cnn_trend = self.cnn_enhance(patch_trend)
        cnn_period = self.cnn_enhance(patch_period)
        cnn_res = self.cnn_enhance(patch_res)

        enc_trend, _ = self.encoder(cnn_trend)
        enc_period, _ = self.encoder(cnn_period)
        enc_res, _ = self.encoder(cnn_res)

        # 维度重塑与融合
        nvars = x_enc.shape[2]
        enc_trend = enc_trend.reshape(-1, nvars, enc_trend.shape[-2], enc_trend.shape[-1]).permute(0,1,3,2)
        enc_period = enc_period.reshape(-1, nvars, enc_period.shape[-2], enc_period.shape[-1]).permute(0,1,3,2)
        enc_res = enc_res.reshape(-1, nvars, enc_res.shape[-2], enc_res.shape[-1]).permute(0,1,3,2)

        weights = F.softmax(torch.cat([self.alpha, self.beta, self.gamma]), dim=0)
        enc_total = weights[0]*enc_trend + weights[1]*enc_period + weights[2]*enc_res

        # 预测与反归一化
        dec_out = self.head(enc_total).permute(0,2,1)
        dec_trend = dec_out * stdev_trend[:,0,:].unsqueeze(1) + mean_trend[:,0,:].unsqueeze(1)
        dec_period = dec_out * stdev_period[:,0,:].unsqueeze(1) + mean_period[:,0,:].unsqueeze(1)
        dec_res = dec_out * stdev_res[:,0,:].unsqueeze(1) + mean_res[:,0,:].unsqueeze(1)
        dec_out = weights[0]*dec_trend + weights[1]*dec_period + weights[2]*dec_res

        return dec_out

    def anomaly_detection(self, x_enc):
        # 逻辑同forecast，仅预测长度为seq_len
        x_trend, x_period, x_res = self.vmd(x_enc)
        enc_trend, mean_trend, stdev_trend = self._process_single_component(x_trend)
        enc_period, mean_period, stdev_period = self._process_single_component(x_period)
        enc_res, mean_res, stdev_res = self._process_single_component(x_res)

        weights = F.softmax(torch.cat([self.alpha, self.beta, self.gamma]), dim=0)
        enc_total = weights[0]*enc_trend + weights[1]*enc_period + weights[2]*enc_res

        dec_out = self.head(enc_total).permute(0,2,1)
        dec_trend = dec_out * stdev_trend[:,0,:].unsqueeze(1) + mean_trend[:,0,:].unsqueeze(1)
        dec_period = dec_out * stdev_period[:,0,:].unsqueeze(1) + mean_period[:,0,:].unsqueeze(1)
        dec_res = dec_out * stdev_res[:,0,:].unsqueeze(1) + mean_res[:,0,:].unsqueeze(1)
        dec_out = weights[0]*dec_trend + weights[1]*dec_period + weights[2]*dec_res

        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # 逻辑同forecast，仅输出为分类概率
        x_trend, x_period, x_res = self.vmd(x_enc)
        enc_trend, _, _ = self._process_single_component(x_trend)
        enc_period, _, _ = self._process_single_component(x_period)
        enc_res, _, _ = self._process_single_component(x_res)

        weights = F.softmax(torch.cat([self.alpha, self.beta, self.gamma]), dim=0)
        enc_total = weights[0]*enc_trend + weights[1]*enc_period + weights[2]*enc_res

        # 分类头（同原有逻辑）
        output = self.flatten(enc_total)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)

        return output

    # --------------------------  forward函数（保持不变） --------------------------
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None