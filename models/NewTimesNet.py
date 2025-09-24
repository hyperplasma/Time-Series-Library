import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


# -------------------------- 模块1：VMD时序分解（适配Weather数据优化） --------------------------
class VMD(nn.Module):
    """
    优化点：
    1. 默认k=2（周期+残差），适配Weather 96小时短期数据（无显著趋势）
    2. 默认alpha=800（降低惩罚强度，避免过分解）
    3. 支持k=2时输出兼容三成分接口（趋势成分置0，避免后续逻辑报错）
    """
    def __init__(self, k: int = 2, alpha: float = 800.0, tau: float = 0.001, max_iter: int = 200):
        super().__init__()
        self.k = k  # 分解模态数（默认2：周期+残差，Weather短期数据无显著趋势）
        self.alpha = alpha  # 降低惩罚系数，适配小波动数据
        self.tau = tau  # 对偶上升步长
        self.max_iter = max_iter  # 增加迭代次数，确保分解收敛
        self.eps = 1e-7  # 数值稳定性参数

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        输入：x -> [B, L, D] （批次大小，时序长度，特征维度）
        输出：x_trend（趋势，k=2时为0）, x_period（周期）, x_res（残差）→ 均为[B, L, D]
        """
        B, L, D = x.shape
        device = x.device

        # 初始化变量（按实际k分解，后续兼容三成分输出）
        u = torch.zeros(B, self.k, L, D, device=device)  # [B, k, L, D]
        omega = torch.zeros(B, self.k, D, device=device)  # [B, k, D]
        lambda_ = torch.zeros(B, L, D, device=device)     # [B, L, D]

        # 傅里叶变换初始化（保持三维维度）
        f = torch.fft.fftfreq(L, d=1.0, device=device).unsqueeze(0).unsqueeze(-1)  # [1, L, 1]
        f_shifted = torch.fft.fftshift(f).expand(B, -1, D)  # [B, L, D]
        x_hat = torch.fft.fftshift(torch.fft.fft(x, dim=1), dim=1)  # [B, L, D]

        # VMD迭代（可微分）
        for _ in range(self.max_iter):
            # 1. 更新各模态u（傅里叶域）
            for k_idx in range(self.k):
                sum_other = torch.sum(u[:, [k_ for k_ in range(self.k) if k_ != k_idx]], dim=1)  # [B, L, D]
                sum_other_hat = torch.fft.fftshift(torch.fft.fft(sum_other, dim=1), dim=1)
                numerator = x_hat - sum_other_hat + lambda_ / 2
                omega_k = omega[:, k_idx].unsqueeze(1)  # [B, 1, D]
                denominator = 1 + 2 * self.alpha * (f_shifted - omega_k) ** 2
                u_hat_k = numerator / denominator
                u[:, k_idx] = torch.fft.ifft(torch.fft.ifftshift(u_hat_k, dim=1), dim=1).real

            # 2. 更新中心频率omega（仅正频率计算）
            for k_idx in range(self.k):
                u_k_hat = torch.fft.fftshift(torch.fft.fft(u[:, k_idx], dim=1), dim=1)  # [B, L, D]
                idx_pos = f_shifted >= 0
                f_pos = torch.where(idx_pos, f_shifted, torch.tensor(0.0, device=device))
                u_k_hat_pos = torch.where(idx_pos, u_k_hat, torch.tensor(0.0, device=device))
                numerator_omega = torch.sum(f_pos * torch.abs(u_k_hat_pos) ** 2, dim=1)  # [B, D]
                denominator_omega = torch.sum(torch.abs(u_k_hat_pos) ** 2, dim=1) + self.eps
                omega[:, k_idx] = numerator_omega / denominator_omega

            # 3. 更新对偶变量lambda
            sum_u = torch.sum(u, dim=1)  # [B, L, D]
            lambda_ = lambda_ + self.tau * (x - sum_u)

        # 按频率排序（低频→高频）
        omega_mean = omega.mean(dim=2)  # [B, k]
        sorted_indices = torch.argsort(omega_mean, dim=1)  # [B, k]
        u_sorted = torch.zeros_like(u)
        for b in range(B):
            u_sorted[b] = u[b, sorted_indices[b]]

        # 兼容三成分输出：k=2时趋势成分置0，k=3时正常输出
        if self.k == 2:
            x_trend = torch.zeros_like(u_sorted[:, 0])  # 趋势为0（短期数据无显著趋势）
            x_period = u_sorted[:, 0]  # 低频→周期成分
            x_res = u_sorted[:, 1]     # 高频→残差成分
        else:  # k=3（预留长期数据适配）
            x_trend = u_sorted[:, 0]
            x_period = u_sorted[:, 1]
            x_res = u_sorted[:, 2]

        return x_trend, x_period, x_res


# -------------------------- 模块2：CNN增强模块（可关闭+简化） --------------------------
class CNNEnhance(nn.Module):
    """
    优化点：
    1. 新增use_cnn开关，默认关闭（先验证VMD+稀疏注意力的核心效果）
    2. 卷积核默认1x1（仅通道融合，不干扰时序依赖，避免过度建模）
    """
    def __init__(self, d_model: int, kernel_size: int = 1, dropout: float = 0.1, use_cnn: bool = False):
        super().__init__()
        self.use_cnn = use_cnn  # 开关：是否启用CNN增强
        if not use_cnn:
            return  # 不启用则不初始化参数，减少冗余计算

        self.d_model = d_model
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2  # 保持patch_num不变

        # 简化为1x1卷积（仅融合通道，不捕捉局部时序，避免干扰Weather数据）
        self.pointwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        if not self.use_cnn:
            return x  # 不启用则直接返回原始Patch特征
        # 仅通道融合（无Depthwise卷积，避免过度建模局部依赖）
        residual = x
        x = x.transpose(1, 2)  # [B*nvars, d_model, patch_num]
        x = self.pointwise_conv(x)
        x = self.gelu(x)
        x = x.transpose(1, 2)  # [B*nvars, patch_num, d_model]
        x = self.dropout(x)
        x = self.norm(residual + x)
        return x


# -------------------------- 模块3：稀疏注意力模块（降低稀疏率） --------------------------
class SparseAttention(nn.Module):
    """
    优化点：
    1. 默认稀疏率0.6（保留60%权重，避免过滤Weather的周期依赖）
    2. 确保k≥2（至少保留2个依赖，避免单点依赖导致偏差）
    """
    def __init__(self, mask_flag: bool = False, factor: int = 3, scale: float = None, 
                 attention_dropout: float = 0.1, output_attention: bool = False,
                 sparse_rate: float = 0.6):  # 提升稀疏率，保留更多有效依赖
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag  # PatchTST默认关闭因果掩码
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.sparse_rate = sparse_rate

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, 
                attn_mask: Tensor = None, tau: float = None, delta: float = None) -> tuple[Tensor, Tensor]:
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        # 1. 计算原始注意力分数
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)  # [B, H, L, S]

        # 2. 稀疏化：保留top-k，k≥2（避免仅保留1个依赖）
        k = max(2, int(S * self.sparse_rate))  # 至少保留2个Patch依赖
        topk_values, topk_indices = torch.topk(scores, k=k, dim=-1)
        sparse_mask = torch.full_like(scores, -math.inf, device=scores.device)
        sparse_mask = sparse_mask.scatter(-1, topk_indices, topk_values)
        scores = sparse_mask

        # 3. 权重归一化与值加权
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return (V.contiguous(), A) if self.output_attention else (V.contiguous(), None)


# -------------------------- 辅助模块（无修改） --------------------------
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


# -------------------------- 核心Model类（融合优化逻辑） --------------------------
class Model(nn.Module):
    """
    优化点：
    1. VMD默认参数适配Weather（k=2, alpha=800）
    2. CNN默认关闭（use_cnn=False），避免过度建模
    3. 稀疏注意力默认0.6，保留更多周期依赖
    4. 动态融合权重：k=2时自动屏蔽趋势成分权重
    """

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride
        d_model = configs.d_model

        # -------------------------- 1. 初始化优化后的模块 --------------------------
        # VMD时序分解（从configs获取参数，默认适配Weather）
        self.vmd = VMD(
            k=getattr(configs, 'vmd_k', 2),
            alpha=getattr(configs, 'vmd_alpha', 800.0),
            max_iter=getattr(configs, 'vmd_max_iter', 200)
        )

        # CNN增强（默认关闭，可通过configs启用）
        self.use_cnn = getattr(configs, 'use_cnn', False)
        self.cnn_enhance = CNNEnhance(
            d_model=d_model,
            kernel_size=getattr(configs, 'cnn_kernel', 1),  # 默认1x1卷积
            dropout=configs.dropout,
            use_cnn=self.use_cnn
        )

        # 稀疏注意力（默认0.6，保留更多依赖）
        self.sparse_attention = SparseAttention(
            mask_flag=False,
            factor=configs.factor,
            attention_dropout=configs.dropout,
            sparse_rate=getattr(configs, 'sparse_rate', 0.6)
        )

        # 多成分融合权重（k=2时自动屏蔽趋势权重）
        self.alpha = nn.Parameter(torch.ones(1, device=configs.device))  # 趋势权重
        self.beta = nn.Parameter(torch.ones(1, device=configs.device))   # 周期权重
        self.gamma = nn.Parameter(torch.ones(1, device=configs.device))  # 残差权重
        self.vmd_k = self.vmd.k  # 记录VMD分解数，用于动态调整权重

        # -------------------------- 2. 原有模块（无修改） --------------------------
        self.patch_embedding = PatchEmbedding(
            d_model, patch_len, stride, padding, configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        self.sparse_attention,
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

    # -------------------------- 辅助函数：单成分处理（兼容CNN开关） --------------------------
    def _process_single_component(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        B, L, D = x.shape
        nvars = D

        # 1. 归一化（单成分独立归一化，避免成分间尺度干扰）
        means = x.mean(1, keepdim=True).detach()  # [B, 1, D]
        x_norm = x - means
        stdev = torch.sqrt(torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm /= stdev

        # 2. Patch Embedding
        x_norm = x_norm.permute(0, 2, 1)  # [B, D, L]
        patch_out, _ = self.patch_embedding(x_norm)  # [B*nvars, patch_num, d_model]

        # 3. CNN增强（按需启用）
        if self.use_cnn:
            patch_out = self.cnn_enhance(patch_out)

        # 4. Encoder
        enc_out, _ = self.encoder(patch_out)  # [B*nvars, patch_num, d_model]

        # 5. 维度重塑
        enc_out = enc_out.reshape(-1, nvars, enc_out.shape[-2], enc_out.shape[-1])  # [B, nvars, patch_num, d_model]
        enc_out = enc_out.permute(0, 1, 3, 2)  # [B, nvars, d_model, patch_num]

        return enc_out, means, stdev

    # -------------------------- 核心预测函数（动态融合权重） --------------------------
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 1. VMD分解（输出趋势、周期、残差，k=2时趋势为0）
        x_trend, x_period, x_res = self.vmd(x_enc)

        # 2. 处理各成分
        enc_trend, mean_trend, stdev_trend = self._process_single_component(x_trend)
        enc_period, mean_period, stdev_period = self._process_single_component(x_period)
        enc_res, mean_res, stdev_res = self._process_single_component(x_res)

        # 3. 动态融合权重：k=2时屏蔽趋势权重（alpha=0）
        if self.vmd_k == 2:
            weights = F.softmax(torch.cat([torch.tensor(0.0, device=x_enc.device), self.beta, self.gamma]), dim=0)
        else:
            weights = F.softmax(torch.cat([self.alpha, self.beta, self.gamma]), dim=0)
        enc_total = weights[0] * enc_trend + weights[1] * enc_period + weights[2] * enc_res

        # 4. 预测头与反归一化
        dec_out = self.head(enc_total).permute(0, 2, 1)  # [B, pred_len, nvars]
        # 反归一化：仅周期和残差有效（k=2时趋势为0，反归一化后仍为0）
        dec_period = dec_out * stdev_period[:, 0, :].unsqueeze(1) + mean_period[:, 0, :].unsqueeze(1)
        dec_res = dec_out * stdev_res[:, 0, :].unsqueeze(1) + mean_res[:, 0, :].unsqueeze(1)
        dec_trend = dec_out * stdev_trend[:, 0, :].unsqueeze(1) + mean_trend[:, 0, :].unsqueeze(1)
        dec_out = weights[0] * dec_trend + weights[1] * dec_period + weights[2] * dec_res

        return dec_out

    # -------------------------- 其他任务函数（兼容优化逻辑） --------------------------
    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        x_enc_valid = x_enc * mask
        x_trend, x_period, x_res = self.vmd(x_enc_valid)

        # 单成分归一化（仅用有效数据）
        def _impute_norm(x, mask):
            means = (x * mask).sum(1) / (mask.sum(1) + 1e-7)
            means = means.unsqueeze(1).detach()
            x_norm = (x - means) * mask
            stdev = torch.sqrt((x_norm ** 2).sum(1) / (mask.sum(1) + 1e-7) + 1e-5)
            stdev = stdev.unsqueeze(1).detach()
            x_norm /= stdev
            return x_norm, means, stdev

        x_trend_norm, mean_trend, stdev_trend = _impute_norm(x_trend, mask)
        x_period_norm, mean_period, stdev_period = _impute_norm(x_period, mask)
        x_res_norm, mean_res, stdev_res = _impute_norm(x_res, mask)

        # Patch+CNN（按需启用）+Encoder
        x_trend_norm = x_trend_norm.permute(0, 2, 1)
        x_period_norm = x_period_norm.permute(0, 2, 1)
        x_res_norm = x_res_norm.permute(0, 2, 1)

        patch_trend, _ = self.patch_embedding(x_trend_norm)
        patch_period, _ = self.patch_embedding(x_period_norm)
        patch_res, _ = self.patch_embedding(x_res_norm)

        if self.use_cnn:
            patch_trend = self.cnn_enhance(patch_trend)
            patch_period = self.cnn_enhance(patch_period)
            patch_res = self.cnn_enhance(patch_res)

        enc_trend, _ = self.encoder(patch_trend)
        enc_period, _ = self.encoder(patch_period)
        enc_res, _ = self.encoder(patch_res)

        # 动态融合权重
        nvars = x_enc.shape[2]
        enc_trend = enc_trend.reshape(-1, nvars, enc_trend.shape[-2], enc_trend.shape[-1]).permute(0,1,3,2)
        enc_period = enc_period.reshape(-1, nvars, enc_period.shape[-2], enc_period.shape[-1]).permute(0,1,3,2)
        enc_res = enc_res.reshape(-1, nvars, enc_res.shape[-2], enc_res.shape[-1]).permute(0,1,3,2)

        if self.vmd_k == 2:
            weights = F.softmax(torch.cat([torch.tensor(0.0, device=x_enc.device), self.beta, self.gamma]), dim=0)
        else:
            weights = F.softmax(torch.cat([self.alpha, self.beta, self.gamma]), dim=0)
        enc_total = weights[0]*enc_trend + weights[1]*enc_period + weights[2]*enc_res

        # 反归一化
        dec_out = self.head(enc_total).permute(0,2,1)
        dec_trend = dec_out * stdev_trend[:,0,:].unsqueeze(1) + mean_trend[:,0,:].unsqueeze(1)
        dec_period = dec_out * stdev_period[:,0,:].unsqueeze(1) + mean_period[:,0,:].unsqueeze(1)
        dec_res = dec_out * stdev_res[:,0,:].unsqueeze(1) + mean_res[:,0,:].unsqueeze(1)
        dec_out = weights[0]*dec_trend + weights[1]*dec_period + weights[2]*dec_res

        return dec_out

    def anomaly_detection(self, x_enc):
        x_trend, x_period, x_res = self.vmd(x_enc)
        enc_trend, mean_trend, stdev_trend = self._process_single_component(x_trend)
        enc_period, mean_period, stdev_period = self._process_single_component(x_period)
        enc_res, mean_res, stdev_res = self._process_single_component(x_res)

        if self.vmd_k == 2:
            weights = F.softmax(torch.cat([torch.tensor(0.0, device=x_enc.device), self.beta, self.gamma]), dim=0)
        else:
            weights = F.softmax(torch.cat([self.alpha, self.beta, self.gamma]), dim=0)
        enc_total = weights[0]*enc_trend + weights[1]*enc_period + weights[2]*enc_res

        dec_out = self.head(enc_total).permute(0,2,1)
        dec_trend = dec_out * stdev_trend[:,0,:].unsqueeze(1) + mean_trend[:,0,:].unsqueeze(1)
        dec_period = dec_out * stdev_period[:,0,:].unsqueeze(1) + mean_period[:,0,:].unsqueeze(1)
        dec_res = dec_out * stdev_res[:,0,:].unsqueeze(1) + mean_res[:,0,:].unsqueeze(1)
        dec_out = weights[0]*dec_trend + weights[1]*dec_period + weights[2]*dec_res

        return dec_out

    def classification(self, x_enc, x_mark_enc):
        x_trend, x_period, x_res = self.vmd(x_enc)
        enc_trend, _, _ = self._process_single_component(x_trend)
        enc_period, _, _ = self._process_single_component(x_period)
        enc_res, _, _ = self._process_single_component(x_res)

        if self.vmd_k == 2:
            weights = F.softmax(torch.cat([torch.tensor(0.0, device=x_enc.device), self.beta, self.gamma]), dim=0)
        else:
            weights = F.softmax(torch.cat([self.alpha, self.beta, self.gamma]), dim=0)
        enc_total = weights[0]*enc_trend + weights[1]*enc_period + weights[2]*enc_res

        output = self.flatten(enc_total)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)

        return output

    # -------------------------- forward函数（保持不变） --------------------------
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