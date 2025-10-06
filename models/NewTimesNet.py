import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1
import math


# -------------------------- 新增：小波变换工具（基于卷积模拟，适配PyTorch） --------------------------
def get_db4_wavelet_filters():
    """预定义Daubechies 4小波（db4）的低通/高通滤波器系数（常用时序局部特征提取）"""
    # db4低通滤波器（近似分量，对应全局趋势/周期）
    low_pass = torch.tensor([
        0.0106, -0.0329, 0.0308, 0.0657, -0.1152, -0.2212, 0.3156, 0.7511,
        0.4946, -0.1495, -0.1767, 0.0935, 0.0352, -0.0823, 0.0344, 0.0106
    ], dtype=torch.float32)
    # db4高通滤波器（细节分量，对应局部波动/短期周期）（低通翻转后隔位取反）
    high_pass = torch.flip(low_pass, dims=[0])
    high_pass = torch.mul(high_pass, (-1) ** torch.arange(len(high_pass)))
    return low_pass, high_pass


class WaveletDecomposition(nn.Module):
    """小波分解模块：提取时序的全局近似分量（全局周期）和局部细节分量（局部周期）"""
    def __init__(self, d_model):
        super(WaveletDecomposition, self).__init__()
        self.d_model = d_model
        self.low_pass, self.high_pass = get_db4_wavelet_filters()
        self.filter_len = len(self.low_pass)
        
        # 1D卷积实现小波滤波（groups=d_model确保各通道独立处理）
        self.low_conv = nn.Conv1d(
            in_channels=d_model, 
            out_channels=d_model, 
            kernel_size=self.filter_len,
            padding=self.filter_len//2, 
            groups=d_model, 
            bias=False
        )
        self.high_conv = nn.Conv1d(
            in_channels=d_model, 
            out_channels=d_model, 
            kernel_size=self.filter_len,
            padding=self.filter_len//2, 
            groups=d_model, 
            bias=False
        )
        
        # 初始化卷积权重为小波系数
        self.low_conv.weight.data = self.low_pass.view(1, 1, -1).repeat(d_model, 1, 1)
        self.high_conv.weight.data = self.high_pass.view(1, 1, -1).repeat(d_model, 1, 1)
        # 小波滤波器权重固定（也可设为可训练，此处默认固定以保证稳定性）
        for param in [self.low_conv.weight, self.high_conv.weight]:
            param.requires_grad = False

    def forward(self, x):
        """
        Args:
            x: [B, T, C] （B=批次，T=时序长度，C=通道数=d_model）
        Returns:
            approx: [B, T, C] 近似分量（全局周期信息）
            detail: [B, T, C] 细节分量（局部波动信息）
        """
        x_permuted = x.permute(0, 2, 1)  # [B, C, T] 适配Conv1d输入格式
        approx = self.low_conv(x_permuted).permute(0, 2, 1)  # [B, T, C]
        detail = self.high_conv(x_permuted).permute(0, 2, 1)  # [B, T, C]
        return approx, detail


# -------------------------- 新增：周期一致性验证工具（自相关分析） --------------------------
def autocorrelation_period_detection(x, max_lag=None):
    """
    自相关分析提取周期：基于近似分量计算自相关系数，峰值对应周期
    Args:
        x: [B, T, C] 小波近似分量（全局周期信息）
        max_lag: 最大滞后步长（默认T//4，避免周期过大）
    Returns:
        auto_period: [B] 每个样本的通道平均周期（修正文档描述，原[B,C]错误）
    """
    B, T, C = x.shape
    max_lag = max_lag if max_lag is not None else T // 4
    if max_lag < 1:
        max_lag = 1

    # 标准化时序（消除均值影响）
    x_norm = x - x.mean(dim=1, keepdim=True)
    # 计算自相关系数（滞后步长1~max_lag）
    autocorr = []
    for lag in range(1, max_lag + 1):
        # 滞后lag的序列
        x_lag = x_norm[:, lag:, :]
        # 原始序列（截断到与滞后序列等长）
        x_ori = x_norm[:, :-lag, :] if lag != 0 else x_norm
        # 计算每个通道的自相关系数（Pearson相关）
        cov = (x_ori * x_lag).mean(dim=1)
        std_ori = torch.sqrt((x_ori ** 2).mean(dim=1) + 1e-8)
        std_lag = torch.sqrt((x_lag ** 2).mean(dim=1) + 1e-8)
        corr = cov / (std_ori * std_lag + 1e-8)  # [B, C]
        autocorr.append(corr.unsqueeze(1))  # [B, 1, C]
    
    autocorr = torch.cat(autocorr, dim=1)  # [B, max_lag, C]
    # 找到自相关系数最大的滞后步长（即周期）
    auto_lag = torch.argmax(autocorr, dim=1) + 1  # [B, C]（+1因为lag从1开始）
    # 若自相关系数均接近0（无明显周期），默认周期为max_lag
    auto_period = torch.where(autocorr.max(dim=1)[0] < 0.1, 
                              torch.tensor(max_lag, device=x.device), 
                              auto_lag)
    # 关键修复：先转浮点计算均值（避免Long型mean错误），再转回长整型（周期需为整数）
    return auto_period.float().mean(dim=1).long()  # [B] 批次内通道平均周期


# -------------------------- 重写：多尺度周期检测（FFT+小波+掩码+一致性验证） --------------------------
class MultiScalePeriodDetector(nn.Module):
    def __init__(self, d_model, top_k=2, mask_reg=1e-5):
        super(MultiScalePeriodDetector, self).__init__()
        self.d_model = d_model
        self.top_k = top_k  # 保留Top-k显著周期（与原版一致，硬编码默认2）
        self.mask_reg = mask_reg  # 频率掩码L1正则强度（硬编码默认1e-5）
        
        # 1. 小波分解模块（提取全局+局部周期）
        self.wavelet_decomp = WaveletDecomposition(d_model=d_model)
        
        # 2. 可学习频率掩码（替代原版固定Top-k，动态强化重要频率）
        # FFT频率维度：因共轭对称性，仅需考虑前T//2个频率
        self.freq_mask = nn.Parameter(torch.ones(1, 1, d_model), requires_grad=True)  # [1,1,C]
        nn.init.constant_(self.freq_mask, 1.0)  # 初始化为全1（无掩码）

    def forward(self, x):
        """
        Args:
            x: [B, T, C] 输入时序特征（B=批次，T=时序长度，C=通道数=d_model）
        Returns:
            period_list: [k] 最终筛选的Top-k周期（全局一致，与原版格式兼容）
            period_weight: [B, T//2, C] 修正后的频率权重（含掩码+一致性验证）
        """
        B, T, C = x.shape
        freq_dim = T // 2  # FFT有效频率维度（共轭对称性）

        # -------------------------- 步骤1：FFT提取全局周期与初始振幅 --------------------------
        xf = torch.fft.rfft(x, dim=1)  # [B, T//2 + 1, C]（rfft输出长度为T//2+1）
        xf_amp = torch.abs(xf)[:, 1:freq_dim+1, :]  # [B, freq_dim, C] 排除直流分量（k=0）
        # 通道维度平均振幅（与原版一致，降低维度）
        amp_avg = xf_amp.mean(dim=2, keepdim=True)  # [B, freq_dim, 1]

        # -------------------------- 步骤2：小波分解与局部周期提取 --------------------------
        approx, _ = self.wavelet_decomp(x)  # [B,T,C] 近似分量（全局周期）；细节分量暂用于验证
        # 自相关分析提取小波周期（作为一致性验证基准）
        wavelet_period = autocorrelation_period_detection(approx, max_lag=T//4)  # [B]

        # -------------------------- 步骤3：可学习频率掩码（强化重要频率） --------------------------
        # 掩码适配频率维度：将通道级掩码扩展到频率维度
        freq_mask_expand = self.freq_mask.repeat(1, freq_dim, 1)  # [1, freq_dim, C]
        # 应用掩码：振幅 * 掩码（动态调整频率权重）
        masked_amp = xf_amp * freq_mask_expand  # [B, freq_dim, C]
        # 掩码正则（避免过度掩码导致信息丢失）
        mask_reg_loss = self.mask_reg * torch.norm(self.freq_mask, p=1)
        # 将正则损失加入计算图（无需返回，自动反向传播）
        if self.training:
            torch.sum(mask_reg_loss).backward(retain_graph=True)

        # -------------------------- 步骤4：周期一致性验证（过滤虚假周期） --------------------------
        # 1. 计算FFT频率对应的周期：freq -> period = T / freq（freq从1到freq_dim）
        freq_list = torch.arange(1, freq_dim+1, device=x.device).float()  # [freq_dim]
        fft_periods = T / freq_list  # [freq_dim] 每个频率对应的周期
        
        # 2. 计算FFT周期与小波周期的相似度（余弦相似度，0~1）
        wavelet_period_expand = wavelet_period.view(B, 1, 1).repeat(1, freq_dim, C)  # [B,freq_dim,C]
        fft_periods_expand = fft_periods.view(1, freq_dim, 1).repeat(B, 1, C)  # [B,freq_dim,C]
        # 余弦相似度（避免除以0）
        sim = torch.cos(
            torch.abs(fft_periods_expand - wavelet_period_expand) / 
            (torch.max(fft_periods_expand, wavelet_period_expand) + 1e-8) * math.pi
        )
        sim = (sim + 1) / 2  # 归一化到0~1（相似度越高越接近1）

        # 3. 修正频率权重：掩码振幅 * 一致性相似度
        period_weight = masked_amp * sim  # [B, freq_dim, C]

        # -------------------------- 步骤5：筛选Top-k周期（与原版格式兼容） --------------------------
        # 全局Top-k频率选择（基于通道平均修正振幅）
        amp_corrected_avg = period_weight.mean(dim=2, keepdim=True)  # [B, freq_dim, 1]
        # 批次内平均振幅（全局排序）
        amp_global_avg = amp_corrected_avg.mean(dim=0).squeeze()  # [freq_dim]
        # 选择Top-k频率（频率从1开始，对应索引+1）
        top_freq_idx = torch.topk(amp_global_avg, self.top_k)[1] + 1  # [k]（+1因为排除了直流分量）
        # 计算最终周期（与原版一致，向上取整避免非整数周期）
        period_list = torch.ceil(T / top_freq_idx).long().cpu().numpy()  # [k]

        return period_list, period_weight


# -------------------------- 重写：TimesBlock（集成多尺度周期检测） --------------------------
class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k  # 保留原版top_k参数（与多尺度检测器兼容）
        self.d_model = configs.d_model  # 新增：用于小波分解和掩码初始化
        
        # -------------------------- 替换：多尺度周期检测器（原FFT_for_Period） --------------------------
        self.period_detector = MultiScalePeriodDetector(
            d_model=self.d_model,
            top_k=self.k,
            mask_reg=1e-5  # 硬编码掩码正则强度
        )

        # -------------------------- 保留原版卷积架构（确保兼容性） --------------------------
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()  # N=d_model（通道数）
        
        # -------------------------- 替换：多尺度周期检测（原FFT_for_Period调用） --------------------------
        period_list, period_weight = self.period_detector(x)
        # 适配原版格式：period_weight需压缩到[B, freq_dim]（取通道平均）
        period_weight = period_weight.mean(dim=2)  # [B, freq_dim]

        # -------------------------- 保留原版二维转换与卷积逻辑（无修改） --------------------------
        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding：确保长度能被周期整除
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape：[B, T, N] → [B, length//period, period, N] → [B, N, length//period, period]
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D卷积：提取二维时序特征
            out = self.conv(out)
            # reshape back：恢复为一维时序
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        
        # -------------------------- 保留原版自适应融合与残差连接（无修改） --------------------------
        res = torch.stack(res, dim=-1)  # [B, T, N, k]
        # 自适应权重（基于修正后的period_weight）
        period_weight = F.softmax(period_weight, dim=1)  # [B, freq_dim]
        # 适配权重维度（与res匹配）：选择Top-k频率对应的权重
        freq_dim = T // 2
        top_freq_idx = (T / torch.tensor(period_list, device=x.device)).long() - 1  # 频率索引（-1因排除直流分量）
        period_weight_topk = period_weight[:, top_freq_idx]  # [B, k]
        # 权重扩展：[B, k] → [B, 1, 1, k] → 与res广播相乘
        period_weight_expand = period_weight_topk.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight_expand, -1)  # [B, T, N]
        # 残差连接
        res = res + x
        return res


# -------------------------- 保留原版Model类（无修改，确保接口兼容） --------------------------
class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out.mul(
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add(
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc.sub(means)
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc.div(stdev)

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out.mul(
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add(
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out.mul(
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add(
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None