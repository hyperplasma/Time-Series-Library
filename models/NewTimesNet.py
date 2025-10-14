import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import numpy as np
from pytorch_wavelets import DWT1D
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


# -------------------------- 1. 基于pytorch-wavelets的GPU小波周期检测（核心替换） --------------------------
def wavelet_period_detection(x, k=2):
    """
    基于pytorch-wavelets的全GPU小波周期检测：批量处理，无CPU传输
    Args:
        x: [B, T, C] 输入时序数据（GPU张量）
        k: 需返回的top-k周期数
    Returns:
        wavelet_periods: [k,] 检测到的top-k周期（所有样本共享）
        wavelet_weights: [B, k] 各周期的权重（基于低频分量能量）
    """
    B, T, C = x.shape
    wavelet_period_candidates = []
    wavelet_energies = []

    # 初始化1级DWT（db4小波，对称边界模式避免边缘效应，移到GPU）
    dwt = DWT1D(wave='db4', level=1, mode='symmetric').to(x.device)

    # 调整输入形状为[B, C, T]（符合DWT1D要求：batch, channel, length）
    x_reshaped = x.permute(0, 2, 1)  # [B, C, T]

    # 批量DWT分解（GPU并行处理所有样本和通道，无需循环）
    cA1, _ = dwt(x_reshaped)  # cA1: [B, C, 1, T//2]（低频分量，含主要周期）
    cA1 = cA1.squeeze(2)  # 移除冗余维度→[B, C, T//2]

    # 遍历每个样本和通道（内部计算全在GPU）
    for b in range(B):
        for c in range(C):
            # 提取当前样本-通道的低频分量（已在GPU上）
            cA1_bc = cA1[b, c, :]  # [T//2,]
            if len(cA1_bc) < 2:
                continue  # 避免FFT结果过短，无法计算周期

            # GPU上做FFT找主频率（实值FFT，效率更高）
            xf_cA1 = torch.fft.rfft(cA1_bc)
            amp_cA1 = torch.abs(xf_cA1)
            amp_cA1[0] = 0  # 排除直流分量（无周期意义）

            # 取top-k频率（GPU排序，避免CPU传输）
            if len(amp_cA1) <= k:
                top_freq_idx = torch.argsort(amp_cA1, descending=True)  # 不足k时全取
            else:
                top_freq_idx = torch.argsort(amp_cA1)[-k:][::-1]  # 降序top-k

            # 计算周期（映射回原序列长度：DWT下采样1倍，周期×2）
            T_cA1 = cA1_bc.shape[0]
            top_periods = (T_cA1 // top_freq_idx) * 2

            # 过滤无效周期（1 < 周期 < T/2，避免极端值）
            valid_mask = (top_periods > 1) & (top_periods < T//2)
            valid_periods = top_periods[valid_mask]
            if len(valid_periods) == 0:
                continue

            # 暂存结果（仅候选周期转CPU，数据量小，无明显开销）
            wavelet_period_candidates.extend(valid_periods.cpu().numpy().tolist())
            # 计算低频分量能量（作为周期权重依据）
            energy_cA1 = torch.sum(cA1_bc ** 2).item()
            wavelet_energies.extend([energy_cA1] * len(valid_periods))

    # 3. 融合周期候选：取出现频次最高的top-k周期
    if not wavelet_period_candidates:
        # 极端情况：无有效周期，用默认值（序列长度的一半）
        wavelet_periods = np.array([T//2] * k)
    else:
        unique_periods, counts = np.unique(wavelet_period_candidates, return_counts=True)
        top_k_idx = np.argsort(counts)[-k:][::-1]
        wavelet_periods = unique_periods[top_k_idx].astype(int)

    # 4. 计算周期权重（GPU张量输出，与输入设备一致）
    wavelet_weights = torch.ones(B, k, device=x.device, dtype=x.dtype)
    if wavelet_energies:
        wavelet_weights *= (np.mean(wavelet_energies) / k)
    else:
        wavelet_weights /= k  # 无能量时用平均权重

    return wavelet_periods, wavelet_weights


# -------------------------- 2. 全GPU VMD类（替换原numpy/CPU版本，核心优化） --------------------------
class VMD(nn.Module):
    def __init__(self, alpha=2000, tau=1e-7, K=3, DC=0, init=1, tol=1e-6):
        """
        全GPU VMD：变分模态分解，所有操作基于PyTorch张量，无CPU交互
        Args:
            alpha: 惩罚参数（平衡重构误差和带宽）
            K: 分解的IMF数量（默认3，覆盖日/周等多尺度周期）
            tol: 收敛阈值（默认1e-6）
        """
        super(VMD, self).__init__()
        self.alpha = alpha
        self.tau = tau
        self.K = K
        self.DC = DC
        self.init = init
        self.tol = tol

    def forward(self, x):
        """
        Args:
            x: [T,] 单通道时序数据（GPU张量）
        Returns:
            imfs: [K, T] K个固有模态函数（IMF，GPU张量）
        """
        device = x.device
        dtype = x.dtype
        T = x.shape[0]

        # 1. 初始化时间轴和频率轴（GPU张量）
        t = torch.arange(1, T + 1, device=device) / T  # [T,]
        freqs = t - 0.5 - 1 / T  # [T,]（频率范围：-0.5~0.5）

        # 2. 初始化中间变量（复数张量，FFT需复数运算）
        complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        u_hat = torch.zeros((self.K, T), dtype=complex_dtype, device=device)  # IMF频谱
        u_hat_prev = torch.zeros_like(u_hat)  # 上一轮IMF频谱
        omega = torch.zeros(self.K, device=device)  # 各IMF的中心频率

        # 3. 初始化中心频率（随机或线性分布）
        if self.init == 1:
            omega = 0.5 * torch.rand(self.K, device=device)  # 随机初始化（0~0.5）
        else:
            omega = torch.linspace(0, 0.5, self.K, device=device)  # 线性分布

        # 4. 初始化拉格朗日乘子（复数张量）
        lambda_hat = torch.zeros((1, T), dtype=complex_dtype, device=device)
        lambda_hat_prev = torch.zeros_like(lambda_hat)

        # 5. 输入信号傅里叶变换（GPU上完成）
        f_hat = torch.fft.fftshift(torch.fft.fft(x.to(complex_dtype)))  # [T,]（频谱中心化）

        # 6. VMD迭代优化（核心步骤，全GPU）
        max_iter = 50  # 最大迭代次数（避免无限循环）
        for n in range(max_iter):
            # 6.1 更新每个IMF的频谱
            for k in range(self.K):
                # 计算分母：频率约束项
                denom = (freqs - omega[k]) ** 2 + self.tau ** 2
                # 相邻IMF的频率约束（避免频谱重叠）
                if k > 0:
                    denom += (freqs - omega[k-1]) ** 2
                if k < self.K - 1:
                    denom += (freqs - omega[k+1]) ** 2
                # 更新当前IMF频谱（变分优化核心公式）
                u_hat[k, :] = (f_hat - lambda_hat_prev / 2) / (1 + self.alpha * denom)

            # 6.2 更新中心频率（按能量加权）
            for k in range(self.K):
                if not self.DC:  # 非直流分量（默认）
                    # 分子：频率×|IMF频谱|² 的积分（能量加权）
                    numerator = torch.sum(freqs * torch.abs(u_hat[k, :]) ** 2)
                    # 分母：|IMF频谱|² 的积分（总能量）
                    denominator = torch.sum(torch.abs(u_hat[k, :]) ** 2) + 1e-8  # 避免除零
                    omega[k] = numerator / denominator
                else:
                    omega[k] = 0  # 直流分量中心频率为0

            # 6.3 更新拉格朗日乘子（对偶优化，加速收敛）
            lambda_hat = lambda_hat_prev + self.tau * (torch.sum(u_hat, dim=0) - f_hat)

            # 6.4 收敛判断（IMF和频率变化小于阈值则停止）
            if n % 10 == 0:  # 每10轮判断一次，减少计算量
                # IMF频谱变化率
                u_diff = torch.sum(torch.abs(u_hat - u_hat_prev) ** 2) / (torch.sum(torch.abs(u_hat_prev) ** 2) + 1e-8)
                # 中心频率变化率
                omega_diff = torch.sum(torch.abs(omega - torch.roll(omega, 1))) / self.K
                if u_diff < self.tol and omega_diff < self.tol:
                    break  # 收敛则退出迭代

            # 保存上一轮结果
            u_hat_prev = u_hat.clone()
            lambda_hat_prev = lambda_hat.clone()

        # 7. 逆傅里叶变换得到时域IMF（取实部，转回原数据类型）
        imfs = torch.zeros((self.K, T), dtype=dtype, device=device)
        for k in range(self.K):
            # 频谱逆中心化→逆FFT→取实部（去除复数虚部误差）
            imf = torch.fft.ifft(torch.fft.ifftshift(u_hat[k, :])).real
            imfs[k, :] = imf

        return imfs


# -------------------------- 3. VMD周期检测（适配全GPU VMD，无CPU传输） --------------------------
def vmd_period_detection(x, vmd_model, k=2):
    """
    全GPU VMD周期检测：对每个IMF做FFT，取能量最大的IMF对应周期
    Args:
        x: [B, T, C] 输入时序数据（GPU张量）
        vmd_model: 全GPU VMD实例
        k: 需返回的top-k周期数
    Returns:
        vmd_periods: [k,] 检测到的top-k周期（所有样本共享）
        vmd_weights: [B, k] 各周期的权重（基于IMF能量）
    """
    B, T, C = x.shape
    vmd_period_candidates = []
    vmd_energies = []

    for b in range(B):
        for c in range(C):
            # 1. 提取单通道数据（GPU）
            seq = x[b, :, c]  # [T,]
            # 2. 全GPU VMD分解（无CPU传输）
            imfs = vmd_model(seq)  # [K, T]
            # 3. 计算每个IMF的能量（GPU上完成）
            imf_energies = torch.sum(imfs ** 2, dim=1)  # [K,]
            # 过滤全零IMF（分解无效）
            if torch.all(imf_energies == 0):
                continue
            
            # 4. 动态取top-n IMF（最多2个，避免数量不足）
            num_imfs = len(imf_energies)
            n = min(2, num_imfs)
            if n == 0:
                continue
            
            # 5. GPU排序+翻转（避免step错误，替代[::-1]切片）
            sorted_idx_asc = torch.argsort(imf_energies)  # 能量升序索引
            top_n_idx_asc = sorted_idx_asc[-n:]  # top-n能量索引（升序）
            top_imf_idx = torch.flip(top_n_idx_asc, dims=[0])  # 转为降序索引

            # 6. 对每个top-IMF做FFT找周期（全GPU）
            for idx in top_imf_idx:
                imf = imfs[idx, :]
                xf_imf = torch.fft.rfft(imf)  # 实值FFT
                amp_imf = torch.abs(xf_imf)
                amp_imf[0] = 0  # 排除直流分量
                if len(amp_imf) <= 1:
                    continue  # 避免无效频率
                
                # 取主频率对应的周期
                top_freq_idx = torch.argsort(amp_imf)[-1]
                if top_freq_idx == 0:
                    continue  # 避免除零
                period = T // top_freq_idx  # 映射回原序列周期
                
                # 暂存结果（仅候选周期转CPU）
                vmd_period_candidates.append(period.item())
                vmd_energies.append(imf_energies[idx].item())

    # 7. 融合周期候选（取频次最高的top-k）
    if not vmd_period_candidates:
        vmd_periods = np.array([T//2] * k)
    else:
        unique_periods, counts = np.unique(vmd_period_candidates, return_counts=True)
        # 候选周期不足k时，用最频繁周期填充
        if len(unique_periods) < k:
            most_common = unique_periods[np.argmax(counts)]
            vmd_periods = np.pad(unique_periods, (0, k - len(unique_periods)),
                                mode='constant', constant_values=most_common)
        else:
            top_k_idx = np.argsort(counts)[-k:][::-1]
            vmd_periods = unique_periods[top_k_idx].astype(int)
    
    # 8. 计算周期权重（GPU张量输出）
    vmd_weights = torch.ones(B, k, device=x.device, dtype=x.dtype)
    if vmd_energies:
        vmd_weights *= (np.mean(vmd_energies) / k)
    else:
        vmd_weights /= k

    return vmd_periods, vmd_weights


# -------------------------- 4. 多尺度周期检测融合模块（FFT+小波+VMD，全GPU） --------------------------
class MultiScalePeriodDetector(nn.Module):
    def __init__(self, configs):
        """
        基础融合版本：整合FFT（原方法）、全GPU小波、全GPU VMD
        Args:
            configs: 项目配置（需包含top_k，vmd_K可选，默认3）
        """
        super(MultiScalePeriodDetector, self).__init__()
        self.top_k = configs.top_k
        # 初始化全GPU VMD（支持通过configs调整K值）
        self.vmd_model = VMD(K=getattr(configs, 'vmd_K', 3), alpha=2000)

    def forward(self, x):
        """
        融合三种方法检测周期，输出格式与原FFT_for_Period完全一致
        Args:
            x: [B, T, C] 输入时序数据（GPU张量）
        Returns:
            fused_periods: [top_k,] 融合后的top-k周期
            fused_weights: [B, top_k] 融合后的周期权重（置信度，GPU张量）
        """
        B, T, C = x.shape

        # 1. 原有FFT周期检测（GPU加速，无需修改）
        fft_periods, fft_weights = FFT_for_Period(x, self.top_k)
        # 2. 全GPU小波周期检测（替换原pywt版本）
        wavelet_periods, wavelet_weights = wavelet_period_detection(x, self.top_k)
        # 3. 全GPU VMD周期检测（替换原numpy版本）
        vmd_periods, vmd_weights = vmd_period_detection(x, self.vmd_model, self.top_k)

        # -------------------------- 基础融合规则：投票+能量加权 --------------------------
        # 步骤1：收集所有周期候选（三种方法）
        all_period_candidates = list(fft_periods) + list(wavelet_periods) + list(vmd_periods)
        unique_periods, counts = np.unique(all_period_candidates, return_counts=True)
        
        # 步骤2：过滤有效周期（1 < 周期 < T/2），避免极端值
        valid_mask = (unique_periods > 1) & (unique_periods < T//2)
        if np.sum(valid_mask) == 0:
            # 极端情况：无有效周期， fallback到FFT结果
            fused_periods = fft_periods
            fused_weights = fft_weights
        else:
            valid_periods = unique_periods[valid_mask]
            valid_counts = counts[valid_mask]
            # 取投票次数最高的top_k周期
            top_k_valid_idx = np.argsort(valid_counts)[-self.top_k:][::-1]
            fused_periods = valid_periods[top_k_valid_idx].astype(int)

            # 步骤3：计算融合权重（三种方法权重均值）
            fused_weights = torch.zeros(B, self.top_k, device=x.device, dtype=x.dtype)
            for i, period in enumerate(fused_periods):
                method_weights = []
                # 收集该周期在各方法中的权重
                if period in fft_periods:
                    fft_idx = list(fft_periods).index(period)
                    method_weights.append(fft_weights[:, fft_idx].unsqueeze(1))
                if period in wavelet_periods:
                    wavelet_idx = list(wavelet_periods).index(period)
                    method_weights.append(wavelet_weights[:, wavelet_idx].unsqueeze(1))
                if period in vmd_periods:
                    vmd_idx = list(vmd_periods).index(period)
                    method_weights.append(vmd_weights[:, vmd_idx].unsqueeze(1))
                # 权重均值（无该周期的方法不参与）
                if method_weights:
                    fused_weights[:, i] = torch.mean(torch.cat(method_weights, dim=1), dim=1)
                else:
                    # 极端情况：无方法检测到，用三种方法的平均权重
                    avg_weight = torch.mean(torch.cat([fft_weights, wavelet_weights, vmd_weights], dim=1), dim=1)
                    fused_weights[:, i] = avg_weight

        return fused_periods, fused_weights


# -------------------------- 5. 原有FFT周期检测（保持不变，GPU加速） --------------------------
def FFT_for_Period(x, k=2):
    # [B, T, C] 输入（GPU张量）
    xf = torch.fft.rfft(x, dim=1)  # 实值FFT，效率更高
    # 计算各频率的平均振幅（按样本和通道平均）
    frequency_list = abs(xf).mean(0).mean(-1)  # [T//2 + 1,]
    frequency_list[0] = 0  # 排除直流分量
    # 取振幅最大的k个频率索引
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()  # 仅索引转CPU（数据量小）
    # 计算周期（序列长度//频率索引）
    period = x.shape[1] // top_list
    # 计算频率权重（各样本的平均振幅）
    weights = abs(xf).mean(-1)[:, top_list]  # [B, k]（GPU张量）
    return period, weights


# -------------------------- 6. TimesBlock（保持接口不变，调用全GPU融合模块） --------------------------
class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # 原有卷积块（保持不变）
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff, num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model, num_kernels=configs.num_kernels)
        )
        # 初始化全GPU多尺度周期检测模块（替换原FFT）
        self.period_detector = MultiScalePeriodDetector(configs)

    def forward(self, x):
        B, T, N = x.size()
        # 用全GPU融合模块替换原FFT_for_Period
        period_list, period_weight = self.period_detector(x)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # 周期对齐padding（保持原有逻辑）
            if (self.seq_len + self.pred_len) % period != 0:
                length = ((self.seq_len + self.pred_len) // period) + 1
                length *= period
                padding = torch.zeros([x.shape[0], length - (self.seq_len + self.pred_len), x.shape[2]],
                                    device=x.device, dtype=x.dtype)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            # 2D卷积处理（保持原有逻辑）
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        # 周期权重自适应聚合（保持原有逻辑）
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # 残差连接（保持原有逻辑）
        res = res + x
        return res


# -------------------------- 7. Model类（完全保持不变，兼容所有原有任务） --------------------------
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
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        # 预测头（保持原有逻辑）
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name in ['imputation', 'anomaly_detection']:
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 归一化（保持原有逻辑）
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # 嵌入层（保持原有逻辑）
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        # TimesNet骨干（保持原有逻辑）
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # 投影回输出维度（保持原有逻辑）
        dec_out = self.projection(enc_out)

        # 反归一化（保持原有逻辑）
        dec_out = dec_out.mul(stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out.add(means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # 归一化（保持原有逻辑）
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc.sub(means)
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / (torch.sum(mask == 1, dim=1) + 1e-5))
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc.div(stdev)

        # 嵌入层+骨干（保持原有逻辑）
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        dec_out = self.projection(enc_out)

        # 反归一化（保持原有逻辑）
        dec_out = dec_out.mul(stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out.add(means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # 归一化+骨干+投影（保持原有逻辑）
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        enc_out = self.enc_embedding(x_enc, None)
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        dec_out = self.projection(enc_out)

        dec_out = dec_out.mul(stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out.add(means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # 嵌入层+骨干+分类头（保持原有逻辑）
        enc_out = self.enc_embedding(x_enc, None)
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # 任务路由（保持原有逻辑）
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        return None