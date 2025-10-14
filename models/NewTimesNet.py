import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import pywt
import numpy as np
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


# -------------------------- 新增：小波变换周期检测（基础版本：db4小波） --------------------------
def wavelet_period_detection(x, k=2):
    """
    基于离散小波变换（DWT）的周期检测：提取低频分量（含主要周期），再通过FFT找周期
    Args:
        x: [B, T, C] 输入时序数据
        k: 需返回的top-k周期数
    Returns:
        wavelet_periods: [k,] 检测到的top-k周期（所有样本共享，基础版本简化）
        wavelet_weights: [B, k] 各周期的权重（基于小波分量能量）
    """
    B, T, C = x.shape
    # 1. 对每个样本的每个通道做小波变换（db4小波，1级分解：低频cA1 + 高频cD1）
    wavelet_period_candidates = []
    wavelet_energies = []  # 用低频分量能量作为权重依据
    for b in range(B):
        for c in range(C):
            # 提取单一样本单通道数据：[T,]
            seq = x[b, :, c].detach().cpu().numpy()
            # 1级DWT分解
            cA1, cD1 = pywt.dwt(seq, 'db4')  # cA1: 低频分量（保留主要周期），长度≈T/2
            # 2. 对低频分量cA1做FFT找周期
            xf_cA1 = np.fft.rfft(cA1)
            amp_cA1 = np.abs(xf_cA1)
            amp_cA1[0] = 0  # 排除直流分量
            # 找top-k频率对应的周期（cA1长度≈T/2，周期需映射回原序列长度）
            top_freq_idx = np.argsort(amp_cA1)[-k:][::-1]  # 降序排列的top-k频率索引
            top_periods = (len(cA1) // top_freq_idx) * 2  # 映射回原序列周期（DWT下采样1倍）
            wavelet_period_candidates.extend(top_periods)
            # 计算低频分量能量（作为该周期的置信度）
            energy_cA1 = np.sum(cA1 ** 2)
            wavelet_energies.extend([energy_cA1] * k)
    
    # 3. 融合所有样本/通道的周期候选：取出现频次最高的top-k周期（基础版本规则融合）
    unique_periods, counts = np.unique(wavelet_period_candidates, return_counts=True)
    top_k_idx = np.argsort(counts)[-k:][::-1]
    wavelet_periods = unique_periods[top_k_idx].astype(int)
    # 4. 计算周期权重：基于能量归一化（每个样本的权重相同，基础版本简化）
    wavelet_weights = torch.ones(B, k, device=x.device) * (np.mean(wavelet_energies) / k)
    return wavelet_periods, wavelet_weights


# -------------------------- 新增：VMD（变分模态分解）类（适配PyTorch） --------------------------
class VMD(nn.Module):
    def __init__(self, alpha=2000, tau=1e-7, K=3, DC=0, init=1, tol=1e-6):
        """
        VMD参数初始化（基础版本：默认分解为3个IMF，覆盖主要周期尺度）
        Args:
            alpha: 惩罚参数（平衡重构误差和带宽）
            K: 分解的IMF数量（需覆盖数据主要周期数）
            tol: 收敛阈值
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
        VMD分解：输入单通道时序，输出K个IMF分量
        Args:
            x: [T,] 单通道时序数据（torch张量）
        Returns:
            imfs: [K, T] K个固有模态函数（IMF）
        """
        x = x.detach().cpu().numpy()
        T = len(x)
        t = np.arange(1, T + 1) / T  # 时间轴
        freqs = t - 0.5 - 1 / T      # 频率轴

        # 初始化：权重、IMF、拉格朗日乘子
        u_hat = np.zeros((self.K, T), dtype=np.complex_)
        u_hat_prev = np.zeros((self.K, T), dtype=np.complex_)
        omega = np.zeros(self.K)
        if self.init == 1:
            omega = 0.5 * np.random.rand(self.K)
        else:
            omega = np.linspace(0, 0.5, self.K)
        lambda_hat = np.zeros((1, T), dtype=np.complex_)
        lambda_hat_prev = np.zeros((1, T), dtype=np.complex_)

        # 傅里叶变换（中心化）
        f_hat = np.fft.fftshift(np.fft.fft(x))
        f_hat_conj = np.conj(f_hat)

        # VMD迭代（基础版本：固定迭代50次，平衡效率与效果）
        for n in range(50):
            # 1. 更新IMF的频谱
            for k in range(self.K):
                # 计算分母
                denom = (freqs - omega[k]) ** 2 + self.tau ** 2
                if k > 0:
                    denom += (freqs - omega[k-1]) ** 2
                if k < self.K - 1:
                    denom += (freqs - omega[k+1]) ** 2
                # 更新当前IMF的频谱
                u_hat[k, :] = (f_hat - lambda_hat_prev / 2) / (1 + self.alpha * denom)
            
            # 2. 更新频率中心（按能量加权）
            for k in range(self.K):
                if not self.DC:
                    omega[k] = np.sum(freqs * np.abs(u_hat[k, :]) ** 2) / np.sum(np.abs(u_hat[k, :]) ** 2)
                else:
                    omega[k] = 0  # DC分量频率为0

            # 3. 更新拉格朗日乘子
            lambda_hat = lambda_hat_prev + self.tau * (np.sum(u_hat, axis=0) - f_hat)

            # 4. 检查收敛（基础版本：简化为固定迭代次数，避免计算量过大）
            if n % 10 == 0:
                u_diff = np.sum(np.abs(u_hat - u_hat_prev) ** 2) / np.sum(np.abs(u_hat_prev) ** 2)
                omega_diff = np.sum(np.abs(omega - np.roll(omega, 1))) / self.K
                if u_diff < self.tol and omega_diff < self.tol:
                    break
            u_hat_prev = u_hat.copy()
            lambda_hat_prev = lambda_hat.copy()

        # 逆傅里叶变换得到IMF（时域）
        imfs = np.zeros((self.K, T))
        for k in range(self.K):
            imfs[k, :] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[k, :])))
        # 转换为torch张量并返回
        return torch.tensor(imfs, device=x.device, dtype=x.dtype)


# -------------------------- 新增：VMD周期检测函数 --------------------------
def vmd_period_detection(x, vmd_model, k=2):
    """
    基于VMD的周期检测：对每个IMF做FFT，取能量最大的IMF对应的周期
    Args:
        x: [B, T, C] 输入时序数据
        vmd_model: 预初始化的VMD实例
        k: 需返回的top-k周期数
    Returns:
        vmd_periods: [k,] 检测到的top-k周期（所有样本共享，基础版本简化）
        vmd_weights: [B, k] 各周期的权重（基于IMF能量）
    """
    B, T, C = x.shape
    vmd_period_candidates = []
    vmd_energies = []  # 用IMF能量作为权重依据

    for b in range(B):
        for c in range(C):
            # 1. VMD分解：得到K个IMF
            seq = x[b, :, c]  # [T,]
            imfs = vmd_model(seq)  # [K, T]
            # 2. 计算每个IMF的能量，选能量最大的2个IMF（聚焦主要周期）
            imf_energies = torch.sum(imfs ** 2, dim=1)  # [K,]
            top_imf_idx = torch.argsort(imf_energies)[-2:][::-1]  # 能量top-2的IMF索引
            # 3. 对每个top-IMF做FFT找周期
            for idx in top_imf_idx:
                imf = imfs[idx, :]
                xf_imf = torch.fft.rfft(imf)
                amp_imf = torch.abs(xf_imf)
                amp_imf[0] = 0  # 排除直流分量
                top_freq_idx = torch.argsort(amp_imf)[-1]  # 每个IMF的主频率
                period = T // top_freq_idx  # 对应周期
                vmd_period_candidates.append(period.item())
                vmd_energies.append(imf_energies[idx].item())

    # 4. 融合周期候选：取出现频次最高的top-k周期（基础版本规则融合）
    unique_periods, counts = np.unique(vmd_period_candidates, return_counts=True)
    top_k_idx = np.argsort(counts)[-k:][::-1]
    vmd_periods = unique_periods[top_k_idx].astype(int)
    # 5. 计算周期权重：基于IMF能量归一化
    vmd_weights = torch.ones(B, k, device=x.device) * (np.mean(vmd_energies) / k)
    return vmd_periods, vmd_weights


# -------------------------- 新增：多尺度周期检测融合模块（FFT+小波+VMD） --------------------------
class MultiScalePeriodDetector(nn.Module):
    def __init__(self, configs):
        """
        基础融合版本周期检测模块：整合FFT（原方法）、小波变换、VMD
        Args:
            configs: 原项目配置（需包含top_k，新增vmd_K=3控制VMD分解的IMF数量）
        """
        super(MultiScalePeriodDetector, self).__init__()
        self.top_k = configs.top_k
        # 初始化VMD模型（基础版本固定参数，可后续通过configs调整）
        self.vmd_model = VMD(K=getattr(configs, 'vmd_K', 3), alpha=2000)

    def forward(self, x):
        """
        融合三种方法检测周期，输出top-k周期及权重（与原FFT_for_Period输出格式完全一致）
        Args:
            x: [B, T, C] 输入时序数据
        Returns:
            fused_periods: [top_k,] 融合后的top-k周期
            fused_weights: [B, top_k] 融合后的周期权重（置信度）
        """
        B, T, C = x.shape

        # 1. 原有FFT周期检测
        fft_periods, fft_weights = FFT_for_Period(x, self.top_k)
        # 2. 小波变换周期检测
        wavelet_periods, wavelet_weights = wavelet_period_detection(x, self.top_k)
        # 3. VMD周期检测
        vmd_periods, vmd_weights = vmd_period_detection(x, self.vmd_model, self.top_k)

        # -------------------------- 基础版本融合规则：投票+能量加权 --------------------------
        # 步骤1：收集所有周期候选（去重）
        all_period_candidates = list(fft_periods) + list(wavelet_periods) + list(vmd_periods)
        unique_periods, counts = np.unique(all_period_candidates, return_counts=True)
        # 步骤2：按投票次数排序，取前top_k周期（确保周期在合理范围：1 < 周期 < T/2）
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

            # 步骤3：计算融合权重（三种方法的权重均值，基础版本简化）
            # 先将各方法权重对齐到融合周期（基础版本：取对应周期的权重均值）
            fused_weights = torch.zeros(B, self.top_k, device=x.device)
            for i, period in enumerate(fused_periods):
                # 收集该周期在各方法中的权重
                method_weights = []
                if period in fft_periods:
                    method_weights.append(fft_weights[:, list(fft_periods).index(period)].unsqueeze(1))
                if period in wavelet_periods:
                    method_weights.append(wavelet_weights[:, list(wavelet_periods).index(period)].unsqueeze(1))
                if period in vmd_periods:
                    method_weights.append(vmd_weights[:, list(vmd_periods).index(period)].unsqueeze(1))
                # 权重均值（无该周期的方法不参与计算）
                if method_weights:
                    fused_weights[:, i] = torch.mean(torch.cat(method_weights, dim=1), dim=1)
                else:
                    # 极端情况：无方法检测到该周期，赋值为平均权重
                    fused_weights[:, i] = torch.mean(torch.cat([fft_weights, wavelet_weights, vmd_weights], dim=1), dim=1)

        return fused_periods, fused_weights


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )
        # 初始化多尺度周期检测模块
        self.period_detector = MultiScalePeriodDetector(configs)

    def forward(self, x):
        B, T, N = x.size()
        # 用多尺度检测替换原FFT_for_Period
        period_list, period_weight = self.period_detector(x)  # 原代码：FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


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