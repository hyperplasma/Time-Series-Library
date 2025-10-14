import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import numpy as np
from pytorch_wavelets import DWT1D
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


# -------------------------- 1. 修复：用torch.flip替代负步长切片 --------------------------
def wavelet_period_detection(x, k=2):
    B, T, C = x.shape
    wavelet_period_candidates = []
    wavelet_energies = []

    # 初始化DWT1D（无level参数）
    dwt = DWT1D(wave='db4', mode='symmetric').to(x.device)
    x_reshaped = x.permute(0, 2, 1)  # [B, C, T]
    cA1, _ = dwt(x_reshaped)
    cA1 = cA1.squeeze(2)  # [B, C, T//2]

    for b in range(B):
        for c in range(C):
            cA1_bc = cA1[b, c, :]  # [T//2,]
            if len(cA1_bc) < 2:
                continue

            xf_cA1 = torch.fft.rfft(cA1_bc)
            amp_cA1 = torch.abs(xf_cA1)
            amp_cA1[0] = 0  # 排除直流分量

            # 修复点1：用torch.flip替代[-k:][::-1]，避免负步长
            if len(amp_cA1) <= k:
                # 当长度<=k时，直接按降序排序（无需切片反转）
                top_freq_idx = torch.argsort(amp_cA1, descending=True)
            else:
                # 先取最后k个元素（升序中的top-k），再用torch.flip反转成降序
                top_k_asc = torch.argsort(amp_cA1)[-k:]  # 升序的top-k索引
                top_freq_idx = torch.flip(top_k_asc, dims=[0])  # 反转成降序

            # 计算周期
            T_cA1 = cA1_bc.shape[0]
            top_periods = (T_cA1 // top_freq_idx) * 2

            # 过滤无效周期
            valid_mask = (top_periods > 1) & (top_periods < T//2)
            valid_periods = top_periods[valid_mask]
            if len(valid_periods) == 0:
                continue

            wavelet_period_candidates.extend(valid_periods.cpu().numpy().tolist())
            energy_cA1 = torch.sum(cA1_bc ** 2).item()
            wavelet_energies.extend([energy_cA1] * len(valid_periods))

    # 融合周期候选
    if not wavelet_period_candidates:
        wavelet_periods = np.array([T//2] * k)
    else:
        unique_periods, counts = np.unique(wavelet_period_candidates, return_counts=True)
        top_k_idx = np.argsort(counts)[-k:][::-1]  # numpy支持负步长，此处无需修改
        wavelet_periods = unique_periods[top_k_idx].astype(int)

    # 计算权重
    wavelet_weights = torch.ones(B, k, device=x.device, dtype=x.dtype)
    if wavelet_energies:
        wavelet_weights *= (np.mean(wavelet_energies) / k)
    else:
        wavelet_weights /= k

    return wavelet_periods, wavelet_weights


# -------------------------- 2. 全GPU VMD类（无修改） --------------------------
class VMD(nn.Module):
    def __init__(self, alpha=2000, tau=1e-7, K=3, DC=0, init=1, tol=1e-6):
        super(VMD, self).__init__()
        self.alpha = alpha
        self.tau = tau
        self.K = K
        self.DC = DC
        self.init = init
        self.tol = tol

    def forward(self, x):
        device = x.device
        dtype = x.dtype
        T = x.shape[0]

        t = torch.arange(1, T + 1, device=device) / T
        freqs = t - 0.5 - 1 / T

        complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        u_hat = torch.zeros((self.K, T), dtype=complex_dtype, device=device)
        u_hat_prev = torch.zeros_like(u_hat)
        omega = torch.zeros(self.K, device=device)

        if self.init == 1:
            omega = 0.5 * torch.rand(self.K, device=device)
        else:
            omega = torch.linspace(0, 0.5, self.K, device=device)

        lambda_hat = torch.zeros((1, T), dtype=complex_dtype, device=device)
        lambda_hat_prev = torch.zeros_like(lambda_hat)

        f_hat = torch.fft.fftshift(torch.fft.fft(x.to(complex_dtype)))

        max_iter = 50
        for n in range(max_iter):
            for k in range(self.K):
                denom = (freqs - omega[k]) ** 2 + self.tau ** 2
                if k > 0:
                    denom += (freqs - omega[k-1]) ** 2
                if k < self.K - 1:
                    denom += (freqs - omega[k+1]) ** 2
                u_hat[k, :] = (f_hat - lambda_hat_prev / 2) / (1 + self.alpha * denom)

            for k in range(self.K):
                if not self.DC:
                    numerator = torch.sum(freqs * torch.abs(u_hat[k, :]) ** 2)
                    denominator = torch.sum(torch.abs(u_hat[k, :]) ** 2) + 1e-8
                    omega[k] = numerator / denominator
                else:
                    omega[k] = 0

            lambda_hat = lambda_hat_prev + self.tau * (torch.sum(u_hat, dim=0) - f_hat)

            if n % 10 == 0:
                u_diff = torch.sum(torch.abs(u_hat - u_hat_prev) ** 2) / (torch.sum(torch.abs(u_hat_prev) ** 2) + 1e-8)
                omega_diff = torch.sum(torch.abs(omega - torch.roll(omega, 1))) / self.K
                if u_diff < self.tol and omega_diff < self.tol:
                    break

            u_hat_prev = u_hat.clone()
            lambda_hat_prev = lambda_hat.clone()

        imfs = torch.zeros((self.K, T), dtype=dtype, device=device)
        for k in range(self.K):
            imf = torch.fft.ifft(torch.fft.ifftshift(u_hat[k, :])).real
            imfs[k, :] = imf

        return imfs


# -------------------------- 3. VMD周期检测（确认torch.flip已正确使用） --------------------------
def vmd_period_detection(x, vmd_model, k=2):
    B, T, C = x.shape
    vmd_period_candidates = []
    vmd_energies = []

    for b in range(B):
        for c in range(C):
            seq = x[b, :, c]
            imfs = vmd_model(seq)
            imf_energies = torch.sum(imfs ** 2, dim=1)

            if torch.all(imf_energies == 0):
                continue

            num_imfs = len(imf_energies)
            n = min(2, num_imfs)
            if n == 0:
                continue

            # 此处已使用torch.flip，无需修改
            sorted_idx_asc = torch.argsort(imf_energies)
            top_n_idx_asc = sorted_idx_asc[-n:]
            top_imf_idx = torch.flip(top_n_idx_asc, dims=[0])  # 正确的反转方式

            for idx in top_imf_idx:
                imf = imfs[idx, :]
                xf_imf = torch.fft.rfft(imf)
                amp_imf = torch.abs(xf_imf)
                amp_imf[0] = 0
                if len(amp_imf) <= 1:
                    continue

                # 修复点2：确保此处也用torch.flip处理排序
                top_freq_idx = torch.flip(torch.argsort(amp_imf)[-1:], dims=[0])[0]  # 取最大频率索引
                if top_freq_idx == 0:
                    continue
                period = T // top_freq_idx

                vmd_period_candidates.append(period.item())
                vmd_energies.append(imf_energies[idx].item())

    if not vmd_period_candidates:
        vmd_periods = np.array([T//2] * k)
    else:
        unique_periods, counts = np.unique(vmd_period_candidates, return_counts=True)
        if len(unique_periods) < k:
            most_common = unique_periods[np.argmax(counts)]
            vmd_periods = np.pad(unique_periods, (0, k - len(unique_periods)),
                                mode='constant', constant_values=most_common)
        else:
            top_k_idx = np.argsort(counts)[-k:][::-1]  # numpy支持负步长
            vmd_periods = unique_periods[top_k_idx].astype(int)

    vmd_weights = torch.ones(B, k, device=x.device, dtype=x.dtype)
    if vmd_energies:
        vmd_weights *= (np.mean(vmd_energies) / k)
    else:
        vmd_weights /= k

    return vmd_periods, vmd_weights


# -------------------------- 4. 多尺度周期检测融合模块（无修改） --------------------------
class MultiScalePeriodDetector(nn.Module):
    def __init__(self, configs):
        super(MultiScalePeriodDetector, self).__init__()
        self.top_k = configs.top_k
        self.vmd_model = VMD(K=getattr(configs, 'vmd_K', 3), alpha=2000)

    def forward(self, x):
        B, T, C = x.shape

        # 1. FFT周期检测
        fft_periods, fft_weights = FFT_for_Period(x, self.top_k)
        # 2. 小波周期检测（调用修复后的函数）
        wavelet_periods, wavelet_weights = wavelet_period_detection(x, self.top_k)
        # 3. VMD周期检测
        vmd_periods, vmd_weights = vmd_period_detection(x, self.vmd_model, self.top_k)

        # 融合逻辑
        all_period_candidates = list(fft_periods) + list(wavelet_periods) + list(vmd_periods)
        unique_periods, counts = np.unique(all_period_candidates, return_counts=True)
        valid_mask = (unique_periods > 1) & (unique_periods < T//2)

        if np.sum(valid_mask) == 0:
            fused_periods = fft_periods
            fused_weights = fft_weights
        else:
            valid_periods = unique_periods[valid_mask]
            valid_counts = counts[valid_mask]
            top_k_valid_idx = np.argsort(valid_counts)[-self.top_k:][::-1]  # numpy支持负步长
            fused_periods = valid_periods[top_k_valid_idx].astype(int)

            fused_weights = torch.zeros(B, self.top_k, device=x.device, dtype=x.dtype)
            for i, period in enumerate(fused_periods):
                method_weights = []
                if period in fft_periods:
                    method_weights.append(fft_weights[:, list(fft_periods).index(period)].unsqueeze(1))
                if period in wavelet_periods:
                    method_weights.append(wavelet_weights[:, list(wavelet_periods).index(period)].unsqueeze(1))
                if period in vmd_periods:
                    method_weights.append(vmd_weights[:, list(vmd_periods).index(period)].unsqueeze(1))
                if method_weights:
                    fused_weights[:, i] = torch.mean(torch.cat(method_weights, dim=1), dim=1)
                else:
                    avg_weight = torch.mean(torch.cat([fft_weights, wavelet_weights, vmd_weights], dim=1), dim=1)
                    fused_weights[:, i] = avg_weight

        return fused_periods, fused_weights


# -------------------------- 5. FFT周期检测（修复排序逻辑） --------------------------
def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0

    # 修复点3：用torch.flip替代[-k:][::-1]
    sorted_idx_asc = torch.argsort(frequency_list)  # 升序索引
    top_k_asc = sorted_idx_asc[-k:]  # 取最后k个（top-k）
    top_list = torch.flip(top_k_asc, dims=[0]).detach().cpu().numpy()  # 反转成降序

    period = x.shape[1] // top_list
    weights = abs(xf).mean(-1)[:, top_list]
    return period, weights


# -------------------------- 6. TimesBlock（无修改） --------------------------
class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff, num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model, num_kernels=configs.num_kernels)
        )
        self.period_detector = MultiScalePeriodDetector(configs)

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = self.period_detector(x)

        res = []
        for i in range(self.k):
            period = period_list[i]
            if (self.seq_len + self.pred_len) % period != 0:
                length = ((self.seq_len + self.pred_len) // period) + 1
                length *= period
                padding = torch.zeros([x.shape[0], length - (self.seq_len + self.pred_len), x.shape[2]],
                                    device=x.device, dtype=x.dtype)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x

            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])

        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        res = res + x
        return res


# -------------------------- 7. Model类（无修改） --------------------------
class Model(nn.Module):
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
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        dec_out = self.projection(enc_out)

        dec_out = dec_out.mul(stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out.add(means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc.sub(means)
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / (torch.sum(mask == 1, dim=1) + 1e-5))
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc.div(stdev)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        dec_out = self.projection(enc_out)

        dec_out = dec_out.mul(stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out.add(means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
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