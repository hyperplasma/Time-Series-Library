import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1
from torch.utils.checkpoint import checkpoint

# 导入pytorch-wavelets库中的小波变换模块
try:
    from pytorch_wavelets import DWT1DForward
    HAS_WAVELETS = True
except ImportError:
    HAS_WAVELETS = False
    print("Warning: pytorch-wavelets not installed. Please install with: pip install pytorch-wavelets")


def FFT_for_Period(x, k=2):
    """原始FFT周期检测，保持原版性能"""
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class VMDPeriodDetection(nn.Module):
    """优化版的VMD周期检测，确保性能提升"""
    def __init__(self, K=5, alpha=1000, tau=1e-6, tol=1e-7, max_iter=50, use_checkpoint=True):
        super(VMDPeriodDetection, self).__init__()
        self.K = K  # IMF数量，设置为5以匹配top_k
        self.alpha = alpha  # 带宽参数
        self.tau = tau      # 噪声容忍度
        self.tol = tol      # 收敛容忍度
        self.max_iter = max_iter
        self.use_checkpoint = use_checkpoint

    def vmd_decomposition(self, signal, K=5, alpha=1000, tau=1e-6, max_iter=50):
        """
        优化的VMD分解实现 - 避免in-place操作
        输入: signal [B, T]
        输出: IMFs [B, K, T]
        """
        B, T = signal.shape
        device = signal.device
        
        # 初始化 - 使用requires_grad=False避免梯度问题
        u = torch.zeros(B, K, T, device=device, requires_grad=False)  # IMFs
        u_hat = torch.zeros(B, K, T//2+1, dtype=torch.complex64, device=device, requires_grad=False)  # 频域IMFs
        omega = torch.zeros(B, K, device=device, requires_grad=False)  # 中心频率
        lambda_hat = torch.zeros(B, T//2+1, dtype=torch.complex64, device=device, requires_grad=False)  # 拉格朗日乘子
        
        # 信号傅里叶变换
        signal_hat = torch.fft.rfft(signal, dim=1)
        
        # 频率轴
        freqs = torch.fft.rfftfreq(T, device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, T//2+1]
        
        # 避免in-place操作的迭代
        for iter_idx in range(max_iter):
            # 保存前一次的u_hat用于收敛判断
            u_hat_prev = u_hat.clone()
            
            # 更新IMFs的频域表示
            sum_u_hat = u_hat.sum(dim=1, keepdim=True)  # [B, 1, T//2+1]
            
            # 创建新的u_hat避免in-place操作
            u_hat_new = torch.zeros_like(u_hat)
            
            # VMD核心更新公式 - 避免in-place操作
            for k in range(K):
                # 分子部分
                numerator = signal_hat - sum_u_hat + lambda_hat/2
                numerator = numerator + u_hat[:, k:k+1]  # 加上当前IMF
                
                # 分母部分: 1 + alpha * (freqs - omega)^2
                denominator = 1.0 + alpha * (freqs - omega[:, k:k+1].unsqueeze(-1)) ** 2
                
                # 更新IMF频域表示 - 不使用in-place操作
                u_hat_new[:, k] = numerator.squeeze(1) / denominator.squeeze(1)
            
            # 更新中心频率 - 避免in-place操作
            omega_new = torch.zeros_like(omega)
            for k in range(K):
                u_hat_k = u_hat_new[:, k]  # [B, T//2+1, C]
                power = torch.abs(u_hat_k) ** 2
                omega_num = torch.sum(freqs.squeeze(0).squeeze(0) * power, dim=1)  # [B, C]
                omega_den = torch.sum(power, dim=1)  # [B, C]
                omega_new[:, k] = torch.mean(omega_num / (omega_den + 1e-10), dim=-1)
            
            # 更新拉格朗日乘子 (对偶上升) - 避免in-place操作
            lambda_hat_new = lambda_hat + tau * (signal_hat - u_hat_new.sum(dim=1))
            
            # 更新变量
            u_hat = u_hat_new
            omega = omega_new
            lambda_hat = lambda_hat_new
            
            # 收敛检查
            if iter_idx > 0:
                convergence = torch.mean(torch.abs(u_hat - u_hat_prev) ** 2)
                if convergence < self.tol:
                    break
        
        # 转换回时域
        for k in range(K):
            u[:, k] = torch.fft.irfft(u_hat[:, k], n=T, dim=1)
        
        return u

    def forward(self, x, k=5):
        """
        输入: x [B, T, C]
        输出: period [k], period_weights [B, k]
        """
        B, T, C = x.shape
        device = x.device
        
        # 确保k不超过K
        k = min(k, self.K)
        
        # 选择方差最大的通道作为代表（通常包含最多信息）
        with torch.no_grad():  # 避免梯度问题
            channel_var = x.var(dim=1).mean(dim=1)  # [B]
            representative_channel = channel_var.argmax(dim=0)  # 选择方差最大的批次
        
        # 使用代表性批次进行VMD分解
        rep_signal = x[representative_channel, :, 0]  # [T] 取第一个通道
        rep_signal = rep_signal.unsqueeze(0)  # [1, T]
        
        # VMD分解 - 使用torch.no_grad()避免梯度计算
        with torch.no_grad():
            imfs = self.vmd_decomposition(rep_signal, K=self.K, alpha=self.alpha, 
                                        tau=self.tau, max_iter=self.max_iter)  # [1, K, T]
        
        # 对每个IMF进行周期检测
        imf_periods = []
        imf_energies = []
        
        for i in range(self.K):
            imf = imfs[0, i].unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
            periods, weights = FFT_for_Period(imf, k=1)  # 每个IMF取1个主要周期
            
            if len(periods) > 0:
                # 确保periods是tensor类型
                period_tensor = torch.tensor(periods[0], device=device) if not torch.is_tensor(periods[0]) else periods[0]
                imf_periods.append(period_tensor)
                energy_tensor = torch.tensor(weights[0].mean(), device=device) if not torch.is_tensor(weights[0].mean()) else weights[0].mean()
                imf_energies.append(energy_tensor)
            else:
                imf_periods.append(torch.tensor(T // 2, device=device))
                imf_energies.append(torch.tensor(0.0, device=device))
        
        # 选择能量最高的k个周期
        energies_tensor = torch.stack(imf_energies)
        topk_energies, topk_indices = torch.topk(energies_tensor, min(k, len(imf_energies)))
        
        # 修复：确保所有period都是tensor类型
        selected_periods = torch.stack([imf_periods[i] for i in topk_indices])
        
        # 如果周期数量不足k个，用FFT补充
        if len(selected_periods) < k:
            # 使用原始信号的FFT检测补充周期
            remaining_k = k - len(selected_periods)
            with torch.no_grad():  # 避免梯度问题
                fft_periods, fft_weights = FFT_for_Period(x[representative_channel:representative_channel+1, :, 0:1], k=remaining_k)
            
            # 将FFT检测的周期添加到selected_periods中
            for i in range(len(fft_periods)):
                period_tensor = torch.tensor(fft_periods[i], device=device) if not torch.is_tensor(fft_periods[i]) else fft_periods[i]
                selected_periods = torch.cat([selected_periods, period_tensor.unsqueeze(0)])
            
            # 扩展权重
            period_weights = topk_energies.unsqueeze(0).repeat(B, 1)
            fft_weight_tensor = torch.tensor(fft_weights.mean(dim=1), device=device) if not torch.is_tensor(fft_weights.mean(dim=1)) else fft_weights.mean(dim=1)
            period_weights = torch.cat([period_weights, fft_weight_tensor.unsqueeze(0).repeat(B, 1)], dim=1)
        else:
            period_weights = topk_energies.unsqueeze(0).repeat(B, 1)  # [B, k]
        
        return selected_periods, period_weights


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.use_checkpoint = getattr(configs, 'use_checkpoint', True)
        
        # 使用VMD作为周期检测方法
        self.period_detector = VMDPeriodDetection(use_checkpoint=self.use_checkpoint)

        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        
        # 使用VMD进行周期检测
        period_list, period_weight = self.period_detector(x, self.k)

        # 确保period_list有足够的元素
        if len(period_list) < self.k:
            # 用默认值补充
            default_periods = torch.tensor([T // 2] * (self.k - len(period_list)), device=x.device)
            period_list = torch.cat([period_list, default_periods])
            default_weights = torch.zeros(B, self.k - period_weight.shape[1], device=x.device)
            period_weight = torch.cat([period_weight, default_weights], dim=1)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # 确保period是整数且合理
            period_val = period.item() if torch.is_tensor(period) else period
            period_val = max(2, min(int(period_val), T))  # 限制在[2, T]范围内
            
            # padding
            total_length = self.seq_len + self.pred_len
            if total_length % period_val != 0:
                length = (((total_length) // period_val) + 1) * period_val
                padding = torch.zeros([x.shape[0], (length - total_length), x.shape[2]], device=x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = total_length
                out = x
                
            # reshape
            out = out.reshape(B, length // period_val, period_val, N).permute(0, 3, 1, 2).contiguous()
            
            # 2D conv
            if self.use_checkpoint and self.training:
                out = checkpoint(self.conv, out)
            else:
                out = self.conv(out)
                
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
            
        res = torch.stack(res, dim=-1)
        
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        
        # residual connection
        res = res + x
        return res


class Model(nn.Module):
    """
    Improved TimesNet with VMD-optimized period detection
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        
        # 使用VMD作为默认周期检测方法
        if not hasattr(configs, 'use_checkpoint'):
            configs.use_checkpoint = True
        
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
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out
        return None