import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    """稳定的FFT周期检测"""
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class StableVMD(nn.Module):
    """数值稳定的VMD实现"""
    def __init__(self, K=5, alpha=2000, tau=0, tol=1e-7, max_iter=50):
        super(StableVMD, self).__init__()
        self.K = K
        self.alpha = alpha
        self.tau = tau
        self.tol = tol
        self.max_iter = max_iter

    def forward(self, signal):
        """
        输入: signal [B, T]
        输出: imfs [B, K, T]
        """
        B, T = signal.shape
        device = signal.device
        
        # 信号标准化，避免数值问题
        signal_mean = signal.mean(dim=1, keepdim=True)
        signal_std = signal.std(dim=1, keepdim=True) + 1e-8
        signal_normalized = (signal - signal_mean) / signal_std
        
        # 初始化
        u = torch.zeros(B, self.K, T, device=device)
        u_hat = torch.zeros(B, self.K, T//2+1, dtype=torch.complex64, device=device)
        omega = torch.zeros(B, self.K, device=device)
        lambda_hat = torch.zeros(B, T//2+1, dtype=torch.complex64, device=device)
        
        # 信号傅里叶变换
        x_hat = torch.fft.rfft(signal_normalized, dim=1)
        
        # 频率轴
        freqs = torch.fft.rfftfreq(T, device=device)
        
        # VMD迭代
        for n in range(self.max_iter):
            # 保存前一次迭代结果用于收敛判断
            u_hat_old = u_hat.clone()
            
            # 更新每个IMF
            for k in range(self.K):
                # 计算sum_{i≠k} u_hat_i
                sum_rest = u_hat.sum(dim=1) - u_hat[:, k]
                
                # 更新u_hat_k
                numerator = x_hat - sum_rest + lambda_hat / 2
                denominator = 1 + self.alpha * (freqs - omega[:, k].unsqueeze(1)) ** 2
                u_hat[:, k] = numerator / denominator
            
            # 更新中心频率
            for k in range(self.K):
                # 计算功率谱
                power = torch.abs(u_hat[:, k]) ** 2
                # 避免除零
                denominator = torch.sum(power, dim=1) + 1e-10
                numerator = torch.sum(freqs.unsqueeze(0) * power, dim=1)
                omega[:, k] = numerator / denominator
            
            # 更新拉格朗日乘子
            sum_u_hat = u_hat.sum(dim=1)
            lambda_hat = lambda_hat + self.tau * (x_hat - sum_u_hat)
            
            # 收敛判断
            if n > 0:
                diff = torch.mean(torch.abs(u_hat - u_hat_old) ** 2)
                if diff < self.tol:
                    break
        
        # 转换回时域并恢复原始尺度
        for k in range(self.K):
            u_k = torch.fft.irfft(u_hat[:, k], n=T, dim=1)
            u[:, k] = u_k * signal_std + signal_mean / self.K
        
        return u


class VMDPeriodDetection(nn.Module):
    """基于VMD的稳定周期检测"""
    def __init__(self, K=5):
        super(VMDPeriodDetection, self).__init__()
        self.K = K
        self.vmd = StableVMD(K=K)
        
    def forward(self, x, k=5):
        B, T, C = x.shape
        device = x.device
        
        # 选择信息量最大的通道
        with torch.no_grad():
            channel_power = torch.mean(x ** 2, dim=1)  # [B, C]
            best_channel = torch.argmax(channel_power.mean(dim=0))
            signal = x[:, :, best_channel]  # [B, T]
        
        # VMD分解
        imfs = self.vmd(signal)  # [B, K, T]
        
        # 对每个IMF检测周期
        all_periods = []
        all_weights = []
        
        for batch_idx in range(B):
            batch_periods = []
            batch_energies = []
            
            for imf_idx in range(self.K):
                imf = imfs[batch_idx, imf_idx]  # [T]
                imf = imf.unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
                
                # 检测周期
                periods, weights = FFT_for_Period(imf, k=1)
                if len(periods) > 0:
                    period_val = periods[0]
                    if isinstance(period_val, torch.Tensor):
                        period_val = period_val.item()
                    batch_periods.append(max(2, min(int(period_val), T)))
                    batch_energies.append(float(weights[0].mean().item()))
                else:
                    batch_periods.append(T // 2)
                    batch_energies.append(0.0)
            
            # 选择能量最高的k个周期
            if len(batch_energies) > 0:
                energies_tensor = torch.tensor(batch_energies, device=device)
                topk_energies, topk_indices = torch.topk(energies_tensor, 
                                                        min(k, len(batch_energies)))
                
                selected_periods = [batch_periods[i] for i in topk_indices.cpu().numpy()]
                # 补充不足的周期
                while len(selected_periods) < k:
                    selected_periods.append(T // 2)
                
                all_periods.append(torch.tensor(selected_periods[:k], device=device, dtype=torch.float))  # 改为float类型
                all_weights.append(topk_energies[:k].unsqueeze(0))
            else:
                # 备用方案
                default_periods = [T // 2] * k
                all_periods.append(torch.tensor(default_periods, device=device, dtype=torch.float))  # 改为float类型
                all_weights.append(torch.zeros(1, k, device=device))
        
        if len(all_periods) == 0:
            # 最终备用方案
            period_list = torch.tensor([T // 2] * k, device=device, dtype=torch.float).unsqueeze(0).repeat(B, 1)  # 改为float类型
            period_weight = torch.ones(B, k, device=device) / k
            return period_list[0].long(), period_weight  # 返回时转换为long
        
        # 合并批次结果 - 修复数据类型问题
        period_list_float = torch.stack(all_periods).mean(dim=0)  # [k], 现在是float类型
        period_list = period_list_float.long()  # 转换为整数类型
        period_weight = torch.cat(all_weights, dim=0)  # [B, k]
        
        # 权重归一化
        period_weight = F.softmax(period_weight, dim=1)
        
        return period_list, period_weight


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        
        # 使用VMD周期检测
        self.period_detector = VMDPeriodDetection(K=min(5, configs.top_k))

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
        
        # 周期检测
        period_list, period_weight = self.period_detector(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # 确保period是整数且合理
            period_val = period.item() if torch.is_tensor(period) else period
            period_val = max(2, min(int(period_val), T))
            
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
    TimesNet with VMD-based period detection
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