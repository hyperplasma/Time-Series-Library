import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1
from torch.utils.checkpoint import checkpoint

# 导入pytorch-wavelets库中的小波变换模块
try:
    from pytorch_wavelets import DWT1DForward, DWT1DInverse
    HAS_WAVELETS = True
except ImportError:
    HAS_WAVELETS = False
    print("Warning: pytorch-wavelets not installed. Please install with: pip install pytorch-wavelets")


def FFT_for_Period(x, k=2):
    """优化的FFT周期检测，全GPU计算"""
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    
    # 修复负步长切片问题
    _, top_k_asc = torch.topk(frequency_list, k)
    top_list = torch.flip(top_k_asc, dims=[0])  # 用flip替代[::-1]
    
    # 保持在GPU上计算，避免CPU传输
    period = (x.shape[1] // top_list).to(x.device)
    period_weights = abs(xf).mean(-1)[:, top_list]
    
    return period, period_weights


class WaveletPeriodDetection(nn.Module):
    """GPU加速的小波变换周期检测模块"""
    def __init__(self, wavelet='db4', mode='symmetric', use_checkpoint=True):
        super(WaveletPeriodDetection, self).__init__()
        self.wavelet = wavelet
        self.mode = mode
        self.use_checkpoint = use_checkpoint
        
        # 修复DWT1D参数错误：删除不存在的level参数
        if HAS_WAVELETS:
            self.dwt = DWT1DForward(wave=wavelet, J=1, mode=mode)
        else:
            self.dwt = None

    def forward(self, x, k=2):
        """
        输入: x [B, T, C]
        输出: period [k], period_weights [B, k]
        """
        if self.dwt is None:
            # 回退到FFT方法
            return FFT_for_Period(x, k)
        
        B, T, C = x.shape
        
        # 调整输入形状以适应DWT1D: [B, C, T]
        x_permuted = x.permute(0, 2, 1)
        
        def _compute_wavelet(x_input):
            # 小波分解 - 全GPU计算
            cA, cD = self.dwt(x_input)
            # 仅使用低频分量cA进行周期检测
            cA_permuted = cA.permute(0, 2, 1)  # [B, T//2, C]
            
            # FFT周期检测
            periods, weights = FFT_for_Period(cA_permuted, k)
            return periods, weights
        
        if self.use_checkpoint and self.training:
            # 使用梯度检查点减少内存
            periods, period_weights = checkpoint(_compute_wavelet, x_permuted)
        else:
            periods, period_weights = _compute_wavelet(x_permuted)
            
        return periods, period_weights


class VMDPeriodDetection(nn.Module):
    """纯PyTorch实现的VMD周期检测（全GPU计算）"""
    def __init__(self, K=2, alpha=2000, tau=0.0, tol=1e-7, max_iter=30, use_checkpoint=True):
        super(VMDPeriodDetection, self).__init__()
        self.K = K  # 减少IMF数量：3→2
        self.alpha = alpha
        self.tau = tau
        self.tol = tol
        self.max_iter = max_iter  # 减少迭代次数：50→30
        self.use_checkpoint = use_checkpoint

    def forward(self, x, k=2):
        """
        输入: x [B, T, C]
        输出: period [k], period_weights [B, k]
        """
        B, T, C = x.shape
        device = x.device
        
        # 初始化变量
        u_hat = torch.zeros(B, self.K, T // 2 + 1, C, dtype=torch.complex64, device=device)
        u_hat_prev = torch.zeros(B, self.K, T // 2 + 1, C, dtype=torch.complex64, device=device)
        lambda_hat = torch.zeros(B, T // 2 + 1, C, dtype=torch.complex64, device=device)
        
        # 信号的傅里叶变换
        x_hat = torch.fft.rfft(x, dim=1)  # [B, T//2+1, C]
        x_hat = x_hat.unsqueeze(1)  # [B, 1, T//2+1, C]
        
        # 中心频率
        omega = torch.zeros(B, self.K, device=device)
        n = torch.arange(1, T//2+2, device=device).float().unsqueeze(0).unsqueeze(-1)  # [1, T//2+1, 1]
        
        def _vmd_iteration(x_hat_input, u_hat_input, u_hat_prev_input, lambda_hat_input, omega_input, n_input):
            # VMD迭代 - 全PyTorch实现
            batch_size = x_hat_input.shape[0]
            
            for iter_idx in range(self.max_iter):
                # 更新IMF频谱
                sum_u_hat = u_hat_input.sum(dim=1, keepdim=True)  # [B, 1, T//2+1, C]
                numerator = x_hat_input - sum_u_hat + lambda_hat_input.unsqueeze(1) / 2
                denominator = 1 + self.alpha * (n_input - omega_input.unsqueeze(-1).unsqueeze(-1)) ** 2
                
                # 使用copy_避免创建新张量
                u_hat_prev_input.copy_(u_hat_input)
                u_hat_input = numerator / denominator
                
                # 更新中心频率
                for k_idx in range(self.K):
                    u_hat_k = u_hat_input[:, k_idx]  # [B, T//2+1, C]
                    power = torch.abs(u_hat_k) ** 2
                    omega_num = torch.sum(n_input * power, dim=1)  # [B, C]
                    omega_den = torch.sum(power, dim=1)  # [B, C]
                    omega_input[:, k_idx] = torch.mean(omega_num / (omega_den + 1e-10), dim=-1)
                
                # 更新拉格朗日乘子
                lambda_hat_input += self.tau * (x_hat_input.squeeze(1) - u_hat_input.sum(dim=1))
                
                # 收敛判断（每5轮判断一次）
                if iter_idx % 5 == 4:
                    convergence = torch.mean(torch.abs(u_hat_input - u_hat_prev_input) ** 2)
                    if convergence < self.tol:
                        break
            
            return u_hat_input, omega_input

        if self.use_checkpoint and self.training:
            u_hat, omega = checkpoint(_vmd_iteration, x_hat, u_hat, u_hat_prev, lambda_hat, omega, n)
        else:
            u_hat, omega = _vmd_iteration(x_hat, u_hat, u_hat_prev, lambda_hat, omega, n)
        
        # 周期检测
        all_periods = []
        all_weights = []
        
        for batch_idx in range(B):
            batch_periods = []
            batch_energies = []
            
            for k_idx in range(self.K):
                # IMF时域信号
                u_k = torch.fft.irfft(u_hat[batch_idx, k_idx], n=T, dim=0)  # [T, C]
                u_k = u_k.unsqueeze(0)  # [1, T, C]
                
                # 使用FFT检测周期
                periods, weights = FFT_for_Period(u_k, k)
                batch_periods.append(periods[0])  # 取第一个batch
                batch_energies.append(weights[0].mean())  # 平均能量
            
            # 选择能量最高的k个周期
            energies_tensor = torch.stack(batch_energies)
            topk_energies, topk_indices = torch.topk(energies_tensor, min(k, len(batch_energies)))
            
            selected_periods = torch.stack([batch_periods[i] for i in topk_indices])
            all_periods.append(selected_periods)
            all_weights.append(topk_energies.unsqueeze(0))
        
        # 清理中间变量释放显存
        del u_hat, u_hat_prev, lambda_hat, x_hat
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        periods_tensor = torch.stack(all_periods).mean(dim=0)  # [k]
        weights_tensor = torch.cat(all_weights, dim=0)  # [B, k]
        
        return periods_tensor, weights_tensor


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.use_checkpoint = getattr(configs, 'use_checkpoint', True)
        
        # 选择周期检测方法（默认为改进的小波变换）
        period_detection_method = getattr(configs, 'period_detection', 'wavelet')
        if period_detection_method == 'wavelet':
            self.period_detector = WaveletPeriodDetection(use_checkpoint=self.use_checkpoint)
        elif period_detection_method == 'vmd':
            self.period_detector = VMDPeriodDetection(use_checkpoint=self.use_checkpoint)
        else:  # 默认使用原始FFT
            self.period_detector = None

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
        if self.period_detector is not None:
            period_list, period_weight = self.period_detector(x, self.k)
        else:
            period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # 确保period是整数
            period = int(period.item() if torch.is_tensor(period) else period)
            
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
                
            # reshape
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            
            # 2D conv: 使用梯度检查点减少激活值缓存
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
    Improved TimesNet with GPU-optimized period detection
    OG TimesNet Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        
        # 添加改进相关的配置参数（使用默认值）
        if not hasattr(configs, 'use_checkpoint'):
            configs.use_checkpoint = True
        if not hasattr(configs, 'period_detection'):
            configs.period_detection = 'vmd'  # 'wavelet', 'vmd', or 'fft'
        
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