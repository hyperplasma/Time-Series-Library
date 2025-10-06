import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import pywt
import numpy as np
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


class MultiScalePeriodDetection:
    """多尺度周期检测：FFT + 小波变换融合"""
    
    def __init__(self, top_k=2, wavelet='db4', scales=None):
        self.top_k = top_k
        self.wavelet = wavelet
        
        # 默认小波尺度
        if scales is None:
            self.scales = [2, 4, 8, 16, 32, 64]
        else:
            self.scales = scales
    
    def __call__(self, x):
        B, T, C = x.shape
        
        # 1. FFT分析
        fft_periods, fft_weights = self._fft_analysis(x)
        
        # 2. 小波变换分析
        wavelet_periods, wavelet_weights = self._wavelet_analysis(x)
        
        # 3. 多方法融合
        fused_periods, fused_weights = self._fuse_periods(
            fft_periods, fft_weights, wavelet_periods, wavelet_weights, T
        )
        
        return fused_periods, fused_weights
    
    def _fft_analysis(self, x):
        """FFT频域分析"""
        # [B, T, C]
        xf = torch.fft.rfft(x, dim=1)
        # find period by amplitudes
        frequency_list = abs(xf).mean(0).mean(-1)
        frequency_list[0] = 0
        _, top_list = torch.topk(frequency_list, self.top_k * 2)  # 多选一些候选
        
        top_list = top_list.detach().cpu().numpy()
        period = x.shape[1] // top_list
        
        # 过滤无效周期
        valid_mask = (period >= 2) & (period <= x.shape[1] // 2)
        valid_periods = period[valid_mask][:self.top_k]
        valid_weights = abs(xf).mean(-1)[:, top_list[valid_mask]][:, :self.top_k]
        
        return valid_periods, valid_weights
    
    def _wavelet_analysis(self, x):
        """小波变换时频分析"""
        B, T, C = x.shape
        device = x.device
        
        # 转换为numpy进行小波变换
        x_np = x.detach().cpu().numpy()
        
        period_candidates = []
        energy_weights = []
        
        for i in range(B):
            batch_periods = []
            batch_energies = []
            
            for c in range(C):
                signal = x_np[i, :, c]
                
                try:
                    # 连续小波变换
                    coefficients, frequencies = pywt.cwt(signal, self.scales, self.wavelet)
                    
                    # 计算每个尺度的能量
                    scale_energies = np.sum(np.abs(coefficients), axis=1)
                    
                    # 选择能量最高的尺度
                    top_scale_indices = np.argsort(scale_energies)[-self.top_k:][::-1]
                    
                    for scale_idx in top_scale_indices:
                        scale = self.scales[scale_idx]
                        # 尺度转换为周期（近似关系）
                        period = max(2, min(int(scale * 2), T // 2))
                        energy = scale_energies[scale_idx]
                        
                        if period not in batch_periods:
                            batch_periods.append(period)
                            batch_energies.append(energy)
                            
                except Exception as e:
                    # 小波变换失败时使用默认值
                    continue
            
            # 如果小波分析没有结果，使用默认周期
            if not batch_periods:
                batch_periods = [max(2, T // (i+2)) for i in range(self.top_k)]
                batch_energies = [1.0] * self.top_k
            
            period_candidates.append(batch_periods[:self.top_k])
            energy_weights.append(batch_energies[:self.top_k])
        
        # 转换为tensor
        period_tensor = np.array(period_candidates)
        weight_tensor = torch.tensor(energy_weights, dtype=torch.float32, device=device)
        
        return period_tensor, weight_tensor
    
    def _fuse_periods(self, fft_periods, fft_weights, wavelet_periods, wavelet_weights, seq_len):
        """融合FFT和小波分析的周期结果"""
        B = fft_weights.shape[0]
        
        fused_periods = []
        fused_weights = []
        
        for i in range(B):
            # 合并周期候选
            all_periods = []
            all_weights = []
            
            # FFT周期
            for j in range(min(len(fft_periods), self.top_k)):
                period = fft_periods[j]
                weight = fft_weights[i, j].item()
                all_periods.append(period)
                all_weights.append(weight * 1.2)  # FFT权重稍高
            
            # 小波周期
            for j in range(min(len(wavelet_periods[i]), self.top_k)):
                period = wavelet_periods[i][j]
                weight = wavelet_weights[i, j].item()
                if period not in all_periods:  # 避免重复
                    all_periods.append(period)
                    all_weights.append(weight)
            
            # 选择top-k个周期
            if len(all_periods) > self.top_k:
                # 按权重排序
                sorted_indices = np.argsort(all_weights)[-self.top_k:][::-1]
                final_periods = [all_periods[idx] for idx in sorted_indices]
                final_weights = [all_weights[idx] for idx in sorted_indices]
            else:
                final_periods = all_periods
                final_weights = all_weights
            
            # 确保周期数量为top_k
            while len(final_periods) < self.top_k:
                default_period = max(2, seq_len // (len(final_periods) + 2))
                final_periods.append(default_period)
                final_weights.append(0.1)  # 较低的默认权重
            
            fused_periods.append(final_periods[:self.top_k])
            fused_weights.append(final_weights[:self.top_k])
        
        period_array = np.array(fused_periods)
        weight_tensor = torch.tensor(fused_weights, dtype=torch.float32, device=fft_weights.device)
        
        return period_array, weight_tensor


class EnhancedInceptionBlock(nn.Module):
    """增强的Inception块：深度可分离卷积 + 多尺度"""
    
    def __init__(self, in_channels, out_channels, num_kernels=3):
        super(EnhancedInceptionBlock, self).__init__()
        self.num_kernels = num_kernels
        kernels = [3, 5, 7][:num_kernels]
        
        # 深度可分离卷积
        self.depthwise_convs = nn.ModuleList()
        self.pointwise_convs = nn.ModuleList()
        
        for kernel_size in kernels:
            # 深度卷积
            self.depthwise_convs.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=(1, kernel_size), 
                         padding=(0, kernel_size//2), groups=in_channels)
            )
            # 点卷积
            self.pointwise_convs.append(
                nn.Conv2d(in_channels, out_channels // num_kernels, kernel_size=1)
            )
        
        # 自适应权重
        self.adaptive_weights = nn.AdaptiveAvgPool2d(1)
        self.weight_net = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, num_kernels),
            nn.Softmax(dim=-1)
        )
        
        self.gelu = nn.GELU()
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 多尺度深度可分离卷积
        outputs = []
        for depth_conv, point_conv in zip(self.depthwise_convs, self.pointwise_convs):
            out = depth_conv(x)
            out = self.gelu(out)
            out = point_conv(out)
            outputs.append(out)
        
        # 自适应权重融合
        if self.num_kernels > 1:
            weight_input = self.adaptive_weights(x).view(B, C)
            weights = self.weight_net(weight_input)  # [B, num_kernels]
            
            weighted_outputs = []
            for i, out in enumerate(outputs):
                weight = weights[:, i].view(B, 1, 1, 1)
                weighted_outputs.append(out * weight)
            
            out = torch.cat(weighted_outputs, dim=1)
        else:
            out = outputs[0]
        
        return out


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        
        # 多尺度周期检测
        self.period_detector = MultiScalePeriodDetection(
            top_k=configs.top_k,
            wavelet='db4'
        )
        
        # 增强的卷积块
        self.conv = nn.Sequential(
            EnhancedInceptionBlock(configs.d_model, configs.d_ff, num_kernels=configs.num_kernels),
            nn.GELU(),
            EnhancedInceptionBlock(configs.d_ff, configs.d_model, num_kernels=configs.num_kernels)
        )
        
        # 周期置信度网络
        self.period_confidence = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(configs.d_model, configs.d_model // 4),
            nn.ReLU(),
            nn.Linear(configs.d_model // 4, self.k)
        )

    def forward(self, x):
        B, T, N = x.size()
        
        # 多尺度周期检测
        period_list, period_weight = self.period_detector(x)
        
        res = []
        learned_weights = []
        
        for i in range(self.k):
            period = period_list[0, i] if period_list.ndim > 1 else period_list[i]
            
            # 周期验证和调整
            if period < 2 or period > (self.seq_len + self.pred_len) // 2:
                period = max(2, (self.seq_len + self.pred_len) // (i + 2))
            
            # padding
            total_length = self.seq_len + self.pred_len
            if total_length % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
                
            # reshape
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            
            # 2D conv with enhanced blocks
            out = self.conv(out)
            
            # 学习周期特定权重
            period_feat = self.period_confidence(out)  # [B, k]
            learned_weights.append(period_feat.unsqueeze(-1))
            
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
            
        res = torch.stack(res, dim=-1)
        
        # 智能周期融合
        if learned_weights:
            learned_weight_tensor = torch.stack(learned_weights, dim=-1).squeeze(2)  # [B, k, k] -> [B, k]
            learned_weight_tensor = learned_weight_tensor.mean(dim=1)  # [B, k]
            learned_weight_tensor = F.softmax(learned_weight_tensor, dim=-1)
        else:
            learned_weight_tensor = torch.ones(B, self.k, device=x.device) / self.k
        
        # 结合FFT权重和学习权重
        original_weight = F.softmax(period_weight, dim=1)
        combined_weights = F.softmax(original_weight * learned_weight_tensor, dim=1)
        
        # adaptive aggregation
        combined_weights = combined_weights.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * combined_weights, -1)
        
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
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

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
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(
            1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(
            1, self.pred_len + self.seq_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc / stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(
            1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(
            1, self.pred_len + self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(
            1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(
            1, self.pred_len + self.seq_len, 1))
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
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out
        return None