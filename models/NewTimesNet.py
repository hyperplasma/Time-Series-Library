import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import pywt
import numpy as np
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


class EnhancedFFTPeriodDetection:
    """增强的FFT周期检测"""
    
    def __init__(self, top_k=2, wavelet='db4'):
        self.top_k = top_k
        self.wavelet = wavelet
        
    def __call__(self, x):
        B, T, C = x.shape
        
        # 1. 多分辨率FFT分析
        fft_periods, fft_weights = self._multi_resolution_fft(x)
        
        # 2. 小波辅助验证
        validated_periods, confidence_scores = self._wavelet_validation(x, fft_periods)
        
        # 3. 动态权重调整
        adjusted_weights = self._dynamic_weight_adjustment(fft_weights, confidence_scores)
        
        return validated_periods, adjusted_weights
    
    def _multi_resolution_fft(self, x):
        """多分辨率FFT分析 - 修复尺寸匹配问题"""
        B, T, C = x.shape
        
        # 方法1: 原始FFT
        xf_full = torch.fft.rfft(x, dim=1)
        freq_list_full = abs(xf_full).mean(0).mean(-1)
        freq_list_full[0] = 0
        
        # 方法2: 滑动窗口FFT - 修复版本
        window_size = T // 4
        if window_size > 10:
            local_freqs = []
            for start in range(0, T - window_size, window_size // 2):
                if start + window_size > T:
                    break
                    
                window_data = x[:, start:start + window_size, :]
                xf_local = torch.fft.rfft(window_data, dim=1)
                local_freq = abs(xf_local).mean(0).mean(-1)
                local_freq[0] = 0
                
                # 关键修复: 插值到相同尺寸
                if len(local_freq) != len(freq_list_full):
                    # 使用线性插值匹配尺寸
                    local_freq_resized = F.interpolate(
                        local_freq.unsqueeze(0).unsqueeze(0), 
                        size=len(freq_list_full), 
                        mode='linear',
                        align_corners=False
                    ).squeeze()
                else:
                    local_freq_resized = local_freq
                    
                local_freqs.append(local_freq_resized)
            
            # 聚合局部频率信息
            if local_freqs:
                local_freq_tensor = torch.stack(local_freqs).mean(0)
                # 确保尺寸匹配
                if local_freq_tensor.shape == freq_list_full.shape:
                    freq_list_full = 0.7 * freq_list_full + 0.3 * local_freq_tensor
        
        # 方法3: 多通道协同分析
        channel_correlations = []
        for i in range(min(C, 5)):  # 限制通道数以减少计算
            for j in range(i + 1, min(C, 5)):
                # 修复相关系数计算
                x_i = x[:, :, i].flatten()
                x_j = x[:, :, j].flatten()
                if x_i.std() > 1e-8 and x_j.std() > 1e-8:  # 避免除零
                    corr_matrix = torch.corrcoef(torch.stack([x_i, x_j]))
                    if corr_matrix.shape == (2, 2):
                        corr = corr_matrix[0, 1]
                        if not torch.isnan(corr):
                            channel_correlations.append(corr.abs())
        
        if channel_correlations:
            avg_correlation = torch.mean(torch.stack(channel_correlations))
            # 高相关性时加强主要频率
            if avg_correlation > 0.3:
                freq_list_full = freq_list_full * (1 + 0.2 * avg_correlation)
        
        # 选择top-k频率
        _, top_list = torch.topk(freq_list_full, self.top_k * 2)
        
        # 频率多样性过滤
        filtered_list = self._frequency_diversity_filter(top_list, T)
        
        top_list = filtered_list[:self.top_k].detach().cpu().numpy()
        period = x.shape[1] // top_list
        
        return period, abs(xf_full).mean(-1)[:, filtered_list][:, :self.top_k]
    
    def _frequency_diversity_filter(self, candidates, seq_len):
        """确保频率多样性，避免选择过于相似的周期"""
        filtered = []
        used_periods = set()
        
        for freq_idx in candidates:
            period = seq_len / (freq_idx.item() + 1e-8)
            period_int = int(period)
            
            # 检查周期是否足够不同
            is_diverse = True
            for used_period in used_periods:
                if abs(period_int - used_period) < min(period_int, used_period) * 0.25:
                    is_diverse = False
                    break
            
            if is_diverse and 2 <= period_int <= seq_len // 2:
                filtered.append(freq_idx)
                used_periods.add(period_int)
            
            if len(filtered) >= self.top_k:
                break
        
        return torch.stack(filtered) if filtered else candidates[:self.top_k]
    
    def _wavelet_validation(self, x, fft_periods):
        """小波变换验证FFT检测的周期"""
        B, T, C = x.shape
        
        try:
            x_np = x.detach().cpu().numpy()
            
            confidence_scores = []
            validated_periods = []
            
            for i in range(min(B, 2)):  # 只验证前2个batch以减少计算
                signal = x_np[i, :, 0]  # 使用第一个通道
                
                # 小波变换
                scales = [2, 4, 8, 16, 32, 64]
                coefficients, _ = pywt.cwt(signal, scales, self.wavelet)
                
                # 计算每个尺度的能量
                scale_energies = np.sum(np.abs(coefficients), axis=1)
                
                batch_scores = []
                batch_periods = []
                
                for period in fft_periods[:self.top_k]:
                    period_int = int(period)
                    
                    # 找到最接近的小波尺度
                    closest_scale_idx = np.argmin([abs(period_int - scale*2) for scale in scales])
                    closest_scale = scales[closest_scale_idx]
                    
                    # 计算置信度：周期与小波尺度的一致性
                    scale_energy = scale_energies[closest_scale_idx]
                    max_energy = np.max(scale_energies)
                    
                    if max_energy > 0:
                        confidence = scale_energy / max_energy
                    else:
                        confidence = 0.5  # 默认值
                    
                    batch_scores.append(confidence)
                    batch_periods.append(period_int)
                
                confidence_scores.append(batch_scores)
                validated_periods.append(batch_periods)
            
            # 扩展到所有batch
            final_scores = np.array(confidence_scores)
            final_periods = np.array(validated_periods)
            
            # 如果只有一个batch被处理，复制到所有batch
            if B > 1:
                final_scores = np.tile(final_scores[0], (B, 1))
                final_periods = np.tile(final_periods[0], (B, 1))
            
            return final_periods, torch.tensor(final_scores, dtype=torch.float32, device=x.device)
            
        except Exception as e:
            # 小波失败时返回原始结果
            return fft_periods, torch.ones(B, self.top_k, device=x.device) * 0.7
    
    def _dynamic_weight_adjustment(self, fft_weights, confidence_scores):
        """基于小波置信度动态调整FFT权重"""
        # 结合FFT幅度和小波置信度
        combined_weights = fft_weights * (0.7 + 0.3 * confidence_scores)
        return combined_weights


class EnhancedTimesBlock(nn.Module):
    """真正改进的TimesBlock"""
    
    def __init__(self, configs):
        super(EnhancedTimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.d_model = configs.d_model
        
        # 增强的周期检测
        self.period_detector = EnhancedFFTPeriodDetection(top_k=configs.top_k)
        
        # 保持架构兼容但优化参数效率
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff, num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model, num_kernels=configs.num_kernels)
        )
        
        # 添加周期特定的适配器
        self.period_adapters = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(configs.d_model, configs.d_model // 4),
                nn.ReLU(),
                nn.Linear(configs.d_model // 4, 1)
            ) for _ in range(configs.top_k)
        ])
        
    def forward(self, x):
        B, T, N = x.size()
        
        # 使用增强的周期检测
        period_list, period_weight = self.period_detector(x)
        
        # 确保period_list是numpy数组格式
        if isinstance(period_list, np.ndarray):
            period_array = period_list
        else:
            period_array = period_list.detach().cpu().numpy() if hasattr(period_list, 'detach') else period_list
        
        res = []
        adapter_weights = []
        
        for i in range(self.k):
            # 安全地获取周期值
            if period_array.ndim == 1:
                period = period_array[i]
            else:
                period = period_array[0, i]  # 取第一个batch
            
            # 确保周期是整数
            period = int(period)
            
            # 智能周期调整
            if period < 2 or period > (self.seq_len + self.pred_len) // 2:
                # 基于数据特性的智能后备
                if T > 200:
                    period = max(4, T // (i + 4))
                else:
                    period = max(2, T // (i + 2))
            
            # 动态填充
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
            
            # 2D卷积处理
            out = self.conv(out)
            
            # 周期特定适配
            if i < len(self.period_adapters):
                adapter_weight = torch.sigmoid(self.period_adapters[i](out))  # [B, 1]
                adapter_weights.append(adapter_weight)
            
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
            
        res = torch.stack(res, dim=-1)
        
        # 增强的权重融合
        period_weight = F.softmax(period_weight, dim=1)
        
        # 结合适配器权重
        if adapter_weights:
            adapter_tensor = torch.cat(adapter_weights, dim=1)  # [B, k]
            adapter_tensor = F.softmax(adapter_tensor, dim=1)
            # 动态融合：FFT权重占60%，学习权重占40%
            combined_weights = 0.6 * period_weight + 0.4 * adapter_tensor
        else:
            combined_weights = period_weight
        
        combined_weights = F.softmax(combined_weights, dim=1)
        combined_weights = combined_weights.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        
        res = torch.sum(res * combined_weights, -1)
        res = res + x  # residual connection
        
        return res


class Model(nn.Module):
    """
    改进的TimesNet模型
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        
        # 使用真正改进的TimesBlock
        self.model = nn.ModuleList([EnhancedTimesBlock(configs)
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
        # 保持原有的归一化
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        
        # 改进的TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
            
        dec_out = self.projection(enc_out)
        
        # 反归一化
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(
            1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(
            1, self.pred_len + self.seq_len, 1))
            
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc / stdev

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        dec_out = self.projection(enc_out)

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(
            1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(
            1, self.pred_len + self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        enc_out = self.enc_embedding(x_enc, None)
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        dec_out = self.projection(enc_out)

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(
            1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(
            1, self.pred_len + self.seq_len, 1))
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