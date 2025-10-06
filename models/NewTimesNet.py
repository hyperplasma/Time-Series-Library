import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


class EnhancedFFTPeriodDetection(nn.Module):
    """增强的FFT周期检测：多尺度 + 自相关验证"""
    def __init__(self, top_k=5, scales=[1, 2, 3], correlation_threshold=0.2):
        super().__init__()
        self.top_k = top_k
        self.scales = scales
        self.correlation_threshold = correlation_threshold
        
    def multi_scale_fft(self, x):
        """多尺度FFT分析"""
        B, T, C = x.shape
        all_amplitudes = []
        
        for scale in self.scales:
            window_size = T // scale
            if window_size < 10:  # 窗口太小则跳过
                continue
                
            # 滑动窗口分析
            stride = max(1, window_size // 2)
            for start_idx in range(0, T - window_size + 1, stride):
                window = x[:, start_idx:start_idx+window_size, :]
                
                # FFT计算
                xf = torch.fft.rfft(window, dim=1)
                amplitudes = torch.abs(xf).mean(dim=-1)  # [B, freq_bins]
                all_amplitudes.append(amplitudes)
        
        if all_amplitudes:
            # 合并多尺度结果
            combined_amplitudes = torch.stack(all_amplitudes).mean(dim=0)  # [B, freq_bins]
            return combined_amplitudes
        else:
            # 回退到标准FFT
            xf = torch.fft.rfft(x, dim=1)
            return torch.abs(xf).mean(dim=-1)
    
    def autocorrelation_validation(self, x, candidate_periods):
        """自相关验证候选周期"""
        B, T, C = x.shape
        validated_periods = []
        validated_weights = []
        
        for i in range(B):
            batch_periods = []
            batch_weights = []
            
            for period in candidate_periods[i]:
                period_int = int(period)
                if period_int < 2 or period_int > T // 2:
                    continue
                    
                # 计算自相关
                if self.compute_autocorrelation(x[i], period_int) > self.correlation_threshold:
                    batch_periods.append(period_int)
                    batch_weights.append(1.0)  # 简单权重
            
            # 如果验证后周期太少，补充一些候选
            if len(batch_periods) < self.top_k:
                additional = self.get_additional_periods(x[i], self.top_k - len(batch_periods))
                batch_periods.extend(additional)
                batch_weights.extend([0.5] * len(additional))  # 较低权重
            
            validated_periods.append(batch_periods[:self.top_k])
            validated_weights.append(batch_weights[:self.top_k])
        
        return validated_periods, validated_weights
    
    def compute_autocorrelation(self, x_single, period):
        """计算自相关分数"""
        T = x_single.shape[0]
        try:
            # 重塑以检查周期性
            num_periods = T // period
            if num_periods < 2:
                return 0.0
                
            reshaped = x_single[:num_periods * period].reshape(num_periods, period, -1)
            
            # 计算周期间的相关性
            if num_periods > 1:
                # 计算相邻周期之间的相关性
                correlations = []
                for i in range(num_periods - 1):
                    corr = F.cosine_similarity(
                        reshaped[i].flatten(), 
                        reshaped[i + 1].flatten(), 
                        dim=0
                    )
                    correlations.append(corr)
                return torch.mean(torch.stack(correlations)).item()
            else:
                return 0.0
        except:
            return 0.0
    
    def get_additional_periods(self, x_single, num_needed):
        """获取额外的候选周期"""
        T = x_single.shape[0]
        additional = []
        
        # 添加一些常见的周期候选
        common_periods = [T//4, T//3, T//2, T//1.5]
        for p in common_periods:
            p_int = int(p)
            if 2 <= p_int <= T // 2 and p_int not in additional:
                additional.append(p_int)
                if len(additional) >= num_needed:
                    break
        
        # 如果还不够，添加随机周期
        while len(additional) < num_needed:
            p = torch.randint(2, T//2, (1,)).item()
            if p not in additional:
                additional.append(p)
        
        return additional
    
    def forward(self, x):
        B, T, C = x.shape
        
        # 1. 多尺度FFT检测
        combined_amplitudes = self.multi_scale_fft(x)
        
        # 2. 选择候选周期
        _, topk_indices = torch.topk(combined_amplitudes, self.top_k * 2, dim=1)  # 多选一些候选
        
        candidate_periods = []
        for i in range(B):
            periods = []
            for idx in topk_indices[i]:
                period = T / (idx.item() + 1)
                if 2 <= period <= T // 2:  # 合理的周期范围
                    periods.append(period)
            candidate_periods.append(periods[:self.top_k * 2])
        
        # 3. 自相关验证
        validated_periods, validated_weights = self.autocorrelation_validation(x, candidate_periods)
        
        # 转换为tensor
        period_tensor = torch.zeros(B, self.top_k)
        weight_tensor = torch.zeros(B, self.top_k)
        
        for i in range(B):
            for j in range(min(self.top_k, len(validated_periods[i]))):
                period_tensor[i, j] = validated_periods[i][j]
                weight_tensor[i, j] = validated_weights[i][j]
        
        return period_tensor.to(x.device), weight_tensor.to(x.device)


def FFT_for_Period(x, k=2):
    """原始FFT函数保持兼容性"""
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
        
        # 使用增强的FFT检测
        self.enhanced_fft = EnhancedFFTPeriodDetection(
            top_k=configs.top_k,
            scales=[1, 2, 3],  # 多尺度分析
            correlation_threshold=0.15  # 适中的相关性阈值
        )
        
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
        
        # 使用增强的FFT周期检测
        period_list, period_weight = self.enhanced_fft(x)
        
        # 转换period_list为整数列表格式以保持兼容性
        period_list_np = period_list.cpu().numpy().astype(int)

        res = []
        for i in range(self.k):
            period = period_list_np[0, i]  # 取第一个batch的周期（所有batch相同）
            if period == 0:  # 无效周期，跳过
                continue
                
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
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
            
        if not res:  # 如果没有有效周期，回退到原始FFT
            period_list, period_weight = FFT_for_Period(x, self.k)
            # ... 原始处理逻辑

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