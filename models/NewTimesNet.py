import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


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


class EnhancedFFTPeriodDetection:
    """增强的FFT周期检测"""
    
    def __init__(self, top_k=2):
        self.top_k = top_k
        
    def __call__(self, x):
        B, T, C = x.shape
        
        # 1. 基础FFT分析
        xf = torch.fft.rfft(x, dim=1)
        amplitudes = torch.abs(xf).mean(dim=-1)  # [B, freq_bins]
        amplitudes[:, 0] = 0  # 移除直流分量
        
        # 2. 多频率融合策略
        # 使用滑动窗口分析频率稳定性
        window_size = min(32, T // 4)
        if window_size > 4:
            stability_scores = self._compute_frequency_stability(x, window_size)
            amplitudes = amplitudes * (1 + 0.2 * stability_scores.unsqueeze(0))
        
        # 3. 改进的top-k选择
        frequency_list = amplitudes.mean(0)  # [freq_bins]
        
        # 避免选择过于接近的频率
        _, top_list = torch.topk(frequency_list, self.top_k * 3)  # 选择更多候选
        top_list = self._filter_similar_frequencies(top_list, frequency_list, T)
        top_list = top_list[:self.top_k]  # 取前k个
        
        top_list = top_list.detach().cpu().numpy()
        period = x.shape[1] // top_list
        
        return period, abs(xf).mean(-1)[:, top_list]
    
    def _compute_frequency_stability(self, x, window_size):
        """计算频率稳定性得分"""
        B, T, C = x.shape
        stability_scores = []
        
        for start_idx in range(0, T - window_size, window_size // 2):
            if start_idx + window_size > T:
                break
                
            window_data = x[:, start_idx:start_idx + window_size, :]
            xf_window = torch.fft.rfft(window_data, dim=1)
            amps_window = torch.abs(xf_window).mean(dim=-1).mean(dim=0)
            
            if len(stability_scores) == 0:
                stability_scores = amps_window
            else:
                stability_scores += amps_window
        
        if len(stability_scores) > 0:
            stability_scores = stability_scores / (len(range(0, T - window_size, window_size // 2)) + 1e-8)
        
        return stability_scores if len(stability_scores) > 0 else torch.zeros(x.shape[1] // 2 + 1, device=x.device)
    
    def _filter_similar_frequencies(self, candidates, frequency_list, seq_len):
        """过滤相似频率，确保多样性"""
        filtered = []
        used_periods = set()
        
        for freq_idx in candidates:
            period = seq_len / (freq_idx.item() + 1e-8)
            period_int = int(period)
            
            # 检查是否与已选周期太接近
            too_close = False
            for used_period in used_periods:
                if abs(period_int - used_period) < min(period_int, used_period) * 0.3:  # 30%阈值
                    too_close = True
                    break
            
            if not too_close and period_int >= 2 and period_int <= seq_len // 2:
                filtered.append(freq_idx)
                used_periods.add(period_int)
            
            if len(filtered) >= self.top_k:
                break
        
        # 如果过滤后数量不足，补充原始候选
        while len(filtered) < self.top_k and len(filtered) < len(candidates):
            for freq_idx in candidates:
                if freq_idx not in filtered:
                    filtered.append(freq_idx)
                    break
        
        return torch.stack(filtered) if filtered else candidates[:self.top_k]


class AdaptiveTimesBlock(nn.Module):
    """自适应TimesBlock：保持接口但优化内部实现"""
    
    def __init__(self, configs):
        super(AdaptiveTimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        
        # 使用增强的周期检测
        self.period_detector = EnhancedFFTPeriodDetection(top_k=configs.top_k)
        
        # 保持原有的Inception块结构，但添加自适应机制
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff, num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model, num_kernels=configs.num_kernels)
        )
        
        # 添加周期自适应权重
        self.period_attention = nn.Linear(configs.d_model, 1)
        
    def forward(self, x):
        B, T, N = x.size()
        
        # 使用增强的周期检测
        period_list, period_weight = self.period_detector(x)
        
        res = []
        period_weights_refined = []
        
        for i in range(self.k):
            period = period_list[i]
            
            # 改进的周期验证
            if period < 2 or period > T // 2:
                period = max(2, T // (i + 3))  # 更合理的后备周期
            
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
            
            # 2D conv with period-aware processing
            out = self.conv(out)
            
            # 计算周期特定权重
            period_feat = out.mean(dim=[2, 3])  # [B, N]
            period_attn = torch.sigmoid(self.period_attention(period_feat))  # [B, 1]
            period_weights_refined.append(period_attn)
            
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
            
        res = torch.stack(res, dim=-1)
        
        # 改进的自适应聚合
        # 结合原始FFT权重和学习的注意力权重
        original_weight = F.softmax(period_weight, dim=1)  # [B, k]
        
        # 合并学习到的权重
        learned_weights = torch.stack(period_weights_refined, dim=-1).squeeze(1)  # [B, k]
        combined_weights = F.softmax(original_weight * learned_weights, dim=1)
        
        # 扩展维度进行加权求和
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
        
        # 使用改进的TimesBlock
        self.model = nn.ModuleList([AdaptiveTimesBlock(configs)
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