import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import pywt
import numpy as np
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


def WPT_for_Period(x, k=2, level=3, wave='db4', energy_threshold=60):
    """
    使用小波包变换检测周期（针对ETT/Weather/Exchange数据集优化）
    Args:
        x: 输入张量 [B, T, C]
        k: 返回top-k个周期
        level: 小波包分解级别（3级=8个子带，适合捕捉日/周多尺度周期）
        wave: 小波函数（db4为工业标准，平衡性能与噪声抑制）
        energy_threshold: 能量阈值百分比（60%过滤中等噪声）
    Returns:
        period: top-k周期数组
        period_weight: 对应的权重 [B, k]
    """
    B, T, C = x.shape
    device = x.device
    dtype = x.dtype
    
    x_np = x.detach().cpu().numpy()
    
    all_periods = []
    all_weights = []
    
    for b in range(B):
        batch_periods = []
        batch_weights = []
        
        for c in range(C):
            signal = x_np[b, :, c]
            
            wp = pywt.WaveletPacket(data=signal, wavelet=wave, mode='symmetric', maxlevel=level)
            
            nodes = [node.path for node in wp.get_level(level, 'freq')]
            
            energies = []
            subbands = []
            for node_path in nodes:
                subband = wp[node_path].data
                energy = np.sum(subband ** 2)
                energies.append(energy)
                subbands.append(subband)
            
            energies = np.array(energies)
            energy_thresh = np.percentile(energies, energy_threshold)
            high_energy_indices = np.where(energies >= energy_thresh)[0]
            
            periods_dict = {}
            for idx in high_energy_indices:
                subband = subbands[idx]
                subband_energy = energies[idx]
                
                fft_vals = np.fft.rfft(subband)
                fft_freqs = np.fft.rfftfreq(len(subband))
                amplitudes = np.abs(fft_vals)
                
                amplitudes[0] = 0
                top_freq_idx = np.argmax(amplitudes)
                
                if fft_freqs[top_freq_idx] > 0:
                    period = int(T / (fft_freqs[top_freq_idx] * (2 ** level)))
                    
                    if 3 <= period <= T:
                        weight = amplitudes[top_freq_idx] * subband_energy
                        if period in periods_dict:
                            periods_dict[period] += weight
                        else:
                            periods_dict[period] = weight
            
            if periods_dict:
                for period, weight in periods_dict.items():
                    batch_periods.append(period)
                    batch_weights.append(weight)
        
        if batch_periods:
            period_weight_dict = {}
            for p, w in zip(batch_periods, batch_weights):
                if p in period_weight_dict:
                    period_weight_dict[p] += w
                else:
                    period_weight_dict[p] = w
            
            sorted_periods = sorted(period_weight_dict.items(), key=lambda x: x[1], reverse=True)
            top_k_periods = sorted_periods[:k]
            
            periods = [p for p, _ in top_k_periods]
            weights = [w for _, w in top_k_periods]
            
            while len(periods) < k:
                periods.append(T // 2)
                weights.append(0.0)
        else:
            periods = [T // (i + 1) for i in range(k)]
            weights = [1.0 / k] * k
        
        all_periods.append(periods)
        all_weights.append(weights)
    
    period_array = np.array(all_periods)
    weight_array = np.array(all_weights)
    
    final_periods = []
    for i in range(k):
        period_col = period_array[:, i]
        unique, counts = np.unique(period_col, return_counts=True)
        final_periods.append(int(unique[np.argmax(counts)]))
    
    period_tensor = np.array(final_periods)
    weight_tensor = torch.tensor(weight_array, dtype=dtype, device=device)
    
    return period_tensor, weight_tensor


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        
        self.use_wpt = True
        self.wpt_level = 3
        self.wave = 'db4'
        self.energy_threshold = 60
        
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        
        if self.use_wpt:
            period_list, period_weight = WPT_for_Period(x, self.k, self.wpt_level, self.wave, self.energy_threshold)
        else:
            period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
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
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        dec_out = self.projection(enc_out)

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc / stdev

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        dec_out = self.projection(enc_out)

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
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

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
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