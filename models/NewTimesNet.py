import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import numpy as np
import pywt
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    # [B, T, C]，保持float32类型
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


def WPT_for_Period(x, k=2, level=2, wave='db4'):
    """
    核心修正：所有张量强制为float32，与模型类型一致
    """
    B, T, C = x.shape
    device = x.device
    dtype = x.dtype  # 获取输入数据类型（通常是float32）
    x_np = x.detach().cpu().numpy().astype(np.float32)  # 强制numpy为float32
    all_periods = []
    all_energies = []

    for b in range(B):
        sample = x_np[b]
        for c in range(C):
            seq = sample[:, c].astype(np.float32)  # 单特征序列保持float32

            # 小波包分解
            wp = pywt.WaveletPacket(data=seq, wavelet=wave, mode='symmetric', maxlevel=level)
            nodes = [node.path for node in wp.get_level(level, 'natural')]

            # 子带能量计算（强制float32）
            subband_energies = {}
            subband_data = {}
            for node in nodes:
                data = wp[node].data.astype(np.float32)  # 子带数据转float32
                energy = torch.sum(torch.tensor(data, dtype=dtype)**2).item()  # 用模型 dtype
                subband_energies[node] = energy
                subband_data[node] = data

            # 筛选高能量子带
            sorted_nodes = sorted(subband_energies.items(), key=lambda x: x[1], reverse=True)
            keep_num = max(1, len(sorted_nodes) // 2)
            valid_nodes = [node for node, _ in sorted_nodes[:keep_num]]

            # 提取周期（全程保持类型一致）
            for node in valid_nodes:
                data = subband_data[node]
                T_sub = len(data)
                if T_sub < 4:
                    continue

                # 子带FFT（强制float32）
                xf_sub = torch.fft.rfft(torch.tensor(data, dtype=dtype))  # 用模型 dtype
                amp_sub = abs(xf_sub)
                amp_sub[0] = 0
                freq_list_sub = amp_sub.cpu().numpy().astype(np.float32)

                # 取top-k频率
                top_idx = np.argsort(freq_list_sub)[-k:][::-1]
                top_freq = top_idx

                # 周期计算与校验
                periods_sub = (T_sub // top_freq) * (2 ** level) if T_sub > 0 else []
                valid_periods = []
                for p in periods_sub:
                    if (19 <= p <= 29) or (148 <= p <= 188) or (p > 2 and p <= T):
                        valid_periods.append(p)
                if not valid_periods:
                    valid_periods = [T_sub // top_freq[0] * (2**level)] if len(top_freq) > 0 else [T//2]

                all_periods.extend(valid_periods)
                all_energies.extend([subband_energies[node]] * len(valid_periods))

    # 周期融合（权重张量强制为模型 dtype）
    if not all_periods:
        return np.array([T//2]*k), torch.ones(B, k, device=device, dtype=dtype)

    unique_periods, counts = np.unique(all_periods, return_counts=True)
    period_energy = {p: 0.0 for p in unique_periods}
    for p, e in zip(all_periods, all_energies):
        period_energy[p] += e
    period_scores = [counts[i] * period_energy[unique_periods[i]] for i in range(len(unique_periods))]
    top_k_idx = np.argsort(period_scores)[-k:][::-1]
    top_periods = unique_periods[top_k_idx].astype(int)

    # 权重张量类型与输入一致
    total_energy = sum(period_energy[p] for p in top_periods)
    weights = np.array([period_energy[p]/total_energy for p in top_periods], dtype=np.float32)
    period_weights = torch.tensor(weights, device=device, dtype=dtype).repeat(B, 1)  # 强制模型 dtype

    return top_periods, period_weights


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.use_wpt = True
        self.wpt_level = 2
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
            period_list, period_weight = WPT_for_Period(x, self.k, self.wpt_level)
        else:
            period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i] if i < len(period_list) else T//2
            if (self.seq_len + self.pred_len) % period != 0:
                length = ((self.seq_len + self.pred_len) // period + 1) * period
                padding = torch.zeros([x.shape[0], length - (self.seq_len + self.pred_len), x.shape[2]],
                                     device=x.device, dtype=x.dtype)  # 保持dtype一致
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

        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))  # 此处类型需一致
        dec_out = self.projection(enc_out)

        dec_out = dec_out.mul((stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add((means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc.sub(means)
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc.div(stdev)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        dec_out = self.projection(enc_out)

        dec_out = dec_out.mul((stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add((means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)))
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

        dec_out = dec_out.mul((stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add((means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)))
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