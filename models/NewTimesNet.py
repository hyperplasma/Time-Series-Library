import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from pytorch_wavelets import DWT1D
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


def WPT_for_Period(x, k=2, level=2, wave='db4'):
    # 用DWT1D手动实现2级分解（模拟小波包4个子带，解决导入错误）
    B, T, C = x.shape
    device = x.device
    # 初始化1级DWT（用于多级分解）
    dwt = DWT1D(wave=wave, mode='symmetric').to(device)
    x_reshaped = x.permute(0, 2, 1)  # [B, C, T]（适配DWT1D输入）

    # 2级分解得到4个子带：cA2(低频-低频)、cD2(低频-高频)、cA1D1(高频-低频)、cD1D1(高频-高频)
    # 第1级分解：得到cA1(低频)、cD1(高频)
    cA1, cD1 = dwt(x_reshaped)  # cA1: [B, C, 1, T//2], cD1: [B, C, 1, T//2]
    cA1 = cA1.squeeze(2)  # [B, C, T//2]
    cD1 = cD1.squeeze(2)  # [B, C, T//2]

    # 第2级分解：对cA1分解得到cA2、cD2；对cD1分解得到cA1D1、cD1D1
    cA2, cD2 = dwt(cA1.unsqueeze(2))  # 需补回通道维度适配DWT1D
    cA1D1, cD1D1 = dwt(cD1.unsqueeze(2))
    # 移除冗余维度，得到4个子带（与原小波包子带对应）
    subbands = [
        cA2.squeeze(2),    # 子带1：低频-低频（aa）
        cD2.squeeze(2),    # 子带2：低频-高频（ad）
        cA1D1.squeeze(2),  # 子带3：高频-低频（da）
        cD1D1.squeeze(2)   # 子带4：高频-高频（dd）
    ]

    all_periods = []
    all_amps = []
    # 遍历4个子带，提取周期
    for subband in subbands:
        T_sub = subband.shape[2]  # 子带长度：T//(2^level)
        # 子带FFT（按时间维度）
        xf_sub = torch.fft.rfft(subband, dim=2)  # [B, C, F]
        amp_sub = abs(xf_sub).mean(1)  # 按通道平均振幅：[B, F]
        freq_list_sub = amp_sub.mean(0)  # 按样本平均：[F]
        freq_list_sub[0] = 0  # 排除直流分量

        # 取子带top-k频率
        if len(freq_list_sub) <= k:
            top_list_sub = torch.argsort(freq_list_sub, descending=True)
        else:
            _, top_list_sub = torch.topk(freq_list_sub, k)

        # 映射回原始序列周期（子带周期 × 2^level）
        periods_sub = (T_sub // top_list_sub) * (2 ** level)
        all_periods.extend(periods_sub.cpu().numpy().tolist())

        # 记录子带振幅（用于权重计算）
        sub_amps = amp_sub[:, top_list_sub]  # [B, k]
        all_amps.append(sub_amps)

    # 融合周期（取频次最高的top-k）
    unique_periods, counts = torch.unique(torch.tensor(all_periods), return_counts=True)
    if len(unique_periods) < k:
        # 候选周期不足时，用序列长度一半填充
        pad_num = k - len(unique_periods)
        pad_periods = torch.tensor([T//2]*pad_num, device=unique_periods.device)
        unique_periods = torch.cat([unique_periods, pad_periods])
        counts = torch.cat([counts, torch.ones(pad_num, device=counts.device)])
    _, top_k_idx = torch.topk(counts, k)
    top_periods = unique_periods[top_k_idx].int().cpu().numpy()

    # 计算周期权重（子带振幅平均）
    all_amps = torch.cat(all_amps, dim=1)  # [B, 4k]
    period_weights = all_amps.mean(dim=1, keepdim=True).repeat(1, k)  # [B, k]

    return top_periods, period_weights


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # 小波包开关（默认关闭，兼容原版）
        self.use_wpt = False  # True启用WPT，False用原版FFT
        self.wpt_level = 2  # 固定2级分解，无需修改
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
        # 选择周期检测方式（模块化切换）
        if self.use_wpt:
            period_list, period_weight = WPT_for_Period(x, self.k, self.wpt_level)
        else:
            period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
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
