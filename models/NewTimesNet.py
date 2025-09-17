import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding


# Depthwise Separable Conv Block
class DepthwiseSeparableConv(nn.Module):
    """
    深度可分离卷积
    [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Squeeze-and-Excitation Block
class SEBlock(nn.Module):
    """
    SE块
    [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.size()
        y = x.mean(dim=(2, 3))  # (B, C)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

# RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))
    def forward(self, x):
        # x: (..., d)
        norm = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return self.scale * x / (norm + self.eps)


# FFT only once，返回周期和幅值
def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


# "My new TimesBlock"，使用 Depthwise Separable Conv、SEBlock、Period Gate
class TimesBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        # Depthwise Separable Conv 替换 Inception
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(self.d_model, self.d_ff, kernel_size=3, padding=1),
            nn.ReLU(),
            DepthwiseSeparableConv(self.d_ff, self.d_model, kernel_size=3, padding=1)
        )
        # Squeeze-and-Excitation
        self.se = SEBlock(self.d_model)
        # Period Gate: 可学习门控
        self.period_gate = nn.Parameter(torch.ones(self.k))

    def forward(self, x, period_list, period_weight):
        B, T, N = x.size()
        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = ((self.seq_len + self.pred_len) // period + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]], device=x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            # reshape
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            # 2D conv
            out = self.conv(out)
            # SEBlock
            out = self.se(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)  # [B, L, N, k]
        # Period Gate: 可学习门控+softmax
        gate = torch.softmax(self.period_gate, dim=0) * torch.softmax(period_weight.mean(0), dim=0)
        gate = gate / gate.sum()  # 归一化
        gate = gate.view(1, 1, 1, self.k)
        res = torch.sum(res * gate, -1)
        # 残差
        res = res + x
        return res


class Model(nn.Module):
    """
    "NewTimesNet" made by Hyperplasma: https://www.hyperplasma.top
    1. 用 Depthwise Separable Conv 替换 Inception_Block_V1，极大减少参数。
    2. FFT 只做一次，周期聚合用可学习门控（Period Gate），减少复杂度。
    3. 堆叠层数减半，每层加 Squeeze-and-Excitation 或通道注意力。
    4. 用 RMSNorm 替换 LayerNorm，激活函数用 ReLU。
    5. 支持混合精度训练。
    """
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        # 堆叠层数减半
        self.layer = max(1, configs.e_layers // 2)
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(self.layer)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.norm = RMSNorm(configs.d_model)
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name in ['imputation', 'anomaly_detection']:
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = nn.ReLU()
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)

        # FFT 只做一次
        period_list, period_weight = FFT_for_Period(enc_out, self.model[0].k)
        # TimesNet
        for i in range(self.layer):
            enc_out = self.model[i](enc_out, period_list, period_weight)
            enc_out = self.norm(enc_out)
        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization
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
        period_list, period_weight = FFT_for_Period(enc_out, self.model[0].k)
        for i in range(self.layer):
            enc_out = self.model[i](enc_out, period_list, period_weight)
            enc_out = self.norm(enc_out)
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
        period_list, period_weight = FFT_for_Period(enc_out, self.model[0].k)
        for i in range(self.layer):
            enc_out = self.model[i](enc_out, period_list, period_weight)
            enc_out = self.norm(enc_out)
        dec_out = self.projection(enc_out)
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        return dec_out


    def classification(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, None)
        period_list, period_weight = FFT_for_Period(enc_out, self.model[0].k)
        for i in range(self.layer):
            enc_out = self.model[i](enc_out, period_list, period_weight)
            enc_out = self.norm(enc_out)
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
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out
        return None
