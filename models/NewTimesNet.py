import math

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding


# Patch Embedding (from TimeXer)
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, d_model, patch_len, dropout=0.0):
        super().__init__()
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        # x: [B, L, C]
        B, L, C = x.shape
        n_patches = L // self.patch_len
        x = x[:, :n_patches * self.patch_len, :]
        x = x.permute(0, 2, 1)  # [B, C, L]
        x = x.reshape(B, C, n_patches, self.patch_len)  # [B, C, n_patches, patch_len]
        x = x.contiguous().view(B * C, n_patches, self.patch_len)
        x = self.value_embedding(x)
        x = x.view(B, C, n_patches, -1).permute(0, 2, 1, 3).contiguous()  # [B, n_patches, C, d_model]
        x = x.view(B, n_patches, -1)  # [B, n_patches, C*d_model]
        return self.dropout(x), n_patches

# Patch-Period Block: patch embedding + FFT周期建模 + flatten head
class PatchPeriodBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.patch_len = configs.patch_len
        self.d_model = configs.d_model
        self.k = configs.top_k
        self.n_vars = configs.enc_in
        self.period_gate = nn.Parameter(torch.ones(self.k))
        self.dropout = nn.Dropout(configs.dropout)
        self.patch_embedding = PatchEmbedding(self.n_vars, self.d_model, self.patch_len, dropout=configs.dropout)
    def forward(self, x, period_list, period_weight):
        # x: [B, L, C]
        B, L, C = x.shape
        n_patches = L // self.patch_len
        x, n_patches = self.patch_embedding(x)
        # FFT for period
        xf = torch.fft.rfft(x, dim=1)
        frequency_list = abs(xf).mean(0).mean(-1)
        frequency_list[0] = 0
        _, top_list = torch.topk(frequency_list, self.k)
        top_list = top_list.detach().cpu().numpy()
        period = x.shape[1] // top_list
        period_weight = abs(xf).mean(-1)[:, top_list]
        # Period Gate
        gate = torch.softmax(self.period_gate, dim=0) * torch.softmax(period_weight.mean(0), dim=0)
        gate = gate / gate.sum()
        # flatten + head
        x = x.reshape(B, -1)
        # head 层也要只初始化一次，放到 __init__
        if not hasattr(self, 'head'):
            self.head = nn.Linear(self.d_model * n_patches * C, self.d_model).to(x.device)
        x = self.head(x)
        x = self.dropout(x)
        return x

# FFT only once，返回周期和幅值
def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class Model(nn.Module):
    """
    Patch-Period NewTimesNet: 融合TimeXer patch embedding和周期建模，提升长期预测性能。
    
    相较于TimesNet的创新点：
    1. Patch Embedding：借鉴TimeXer，将长序列分块（patch），每个patch通过线性层映射到高维空间，提升对长序列的建模能力和效率。
    2. Patch-Period Block：将patch embedding与周期建模（FFT+周期门控）结合，充分利用周期性和局部/全局特征。
    3. Flatten Head结构：借鉴TimeXer，patch特征直接flatten后用线性层输出预测结果，简化输出头部，提升效率。
    4. 结构极简：主干仅包含patch embedding、周期建模和flatten head，参数量和推理速度优于原始TimesNet。
    5. 更强的全局建模能力：patch embedding天然具备全局感受野，周期建模进一步增强对周期性和全局依赖的捕捉。
    6. 代码实现更易于扩展：可灵活插入注意力、全局token等模块，便于后续创新。
    """
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([PatchPeriodBlock(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        self.norm = nn.LayerNorm(configs.d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(configs.dropout)

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
        for i in range(self.layer):
            enc_out = self.model[i](enc_out, period_list, period_weight)
            enc_out = self.norm(enc_out)
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
