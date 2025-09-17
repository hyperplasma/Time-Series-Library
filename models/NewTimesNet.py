
# ========== 重新实现，严格对齐 TSLib/TimesNet 接口，创新点保留 ===========
import torch
import torch.fft
import torch.nn as nn
from layers.Embed import DataEmbedding


class PatchEmbedding(nn.Module):
    """
    Patch Embedding: [B, L, C] -> [B, n_patches, d_model]
    """
    def __init__(self, in_channels, d_model, patch_len, dropout=0.0):
        super().__init__()
        self.patch_len = patch_len
        self.d_model = d_model
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, C]
        B, L, C = x.shape
        n_patches = L // self.patch_len
        x = x[:, :n_patches * self.patch_len, :]  # 截断多余部分
        x = x.permute(0, 2, 1)  # [B, C, L]
        x = x.reshape(B * C, n_patches, self.patch_len)  # [B*C, n_patches, patch_len]
        x = self.value_embedding(x)  # [B*C, n_patches, d_model]
        x = x.view(B, C, n_patches, self.d_model).permute(0, 2, 1, 3).contiguous()  # [B, n_patches, C, d_model]
        x = x.view(B, n_patches, C * self.d_model)  # [B, n_patches, C*d_model]
        return self.dropout(x), n_patches

class PatchPeriodBlock(nn.Module):
    """
    Patch+Period Block: patch embedding + FFT周期建模 + flatten head
    """
    def __init__(self, configs):
        super().__init__()
        self.patch_len = configs.patch_len
        self.d_model = configs.d_model
        self.k = configs.top_k
        self.n_vars = configs.enc_in
        self.dropout = nn.Dropout(configs.dropout)
        self.patch_embedding = PatchEmbedding(self.n_vars, self.d_model, self.patch_len, dropout=configs.dropout)
        # 不再用固定 head 层，改为 mean pooling + projection
        self.proj = nn.Linear(self.d_model * self.n_vars, self.d_model)
        self.period_gate = nn.Parameter(torch.ones(self.k))

    def forward(self, x):
        # x: [B, L, C]
        B, L, C = x.shape
        x, n_patches = self.patch_embedding(x)  # [B, n_patches, C*d_model]
        # FFT for period
        xf = torch.fft.rfft(x, dim=1)
        frequency_list = abs(xf).mean(0).mean(-1)
        frequency_list[0] = 0
        freq_len = frequency_list.shape[0]
        k = min(self.k, freq_len)
        _, top_list = torch.topk(frequency_list, k)
        period_weight = abs(xf).mean(-1)[:, top_list]
        # Period Gate
        gate = torch.softmax(self.period_gate[:k], dim=0) * torch.softmax(period_weight.mean(0), dim=0)
        gate = gate / gate.sum()
        # mean pooling over patch 维度
        x = x.mean(dim=1)  # [B, C*d_model]
        x = self.proj(x)   # [B, d_model]
        x = self.dropout(x)
        return x

def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

class Model(nn.Module):
    """
    Patch-Period NewTimesNet: Patch Embedding + FFT周期建模 + flatten head
    接口、参数、输入输出 shape 与 TimesNet/TSLib 完全一致
    """
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.d_model = configs.d_model
        self.e_layers = configs.e_layers
        self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, configs.embed, configs.freq, configs.dropout)
        self.blocks = nn.ModuleList([PatchPeriodBlock(configs) for _ in range(self.e_layers)])
        self.norm = nn.LayerNorm(self.d_model)
        self.projection = nn.Linear(self.d_model, self.c_out, bias=True)
        self.dropout = nn.Dropout(configs.dropout)
        self.act = nn.GELU()

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, L, d_model]
        # Patch-Period Block
        for block in self.blocks:
            enc_out = block(enc_out)
            enc_out = self.norm(enc_out)
        # [B, d_model]
        dec_out = self.projection(enc_out).unsqueeze(1)  # [B, 1, c_out]
        # De-Normalization
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1)
        # 输出 shape [B, pred_len, c_out]，直接复制预测长度
        dec_out = dec_out.repeat(1, self.pred_len, 1)
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
        for block in self.blocks:
            enc_out = block(enc_out)
            enc_out = self.norm(enc_out)
        dec_out = self.projection(enc_out).unsqueeze(1)
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1)
        dec_out = dec_out.repeat(1, self.pred_len, 1)
        return dec_out

    def anomaly_detection(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        enc_out = self.enc_embedding(x_enc, None)
        for block in self.blocks:
            enc_out = block(enc_out)
            enc_out = self.norm(enc_out)
        dec_out = self.projection(enc_out).unsqueeze(1)
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1)
        dec_out = dec_out.repeat(1, self.pred_len, 1)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        for block in self.blocks:
            enc_out = block(enc_out)
            enc_out = self.norm(enc_out)
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
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
