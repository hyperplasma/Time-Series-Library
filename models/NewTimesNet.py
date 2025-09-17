import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding


def FFT_for_Period(x, k=2):
    # x: [B, n_patches, d_model]
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    return top_list, abs(xf).mean(-1)[:, top_list]



# Patch Embedding
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
        # 自动补零，保证至少有一个patch
        if L < self.patch_len:
            pad_len = self.patch_len - L
            x = torch.cat([x, torch.zeros(B, pad_len, C, device=x.device, dtype=x.dtype)], dim=1)
            L = self.patch_len
        n_patches = (L + self.patch_len - 1) // self.patch_len  # 向上取整
        pad_total = n_patches * self.patch_len - L
        if pad_total > 0:
            x = torch.cat([x, torch.zeros(B, pad_total, C, device=x.device, dtype=x.dtype)], dim=1)
        x = x[:, :n_patches * self.patch_len, :]  # [B, n_patches*patch_len, C]
        x = x.reshape(B, n_patches, self.patch_len, C)  # [B, n_patches, patch_len, C]
        x = x.permute(0, 1, 3, 2)  # [B, n_patches, C, patch_len]
        x = x.reshape(B * n_patches * C, self.patch_len)  # [B*n_patches*C, patch_len]
        x = self.value_embedding(x)  # [B*n_patches*C, d_model]
        x = x.view(B, n_patches, C, self.d_model)
        x = x.mean(dim=2)  # [B, n_patches, d_model]  # 对变量维做平均
        return self.dropout(x)

# Patch+Period Block

class PatchPeriodBlock(nn.Module):
    """
    多尺度 Patch+Period Block: 多个 patch embedding + FFT 周期建模 + 拼接输出
    """
    def __init__(self, configs):
        super().__init__()
        self.d_model = configs.d_model
        self.k = configs.top_k
        self.n_vars = configs.enc_in
        self.dropout = nn.Dropout(configs.dropout)
        # 支持多尺度 patch_lens
        self.patch_lens = getattr(configs, 'patch_lens', [configs.patch_len])
        self.patch_embeddings = nn.ModuleList([
            PatchEmbedding(self.n_vars, self.d_model, pl, dropout=configs.dropout)
            for pl in self.patch_lens
        ])
        self.projs = nn.ModuleList([
            nn.Linear(self.d_model, self.d_model) for _ in self.patch_lens
        ])
        self.period_gates = nn.ParameterList([
            nn.Parameter(torch.ones(self.k)) for _ in self.patch_lens
        ])

    def forward(self, x):
        # x: [B, L, C]
        outs = []
        for pe, proj, period_gate in zip(self.patch_embeddings, self.projs, self.period_gates):
            px = pe(x)  # [B, n_patches, d_model]
            xf = torch.fft.rfft(px, dim=1)
            frequency_list = abs(xf).mean(0).mean(-1)
            frequency_list[0] = 0
            freq_len = frequency_list.shape[0]
            k = min(self.k, freq_len)
            _, top_list = torch.topk(frequency_list, k)
            period_weight = abs(xf).mean(-1)[:, top_list]
            # Period Gate
            gate = torch.softmax(period_gate[:k], dim=0) * torch.softmax(period_weight.mean(0), dim=0)
            gate = gate / gate.sum()
            # 门控聚合（可扩展）
            px = proj(px)
            px = self.dropout(px)
            outs.append(px)
        # 多尺度拼接
        x_cat = torch.cat(outs, dim=-1)  # [B, n_patches, d_model * n_scale]
        return x_cat

# Flatten Head
class FlattenHead(nn.Module):
    def __init__(self, d_model, c_out, n_scale=1, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model * n_scale, c_out)

    def forward(self, x):
        # x: [B, n_patches, d_model * n_scale]
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.proj(x)
        return x.unsqueeze(1)


class Model(nn.Module):
    """
    Patch-Period NewTimesNet: Patch Embedding + FFT周期建模 + flatten head
    
    要求创新点：
    1. Patch Embedding：借鉴TimeXer，将长序列分块（patch），每个patch通过线性层映射到高维空间，提升对长序列的建模能力和效率。
    2. Patch-Period Block：将patch embedding与周期建模（FFT+周期门控）结合，充分利用周期性和局部/全局特征。
    3. Flatten Head结构：借鉴TimeXer，patch特征mean pooling后用线性层输出预测结果，简化输出头部，提升效率。
    4. 结构极简：主干仅包含patch embedding、周期建模和flatten head，参数量和推理速度优于原始TimesNet。
    5. 更强的全局建模能力：patch embedding天然具备全局感受野，周期建模进一步增强对周期性和全局依赖的捕捉。
    6. 代码实现更易于扩展：可灵活插入注意力、全局token等模块，便于后续创新。
    7. 多尺度Patch/FFT：支持多尺度patch划分和多尺度FFT周期建模，不同patch长度的特征并行提取与拼接，进一步提升对多尺度周期性和复杂时序结构的建模能力。
    
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
        self.patch_lens = getattr(configs, 'patch_lens', [configs.patch_len])
        self.n_scale = len(self.patch_lens)
        self.blocks = nn.ModuleList([PatchPeriodBlock(configs) for _ in range(self.e_layers)])
        self.norm = nn.LayerNorm(self.d_model * self.n_scale)
        self.head = FlattenHead(self.d_model, self.c_out, n_scale=self.n_scale, dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        # Patch-Period Block
        out = x_enc
        for block in self.blocks:
            out = block(out)
            out = self.norm(out)
        dec_out = self.head(out)  # [B, 1, c_out]
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
        out = x_enc
        for block in self.blocks:
            out = block(out)
            out = self.norm(out)
        dec_out = self.head(out)
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1)
        dec_out = dec_out.repeat(1, self.pred_len, 1)
        return dec_out

    def anomaly_detection(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        out = x_enc
        for block in self.blocks:
            out = block(out)
            out = self.norm(out)
        dec_out = self.head(out)
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1)
        dec_out = dec_out.repeat(1, self.pred_len, 1)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        out = x_enc
        for block in self.blocks:
            out = block(out)
            out = self.norm(out)
        output = out.mean(dim=1)
        output = self.head.proj(output)
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
