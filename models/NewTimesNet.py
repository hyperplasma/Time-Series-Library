import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
import math


class RevIN(nn.Module):
    """Reversible Instance Normalization"""
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class WAVEChannelAttention(nn.Module):
    """
    WAVE: Weighted Autoregressive Varying Gate Attention (2025年最新)
    论文: WAVE: Weighted Autoregressive Varying Gate for Time Series Forecasting
    arxiv: 2410.03159v3 (更新于2025年2月)
    
    核心创新:
    1. AR (Autoregressive) 成分: 捕捉长期依赖
    2. MA (Moving Average) 成分: 捕捉局部模式
    3. Varying Gate: 自适应调节AR和MA的权重
    """
    def __init__(self, num_channels, window_size=3):
        super(WAVEChannelAttention, self).__init__()
        self.num_channels = num_channels
        self.window_size = window_size
        
        # 全局池化（用于AR成分）
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 局部池化（用于MA成分）
        self.local_pool = nn.AvgPool1d(kernel_size=window_size, stride=1, padding=window_size//2)
        
        # AR路径：捕捉全局长期依赖
        self.ar_pathway = nn.Sequential(
            nn.Linear(num_channels, max(num_channels // 4, 1)),
            nn.ReLU(inplace=True),
            nn.Linear(max(num_channels // 4, 1), num_channels)
        )
        
        # MA路径：捕捉局部短期模式
        self.ma_pathway = nn.Sequential(
            nn.Conv1d(num_channels, max(num_channels // 4, 1), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(max(num_channels // 4, 1), num_channels, kernel_size=1)
        )
        
        # Varying Gate：自适应权重调节
        self.gate = nn.Sequential(
            nn.Linear(num_channels * 2, num_channels),
            nn.Sigmoid()
        )
        
        self.sigmoid = nn.Sigmoid()
        
        print(f">>> Using WAVE Attention with window_size={window_size} for {num_channels} channels")

    def forward(self, x):
        # x: [bs x nvars x d_model x patch_num]
        b, c, d, p = x.size()
        x_reshape = x.view(b, c, -1)  # [B, C, D*P]
        
        # AR成分：全局池化 + 全连接
        ar_feat = self.global_pool(x_reshape).squeeze(-1)  # [B, C]
        ar_out = self.ar_pathway(ar_feat)  # [B, C]
        
        # MA成分：局部池化 + 卷积
        ma_feat = self.local_pool(x_reshape)  # [B, C, D*P]
        ma_feat_pooled = self.global_pool(ma_feat).squeeze(-1)  # [B, C]
        ma_out = ma_feat_pooled  # 简化版
        
        # Varying Gate：融合AR和MA
        gate_input = torch.cat([ar_out, ma_out], dim=1)  # [B, 2C]
        gate_weight = self.gate(gate_input)  # [B, C]
        
        # 加权融合
        fused = gate_weight * ar_out + (1 - gate_weight) * ma_out  # [B, C]
        
        # Sigmoid激活
        attention_weights = self.sigmoid(fused).view(b, c, 1, 1)  # [B, C, 1, 1]
        
        return x * attention_weights.expand_as(x)


class ECAAttention(nn.Module):
    """ECA-Net (备选)"""
    def __init__(self, num_channels, gamma=2, b=1):
        super(ECAAttention, self).__init__()
        t = int(abs((math.log(num_channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        print(f">>> ECA kernel size: {k} for {num_channels} channels")

    def forward(self, x):
        b, c, d, p = x.size()
        y = self.avg_pool(x.view(b, c, -1))
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    Enhanced PatchTST with:
    1. RevIN normalization (可开关)
    2. WAVE Attention (2025最新，可开关)
    
    可通过configs.use_revin和configs.use_channel_attn控制
    可通过configs.channel_attn_type选择: 'wave', 'eca'
    """

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride

        # RevIN
        self.use_revin = getattr(configs, 'use_revin', True)
        if self.use_revin:
            self.revin = RevIN(configs.enc_in, affine=True)
            print(f">>> Using RevIN normalization")
        else:
            print(f">>> Using standard normalization")

        # Patch嵌入
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        )

        # Channel Attention
        self.use_channel_attn = getattr(configs, 'use_channel_attn', True)
        if self.use_channel_attn:
            attn_type = getattr(configs, 'channel_attn_type', 'wave')
            if attn_type == 'wave':
                window_size = getattr(configs, 'wave_window_size', 3)
                self.channel_attention = WAVEChannelAttention(configs.enc_in, window_size=window_size)
                print(f">>> Using WAVE Channel Attention")
            elif attn_type == 'eca':
                self.channel_attention = ECAAttention(configs.enc_in)
                print(f">>> Using ECA Channel Attention")
            else:
                raise ValueError(f"Unknown channel attention type: {attn_type}")
        else:
            print(f">>> Not using Channel Attention")

        # Prediction Head
        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_revin:
            x_enc = self.revin(x_enc, 'norm')
        else:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        if self.use_channel_attn:
            enc_out = self.channel_attention(enc_out)

        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_revin:
            dec_out = self.revin(dec_out, 'denorm')
        else:
            dec_out = dec_out * \
                      (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + \
                      (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        if self.use_channel_attn:
            enc_out = self.channel_attention(enc_out)

        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        if self.use_revin:
            x_enc = self.revin(x_enc, 'norm')
        else:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        if self.use_channel_attn:
            enc_out = self.channel_attention(enc_out)

        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_revin:
            dec_out = self.revin(dec_out, 'denorm')
        else:
            dec_out = dec_out * \
                      (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
            dec_out = dec_out + \
                      (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        if self.use_revin:
            x_enc = self.revin(x_enc, 'norm')
        else:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        if self.use_channel_attn:
            enc_out = self.channel_attention(enc_out)

        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out
        return None