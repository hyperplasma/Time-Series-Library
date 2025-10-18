import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


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


class ChannelAttention(nn.Module):
    """
    核心创新1：轻量级跨通道注意力
    在保持Channel Independence优势的同时，允许有限的通道间信息交换
    """
    def __init__(self, num_channels, reduction=4):
        super().__init__()
        self.num_channels = num_channels
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction, num_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [bs x nvars x d_model x patch_num]
        b, c, d, p = x.size()
        # 在patch维度上池化
        y = self.avg_pool(x.view(b, c, -1)).view(b, c)  # [bs x nvars]
        # 学习通道权重
        y = self.fc(y).view(b, c, 1, 1)  # [bs x nvars x 1 x 1]
        # 广播相乘
        return x * y.expand_as(x)


class MultiScalePatchEmbedding(nn.Module):
    """
    核心创新2：多尺度Patch嵌入
    使用不同的patch_len捕捉不同时间尺度的模式
    """
    def __init__(self, d_model, patch_lens=[16, 8, 32], stride=8, padding=8, dropout=0.1):
        super().__init__()
        self.patch_lens = patch_lens
        self.num_scales = len(patch_lens)
        
        # 为每个尺度创建独立的patch embedding
        self.patch_embeddings = nn.ModuleList([
            PatchEmbedding(d_model, patch_len, stride, padding, dropout)
            for patch_len in patch_lens
        ])
        
        # 尺度融合权重（可学习）
        self.scale_weights = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)
        
    def forward(self, x):
        # x: [bs, nvars, seq_len]
        multi_scale_features = []
        n_vars = None
        
        for i, patch_embed in enumerate(self.patch_embeddings):
            # 每个尺度的patch embedding
            feat, n_vars = patch_embed(x)  # [bs * nvars, patch_num, d_model]
            multi_scale_features.append(feat)
        
        # 对齐patch数量（填充或截断到最大）
        max_patches = max([f.shape[1] for f in multi_scale_features])
        aligned_features = []
        for feat in multi_scale_features:
            if feat.shape[1] < max_patches:
                # 填充
                padding = torch.zeros(feat.shape[0], max_patches - feat.shape[1], feat.shape[2]).to(feat.device)
                feat = torch.cat([feat, padding], dim=1)
            elif feat.shape[1] > max_patches:
                # 截断
                feat = feat[:, :max_patches, :]
            aligned_features.append(feat)
        
        # 加权融合多尺度特征
        scale_weights = torch.softmax(self.scale_weights, dim=0)
        fused_features = sum([w * f for w, f in zip(scale_weights, aligned_features)])
        
        return fused_features, n_vars


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
    1. RevIN normalization (from original PatchTST paper)
    2. Multi-Scale Patch Embedding (capture different temporal scales)
    3. Channel-wise Attention (limited cross-channel interaction)
    """

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride

        # RevIN
        self.revin = RevIN(configs.enc_in, affine=True)

        # 多尺度Patch嵌入
        self.use_multi_scale = getattr(configs, 'use_multi_scale', True)
        if self.use_multi_scale:
            self.patch_embedding = MultiScalePatchEmbedding(
                configs.d_model, 
                patch_lens=[patch_len // 2, patch_len, patch_len * 2],  # [8, 16, 32]
                stride=stride, 
                padding=padding, 
                dropout=configs.dropout
            )
            # 更新head_nf以匹配multi-scale输出
            max_patch_len = patch_len * 2
            self.head_nf = configs.d_model * int((configs.seq_len - max_patch_len) / stride + 2)
        else:
            self.patch_embedding = PatchEmbedding(
                configs.d_model, patch_len, stride, padding, configs.dropout)
            self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)

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
            self.channel_attention = ChannelAttention(configs.enc_in, reduction=4)

        # Prediction Head
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
        # RevIN normalization
        x_enc = self.revin(x_enc, 'norm')

        # Patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        enc_out, attns = self.encoder(enc_out)
        
        # Reshape
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Channel Attention
        if self.use_channel_attn:
            enc_out = self.channel_attention(enc_out)

        # Decoder
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        # RevIN denormalization
        dec_out = self.revin(dec_out, 'denorm')
        
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Use original normalization for imputation
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
        x_enc = self.revin(x_enc, 'norm')

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

        dec_out = self.revin(dec_out, 'denorm')
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        x_enc = self.revin(x_enc, 'norm')

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