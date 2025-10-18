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
    """WAVE: Weighted Autoregressive Varying Gate Attention"""
    def __init__(self, num_channels, window_size=3):
        super(WAVEChannelAttention, self).__init__()
        self.num_channels = num_channels
        self.window_size = window_size
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.local_pool = nn.AvgPool1d(kernel_size=window_size, stride=1, padding=window_size//2)
        
        self.ar_pathway = nn.Sequential(
            nn.Linear(num_channels, max(num_channels // 4, 1)),
            nn.ReLU(inplace=True),
            nn.Linear(max(num_channels // 4, 1), num_channels)
        )
        
        self.ma_pathway = nn.Sequential(
            nn.Conv1d(num_channels, max(num_channels // 4, 1), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(max(num_channels // 4, 1), num_channels, kernel_size=1)
        )
        
        self.gate = nn.Sequential(
            nn.Linear(num_channels * 2, num_channels),
            nn.Sigmoid()
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, d, p = x.size()
        x_reshape = x.view(b, c, -1)
        
        ar_feat = self.global_pool(x_reshape).squeeze(-1)
        ar_out = self.ar_pathway(ar_feat)
        
        ma_feat = self.local_pool(x_reshape)
        ma_feat_pooled = self.global_pool(ma_feat).squeeze(-1)
        ma_out = ma_feat_pooled
        
        gate_input = torch.cat([ar_out, ma_out], dim=1)
        gate_weight = self.gate(gate_input)
        
        fused = gate_weight * ar_out + (1 - gate_weight) * ma_out
        attention_weights = self.sigmoid(fused).view(b, c, 1, 1)
        
        return x * attention_weights.expand_as(x)


class InvertedDecoderLayer(nn.Module):
    """
    核心创新3: Inverted Decoder Layer
    灵感来自iTransformer，在变量维度应用attention
    
    架构特点:
    1. 每个变量作为一个token (inverted)
    2. Causal Mask支持自回归预测
    3. 轻量级设计，只需1-2层
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(InvertedDecoderLayer, self).__init__()
        
        # Multi-Head Self-Attention (在变量维度)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask=None):
        """
        x: [batch_size, num_vars, d_model]
        """
        # Self-Attention with optional causal mask
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=causal_mask,
            need_weights=False
        )
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-Forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class InvertedDecoder(nn.Module):
    """
    Inverted Decoder模块
    将Encoder输出的特征在变量维度进行建模
    """
    def __init__(self, num_vars, d_model, n_heads=4, d_ff=None, num_layers=2, dropout=0.1):
        super(InvertedDecoder, self).__init__()
        self.num_vars = num_vars
        self.d_model = d_model
        d_ff = d_ff or d_model * 4
        
        # Variable Embedding (可学习的变量embedding)
        self.var_embedding = nn.Parameter(torch.randn(1, num_vars, d_model))
        
        # Decoder Layers
        self.layers = nn.ModuleList([
            InvertedDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output Projection (可选)
        self.output_proj = nn.Linear(d_model, d_model)
        
        print(f">>> Using Inverted Decoder: {num_layers} layers, {n_heads} heads")

    def forward(self, x, use_causal_mask=False):
        """
        x: [batch_size, num_vars, patch_num, d_model]
        输出: [batch_size, num_vars, patch_num, d_model]
        """
        B, V, P, D = x.shape
        
        # 变量维度的全局池化作为初始输入
        x_pooled = x.mean(dim=2)  # [B, V, D]
        
        # 添加可学习的变量embedding
        x_pooled = x_pooled + self.var_embedding
        
        # 生成causal mask (可选)
        causal_mask = None
        if use_causal_mask:
            causal_mask = torch.triu(
                torch.ones(V, V, device=x.device) * float('-inf'),
                diagonal=1
            )
        
        # 通过Decoder Layers
        for layer in self.layers:
            x_pooled = layer(x_pooled, causal_mask)
        
        # Output projection
        x_pooled = self.output_proj(x_pooled)  # [B, V, D]
        
        # 广播回原始形状并与输入融合
        x_pooled = x_pooled.unsqueeze(2).expand(-1, -1, P, -1)
        
        # 残差连接
        output = x + x_pooled
        
        return output


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
    Triple-Enhanced PatchTST:
    1. RevIN normalization
    2. WAVE channel attention (AR+MA)
    3. Inverted Decoder (variable-wise modeling)
    """

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride

        # 1. RevIN
        self.use_revin = getattr(configs, 'use_revin', True)
        if self.use_revin:
            self.revin = RevIN(configs.enc_in, affine=True)
            print(f">>> [1/3] Using RevIN normalization")
        else:
            print(f">>> [1/3] Using standard normalization")

        # Patch嵌入
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # Standard Encoder (时间维度建模)
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

        # 2. WAVE Channel Attention
        self.use_channel_attn = getattr(configs, 'use_channel_attn', True)
        if self.use_channel_attn:
            window_size = getattr(configs, 'wave_window_size', 3)
            self.channel_attention = WAVEChannelAttention(configs.enc_in, window_size=window_size)
            print(f">>> [2/3] Using WAVE Channel Attention")
        else:
            print(f">>> [2/3] Not using Channel Attention")

        # 3. Inverted Decoder (核心创新)
        self.use_inverted_decoder = getattr(configs, 'use_inverted_decoder', True)
        if self.use_inverted_decoder:
            inv_n_heads = getattr(configs, 'inv_n_heads', 4)
            inv_layers = getattr(configs, 'inv_layers', 2)
            self.inverted_decoder = InvertedDecoder(
                num_vars=configs.enc_in,
                d_model=configs.d_model,
                n_heads=inv_n_heads,
                d_ff=configs.d_ff,
                num_layers=inv_layers,
                dropout=configs.dropout
            )
            print(f">>> [3/3] Using Inverted Decoder")
        else:
            print(f">>> [3/3] Not using Inverted Decoder")

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
        # [1] RevIN normalization
        if self.use_revin:
            x_enc = self.revin(x_enc, 'norm')
        else:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # Patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder (时间维度建模)
        enc_out, attns = self.encoder(enc_out)
        
        # Reshape: [B*V, P, D] -> [B, V, P, D]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        
        # [3] Inverted Decoder (变量维度建模)
        if self.use_inverted_decoder:
            enc_out = self.inverted_decoder(enc_out)
        
        # [B, V, P, D] -> [B, V, D, P]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # [2] WAVE Channel Attention
        if self.use_channel_attn:
            enc_out = self.channel_attention(enc_out)

        # Decoder
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization
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
        
        if self.use_inverted_decoder:
            enc_out = self.inverted_decoder(enc_out)
            
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
        
        if self.use_inverted_decoder:
            enc_out = self.inverted_decoder(enc_out)
            
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
        
        if self.use_inverted_decoder:
            enc_out = self.inverted_decoder(enc_out)
            
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