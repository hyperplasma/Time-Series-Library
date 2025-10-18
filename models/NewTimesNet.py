import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


class RevIN(nn.Module):
    """
    Reversible Instance Normalization
    默认不使用，但保留供实际应用时开启
    """
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


class InvertedDecoderLayer(nn.Module):
    """
    Inverted Decoder Layer
    在变量维度应用Self-Attention，捕捉跨变量的依赖关系
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

    def forward(self, x, attn_mask=None):
        """
        x: [batch_size, num_vars, d_model]
        """
        # Self-Attention
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            need_weights=False
        )
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-Forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class InvertedDecoder(nn.Module):
    """
    自适应Inverted Decoder模块
    
    特性：
    1. 自动根据变量数调整配置（可被命令行参数覆盖）
    2. 支持可调节的残差权重（用于不同数据集的适配）
    3. 双重视角建模：变量维度attention + 与时间视角的融合
    """
    def __init__(self, num_vars, d_model, n_heads=4, d_ff=None, num_layers=2, 
                 dropout=0.1, residual_weight=1.0):
        super(InvertedDecoder, self).__init__()
        self.num_vars = num_vars
        self.d_model = d_model
        self.residual_weight = residual_weight
        d_ff = d_ff or d_model * 4
        
        # 可学习的变量embedding（编码变量的先验信息）
        self.var_embedding = nn.Parameter(torch.randn(1, num_vars, d_model) * 0.02)
        
        # Decoder Layers
        self.layers = nn.ModuleList([
            InvertedDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output Projection
        self.output_proj = nn.Linear(d_model, d_model)
        
        print(f">>> InvDec Config: {num_layers} layers, {n_heads} heads, "
              f"{num_vars} vars, residual_weight={residual_weight:.2f}")

    def forward(self, x):
        """
        x: [batch_size, num_vars, patch_num, d_model]
        输出: [batch_size, num_vars, patch_num, d_model]
        """
        B, V, P, D = x.shape
        
        # 在patch维度做全局池化，得到每个变量的全局表示
        x_pooled = x.mean(dim=2)  # [B, V, D]
        
        # 添加可学习的变量embedding
        x_pooled = x_pooled + self.var_embedding  # [B, V, D]
        
        # 通过Decoder Layers（在变量维度做attention）
        for layer in self.layers:
            x_pooled = layer(x_pooled)
        
        # Output projection
        x_pooled = self.output_proj(x_pooled)  # [B, V, D]
        
        # 广播回原始形状
        x_pooled = x_pooled.unsqueeze(2).expand(-1, -1, P, -1)  # [B, V, P, D]
        
        # 自适应残差连接：融合时间和变量两个视角
        output = x + self.residual_weight * x_pooled
        
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
    InvDec: Inverted Decoder for Time Series Forecasting
    
    核心创新：Hybrid Inverted Architecture
    - Encoder: 在时间维度建模（temporal patterns）
    - InvDec: 在变量维度建模（variate dependencies）
    - 双重视角形成互补，增强多变量时序建模能力
    
    自适应特性：
    - 低维数据（≤10变量）：轻量级配置（1层，2头，权重0.3）
    - 高维数据（>10变量）：完整配置（2层，4头，权重1.0）
    - 支持命令行手动覆盖所有参数
    """

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride

        # RevIN（可选，默认关闭）
        self.use_revin = getattr(configs, 'use_revin', False)
        if self.use_revin:
            self.revin = RevIN(configs.enc_in, affine=True)
            print(f">>> [Optional] Using RevIN normalization")
        else:
            print(f">>> Using standard normalization (RevIN disabled)")

        # Patch Embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # Encoder (时间维度建模)
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

        # Inverted Decoder (变量维度建模) - 核心创新
        self.use_inverted_decoder = getattr(configs, 'use_inverted_decoder', True)
        if self.use_inverted_decoder:
            # 自动适配策略（优先级低）
            if configs.enc_in <= 10:
                default_n_heads = 2
                default_layers = 1
                default_residual_weight = 0.3
                mode = "Low-dimensional"
            else:
                default_n_heads = 4
                default_layers = 2
                default_residual_weight = 1.0
                mode = "High-dimensional"
            
            # 命令行参数覆盖（优先级高）
            inv_n_heads = getattr(configs, 'inv_n_heads', default_n_heads)
            inv_layers = getattr(configs, 'inv_layers', default_layers)
            inv_residual_weight = getattr(configs, 'inv_residual_weight', default_residual_weight)
            
            # 打印配置信息
            is_default = (inv_n_heads == default_n_heads and 
                         inv_layers == default_layers and 
                         inv_residual_weight == default_residual_weight)
            if is_default:
                print(f">>> [Core] {mode} mode auto-configured for {configs.enc_in} variables")
            else:
                print(f">>> [Core] Manual configuration for {configs.enc_in} variables")
            
            self.inverted_decoder = InvertedDecoder(
                num_vars=configs.enc_in,
                d_model=configs.d_model,
                n_heads=inv_n_heads,
                d_ff=configs.d_ff,
                num_layers=inv_layers,
                dropout=configs.dropout,
                residual_weight=inv_residual_weight
            )
        else:
            print(f">>> Baseline PatchTST (no Inverted Decoder)")

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
        # Normalization
        if self.use_revin:
            # RevIN normalization
            x_enc = self.revin(x_enc, 'norm')
        else:
            # Standard normalization
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
        
        # Inverted Decoder (变量维度建模)
        if self.use_inverted_decoder:
            enc_out = self.inverted_decoder(enc_out)
        
        # [B, V, P, D] -> [B, V, D, P]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Prediction Head
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization
        if self.use_revin:
            # RevIN denormalization
            dec_out = self.revin(dec_out, 'denorm')
        else:
            # Standard denormalization
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