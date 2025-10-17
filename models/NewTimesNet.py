import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


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


class Enhanced_Inception_Block(nn.Module):
    """
    核心创新：增强的Inception块
    1. 添加深度可分离卷积（减少参数、增强特征提取）
    2. 添加SE注意力模块（通道自适应重标定）
    """
    def __init__(self, in_channels, out_channels, num_kernels=6):
        super(Enhanced_Inception_Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        
        # 原始多尺度卷积
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(
                nn.Conv2d(in_channels, out_channels, 
                         kernel_size=2 * i + 1, padding=i)
            )
        self.kernels = nn.ModuleList(kernels)
        
        # SE注意力（轻量级通道注意力）
        self.se_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 多尺度卷积
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        
        # SE通道注意力
        se_weight = self.se_layer(res)
        res = res * se_weight
        
        return res


class AdaptiveFreqFilter(nn.Module):
    """轻量级自适应频率滤波"""
    def __init__(self, seq_len, channels):
        super(AdaptiveFreqFilter, self).__init__()
        self.seq_len = seq_len
        freq_dim = seq_len // 2 + 1
        # 简化：只学习全局频率权重（不分通道）
        self.freq_weight = nn.Parameter(torch.ones(1, freq_dim, 1))
        
    def forward(self, x):
        B, T, C = x.shape
        x_freq = torch.fft.rfft(x, dim=1)
        weight = torch.sigmoid(self.freq_weight)
        x_freq_filtered = x_freq * weight
        x_filtered = torch.fft.irfft(x_freq_filtered, n=T, dim=1)
        return x_filtered


class CrossPeriodInteraction(nn.Module):
    """
    核心创新：跨周期信息交互
    不同周期捕捉的模式之间进行轻量级信息交换
    """
    def __init__(self, d_model, num_periods):
        super(CrossPeriodInteraction, self).__init__()
        self.num_periods = num_periods
        
        # 使用1x1卷积实现跨周期特征混合（轻量级）
        self.period_mixer = nn.Conv1d(
            in_channels=num_periods,
            out_channels=num_periods,
            kernel_size=1,
            groups=1  # 全连接混合
        )
        
    def forward(self, multi_period_features):
        # multi_period_features: [B, T, C, K]
        B, T, C, K = multi_period_features.shape
        
        # 重塑为 [B*T*C, K]
        x = multi_period_features.reshape(B * T * C, K).unsqueeze(-1)  # [B*T*C, K, 1]
        
        # 跨周期混合
        x = self.period_mixer(x).squeeze(-1)  # [B*T*C, K]
        
        # 恢复形状
        x = x.reshape(B, T, C, K)
        
        return x


class EnhancedTimesBlock(nn.Module):
    """
    改进的TimesBlock：
    1. 使用增强的Inception块（SE注意力）
    2. 添加跨周期信息交互
    3. 轻量级频率滤波
    4. 可学习残差缩放
    """
    def __init__(self, configs):
        super(EnhancedTimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        
        # 增强的Inception卷积
        self.conv = nn.Sequential(
            Enhanced_Inception_Block(configs.d_model, configs.d_ff,
                                    num_kernels=configs.num_kernels),
            nn.GELU(),
            Enhanced_Inception_Block(configs.d_ff, configs.d_model,
                                    num_kernels=configs.num_kernels)
        )
        
        # 跨周期信息交互
        self.cross_period = CrossPeriodInteraction(configs.d_model, configs.top_k)
        
        # 频率滤波
        self.freq_filter = AdaptiveFreqFilter(
            seq_len=configs.seq_len + configs.pred_len,
            channels=configs.d_model
        )
        
        # 可学习残差缩放
        self.residual_scale = nn.Parameter(torch.tensor(0.1))  # 初始化为0.1

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            
            # 2D变换 + 增强的Inception卷积
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        
        # 堆叠多周期结果
        res = torch.stack(res, dim=-1)  # [B, T, N, K]
        
        # 跨周期信息交互（核心创新）
        res = self.cross_period(res)
        
        # 周期加权聚合
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        
        # 频率滤波
        res = self.freq_filter(res)
        
        # 残差连接（可学习缩放）
        res = res + self.residual_scale * x
        
        return res


class Model(nn.Module):
    """
    Enhanced TimesNet with:
    1. RevIN normalization
    2. Enhanced Inception Block (with SE attention)
    3. Cross-Period Interaction
    4. Adaptive Frequency Filtering
    5. Learnable Residual Scaling
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        
        # RevIN
        self.use_revin = getattr(configs, 'use_revin', True)
        if self.use_revin:
            self.revin_layer = RevIN(configs.enc_in, affine=True)
        
        # 增强的TimesBlock
        self.model = nn.ModuleList([EnhancedTimesBlock(configs) 
                                    for _ in range(configs.e_layers)])
        
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, 
                                          configs.embed, configs.freq,
                                          configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_revin:
            x_enc = self.revin_layer(x_enc, 'norm')
        else:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        
        dec_out = self.projection(enc_out)

        if self.use_revin:
            dec_out = self.revin_layer(dec_out, 'denorm')
        else:
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
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
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
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        dec_out = self.projection(enc_out)

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
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
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
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