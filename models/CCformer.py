import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
import numpy as np
from layers.convffn import FeedForwardNetwork

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)  # 把倒数第二个维度开始展平（通常是 [d_model, patch_num]）
        self.linear = nn.Linear(nf, target_window)  # 把展平后的特征映射到预测窗口长度
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)              # -> [bs, nvars, d_model*patch_num]
        x = self.linear(x)               # -> [bs, nvars, target_window]
        x = self.dropout(x)
        return x


class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)  # 每个patch长度 -> d_model
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model)) # 每个变量一个“全局token”
        self.position_embedding = PositionalEmbedding(d_model)            # 时序位置编码

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x 预期形状： [B, n_vars, L] 或 [B, 1, L]；这里函数外会控制传入
        # do patching
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))  # 扩到 batch 维度，形状 [B, n_vars, 1, d_model]

        # 用 unfold 把时间维切成不重叠patch：size=patch_len, step=patch_len
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)  # [B, n_vars, n_patches, patch_len]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # 合并B和n_vars -> [B*n_vars, n_patches, patch_len]
        # Input encoding：线性投到d_model并加上位置编码
        x = self.value_embedding(x) + self.position_embedding(x)               # [B*n_vars, n_patches, d_model]
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))           # -> [B, n_vars, n_patches, d_model]
        x = torch.cat([x, glb], dim=2)                                         # 在“序列”维上拼接全局token -> [B, n_vars, n_patches+1, d_model]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))# 再次合并 n_vars -> [B*n_vars, n_patches+1, d_model]
        return self.dropout(x), n_vars


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)  # 多个 EncoderLayer 顺序堆叠
        self.norm = norm_layer               # 末尾可选 LayerNorm
        self.projection = projection         # 末尾可选投影层

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        # x 是“主分支”（这里是 patch+glb 的变量通道序列），cross 是外部提供的“上下文/时间编码”等
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention      # 自注意力（作用在 x 上）
        self.cross_attention = self.cross_attention = cross_attention  # 交叉注意力（用 cross 作为K/V）
        self.ffn = FeedForwardNetwork(nvars=1, dmodel=d_model, dff=d_ff, drop=dropout)  # 前馈块（这里用的是自定义的conv-ffn）
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)   # 残差里的 point-wise 卷积
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)  # 对应自注意力后的归一化
        self.norm2 = nn.LayerNorm(d_model)  # 对应 cross-attn 后的归一化（只用在 glb token）
        self.norm3 = nn.LayerNorm(d_model)  # 对应 FFN 残差后的归一化
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        # x: [B*n_vars, Lx, D]；cross: [B, Lc, D]（外层会保证形状与batch匹配）
        B, L, D = cross.shape

        # 先过一遍自定义的前馈（FeedForwardNetwork内部通常按 [B, D, L] 处理）
        x = x.permute(0,2,1)             # -> [B*n_vars, D, Lx]
        x = self.ffn(x)                  # -> [B*n_vars, D, Lx]
        x = x.permute(0,2,1)             # -> [B*n_vars, Lx, D]

        # 自注意力 + 残差 + Norm
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        # 只把序列里最后一个位置当作“全局token”，拿出来做 cross-attn
        x_glb_ori = x[:, -1, :].unsqueeze(1)      # [B*n_vars, 1, D]
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))  # 按 batch 还原成 [B, (n_vars), D]，把 n_vars 合在中间维

        # 全局token 与 cross 做交叉注意力（让 glb 聚合到 “外部时序/日历特征”上）
        x_glb_attn = self.dropout(self.cross_attention(
            x_glb, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])                              # [B, (n_vars), D]

        # 形状还原回去，叠回原来的 glb token 位置
        x_glb_attn = torch.reshape(x_glb_attn,
                                   (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)  # [B*n_vars,1,D]
        x_glb = x_glb_ori + x_glb_attn     # 残差
        x_glb = self.norm2(x_glb)

        # 把新的 glb 放回到序列末尾
        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)  # [B*n_vars, Lx, D]

        # 一个轻量的“卷积前馈”：1x1卷积提维->激活->1x1卷积降维
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))  # -> [B*n_vars, d_ff, Lx]
        y = self.dropout(self.conv2(y).transpose(-1, 1))                   # -> [B*n_vars, Lx, D]

        return self.norm3(x + y)  # 残差 + Norm


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.patch_len = configs.patch_len
        self.patch_num = int(configs.seq_len // configs.patch_len)  # 切成多少个不重叠patch
        self.n_vars = 1 if configs.features == 'MS' else configs.enc_in  # MS: 单变量；M: 多变量

        # Embedding
        self.en_embedding = EnEmbedding(self.n_vars, configs.d_model, self.patch_len, configs.dropout)
        # ex_embedding 通常把 “原始序列(不含最后1维)” + 时间标记 做成 d_model（作为 cross 分支）
        self.ex_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                   configs.dropout)

        # Encoder-only architecture（多层 EncoderLayer 堆叠）
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # 头的输入维度 = d_model * (patch_num + 1)，多出来的 +1 是全局token
        self.head_nf = configs.d_model * (self.patch_num + 1)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 单变量场景（features != 'M'，外面传进来的是 [B, L, N]，N一般=1）
        if self.use_norm:
            # Non-stationary Transformer 的标准化：按变量维做均值方差
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape  # N 通常是通道/变量数

        # 这里只取最后一个通道做 EnEmbedding：x_enc[:, :, -1] -> [B,L] -> unsqueeze(-1)再 permute到 [B, n_vars(=1), L]
        en_embed, n_vars = self.en_embedding(x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1))
        # ex_embed 用 “除去最后通道的多维序列 + 时间特征” 做上下文
        ex_embed = self.ex_embedding(x_enc[:, :, :-1], x_mark_enc)

        # 编码器把 en_embed 当主分支，把 ex_embed 当 cross
        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))   # 还原 n_vars -> [B, n_vars, Lx, D]
        # z: [bs x nvars x d_model x patch_num] 这里转成 [B, n_vars, D, Lx] 以适配 FlattenHead 的假设
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)  # -> [B, nvars, pred_len]
        dec_out = dec_out.permute(0, 2, 1)  # -> [B, pred_len, nvars]

        if self.use_norm:
            # 反标准化，只对最后通道做（与输入对应）
            dec_out = dec_out * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out


    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 多变量场景（features == 'M'）：这里直接把所有变量都打成 patch
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape

        # 多变量：直接把 [B, L, N] 先 permute 到 [B, N, L]，每个变量各自做 patch+embedding+加glb
        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))
        # ex_embed 用全量序列 + 时间标记
        ex_embed = self.ex_embedding(x_enc, x_mark_enc)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))    # -> [B, n_vars, Lx, D]
        enc_out = enc_out.permute(0, 1, 3, 2)                                # -> [B, n_vars, D, Lx]

        dec_out = self.head(enc_out)  # -> [B, nvars, pred_len]
        dec_out = dec_out.permute(0, 2, 1)  # -> [B, pred_len, nvars]

        if self.use_norm:
            # 反标准化：多变量一起反
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # 统一入口：根据任务名 + 特征类型路由到单变量或多变量预测
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if self.features == 'M':
                dec_out = self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :]  # [B, L, D]，保险再切一次最后 pred_len
            else:
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else:
            return None
