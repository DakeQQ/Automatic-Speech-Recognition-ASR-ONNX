import torch
import torch.nn as nn
import torch.nn.functional as F


class ConformerEncoder(nn.Module):
    def __init__(self, idim, n_layers, n_head, d_model,
                 residual_dropout=0.1, dropout_rate=0.1, kernel_size=33,
                 pe_maxlen=5000):
        super().__init__()
        self.odim = d_model

        self.input_preprocessor = Conv2dSubsampling(idim, d_model)
        self.positional_encoding = RelPositionalEncoding(d_model)
        self.dropout = nn.Dropout(residual_dropout)
        self.pad_zeros = torch.zeros((1, 6, 80), dtype=torch.float32)  # 80 = n_mels

        self.layer_stack = nn.ModuleList()
        for l in range(n_layers):
            block = RelPosEmbConformerBlock(d_model, n_head,
                        residual_dropout,
                        dropout_rate, kernel_size)
            self.layer_stack.append(block)

    def forward(self, padded_input, input_lengths=None, pad=True):
        padded_input = torch.cat((padded_input, self.pad_zeros), dim=1)
        enc_output, input_lengths = self.input_preprocessor(padded_input)
        pos_emb = self.positional_encoding(input_lengths)
        enc_outputs = []
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, pos_emb, input_lengths, slf_attn_mask=None, pad_mask=None)
            enc_outputs.append(enc_output)
        return enc_output

    def padding_position_is_0(self, padded_input, input_lengths=None):
        N, T = padded_input.size()[:2]
        mask = torch.ones((N, T)).to(padded_input.device)
        for i in range(N):
            mask[i, input_lengths[i]:] = 0
        mask = mask.unsqueeze(dim=1)
        return mask.to(torch.uint8)


class RelPosEmbConformerBlock(nn.Module):
    def __init__(self, d_model, n_head,
                 residual_dropout=0.1,
                 dropout_rate=0.1, kernel_size=33):
        super().__init__()
        self.ffn1 = ConformerFeedForward(d_model, dropout_rate)
        self.mhsa = RelPosMultiHeadAttention(n_head, d_model,
                                             residual_dropout)
        self.conv = ConformerConvolution(d_model, kernel_size,
                                         dropout_rate)
        self.ffn2 = ConformerFeedForward(d_model, dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, pos_emb, x_len, slf_attn_mask=None, pad_mask=None):
        out = 0.5 * (x + self.ffn1(x))
        out = self.mhsa(out, out, out, pos_emb, x_len, mask=slf_attn_mask)
        out = self.conv(out, pad_mask)
        out = 0.5 * (out + self.ffn2(out))
        out = self.layer_norm(out)
        return out


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Conv2dSubsampling(nn.Module):
    def __init__(self, idim, d_model, out_channels=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, out_channels, 3, 2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 2),
            nn.ReLU(),
        )
        subsample_idim = ((idim - 1) // 2 - 1) // 2
        self.out = nn.Linear(out_channels * subsample_idim, d_model)
        self.out_size = self.out.in_features

        self.subsampling = 4
        left_context = right_context = 3  # both exclude currect frame
        self.context = left_context + 1 + right_context  # 7

    def forward(self, x, x_mask=None):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x_len = x.shape[2].unsqueeze(0)
        x = self.out(x.transpose(1, 2).contiguous().view(1, -1, self.out_size))
        return x, x_len


class RelPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe_positive = torch.zeros(max_len, d_model, requires_grad=False)
        pe_negative = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(torch.log(torch.tensor(10000.0)).item()/d_model))
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.register_buffer('pe', pe)
        self.Tmax_half = pe.size(1) // 2
        self.Tmax_half_plus = self.Tmax_half + 1

    def forward(self, x_len):
        return self.pe[:, self.Tmax_half_plus - x_len: self.Tmax_half + x_len].float()


class ConformerFeedForward(nn.Module):
    def __init__(self, d_model, dropout_rate=0.1):
        super().__init__()
        pre_layer_norm = nn.LayerNorm(d_model)
        linear_expand = nn.Linear(d_model, d_model*4)
        nonlinear = Swish()
        dropout_pre = nn.Dropout(dropout_rate)
        linear_project = nn.Linear(d_model*4, d_model)
        dropout_post = nn.Dropout(dropout_rate)
        self.net = nn.Sequential(pre_layer_norm,
                                 linear_expand,
                                 nonlinear,
                                 dropout_pre,
                                 linear_project,
                                 dropout_post)

    def forward(self, x):
        return self.net(x) + x


class ConformerConvolution(nn.Module):
    def __init__(self, d_model, kernel_size=33, dropout_rate=0.1):
        super().__init__()
        assert kernel_size % 2 == 1
        self.pre_layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model*4, kernel_size=1, bias=False)
        self.glu = F.glu
        self.padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(d_model*2, d_model*2,
                                        kernel_size, stride=1,
                                        padding=self.padding,
                                        groups=d_model*2, bias=False)
        self.batch_norm = nn.LayerNorm(d_model*2)
        self.swish = Swish()
        self.pointwise_conv2 = nn.Conv1d(d_model*2, d_model, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        residual = x
        out = self.pre_layer_norm(x)
        out = out.transpose(1, 2)
        out = self.pointwise_conv1(out)
        out = F.glu(out, dim=1)
        out = self.depthwise_conv(out)
        out = out.transpose(1, 2)
        out = self.swish(self.batch_norm(out))
        out = out.transpose(1, 2)
        out = self.pointwise_conv2(out)
        out = out.transpose(1, 2)
        return out + residual


class EncoderMultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model,
                 residual_dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = self.d_k

        self.w_qs = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * self.d_v, bias=False)

        self.layer_norm_q = nn.LayerNorm(d_model)
        self.layer_norm_k = nn.LayerNorm(d_model)
        self.layer_norm_v = nn.LayerNorm(d_model)

        self.attention = ScaledDotProductAttention(temperature=self.d_k ** 0.5)
        self.fc = nn.Linear(n_head * self.d_v, d_model, bias=False)
        self.fc_size = self.fc.in_features
        self.dropout = nn.Dropout(residual_dropout)

    def forward(self, q, k, v, mask=None):
        residual = q
        q, k, v = self.forward_qkv(q, k, v)

        output, attn = self.attention(q, k, v, mask=mask)

        output = self.forward_output(output, residual)
        return output, attn

    def forward_qkv(self, q, k, v):
        q = torch.matmul(self.layer_norm_q(q), self.w_qs.weight)
        k = torch.matmul(self.layer_norm_k(k), self.w_ks.weight).transpose(1, 2)
        v = torch.matmul(self.layer_norm_v(v), self.w_vs.weight)
        return q, k, v

    def forward_output(self, output, residual):
        output = torch.matmul(output, self.fc.weight).sum(dim=0, keepdim=True)
        return output + residual


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(0.0)
        self.INF = float('inf')

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k)
        output, attn = self.forward_attention(attn, v, mask)
        return output, attn

    def forward_attention(self, attn, v, mask=None):
        return torch.matmul(torch.softmax(attn, dim=-1), v)


class RelPosMultiHeadAttention(EncoderMultiHeadAttention):
    def __init__(self, n_head, d_model,
                 residual_dropout=0.1):
        super().__init__(n_head, d_model,
                         residual_dropout)
        d_k = d_model // n_head
        self.linear_pos = nn.Linear(d_model, n_head * d_k, bias=False)
        self.pos_bias_u = nn.Parameter(torch.FloatTensor(n_head, d_k))
        self.pos_bias_v = nn.Parameter(torch.FloatTensor(n_head, d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)
        self.zero_pad = torch.zeros((n_head, 2048, 1), dtype=torch.int8)  # 2048 is about 30 seconds audio input.
        self.n_head = n_head

    def _rel_shift(self, x, x_len):
        x_padded = torch.cat([self.zero_pad[:, :x_len].float(), x], dim=-1)
        x_padded = x_padded.view(self.n_head, -1, x_len)
        x = x_padded[:, 1:].view_as(x)
        return x[:, :, :x_len]

    def forward(self, q, k, v, pos_emb, x_len, mask=None):
        residual = q
        q, k, v = self.forward_qkv(q, k, v)
        p = torch.matmul(pos_emb, self.linear_pos.weight).transpose(1, 2)
        q_with_bias_u = q + self.pos_bias_u
        q_with_bias_v = q + self.pos_bias_v
        matrix_ac = torch.matmul(q_with_bias_u, k)
        matrix_bd = torch.matmul(q_with_bias_v, p)
        matrix_bd = self._rel_shift(matrix_bd, x_len)
        attn_scores = matrix_ac + matrix_bd
        output = self.attention.forward_attention(attn_scores, v, mask=None)
        return self.forward_output(output, residual)
