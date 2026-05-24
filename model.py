import math
import torch
import torch.nn as nn

from dataset import (
    PAD_IDX,
    SOS_IDX,
    EOS_IDX
)

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads
    ):

        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.fc_out = nn.Linear(d_model, d_model)

    def split_heads(self, x):

        batch_size = x.shape[0]

        x = x.view(
            batch_size,
            -1,
            self.num_heads,
            self.head_dim
        )

        return x.transpose(1, 2)

    def forward(
        self,
        query,
        key,
        value,
        mask=None
    ):

        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = torch.softmax(scores, dim=-1)

        out = torch.matmul(attention, V)

        out = out.transpose(1, 2).contiguous()

        batch_size = out.shape[0]

        out = out.view(
            batch_size,
            -1,
            self.d_model
        )

        out = self.fc_out(out)

        return out

class FeedForward(nn.Module):

    def __init__(
        self,
        d_model,
        d_ff,
        dropout
    ):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        dropout
    ):

        super().__init__()

        self.attention = MultiHeadAttention(
            d_model,
            num_heads
        )

        self.norm1 = nn.LayerNorm(d_model)

        self.ff = FeedForward(
            d_model,
            d_ff,
            dropout
        )

        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):

        attn = self.attention(
            x,
            x,
            x,
            mask
        )

        x = self.norm1(
            x + self.dropout(attn)
        )

        ff = self.ff(x)

        x = self.norm2(
            x + self.dropout(ff)
        )

        return x

class DecoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        dropout
    ):

        super().__init__()

        self.self_attention = MultiHeadAttention(
            d_model,
            num_heads
        )

        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attention = MultiHeadAttention(
            d_model,
            num_heads
        )

        self.norm2 = nn.LayerNorm(d_model)

        self.ff = FeedForward(
            d_model,
            d_ff,
            dropout
        )

        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        enc_out,
        src_mask,
        tgt_mask
    ):

        attn = self.self_attention(
            x,
            x,
            x,
            tgt_mask
        )

        x = self.norm1(
            x + self.dropout(attn)
        )

        cross = self.cross_attention(
            x,
            enc_out,
            enc_out,
            src_mask
        )

        x = self.norm2(
            x + self.dropout(cross)
        )

        ff = self.ff(x)

        x = self.norm3(
            x + self.dropout(ff)
        )

        return x

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_len=5000
    ):

        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            0,
            max_len
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):

        return x + self.pe[:, :x.size(1)]

class Transformer(nn.Module):

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=1024,
        dropout=0.1
    ):

        super().__init__()

        self.src_embedding = nn.Embedding(
            src_vocab_size,
            d_model,
            padding_idx=PAD_IDX
        )

        self.tgt_embedding = nn.Embedding(
            tgt_vocab_size,
            d_model,
            padding_idx=PAD_IDX
        )

        self.positional_encoding = PositionalEncoding(d_model)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                d_model,
                num_heads,
                d_ff,
                dropout
            )
            for _ in range(num_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                d_model,
                num_heads,
                d_ff,
                dropout
            )
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(
            d_model,
            tgt_vocab_size
        )

        self.dropout = nn.Dropout(dropout)

    def make_src_mask(self, src):

        return (
            src != PAD_IDX
        ).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):

        batch_size, tgt_len = tgt.shape

        tgt_pad_mask = (
            tgt != PAD_IDX
        ).unsqueeze(1).unsqueeze(2)

        tgt_sub_mask = torch.tril(
            torch.ones(
                (tgt_len, tgt_len),
                device=tgt.device
            )
        ).bool()

        return tgt_pad_mask & tgt_sub_mask

    def encode(self, src, src_mask):

        x = self.dropout(
            self.positional_encoding(
                self.src_embedding(src)
            )
        )

        for layer in self.encoder_layers:
            x = layer(x, src_mask)

        return x

    def decode(
        self,
        tgt,
        enc_out,
        src_mask,
        tgt_mask
    ):

        x = self.dropout(
            self.positional_encoding(
                self.tgt_embedding(tgt)
            )
        )

        for layer in self.decoder_layers:
            x = layer(
                x,
                enc_out,
                src_mask,
                tgt_mask
            )

        return self.fc_out(x)

    def forward(self, src, tgt):

        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        enc_out = self.encode(src, src_mask)

        out = self.decode(
            tgt,
            enc_out,
            src_mask,
            tgt_mask
        )

        return out

    @torch.no_grad()
    def infer(
        self,
        src,
        max_len=50
    ):

        self.eval()

        src_mask = self.make_src_mask(src)

        enc_out = self.encode(
            src,
            src_mask
        )

        generated = torch.tensor(
            [[SOS_IDX]],
            device=src.device
        )

        for _ in range(max_len):

            tgt_mask = self.make_tgt_mask(generated)

            out = self.decode(
                generated,
                enc_out,
                src_mask,
                tgt_mask
            )

            next_token = out[:, -1, :].argmax(-1).unsqueeze(1)

            generated = torch.cat(
                [generated, next_token],
                dim=1
            )

            if next_token.item() == EOS_IDX:
                break

        return generated.squeeze(0).tolist()