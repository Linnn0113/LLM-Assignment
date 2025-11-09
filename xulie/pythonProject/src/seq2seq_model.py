import torch
import torch.nn as nn
import math

# 从组件文件导入
from transformer_blocks import (  # <-- 确保从重命名的文件导入
    PositionalEncoding,
    MultiHeadAttention,
    PositionWiseFeedForward,
    EncoderLayer
)
# 从 utils 导入掩码函数
from utils import create_causal_mask


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.encoder_decoder_attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, encoder_output, tgt_mask, src_tgt_mask):
        normed_tgt = self.norm1(tgt)
        attn_output1 = self.masked_self_attn(normed_tgt, normed_tgt, normed_tgt, tgt_mask)
        tgt = tgt + self.dropout(attn_output1)

        normed_tgt2 = self.norm2(tgt)
        attn_output2 = self.encoder_decoder_attn(
            normed_tgt2, encoder_output, encoder_output, src_tgt_mask
        )
        tgt = tgt + self.dropout(attn_output2)

        normed_tgt3 = self.norm3(tgt)
        ffn_output = self.ffn(normed_tgt3)
        tgt = tgt + self.dropout(ffn_output)

        return tgt


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout=0.1, enable_pe=True):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(d_model)
        self.enable_pe = enable_pe

    def forward(self, src, mask):
        x = self.embedding(src) * math.sqrt(self.d_model)

        if self.enable_pe:
            x = self.pos_encoding(x)

        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.final_norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout=0.1, enable_pe=True):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(d_model)
        self.enable_pe = enable_pe

    def forward(self, tgt, encoder_output, tgt_mask, src_tgt_mask):
        x = self.embedding(tgt) * math.sqrt(self.d_model)

        if self.enable_pe:
            x = self.pos_encoding(x)

        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_tgt_mask)
        return self.final_norm(x)


class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model,
                 num_layers,
                 num_heads,
                 d_ff,
                 max_len,
                 pad_idx,
                 dropout=0.1,
                 enable_pe=True):
        super(Seq2SeqTransformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout, enable_pe=enable_pe
        )
        self.decoder = Decoder(
            tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout, enable_pe=enable_pe
        )

        self.final_linear = nn.Linear(d_model, tgt_vocab_size)
        self.pad_idx = pad_idx

    def create_masks(self, src, tgt):
        device = src.device
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(-1)
        tgt_len = tgt.shape[1]
        tgt_look_ahead_mask = create_causal_mask(tgt_len, device)
        tgt_mask = tgt_pad_mask & tgt_look_ahead_mask
        src_tgt_mask = src_mask
        return src_mask, tgt_mask, src_tgt_mask

    def forward(self, src_tokens, tgt_tokens):
        src_mask, tgt_mask, src_tgt_mask = self.create_masks(src_tokens, tgt_tokens)
        encoder_output = self.encoder(src_tokens, mask=src_mask)
        decoder_output = self.decoder(
            tgt_tokens, encoder_output, tgt_mask, src_tgt_mask
        )
        logits = self.final_linear(decoder_output)
        return logits