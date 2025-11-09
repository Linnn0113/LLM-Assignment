import torch
import torch.nn as nn
import math

from src.transformer_blocks import (
    PositionalEncoding,
    EncoderLayer,
    DecoderLayer
)

from src.utils import create_causal_mask

class EncoderOnlyLM(nn.Module):

    def __init__(self,
                 vocab_size,
                 d_model,
                 num_layers,
                 num_heads,
                 d_ff,
                 max_len,
                 dropout=0.1,
                 enable_pe=True):

        super(EncoderOnlyLM, self).__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.enable_pe = enable_pe

    def forward(self, src_tokens):
        device = src_tokens.device
        seq_len = src_tokens.size(1)

        causal_mask = create_causal_mask(seq_len, device)

        x = self.token_embedding(src_tokens) * math.sqrt(self.d_model)

        if self.enable_pe:
            x = self.pos_encoding(x)

        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask=causal_mask)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits