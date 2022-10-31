"""
    Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0
"""

# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Function:
@Reference: https://pytorch.org/docs/stable/nn.init.html
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.Utils import weight_reset


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, heads):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads

        self.V_linear = nn.Linear(self.head_dim, self.head_dim)
        self.K_linear = nn.Linear(self.head_dim, self.head_dim)
        self.Q_linear = nn.Linear(self.head_dim, self.head_dim)
        self.FC_linear = nn.Linear(heads * self.head_dim, hidden_dim)

        self.reset_parameters()

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        # bacth_size, attention len, n_heads, head_dim
        Q = self.Q_linear(query.view(batch_size, -1, self.heads, self.head_dim))
        K = self.K_linear(key.view(batch_size, -1, self.heads, self.head_dim))
        V = self.V_linear(value.view(batch_size, -1, self.heads, self.head_dim))
        # print('Q, K, V', Q.size(), K.size(), V.size())
        key_out = torch.einsum("nqhd,nkhd->nhqk", [Q, K])  # batchx heads x query_len, key_len

        if mask is not None:
            key_out = key_out.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(key_out / (self.hidden_dim ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, V]).reshape(
            batch_size, query.shape[1], self.heads * self.head_dim)
        out = self.FC_linear(out)
        return out

    def reset_parameters(self):
        self.V_linear.reset_parameters()
        self.K_linear.reset_parameters()
        self.Q_linear.reset_parameters()
        self.FC_linear.reset_parameters()


class StoSelfAttention(nn.Module):
    def __init__(self, hidden_dim, heads, tau):
        super(StoSelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.tau = tau

        self.V_linear = nn.Linear(self.head_dim, self.head_dim)
        self.K_linear = nn.Linear(self.head_dim, self.head_dim)
        self.Q_linear = nn.Linear(self.head_dim, self.head_dim)
        self.FC_linear = nn.Linear(heads * self.head_dim, hidden_dim)

        self.reset_parameters()

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        # [batch_size, sentence_len, heads, heads_dim]
        Q = self.Q_linear(query.view(batch_size, -1, self.heads, self.head_dim))
        K = self.K_linear(key.view(batch_size, -1, self.heads, self.head_dim))
        V = self.V_linear(value.view(batch_size, -1, self.heads, self.head_dim))

        key_out = torch.einsum("nqhd,nkhd->nhqk", [Q, K])  # batchx heads x query_sen_len, key_sen_len

        if mask is not None:
            key_out = key_out.masked_fill(mask == 0, float("-1e20"))

        sto_attention = F.gumbel_softmax(key_out, tau=self.tau, hard=False, dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [sto_attention, V]).reshape(
            batch_size, query.shape[1], self.heads * self.head_dim)
        out = self.FC_linear(out)
        return out

    def reset_parameters(self):
        self.V_linear.reset_parameters()
        self.K_linear.reset_parameters()
        self.Q_linear.reset_parameters()
        self.FC_linear.reset_parameters()


class StoSelfDualAttention(nn.Module):
    def __init__(self, hidden_dim, heads, tau1, tau2, k_centroid, init_function=torch.nn.init.uniform_):
        super(StoSelfDualAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.tau1 = tau1
        self.tau2 = tau2
        self.centroid = torch.nn.Parameter(init_function(torch.empty(self.head_dim, k_centroid), a=-0.5, b=0.5),
                                           requires_grad=True)

        self.V_linear = nn.Linear(self.head_dim, self.head_dim)
        self.K_linear = nn.Linear(self.head_dim, self.head_dim)
        self.Q_linear = nn.Linear(self.head_dim, self.head_dim)
        self.FC_linear = nn.Linear(heads * self.head_dim, hidden_dim)

        self.reset_parameters()

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # [batch_size, sentence_len, heads, heads_dim]
        Q = self.Q_linear(query.view(batch_size, -1, self.heads, self.head_dim))
        K = self.K_linear(key.view(batch_size, -1, self.heads, self.head_dim))
        V = self.V_linear(value.view(batch_size, -1, self.heads, self.head_dim))

        K_ = torch.einsum("nshd,dc->nshc", [K, self.centroid])
        prob = F.gumbel_softmax(K_, tau=self.tau1, hard=False, dim=-1)
        sto_K = torch.einsum("nshc,cd->nshd", [prob, self.centroid.T])
        key_out = torch.einsum("nqhd,nkhd->nhqk", [Q, sto_K])

        if mask is not None:
            key_out = key_out.masked_fill(mask == 0, float("-1e20"))

        sto_attention = F.gumbel_softmax(key_out, tau=self.tau2, hard=False, dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [sto_attention, V]).reshape(
            batch_size, query.shape[1], self.heads * self.head_dim)
        out = self.FC_linear(out)
        return out

    def reset_parameters(self):
        self.V_linear.reset_parameters()
        self.K_linear.reset_parameters()
        self.Q_linear.reset_parameters()
        self.FC_linear.reset_parameters()


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(emb_dim, heads)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(emb_dim, forward_expansion * emb_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * emb_dim, emb_dim),
        )

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def forward(self, value, key, query, mask):
        """
        Args:
            value, key, query: [batch_size, sentence_len, emb_dim]
            mask: [batch_size, 1, 1, sentence_len]

        Returns:

        """
        attention = self.attention(query, key, value, mask)  # [batch_size, sentence_len, emb_dim]
        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

    def reset_parameters(self) -> None:
        """pp added: Resets all trainable parameters of the module."""
        self.attention.reset_parameters()
        self.feed_forward.apply(weight_reset)


class EncoderNetwork(nn.Module):

    def __init__(
            self,
            vocab_size,
            emb_dim,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
    ):
        super(EncoderNetwork, self).__init__()
        self.emb_dim = emb_dim
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(max_length, emb_dim)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    emb_dim,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, emb=None):
        batch, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(batch, seq_len).to(self.device)
        sum_emb = self.word_embedding(x) + self.position_embedding(positions)
        if emb is not None:
            sum_emb = sum_emb + emb
        out = self.dropout(sum_emb)

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

    def reset_parameters(self) -> None:
        """pp added: Resets all trainable parameters of the module."""
        for l in self.layers:
            l.reset_parameters()


class DecoderBlock(nn.Module):
    def __init__(self, emb_dim, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(emb_dim)
        self.attention = SelfAttention(emb_dim, heads=heads)
        self.transformer_block = TransformerBlock(
            emb_dim, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out


class DecoderNetwork(nn.Module):

    def __init__(
            self,
            vocab_size,
            emb_dim,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
    ):
        super(DecoderNetwork, self).__init__()
        self.emb_dim = emb_dim
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(max_length, emb_dim)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(emb_dim, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out, en_mask, de_mask):
        batch, seq_len = x.shape

        positions = torch.arange(0, seq_len).expand(batch, seq_len).to(self.device)
        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(positions))
        )

        for layer in self.layers:
            out = layer(out, encoder_out, encoder_out, en_mask, de_mask)

        out = self.fc_out(out)

        return out


class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            trg_pad_idx,
            emb_dim=512,
            num_layers=6,
            forward_expansion=4,
            heads=8,
            dropout=0,
            device="cpu",
            max_length=100,
    ):
        super(Transformer, self).__init__()

        self.encoder = EncoderNetwork(
            src_vocab_size,
            emb_dim,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = DecoderNetwork(
            trg_vocab_size,
            emb_dim,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )

        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out


# pp added
class TransformerEncoder(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            src_pad_idx,
            emb_dim=512,
            num_layers=6,
            forward_expansion=4,
            heads=8,
            dropout=0,
            device="cpu",
            max_length=100,
    ):
        super(TransformerEncoder, self).__init__()

        self.encoder = EncoderNetwork(
            src_vocab_size,
            emb_dim,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def forward(self, src, emb=None):
        src_mask = self.make_src_mask(src)
        enc_src = self.encoder(src, src_mask, emb)
        return enc_src

    def reset_parameters(self) -> None:
        """pp added: Resets all trainable parameters of the module."""
        self.encoder.reset_parameters()


class StoTransformerBlock(nn.Module):
    def __init__(self, emb_dim, heads, dropout, forward_expansion, k_centroid, tau1=1.0, tau2=1.0, direction=2):
        super(StoTransformerBlock, self).__init__()

        if direction == 1:
            self.attention = StoSelfAttention(emb_dim, heads, tau2)
        elif direction == 2:
            self.attention = StoSelfDualAttention(emb_dim, heads, tau1, tau2, k_centroid)

        self.tau1 = tau1
        self.tau2 = tau2
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(emb_dim, forward_expansion * emb_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * emb_dim, emb_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):

        attention = self.attention(query, key, value, mask)  # attention [batch_size, sentence_len, emb_dim]
        # tensor_l_k = torch.matmul(attention, self.centroid)        # tensor_l_k [batch_size, sentence_len, k_centroid] <= [batch_size, sentence_len, emb_dim] * [emb_dim, k_centroid]
        # option 1:
        # new_attention = torch.matmul(tensor_l_k, self.centroid.T)  # new_attention [batch_size, sentence_len, emb_dim] <= [batch_size, sentence_len, k_centroid] * [k_centroid, emb_dim]
        # option 2:
        # new_attention = torch.matmul(F.gumbel_softmax(tensor_l_k, tau=self.tau, hard=False), self.centroid.T)  # new_attention [batch_size, sentence_len, emb_dim] <= [batch_size, sentence_len, k_centroid] * [k_centroid, emb_dim]
        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))  # query [batch_size, sentence_len, emb_dim]
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

    def reset_parameters(self) -> None:
        """pp added: Resets all trainable parameters of the module."""
        self.attention.reset_parameters()
        self.feed_forward.apply(weight_reset)


class StoEncoderNetwork(nn.Module):

    def __init__(
            self,
            vocab_size,
            emb_dim,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
            k_centroid,
            tau1,
            tau2,
            direction
    ):
        super(StoEncoderNetwork, self).__init__()
        self.emb_dim = emb_dim
        self.k_centroid = k_centroid
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(max_length, emb_dim)

        self.layers = nn.ModuleList(
            [
                StoTransformerBlock(
                    emb_dim,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                    k_centroid=k_centroid,
                    tau1=tau1,
                    tau2=tau2,
                    direction=direction
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, emb=None):
        batch, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(batch, seq_len).to(self.device)
        sum_emb = self.word_embedding(x) + self.position_embedding(positions)
        if emb is not None:
            sum_emb = sum_emb + emb
        out = self.dropout(sum_emb)

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

    def reset_parameters(self) -> None:
        """pp added: Resets all trainable parameters of the module."""
        for l in self.layers:
            l.reset_parameters()


class StoTransformerEncoder(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            src_pad_idx,
            emb_dim=512,
            num_layers=6,
            forward_expansion=4,
            heads=8,
            dropout=0,
            device="cpu",
            max_length=100,
            k_centroid=2,
            tau1=1.0,
            tau2=1.0,
            direction=2
    ):
        super(StoTransformerEncoder, self).__init__()

        self.encoder = StoEncoderNetwork(
            src_vocab_size,
            emb_dim,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
            k_centroid,
            tau1,
            tau2,
            direction
        )

        self.src_pad_idx = src_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def forward(self, src, emb=None):
        src_mask = self.make_src_mask(src)
        enc_src = self.encoder(src, src_mask, emb)
        return enc_src

    def reset_parameters(self) -> None:
        """pp added: Resets all trainable parameters of the module."""
        self.encoder.reset_parameters()
        # self.input_projection.reset_parameters()
        # self.output_projection.reset_parameters()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(
        device
    )
    out = model(x, trg[:, :-1])
    print(out.shape)
