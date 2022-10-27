"""
    Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0
"""

import torch.nn as nn
import torch.nn.functional as F
from common.transformer import *
from IMDB.Config import *


class IMDB(nn.Module):
    def __init__(self, emb_dim, output_dim, vocab, dropout=0.1):
        super(IMDB, self).__init__()
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.vocab2id = vocab.stoi
        self.id2vocab = vocab.itos
        self.vocab_size = len(vocab)
        self.transformer_encoder = TransformerEncoder(src_vocab_size=self.vocab_size, src_pad_idx=0, emb_dim=self.emb_dim, num_layers=args.n_layer,
                                                      forward_expansion=args.n_fe, heads=args.n_head, dropout=args.dropout, device=args.device, max_length=args.sentence_len)
        self.linear_out = nn.Linear(self.emb_dim, self.output_dim, bias=False)
        self.loss = nn.CrossEntropyLoss()
        # self.layer_norm = nn.LayerNorm(self.output_dim)
        # self.dropout = nn.Dropout(dropout)

    def transformers_encode(self, token_seq):   # batch_numseq_seqlen
        batch_size, num_seq, seq_len = token_seq.size()
        token_seq = token_seq.reshape(-1, seq_len)  # [3, 4, 20] -> [12, 20]
        seq_mask = token_seq.ne(0).detach()  # True if element is not 0: PAD

        seq_enc = self.transformer_encoder(token_seq)   # [batch_size, seq_len, n_dim]
        seq_enc = seq_enc * seq_mask.unsqueeze(-1)

        seq_enc = seq_enc.reshape(batch_size, num_seq, seq_len, -1)  # [12, 20, 4] -> [3, 4, 20, 4]
        seq_mask = seq_mask.reshape(batch_size, num_seq, seq_len)  # [3, 4, 20]
        return seq_enc, seq_mask

    def sentiment_classification(self, query):  # [batch_size, seq_len]
        q = self.transformer_encoder(query)     # [batch_size, seq_len, emb_dim]
        enc_q = q[:, 0, :]                      # use the [CLS] for sentence classfication
        logits = self.linear_out(enc_q)  # [batch_size, output_dim]
        logits = F.softmax(logits, dim=-1)
        return logits

    def sentiment_classification_v0(self, query):
        query_enc, query_mask = self.transformers_encode(query)  # [batch_size, 1, seq_len] -> [batch_size, 1, seq_len, n_dim]
        query_enc = query_enc.squeeze(1)  # [batch_size, seq_len, n_dim]
        query_mask = query_mask.squeeze(1)  # [batch_size, seq_len]
        q = universal_sentence_embedding(query_enc, query_mask)  # sum [batch_size, n_dim]
        logits = self.linear_out(q)  # [batch_size, output_dim]
        logits = F.softmax(logits, dim=0)
        return logits

    def do_train(self, data):
        losses = []
        text = data['text']#.unsqueeze(1)
        label = data['label']
        prediction = self.sentiment_classification(text)    # [batch_size, output_dim]
        target = torch.zeros_like(prediction)  # [batch_size, output_dim]

        # for i, l in enumerate(label):
        #     target[i, l] = 1

        # loss_sel = F.binary_cross_entropy_with_logits(input=prediction, target=target)
        loss_sel = self.loss(input=prediction, target=label)
        # loss_sel = F.binary_cross_entropy(input=prediction, target=target)
        losses.append(loss_sel)
        return losses

    def do_infer(self, data):
        text = data['text']#.unsqueeze(1)
        label = data['label']
        prediction = self.sentiment_classification(text)  # [batch_size, output_dim]
        return prediction.argmax(axis=1)

    def forward(self, data, method='train'):
        if method == 'train':
            return self.do_train(data)
        elif method == 'infer':
            return self.do_infer(data)
