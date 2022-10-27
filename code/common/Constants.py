"""
    Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0
"""

PAD_WORD = '[PAD]'
BOS_WORD = '[unused0]'
UNK_WORD = '[UNK]'
EOS_WORD = '[unused1]'
SEP_WORD = '[SEP]'
CLS_WORD ='[CLS]'
MASK_WORD='[MASK]'

Universal_POS=['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'SPACE']
NER=['O', 'PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']
pos2id=dict()
id2pos=dict()
pos2id[PAD_WORD]=0
id2pos[0]=PAD_WORD
pos2id[CLS_WORD]=1
id2pos[1]=CLS_WORD
pos2id[EOS_WORD]=2
id2pos[2]=EOS_WORD
for pos in Universal_POS:
    pos2id[pos]=len(pos2id)
    id2pos[len(id2pos)]=pos

ner2id=dict()
id2ner=dict()
ner2id[PAD_WORD]=0
id2ner[0]=PAD_WORD
ner2id[CLS_WORD]=1
id2ner[1]=CLS_WORD
ner2id[EOS_WORD]=2
id2ner[2]=EOS_WORD
for ner in NER:
    ner2id[ner]=len(ner2id)
    id2ner[len(id2ner)]=ner


