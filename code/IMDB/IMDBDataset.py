"""
// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Jiahuan Pei
@Contact : jpei@amazon.com
@Date: 2021-04-01
@Reference:
https://github.com/jensjepsen/imdb-transformer/blob/48ce71d2520b5b59152c3accc6114239881aa513/dataloader.py
https://github.com/rahulbhalley/sentiment-analysis-bert.pytorch/blob/master/main.py
"""
from IMDB.Config import *
from IMDB.Utils import *
from common.Constants import *
#import torch
from torchtext.legacy import data as t_data
#import torchtext
from torchtext.legacy import datasets
from torchtext.vocab import GloVe
import string
from transformers import BertTokenizer
import random
import os
from tqdm import tqdm
from datetime import datetime


def get_imdb(data_root_dir, max_length=args.sentence_len, model_type=args.model_type, pretrained_model=args.pretrained_model, SEED=args.seed):
    """
    Given the root dir of imdb folder, load train/valid/test dataset by torchtext
    """
    # load TEXT & LABEL
    if model_type == 'tf':              # transformer
        TEXT = t_data.Field(
            lower=True,
            include_lengths=True,
            batch_first=True,
            tokenize=imdb_tokenize,
            # truncate_first=True,
            fix_length=max_length-2    # minus init_token & eos_token
        )
        LABEL = t_data.Field(
            sequential=False,
            use_vocab=False,
            unk_token=None,
            pad_token=None
        )
    elif model_type == 'bert':          # bert
        tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        sentence_len = min(max_length, tokenizer.max_model_input_sizes[pretrained_model])
        TEXT = t_data.Field(
            batch_first=True,
            use_vocab=False,
            tokenize=tokenizer.tokenize,
            preprocessing=tokenizer.convert_tokens_to_ids,
            # init_token=tokenizer.cls_token,
            # pad_token=tokenizer.pad_token,
            # unk_token=tokenizer.unk_token,
            # eos_token=tokenizer.eos_token,
            fix_length=sentence_len-2
        )

        LABEL = t_data.Field(
            sequential=False,
            use_vocab=False,
            unk_token=None,
            pad_token=None
        )
    else:
        pass

    if FLAG_OUTPUT:
        print("Loading data using torchtext...\n")

    # make splits for data
    _train_dataset, _test_dataset = datasets.IMDB.splits(TEXT, LABEL, root=data_root_dir)
    _train_dataset, _valid_dataset = _train_dataset.split(args.train_valid_split_ratio, random_state=random.seed(SEED))
    # 0.7: (17500, 7500, 25000)     0.9: (22500, 2500, 25000)     0.8: (20000, 5000, 25000)
    # build the vocabulary
    if args.model_type == 'tf':
        TEXT.build_vocab(_train_dataset, vectors=GloVe(name='42B', dim=300, max_vectors=10000, cache='%s/vectors_cache'%data_root_dir))
        LABEL.build_vocab(_train_dataset)
        _vocab = TEXT.vocab
    elif args.model_type == 'bert':
        TEXT.build_vocab(_train_dataset)
        LABEL.build_vocab(_train_dataset)
        _vocab = TEXT.vocab
    else:
        pass

    # print information about the data
    if FLAG_OUTPUT:
        print(f"training examples count: {len(_train_dataset)}")
        print(f"test examples count: {len(_test_dataset)}")
        print(f"validation examples count: {len(_valid_dataset)}")
        print('train.fields=%s' % _train_dataset.fields)
        print('len(vocab)=%s' % len(_vocab))
        # print('vocab.vectors.size()=%s' % str(_vocab.vectors.size()))

    return _train_dataset, _valid_dataset, _test_dataset, _vocab


class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, torchtext_dataset, vocab, sentence_len=args.sentence_len, n=1E10, sample_tensor=None):
        """
        Given a torchtext_dataset and vocab object as an input, wrap a torch_dataset object
        """
        super(IMDBDataset, self).__init__()
        self.sentence_len = sentence_len

        if sample_tensor is None:
            self.samples = torchtext_dataset.examples
            self.vocab2id = vocab.stoi
            self.id2vocab = vocab.itos
            # self.label2id = torchtext_dataset.fields['label'].vocab.stoi
            # self.id2label = torchtext_dataset.fields['label'].vocab.itos
            self.label2id = {'neg': 0, 'pos': 1}
            self.id2label = {0: 'neg', 1: 'pos'}
            self.n = n
            self.sample_tensor = []
            self.load()
        else:
            self.samples = None
            self.sample_tensor = sample_tensor

        self.len = len(self.sample_tensor)
        print('data size: %s' % self.len)

    def load(self):
        for id in tqdm(range(len(self.samples)), desc='sample', disable=False if args.debug else True):
            sample = self.samples[id]
            id_tensor = torch.tensor([id]).long()
            text = sample.text[:self.sentence_len-2]
            sent = [CLS_WORD] + text + [PAD_WORD] * (self.sentence_len - len(text) - 2) + [EOS_WORD]   # padding a sentence
            # sent = sample.text
            # Numericalization
            text_tensor = torch.tensor([self.vocab2id.get(w, self.vocab2id[UNK_WORD]) for w in sent]).long()
            label_tensor = torch.tensor([self.label2id.get(sample.label)]).long()
            self.sample_tensor.append([id_tensor, text_tensor, label_tensor])

            if id >= self.n:
                break

    def __getitem__(self, index):
        return self.sample_tensor[index]

    def __len__(self):
        return self.len


class CRDataset(torch.utils.data.Dataset):
    def __init__(self, samples, vocab, sentence_len=args.sentence_len, n=1E10, sample_tensor=None):
        """
        Given a torchtext_dataset and vocab object as an input, wrap a torch_dataset object
        """
        super(CRDataset, self).__init__()
        self.sentence_len = sentence_len

        if sample_tensor is None:
            self.samples = samples
            self.vocab2id = vocab.stoi
            self.id2vocab = vocab.itos
            self.label2id = {'neg': 0, 'pos': 1}
            self.id2label = {0: 'neg', 1: 'pos'}
            self.n = n
            self.sample_tensor = []
            self.load()
        else:
            self.samples = None
            self.sample_tensor = sample_tensor

        self.len = len(self.sample_tensor)
        print('data size: %s' % self.len)

    def load(self):
        for id in tqdm(range(len(self.samples)), desc='sample', disable=False if args.debug else True):
            sample = self.samples[id]
            id_tensor = torch.tensor([id]).long()
            text = sample.text[:self.sentence_len-2]
            sent = [CLS_WORD] + text + [PAD_WORD] * (self.sentence_len - len(text) - 2) + [EOS_WORD]   # padding a sentence
            # sent = sample.text
            # Numericalization
            text_tensor = torch.tensor([self.vocab2id.get(w, self.vocab2id[UNK_WORD]) for w in sent]).long()
            label_tensor = torch.tensor([sample.label]).long()
            self.sample_tensor.append([id_tensor, text_tensor, label_tensor])

            if id >= self.n:
                break

    def __getitem__(self, index):
        return self.sample_tensor[index]

    def __len__(self):
        return self.len


def collate_fn(data): # combine batch
    id, text, label = zip(*data)

    return {
            'id': torch.cat(id),
            'text': torch.stack(text),
            'label': torch.cat(label),
    }


class Sample:
    def __init__(self, sample_id, text, label):
        self.sample_id = sample_id
        self.text = text
        self.label = label


def create_ood_torchtext_dataset(file_name='CR.test', data_dir=DATA_DIR):
    ood_path = os.path.join(data_dir, file_name)
    torchtext_ood_samples = []
    if os.path.exists(ood_path):
        with open(ood_path, 'r') as fr:
            for i, line in enumerate(fr.readlines()):
                text, label = line.strip('\n').split('\t')
                sample = Sample(sample_id=i, text=imdb_tokenize(text), label=int(label))
                torchtext_ood_samples.append(sample)
    return torchtext_ood_samples


def wrap_dataset(torchtext_dataset, vocab, mode, ratio=args.train_valid_split_ratio, model_name=args.model_name, model_type=args.model_type.split('-')[0], data_dir=DATA_DIR):
    """
    wrap torchtext_dataset to tensor and save pickle files
    """
    print('Prepare %s dataset objects...' % mode)
    dataset_path = '%s/%s.%s.%s-%s.pkl' % (data_dir, model_name, model_type, mode, ratio)  # e.g., data/imdb/IMDB.tf.train.pkl
    dataset, sample_tensor = None, None
    if os.path.exists(dataset_path):
        sample_tensor = torch.load(dataset_path)
    else:
        if mode == 'test_ood':
            dataset = CRDataset(torchtext_dataset, vocab)
        else:
            dataset = IMDBDataset(torchtext_dataset, vocab)
        torch.save(dataset.sample_tensor, dataset_path)     # save sample_tensor object
    return sample_tensor


def prepare_dataset(args):
    # 1. Load raw dataset by torchtext
    torchtext_train_dataset, torchtext_valid_dataset, torchtext_test_dataset, vocab = get_imdb(data_root_dir=args.data_root_dir)
    vocab_path = '%s/%s.%s.vocab-%s.pkl' % (DATA_DIR, args.model_name, args.model_type, args.train_valid_split_ratio)
    if not os.path.exists(vocab_path):
        torch.save(vocab, vocab_path)

    torchtext_test_ood_dataset = create_ood_torchtext_dataset()

    # 2. Wrap and number it to torch dataset
    train_sample_tensor = wrap_dataset(torchtext_train_dataset, vocab, mode='train')
    valid_sample_tensor = wrap_dataset(torchtext_valid_dataset, vocab, mode='valid')
    test_sample_tensor = wrap_dataset(torchtext_test_dataset, vocab, mode='test')
    test_ood_sample_tensor = wrap_dataset(torchtext_test_ood_dataset, vocab, mode='test_ood')

    # 3. Iter and see train dataset
    train_dataset = IMDBDataset(torchtext_train_dataset, vocab, sample_tensor=train_sample_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.train_batch_size, pin_memory=True)
    for j, data in enumerate(train_loader, 0):
        print(j, data)
        if j > 3:
            break



if __name__ == "__main__":
    """
        Run prepare_dataset() before training
    """
    start = datetime.now()
    prepare_dataset(args=args)
    end = datetime.now()
    print('run time:%.2f mins' % ((end-start).seconds/60))
