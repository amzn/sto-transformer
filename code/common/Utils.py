"""
    Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0
"""

from common.Constants import *
from torch.nn.init import *
import torch.nn as nn
from torch.distributions.categorical import *
import torch.backends.cudnn as cudnn
import pickle
import bcolz
import numpy as np
import random
import time
import codecs
import nltk
from transformers import *


NEAR_INF = 1e20
NEAR_INF_FP16 = 65504


def neginf(dtype):
    """Return a representable finite number near -inf for a dtype."""
    if dtype is torch.float16:
        return -NEAR_INF_FP16
    else:
        return -NEAR_INF


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(~mask, neginf(torch.float32)).masked_fill(mask, float(0.0))
    if torch.cuda.is_available():
        mask = mask.cuda()
    return mask


def bert_tokenizer(uncased=True):
    if uncased:
        t= BertTokenizer.from_pretrained(
        "bert-base-uncased", do_lower_case=True)
    else:
        t = BertTokenizer.from_pretrained(
            "bert-base-cased", do_lower_case=False)
    return t.tokenize, t.vocab, t.ids_to_tokens


def bert_detokenizer():
    def detokenizer(tokens):
        return ' '.join(tokens).replace(' ##', '').strip()
    return detokenizer


def nltk_tokenizer():
    def tokenizer(sent):
        return nltk.word_tokenize(sent.lower())
    return tokenizer


def nltk_detokenizer():
    def detokenizer(tokens):
        return ' '.join(tokens)
    return detokenizer


def get_ms():
    return time.time() * 1000


def set_random_seed(seed=None):
    if seed is None:
        seed = random.randrange(2 ** 32 - 1)

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True

    print('seed=%s' % seed)


def build_copy_things(id, vocab2id, copy_source, output):
    dyn_vocab2id, dyn_id2vocab = build_vocab(copy_source)
    id_dyn_vocab2id = (id, dyn_vocab2id)
    id_dyn_id2vocab = (id, dyn_id2vocab)

    dyn_output_tensor = torch.tensor(
        [dyn_vocab2id.get(w) if w in dyn_vocab2id else 0 for w in output], requires_grad=False).long()
    source_map_tensor = torch.tensor([dyn_vocab2id.get(w) for w in copy_source], requires_grad=False)

    vocab_map = []
    vocab_overlap = []
    for i in range(len(dyn_id2vocab)):
        vocab_map.append(vocab2id.get(dyn_id2vocab[i], vocab2id[UNK_WORD]))
        if dyn_id2vocab[i] in vocab2id:
            vocab_overlap.append(0.)
        else:
            vocab_overlap.append(1.)
    vocab_map_tensor = torch.tensor(vocab_map, requires_grad=False)
    vocab_overlap_tensor = torch.tensor(vocab_overlap, requires_grad=False)

    return dyn_output_tensor, source_map_tensor, vocab_map_tensor, vocab_overlap_tensor, id_dyn_vocab2id, id_dyn_id2vocab

def importance_sampling(prob,topk):
    m = Categorical(logits=prob)
    indices = m.sample((topk,)).transpose(0,1)  # batch, topk

    values = prob.gather(1, indices)
    return values, indices

def decode_to_end(model, data, vocab2id, max_target_length=None, schedule_rate=1, softmax=False, encode_outputs=None, init_decoder_states=None, tgt=None):
    # if tgt is None:
    #     tgt = data['output']
    batch_size = len(data['id'])
    if max_target_length is None:
        max_target_length = tgt.size(1)

    if encode_outputs is None:
        encode_outputs = model.encode(data)
    if init_decoder_states is None:
        init_decoder_states = model.init_decoder_states(data, encode_outputs)

    decoder_input = new_tensor([vocab2id[BOS_WORD]] * batch_size, requires_grad=False)

    prob = torch.ones((batch_size,)) * schedule_rate
    if torch.cuda.is_available():
        prob=prob.cuda()

    all_gen_outputs = list()
    all_decode_outputs = [dict({'state': init_decoder_states})]

    for t in range(max_target_length):
        # decoder_outputs, decoder_states,...
        decode_outputs = model.decode(
            data, decoder_input, encode_outputs, all_decode_outputs[-1]
        )

        output = model.generate(data, encode_outputs, decode_outputs, softmax=softmax)

        all_gen_outputs.append(output)
        all_decode_outputs.append(decode_outputs)

        if schedule_rate >=1:
            decoder_input = tgt[:, t]
        elif schedule_rate<=0:
            probs, ids = model.to_word(data, output, 1)
            decoder_input = model.generation_to_decoder_input(data, ids[:, 0])
        else:
            probs, ids = model.to_word(data, output, 1)
            indices = model.generation_to_decoder_input(data, ids[:, 0])

            draws = torch.bernoulli(prob).long()
            decoder_input = tgt[:, t] * draws + indices * (1 - draws)

    # all_gen_outputs = torch.cat(all_gen_outputs, dim=0).transpose(0, 1).contiguous()

    return encode_outputs, init_decoder_states, all_decode_outputs, all_gen_outputs

def randomk(gen_output, k=5, PAD=None, BOS=None, UNK=None):
    if PAD is not None:
        gen_output[:, PAD] = -float('inf')
    if BOS is not None:
        gen_output[:, BOS] = -float('inf')
    if UNK is not None:
        gen_output[:, UNK] = -float('inf')
    values, indices = importance_sampling(gen_output, k)
    # words=[[tgt_id2vocab[id.item()] for id in one] for one in indices]
    return values, indices

def topk(gen_output, k=5, PAD=None, BOS=None, UNK=None):
    if PAD is not None:
        gen_output[:, PAD] = 0
    if BOS is not None:
        gen_output[:, BOS] = 0
    if UNK is not None:
        gen_output[:, UNK] = 0
    if k>1:
        values, indices = torch.topk(gen_output, k, dim=1, largest=True,
                                     sorted=True, out=None)
    else:
        values, indices = torch.max(gen_output, dim=1, keepdim=True)
    return values, indices

def copy_topk(gen_output, vocab_map, vocab_overlap, k=5, PAD=None, BOS=None, UNK=None):
    vocab=gen_output[:, :vocab_map.size(-1)]
    dy_vocab=gen_output[:, vocab_map.size(-1):]

    vocab=vocab+torch.bmm(dy_vocab.unsqueeze(1), vocab_map).squeeze(1)
    dy_vocab=dy_vocab*vocab_overlap

    gen_output=torch.cat([vocab, dy_vocab], dim=-1)
    return topk(gen_output, k, PAD=PAD, BOS=BOS, UNK=UNK)

def remove_duplicate_once(sents, n=3):
    changed=False
    for b in range(len(sents)):
        sent=sents[b]
        if len(sent)<=n:
            continue

        for i in range(len(sent)-n):
            index= len(sent) - i - n
            if all(elem in sent[:index] for elem in sent[index:]):
                sents[b]=sent[:index]
                changed=True
                break
    return changed

def remove_duplicate(sents, n=3):
    changed = remove_duplicate_once(sents, n)
    while changed:
        changed = remove_duplicate_once(sents, n)

def to_sentence(batch_indices, id2vocab):
    batch_size=len(batch_indices)
    summ=list()
    for i in range(batch_size):
        indexes=batch_indices[i]
        text_summ2 = []
        for index in indexes:
            index = index.item()
            w = id2vocab[index]
            if w == BOS_WORD or w == PAD_WORD:
                continue
            if w == EOS_WORD:
                break
            text_summ2.append(w)
        if len(text_summ2)==0:
            text_summ2.append(UNK_WORD)
        summ.append(text_summ2)
    return summ

def to_copy_sentence(data, batch_indices, id2vocab, dyn_id2vocab_map):
    ids=data['id']
    batch_size=len(batch_indices)
    summ=list()
    for i in range(batch_size):
        indexes=batch_indices[i]
        text_summ2 = []
        dyn_id2vocab=dyn_id2vocab_map[ids[i].item()]
        for index in indexes:
            index = index.item()
            if index < len(id2vocab):
                w = id2vocab[index]
            elif index - len(id2vocab) in dyn_id2vocab:
                w = dyn_id2vocab[index - len(id2vocab)]
            else:
                w = PAD_WORD

            if w == BOS_WORD or w == PAD_WORD:
                continue

            if w == EOS_WORD:
                break

            text_summ2.append(w)

        if len(text_summ2)==0:
            text_summ2.append(UNK_WORD)

        summ.append(text_summ2)
    return summ

def create_emb_layer(emb_matrix, non_trainable=True):
    num_embeddings, embedding_dim = emb_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
    emb_layer.load_state_dict({'weight': emb_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer

def load_embeddings(emb_text_filepath, id2vocab, emb_dim):
    vectors = bcolz.open(emb_text_filepath+'.dat')[:]
    words = pickle.load(open(emb_text_filepath+'.words.pkl', 'rb'))
    word2idx = pickle.load(open(emb_text_filepath+'.ids.pkl', 'rb'))

    glove = {w: vectors[word2idx[w]] for w in words}

    matrix_len = len(id2vocab)
    emb_matrix = torch.zeros((matrix_len, emb_dim))
    words_found = 0

    for i in range(len(id2vocab)):
        word=id2vocab[i]
        try:
            emb_matrix[i] = torch.Tensor(glove[word])
            words_found += 1
        except KeyError:
            emb_matrix[i] = torch.Tensor(np.random.normal(scale=0.6, size=(emb_dim,)))
    return emb_matrix

def prepare_embeddings(emb_text_filepath):
    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir=emb_text_filepath+'.temp', mode='w')

    with open(emb_text_filepath, 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    vectors = bcolz.carray(vectors[1:].reshape((idx, -1)), rootdir=emb_text_filepath+'.dat', mode='w')
    vectors.flush()
    pickle.dump(words, open(emb_text_filepath+'.words.pkl', 'wb'))
    pickle.dump(word2idx, open(emb_text_filepath+'.ids.pkl', 'wb'))

def new_tensor(array, requires_grad=False):
    if torch.cuda.is_available():
        tensor =torch.tensor(array, device=torch.cuda.current_device(), requires_grad=requires_grad)
    else:
        tensor = torch.tensor(array, requires_grad=requires_grad)
    return tensor

def ones(size, requires_grad=False):
    if torch.cuda.is_available():
        tensor = torch.cuda.FloatTensor(size).fill_(1)
    else:
        tensor = torch.ones(size, requires_grad=requires_grad)
    tensor.requires_grad=False
    return tensor

# def hotfix_pack_padded_sequence(input, lengths, batch_first=True):
#     lengths = torch.as_tensor(lengths, dtype=torch.int64)
#     lengths = lengths.cpu()
#     return PackedSequence(torch._C._VariableFunctions._pack_padded_sequence(input, lengths, batch_first))

def gru_forward(gru, input, lengths, state=None, batch_first=True):
    gru.flatten_parameters()
    input_lengths, perm = torch.sort(lengths, descending=True)

    input = input[perm]
    if state is not None:
        state = state[perm].transpose(0, 1).contiguous()

    total_length=input.size(1)
    if not batch_first:
        input = input.transpose(0, 1)  # B x L x N -> L x B x N
    packed = torch.nn.utils.rnn.pack_padded_sequence(input, input_lengths, batch_first)
    # packed = hotfix_pack_padded_sequence(embedded, input_lengths, batch_first)
    # self.gru.flatten_parameters()
    outputs, state = gru(packed, state)  # -> L x B x N * n_directions, 1, B, N
    outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=batch_first, total_length=total_length)  # unpack (back to padded)

    _, perm = torch.sort(perm, descending=False)
    if not batch_first:
        outputs = outputs.transpose(0, 1)
    outputs=outputs[perm]
    state = state.transpose(0, 1)[perm]

    return outputs, state

def build_map(b_map, max=None):
    batch_size, b_len = b_map.size()
    if max is None:
        max=b_map.max() + 1
    if torch.cuda.is_available():
        b_map_ = torch.cuda.FloatTensor(batch_size, b_len, max).fill_(0)
    else:
        b_map_ = torch.zeros(batch_size, b_len, max)
    b_map_.scatter_(2, b_map.unsqueeze(2), 1.)
    # b_map_[:, :, 0] = 0.
    b_map_.requires_grad=False
    return b_map_

def build_vocab(words, max=100000):
    dyn_vocab2id = dict({PAD_WORD: 0, BOS_WORD: 1, UNK_WORD: 2, EOS_WORD: 3, SEP_WORD: 4, CLS_WORD: 5, MASK_WORD: 6})
    dyn_id2vocab = dict({0: PAD_WORD, 1: BOS_WORD, 2: UNK_WORD, 3: EOS_WORD, 4: SEP_WORD, 5: CLS_WORD, 6: MASK_WORD})
    for w in words:
        if w not in dyn_vocab2id and len(dyn_id2vocab)<max:
            dyn_vocab2id[w] = len(dyn_vocab2id)
            dyn_id2vocab[len(dyn_id2vocab)] = w
    return dyn_vocab2id,dyn_id2vocab

def merge1D(sequences,max_len=None,pad_value=None):
    lengths = [len(seq) for seq in sequences]
    max_len=max(lengths) if max_len is None else max_len
    if pad_value is None:
        padded_seqs = torch.zeros(len(sequences), max_len,requires_grad=False).type_as(sequences[0])
    else:
        padded_seqs = torch.full((len(sequences), max_len),pad_value,requires_grad=False).type_as(sequences[0])

    for i, seq in enumerate(sequences):
        end = min(lengths[i], max_len)
        padded_seqs[i, :end] = seq[:end]
    return padded_seqs

def get_data(i,data):
    ones=dict()
    for key, value in data.items():
        if value is not None:
            if isinstance(value, torch.Tensor):
                ones[key] = value[i].unsqueeze(0)
            elif isinstance(value, dict):
                ones[key] = value[data['id'][i].item()]
            else:
                ones[key] = [value[i]]
        else:
            ones[key] =None
    return ones

def concat_data(datalist):
    data=dict()

    size=len(datalist)

    for key in datalist[0]:
        value=datalist[0][key]
        if value is not None:
            if isinstance(value, torch.Tensor):
                data[key]=torch.cat([datalist[i][key] for i in range(size)], dim=0)
            elif isinstance(value, dict):
                data[key] =dict()
                for i in range(size):
                    data[key][datalist[i]['id'].item()]=datalist[i][key]
            else:
                data[key] =[datalist[i][key] for i in range(size)]
        else:
            data[key]=None
    return data

def load_vocab(vocab_file,t=0):
    thisvocab2id = dict({PAD_WORD:0, BOS_WORD:1, UNK_WORD:2, EOS_WORD:3, SEP_WORD:4, CLS_WORD:5, MASK_WORD:6})
    thisid2vocab = dict({0:PAD_WORD, 1:BOS_WORD, 2:UNK_WORD, 3:EOS_WORD, 4:SEP_WORD, 5:CLS_WORD, 6:MASK_WORD})
    id2freq = {}

    sum_freq = 0
    with codecs.open(vocab_file, encoding='utf-8') as f:
        for line in f:
            try:
                name,freq = line.strip('\n').strip('\r').split('\t')
            except Exception:
                continue
            if int(freq)>=t:
                id=len(thisvocab2id)
                thisvocab2id[name] = id
                thisid2vocab[id] = name
                id2freq[id]=int(freq)
                sum_freq+=int(freq)
    id2freq[0] = sum_freq/len(id2freq)
    id2freq[1] = id2freq[0]
    id2freq[2] = id2freq[0]
    id2freq[3] = id2freq[0]

    print('item size: ', len(thisvocab2id))

    return thisvocab2id, thisid2vocab, id2freq


def load_embedding(src_vocab2id, file):
    model=dict()
    with codecs.open(file, encoding='utf-8') as f:
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            model[word] = torch.tensor([float(val) for val in splitLine[1:]])
    matrix = torch.zeros((len(src_vocab2id), 100))
    xavier_uniform_(matrix)
    for word in model:
        if word in src_vocab2id:
            matrix[src_vocab2id[word]]=model[word]
    return matrix

def universal_sentence_embedding(sentences, mask, sqrt=False, avg=False):
    '''
    :param sentences: [batch_size, seq_len, hidden_size]
    :param mask: [batch_size, seq_len]
    :param sqrt:
    :return: [batch_size, hidden_size]
    '''
    # need to mask out the padded chars
    sentence_sums = torch.bmm(
        sentences.permute(0, 2, 1), mask.float().unsqueeze(-1)
    ).squeeze(-1)
    if not avg:
        return sentence_sums
    divisor = (mask.sum(dim=1).view(-1, 1).float())
    if sqrt:
        divisor = divisor.sqrt()
    sentence_sums /= divisor
    return sentence_sums

NEAR_INF = 1e20
NEAR_INF_FP16 = 65504
def neginf(dtype):
    """Return a representable finite number near -inf for a dtype."""
    if dtype is torch.float16:
        return -NEAR_INF_FP16
    else:
        return -NEAR_INF


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()