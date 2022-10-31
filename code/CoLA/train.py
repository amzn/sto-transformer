import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import argparse
import os
import csv

from data_loader import get_data, get_data_without_label, get_vocab, accuracy_cal
from data_loader import DataLoader, TestLoader, TestLoaderNoLabel

from LSTM import Model_B
from Transformer import Model_T
from Sto_Transformer import Model_S

from sklearn.metrics import accuracy_score, matthews_corrcoef

import numpy as np
import random


def main(args):
    data_train, gold_train = get_data(args.train_path)
    data_dev, gold_dev = get_data(args.dev_path)

    word_to_int, int_to_word = get_vocab(data_train, args.min_word_count)

    vocab_size = len(word_to_int)
    max_len = 100

    train_loader = DataLoader(args, data_train, gold_train, args.batch_size,
                              word_to_int)
    dev_loader = TestLoader(args, data_dev, gold_dev, word_to_int)

    lossFunction = nn.CrossEntropyLoss()
    if args.transformer:
        model = Model_T(args.embed_size, args.hidden_size, args.inter_size, vocab_size,
                        max_len, args.n_heads, args.n_layers, args.per_layer,
                        args.dropout_prob_classifier, args.dropout_prob_attn,
                        args.dropout_prob_hidden, args.use_elmo, args.num_rep, args.elmo_drop, args.gpu_id)  # .cuda()
    elif args.sto_transformer:
        model = Model_S(args.dual, args.embed_size, args.hidden_size, args.inter_size, vocab_size,
                        max_len, args.n_heads, args.n_layers, args.per_layer,
                        args.dropout_prob_classifier, args.dropout_prob_attn,
                        args.dropout_prob_hidden, args.use_elmo, args.tau1, args.tau2, args.centroids,
                        args.num_rep, args.elmo_drop, args.gpu_id)  # .cuda()
    elif args.lstm:
        model = Model_B(args.embed_size, args.hidden_size, vocab_size,
                        args.use_elmo, args.num_rep, args.elmo_drop, args.gpu_id)  # .cuda()
    if args.gpu_id > -1:
        model = model.cuda(args.gpu_id)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.inference:
        ood_data_dev, ood_gold_dev = get_data(args.ood_val_path)
        ood_dev_loader = TestLoader(args, ood_data_dev, ood_gold_dev, word_to_int)
        inference(model, lossFunction, ood_dev_loader, args)
    elif args.prediction:
        test_data, gold_test = get_data(args.in_test_path)
        test_loader_ = TestLoader(args, test_data, gold_test, word_to_int)
        prediction(model, test_loader_, args)
    else:
        train(model, optimizer, lossFunction, train_loader, dev_loader, args.epochs, args.eval_every)


def inference(model, lossFunction, ood_dev_loader, args):
    start = datetime.now()
    if args.gpu_id > -1:
        model.load_state_dict(torch.load(args.model_path + '_' + args.model_name + '.pkl'))
    else:
        model.load_state_dict(torch.load(args.model_path + '_' + args.model_name + '.pkl', map_location='cpu'))

    model.eval()
    end = datetime.now()
    model_load_time = ((end - start).seconds)

    ood_vals_acc = []
    ood_vals_mcc = []
    totoal_time = 0
    start = datetime.now()
    for i in range(10):
        acc, mcc = validate(model, lossFunction, ood_dev_loader)
        ood_vals_acc.append(acc)
        ood_vals_mcc.append(mcc)
    end = datetime.now()
    totoal_time = ((end - start).seconds)
    mean = np.array(ood_vals_mcc).mean()
    var = np.array(ood_vals_mcc).var()
    print('Mcc  mean', mean, 'MCC var', var)
    print("model loading time:", model_load_time)
    print("average_inference_time:", totoal_time / 10.0)

    mean_ = np.array(ood_vals_acc).mean()
    var_ = np.array(ood_vals_acc).var()
    print('Acc mean', mean_, 'MCC var', var_)


def train(model, optimizer, lossFunction, train_loader, dev_loader, epochs, eval_every):
    val_best = -1
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        data, input_data, input_mask, \
        positional, answers = train_loader.__load_next__()

        if args.transformer:
            if args.use_elmo:
                output = model(input_data, positional, input_mask, data)
            else:
                output = model(input_data, positional, input_mask)
        elif args.sto_transformer:
            if args.use_elmo:
                output = model(input_data, positional, input_mask, data)
            else:
                output = model(input_data, positional, input_mask)
        elif args.lstm:
            if args.use_elmo:
                output = model(input_data, input_mask, data)
            else:
                output = model(input_data, input_mask)

        loss = lossFunction(output, answers)
        scalar = loss.item()
        loss.backward()
        optimizer.step()

        print('epoch=', epoch + 1, 'training loss=', scalar)
        if (epoch + 1) % eval_every == 0:
            val_acc, val_mcc = validate(model, lossFunction, dev_loader)
            if val_best < val_mcc:
                val_best = val_mcc
                torch.save(model.state_dict(), args.model_path + '_' + args.model_name + '.pkl')
            with open(args.log_name, 'a') as f:
                f.write('epoch=' + str(epoch + 1) + 'val acc=' + str(val_acc) + 'val mcc=' + str(val_mcc) + '\n')
                f.flush()
    # f.close()

    print('best val mcc', val_best)
    return val_best


def prediction(model, test_loader, args):
    model.eval()
    scalar = 0
    tn, tp, fn, fp = 0, 0, 0, 0
    with open('predictions/' + args.model_type + '_in_test_prediction.csv', 'w', newline='') as f:
        # create the csv writer
        # fieldnames = ['Id', 'Label']
        # writer = csv.DictWriter(f, fieldnames=fieldnames)
        # header =
        # writer.writeheader()
        # write a row to the csv file

        for _ in range(test_loader.len):
            data, input_data, input_mask, \
            positional, label = test_loader.__load_next__()
            # print(label)
            max_probs = []
            pos = 0
            for i in range(0, 10):
                seed = random.randint(1, 1000000)
                torch.manual_seed(seed)
                if args.transformer:
                    if args.use_elmo:
                        output = model(input_data, positional, input_mask, data)
                    else:
                        output = model(input_data, positional, input_mask)
                elif args.sto_transformer:
                    if args.use_elmo:
                        output = model(input_data, positional, input_mask, data)
                    else:
                        output = model(input_data, positional, input_mask)
                elif args.lstm:
                    if args.use_elmo:
                        output = model(input_data, input_mask, data)
                    else:
                        output = model(input_data, input_mask)
                prob = F.softmax(output, dim=-1)
                pred = torch.argmax(prob)
                m_prob = prob[0][label[0]].detach().cpu().numpy().tolist()

                max_probs.append(m_prob)
                if pred == label[0]:
                    pos += 1

            print(" ".join(data[0][1:]))

            print(np.mean(max_probs), np.var(max_probs))
            print(pos, label[0].detach().cpu().numpy())


def validate(model, lossFunction, dev_loader):
    model.eval()
    scalar = 0
    tn, tp, fn, fp = 0, 0, 0, 0
    preds = []
    truth = []
    for _ in range(dev_loader.len):
        seed = random.randint(1, 1000000)
        torch.manual_seed(seed)
        data, input_data, input_mask, \
        positional, answers = dev_loader.__load_next__()

        if args.transformer:
            if args.use_elmo:
                output = model(input_data, positional, input_mask, data)
            else:
                output = model(input_data, positional, input_mask)
        elif args.sto_transformer:
            if args.use_elmo:
                output = model(input_data, positional, input_mask, data)
            else:
                output = model(input_data, positional, input_mask)
        elif args.lstm:
            if args.use_elmo:
                output = model(input_data, input_mask, data)
            else:
                output = model(input_data, input_mask)
        pred = torch.argmax(F.softmax(output, dim=-1))
        truth.append(answers.cpu().detach().numpy()[0])
        # pdb.set_trace(pred.cpu().detach().numpy())
        preds.append(pred.cpu().detach().numpy())
    # pdb.set_trace()
    accuracy = accuracy_score(truth, preds)
    mcc = matthews_corrcoef(truth, preds)
    print('accuracy', accuracy * 100, 'mcc', mcc * 100)

    return accuracy * 100, mcc * 100


def setup():
    parser = argparse.ArgumentParser('Argument Parser')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--embed_size', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--per_layer', type=int, default=1)
    parser.add_argument('--inter_size', type=int, default=512)
    parser.add_argument('--train_path', type=str, default='data/cola_public/tokenized_splits/in_domain_train_split.tsv')
    parser.add_argument('--dev_path', type=str, default='data/cola_public/tokenized_splits/in_domain_val_split.tsv')
    parser.add_argument('--ood_val_path', type=str, default='data/cola_public/tokenized/out_of_domain_dev.tsv')
    parser.add_argument('--in_test_path', type=str, default='data/cola_public/tokenized/out_of_domain_dev.tsv')
    # parser.add_argument('--in_test_path', type=str, default='data/cola_public/tokenized_splits/in_domain_test_split.tsv')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--min_word_count', type=int, default=0)
    parser.add_argument('--eval_every', type=int, default=50)
    parser.add_argument('--dropout_prob_classifier', type=float, default=0.1)
    parser.add_argument('--dropout_prob_attn', type=float, default=0)
    parser.add_argument('--dropout_prob_hidden', type=float, default=0)
    parser.add_argument('--num_rep', type=int, default=1)
    parser.add_argument('--elmo_drop', type=float, default=0)
    parser.add_argument('--use_elmo', type=bool, default=True)
    parser.add_argument('--lstm', type=bool, default=False)
    parser.add_argument('--transformer', type=bool, default=False)

    parser.add_argument('--sto_transformer', type=bool, default=False)

    parser.add_argument('--tau1', type=float, default=1.0)
    parser.add_argument('--tau2', type=float, default=1.0)
    parser.add_argument('--centroids', type=int, default=2)

    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--log_name', type=str, default='train_log')
    parser.add_argument('--use_transformers', type=bool, default=True)
    parser.add_argument('--model_type', type=str, default='transformer')
    parser.add_argument('--model_name', type=str, default='')

    parser.add_argument('--save_model_path', type=str, default='./trained_models/')
    parser.add_argument('--inference', type=bool, default=False)
    parser.add_argument('--prediction', type=bool, default=False)
    parser.add_argument('--dual', type=bool, default=False)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = setup()
    print(args)
    seed = random.randint(1, 1000000)
    torch.manual_seed(seed)
    args.log_name = args.model_type
    args.model_path = args.save_model_path + args.model_type
    if args.model_type == 'lstm':
        args.lstm = True
        args.use_transformers = False
    elif args.model_type == 'sto_transformer':
        args.sto_transformer = True
    elif args.model_type == 'transformer':
        args.transformer = True
    main(args)
