"""
    Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0
"""

import torch
from sklearn.metrics import classification_report
from IMDB.Config import *
from tqdm import tqdm
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt


def calculate_acc(output, target):
    """ Calculates binary accuracy based
        on given predictions and target labels.

    Arguments:
        output (torch.tensor): predictions
        target (torch.tensor): target labels
    Returns:
        acc (float): binary accuracy
    """
    output = torch.round(output) if output.dtype == torch.float else output
    correct = torch.sum(output == target).float()
    acc = (correct / len(target)).item()
    return acc * 100


def load_n_result(output_result_dir):
    """
    Load distributed results to 1. A dict (e.g., all_label) can be represented as:
    all_label = {
        'train': {
            0: [tensor_r1, tensor_r2, tensor_r3],
            1: [tensor_r1', tensor_r2', tensor_r3']
        }
        'valid': {
            0: [],
            1: []
        }
        'test': {
            0: [],
            1: []
        }
    }
    """
    files = sorted(os.listdir(output_result_dir))
    all_label, all_pred, all_acc = {}, {}, {}
    for file in files:
        mode, epoch, local_rank = file.split('.')
        epoch = int(epoch)
        # 1. Init dicts
        if mode not in all_label:
            all_label[mode] = {}
            all_pred[mode] = {}
            all_acc[mode] = {}
        if epoch not in all_label[mode]:
            all_label[mode][epoch] = []
            all_pred[mode][epoch] = []
            all_acc[mode][epoch] = 0

        # 2. Load a result file
        list_output = torch.load(os.path.join(output_result_dir, file), map_location=torch.device('cpu'))
        # print('Load:', file)
        for golden_data, pred_output in tqdm(list_output, disable=False if args.debug else True):  # each output from each gpu
            try:
                golden_label = golden_data['label']
                pred_label = pred_output
            except:
                print(golden_data)
                print(pred_output)
                exit(0)
            all_label[mode][epoch].append(golden_label)
            all_pred[mode][epoch].append(pred_label)
    return all_label, all_pred, all_acc


def show_result(all_acc, output_result_dir, save_figure=1):
    # pprint(all_acc)
    df = pd.DataFrame(all_acc).sort_index()
    pprint(df)

    max_valid_ep = df['valid'].idxmax()
    max_valid_acc = df['valid'].max()
    test_acc = df['test'][df['valid'].idxmax()]

    print('Summary: Epoch=%s, Valid_Acc=%.4f %%, Test_Acc=%.4f %%' %
          (max_valid_ep, max_valid_acc, test_acc))

    df.plot()
    # plt.title('Learning curve')
    plt.title(output_result_dir)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    # plt.xticks(range(len(df) + 1))
    plt.xticks(np.arange(0, len(df) + 1, 10))
    if save_figure:
        fig_path = os.path.join(output_result_dir.replace('result', 'figure'))
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)

        plt.savefig(os.path.join(fig_path,
                                 'eval_ep%s_valid%.4f_test%.4f.png' %
                                 (max_valid_ep, max_valid_acc, test_acc)))
    else:
        plt.show()
    return max_valid_ep, max_valid_acc, test_acc


def eval_from_files(output_result_dir):
    print('-' * 50)
    print('Evaluating test set...%s' % output_result_dir)

    all_label, all_pred, all_acc = load_n_result(output_result_dir)

    for eval_mode in all_label:
        for ep in all_label[eval_mode]:
            # evaluate for each epoch
            acc = calculate_acc(output=torch.cat(all_pred[eval_mode][ep]),
                                target=torch.cat(all_label[eval_mode][ep]))
            all_acc[eval_mode][ep] = acc

    _, _, test_acc = show_result(all_acc, output_result_dir)

    return test_acc


def eval_from_list_output(data_list_output):
    output, target = [], []
    for gold_data, pred in data_list_output:
        output.extend(pred)
        target.extend(gold_data['label'])
    return calculate_acc(output=torch.stack(output), target=torch.stack(target))


if __name__ == '__main__':
    # y_true = [0, 1, 2, 2, 2]
    # y_pred = [0, 0, 2, 2, 1]
    # target_names = ['class 0', 'class 1', 'class 2']
    # print(classification_report(y_true, y_pred, target_names=target_names))
    # eval_from_files(args.eval_result, args.data_dir, args.save_figure, args.model_choice)
    eval_from_files(OUTPUT_RESULT_DIR)