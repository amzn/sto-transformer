#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0
"""
"""
@Function : All parameters & config settings
Please note that args assign by = can not be optimized as hyper-parameters.
"""
from common.Utils import *
import torch
import torch.backends.cudnn as cudnn
import os
import sys
from sys import platform
import argparse
from datetime import datetime
import re


CODE_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(CODE_ROOT_DIR)

parser = argparse.ArgumentParser()
# ** device **
parser.add_argument("--local_rank", type=int)


# ** model name & structure **
parser.add_argument("--model_name", type=str, default='IMDB', help='Assign the subdirectory of the running model')
parser.add_argument("--exp_name", type=str, default='default', help='extra tag to memorize experimental setup')
parser.add_argument("--model_type", type=str, default='tf', help='tf / tf-sto / bert')
parser.add_argument("--pretrained_model", type=str, default='bert-base-uncased', help='Name of the pretrained model')


# ** learning **
parser.add_argument("--n_epoch", type=int, default=50)
parser.add_argument("--job_id", default='', type=str, help="The id of the job")
parser.add_argument("--mode", type=str, default='train', help='pre/train/test/eval/uncertain')
parser.add_argument("--debug", type=int, default=0, help='to speed up running a result during debugging phase')
parser.add_argument("--early_stop", type=int, default=30, help='if loss does not decrease N epoch then stop training.')
parser.add_argument("--seed", type=int, default=123456)
parser.add_argument("--test_epoch", type=int, default=-1, help='test one epoch; -1 means test latest epoch')
parser.add_argument("--device", type=object, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
parser.add_argument("--device_ids", type=str, default='0,1')
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--train_valid_split_ratio", type=float, default=0.9)
parser.add_argument("--train_batch_size", type=int, default=256)        # TF
parser.add_argument("--test_batch_size", type=int, default=256)        # TF
parser.add_argument("--sentence_len", type=int, default=500, help='Maximum sequence length')  # TF
parser.add_argument("--train_epoch_start", type=int, default=0)
parser.add_argument("--infer_epoch_start", type=int, default=0)  #
parser.add_argument("--warmup", type=float, default=-1, help='Percentage of warm up steps')
parser.add_argument("--max_grad_norm", type=int, default=10, help="Clip gradients to this norm.")   # 1, 10.0, 40.0
parser.add_argument("--weight_decay", type=float, default=5e-4, help="Cweight decay.")    # 1, 10.0, 40.0
parser.add_argument("--accumulation_steps", type=int, default=1)
parser.add_argument("--num_negative_samples", type=int, default=-1, help='Number of samples; -1: do not sample.')
# transformers
parser.add_argument("--n_layer", type=int, default=1)
parser.add_argument("--emb_dim", type=int, default=128)
parser.add_argument("--output_dim", type=int, default=2, help='Number of classes.')           # TF
parser.add_argument("--n_head", type=int, default=4)
# parser.add_argument("--head_dim", type=int, default=32)       # emb_dim // n_head
parser.add_argument("--n_fe", type=int, default=4)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--n_gpu", type=int, default=1)
parser.add_argument("--gpu_ids", type=str, default="")
parser.add_argument("--k_centroid", type=int, default=2, help='Number of centroid of attentions in transformers.')
parser.add_argument("--tau1", type=float, default=1.0)
parser.add_argument("--tau2", type=float, default=1.0)
parser.add_argument("--n_run", type=int, default=10)
parser.add_argument("--direction", type=int, default=1) # 1 or 2
# ** dir & path **
parser.add_argument("--data_root_dir", type=str, default='%s/data' % CODE_ROOT_DIR)
parser.add_argument("--log_root_dir", type=str, default='%s/log' % CODE_ROOT_DIR)
parser.add_argument("--output_root_dir", type=str, default='%s/output' % CODE_ROOT_DIR)
parser.add_argument("--output_model_dir", type=str, default=None, help='use only when to specify a model path')
# parser.add_argument("--task", type=str, default='babi-small-t5')

args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids
if torch.cuda.is_available() and args.n_gpu > 1:
    torch.distributed.init_process_group(backend='NCCL', init_method='env://')


set_random_seed(args.seed)

# ** construct macros **
FLAG_OUTPUT = (args.local_rank is None or args.local_rank == 0)      # for multiple gpu output only once
SETTING = '%s_%s_%s' % (args.model_name, args.model_type, args.exp_name)  # e.g., IMDB_tf_default/20210415_123456
#### Cheng: to give a job id to create output folder
if args.job_id == '':
    if args.mode != 'test':
        args.job_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    else:
        # test the latest setting by default if do not assign a job_id
        args.job_id = os.listdir(os.path.join(args.output_root_dir, SETTING))[-1]
LOG_DIR = os.path.join(args.log_root_dir, '%s/%s' % (SETTING, args.job_id))
DATA_DIR = '%s/imdb' % args.data_root_dir
OUTPUT_DIR = os.path.join(args.output_root_dir, '%s/%s' % (SETTING, args.job_id))
OUTPUT_MODEL_DIR = os.path.join(OUTPUT_DIR, 'model') if not args.output_model_dir else args.output_model_dir  # Model dir
OUTPUT_RESULT_DIR = os.path.join(OUTPUT_DIR, 'result')  # Result dir

if args.debug:
    args.train_batch_size = args.test_batch_size = 8
    cut_data_index = 20 * args.train_batch_size
    args.n_epoch = 5

    args.n_head = 2
    args.n_layer = 1
    args.n_fe = 4

    args.infer_epoch_start = 0
    args.train_epoch_start = 0
else:
    cut_data_index = -1

# ** create dirs & print information**
if FLAG_OUTPUT:
    create_dir_list = [LOG_DIR, OUTPUT_DIR, OUTPUT_MODEL_DIR, OUTPUT_RESULT_DIR]
    for d in create_dir_list:
        if not os.path.exists(d):
            os.makedirs(d)
    print('pytorch=%s; cuda=%s; cudnn=%s' % (torch.__version__, torch.version.cuda, cudnn.version()))
    print('DATA_DIR=', DATA_DIR)
    print('LOG_DIR=', LOG_DIR)
    print('OUTPUT_DIR=', OUTPUT_DIR)
    print('OUTPUT_MODEL_DIR=', OUTPUT_MODEL_DIR)
    print('OUTPUT_RESULT_DIR=', OUTPUT_RESULT_DIR)

if args.mode == 'test':
    # automatically to find the latest model , only for test
    latest_model_path = max([os.path.join(OUTPUT_MODEL_DIR,d) for d in os.listdir(OUTPUT_MODEL_DIR)], key=os.path.getmtime)
    args.test_epoch = int(re.split('/|\.', latest_model_path)[-2]) if args.test_epoch == -1 else args.test_epoch