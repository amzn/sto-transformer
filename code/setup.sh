#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

## Creat a virtual environment, here keep the env name same as ec2
# conda create -n pytorch_latest_p37 python=3.7 anaconda
# source activate pytorch_latest_p37

## To install basic PyTorch via Anaconda
conda install anaconda bcolz
conda install pytorch torchvision -c pytorch
# for similarity search and clustering of dense vectors, CPU versyion only
conda install faiss-cpu -c pytorch

## Install transformers
pip install transformers whoosh tensorflow tensorboardX torchtext dill sentencepiece

## Others might he useful
# realpath cmd
# /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# brew install coreutils