Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0

# Environment
We implemented the model based on pytorch 1.8.1 and python 3.7.6. And config experimental enviroment by the following steps.
```bash
conda create -n pytorch_latest_p37 python=3.7 anaconda  # creat the virtual environment
source activate pytorch_latest_p37                      # activate the environment
sh setup.sh                                             # install all dependent packages
```

# Introduction
We implemented models for the Sentiment Analysis task (code/IMDB);


pre.sh is a bash script for preprocessing a dataset and run.sh is a bash script for running the experiments. The commands we use on ec2 terminal is formatted as:
```bash
sh pre.sh -m <model_name> -t <model_type> -e <exp_name> -p <other_params>
sh run.sh -r <run_mode> -m <model_name> -t <model_type> -e <exp_name> -p <other_params>  -i <gpu_id>
```
An quick start for you to undertand how to run an experiment, we take task 1 as an example:
first, we need to preprocess the original dataset to processed formatted pytorch datasets;
second, we train a model with 5 epochs and then test it 10 times with uncertain estimation.

```bash
# Step 1: preprocess dataset
sh pre.sh -m IMDB -t tf -e default
# Step 2: run with 5 epoches
sh run.sh -r train -m IMDB -t tf -e default -p '--n_epoch=5' -i 0
```
Next, we list more experimental details for each tasks on ec2, respectively.

# Task 1: Sentiment Analysis
1. Preprocessing
```bash
# sh pre.sh -m <model_name> -t <model_type> -e <exp_name> -p <other_params>
sh pre.sh -m IMDB -t tf -e default
```

2. Train & test N_RUN times with uncertainty
```bash
sh run.sh -r uncertain-train-test -m IMDB -t tf -e default -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128' -i 0
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e single_t1 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=1 --tau=1' -i 2
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e single_t11 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=1 --tau=11.3137' -i 2
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e dual_t1 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=2 --tau=1' -i 3
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e dual_t1_h2_k16 -p '--n_layer=1 --n_head=2 --k_centroid=16 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=2 --tau=1' -i 0
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e dual_t1_h4_k32 -p '--n_layer=1 --n_head=4 --k_centroid=16 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=2 --tau=1' -i 1
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e dual_t1_w0.5 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=2 --tau=1' -i 0
```

3. Hyperparameter dual tau

(1) StoTransformer tau
tau = [1, 3, 5, 7, 9, 11, 20, 30, 40, 50]
```bash
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e single_t3 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=1 --tau=3' -i 4
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e single_t5 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=1 --tau=5' -i 5
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e single_t7 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=1 --tau=7' -i 6
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e single_t9 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=1 --tau=9' -i 7
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e single_t_11 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=1 --tau2=11' -i 2
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e single_t20 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=1 --tau=20' -i 0
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e single_t30 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=1 --tau=30' -i 1
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e single_t40 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=1 --tau=40' -i 2
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e single_t50 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=1 --tau=50' -i 3
sh run.sh -r uncertain-test -m IMDB -t tf-sto -e dual_t1_h4_k32 -p '--n_layer=1 --n_head=4 --k_centroid=16 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=2 --tau=1 --job_id=20210503_091647' -i 1
```
(2) DualTransformer tau2
```bash
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e dual_t2_1 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=2 --tau2=1' -i 0
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e dual_t2_3 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=2 --tau2=3' -i 1
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e dual_t2_5 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=2 --tau2=5' -i 2
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e dual_t2_7 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=2 --tau2=7' -i 3
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e dual_t2_9 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=2 --tau2=9' -i 4
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e dual_t2_11 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=2 --tau2=11' -i 5
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e dual_t2_20 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=2 --tau2=20' -i 6
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e dual_t2_30 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=2 --tau2=30' -i 7
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e dual_t2_40 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=2 --tau2=40' -i 0
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e dual_t2_50 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=2 --tau2=50' -i 1
```
(3) DualTransformer tau 1
```bash
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e dual_t1_3 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=2 --tau1=3' -i 0
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e dual_t1_5 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=2 --tau1=5' -i 1
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e dual_t1_7 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=2 --tau1=7' -i 2
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e dual_t1_9 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=2 --tau1=9' -i 3
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e dual_t1_11 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=2 --tau1=11' -i 4
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e dual_t1_20 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=2 --tau1=20' -i 5
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e dual_t1_30 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=2 --tau1=30' -i 6
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e dual_t1_40 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=2 --tau1=40' -i 7
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e dual_t1_50 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=2 --tau1=50' -i 0
```
(4) DualTransformer tau1 & tau2
```bash
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e dual_t1_20_t2_20 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=2 --tau1=20 --tau2=20' -i 1
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e dual_t1_30_t2_20 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=2 --tau1=30 --tau2=20' -i 2
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e dual_t1_40_t2_20 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=2 --tau1=40 --tau2=20' -i 3
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e dual_t1_20_t2_30 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=2 --tau1=20 --tau2=30' -i 4
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e dual_t1_30_t2_30 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=2 --tau1=30 --tau2=30' -i 5
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e dual_t1_40_t2_30 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=2 --tau1=40 --tau2=30' -i 6
```
(5) Centroid k
```bash
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e dual_t2_30_k4 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=2 --tau2=30 --k_centroid=4' -i 0
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e dual_t2_30_k8 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=2 --tau2=30 --k_centroid=8' -i 1
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e dual_t2_30_k16 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=2 --tau2=30 --k_centroid=16' -i 2
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e dual_t2_30_k32 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=2 --tau2=30 --k_centroid=32' -i 3
sh run.sh -r uncertain-train-test -m IMDB -t tf-sto -e dual_t2_30_k64 -p '--n_layer=1 --n_head=8 --emb_dim=128 --n_epoch=50 --early_stop=30 --dropout=0.5 --train_valid_split_ratio=0.9 --train_batch_size=128 --direction=2 --tau2=30 --k_centroid=64' -i 4
```

# Analysis learning curves in realtime
We use tensorboard to track the realtime learning curves (i.e., loss & accuracy) by the following steps:

1. From your local machine, run
```bash
# ssh -N -f -L localhost:16006:localhost:6006 <user_name>@<server_address>
ssh -N -f -L localhost:16006:localhost:6006 ec2-pp
```

2. On the remote machine, run:
```bash
# tensorboard --logdir <path> --port 6006
tensorboard --logdir . --port 6006
```

3. Then, navigate to (in this example) <http://localhost:16006> on your local machine.


# Watch gpu usage
We use the following command for checking gpu usage on ec2.
```
watch "nvidia-smi"
```

# Watch & cancel running jobs
We use the following commands for checking and canceling the submitted jobs.
```
# Use the following cmd to see the submitted job
ps -ef | grep "Run.py"

# See last few lines of log file
tail -l train.log

# Use the following cmd to kill the job
kill -s 9 <pid>

# See how long does the job run
ps -o etime= -p <pid>

```

# Debug & run on laptop
We debug the model on laptop on small data in Pycharm IDE. The settings are listed as follows:
```
Name                            Parameters
IMDB-pre-tf                     --mode=pre --model_type=tf --debug=1

IMDB-run-tf                     --mode=uncertain-train-test --model_type=tf-sto  --debug=1
IMDB-run-tf-sto                 --mode=uncertain-train-test --model_type=tf-sto --debug=1
```