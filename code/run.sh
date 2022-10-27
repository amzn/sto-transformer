# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#!/bin/bash
## Set-up the environment.
#source activate pytorch_latest_p37
ROOT=$(dirname $(realpath ${0}))
N_GPU=$(nvidia-smi --list-gpus | wc -l)

which python
export PYTHONPATH=${ROOT}
#set OMP_NUM_THREADS=2

## parse bash parameters
while getopts r:i:m:t:e:p: option
do
   case "${option}"  in
                i) GPU_ID=${OPTARG};;  # gpu id, 0-7
                r) MODE=${OPTARG};;    # train/test/eval/uncertain
                m) MODEL=${OPTARG};;   # model name
                t) TYPE=${OPTARG};;    # model type
                e) EXP=${OPTARG};;     # exp name
                p) PARAMS=${OPTARG};;  # other parameters
   esac
done

echo "sh run.sh -r ${MODE} -m ${MODEL} -t ${TYPE} -e ${EXP} -p ${PARAMS} -i ${GPU_ID}" >> ${log_dir}/${MODE}.log 2>&1 &

## Construct log dir
TS=$(date +"%Y%m%d_%H%M%S")
log_dir=${ROOT}/log/${MODEL}_${TYPE}_${EXP}/${TS}
if [ ! -d ${log_dir}  ];then
  mkdir -p ${log_dir}
fi


## Run code
if [ ${GPU_ID} == "" ]; then
  GPU_ID=$((( RANDOM % N_GPU )))
fi
export CUDA_VISIBLE_DEVICES=${GPU_ID}
hyper_params="--mode=${MODE} --model_name=${MODEL} --model_type=${TYPE} --exp_name=${EXP} --job_id=${TS} --n_gpu=1 ${PARAMS} --debug=0"
nohup python ${ROOT}/${MODEL}/Run.py ${hyper_params} >> ${log_dir}/${MODE}.log 2>&1 &
ps -ef | grep "${ROOT}/${MODEL}/Run.py ${hyper_params}"
# Examples:
# (1) IMDB
# sh run.sh -m IMDB -t tf -e default -p '--n_epoch=5'
# sh run.sh -m IMDB -t tf-sto -e default -p '--n_epoch=5'
# (2) CTDS
# sh run.sh -m CTDS -t comemnn -e small_t5_sota -p '--n_epoch=5'
# sh run.sh -m CTDS -t tf -e small_t5_tf -p '--n_epoch=5'
# sh run.sh -m CTDS -t tf-sto -e small_t5_tf-sto -p '--n_epoch=5'

echo "LOG_DIR= ${log_dir}/${MODE}.log"
echo "PARAMS= ${hyper_params}"
watch "tail -l ${log_dir}/${MODE}.log"
## use the following cmd to see the submitted job
# ps -ef | grep python -u CTDS/Run.py
# use the following cmd to kill the job
# kill -s 9 <pid>
# See how long does the job run
# ps -o etime= -p <pid>