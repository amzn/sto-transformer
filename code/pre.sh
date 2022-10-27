# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


#!/bin/bash
ROOT=$(dirname $(realpath ${0}))
# run commands example:
# sh pre.sh -m IMDB -t tf -e default

# sh pre.sh -m CTDS -t comemnn -e small_t5_sota -p '--task=babi-small-t5'
# sh pre.sh -m CTDS -t comemnn -e full_t5_sota -p '--task=babi-full-t5'

# Set-up the environment.
#source ${HOME}/.bashrc
#source activate pytorch_latest_p37
which python
export PYTHONPATH=${ROOT}

## parse bash parameters
while getopts m:t:e:p: option
do
   case "${option}"  in
                m) MODEL=${OPTARG};;   # model name
                t) TYPE=${OPTARG};;    # model type
                e) EXP=${OPTARG};;     # exp name
                p) PARAMS=${OPTARG};;  # other parameters
   esac
done

## Construct log dir
TS=$(date +"%Y%m%d_%H%M%S")
log_dir=${ROOT}/log/${MODEL}_${TYPE}_${EXP}
if [ ! -d ${log_dir}  ];then
  mkdir -p ${log_dir}
fi


## Run code
hyper_params="--mode=pre --model_name=${MODEL} --model_type=${TYPE} --exp_name=${EXP} --job_id=${TS} ${PARAMS} --n_gpu=1 --debug=0"
nohup python ${ROOT}/${MODEL}/Run.py ${hyper_params} >> ${log_dir}/${TS}_pre.log 2>&1 &
ps -ef | grep "${ROOT}/${MODEL}/Run.py ${hyper_params}"

echo "LOG_DIR= ${log_dir}/${TS}_pre.log"
echo "PARAMS= ${hyper_params}"
watch "tail -l ${log_dir}/${TS}_pre.log"