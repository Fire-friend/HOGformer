#!/usr/bin/env bash

CONFIG=$1
PORT_=$2
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt Allweather/Options/Allweather_HOGformer.yml --launcher pytorch
