#!/bin/bash
set -e

vision="deq/mdeq_vision"
tiny_args="--percent=0.0035 TRAIN.END_EPOCH 2 TRAIN.PRETRAIN_STEPS 1 DEQ.F_THRES 5 DEQ.B_THRES 5 MODEL.NUM_LAYERS 2"

python ${vision}/tools/cls_train.py --cfg ${vision}/experiments/cifar/cls_mdeq_TINY.yaml $tiny_args
python ${vision}/tools/cls_valid.py --cfg ${vision}/experiments/cifar/cls_mdeq_TINY.yaml $tiny_args
