#!/bin/bash

cd deq/mdeq_vision

tiny_args="--percent=0.0035 TRAIN.END_EPOCH 2 TRAIN.PRETRAIN_STEPS 1 DEQ.F_THRES 5 DEQ.B_THRES 5 MODEL.NUM_LAYERS 2"

python tools/cls_train.py --cfg experiments/cifar/cls_mdeq_TINY.yaml $tiny_args
python tools/cls_train.py --cfg experiments/cifar/cls_mdeq_LARGE_reg.yaml $tiny_args
