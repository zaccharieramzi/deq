#!/bin/bash

cd deq/mdeq_vision

tiny_args="--percent 0.01 --opts END_EPOCH 2 PRETRAIN_STEPS 1"

python tools/cls_train.py --cfg experiments/cifar/cls_mdeq_TINY.yaml $tiny_args
python tools/cls_train.py --cfg experiments/cifar/cls_mdeq_LARGE_reg.yaml $tiny_args
