#!/bin/bash

cd deq/mdeq_vision

tiny_args="--percent=0.0035 MODEL.PRETRAINED None TRAIN.END_EPOCH 2 TRAIN.PRETRAIN_STEPS 1 DEQ.F_THRES 5 DEQ.B_THRES 5 MODEL.NUM_LAYERS 2 TRAIN.BATCH_SIZE 64 TEST.BATCH_SIZE 64"

python tools/seg_train.py --cfg experiments/cifar/seg_mdeq_SMALL.yaml $tiny_args