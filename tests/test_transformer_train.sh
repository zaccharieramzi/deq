#!/bin/bash

cd deq/deq_sequence

bash deq/deq_sequence/get_data.sh

python train_transformer.py \
        --data ./data/wikitext-103/ \
        --dataset wt103 \
        --adaptive \
        --div_val 4 \
        --n_layer 2 \
        --eval_n_layer 2 \
        --d_embed 20 \
        --d_model 20 \
        --n_head 2 \
        --d_head 10 \
        --d_inner 100 \
        --dropout 0.05 \
        --dropatt 0.0 \
        --optim Adam \
        --lr 0.00025 \
        --warmup_step 1 \
        --pretrain_steps 2 \
        --eval-interval 2 \
        --max_step 4 \
        --tgt_len 20 \
        --mem_len 20 \
        --eval_tgt_len 20 \
        --wnorm \
        --f_solver anderson \
        --b_solver broyden \
        --stop_mode rel \
        --f_thres 5 \
        --b_thres 5 \
        --batch_size 56 \
        --name test
