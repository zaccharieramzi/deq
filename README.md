# DEQ experiments (except for Optical Flow estimation)

The original readme can still be read [here](original_readme.md).

## Requirements
In order to run this code you need a Python installation with version at least 3.9.
You can then install the requirements using the following command:

```
pip install .
```

## Reproducing Fig. 2 of the paper
In order to reproduce Fig 2. of the paper you will need to download the data and the model weights for each of the experiments.

In the following Table you have the links to each of the data and weights for the experiments:


| Experiment | Data | Weights |
| --- | --- | --- |
|Image classification | [ImageNet](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php) | [MDEQ-XL](https://drive.google.com/file/d/1vcWnlb5yUXE-3VHiuVJdfRZVeyx0U9-W/view?usp=sharing)|
|Image Segmentation | [Cityscapes](https://www.cityscapes-dataset.com/downloads/) | [MDEQ-XL](https://drive.google.com/file/d/1Gu7pJLGvXBbU_sPxNfjiaROJwEwak2Z8/view?usp=sharing)|
|Language Modeling | [WikiText-103](deq/deq_sequence/get_data.sh) | [DEQ-Transformer](https://drive.google.com/file/d/1lZx_sHt0-1gJVgXx90LDRizq3k-ZI0SW/view?usp=sharing)|

You can read about where to place the data and weights in the original per-experiment readme files ([for vision](deq/mdeq_vision/README.md) and [for NLP](deq/deq_sequence/README.md)).

Once you have the data and weights in the correct place you can run each experiment using the following commands.
Each of these experiments can be run on a single GPU.
For Image Classification, with `f_thres` the number of test-time iterations:

```
python deq/mdeq_vision/tools/cls_valid.py \
    --cfg deq/mdeq_vision/experiments/imagenet/cls_mdeq_XL.yaml \
    --results_name imagenet_n_iter_results.csv \
    TEST.MODEL_FILE deq/mdeq_vision/mdeq_XL_cls_new.pkl \
    DATASET.ROOT data/imagenet \
    GPUS 0, \
    DEQ.F_EPS 0.0000001 \
    WORKERS 10 \
    DEQ.F_THRES $f_thres

```

Add the `--use_loss_as_perf` flag to report the loss rather than the top1 accuracy.
Add the `--valid_on_train` flag to report the train data metrics rather than the test data metrics.

For Image Segmentation, with `f_thres` the number of test-time iterations:

```
python deq/mdeq_vision/tools/seg_test.py \
    --cfg deq/mdeq_vision/experiments/cityscapes/seg_mdeq_XL.yaml \
    --results_name cityscapes_n_iter_results.csv \
    TEST.MODEL_FILE deq/mdeq_vision/MDEQ_XL_Seg.pkl \
    DEQ.F_EPS 0.0000001 \
    GPUS 0, \
    WORKERS 10 \
    DEQ.F_THRES $f_thres
```

For NLP, with `f_thres` the number of test-time iterations:

```
. wt103_deq_transformer.sh train \
  --data wikitext-103/ \
  --load pretrained_wt103_deqtrans_v3.pkl \
  --name eval_$f_thres \
  --eval \
  --mem_len 300 \
  --f_thres $f_thres \
  --f_eps 0.0000001 \
  --f_solver broyden \
  --pretrain_step 0 \
  --debug \
  --results_file transformer_results.csv
```
