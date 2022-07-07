import os

import pytest
from unittest.mock import patch
import torch

from deq.mdeq_vision.tools.cls_train import main


@pytest.mark.parametrize("config", [
    "TINY",
    "LARGE_reg",
    "TINY_warm",
])
def test_cls_train(config):
    args = [
        "main",
        "--cfg",
        f"deq/mdeq_vision/experiments/cifar/cls_mdeq_{config}.yaml",
        "--percent",
        "0.0035",
        "TRAIN.END_EPOCH",
        "2",
        "TRAIN.PRETRAIN_STEPS",
        "1",
        "DEQ.F_THRES",
        "5",
        "DEQ.B_THRES",
        "5",
        "MODEL.NUM_LAYERS",
        "2",
    ]
    with patch("sys.argv", args):
        main()


@pytest.mark.parametrize("config", [
    "TINY",
    "LARGE_reg",
    "TINY_warm",
])
def test_cls_train_warm_init_back(config):
    args = [
        "main",
        "--cfg",
        f"deq/mdeq_vision/experiments/cifar/cls_mdeq_{config}.yaml",
        "--percent", "0.0035",
        "TRAIN.END_EPOCH", "2",
        "TRAIN.PRETRAIN_STEPS", "1",
        "DEQ.F_THRES", "5",
        "DEQ.B_THRES", "5",
        "MODEL.NUM_LAYERS", "2",
        "TRAIN.WARM_INIT_BACK", "True",
        "TRAIN.WARM_INIT_DIR", "./",
        "WORKERS", "1",
    ]
    with patch("sys.argv", args):
        main()
    # clean up all *.pt files in the ./ dir
    for f in os.listdir("./"):
        if f.endswith(".pt"):
            os.remove(f)


@pytest.mark.skipif(
    os.environ.get('CI', False) == 'true',
    reason='The full warm init test is too long for CI',
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='No GPU found',
)
def test_cls_train_warm_init():
    args = [
        "main",
        "--cfg",
        "deq/mdeq_vision/experiments/cifar/cls_mdeq_TINY_warm.yaml",
        "TRAIN.END_EPOCH",
        "2",
        "TRAIN.PRETRAIN_STEPS",
        "0",
        "TRAIN.RESUME",
        "False",
        "DEQ.F_THRES",
        "1",
        "DEQ.B_THRES",
        "1",
        "MODEL.NUM_LAYERS",
        "1",
        "MODEL.EXTRA.FULL_STAGE.FINAL_CHANSIZE",
        "10",
    ]
    with patch("sys.argv", args):
        main()
