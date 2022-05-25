import pytest
from unittest.mock import patch

from deq.mdeq_vision.tools.cls_train import main


@pytest.mark.parametrize("config", [
    "TINY",
    "LARGE_reg",
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
