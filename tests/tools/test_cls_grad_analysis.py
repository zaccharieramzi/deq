import pytest
from unittest.mock import patch

from deq.mdeq_vision.tools.cls_train import main
from deq.mdeq_vision.tools.cls_grad_analysis import main as cls_grad_analysis_main


@pytest.mark.parametrize("config", [
    "TINY",
    "LARGE_reg",
])
def test_cls_grad(config):
    args = [
        "main",
        "--save_at",
        "1",
        "--cfg",
        f"deq/mdeq_vision/experiments/cifar/cls_mdeq_{config}.yaml",
        "--percent",
        "0.0035",
        "TRAIN.END_EPOCH",
        "2",
        "TRAIN.PRETRAIN_STEPS",
        "1",
        "DEQ.F_THRES",
        "2",
        "DEQ.B_THRES",
        "2",
        "MODEL.NUM_LAYERS",
        "2",
    ]
    with patch("sys.argv", args):
        main()

    args += [
        "TRAIN.BEGIN_EPOCH",
        "1",
        "--n_images",
        "1",
    ]
    with patch("sys.argv", args):
        cls_grad_analysis_main()
