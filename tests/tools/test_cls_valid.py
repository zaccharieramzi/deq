from unittest.mock import patch

from deq.mdeq_vision.tools.cls_train import main
from deq.mdeq_vision.tools.cls_valid import main as main_valid


def test_cls_eq():
    args = [
        "main",
        "--cfg",
        "deq/mdeq_vision/experiments/cifar/cls_mdeq_TINY.yaml",
        "--percent",
        "0.0035",
        "--results_name", "results.csv",
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

    with patch("sys.argv", args):
        main_valid()
