import pytest
from unittest.mock import patch

from deq.mdeq_vision.tools.cls_train import main
from deq.mdeq_vision.tools.cls_eq_analysis import main as cls_eq_analysis_main


@pytest.mark.parametrize("config", [
    "TINY",
    "LARGE_reg",
])
def test_cls_eq(config):
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
    ]
    args[1] = "--n_images"
    with patch("sys.argv", args):
        cls_eq_analysis_main()


def test_cls_eq_broyden_matrices(config):
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
    ]
    args[1] = "--n_images"
    args.insert(1, "--broyden_matrices")
    with patch("sys.argv", args):
        cls_eq_analysis_main()
