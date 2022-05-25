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


def test_cls_train_restart():
    args = [
        "main",
        "--save_at",
        "1",
        "--cfg",
        "deq/mdeq_vision/experiments/cifar/cls_mdeq_TINY.yaml",
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

    args[args.index("save_at")] = "--restart_at"
    with patch("sys.argv", args):
        main()


@pytest.mark.parametrize("config", [
    "TINY",
    "LARGE_reg",
])
@pytest.mark.parametrize("n_refine", [
    0,
    1,
])
def test_cls_train_shine(config, n_refine):
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
        f"{n_refine}",
        "DEQ.SHINE",
        "True",
        "DEQ.SHINE_FALLBACK",
        "True",
        "MODEL.NUM_LAYERS",
        "2",
    ]
    with patch("sys.argv", args):
        main()
