import pytest
from unittest.mock import patch

from deq.mdeq_vision.tools.cls_train import main
from deq.mdeq_vision.tools.cls_fixed_point_da_invariance import (
    main as fpdi_main
)


@pytest.mark.parametrize("config", [
    "TINY",
    "LARGE_reg",
])
def test_cls_fdpi(config):
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
        fpdi_main()


