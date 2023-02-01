import pytest
from unittest.mock import patch

from deq.mdeq_vision.tools.cls_train import main
from deq.mdeq_vision.tools.cls_fixed_point_description import (
    main as fp_norm_main
)


@pytest.mark.parametrize("config", [
    "TINY",
])
def test_cls_fp_norm(config):
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

    args[1] = "--n_images"
    args += [
        "TEST.MODEL_FILE", "output/cifar10/cls_mdeq_TINY/final_state.pth.tar",
    ]
    with patch("sys.argv", args):
        fp_norm_main()


