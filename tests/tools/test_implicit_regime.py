import os

import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch

from deq.mdeq_vision.tools.cls_train import main
from deq.mdeq_vision.tools.cls_implicit_regime_identification import (
    main as implicit_regime_main
)


@pytest.mark.parametrize("config", [
    "TINY",
])
def test_cls_implicit_regime(config, clean_up_files):
    args = [
        "main",
        "--save_at",
        "1",
        "--cfg",
        f"deq/mdeq_vision/experiments/cifar/cls_mdeq_{config}.yaml",
        "--percent",
        "0.0035",
    ]
    opts = [
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
    with patch("sys.argv", args + opts):
        main()

    args[1] = "--n_batches"
    args.extend([
        "--b_thres_range", "1", "2", "1",
        "--f_thres_range", "1", "2", "1",
    ])
    opts += [
        "TEST.MODEL_FILE", "output/cifar10/cls_mdeq_TINY/checkpoint_1.pth.tar",
        "TRAIN.BEGIN_EPOCH", "1",
        "TRAIN.WARM_INIT_DIR", "./",
        "TRAIN.BATCH_SIZE_PER_GPU", "2",
    ]
    with patch("sys.argv", args + opts):
        implicit_regime_main()


@pytest.mark.parametrize("config", [
    "TINY",
])
@pytest.mark.skipif(
    os.environ.get('CI', False) == 'true',
    reason='The full warm init test is too long for CI',
)
def test_cls_implicit_regime_sanity_check(config, clean_up_files):
    args = [
        "main",
        "--save_at",
        "1",
        "--cfg",
        f"deq/mdeq_vision/experiments/cifar/cls_mdeq_{config}.yaml",
        "--percent",
        "0.0035",
    ]
    opts = [
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
    with patch("sys.argv", args + opts):
        main()

    args[1] = "--n_batches"
    args.extend([
        "--b_thres_range", "100", "101", "1",
        "--f_thres_range", "100", "101", "1",
    ])
    opts += [
        "TEST.MODEL_FILE", "output/cifar10/cls_mdeq_TINY/checkpoint_1.pth.tar",
        "TRAIN.BEGIN_EPOCH", "1",
        "TRAIN.WARM_INIT_DIR", "./",
        "TRAIN.BATCH_SIZE_PER_GPU", "2",
    ]
    with patch("sys.argv", args + opts):
        implicit_regime_main()

    df_results = pd.read_csv("implicit_regime_identification.csv")
    diff_norm = df_results['grad_diff_norm']
    unrolled_diff_norm = df_results['unrolled_grad_diff_norm']
    np.testing.assert_almost_equal(diff_norm, np.zeros(len(diff_norm)))
    np.testing.assert_almost_equal(unrolled_diff_norm, np.zeros(len(unrolled_diff_norm)))
    assert len(df_results) == 2
