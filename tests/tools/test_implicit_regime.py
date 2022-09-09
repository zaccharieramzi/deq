import os
from pathlib import Path

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

    model_directory = Path("output/cifar10/cls_mdeq_TINY")
    trained_model_file = model_directory / "checkpoint_14.pth.tar"
    if trained_model_file.exists():
        model_file = trained_model_file
        epoch = 14
    else:
        model_file = model_directory / "checkpoint_1.pth.tar"
        epoch = 1
        with patch("sys.argv", args + opts):
            main()

    args[1] = "--n_batches"
    n_iter = 50
    args.extend([
        "--b_thres_range", str(n_iter), str(n_iter+1), "1",
        "--f_thres_range", str(n_iter), str(n_iter+1), "1",
    ])

    opts += [
        "TEST.MODEL_FILE", str(model_file),
        "TRAIN.BEGIN_EPOCH", str(epoch),
        "TRAIN.WARM_INIT_DIR", "./",
        "TRAIN.BATCH_SIZE_PER_GPU", "1",
    ]
    with patch("sys.argv", args + opts):
        implicit_regime_main()

    df_results = pd.read_csv("implicit_regime_identification.csv")
    diff_norm = df_results['true_grad_diff_norm']
    np.testing.assert_allclose(diff_norm, np.zeros(len(diff_norm)), atol=2)
    # we cannot test for the consistency between unrolled and ift without a
    # trained model
    # because the fixed point iterations are not stable
    if trained_model_file.exists():
        # load the model weights
        unrolled_diff_norm = df_results['unrolled_grad_diff_norm']
        np.testing.assert_allclose(unrolled_diff_norm, np.zeros(len(unrolled_diff_norm)), atol=2)
    assert len(df_results) == 1
