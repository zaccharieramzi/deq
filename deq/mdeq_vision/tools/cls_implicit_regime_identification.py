# Modified based on the HRNet repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import itertools
import os
from pathlib import Path
import pprint

import pandas as pd
from termcolor import colored
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from deq.lib.optimizations import VariationalHidDropout
from deq.mdeq_vision.lib import models  # noqa F401
from deq.mdeq_vision.lib.config import config
from deq.mdeq_vision.lib.config import update_config
from deq.mdeq_vision.lib.datasets.indexed_dataset import IndexedDataset
from deq.mdeq_vision.lib.utils.utils import create_logger


def set_modules_inactive(model, deactivate_dropout=False):
    inactive_types = [nn.BatchNorm2d, nn.GroupNorm]
    if deactivate_dropout:
        inactive_types.append(VariationalHidDropout)
    inactive_types = tuple(inactive_types)
    for m in model.modules():
        if isinstance(m, inactive_types):
            m.eval()


def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        default='')
    parser.add_argument('--percent',
                        help='percentage of training data to use',
                        type=float,
                        default=1.0)
    parser.add_argument('--save_at',
                        help='''save checkpoint at certain epochs,
                        space-separated''',
                        type=int,
                        nargs='+')
    parser.add_argument('--n_batches',
                        help='number of batches to use for evaluation',
                        type=int,
                        default=10)
    parser.add_argument('--b_thres_range',
                        help='range of backward threshold to use for evaluation',
                        type=int,
                        nargs=3)
    parser.add_argument('--f_thres_range',
                        help='range of forward threshold to use for evaluation',
                        type=int,
                        nargs=3)
    parser.add_argument('--dropout_eval',
                        help='whether to use dropout during the evaluation',
                        action='store_true')
    parser.add_argument('--seed',
                        help='random seed',
                        type=int,
                        default=None)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    """
    Set the --percent to make the duration of training vary.
    Set the TRAIN.BEGIN_EPOCH for the checkpoint
    """
    args = parse_args()
    seed = args.seed
    seeding = seed is not None
    torch.manual_seed(seed if seeding else 42)
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    if torch.cuda.is_available():
        print(colored("Setting default tensor type to cuda.FloatTensor", "cyan"))
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device_str = 'cuda'
    else:
        device_str = 'cpu'

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    if not seeding:
        cudnn.benchmark = config.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    else:
        cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.'+config.MODEL.NAME+'.get_cls_net')(config)
    if torch.cuda.is_available():
        model = model.cuda()

    if torch.cuda.is_available():
        gpus = list(config.GPUS)
        model = nn.DataParallel(model, device_ids=gpus).cuda()
    print("Finished constructing model!")

    last_epoch = config.TRAIN.BEGIN_EPOCH
    checkpoint_name = 'checkpoint'
    if seeding:
        checkpoint_name += f'_seed{seed}'
    model_state_file = os.path.join(
        final_output_dir,
        f'{checkpoint_name}_{last_epoch}.pth.tar',
    )
    if torch.cuda.is_available():
        checkpoint = torch.load(model_state_file)
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        checkpoint = torch.load(model_state_file, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])

    logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    # Data loading code
    dataset_name = config.DATASET.DATASET

    if dataset_name == 'imagenet':
        traindir = os.path.join(config.DATASET.ROOT+'/images', config.DATASET.TRAIN_SET)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        augment_list = [
            transforms.RandomResizedCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.RandomHorizontalFlip(),
        ]
        transform_train = transforms.Compose(augment_list + [
            transforms.ToTensor(),
            normalize,
        ])
        aug_train_dataset = datasets.ImageFolder(traindir, transform_train)
    else:
        assert dataset_name == "cifar10", "Only CIFAR-10 and ImageNet are supported at this phase"
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  # For reference

        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        augment_list = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        transform_train = transforms.Compose(augment_list + [
            transforms.ToTensor(),
            normalize,
        ])
        aug_train_dataset = datasets.CIFAR10(
            root=f'{config.DATASET.ROOT}',
            train=True,
            download=True,
            transform=transform_train,
        )

    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU
    if torch.cuda.is_available():
        batch_size = batch_size * len(config.GPUS)

    aug_train_dataset = IndexedDataset(aug_train_dataset)

    aug_train_loader = torch.utils.data.DataLoader(
        aug_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True,
        generator=torch.Generator(device=device_str),
    )

    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    df_results = pd.DataFrame(columns=[
        'b_thres',
        'f_thres',
        'true_grad_diff',
        'true_grad_diff_norm',
        'unrolled_grad_diff',
        'unrolled_grad_diff_norm',
        'image_index',
        'epoch',
        'seed',
        'opts',
    ])

    # Evaluating convergence before training
    model.train()
    set_modules_inactive(model, deactivate_dropout=not args.dropout_eval)

    warm_init_dir = Path(config.TRAIN.WARM_INIT_DIR)
    n_batches_seen = 0

    for batch in aug_train_loader:
        if n_batches_seen >= args.n_batches:
            break
        n_batches_seen += 1
        image, target, indices = batch
        if torch.cuda.is_available():
            image = image.cuda()
            target = target.cuda(non_blocking=True)

        def get_grad(model, f_thres, b_thres):
            output, *_ = model(
                image,
                train_step=-1,
                indices=indices,
                f_thres=f_thres,
                f_eps=1e-6,
                b_thres=b_thres,
                b_eps=1e-6,
            )
            loss = criterion(output, target)
            loss.backward()
            model.zero_grad()
            gradients = {}
            for image_index in indices:
                fname = warm_init_dir / f'{image_index}_back.pt'
                gradients[image_index.item()] = torch.load(fname)
            return gradients

        def get_broyden_unrolled_grad(model, f_thres):
            output, *_ = model(
                image,
                train_step=-1,
                indices=indices,
                f_thres=f_thres,
                f_eps=1e-6,
                unrolled_broyden=True,
            )
            loss = criterion(output, target)
            loss.backward()
            model.zero_grad()
            gradients = {}
            for image_index in indices:
                fname = warm_init_dir / f'{image_index}_back.pt'
                gradients[image_index.item()] = torch.load(fname)
            return gradients
        # pot in kwargs we can have: f_thres, b_thres, lim_mem
        # first let's get the true gradients
        # with a lot of iterations
        true_gradients = get_grad(model, 100, 100)
        # we now look at the approximate gradient
        # with a reduced number of iterations
        # we loop over the cartesian product
        # between args.f_thres_range and args.b_thres_range
        f_thres_range = range(*args.f_thres_range)
        b_thres_range = range(*args.b_thres_range)
        for f_thres, b_thres in itertools.product(f_thres_range, b_thres_range):
            approx_grad = get_grad(model, f_thres, b_thres)
            unrolled_broyden_grad = get_broyden_unrolled_grad(model, f_thres)
            # now we compute the difference between the two gradients
            # and we store the results in a dataframe
            for image_index in indices:
                i = image_index.item()
                grad_diff = torch.abs(true_gradients[i] - approx_grad[i]).sum().cpu().numpy().item()
                grad_diff_norm = grad_diff / torch.abs(true_gradients[i]).sum().cpu().numpy().item()
                unrolled_grad_diff = torch.abs(true_gradients[i] - unrolled_broyden_grad[i]).sum().cpu().numpy().item()
                unrolled_grad_diff_norm = unrolled_grad_diff / torch.abs(true_gradients[i]).sum().cpu().numpy().item()
                df_results.loc[len(df_results)] = [
                    b_thres,
                    f_thres,
                    grad_diff,
                    grad_diff_norm,
                    unrolled_grad_diff,
                    unrolled_grad_diff_norm,
                    i,
                    last_epoch,
                    seed,
                    args.opts,
                ]

    results_path = Path('implicit_regime_identification.csv')
    if results_path.exists():
        df_results.to_csv(results_path, mode='a', header=False)
    else:
        df_results.to_csv(results_path)


if __name__ == '__main__':
    main()
