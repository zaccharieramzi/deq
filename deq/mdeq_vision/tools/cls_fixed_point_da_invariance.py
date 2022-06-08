# Modified based on the HRNet repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from pathlib import Path

import numpy as np
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

from deq.lib.optimizations import VariationalHidDropout2d
from deq.mdeq_vision.lib import models  # noqa F401
from deq.mdeq_vision.lib.config import config
from deq.mdeq_vision.lib.config import update_config
from deq.mdeq_vision.lib.utils.utils import create_logger


def set_dropout_modules_active(model):
    for m in model.modules():
        if isinstance(m, VariationalHidDropout2d):
            m.train()


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
    parser.add_argument('--dropout_eval',
                        help='whether to use dropout during the evaluation',
                        action='store_true')
    parser.add_argument('--n_images',
                        help='number of images to use for evaluation',
                        type=int,
                        default=10)
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
    Set the --dropout_eval to use dropout during the evaluation.
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

    _, final_output_dir, _ = create_logger(
        config, args.cfg, 'train')

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

    # define loss function (criterion) and optimizer
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

    # Data loading code
    dataset_name = config.DATASET.DATASET

    if dataset_name == 'imagenet':
        traindir = os.path.join(config.DATASET.ROOT+'/images', config.DATASET.TRAIN_SET)
        valdir = os.path.join(config.DATASET.ROOT+'/images', config.DATASET.TEST_SET)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        augment_list = [
            transforms.RandomResizedCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.RandomHorizontalFlip(),
        ]
        transform_train = transforms.Compose(augment_list + [
            transforms.ToTensor(),
            normalize,
        ])
        transform_valid = transforms.Compose([
            transforms.Resize(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
            transforms.CenterCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = datasets.ImageFolder(traindir, transform_valid)
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
        transform_valid = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = datasets.CIFAR10(
            root=f'{config.DATASET.ROOT}',
            train=True,
            download=True,
            transform=transform_valid,
        )
        aug_train_dataset = datasets.CIFAR10(
            root=f'{config.DATASET.ROOT}',
            train=True,
            download=True,
            transform=transform_train,
        )

    # Evaluating convergence before training
    model.eval()
    if args.dropout_eval:
        set_dropout_modules_active(model)

    image_indices = np.random.choice(
        len(train_dataset),
        args.n_images,
        replace=False,
    )
    fn = model
    df_results = pd.DataFrame(columns=[
        'image_index',
        'mse_fixed_points',
        'relative_mse_fixed_points',
        'mse_logits',
        'relative_mse_logits',
        'seed',
        'dropout',
        'checkpoint',
        'model_size',
        'dataset',
    ])
    model_size = Path(args.cfg).stem[9:]
    for image_index in image_indices:
        image, _ = train_dataset[image_index]
        image = image.unsqueeze(0)
        if torch.cuda.is_available():
            image = image.cuda()
        # pot in kwargs we can have: f_thres, b_thres, lim_mem
        vanilla_logits, *_, [vanilla_z1, *_] = fn(
            image,
            train_step=-1,
            return_inits=True,
        )
        aug_image, _ = aug_train_dataset[image_index]
        aug_image = aug_image.unsqueeze(0)
        if torch.cuda.is_available():
            aug_image = aug_image.cuda()
        aug_logits, *_, [aug_z1, *_] = fn(
            aug_image,
            train_step=-1,
            return_inits=True,
        )
        mse_fixed_points = torch.mean((vanilla_z1 - aug_z1)**2)
        relative_mse_fixed_points = mse_fixed_points / torch.mean(vanilla_z1**2)
        mse_logits = torch.mean((vanilla_logits - aug_logits)**2)
        relative_mse_logits = mse_logits / torch.mean(vanilla_logits**2)
        (
            mse_fixed_points,
            relative_mse_fixed_points,
            mse_logits,
            relative_mse_logits,
        ) = [t.detach().cpu().numpy().item() for t in (
            mse_fixed_points,
            relative_mse_fixed_points,
            mse_logits,
            relative_mse_logits,
        )]
        df_results = df_results.append({
            'image_index': image_index,
            'mse_fixed_points': mse_fixed_points,
            'relative_mse_fixed_points': relative_mse_fixed_points,
            'mse_logits': mse_logits,
            'relative_mse_logits': relative_mse_logits,
            'seed': args.seed,
            'dropout': args.dropout_eval,
            'checkpoint': last_epoch,
            'model_size': model_size,
            'dataset': dataset_name,
        }, ignore_index=True)

    results_name = 'fixed_point_invariance_results.csv'
    write_header = not Path(results_name).is_file()
    df_results.to_csv(
        results_name,
        mode='a',
        header=write_header,
        index=False,
    )
    return df_results


if __name__ == '__main__':
    main()
