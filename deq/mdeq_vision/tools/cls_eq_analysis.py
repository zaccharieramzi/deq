# Modified based on the HRNet repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from pathlib import Path
import pprint
import time

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
from torch.utils.tensorboard import SummaryWriter

from deq.lib.optimizations import VariationalHidDropout2d
from deq.mdeq_vision.lib import models  # noqa F401
from deq.mdeq_vision.lib.config import config
from deq.mdeq_vision.lib.config import update_config
from deq.mdeq_vision.lib.core.cls_function import train
from deq.mdeq_vision.lib.utils.utils import get_optimizer
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
    parser.add_argument('--save_at',
                        help='''save checkpoint at certain epochs,
                        space-separated''',
                        type=int,
                        nargs='+')
    parser.add_argument('--dropout_eval',
                        help='whether to use dropout during the evaluation',
                        action='store_true')
    parser.add_argument('--broyden_matrices',
                        help='whether to use broyden matrices when doing warm init',
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
    Set the --broyden_matrices to use broyden matrices when doing warm init.
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

    writer_dict = {
        'writer': (
            SummaryWriter(log_dir=tb_log_dir)
            if not config.DEBUG.DEBUG else None
        ),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    if torch.cuda.is_available():
        gpus = list(config.GPUS)
        model = nn.DataParallel(model, device_ids=gpus).cuda()
    print("Finished constructing model!")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    optimizer = get_optimizer(config, model)
    lr_scheduler = None

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

    # Update weight decay if needed
    checkpoint['optimizer']['param_groups'][0]['weight_decay'] = config.TRAIN.WD
    optimizer.load_state_dict(checkpoint['optimizer'])

    if 'lr_scheduler' in checkpoint:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1e5,
                            last_epoch=checkpoint['lr_scheduler']['last_epoch'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

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

    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU
    if torch.cuda.is_available():
        batch_size = batch_size * len(config.GPUS)
    aug_train_loader = torch.utils.data.DataLoader(
        aug_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True,
        generator=torch.Generator(device=device_str),
    )

    # Learning rate scheduler
    if lr_scheduler is None:
        if config.TRAIN.LR_SCHEDULER != 'step':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, len(aug_train_loader)*config.TRAIN.END_EPOCH, eta_min=1e-6)
        elif isinstance(config.TRAIN.LR_STEP, list):
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
                last_epoch-1)
        else:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
                last_epoch-1)

    # Loaded a checkpoint
    writer_dict['train_global_steps'] = last_epoch * len(aug_train_loader)

    # Evaluating convergence before training
    model.eval()
    if args.dropout_eval:
        set_dropout_modules_active(model)
    df_results = pd.DataFrame(columns=[
        'image_index',
        'before_training',
        'trace',
        'abs_trace',
        'init_type',
        'is_aug',
        'i_iter',
        'analysis_time',
        'seed',
        'dataset',
        'model_size',
        'checkpoint',
        'dropout',
        'percent',
        'broyden_matrices',
    ])
    model_size = Path(args.cfg).stem[9:]
    common_args = dict(
        model_size=model_size,
        dataset=dataset_name,
        checkpoint=last_epoch,
        seed=args.seed,
        analysis_time=time.time(),
        dropout=args.dropout_eval,
        percent=args.percent_eval*100,
        broyden_matrices=args.broyden_matrices,
    )

    def fill_df_results(df_results, result_info,  **data_kwargs):
        n_step = result_info['nstep']
        trace = result_info['rel_trace'][:n_step]
        abs_trace = result_info['abs_trace'][:n_step]
        i_iter = np.arange(n_step)
        df_trace = pd.DataFrame(data={
            'trace': trace,
            'abs_trace': abs_trace,
            'i_iter': i_iter,
            **data_kwargs,
            **common_args,
        })
        df_results = df_results.append(df_trace, ignore_index=True)
        return df_results

    image_indices = np.random.choice(
        len(train_dataset),
        args.n_images,
        replace=False,
    )
    vanilla_inits = {}
    aug_inits = {}
    fn = model.module._forward if torch.cuda.is_available() else model._forward
    for image_index in image_indices:
        image, _ = train_dataset[image_index]
        image = image.unsqueeze(0)
        if torch.cuda.is_available():
            image = image.cuda()
        # pot in kwargs we can have: f_thres, b_thres, lim_mem
        y_list, *_, result_info = fn(image, train_step=-1, return_result=True)
        vanilla_inits[image_index] = {
            'y_list': y_list,
            'Us': result_info['Us'],
            'VTs': result_info['VTs'],
            'nstep': result_info['nstep'],
        }
        df_results = fill_df_results(
            df_results,
            result_info,
            image_index=image_index,
            before_training=True,
            init_type=None,
            is_aug=False,
        )
        aug_image, _ = aug_train_dataset[image_index]
        aug_image = aug_image.unsqueeze(0)
        if torch.cuda.is_available():
            aug_image = aug_image.cuda()
        aug_y_list, *_, result_info = fn(aug_image, train_step=-1, return_result=True)
        aug_inits[image_index] = {
            'y_list': aug_y_list,
            'Us': result_info['Us'],
            'VTs': result_info['VTs'],
            'nstep': result_info['nstep'],
        }

    # Training code
    topk = (1, 5) if dataset_name == 'imagenet' else (1,)
    if config.TRAIN.LR_SCHEDULER == 'step':
        lr_scheduler.step()

    # train for one epoch
    train(
        config,
        aug_train_loader,
        model,
        criterion,
        optimizer,
        lr_scheduler,
        last_epoch,
        final_output_dir,
        tb_log_dir,
        writer_dict,
        topk=topk,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if writer_dict['writer'] is not None:
        writer_dict['writer'].flush()

    if writer_dict['writer'] is not None:
        writer_dict['writer'].close()

    model.eval()
    if args.dropout_eval:
        set_dropout_modules_active(model)
    for image_index in image_indices:
        image, _ = train_dataset[image_index]
        image = image.unsqueeze(0)
        if torch.cuda.is_available():
            image = image.cuda()
        *_, result = fn(image, train_step=-1, return_result=True)
        df_results = fill_df_results(
            df_results,
            result,
            image_index=image_index,
            before_training=False,
            init_type=None,
            is_aug=False,
        )
        add_kwargs_vanilla = {}
        add_kwargs_aug = {}
        if args.broyden_matrices:
            for add_kwargs, inits in zip([add_kwargs_vanilla, add_kwargs_aug], [vanilla_inits, aug_inits]):
                init_tensors = [
                    inits[image_index][k]
                    for k in ['nstep', 'Us', 'VTs']
                ]
                add_kwargs['init_tensors'] = init_tensors
        *_, result = fn(
            image,
            train_step=-1,
            return_result=True,
            z_list=vanilla_inits[image_index]['y_list'],
            **add_kwargs_vanilla,
        )
        df_results = fill_df_results(
            df_results,
            result,
            image_index=image_index,
            before_training=False,
            init_type='vanilla',
            is_aug=False,
        )
        new_aug_image, _ = aug_train_dataset[image_index]
        new_aug_image = new_aug_image.unsqueeze(0)
        if torch.cuda.is_available():
            new_aug_image = new_aug_image.cuda()
        *_, result = fn(
            new_aug_image,
            train_step=-1,
            return_result=True,
            z_list=aug_inits[image_index]['y_list'],
            **add_kwargs_aug,
        )
        df_results = fill_df_results(
            df_results,
            result,
            image_index=image_index,
            before_training=False,
            init_type='aug',
            is_aug=True,
        )
    results_name = 'eq_init_results.csv'
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
