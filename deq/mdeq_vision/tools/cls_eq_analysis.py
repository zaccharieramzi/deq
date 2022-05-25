# Modified based on the HRNet repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import sys

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

from deq.mdeq_vision.lib import models
from deq.mdeq_vision.lib.config import config
from deq.mdeq_vision.lib.config import update_config
from deq.mdeq_vision.lib.core.cls_function import train
from deq.mdeq_vision.lib.utils.modelsummary import get_model_summary
from deq.mdeq_vision.lib.utils.utils import get_optimizer
from deq.mdeq_vision.lib.utils.utils import save_checkpoint
from deq.mdeq_vision.lib.utils.utils import create_logger


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
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    torch.manual_seed(42)
    args = parse_args()
    torch.multiprocessing.set_start_method('spawn')
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
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.'+config.MODEL.NAME+'.get_cls_net')(config)
    if torch.cuda.is_available():
        model = model.cuda()


    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir) if not config.DEBUG.DEBUG else None,
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

    best_perf = 0.0
    best_model = False
    last_epoch = config.TRAIN.BEGIN_EPOCH
    model_state_file = os.path.join(final_output_dir, f'checkpoint_{last_epoch}.pth.tar')
    checkpoint = torch.load(model_state_file)
    model.module.load_state_dict(checkpoint['state_dict'])

    # Update weight decay if needed
    checkpoint['optimizer']['param_groups'][0]['weight_decay'] = config.TRAIN.WD
    optimizer.load_state_dict(checkpoint['optimizer'])

    if 'lr_scheduler' in checkpoint:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1e5,
                            last_epoch=checkpoint['lr_scheduler']['last_epoch'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    best_model = True

    # Data loading code
    dataset_name = config.DATASET.DATASET

    if dataset_name == 'imagenet':
        traindir = os.path.join(config.DATASET.ROOT+'/images', config.DATASET.TRAIN_SET)
        valdir = os.path.join(config.DATASET.ROOT+'/images', config.DATASET.TEST_SET)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = datasets.ImageFolder(traindir, transform_train)
    else:
        assert dataset_name == "cifar10", "Only CIFAR-10 and ImageNet are supported at this phase"
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  # For reference

        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        augment_list = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if config.DATASET.AUGMENT else []
        transform_train = transforms.Compose(augment_list + [
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = datasets.CIFAR10(root=f'{config.DATASET.ROOT}', train=True, download=True, transform=transform_train)

    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU
    test_batch_size = config.TEST.BATCH_SIZE_PER_GPU
    if torch.cuda.is_available():
        batch_size = batch_size * len(config.GPUS)
        test_batch_size = test_batch_size * len(config.GPUS)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
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
                optimizer, len(train_loader)*config.TRAIN.END_EPOCH, eta_min=1e-6)
        elif isinstance(config.TRAIN.LR_STEP, list):
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
                last_epoch-1)
        else:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
                last_epoch-1)

    if best_model:
        # Loaded a checkpoint
        writer_dict['train_global_steps'] = last_epoch * len(train_loader)

    # Training code
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        topk = (1,5) if dataset_name == 'imagenet' else (1,)
        if config.TRAIN.LR_SCHEDULER == 'step':
            lr_scheduler.step()

        # train for one epoch
        train(config, train_loader, model, criterion, optimizer, lr_scheduler, epoch,
              final_output_dir, tb_log_dir, writer_dict, topk=topk)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if writer_dict['writer'] is not None:
            writer_dict['writer'].flush()

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        checkpoint_files = ['checkpoint.pth.tar']
        if epoch in args.save_at:
            checkpoint_files.append(f'checkpoint_{epoch}.pth.tar')
        for checkpoint_file in checkpoint_files:
            save_checkpoint({
                'epoch': epoch + 1,
                'model': config.MODEL.NAME,
                'state_dict': model.module.state_dict() if torch.cuda.is_available() else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'train_global_steps': writer_dict['train_global_steps'],
            }, best_model, final_output_dir, filename=checkpoint_file)

    final_model_state_file = os.path.join(final_output_dir, 'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(final_model_state_file))
    state_dict = model.module.state_dict() if torch.cuda.is_available() else model.state_dict()
    torch.save(state_dict, final_model_state_file)
    if writer_dict['writer'] is not None:
        writer_dict['writer'].close()


if __name__ == '__main__':
    main()
