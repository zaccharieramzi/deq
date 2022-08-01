# Modified based on the HRNet repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import shutil
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from deq.mdeq_vision.lib import models
from deq.mdeq_vision.lib.config import config
from deq.mdeq_vision.lib.config import update_config
from deq.mdeq_vision.lib.core.cls_function import validate
from deq.mdeq_vision.lib.utils.modelsummary import get_model_summary
from deq.mdeq_vision.lib.utils.utils import create_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')

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
    parser.add_argument('--seed',
                        help='random seed',
                        type=int,
                        default=None)
    parser.add_argument('--valid_on_train',
                        help='validate on training data',
                        action='store_true')
    parser.add_argument('--use_warm_init',
                        help='use warm inits when validating on train data',
                        action='store_true')
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()
    seed = args.seed
    seeding = seed is not None
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.'+config.MODEL.NAME+'.get_cls_net')(config)

    if config.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
        model_state_file = config.TEST.MODEL_FILE
    else:
        base_final_state_name = 'final_state'
        if seeding:
            base_final_state_name += f'_seed{seed}'
        model_state_file = os.path.join(
            final_output_dir,
            f'{base_final_state_name}.pth.tar',
        )
        logger.info('=> loading model from {}'.format(model_state_file))
    if torch.cuda.is_available():
        state_dict = torch.load(model_state_file)
        device_str = 'cuda'
    else:
        state_dict = torch.load(
            model_state_file,
            map_location='cpu',
        )
        device_str = 'cpu'

    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # loading from a checkpoint and not the final state
        model.load_state_dict(state_dict['state_dict'])

    if torch.cuda.is_available():
        gpus = list(config.GPUS)
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    # Data loading code
    dataset_name = config.DATASET.DATASET
    topk = (1,5) if dataset_name == 'imagenet' else (1,)
    if dataset_name == 'imagenet':
        test_path = config.DATASET.TEST_SET if not args.valid_on_train else config.DATASET.TRAIN_SET
        valdir = os.path.join(config.DATASET.ROOT+'/images', test_path)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_valid = transforms.Compose([
            transforms.Resize(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
            transforms.CenterCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.ToTensor(),
            normalize,
        ])
        valid_dataset = datasets.ImageFolder(valdir, transform_valid)
    else:
        assert dataset_name == "cifar10", "Only CIFAR-10 and ImageNet are supported at this phase"
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  # For reference

        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transform_valid = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        valid_dataset = datasets.CIFAR10(root=f'{config.DATASET.ROOT}', train=args.valid_on_train, download=True, transform=transform_valid)

    test_batch_size = config.TEST.BATCH_SIZE_PER_GPU
    if torch.cuda.is_available():
        test_batch_size *= len(gpus)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
        generator=torch.Generator(device=device_str),
    )

    # evaluate on validation set
    validate(config, valid_loader, model, criterion, None, epoch=-1, output_dir=final_output_dir,
             tb_log_dir=tb_log_dir, writer_dict=None, topk=topk, spectral_radius_mode=config.DEQ.SPECTRAL_RADIUS_MODE)


if __name__ == '__main__':
    main()
