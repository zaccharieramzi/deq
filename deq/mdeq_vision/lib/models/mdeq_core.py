from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import functools
import logging
import os
from pathlib import Path
import sys

import numpy as np
from termcolor import colored
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._utils
import torch.nn.functional as F
import torch.autograd as autograd

from deq.lib.optimizations import VariationalHidDropout2d, weight_norm
from deq.lib.solvers import anderson, broyden, backprop_broyden
from deq.lib.solvers import power_method as fixed_point_iteration
from deq.lib.jacobian import jac_loss_estimate, power_method
from deq.lib.layer_utils import list2vec, vec2list, norm_diff, conv3x3, conv5x5
from deq.mdeq_vision.lib.utils.utils import get_world_size, get_rank


BN_MOMENTUM = 0.1
BLOCK_GN_AFFINE = True    # Don't change the value here. The value is controlled by the yaml files.
FUSE_GN_AFFINE = True     # Don't change the value here. The value is controlled by the yaml files.
POST_GN_AFFINE = True     # Don't change the value here. The value is controlled by the yaml files.
DEQ_EXPAND = 5        # Don't change the value here. The value is controlled by the yaml files.
NUM_GROUPS = 4        # Don't change the value here. The value is controlled by the yaml files.
logger = logging.getLogger(__name__)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, n_big_kernels=0, dropout=0.0, wnorm=False):
        """
        A canonical residual block with two 3x3 convolutions and an intermediate ReLU. Corresponds to Figure 2
        in the paper.
        """
        super(BasicBlock, self).__init__()
        conv1 = conv5x5 if n_big_kernels >= 1 else conv3x3
        conv2 = conv5x5 if n_big_kernels >= 2 else conv3x3
        inner_planes = int(DEQ_EXPAND*planes)

        self.conv1 = conv1(inplanes, inner_planes)
        self.gn1 = nn.GroupNorm(NUM_GROUPS, inner_planes, affine=BLOCK_GN_AFFINE)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv2(inner_planes, planes)
        self.gn2 = nn.GroupNorm(NUM_GROUPS, planes, affine=BLOCK_GN_AFFINE)

        self.gn3 = nn.GroupNorm(NUM_GROUPS, planes, affine=BLOCK_GN_AFFINE)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.drop = VariationalHidDropout2d(dropout)
        if wnorm: self._wnorm()

    def _wnorm(self):
        """
        Register weight normalization
        """
        self.conv1, self.conv1_fn = weight_norm(self.conv1, names=['weight'], dim=0)
        self.conv2, self.conv2_fn = weight_norm(self.conv2, names=['weight'], dim=0)

    def _reset(self, bsz, d, H, W):
        """
        Reset dropout mask and recompute weight via weight normalization
        """
        if 'conv1_fn' in self.__dict__:
            self.conv1_fn.reset(self.conv1)
        if 'conv2_fn' in self.__dict__:
            self.conv2_fn.reset(self.conv2)
        self.drop.reset_mask(bsz, d, H, W)

    def forward(self, x, injection=None):
        if injection is None: injection = 0
        residual = x

        out = self.relu(self.gn1(self.conv1(x)))
        out = self.drop(self.conv2(out)) + injection
        out = self.gn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.gn3(self.relu3(out))
        return out


blocks_dict = { 'BASIC': BasicBlock }


class BranchNet(nn.Module):
    def __init__(self, blocks):
        """
        The residual block part of each resolution stream
        """
        super().__init__()
        self.blocks = blocks

    def forward(self, x, injection=None):
        blocks = self.blocks
        y = blocks[0](x, injection)
        for i in range(1, len(blocks)):
            y = blocks[i](y)
        return y


class DownsampleModule(nn.Module):
    def __init__(self, num_channels, in_res, out_res):
        """
        A downsample step from resolution j (with in_res) to resolution i (with out_res). A series of 2-strided convolutions.
        """
        super(DownsampleModule, self).__init__()
        # downsample (in_res=j, out_res=i)
        convs = []
        inp_chan = num_channels[in_res]
        out_chan = num_channels[out_res]
        self.level_diff = level_diff = out_res - in_res

        kwargs = {"kernel_size": 3, "stride": 2, "padding": 1, "bias": False}
        for k in range(level_diff):
            intermediate_out = out_chan if k == (level_diff-1) else inp_chan
            components = [('conv', nn.Conv2d(inp_chan, intermediate_out, **kwargs)),
                          ('gnorm', nn.GroupNorm(NUM_GROUPS, intermediate_out, affine=FUSE_GN_AFFINE))]
            if k != (level_diff-1):
                components.append(('relu', nn.ReLU(inplace=True)))
            convs.append(nn.Sequential(OrderedDict(components)))
        self.net = nn.Sequential(*convs)

    def forward(self, x):
        return self.net(x)


class UpsampleModule(nn.Module):
    def __init__(self, num_channels, in_res, out_res):
        """
        An upsample step from resolution j (with in_res) to resolution i (with out_res).
        Simply a 1x1 convolution followed by an interpolation.
        """
        super(UpsampleModule, self).__init__()
        # upsample (in_res=j, out_res=i)
        inp_chan = num_channels[in_res]
        out_chan = num_channels[out_res]
        self.level_diff = level_diff = in_res - out_res

        self.net = nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(inp_chan, out_chan, kernel_size=1, bias=False)),
                        ('gnorm', nn.GroupNorm(NUM_GROUPS, out_chan, affine=FUSE_GN_AFFINE)),
                        ('upsample', nn.Upsample(scale_factor=2**level_diff, mode='nearest'))]))

    def forward(self, x):
        return self.net(x)


class MDEQModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_channels, big_kernels, dropout=0.0):
        """
        An MDEQ layer (note that MDEQ only has one layer).
        """
        super(MDEQModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_channels, big_kernels)

        self.num_branches = num_branches
        self.num_channels = num_channels
        self.big_kernels = big_kernels

        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels, big_kernels, dropout=dropout)
        self.fuse_layers = self._make_fuse_layers()
        self.post_fuse_layers = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('relu', nn.ReLU(False)),
                ('conv', nn.Conv2d(num_channels[i], num_channels[i], kernel_size=1, bias=False)),
                ('gnorm', nn.GroupNorm(NUM_GROUPS // 2, num_channels[i], affine=POST_GN_AFFINE))
            ])) for i in range(num_branches)])

    def _check_branches(self, num_branches, blocks, num_blocks, num_channels, big_kernels):
        """
        To check if the config file is consistent
        """
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(big_kernels):
            error_msg = 'NUM_BRANCHES({}) <> BIG_KERNELS({})'.format(
                num_branches, len(big_kernels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _wnorm(self):
        """
        Apply weight normalization to the learnable parameters of MDEQ
        """
        self.post_fuse_fns = []
        for i, branch in enumerate(self.branches):
            for block in branch.blocks:
                block._wnorm()
            conv, fn = weight_norm(self.post_fuse_layers[i].conv, names=['weight'], dim=0)
            self.post_fuse_fns.append(fn)
            self.post_fuse_layers[i].conv = conv

        # Throw away garbage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _reset(self, xs):
        """
        Reset the dropout mask and the learnable parameters (if weight normalization is applied)
        """
        for i, branch in enumerate(self.branches):
            for block in branch.blocks:
                block._reset(*xs[i].shape)
            if 'post_fuse_fns' in self.__dict__:
                self.post_fuse_fns[i].reset(self.post_fuse_layers[i].conv)    # Re-compute (...).conv.weight using _g and _v

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, big_kernels, stride=1, dropout=0.0):
        """
        Make a specific branch indexed by `branch_index`. This branch contains `num_blocks` residual blocks of type `block`.
        """
        layers = nn.ModuleList()
        n_channel = num_channels[branch_index]
        n_big_kernels = big_kernels[branch_index]
        for i in range(num_blocks[branch_index]):
            layers.append(block(n_channel, n_channel, n_big_kernels=n_big_kernels, dropout=dropout))
        return BranchNet(layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels, big_kernels, dropout=0.0):
        """
        Make the residual block (s; default=1 block) of MDEQ's f_\theta layer. Specifically,
        it returns `branch_layers[i]` gives the module that operates on input from resolution i.
        """
        branch_layers = [self._make_one_branch(i, block, num_blocks, num_channels, big_kernels, dropout=dropout) for i in range(num_branches)]
        return nn.ModuleList(branch_layers)

    def _make_fuse_layers(self):
        """
        Create the multiscale fusion layer (which does simultaneous up- and downsamplings).
        """
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_channels = self.num_channels
        fuse_layers = []
        for i in range(num_branches):
            fuse_layer = []                    # The fuse modules into branch #i
            for j in range(num_branches):
                if i == j:
                    fuse_layer.append(None)    # Identity if the same branch
                else:
                    module = UpsampleModule if j > i else DownsampleModule
                    fuse_layer.append(module(num_channels, in_res=j, out_res=i))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        # fuse_layers[i][j] gives the (series of) conv3x3s that convert input from branch j to branch i
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_channels

    def forward(self, x, injection, *args):
        """
        The two steps of a multiscale DEQ module (see paper): a per-resolution residual block and
        a parallel multiscale fusion step.
        """
        if injection is None:
            injection = [0] * len(x)
        if self.num_branches == 1:
            return [self.branches[0](x[0], injection[0])]

        # Step 1: Per-resolution residual block
        x_block = []
        for i in range(self.num_branches):
            x_block.append(self.branches[i](x[i], injection[i]))

        # Step 2: Multiscale fusion
        x_fuse = []
        for i in range(self.num_branches):
            y = 0
            # Start fusing all #j -> #i up/down-samplings
            for j in range(self.num_branches):
                y += x_block[j] if i == j else self.fuse_layers[i][j](x_block[j])
            x_fuse.append(self.post_fuse_layers[i](y))
        return x_fuse


class MDEQNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        """
        Build an MDEQ model with the given hyperparameters

        Args:
            cfg ([config]): The configuration file (parsed from yaml) specifying the model settings
        """
        super(MDEQNet, self).__init__()
        global BN_MOMENTUM
        BN_MOMENTUM = kwargs.get('BN_MOMENTUM', 0.1)
        self.parse_cfg(cfg)
        init_chansize = self.init_chansize

        self.downsample = nn.Sequential(
            conv3x3(3, init_chansize, stride=(2 if self.downsample_times >= 1 else 1)),
            nn.BatchNorm2d(init_chansize, momentum=BN_MOMENTUM, affine=True),
            nn.ReLU(inplace=True),
            conv3x3(init_chansize, init_chansize, stride=(2 if self.downsample_times >= 2 else 1)),
            nn.BatchNorm2d(init_chansize, momentum=BN_MOMENTUM, affine=True),
            nn.ReLU(inplace=True))

        if self.downsample_times > 2:
            for i in range(3, self.downsample_times+1):
                self.downsample.add_module(f"DS{i}", conv3x3(init_chansize, init_chansize, stride=2))
                self.downsample.add_module(f"DS{i}-BN", nn.BatchNorm2d(init_chansize, momentum=BN_MOMENTUM, affine=True))
                self.downsample.add_module(f"DS{i}-RELU", nn.ReLU(inplace=True))

        # PART I: Input injection module
        if self.downsample_times == 0 and self.num_branches <= 2:
            # We use the downsample module above as the injection transformation
            self.stage0 = None
        else:
            self.stage0 = nn.Sequential(nn.Conv2d(self.init_chansize, self.init_chansize, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(self.init_chansize, momentum=BN_MOMENTUM, affine=True),
                                        nn.ReLU(inplace=True))

        # PART II: MDEQ's f_\theta layer
        self.fullstage = self._make_stage(self.fullstage_cfg, self.num_channels, dropout=self.dropout)
        self.alternative_mode = "abs" if self.stop_mode == "rel" else "rel"
        if self.wnorm:
            self.fullstage._wnorm()

        self.iodrop = VariationalHidDropout2d(0.0)  # this is a no-op
        self.hook = None

    def parse_cfg(self, cfg):
        """
        Parse a configuration file
        """
        global DEQ_EXPAND, NUM_GROUPS, BLOCK_GN_AFFINE, FUSE_GN_AFFINE, POST_GN_AFFINE
        self.num_branches = cfg['MODEL']['EXTRA']['FULL_STAGE']['NUM_BRANCHES']
        self.num_channels = cfg['MODEL']['EXTRA']['FULL_STAGE']['NUM_CHANNELS']
        self.init_chansize = self.num_channels[0]
        self.num_layers = cfg['MODEL']['NUM_LAYERS']
        self.dropout = cfg['MODEL']['DROPOUT']
        self.wnorm = cfg['MODEL']['WNORM']
        self.num_classes = cfg['MODEL']['NUM_CLASSES']
        self.downsample_times = cfg['MODEL']['DOWNSAMPLE_TIMES']
        self.fullstage_cfg = cfg['MODEL']['EXTRA']['FULL_STAGE']
        self.pretrain_steps = cfg['TRAIN']['PRETRAIN_STEPS']
        self.all_unrolled = cfg['TRAIN']['ALL_UNROLLED']
        self.head_only = cfg['TRAIN']['HEAD_ONLY']

        # DEQ related
        self.f_solver = eval(cfg['DEQ']['F_SOLVER'])
        self.b_solver = eval(cfg['DEQ']['B_SOLVER'])
        if self.b_solver is None:
            self.b_solver = self.f_solver
        if cfg['DEQ']['F_SOLVER'] == 'fixed_point_iteration':
            self.f_solver_kwargs = {
                'step_size': cfg['DEQ']['F_STEP_SIZE'],
                'ls': cfg['DEQ']['F_LS'],
            }
        else:
            self.f_solver_kwargs = {}
        if cfg['DEQ']['B_SOLVER'] == 'fixed_point_iteration':
            self.b_solver_kwargs = {
                'step_size': cfg['DEQ']['B_STEP_SIZE'],
                'ls': cfg['DEQ']['B_LS'],
            }
        else:
            self.b_solver_kwargs = {}
        self.f_thres = cfg['DEQ']['F_THRES']
        self.b_thres = cfg['DEQ']['B_THRES']
        self.stop_mode = cfg['DEQ']['STOP_MODE']
        self.f_eps = cfg['DEQ']['F_EPS']
        self.b_eps = cfg['DEQ']['B_EPS']
        self.rand_init = cfg['DEQ']['RAND_INIT']

        # Update global variables
        DEQ_EXPAND = cfg['MODEL']['EXPANSION_FACTOR']
        NUM_GROUPS = cfg['MODEL']['NUM_GROUPS']
        BLOCK_GN_AFFINE = cfg['MODEL']['BLOCK_GN_AFFINE']
        FUSE_GN_AFFINE = cfg['MODEL']['FUSE_GN_AFFINE']
        POST_GN_AFFINE = cfg['MODEL']['POST_GN_AFFINE']

        # warm init related
        self.warm_init_dir = cfg['TRAIN']['WARM_INIT_DIR']
        if self.warm_init_dir is not None:
            self.warm_init_dir = Path(self.warm_init_dir)

    def _make_stage(self, layer_config, num_channels, dropout=0.0):
        """
        Build an MDEQ block with the given hyperparameters
        """
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        block_type = blocks_dict[layer_config['BLOCK']]
        big_kernels = layer_config['BIG_KERNELS']
        return MDEQModule(num_branches, block_type, num_blocks, num_channels, big_kernels, dropout=dropout)

    def _forward(self, x, train_step=-1, compute_jac_loss=True, spectral_radius_mode=False, writer=None, **kwargs):
        """
        The core MDEQ module. In the starting phase, we can (optionally) enter a shallow stacked f_\theta training mode
        to warm up the weights (specified by the self.pretrain_steps; see below)
        """
        num_branches = self.num_branches
        f_thres = kwargs.get('f_thres', self.f_thres)
        f_eps = kwargs.get('f_eps', self.f_eps)
        f_ls = kwargs.get('f_ls', None)
        b_thres = kwargs.get('b_thres', self.b_thres)
        b_eps = kwargs.get('b_eps', self.b_eps)
        z_list = kwargs.get('z_list', None)
        z1 = kwargs.get('z1', None)
        init_tensors = kwargs.get('init_tensors', None)
        grad_init = kwargs.get('grad_init', None)
        indices = kwargs.get('indices', None)
        return_inits = kwargs.get('return_inits', False)
        return_result = kwargs.get('return_result', False)
        data_aug_invariance = kwargs.get('data_aug_invariance', False)
        unrolled_broyden = kwargs.get('unrolled_broyden', False)
        if data_aug_invariance:
            # in this case x shape is
            # (batch_size, n_aug, channels, height, width)
            n_aug = x.size()[1]
            n_unique_images = x.size()[0]
            x = x.reshape(n_aug*n_unique_images, *x.size()[2:])
        save_grad_result = kwargs.get('save_grad_result', False)
        x = self.downsample(x)
        rank = get_rank()
        deq_mode = (train_step < 0) or (train_step >= self.pretrain_steps)

        # Inject only to the highest resolution...
        x_list = [self.stage0(x) if self.stage0 else x]
        for i in range(1, num_branches):
            bsz, _, H, W = x_list[-1].shape
            x_list.append(torch.zeros(bsz, self.num_channels[i], H//2, W//2).to(x))   # ... and the rest are all zeros

        if z_list is None or not deq_mode:
            z_list = [
                torch.randn_like(elem) if self.rand_init else torch.zeros_like(elem)
                for elem in x_list
            ]
        if z1 is None or not deq_mode:
            z1 = list2vec(z_list)
        cutoffs = [(elem.size(1), elem.size(2), elem.size(3)) for elem in z_list]
        func = lambda z: list2vec(self.fullstage(vec2list(z, cutoffs), x_list))

        # For variational dropout mask resetting and weight normalization re-computations
        self.fullstage._reset(z_list)

        jac_loss = torch.tensor(0.0).to(x)
        sradius = torch.zeros(bsz, 1).to(x)

        # Multiscale Deep Equilibrium!
        if not deq_mode:
            for layer_ind in range(self.num_layers):
                if self.f_solver_kwargs:
                    step_size = self.f_solver_kwargs['step_size']
                else:
                    step_size = 1.0
                z1 = z1 + step_size * (func(z1) - z1)
            new_z1 = z1

            if self.training:
                if compute_jac_loss:
                    z2 = z1.clone().detach().requires_grad_()
                    new_z2 = func(z2)
                    jac_loss = jac_loss_estimate(new_z2, z2)
        elif unrolled_broyden or self.all_unrolled:
            result_fw = backprop_broyden(
                func,
                z1,
                threshold=f_thres,
                stop_mode=self.stop_mode,
                name="forward",
                eps=1e-6,
            )
            new_z1 = func(result_fw.pop('result'))
        else:
            with torch.no_grad():
                f_solver_kwargs = self.f_solver_kwargs
                if f_ls is not None:
                    f_solver_kwargs.update(ls=f_ls)
                result_fw = self.f_solver(
                    func,
                    z1,
                    threshold=f_thres,
                    stop_mode=self.stop_mode,
                    name="forward",
                    init_tensors=init_tensors,
                    eps=f_eps,
                    **self.f_solver_kwargs,
                )
                z1 = result_fw.pop('result')
            new_z1 = z1

            if (not self.training) and spectral_radius_mode:
                with torch.enable_grad():
                    new_z1 = func(z1.requires_grad_())
                _, sradius = power_method(new_z1, z1, n_iters=150)

            if self.training and not self.head_only:
                new_z1 = func(z1.requires_grad_())
                if compute_jac_loss:
                    jac_loss = jac_loss_estimate(new_z1, z1)

                def jac(y, grad):
                    return autograd.grad(new_z1, z1, y, retain_graph=True)[0] + grad

                def backward_hook(grad):
                    if self.hook is not None:
                        self.hook.remove()
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                    if grad_init is None:
                        grad_init_ = torch.zeros_like(grad)
                    else:
                        grad_init_ = grad_init
                    result_bw = self.b_solver(
                        functools.partial(jac, grad=grad), grad_init_,
                        threshold=b_thres, stop_mode=self.stop_mode, name="backward", eps=b_eps,
                        **self.b_solver_kwargs,
                    )
                    new_grad = result_bw.pop('result')
                    # save the new gradients per elements of the batch
                    # according to their indices
                    if self.warm_init_dir is not None and indices is not None:
                        for i_batch, idx in enumerate(indices):
                            g = new_grad[i_batch].cpu()
                            fname = f'{idx.cpu().numpy().item()}_back.pt'
                            torch.save(
                                g,
                                self.warm_init_dir / fname,
                            )
                    if save_grad_result:
                        torch.save(
                            result_bw,
                            self.warm_init_dir / 'grad_result.pt',
                        )
                    return new_grad
                self.hook = new_z1.register_hook(backward_hook)
        if data_aug_invariance:
            batched_z1 = new_z1[..., 0].reshape(
                n_unique_images,
                n_aug,
                -1,
            )
            distance_matrix = torch.cat([torch.cdist(
                batched_z1[i_batch].unsqueeze(0),
                batched_z1[i_batch].unsqueeze(0),
            )[0] for i_batch in range(n_unique_images)], dim=0)

        y_list = self.iodrop(vec2list(new_z1, cutoffs))  # this is a no-op
        return_objects = (y_list, jac_loss.view(1, -1), sradius.view(-1, 1))
        if data_aug_invariance:
            return_objects += (distance_matrix,)
        if return_inits:
            if deq_mode:
                new_inits = [
                    new_z1.detach().clone(),
                    result_fw.get('Us', None),
                    result_fw.get('VTs', None),
                    torch.tensor(result_fw['nstep']).to(x).repeat(bsz),
                ]
            else:
                new_inits = None
            return *return_objects, new_inits
        if return_result:
            return *return_objects, result_fw
        else:
            return return_objects

    def forward(self, x, train_step=-1, **kwargs):
        raise NotImplemented    # To be inherited & implemented by MDEQClsNet and MDEQSegNet (see mdeq.py)
