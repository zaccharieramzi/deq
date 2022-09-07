import torch

from deq.lib.layer_utils import list2vec, vec2list
from deq.lib.solvers import broyden, backprop_broyden
from deq.mdeq_vision.lib.config import config as cfg
from deq.mdeq_vision.lib import models


def get_function_and_init():
    # get the TINY config
    cfg.defrost()
    cfg.merge_from_file("deq/mdeq_vision/experiments/cifar/cls_mdeq_TINY.yaml")
    cfg.freeze()

    # initialize the model
    model = models.mdeq.get_cls_net(cfg)

    # set up the function whose fixed point needs to be found
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    x = model.downsample(x)
    x_list = [model.stage0(x) if model.stage0 else x]
    for i in range(1, model.num_branches):
        bsz, _, H, W = x_list[-1].shape
        x_list.append(torch.zeros(bsz, model.num_channels[i], H//2, W//2).to(x))
    z_list = [torch.zeros_like(elem) for elem in x_list]
    z1 = list2vec(z_list)
    z1.requires_grad_()
    cutoffs = [(elem.size(1), elem.size(2), elem.size(3)) for elem in z_list]
    func = lambda z: list2vec(model.fullstage(vec2list(z, cutoffs), x_list))
    model.fullstage._reset(z_list)
    return z1, func


def test_can_backprop_broyden():
    """
    Tests whether we can backpropagate, i.e. apply backprop_broyden
    on a tensor that requires grad.
    For the sake of adequacy, we will use the true MDEQ function.
    """
    z1, func = get_function_and_init()

    result_fw = backprop_broyden(
        func,
        z1,
        threshold=2,
        stop_mode='abs',
        name="backprop",
        eps=1e-6,
        ls=False,
    )
    new_z1 = result_fw.pop('result')
    new_z1.backward(torch.ones_like(new_z1))
    assert z1.grad is not None


def test_broyden_consistency():
    """
    Tests whether the broyden solver and the backprop_broyden solver
    give the same result.
    For the sake of adequacy, we will use the true MDEQ function.
    """
    z1, func = get_function_and_init()

    result_fw = backprop_broyden(
        func,
        z1,
        threshold=2,
        stop_mode='abs',
        name="backprop",
        eps=1e-6,
        ls=False,
    )
    new_z1 = result_fw.pop('result')

    result_bw = broyden(
        func,
        z1,
        threshold=2,
        stop_mode='abs',
        name="original",
        eps=1e-6,
        ls=False,
    )
    assert torch.allclose(result_bw['result'], new_z1)
