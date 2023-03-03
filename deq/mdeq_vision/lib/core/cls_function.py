# Modified based on the HRNet repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

from deq.mdeq_vision.lib.core.cls_evaluate import accuracy
from deq.mdeq_vision.lib.utils.utils import save_checkpoint, AverageMeter


logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, lr_scheduler, epoch,
          output_dir, tb_log_dir, writer_dict=None, topk=(1,5), warm_inits=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    jac_losses = AverageMeter()
    data_aug_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    writer = writer_dict['writer'] if writer_dict else None
    global_steps = writer_dict['train_global_steps']
    update_freq = config.LOSS.JAC_INCREMENTAL

    # switch to train mode
    model.train()

    end = time.time()
    total_batch_num = len(train_loader)
    effec_batch_num = int(config.PERCENT * total_batch_num)
    for i, batch in enumerate(train_loader):
        # train on partial training data
        if i >= effec_batch_num: break

        if warm_inits is None:
            if config.TRAIN.WARM_INIT or config.TRAIN.WARM_INIT_BACK:
                input, target, z1, grad_init, indices = batch
                warm_init_dir = Path(config.TRAIN.WARM_INIT_DIR)
            else:
                input, target = batch
                z1 = None
            init_tensors = None
        else:
            use_broyden_matrices = config.TRAIN.USE_BROYDEN_MATRICES
            input, target, indices = batch
            try:
                if use_broyden_matrices:
                    warm_inits_batch = [[
                        warm_inits[idx.cpu().numpy().item()][i].unsqueeze(0)
                        for idx in indices
                    ] for i in range(3)]
                    nstep = min([
                        warm_inits[idx.cpu().numpy().item()][-1].unsqueeze(0)
                        for idx in indices
                    ])
                else:
                    warm_inits_batch = [
                        warm_inits[idx.cpu().numpy().item()].unsqueeze(0)
                        for idx in indices
                    ]
            except KeyError:
                z1 = None
                init_tensors = None
            else:
                # in z1 we concatenate all the warm inits elements
                if use_broyden_matrices:
                    z1, Us, VTs = [
                        torch.cat(wi_batch, dim=0).to(input)
                        for wi_batch in warm_inits_batch
                    ]
                    init_tensors = [Us, VTs, nstep]
                else:
                    z1 = torch.cat(warm_inits_batch, dim=0).to(input)
                    init_tensors = None

        # measure data loading time
        data_time.update(time.time() - end)

        # compute jacobian loss weight (which is dynamically scheduled)
        deq_steps = global_steps - config.TRAIN.PRETRAIN_STEPS
        if deq_steps < 0:
            # We can also regularize output Jacobian when pretraining
            factor = config.LOSS.PRETRAIN_JAC_LOSS_WEIGHT
        elif epoch >= config.LOSS.JAC_STOP_EPOCH:
            # If are above certain epoch, we may want to stop jacobian regularization training
            # (e.g., when the original loss is 0.01 and jac loss is 0.05, the jacobian regularization
            # will be dominating and hurt performance!)
            factor = 0
        else:
            # Dynamically schedule the Jacobian reguarlization loss weight, if needed
            factor = config.LOSS.JAC_LOSS_WEIGHT + 0.1 * (deq_steps // update_freq)
        compute_jac_loss = (torch.rand([]).item() < config.LOSS.JAC_LOSS_FREQ) and (factor > 0)
        delta_f_thres = torch.randint(-config.DEQ.RAND_F_THRES_DELTA,2,[]).item() if (config.DEQ.RAND_F_THRES_DELTA > 0 and compute_jac_loss) else 0
        f_thres = max(config.DEQ.F_THRES + delta_f_thres, 1)
        if warm_inits is not None and z1 is None:
            f_thres *= 2
        b_thres = config.DEQ.B_THRES
        if config.TRAIN.WARM_INIT_BACK:
            extra_kwargs = dict(
                grad_init=grad_init,
                indices=indices,
            )
        else:
            extra_kwargs = dict()
        output, jac_loss, *others, new_inits = model(
            input,
            train_step=(lr_scheduler._step_count-1),
            compute_jac_loss=compute_jac_loss,
            z1=z1,
            init_tensors=init_tensors,
            f_thres=f_thres,
            b_thres=b_thres,
            writer=writer,
            return_inits=True,
            data_aug_invariance=config.LOSS.DATA_AUG_INVARIANCE,
            **extra_kwargs,
        )
        if config.LOSS.DATA_AUG_INVARIANCE:
            distance_matrix = others[1]
            target = target.reshape(-1)
        start_warm_init_write = time.time()
        if warm_inits is not None and new_inits is not None:
            for i_batch, idx in enumerate(indices):
                if use_broyden_matrices:
                    warm_inits[idx.cpu().numpy().item()] = [
                        ni[i_batch].cpu()
                        for ni in new_inits[:-1]
                    ]
                else:
                    ni = new_inits[0][i_batch].cpu()
                    warm_inits[idx.cpu().numpy().item()] = ni
        elif config.TRAIN.WARM_INIT and new_inits is not None:
            for i_batch, idx in enumerate(indices):
                ni = new_inits[0][i_batch].cpu()
                torch.save(
                    ni,
                    warm_init_dir / f'{idx.cpu().numpy().item()}.pt',
                )
        end_warm_init_write = time.time()
        if i % config.PRINT_FREQ == 0:
            logger.info(f'Warm init write time: {end_warm_init_write - start_warm_init_write}')
        if torch.cuda.is_available():
            target = target.cuda(non_blocking=True)
        loss = criterion(output, target)
        jac_loss = jac_loss.mean()
        # data aug invariance loss
        if config.LOSS.DATA_AUG_INVARIANCE:
            data_aug_weight = config.LOSS.DATA_AUG_INVARIANCE_WEIGHT
            data_aug_loss = distance_matrix.mean()

        # measure accuracy and record loss
        losses.update(loss.item(), target.size(0))
        if compute_jac_loss:
            jac_losses.update(jac_loss.item(), target.size(0))

        if config.LOSS.DATA_AUG_INVARIANCE:
            data_aug_losses.update(data_aug_loss.item(), target.size(0))

        if factor > 0:
            loss += jac_loss * factor
        if config.LOSS.DATA_AUG_INVARIANCE:
            loss += data_aug_loss * data_aug_weight

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        if config.TRAIN.CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP)
        optimizer.step()
        if config.TRAIN.LR_SCHEDULER != 'step':
            lr_scheduler.step()

        prec1, prec5 = accuracy(output, target, topk=topk)
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}] ({3})\t' \
                  'Time {batch_time.avg:.3f}s\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.avg:.3f}s\t' \
                  'Loss {loss.avg:.5f}\t' \
                  'Jac (gamma) {jac_losses.avg:.4f} ({factor:.4f})\t' \
                  'Data aug. loss {data_aug_losses.avg:.4f}\t' \
                  'Acc@1 {top1.avg:.3f}\t'.format(
                      epoch, i, effec_batch_num, global_steps, batch_time=batch_time,
                      speed=input.size(0)/batch_time.avg,
                      data_time=data_time, loss=losses, jac_losses=jac_losses,
                      data_aug_losses=data_aug_losses,
                      factor=factor, top1=top1)
            if 5 in topk:
                msg += 'Acc@5 {top5.avg:.3f}\t'.format(top5=top5)
            logger.info(msg)

        global_steps += 1
        writer_dict['train_global_steps'] = global_steps

        if factor > 0 and global_steps > config.TRAIN.PRETRAIN_STEPS and (deq_steps+1) % update_freq == 0:
             logger.info(f'Note: Adding 0.1 to Jacobian regularization weight.')


def validate(config, val_loader, model, criterion, lr_scheduler, epoch, output_dir, tb_log_dir,
             writer_dict=None, topk=(1,5), spectral_radius_mode=False, warm_inits=None, return_loss=False, return_convergence=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    spectral_radius_mode = spectral_radius_mode and (epoch % 10 == 0)
    if spectral_radius_mode:
        sradiuses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    if return_convergence:
        convergence_rel = AverageMeter()
        convergence_abs = AverageMeter()
    writer = writer_dict['writer'] if writer_dict else None

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        # tk0 = tqdm(enumerate(val_loader), total=len(val_loader), position=0, leave=True)
        total_batch_num = len(val_loader)
        effec_batch_num = int(config.PERCENT * total_batch_num)
        for i, batch in enumerate(val_loader):
            # eval on partial data as well for debugging purposes
            if i >= effec_batch_num and config.PERCENT < 1.0:
                break

            if warm_inits is None:
                input, target = batch
                z1 = None
            else:
                input, target, indices = batch
                warm_inits_batch = [
                    warm_inits[idx.cpu().numpy().item()].unsqueeze(0)
                    for idx in indices
                ]
                # in z1 we concatenate all the warm inits elements
                z1 = torch.cat(warm_inits_batch, dim=0).to(input)

            # compute output
            model_out = model(
                input,
                train_step=(-1 if epoch < 0 else (lr_scheduler._step_count-1)),
                compute_jac_loss=False,
                spectral_radius_mode=spectral_radius_mode,
                writer=writer,
                z1=z1,
                return_result=return_convergence,
            )
            if return_convergence:
                output, _, sradius = model_out
            else:
                output, _, sradius, result_fw = model_out
            if torch.cuda.is_available():
                target = target.cuda(non_blocking=True)
            loss = criterion(output, target)

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            prec1, prec5 = accuracy(output, target, topk=topk)
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))
            if return_convergence:
                convergence_rel.update(result_fw['rel_trace'].item(), input.size(0))
                convergence_abs.update(result_fw['abs_trace'].item(), input.size(0))

            if spectral_radius_mode:
                sradius = sradius.mean()
                sradiuses.update(sradius.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    if spectral_radius_mode:
        logger.info(f"Spectral radius over validation set: {sradiuses.avg}")
    msg = 'Test: Time {batch_time.avg:.3f}\t' \
            'Loss {loss.avg:.4f}\t' \
            'Acc@1 {top1.avg:.3f}\t'.format(
                batch_time=batch_time, loss=losses, top1=top1)
    if 5 in topk:
        msg += 'Acc@5 {top5.avg:.3f}\t'.format(top5=top5)
    logger.info(msg)

    if writer:
        writer.add_scalar('accuracy/valid_top1', top1.avg, epoch)
        if spectral_radius_mode:
            writer.add_scalar('stability/sradius', sradiuses.avg, epoch)

    if return_loss:
        return losses.avg
    else:
        if return_convergence:
            return top1.avg, convergence_rel.avg, convergence_abs.avg
        else
            return top1.avg
