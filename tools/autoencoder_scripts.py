import os
import sys
import warnings
from contextlib import nullcontext

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import numpy as np
from tqdm import tqdm

from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

import torch
import torch.nn as nn
import torch.nn.functional as F

import deepspeed

from torch.amp.autocast_mode import autocast

from SimpleGeneration.flux_autoencoder.common import AverageMeter


def all_reduce_operation_in_group_for_variables(variables, operator, group):
    for i in range(len(variables)):
        if not torch.is_tensor(variables[i]):
            variables[i] = torch.tensor(variables[i]).cuda()
        torch.distributed.all_reduce(variables[i], op=operator, group=group)
        variables[i] = variables[i].item()

    return variables


def train_flux_autoencoder_model(train_loader, generator_model,
                                 discriminator_model, criterion,
                                 generator_optimizer, discriminator_optimizer,
                                 generator_scheduler, discriminator_scheduler,
                                 epoch, logger, config):
    generator_losses = AverageMeter()
    discriminator_losses = AverageMeter()
    total_losses = AverageMeter()

    # switch to train mode
    generator_model.train()
    discriminator_model.train()

    # Extract bare models to support both DDP and torch.compile+DDP
    if config.use_compile:
        bare_generator = generator_model._orig_mod.module
        bare_discriminator = discriminator_model._orig_mod.module
    else:
        bare_generator = generator_model.module
        bare_discriminator = discriminator_model.module

    local_rank = config.local_rank
    if hasattr(config, 'total_rank'):
        total_rank = config.total_rank
    else:
        total_rank = 0

    log_info = f'use_amp: {config.use_amp}, amp_type: {config.amp_type}!'
    logger.info(log_info) if local_rank == 0 and total_rank == 0 else None

    iters = len(train_loader.dataset) // config.batch_size
    iter_index = 1
    assert config.accumulation_steps >= 1, 'illegal accumulation_steps!'

    for _, data in enumerate(train_loader):
        images = data['image']
        images = images.cuda()

        skip_batch_flag = False

        if torch.any(torch.isinf(images)):
            skip_batch_flag = True

        if torch.any(torch.isnan(images)):
            skip_batch_flag = True

        accumulation_iter_index, accumulation_iters = int(
            iter_index // config.accumulation_steps), int(
                iters // config.accumulation_steps)
        total_accumulation_iters = accumulation_iters * (
            epoch - 1) + accumulation_iter_index

        # for generator training
        ########################################################################################
        if (iter_index - 1) % config.accumulation_steps == 0:
            generator_optimizer.zero_grad()

        if config.use_amp:
            with autocast(device_type="cuda", dtype=config.amp_type):
                reconstruction_images, kl_out = bare_generator.forward_train(
                    images)
                loss_dict = criterion(
                    images,
                    reconstruction_images,
                    logvar=bare_generator.logvar,
                    kl_out=kl_out,
                    loss_type='generator_loss',
                    last_layer=bare_generator.get_last_layer())
        else:
            reconstruction_images, kl_out = bare_generator.forward_train(
                images)
            loss_dict = criterion(images,
                                  reconstruction_images,
                                  logvar=bare_generator.logvar,
                                  kl_out=kl_out,
                                  loss_type='generator_loss',
                                  last_layer=bare_generator.get_last_layer())

        generator_loss_dict = {}
        generator_fake = 0.
        for key, value in loss_dict.items():
            if key == 'generator_fake':
                generator_fake = value
            else:
                generator_loss_dict[key] = value

        # before generator_loss_start_step,don't train generator_model
        if total_accumulation_iters < config.generator_loss_start_step:
            for key in generator_loss_dict.keys():
                generator_loss_dict[key] = 0. * generator_loss_dict[key]

        # before discriminator_loss_start_step,don't train discriminator_model
        if total_accumulation_iters < config.discriminator_loss_start_step:
            generator_loss_dict[
                'generator_adversarial_loss'] = 0. * generator_loss_dict[
                    'generator_adversarial_loss']

        generator_loss = 0.
        for key, value in generator_loss_dict.items():
            generator_loss += value

        inf_nan_flag = False
        for key, value in generator_loss_dict.items():
            if torch.any(torch.isinf(value)) or torch.any(torch.isnan(value)):
                inf_nan_flag = True

        if torch.any(torch.isinf(generator_loss)) or torch.any(
                torch.isnan(generator_loss)):
            inf_nan_flag = True

        if inf_nan_flag:
            print(f'GPU id:{local_rank},zero loss or nan loss or inf loss!')
            skip_batch_flag = True

        generator_loss = generator_loss / config.accumulation_steps
        generator_fake = generator_fake / config.accumulation_steps
        for key, value in generator_loss_dict.items():
            generator_loss_dict[key] = value / config.accumulation_steps

        if config.use_amp:
            config.generator_scaler.scale(generator_loss).backward()
        else:
            generator_loss.backward()

        if hasattr(config, 'skip_inf_nan_grad') and config.skip_inf_nan_grad:
            grad_inf_nan_flag = False
            for _, param in generator_model.named_parameters():
                per_weight_grad = param.grad
                if per_weight_grad is not None:
                    if torch.any(torch.isinf(per_weight_grad)) or torch.any(
                            torch.isnan(per_weight_grad)):
                        grad_inf_nan_flag = True
            if grad_inf_nan_flag:
                print(f'GPU id:{local_rank},nan grad or inf grad!')
                skip_batch_flag = True

        [skip_batch_flag] = all_reduce_operation_in_group_for_variables(
            variables=[skip_batch_flag],
            operator=torch.distributed.ReduceOp.SUM,
            group=config.group)

        if skip_batch_flag:
            log_info = f'skip this batch!'
            logger.info(
                log_info) if local_rank == 0 and total_rank == 0 else None
            generator_optimizer.zero_grad()
            continue

        torch.distributed.barrier(device_ids=[local_rank])

        if config.use_amp:
            if iter_index % config.accumulation_steps == 0:
                # 手动 allreduce 梯度
                for param in generator_model.parameters():
                    if param.grad is not None:
                        torch.distributed.all_reduce(
                            param.grad, op=torch.distributed.ReduceOp.AVG)

                if (hasattr(config, 'generator_clip_grad_value')
                        and config.generator_clip_grad_value
                        > 0) or (hasattr(config, 'generator_clip_max_norm')
                                 and config.generator_clip_max_norm > 0):
                    config.generator_scaler.unscale_(generator_optimizer)

                    if hasattr(config, 'generator_clip_grad_value'):
                        torch.nn.utils.clip_grad_value_(
                            generator_model.parameters(),
                            config.generator_clip_grad_value)

                    if hasattr(config, 'generator_clip_max_norm'):
                        torch.nn.utils.clip_grad_norm_(
                            generator_model.parameters(),
                            config.generator_clip_max_norm)

                config.generator_scaler.step(generator_optimizer)
                config.generator_scaler.update()
        else:
            if iter_index % config.accumulation_steps == 0:
                # 手动 allreduce 梯度
                for param in generator_model.parameters():
                    if param.grad is not None:
                        torch.distributed.all_reduce(
                            param.grad, op=torch.distributed.ReduceOp.AVG)

                if (hasattr(config, 'generator_clip_grad_value')
                        and config.generator_clip_grad_value
                        > 0) or (hasattr(config, 'generator_clip_max_norm')
                                 and config.generator_clip_max_norm > 0):

                    if hasattr(config, 'generator_clip_grad_value'):
                        torch.nn.utils.clip_grad_value_(
                            generator_model.parameters(),
                            config.generator_clip_grad_value)

                    if hasattr(config, 'generator_clip_max_norm'):
                        torch.nn.utils.clip_grad_norm_(
                            generator_model.parameters(),
                            config.generator_clip_max_norm)

                generator_optimizer.step()

        ########################################################################################
        # for discriminator training
        if (iter_index - 1) % config.accumulation_steps == 0:
            discriminator_optimizer.zero_grad()

        if config.use_amp:
            with autocast(device_type="cuda", dtype=config.amp_type):
                loss_dict = criterion(images,
                                      reconstruction_images,
                                      loss_type='discriminator_loss')

        else:
            loss_dict = criterion(images,
                                  reconstruction_images,
                                  loss_type='discriminator_loss')

        discriminator_loss_dict = {}
        discriminator_real = 0.
        discriminator_fake = 0.
        for key, value in loss_dict.items():
            if key == 'discriminator_real':
                discriminator_real = value
            elif key == 'discriminator_fake':
                discriminator_fake = value
            else:
                discriminator_loss_dict[key] = value

        # before discriminator_loss_start_step,don't train discriminator_model
        if total_accumulation_iters < config.discriminator_loss_start_step:
            discriminator_loss_dict[
                'discriminator_adversarial_loss'] = 0. * discriminator_loss_dict[
                    'discriminator_adversarial_loss']

        discriminator_loss = 0.
        for key, value in discriminator_loss_dict.items():
            discriminator_loss += value

        inf_nan_flag = False
        for key, value in discriminator_loss_dict.items():
            if torch.any(torch.isinf(value)) or torch.any(torch.isnan(value)):
                inf_nan_flag = True

        if torch.any(torch.isinf(discriminator_loss)) or torch.any(
                torch.isnan(discriminator_loss)):
            inf_nan_flag = True

        if inf_nan_flag:
            print(f'GPU id:{local_rank},zero loss or nan loss or inf loss!')
            skip_batch_flag = True

        discriminator_loss = discriminator_loss / config.accumulation_steps
        discriminator_real = discriminator_real / config.accumulation_steps
        discriminator_fake = discriminator_fake / config.accumulation_steps
        for key, value in discriminator_loss_dict.items():
            discriminator_loss_dict[key] = value / config.accumulation_steps

        if config.use_amp:
            config.discriminator_scaler.scale(discriminator_loss).backward()
        else:
            discriminator_loss.backward()

        if hasattr(config, 'skip_inf_nan_grad') and config.skip_inf_nan_grad:
            grad_inf_nan_flag = False
            for _, param in discriminator_model.named_parameters():
                per_weight_grad = param.grad
                if per_weight_grad is not None:
                    if torch.any(torch.isinf(per_weight_grad)) or torch.any(
                            torch.isnan(per_weight_grad)):
                        grad_inf_nan_flag = True
            if grad_inf_nan_flag:
                print(f'GPU id:{local_rank},nan grad or inf grad!')
                skip_batch_flag = True

        [skip_batch_flag] = all_reduce_operation_in_group_for_variables(
            variables=[skip_batch_flag],
            operator=torch.distributed.ReduceOp.SUM,
            group=config.group)

        if skip_batch_flag:
            log_info = f'skip this batch!'
            logger.info(
                log_info) if local_rank == 0 and total_rank == 0 else None
            discriminator_optimizer.zero_grad()
            continue

        torch.distributed.barrier(device_ids=[local_rank])

        if config.use_amp:
            if iter_index % config.accumulation_steps == 0:
                # 手动 allreduce 梯度
                for param in discriminator_model.parameters():
                    if param.grad is not None:
                        torch.distributed.all_reduce(
                            param.grad, op=torch.distributed.ReduceOp.AVG)

                if (hasattr(config, 'discriminator_clip_grad_value')
                        and config.discriminator_clip_grad_value
                        > 0) or (hasattr(config, 'discriminator_clip_max_norm')
                                 and config.discriminator_clip_max_norm > 0):
                    config.discriminator_scaler.unscale_(
                        discriminator_optimizer)

                    if hasattr(config, 'discriminator_clip_grad_value'):
                        torch.nn.utils.clip_grad_value_(
                            discriminator_model.parameters(),
                            config.discriminator_clip_grad_value)

                    if hasattr(config, 'discriminator_clip_max_norm'):
                        torch.nn.utils.clip_grad_norm_(
                            discriminator_model.parameters(),
                            config.discriminator_clip_max_norm)

                config.discriminator_scaler.step(discriminator_optimizer)
                config.discriminator_scaler.update()
        else:
            if iter_index % config.accumulation_steps == 0:
                # 手动 allreduce 梯度
                for param in discriminator_model.parameters():
                    if param.grad is not None:
                        torch.distributed.all_reduce(
                            param.grad, op=torch.distributed.ReduceOp.AVG)

                if (hasattr(config, 'discriminator_clip_grad_value')
                        and config.discriminator_clip_grad_value
                        > 0) or (hasattr(config, 'discriminator_clip_max_norm')
                                 and config.discriminator_clip_max_norm > 0):

                    if hasattr(config, 'discriminator_clip_grad_value'):
                        torch.nn.utils.clip_grad_value_(
                            discriminator_model.parameters(),
                            config.discriminator_clip_grad_value)

                    if hasattr(config, 'discriminator_clip_max_norm'):
                        torch.nn.utils.clip_grad_norm_(
                            discriminator_model.parameters(),
                            config.discriminator_clip_max_norm)

                discriminator_optimizer.step()
        #########################################################################

        if iter_index % config.accumulation_steps == 0:
            for key, value in generator_loss_dict.items():
                [value] = all_reduce_operation_in_group_for_variables(
                    variables=[value],
                    operator=torch.distributed.ReduceOp.SUM,
                    group=config.group)
                generator_loss_dict[key] = value / float(config.gpus_num)

            for key, value in discriminator_loss_dict.items():
                [value] = all_reduce_operation_in_group_for_variables(
                    variables=[value],
                    operator=torch.distributed.ReduceOp.SUM,
                    group=config.group)
                discriminator_loss_dict[key] = value / float(config.gpus_num)

            generator_fake = generator_fake.detach().mean()
            [generator_fake] = all_reduce_operation_in_group_for_variables(
                variables=[generator_fake],
                operator=torch.distributed.ReduceOp.SUM,
                group=config.group)
            generator_fake = generator_fake / float(config.gpus_num)

            discriminator_real = discriminator_real.detach().mean()
            [discriminator_real] = all_reduce_operation_in_group_for_variables(
                variables=[discriminator_real],
                operator=torch.distributed.ReduceOp.SUM,
                group=config.group)
            discriminator_real = discriminator_real / float(config.gpus_num)

            discriminator_fake = discriminator_fake.detach().mean()
            [discriminator_fake] = all_reduce_operation_in_group_for_variables(
                variables=[discriminator_fake],
                operator=torch.distributed.ReduceOp.SUM,
                group=config.group)
            discriminator_fake = discriminator_fake / float(config.gpus_num)

            [generator_loss] = all_reduce_operation_in_group_for_variables(
                variables=[generator_loss],
                operator=torch.distributed.ReduceOp.SUM,
                group=config.group)
            generator_loss = generator_loss / float(config.gpus_num)
            generator_losses.update(generator_loss, images.size(0))

            [discriminator_loss] = all_reduce_operation_in_group_for_variables(
                variables=[discriminator_loss],
                operator=torch.distributed.ReduceOp.SUM,
                group=config.group)
            discriminator_loss = discriminator_loss / float(config.gpus_num)
            discriminator_losses.update(discriminator_loss, images.size(0))

            total_loss = generator_loss + discriminator_loss
            total_losses.update(total_loss, images.size(0))

        if iter_index % config.accumulation_steps == 0:
            generator_scheduler.step(generator_optimizer,
                                     iter_index / iters + (epoch - 1))
            discriminator_scheduler.step(discriminator_optimizer,
                                         iter_index / iters + (epoch - 1))

        accumulation_iter_index, accumulation_iters = int(
            iter_index // config.accumulation_steps), int(
                iters // config.accumulation_steps)
        if iter_index % int(
                config.print_interval * config.accumulation_steps) == 0:
            log_info = f'train: epoch {epoch:0>4d}, iter [{accumulation_iter_index:0>5d}, {accumulation_iters:0>5d}], generator_lr: {generator_scheduler.current_lr:.6f}, discriminator_lr: {discriminator_scheduler.current_lr:.6f}, \n'
            log_info += f'generator_loss: {generator_loss*config.accumulation_steps:.4f}, discriminator_loss: {discriminator_loss*config.accumulation_steps:.4f}, total_loss: {total_loss*config.accumulation_steps:.4f}\n'
            log_info += f'generator_fake: {generator_fake*config.accumulation_steps:.4f}, discriminator_real: {discriminator_real*config.accumulation_steps:.4f}, discriminator_fake: {discriminator_fake*config.accumulation_steps:.4f}, \n'
            for key, value in generator_loss_dict.items():
                log_info += f'{key}: {value*config.accumulation_steps:.8f}, '
            log_info += f'\n'
            for key, value in discriminator_loss_dict.items():
                log_info += f'{key}: {value*config.accumulation_steps:.8f}, '
            logger.info(
                log_info) if local_rank == 0 and total_rank == 0 else None

        total_accumulation_iters = accumulation_iters * (
            epoch - 1) + accumulation_iter_index
        if hasattr(config,
                   'use_step_save_interval') and config.use_step_save_interval:
            if total_accumulation_iters % config.step_save_interval == 0:
                if local_rank == 0 and total_rank == 0:
                    if config.use_compile:
                        save_generator_model = generator_model._orig_mod.module.state_dict(
                        )
                        save_discriminator_model = discriminator_model._orig_mod.module.state_dict(
                        )
                    else:
                        save_generator_model = generator_model.module.state_dict(
                        )
                        save_discriminator_model = discriminator_model.module.state_dict(
                        )

                    torch.save(
                        save_generator_model,
                        os.path.join(
                            config.checkpoint_dir,
                            f'step_{total_accumulation_iters}_generator_model.pth'
                        ))
                    torch.save(
                        save_discriminator_model,
                        os.path.join(
                            config.checkpoint_dir,
                            f'step_{total_accumulation_iters}_discriminator_model.pth'
                        ))

        iter_index += 1

    avg_generator_loss = generator_losses.avg
    avg_generator_loss = avg_generator_loss * config.accumulation_steps

    avg_discriminator_loss = discriminator_losses.avg
    avg_discriminator_loss = avg_discriminator_loss * config.accumulation_steps

    total_loss = total_losses.avg
    total_loss = total_loss * config.accumulation_steps

    return avg_generator_loss, avg_discriminator_loss, total_loss


def train_flux_autoencoder_model_deepspeed(
        train_loader, generator_model, discriminator_model, criterion,
        generator_optimizer, discriminator_optimizer, generator_scheduler,
        discriminator_scheduler, epoch, logger, config):
    '''
    train flux autoencoder model for one epoch using DeepSpeed engine.
    generator_model: DeepSpeed engine for generator
    discriminator_model: DeepSpeed engine for discriminator
    '''
    generator_losses = AverageMeter()
    discriminator_losses = AverageMeter()
    total_losses = AverageMeter()

    # switch to train mode
    generator_model.train()
    discriminator_model.train()

    # Extract bare models: DeepSpeed engine wraps model in .module
    if config.use_compile:
        bare_generator = generator_model.module._orig_mod
        bare_discriminator = discriminator_model.module._orig_mod
    else:
        bare_generator = generator_model.module
        bare_discriminator = discriminator_model.module

    local_rank = config.local_rank
    if hasattr(config, 'total_rank'):
        total_rank = config.total_rank
    else:
        total_rank = 0

    log_info = f'use_amp: {config.use_amp}, amp_type: {config.amp_type}!'
    logger.info(log_info) if local_rank == 0 and total_rank == 0 else None

    iters = len(train_loader.dataset) // config.batch_size
    iter_index = 1
    assert config.accumulation_steps >= 1, 'illegal accumulation_steps!'

    for _, data in enumerate(train_loader):
        images = data['image']
        images = images.cuda()

        accumulation_iter_index, accumulation_iters = int(
            iter_index // config.accumulation_steps), int(
                iters // config.accumulation_steps)
        total_accumulation_iters = accumulation_iters * (
            epoch - 1) + accumulation_iter_index

        # for generator training
        ########################################################################################
        # Build GatheredParameters context for ZeRO Stage 3: logvar and
        # decoder.conv_out.weight are accessed after forward_train completes,
        # at which point ZeRO-3 has released them back to local shards.
        if config.deepspeed_zero_stage == 3:
            gather_ctx = deepspeed.zero.GatheredParameters(
                [bare_generator.logvar,
                 bare_generator.get_last_layer()])
        else:
            gather_ctx = nullcontext()

        if config.use_amp:
            with autocast(device_type="cuda", dtype=config.amp_type):
                reconstruction_images, kl_out = bare_generator.forward_train(
                    images)
                with gather_ctx:
                    loss_dict = criterion(
                        images,
                        reconstruction_images,
                        logvar=bare_generator.logvar,
                        kl_out=kl_out,
                        loss_type='generator_loss',
                        last_layer=bare_generator.get_last_layer())
        else:
            reconstruction_images, kl_out = bare_generator.forward_train(
                images)
            with gather_ctx:
                loss_dict = criterion(
                    images,
                    reconstruction_images,
                    logvar=bare_generator.logvar,
                    kl_out=kl_out,
                    loss_type='generator_loss',
                    last_layer=bare_generator.get_last_layer())

        generator_loss_dict = {}
        generator_fake = 0.
        for key, value in loss_dict.items():
            if key == 'generator_fake':
                generator_fake = value
            else:
                generator_loss_dict[key] = value

        # before generator_loss_start_step,don't train generator_model
        if total_accumulation_iters < config.generator_loss_start_step:
            for key in generator_loss_dict.keys():
                generator_loss_dict[key] = 0. * generator_loss_dict[key]

        # before discriminator_loss_start_step,don't train discriminator_model
        if total_accumulation_iters < config.discriminator_loss_start_step:
            generator_loss_dict[
                'generator_adversarial_loss'] = 0. * generator_loss_dict[
                    'generator_adversarial_loss']

        generator_loss = 0.
        for key, value in generator_loss_dict.items():
            generator_loss += value

        # Freeze discriminator params before generator backward to prevent
        # DeepSpeed ZeRO Stage 2/3 backward hooks from triggering on
        # discriminator parameters (gradients still flow through discriminator
        # ops to generator via autograd computation graph).
        for param in discriminator_model.module.parameters():
            param.requires_grad = False

        # DeepSpeed backward and step for generator
        generator_model.backward(generator_loss)
        generator_model.step()

        # Unfreeze discriminator params for discriminator training
        for param in discriminator_model.module.parameters():
            param.requires_grad = True

        ########################################################################################
        # for discriminator training
        if config.use_amp:
            with autocast(device_type="cuda", dtype=config.amp_type):
                loss_dict = criterion(images,
                                      reconstruction_images,
                                      loss_type='discriminator_loss')
        else:
            loss_dict = criterion(images,
                                  reconstruction_images,
                                  loss_type='discriminator_loss')

        discriminator_loss_dict = {}
        discriminator_real = 0.
        discriminator_fake = 0.
        for key, value in loss_dict.items():
            if key == 'discriminator_real':
                discriminator_real = value
            elif key == 'discriminator_fake':
                discriminator_fake = value
            else:
                discriminator_loss_dict[key] = value

        # before discriminator_loss_start_step,don't train discriminator_model
        if total_accumulation_iters < config.discriminator_loss_start_step:
            discriminator_loss_dict[
                'discriminator_adversarial_loss'] = 0. * discriminator_loss_dict[
                    'discriminator_adversarial_loss']

        discriminator_loss = 0.
        for key, value in discriminator_loss_dict.items():
            discriminator_loss += value

        # DeepSpeed backward and step for discriminator
        discriminator_model.backward(discriminator_loss)
        discriminator_model.step()

        #########################################################################

        if iter_index % config.accumulation_steps == 0:
            for key, value in generator_loss_dict.items():
                [value] = all_reduce_operation_in_group_for_variables(
                    variables=[value],
                    operator=torch.distributed.ReduceOp.SUM,
                    group=config.group)
                generator_loss_dict[key] = value / float(config.gpus_num)

            for key, value in discriminator_loss_dict.items():
                [value] = all_reduce_operation_in_group_for_variables(
                    variables=[value],
                    operator=torch.distributed.ReduceOp.SUM,
                    group=config.group)
                discriminator_loss_dict[key] = value / float(config.gpus_num)

            generator_fake = generator_fake.detach().mean()
            [generator_fake] = all_reduce_operation_in_group_for_variables(
                variables=[generator_fake],
                operator=torch.distributed.ReduceOp.SUM,
                group=config.group)
            generator_fake = generator_fake / float(config.gpus_num)

            discriminator_real = discriminator_real.detach().mean()
            [discriminator_real] = all_reduce_operation_in_group_for_variables(
                variables=[discriminator_real],
                operator=torch.distributed.ReduceOp.SUM,
                group=config.group)
            discriminator_real = discriminator_real / float(config.gpus_num)

            discriminator_fake = discriminator_fake.detach().mean()
            [discriminator_fake] = all_reduce_operation_in_group_for_variables(
                variables=[discriminator_fake],
                operator=torch.distributed.ReduceOp.SUM,
                group=config.group)
            discriminator_fake = discriminator_fake / float(config.gpus_num)

            [generator_loss] = all_reduce_operation_in_group_for_variables(
                variables=[generator_loss],
                operator=torch.distributed.ReduceOp.SUM,
                group=config.group)
            generator_loss = generator_loss / float(config.gpus_num)
            generator_losses.update(generator_loss, images.size(0))

            [discriminator_loss] = all_reduce_operation_in_group_for_variables(
                variables=[discriminator_loss],
                operator=torch.distributed.ReduceOp.SUM,
                group=config.group)
            discriminator_loss = discriminator_loss / float(config.gpus_num)
            discriminator_losses.update(discriminator_loss, images.size(0))

            total_loss = generator_loss + discriminator_loss
            total_losses.update(total_loss, images.size(0))

        if iter_index % config.accumulation_steps == 0:
            generator_scheduler.step(generator_optimizer,
                                     iter_index / iters + (epoch - 1))
            discriminator_scheduler.step(discriminator_optimizer,
                                         iter_index / iters + (epoch - 1))

        accumulation_iter_index, accumulation_iters = int(
            iter_index // config.accumulation_steps), int(
                iters // config.accumulation_steps)
        if iter_index % int(
                config.print_interval * config.accumulation_steps) == 0:
            log_info = f'train: epoch {epoch:0>4d}, iter [{accumulation_iter_index:0>5d}, {accumulation_iters:0>5d}], generator_lr: {generator_scheduler.current_lr:.6f}, discriminator_lr: {discriminator_scheduler.current_lr:.6f}, \n'
            log_info += f'generator_loss: {generator_loss:.4f}, discriminator_loss: {discriminator_loss:.4f}, total_loss: {total_loss:.4f}\n'
            log_info += f'generator_fake: {generator_fake:.4f}, discriminator_real: {discriminator_real:.4f}, discriminator_fake: {discriminator_fake:.4f}, \n'
            for key, value in generator_loss_dict.items():
                log_info += f'{key}: {value:.8f}, '
            log_info += f'\n'
            for key, value in discriminator_loss_dict.items():
                log_info += f'{key}: {value:.8f}, '
            logger.info(
                log_info) if local_rank == 0 and total_rank == 0 else None

        total_accumulation_iters = accumulation_iters * (
            epoch - 1) + accumulation_iter_index
        if hasattr(config,
                   'use_step_save_interval') and config.use_step_save_interval:
            if total_accumulation_iters % config.step_save_interval == 0:
                if config.use_compile:
                    generator_module = generator_model.module._orig_mod
                    discriminator_module = discriminator_model.module._orig_mod
                else:
                    generator_module = generator_model.module
                    discriminator_module = discriminator_model.module

                generator_save_path = os.path.join(
                    config.checkpoint_dir,
                    f'step_{total_accumulation_iters}_generator_model.pth')
                discriminator_save_path = os.path.join(
                    config.checkpoint_dir,
                    f'step_{total_accumulation_iters}_discriminator_model.pth')

                # Generator: respect ZeRO stage for parameter gathering
                if config.deepspeed_zero_stage == 3:
                    generator_state_dict = {}
                    for name, param in generator_module.named_parameters():
                        with deepspeed.zero.GatheredParameters(param):
                            if local_rank == 0 and total_rank == 0:
                                generator_state_dict[name] = param.data.cpu(
                                ).clone()
                    for name, buf in generator_module.named_buffers():
                        if local_rank == 0 and total_rank == 0:
                            generator_state_dict[name] = buf.cpu().clone()
                    if local_rank == 0 and total_rank == 0:
                        torch.save(generator_state_dict, generator_save_path)
                else:
                    if local_rank == 0 and total_rank == 0:
                        torch.save(generator_module.state_dict(),
                                   generator_save_path)

                # Discriminator: always ZeRO Stage 0, no parameter partitioning
                if local_rank == 0 and total_rank == 0:
                    torch.save(discriminator_module.state_dict(),
                               discriminator_save_path)

        iter_index += 1

    avg_generator_loss = generator_losses.avg
    avg_discriminator_loss = discriminator_losses.avg
    total_loss = total_losses.avg

    return avg_generator_loss, avg_discriminator_loss, total_loss


def test_flux_autoencoder_model(test_loader, generator_model, config):
    # switch to evaluate mode
    generator_model.eval()

    mae, psnr, ssim = [], [], []
    with torch.no_grad():
        model_on_cuda = next(generator_model.parameters()).is_cuda
        for _, data in tqdm(enumerate(test_loader)):
            images = data['image']
            if model_on_cuda:
                images = images.cuda()

            sizes = data['size']

            if torch.any(torch.isinf(images)):
                continue

            if torch.any(torch.isnan(images)):
                continue

            torch.cuda.synchronize()

            if config.use_amp:
                with autocast(device_type="cuda", dtype=config.amp_type):
                    outputs = generator_model(images)
            else:
                outputs = generator_model(images)

            torch.cuda.synchronize()

            images = images.to(torch.float32)
            outputs = outputs.to(torch.float32)

            for per_image, per_output, per_image_size in zip(
                    images, outputs, sizes):
                per_image = per_image.permute(1, 2, 0)
                per_image = per_image[:int(per_image_size[0]
                                           ), :int(per_image_size[1])]

                # per_image value is between [0, 1]
                per_image = per_image * 0.5 + 0.5
                per_image = per_image.cpu().numpy()

                per_output = per_output.permute(1, 2, 0)
                per_output = per_output[:int(per_image_size[0]
                                             ), :int(per_image_size[1])]

                # per_image value is between [0, 1]
                per_output = (per_output * 0.5 + 0.5)
                per_output = torch.clamp(per_output, min=0., max=1.0)
                per_output = per_output.cpu().numpy()

                per_pair_mae = np.sum(
                    np.abs(
                        per_image.astype(np.float32) -
                        per_output.astype(np.float32))) / np.sum(
                            per_image.astype(np.float32) +
                            per_output.astype(np.float32))
                per_pair_psnr = compare_psnr(per_image,
                                             per_output,
                                             data_range=1.0)
                per_pair_ssim = compare_ssim(per_image,
                                             per_output,
                                             data_range=1.0,
                                             channel_axis=-1)

                mae.append(per_pair_mae)
                psnr.append(per_pair_psnr)
                ssim.append(per_pair_ssim)

    mae = sum(mae) / len(mae)
    psnr = sum(psnr) / len(psnr)
    ssim = sum(ssim) / len(ssim)

    result_dict = {
        'mae': mae,
        'psnr': psnr,
        'ssim': ssim,
    }

    return result_dict
