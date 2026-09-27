import os
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed

from torch.amp.autocast_mode import autocast

from SimpleGeneration.universal_generation_edit.t2i_common import AverageMeter
from SimpleGeneration.universal_generation_edit.losses import MSELoss, SigmaAwareClippedMSELoss
from SimpleGeneration.universal_generation_edit.models.scheduler import FlowMatchingUniformTimestepSampler, FlowMatchingLogitNormalTimestepSampler, FlowMatchingResolutionShiftTimestepSampler


def all_reduce_operation_in_group_for_variables(variables,
                                                operator,
                                                group=None):
    for i in range(len(variables)):
        if not torch.is_tensor(variables[i]):
            variables[i] = torch.tensor(variables[i]).cuda()
        torch.distributed.all_reduce(variables[i], op=operator, group=group)
        variables[i] = variables[i].item()

    return variables


def train_universal_generation_edit_model(train_loader, model, criterion,
                                          optimizer, scheduler, epoch, logger,
                                          config):
    '''
    train universal generation edit model for one epoch
    '''
    losses = AverageMeter()

    # switch to train mode
    model.train()

    if config.use_compile:
        model._orig_mod.module.ae.eval()
    else:
        model.module.ae.eval()

    local_rank = config.local_rank
    if hasattr(config, 'total_rank'):
        total_rank = config.total_rank
    else:
        total_rank = 0

    log_info = f'use_amp: {config.use_amp}, amp_type: {config.amp_type}!'
    logger.info(log_info) if local_rank == 0 and total_rank == 0 else None

    iters = len(train_loader)
    iter_index = 1
    assert config.accumulation_steps >= 1, 'illegal accumulation_steps!'

    for _, data in enumerate(train_loader):
        images, captions = data['image'], data['caption']
        images = images.cuda()

        assert config.task_type in ['T2I', 'TI2I', 'MIX']
        if config.task_type == 'MIX':
            batch_task_type = data['task_type']
        else:
            batch_task_type = config.task_type
        assert batch_task_type in ['T2I', 'TI2I']

        # TI2I only: the VAE-side reference images (normalized to [-1, 1]) and
        # the VLM-side reference PIL images. T2I batches carry neither.
        reference_images, reference_pil_images = None, None
        if batch_task_type == 'TI2I':
            reference_images = [[
                per_reference_image.cuda()
                for per_reference_image in per_sample_reference_images
            ] for per_sample_reference_images in data['reference_image']]
            reference_pil_images = data['reference_pil_image']

        skip_batch_flag = False

        if torch.any(torch.isinf(images)):
            skip_batch_flag = True

        if torch.any(torch.isnan(images)):
            skip_batch_flag = True

        # ---- Tokenize ----
        if batch_task_type == 'T2I':
            pil_images_list = None
        elif batch_task_type == 'TI2I':
            pil_images_list = reference_pil_images

        tokenized = config.tokenizer.encode(prompt_texts=captions,
                                            sample_type=batch_task_type,
                                            pil_images_list=pil_images_list)

        input_ids = tokenized['input_ids'].cuda()
        attention_mask = tokenized['attention_mask'].cuda()
        prompt_start_idx = tokenized['prompt_start_idx'].cuda()
        pixel_values = tokenized['pixel_values']
        image_grid_thw = tokenized['image_grid_thw']
        mm_token_type_ids = tokenized['mm_token_type_ids']

        if pixel_values is not None:
            pixel_values = pixel_values.cuda()
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.cuda()
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids.cuda()

        # ---- Flow matching timesteps ----
        # Every batch of an aspect-ratio bucket run may land on a different
        # resolution, so image_seq_len is recomputed here instead of being read
        # from the config. The 16x comes from the AE 8x downsample plus the 2x2
        # patchify folded into the flux2 latent channels.
        batch_size = images.shape[0]
        image_seq_len = int(images.shape[2] // 16) * int(images.shape[3] // 16)

        train_timestep_sampler = config.train_timestep_sampler
        if isinstance(train_timestep_sampler,
                      FlowMatchingUniformTimestepSampler):
            timesteps = train_timestep_sampler.sample(batch_size=batch_size)
        elif isinstance(train_timestep_sampler,
                        FlowMatchingLogitNormalTimestepSampler):
            timesteps = train_timestep_sampler.sample(batch_size=batch_size)
        elif isinstance(train_timestep_sampler,
                        FlowMatchingResolutionShiftTimestepSampler):
            timesteps = train_timestep_sampler.sample(
                batch_size=batch_size, image_seq_len=image_seq_len)
        timesteps = timesteps.cuda()

        if config.use_amp:
            with autocast(device_type="cuda", dtype=config.amp_type):
                model_pred, target = model(target_image=images,
                                           timesteps=timesteps,
                                           input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           prompt_start_idx=prompt_start_idx,
                                           pixel_values=pixel_values,
                                           image_grid_thw=image_grid_thw,
                                           mm_token_type_ids=mm_token_type_ids,
                                           reference_images=reference_images)

                if isinstance(criterion, SigmaAwareClippedMSELoss):
                    loss_value = criterion(model_pred,
                                           target,
                                           sigmas=timesteps)
                elif isinstance(criterion, MSELoss):
                    loss_value = criterion(model_pred, target)
        else:
            model_pred, target = model(target_image=images,
                                       timesteps=timesteps,
                                       input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       prompt_start_idx=prompt_start_idx,
                                       pixel_values=pixel_values,
                                       image_grid_thw=image_grid_thw,
                                       mm_token_type_ids=mm_token_type_ids,
                                       reference_images=reference_images)

            if isinstance(criterion, SigmaAwareClippedMSELoss):
                loss_value = criterion(model_pred, target, sigmas=timesteps)
            elif isinstance(criterion, MSELoss):
                loss_value = criterion(model_pred, target)

        loss = sum(loss_value.values())

        inf_nan_flag = False
        for key, value in loss_value.items():
            if torch.any(torch.isinf(value)) or torch.any(torch.isnan(value)):
                inf_nan_flag = True

        if torch.any(torch.isinf(loss)) or torch.any(torch.isnan(loss)):
            inf_nan_flag = True

        if loss == 0. or inf_nan_flag:
            print(f'GPU id:{local_rank},zero loss or nan loss or inf loss!')
            skip_batch_flag = True

        loss = loss / config.accumulation_steps
        for key, value in loss_value.items():
            loss_value[key] = value / config.accumulation_steps

        if config.use_amp:
            if iter_index % config.accumulation_steps == 0:
                config.scaler.scale(loss).backward()
            else:
                # not reduce gradient while iter_index % config.accumulation_steps != 0
                with model.no_sync():
                    config.scaler.scale(loss).backward()
        else:
            if iter_index % config.accumulation_steps == 0:
                loss.backward()
            else:
                # not reduce gradient while iter_index % config.accumulation_steps != 0
                with model.no_sync():
                    loss.backward()

        if hasattr(config, 'skip_inf_nan_grad') and config.skip_inf_nan_grad:
            grad_inf_nan_flag = False
            for _, param in model.named_parameters():
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
            operator=torch.distributed.ReduceOp.SUM)

        if skip_batch_flag:
            log_info = f'skip this batch!'
            logger.info(
                log_info) if local_rank == 0 and total_rank == 0 else None
            optimizer.zero_grad()
            continue

        if config.use_amp:
            if iter_index % config.accumulation_steps == 0:
                if (hasattr(config, 'clip_grad_value')
                        and config.clip_grad_value
                        > 0) or (hasattr(config, 'clip_max_norm')
                                 and config.clip_max_norm > 0):
                    config.scaler.unscale_(optimizer)

                    if hasattr(config, 'clip_grad_value'):
                        torch.nn.utils.clip_grad_value_(
                            model.parameters(), config.clip_grad_value)

                    if hasattr(config, 'clip_max_norm'):
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       config.clip_max_norm)

                config.scaler.step(optimizer)
                config.scaler.update()
                optimizer.zero_grad()
        else:
            if iter_index % config.accumulation_steps == 0:
                if (hasattr(config, 'clip_grad_value')
                        and config.clip_grad_value
                        > 0) or (hasattr(config, 'clip_max_norm')
                                 and config.clip_max_norm > 0):

                    if hasattr(config, 'clip_grad_value'):
                        torch.nn.utils.clip_grad_value_(
                            model.parameters(), config.clip_grad_value)

                    if hasattr(config, 'clip_max_norm'):
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       config.clip_max_norm)

                optimizer.step()
                optimizer.zero_grad()

        if iter_index % config.accumulation_steps == 0:
            for key, value in loss_value.items():
                [value] = all_reduce_operation_in_group_for_variables(
                    variables=[value], operator=torch.distributed.ReduceOp.SUM)
                loss_value[key] = value / float(config.gpus_num)

            [loss] = all_reduce_operation_in_group_for_variables(
                variables=[loss], operator=torch.distributed.ReduceOp.SUM)
            loss = loss / float(config.gpus_num)
            losses.update(loss, images.size(0))

        if iter_index % config.accumulation_steps == 0:
            scheduler.step(optimizer, iter_index / iters + (epoch - 1))

        accumulation_iter_index, accumulation_iters = int(
            iter_index // config.accumulation_steps), int(
                iters // config.accumulation_steps)
        if iter_index % int(
                config.print_interval * config.accumulation_steps) == 0:
            log_info = f'train: epoch {epoch:0>4d}, iter [{accumulation_iter_index:0>5d}, {accumulation_iters:0>5d}], lr: {scheduler.current_lr:.6f}, total_loss: {loss*config.accumulation_steps:.4f}, '
            for key, value in loss_value.items():
                log_info += f'{key}: {value*config.accumulation_steps:.4f}, '
            logger.info(
                log_info) if local_rank == 0 and total_rank == 0 else None

        total_accumulation_iters = accumulation_iters * (
            epoch - 1) + accumulation_iter_index
        if hasattr(config,
                   'use_step_save_interval') and config.use_step_save_interval:
            if (iter_index % config.accumulation_steps == 0 and
                    total_accumulation_iters % config.step_save_interval == 0):
                if local_rank == 0 and total_rank == 0:
                    if config.use_compile:
                        save_model = model._orig_mod.module.state_dict()
                    else:
                        save_model = model.module.state_dict()

                    torch.save(
                        save_model,
                        os.path.join(config.checkpoint_dir,
                                     f'step_{total_accumulation_iters}.pth'))

        iter_index += 1

    avg_loss = losses.avg
    avg_loss = avg_loss * config.accumulation_steps

    return avg_loss


def train_universal_generation_edit_model_deepspeed(train_loader, model,
                                                    criterion, optimizer,
                                                    scheduler, epoch, logger,
                                                    config):
    '''
    train universal generation edit model for one epoch using DeepSpeed engine.
    '''
    losses = AverageMeter()

    # switch to train mode
    model.train()

    if config.use_compile:
        model.module._orig_mod.ae.eval()
    else:
        model.module.ae.eval()

    local_rank = config.local_rank
    if hasattr(config, 'total_rank'):
        total_rank = config.total_rank
    else:
        total_rank = 0

    log_info = f'use_amp: {config.use_amp}, amp_type: {config.amp_type}!'
    logger.info(log_info) if local_rank == 0 and total_rank == 0 else None

    iters = len(train_loader)
    iter_index = 1
    assert config.accumulation_steps >= 1, 'illegal accumulation_steps!'
    assert config.accumulation_steps == model.gradient_accumulation_steps()

    for _, data in enumerate(train_loader):
        images, captions = data['image'], data['caption']
        images = images.cuda()

        assert config.task_type in ['T2I', 'TI2I', 'MIX']
        if config.task_type == 'MIX':
            batch_task_type = data['task_type']
        else:
            batch_task_type = config.task_type
        assert batch_task_type in ['T2I', 'TI2I']

        # TI2I only: the VAE-side reference images (normalized to [-1, 1]) and
        # the VLM-side reference PIL images. T2I batches carry neither.
        reference_images, reference_pil_images = None, None
        if batch_task_type == 'TI2I':
            reference_images = [[
                per_reference_image.cuda()
                for per_reference_image in per_sample_reference_images
            ] for per_sample_reference_images in data['reference_image']]
            reference_pil_images = data['reference_pil_image']

        # ---- Tokenize ----
        if batch_task_type == 'T2I':
            pil_images_list = None
        elif batch_task_type == 'TI2I':
            pil_images_list = reference_pil_images

        tokenized = config.tokenizer.encode(prompt_texts=captions,
                                            sample_type=batch_task_type,
                                            pil_images_list=pil_images_list)

        input_ids = tokenized['input_ids'].cuda()
        attention_mask = tokenized['attention_mask'].cuda()
        prompt_start_idx = tokenized['prompt_start_idx'].cuda()
        pixel_values = tokenized['pixel_values']
        image_grid_thw = tokenized['image_grid_thw']
        mm_token_type_ids = tokenized['mm_token_type_ids']

        if pixel_values is not None:
            pixel_values = pixel_values.cuda()
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.cuda()
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids.cuda()

        # ---- Flow matching timesteps ----
        # Every batch of an aspect-ratio bucket run may land on a different
        # resolution, so image_seq_len is recomputed here instead of being read
        # from the config. The 16x comes from the AE 8x downsample plus the 2x2
        # patchify folded into the flux2 latent channels.
        batch_size = images.shape[0]
        image_seq_len = int(images.shape[2] // 16) * int(images.shape[3] // 16)

        train_timestep_sampler = config.train_timestep_sampler
        if isinstance(train_timestep_sampler,
                      FlowMatchingUniformTimestepSampler):
            timesteps = train_timestep_sampler.sample(batch_size=batch_size)
        elif isinstance(train_timestep_sampler,
                        FlowMatchingLogitNormalTimestepSampler):
            timesteps = train_timestep_sampler.sample(batch_size=batch_size)
        elif isinstance(train_timestep_sampler,
                        FlowMatchingResolutionShiftTimestepSampler):
            timesteps = train_timestep_sampler.sample(
                batch_size=batch_size, image_seq_len=image_seq_len)
        timesteps = timesteps.cuda()

        # DeepSpeed engine handles fp16/bf16 casting natively,
        # but criterion is not managed by DeepSpeed, so we use autocast for it
        model_pred, target = model(target_image=images,
                                   timesteps=timesteps,
                                   input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   prompt_start_idx=prompt_start_idx,
                                   pixel_values=pixel_values,
                                   image_grid_thw=image_grid_thw,
                                   mm_token_type_ids=mm_token_type_ids,
                                   reference_images=reference_images)

        if config.use_amp:
            with autocast(device_type="cuda", dtype=config.amp_type):
                if isinstance(criterion, SigmaAwareClippedMSELoss):
                    loss_value = criterion(model_pred,
                                           target,
                                           sigmas=timesteps)
                elif isinstance(criterion, MSELoss):
                    loss_value = criterion(model_pred, target)

        else:
            if isinstance(criterion, SigmaAwareClippedMSELoss):
                loss_value = criterion(model_pred, target, sigmas=timesteps)
            elif isinstance(criterion, MSELoss):
                loss_value = criterion(model_pred, target)

        loss = sum(loss_value.values())

        # DeepSpeed backward (handles loss scaling for fp16 internally)
        model.backward(loss)
        # DeepSpeed step (handles gradient clipping + optimizer step + gradient accumulation internally)
        model.step()

        if iter_index % config.accumulation_steps == 0:
            for key, value in loss_value.items():
                [value] = all_reduce_operation_in_group_for_variables(
                    variables=[value], operator=torch.distributed.ReduceOp.SUM)
                loss_value[key] = value / float(config.gpus_num)

            [loss] = all_reduce_operation_in_group_for_variables(
                variables=[loss], operator=torch.distributed.ReduceOp.SUM)
            loss = loss / float(config.gpus_num)
            losses.update(loss, images.size(0))

        if iter_index % config.accumulation_steps == 0:
            scheduler.step(optimizer, iter_index / iters + (epoch - 1))

        accumulation_iter_index, accumulation_iters = int(
            iter_index // config.accumulation_steps), int(
                iters // config.accumulation_steps)
        if iter_index % int(
                config.print_interval * config.accumulation_steps) == 0:
            log_info = f'train: epoch {epoch:0>4d}, iter [{accumulation_iter_index:0>5d}, {accumulation_iters:0>5d}], lr: {scheduler.current_lr:.6f}, total_loss: {loss:.4f}, '
            for key, value in loss_value.items():
                log_info += f'{key}: {value:.4f}, '
            logger.info(
                log_info) if local_rank == 0 and total_rank == 0 else None

        total_accumulation_iters = accumulation_iters * (
            epoch - 1) + accumulation_iter_index
        if hasattr(config,
                   'use_step_save_interval') and config.use_step_save_interval:
            if (iter_index % config.accumulation_steps == 0 and
                    total_accumulation_iters % config.step_save_interval == 0):
                save_path = os.path.join(
                    config.checkpoint_dir,
                    f'step_{total_accumulation_iters}.pth')

                if config.use_compile:
                    module = model.module._orig_mod
                else:
                    module = model.module

                if config.deepspeed_zero_stage == 3:
                    # Batch gather all parameters at once to reduce
                    # communication rounds under ZeRO-3
                    all_params = list(module.parameters())
                    with deepspeed.zero.GatheredParameters(all_params):
                        if local_rank == 0 and total_rank == 0:
                            state_dict = {
                                k: v.cpu().clone()
                                for k, v in module.state_dict().items()
                            }
                    if local_rank == 0 and total_rank == 0:
                        torch.save(state_dict, save_path)
                else:
                    if local_rank == 0 and total_rank == 0:
                        torch.save(module.state_dict(), save_path)

        iter_index += 1

    avg_loss = losses.avg

    return avg_loss
