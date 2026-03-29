import os
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import argparse
import functools
import re
import time

import torch
import deepspeed
from torch.utils.data import DataLoader

from tools.image_tokenizer_scripts import train_fsq_model_deepspeed
from tools.utils import (set_seed, get_logger, worker_seed_init_fn, Scheduler)


def build_param_groups(config, model):
    """Build parameter groups for optimizer (weight decay differentiation).
    Returns (model_params_weight_decay_list, model_layer_weight_decay_list).
    Does NOT create the optimizer — DeepSpeed will create it from ds_config.
    """
    optimizer_name = config.optimizer[0]
    optimizer_parameters = config.optimizer[1]
    assert optimizer_name in ['SGD', 'AdamW', 'Muon'], 'Unsupported optimizer!'

    lr = optimizer_parameters['lr']
    weight_decay = optimizer_parameters['weight_decay']

    # For Muon, DeepSpeed native implementation handles muon/adamw param
    # split internally (>=2D params use Muon, others use AdamW fallback).
    # We only need to pass all trainable parameters and build logging info.
    if optimizer_name == 'Muon':
        exclude_muon_layer_name_list = [
            'position_encoding',
            'cls_token',
            'patch_embedding',
        ]
        if 'exclude_muon_layer_name_list' in optimizer_parameters.keys(
        ) and isinstance(optimizer_parameters['exclude_muon_layer_name_list'],
                         list):
            exclude_muon_layer_name_list = (
                exclude_muon_layer_name_list +
                optimizer_parameters['exclude_muon_layer_name_list'])

        muon_param_names = []
        adamw_param_names = []
        all_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            all_params.append(param)
            use_muon = (
                param.ndim >= 2
                and not any(exclude_name in name
                            for exclude_name in exclude_muon_layer_name_list))
            param.use_muon = use_muon
            if use_muon:
                muon_param_names.append(name)
            else:
                adamw_param_names.append(name)

        model_params_weight_decay_list = all_params

        model_layer_weight_decay_list = []
        if muon_param_names:
            model_layer_weight_decay_list.append({
                'name': muon_param_names,
                'optimizer': 'Muon',
                'lr': lr,
                'weight_decay': weight_decay,
            })
        if adamw_param_names:
            model_layer_weight_decay_list.append({
                'name': adamw_param_names,
                'optimizer': 'AdamW',
                'lr': lr,
                'weight_decay': weight_decay,
            })

        return model_params_weight_decay_list, model_layer_weight_decay_list

    # For SGD/AdamW, handle per-layer weight decay and lr differentiation.
    global_weight_decay = True if 'global_weight_decay' not in optimizer_parameters.keys(
    ) else optimizer_parameters['global_weight_decay']

    no_weight_decay_layer_name_list = []
    if 'no_weight_decay_layer_name_list' in optimizer_parameters.keys(
    ) and isinstance(optimizer_parameters['no_weight_decay_layer_name_list'],
                     list):
        no_weight_decay_layer_name_list = optimizer_parameters[
            'no_weight_decay_layer_name_list']

    # training trick only for VIT
    if 'lr_layer_decay' in optimizer_parameters.keys(
    ) and 'lr_layer_decay_block' in optimizer_parameters.keys(
    ) and 'block_name' in optimizer_parameters.keys():
        lr_layer_decay = optimizer_parameters['lr_layer_decay']
        lr_layer_decay_block = optimizer_parameters['lr_layer_decay_block']
        block_name = optimizer_parameters['block_name']

        num_layers = len(lr_layer_decay_block) + 1
        lr_layer_scales = list(lr_layer_decay**(num_layers - i)
                               for i in range(num_layers + 1))

        layer_scale_id_0_name_list = [
            'position_encoding',
            'cls_token',
            'patch_embedding',
        ]

        param_layer_name_list = []
        param_layer_weight_dict = {}
        param_layer_decay_dict, param_layer_lr_dict = {}, {}
        param_layer_lr_scale_dict = {}

        not_group_layer_name_list = []
        not_group_layer_weight_dict = {}
        not_group_layer_decay_dict, not_group_layer_lr_dict = {}, {}

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            in_not_group_layer = False
            if block_name in name:
                not_group_layer_name_list.append(name)
                not_group_layer_weight_dict[name] = param
                in_not_group_layer = True
            else:
                param_layer_name_list.append(name)
                param_layer_weight_dict[name] = param

            if in_not_group_layer is False:
                if any(per_layer_scale_id_0_name in name
                       for per_layer_scale_id_0_name in
                       layer_scale_id_0_name_list):
                    param_layer_lr_scale_dict[name] = lr_layer_scales[0]
                else:
                    param_layer_lr_scale_dict[name] = 1.

            if global_weight_decay is False:
                if param.ndim == 1 or any(no_weight_decay_layer_name in name
                                          for no_weight_decay_layer_name in
                                          no_weight_decay_layer_name_list):
                    if in_not_group_layer:
                        not_group_layer_decay_dict[name] = 0.
                    else:
                        param_layer_decay_dict[name] = 0.
                else:
                    per_layer_weight_decay = weight_decay
                    if 'sub_layer_weight_decay' in optimizer_parameters.keys(
                    ) and isinstance(
                            optimizer_parameters['sub_layer_weight_decay'],
                            dict):
                        for per_sub_layer_name_prefix, per_sub_layer_weight_decay in optimizer_parameters[
                                'sub_layer_weight_decay'].items():
                            if per_sub_layer_name_prefix in name:
                                per_layer_weight_decay = per_sub_layer_weight_decay
                                break

                    if in_not_group_layer:
                        not_group_layer_decay_dict[
                            name] = per_layer_weight_decay
                    else:
                        param_layer_decay_dict[name] = per_layer_weight_decay
            else:
                if in_not_group_layer:
                    not_group_layer_decay_dict[name] = weight_decay
                else:
                    param_layer_decay_dict[name] = weight_decay

            per_layer_lr = lr
            if 'sub_layer_lr' in optimizer_parameters.keys() and isinstance(
                    optimizer_parameters['sub_layer_lr'], dict):
                for per_sub_layer_name_prefix, per_sub_layer_lr in optimizer_parameters[
                        'sub_layer_lr'].items():
                    if per_sub_layer_name_prefix in name:
                        per_layer_lr = per_sub_layer_lr
                        break
            if in_not_group_layer:
                not_group_layer_lr_dict[name] = per_layer_lr
            else:
                param_layer_lr_dict[name] = per_layer_lr

        assert len(param_layer_name_list) == len(
            param_layer_weight_dict) == len(param_layer_decay_dict) == len(
                param_layer_lr_dict) == len(param_layer_lr_scale_dict)

        assert len(not_group_layer_name_list) == len(
            not_group_layer_weight_dict) == len(
                not_group_layer_decay_dict) == len(not_group_layer_lr_dict)

        per_group_weight_nums = len(not_group_layer_name_list) // len(
            lr_layer_decay_block)
        for layer_id in range(0, len(lr_layer_decay_block)):
            for per_group_id in range(per_group_weight_nums):
                per_group_layer_names = not_group_layer_name_list[
                    layer_id * per_group_weight_nums + per_group_id]

                if not isinstance(per_group_layer_names, list):
                    per_layer_name = per_group_layer_names
                    param_layer_name_list.append(per_layer_name)
                    param_layer_weight_dict[
                        per_layer_name] = not_group_layer_weight_dict[
                            per_layer_name]
                    param_layer_decay_dict[
                        per_layer_name] = not_group_layer_decay_dict[
                            per_layer_name]
                    param_layer_lr_dict[
                        per_layer_name] = not_group_layer_lr_dict[
                            per_layer_name]
                    param_layer_lr_scale_dict[
                        per_layer_name] = lr_layer_scales[layer_id + 1]
                else:
                    for per_layer_name in per_group_layer_names:
                        param_layer_name_list.append(per_layer_name)
                        param_layer_weight_dict[
                            per_layer_name] = not_group_layer_weight_dict[
                                per_layer_name]
                        param_layer_decay_dict[
                            per_layer_name] = not_group_layer_decay_dict[
                                per_layer_name]
                        param_layer_lr_dict[
                            per_layer_name] = not_group_layer_lr_dict[
                                per_layer_name]
                        param_layer_lr_scale_dict[
                            per_layer_name] = lr_layer_scales[layer_id + 1]

        assert len(param_layer_name_list) == len(
            param_layer_weight_dict) == len(param_layer_decay_dict) == len(
                param_layer_lr_dict) == len(param_layer_lr_scale_dict)

        unique_decays = list(set(param_layer_decay_dict.values()))
        unique_lrs = list(set(param_layer_lr_dict.values()))
        unique_lr_scales = list(set(param_layer_lr_scale_dict.values()))

        lr_weight_decay_combination = []
        for per_decay in unique_decays:
            for per_lr in unique_lrs:
                for per_lr_scale in unique_lr_scales:
                    lr_weight_decay_combination.append(
                        [per_decay, per_lr, per_lr_scale])

        model_params_weight_decay_list = []
        model_layer_weight_decay_list = []
        for per_decay, per_lr, per_lr_scale in lr_weight_decay_combination:
            per_decay_lr_lrscale_param_list, per_decay_lr_lrscale_name_list = [], []
            for per_layer_name in param_layer_name_list:
                per_layer_weight = param_layer_weight_dict[per_layer_name]
                per_layer_weight_decay = param_layer_decay_dict[per_layer_name]
                per_layer_lr = param_layer_lr_dict[per_layer_name]
                per_layer_lr_scale = param_layer_lr_scale_dict[per_layer_name]

                if per_layer_weight_decay == per_decay and per_layer_lr == per_lr and per_layer_lr_scale == per_lr_scale:
                    per_decay_lr_lrscale_param_list.append(per_layer_weight)
                    per_decay_lr_lrscale_name_list.append(per_layer_name)

            assert len(per_decay_lr_lrscale_param_list) == len(
                per_decay_lr_lrscale_name_list)

            if len(per_decay_lr_lrscale_param_list) > 0:
                model_params_weight_decay_list.append({
                    'params':
                    per_decay_lr_lrscale_param_list,
                    'weight_decay':
                    per_decay,
                    'lr':
                    per_lr * per_lr_scale,
                })
                model_layer_weight_decay_list.append({
                    'name': per_decay_lr_lrscale_name_list,
                    'weight_decay': per_decay,
                    'lr': per_lr,
                    'lr_scale': per_lr_scale,
                })

        assert len(model_params_weight_decay_list) == len(
            model_layer_weight_decay_list)

    else:
        param_layer_name_list = []
        param_layer_weight_dict = {}
        param_layer_decay_dict, param_layer_lr_dict = {}, {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            param_layer_name_list.append(name)
            param_layer_weight_dict[name] = param

            if global_weight_decay is False:
                if param.ndim == 1 or any(no_weight_decay_layer_name in name
                                          for no_weight_decay_layer_name in
                                          no_weight_decay_layer_name_list):
                    param_layer_decay_dict[name] = 0.
                else:
                    per_layer_weight_decay = weight_decay
                    if 'sub_layer_weight_decay' in optimizer_parameters.keys(
                    ) and isinstance(
                            optimizer_parameters['sub_layer_weight_decay'],
                            dict):
                        for per_sub_layer_name_prefix, per_sub_layer_weight_decay in optimizer_parameters[
                                'sub_layer_weight_decay'].items():
                            if per_sub_layer_name_prefix in name:
                                per_layer_weight_decay = per_sub_layer_weight_decay
                                break
                    param_layer_decay_dict[name] = per_layer_weight_decay
            else:
                param_layer_decay_dict[name] = weight_decay

            per_layer_lr = lr
            if 'sub_layer_lr' in optimizer_parameters.keys() and isinstance(
                    optimizer_parameters['sub_layer_lr'], dict):
                for per_sub_layer_name_prefix, per_sub_layer_lr in optimizer_parameters[
                        'sub_layer_lr'].items():
                    if per_sub_layer_name_prefix in name:
                        per_layer_lr = per_sub_layer_lr
                        break
            param_layer_lr_dict[name] = per_layer_lr

        assert len(param_layer_name_list) == len(
            param_layer_weight_dict) == len(param_layer_decay_dict) == len(
                param_layer_lr_dict)

        unique_decays = list(set(param_layer_decay_dict.values()))
        unique_lrs = list(set(param_layer_lr_dict.values()))

        lr_weight_decay_combination = []
        for per_decay in unique_decays:
            for per_lr in unique_lrs:
                lr_weight_decay_combination.append([per_decay, per_lr])

        model_params_weight_decay_list = []
        model_layer_weight_decay_list = []
        for per_decay, per_lr in lr_weight_decay_combination:
            per_decay_lr_param_list, per_decay_lr_name_list = [], []
            for per_layer_name in param_layer_name_list:
                per_layer_weight = param_layer_weight_dict[per_layer_name]
                per_layer_weight_decay = param_layer_decay_dict[per_layer_name]
                per_layer_lr = param_layer_lr_dict[per_layer_name]

                if per_layer_weight_decay == per_decay and per_layer_lr == per_lr:
                    per_decay_lr_param_list.append(per_layer_weight)
                    per_decay_lr_name_list.append(per_layer_name)

            assert len(per_decay_lr_param_list) == len(per_decay_lr_name_list)

            if len(per_decay_lr_param_list) > 0:
                model_params_weight_decay_list.append({
                    'params': per_decay_lr_param_list,
                    'weight_decay': per_decay,
                    'lr': per_lr,
                })
                model_layer_weight_decay_list.append({
                    'name': per_decay_lr_name_list,
                    'weight_decay': per_decay,
                    'lr': per_lr,
                })

        assert len(model_params_weight_decay_list) == len(
            model_layer_weight_decay_list)

    return model_params_weight_decay_list, model_layer_weight_decay_list


def build_deepspeed_config(config):
    """Build DeepSpeed config dict from training config."""
    ds_config = {
        "train_micro_batch_size_per_gpu": config.batch_size // config.gpus_num,
        "gradient_accumulation_steps": config.accumulation_steps,
        # never print by deepspeed
        "steps_per_print": 2**31,
        "wall_clock_breakdown": False,
        "zero_optimization": {
            "stage": config.deepspeed_zero_stage,
        },
    }

    # Gradient clipping
    if hasattr(config, 'clip_max_norm') and config.clip_max_norm > 0:
        ds_config["gradient_clipping"] = config.clip_max_norm
    else:
        ds_config["gradient_clipping"] = 0.0

    # Mixed precision
    if config.use_amp:
        if config.amp_type == torch.float16:
            ds_config["fp16"] = {
                "enabled": True,
                # 0代表启用dynamic loss scaling
                "loss_scale": 0,
                # dynamic loss scaling的初始值为65536,2的16次方
                "initial_scale_power": 16,
                # 连续1000个step没有出现overflow的情况下loss scale会翻倍(尝试更激进的缩放以获得更好的梯度精度),1000是DeepSpeed默认值
                "loss_scale_window": 1000,
                # 连续发生2次overflow之后才真正将loss scale减半
                "hysteresis": 2,
                # loss scale下限为1
                "min_loss_scale": 1,
            }
            ds_config["torch_autocast"] = {
                "enabled": True,
                "dtype": "float16",
            }
        elif config.amp_type == torch.bfloat16:
            ds_config["bf16"] = {
                "enabled": True,
            }
            ds_config["torch_autocast"] = {
                "enabled": True,
                "dtype": "bfloat16",
            }
    else:
        ds_config["fp16"] = {"enabled": False}
        ds_config["bf16"] = {"enabled": False}

    # ZeRO-Offload
    if hasattr(config, 'deepspeed_offload') and config.deepspeed_offload:
        if config.deepspeed_zero_stage >= 2:
            # 将优化器状态(如Adam的一阶矩m和二阶矩v)卸载到CPU内存,仅在ZeRO Stage≥2时有意义
            ds_config["zero_optimization"]["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": True,
            }
        if config.deepspeed_zero_stage == 3:
            # 将模型参数本身卸载到CPU内存,仅在ZeRO Stage=3时可用
            ds_config["zero_optimization"]["offload_param"] = {
                "device": "cpu",
                "pin_memory": True,
            }

    # ZeRO Stage 3 specific
    if config.deepspeed_zero_stage == 3:
        # ZeRO-3下每个rank只持有1/N参数分片,直接state_dict()只能拿到本rank的模型分片参数。开启此选项后,调用model_engine.save_checkpoint()时DeepSpeed 会自动执行all-gather将完整的16-bit权重收集到一起保存
        ds_config["zero_optimization"][
            "stage3_gather_16bit_weights_on_model_save"] = True

    # Optimizer (let DeepSpeed create the optimizer natively for ZeRO
    # compatibility, especially for Muon which requires native support
    # under ZeRO stage 1/2/3).
    optimizer_name = config.optimizer[0]
    optimizer_parameters = config.optimizer[1]
    assert optimizer_name in ['SGD', 'AdamW', 'Muon'], 'Unsupported optimizer!'

    if optimizer_name == 'SGD':
        momentum = 0.9 if 'momentum' not in optimizer_parameters.keys(
        ) else optimizer_parameters['momentum']
        nesterov = False if 'nesterov' not in optimizer_parameters.keys(
        ) else optimizer_parameters['nesterov']
        ds_config["optimizer"] = {
            "type": "SGD",
            "params": {
                "lr": optimizer_parameters['lr'],
                "momentum": momentum,
                "nesterov": nesterov,
                "weight_decay": optimizer_parameters['weight_decay'],
            }
        }
    elif optimizer_name == 'AdamW':
        beta1 = 0.9 if 'beta1' not in optimizer_parameters.keys(
        ) else optimizer_parameters['beta1']
        beta2 = 0.999 if 'beta2' not in optimizer_parameters.keys(
        ) else optimizer_parameters['beta2']
        eps = 1e-08 if 'eps' not in optimizer_parameters.keys(
        ) else optimizer_parameters['eps']
        ds_config["optimizer"] = {
            "type": "AdamW",
            "params": {
                "lr": optimizer_parameters['lr'],
                "betas": [beta1, beta2],
                "eps": eps,
                "weight_decay": optimizer_parameters['weight_decay'],
            }
        }
    elif optimizer_name == 'Muon':
        momentum = 0.95 if 'momentum' not in optimizer_parameters.keys(
        ) else optimizer_parameters['momentum']
        nesterov = True if 'nesterov' not in optimizer_parameters.keys(
        ) else optimizer_parameters['nesterov']
        ns_steps = 5 if 'ns_steps' not in optimizer_parameters.keys(
        ) else optimizer_parameters['ns_steps']
        adamw_beta1 = 0.9 if 'adamw_beta1' not in optimizer_parameters.keys(
        ) else optimizer_parameters['adamw_beta1']
        adamw_beta2 = 0.999 if 'adamw_beta2' not in optimizer_parameters.keys(
        ) else optimizer_parameters['adamw_beta2']
        adamw_eps = 1e-08 if 'adamw_eps' not in optimizer_parameters.keys(
        ) else optimizer_parameters['adamw_eps']
        ds_config["optimizer"] = {
            "type": "Muon",
            "params": {
                "lr": optimizer_parameters['lr'],
                "wd": optimizer_parameters['weight_decay'],
                "momentum": momentum,
                "nesterov": nesterov,
                "ns_steps": ns_steps,
                "adamw_betas": [adamw_beta1, adamw_beta2],
                "adamw_eps": adamw_eps,
            }
        }

    return ds_config


def get_model_state_dict(model_engine, config):
    """
    Get full model state dict for saving.
    For ZeRO-3, all ranks must call this function (GatheredParameters is
    collective), but only rank 0 returns a non-None dict.
    """
    if config.use_compile:
        module = model_engine.module._orig_mod
    else:
        module = model_engine.module

    if config.deepspeed_zero_stage == 3:
        state_dict = {}
        for name, param in module.named_parameters():
            with deepspeed.zero.GatheredParameters(param):
                if config.total_rank == 0 and config.local_rank == 0:
                    state_dict[name] = param.data.cpu().clone()
        for name, buf in module.named_buffers():
            if config.total_rank == 0 and config.local_rank == 0:
                state_dict[name] = buf.cpu().clone()
        return state_dict if config.total_rank == 0 and config.local_rank == 0 else None
    else:
        return module.state_dict()


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch FSQ Training')
    parser.add_argument(
        '--work-dir',
        type=str,
        help='path for get training config and saving log/models')
    args, _ = parser.parse_known_args()

    return args


def main():
    assert torch.cuda.is_available(), 'need gpu to train network!'
    torch.cuda.empty_cache()

    args = parse_args()
    sys.path.append(args.work_dir)
    from train_config import config
    log_dir = os.path.join(args.work_dir, 'log')
    checkpoint_dir = os.path.join(args.work_dir, 'checkpoints')
    config.checkpoint_dir = checkpoint_dir
    config.gpus_type = torch.cuda.get_device_name()
    config.gpus_num = torch.cuda.device_count()

    if config.deepspeed_zero_stage == 3:
        resume_model = os.path.join(
            checkpoint_dir, 'zero_pp_rank_0_mp_rank_00_model_states.pt')
    else:
        resume_model = os.path.join(checkpoint_dir,
                                    'mp_rank_00_model_states.pt')

    set_seed(config.seed)

    local_rank = int(os.environ['LOCAL_RANK'])
    config.local_rank = local_rank
    # start init process
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed(dist_backend='nccl')

    # 获取total_rank
    total_rank = torch.distributed.get_rank()
    config.total_rank = total_rank

    # 假设每个进程只使用一个GPU
    # 获取当前node上进程数量
    per_node_process_nums = int(os.environ['LOCAL_WORLD_SIZE'])
    # 获取当前node上GPU数量
    per_node_gpus_num = torch.cuda.device_count()
    # 获取当前node上每个进程分配的GPU数量
    per_node_per_process_gpus_num = int(per_node_gpus_num /
                                        per_node_process_nums)
    # 获取所有node上进程数量
    world_size = torch.distributed.get_world_size()
    # 获取所有node上GPU数量:每个进程分配的GPU数量×所有node上进程数量
    config.gpus_num = int(per_node_per_process_gpus_num * world_size)
    config.group = torch.distributed.new_group(list(range(config.gpus_num)))

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    torch.distributed.barrier(device_ids=[local_rank])

    logger = get_logger('train', log_dir)

    batch_size, num_workers = config.batch_size, config.num_workers
    assert config.batch_size % config.gpus_num == 0, 'config.batch_size is not divisible by config.gpus_num!'
    assert config.num_workers % config.gpus_num == 0, 'config.num_workers is not divisible by config.gpus_num!'
    batch_size = int(config.batch_size // config.gpus_num)
    num_workers = int(config.num_workers // config.gpus_num)

    init_fn = functools.partial(worker_seed_init_fn,
                                num_workers=num_workers,
                                local_rank=local_rank,
                                seed=config.seed)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        config.train_dataset, shuffle=True)
    train_loader = DataLoader(config.train_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=True,
                              drop_last=True,
                              num_workers=num_workers,
                              collate_fn=config.train_collater,
                              sampler=train_sampler,
                              worker_init_fn=init_fn)

    for key, value in config.__dict__.items():
        if not key.startswith('__'):
            if key not in [
                    'model',
                    'train_criterion',
            ]:
                log_info = f'{key}: {value}'
                logger.info(
                    log_info) if local_rank == 0 and total_rank == 0 else None

    model = config.model.cuda()
    train_criterion = config.train_criterion

    for name in train_criterion.keys():
        train_criterion[name] = train_criterion[name].cuda()

    # parameters needs to be updated by the optimizer
    # buffers doesn't needs to be updated by the optimizer
    log_info = f'--------------------parameters--------------------'
    logger.info(log_info) if local_rank == 0 and total_rank == 0 else None
    for name, param in model.named_parameters():
        log_info = f'name: {name}, grad: {param.requires_grad}'
        logger.info(log_info) if local_rank == 0 and total_rank == 0 else None

    log_info = f'--------------------buffers--------------------'
    logger.info(log_info) if local_rank == 0 and total_rank == 0 else None
    for name, buffer in model.named_buffers():
        log_info = f'name: {name}, grad: {buffer.requires_grad}'
        logger.info(log_info) if local_rank == 0 and total_rank == 0 else None

    model_params_weight_decay_list, model_layer_weight_decay_list = build_param_groups(
        config, model)

    log_info = f'-------------layers weight decay---------------'
    logger.info(log_info) if local_rank == 0 and total_rank == 0 else None
    for per_layer_list in model_layer_weight_decay_list:
        layer_name_list, layer_lr, layer_weight_decay = per_layer_list[
            'name'], per_layer_list['lr'], per_layer_list['weight_decay']

        lr_scale = 'not setting!'
        if 'lr_scale' in per_layer_list.keys():
            lr_scale = per_layer_list['lr_scale']

        for name in layer_name_list:
            log_info = f'name: {name}, lr: {layer_lr}, weight_decay: {layer_weight_decay}, lr_scale: {lr_scale}'
            logger.info(
                log_info) if local_rank == 0 and total_rank == 0 else None

    # Check torch compile support
    config.compile_support = False
    log_info = f'using torch version:{torch.__version__}'
    logger.info(log_info) if local_rank == 0 and total_rank == 0 else None
    if re.match(r'2.\d*.\d*', torch.__version__):
        config.compile_support = True
        log_info = f'this torch version support torch.compile function.'
        logger.info(log_info) if local_rank == 0 and total_rank == 0 else None
    elif re.match(r'1.\d*.\d*', torch.__version__):
        log_info = f'this torch version unsupport torch.compile function.'
        logger.info(log_info) if local_rank == 0 and total_rank == 0 else None
    else:
        log_info = f'unsupport torch version:{torch.__version__}'
        logger.info(log_info) if local_rank == 0 and total_rank == 0 else None
        return

    config.use_compile = (config.compile_support and config.use_compile)

    if config.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if config.use_compile:
        # _orig_mod
        model = torch.compile(model, **config.compile_params)

    # Build DeepSpeed config and initialize engine
    ds_config = build_deepspeed_config(config)
    log_info = f'DeepSpeed config: {ds_config}'
    logger.info(log_info) if local_rank == 0 and total_rank == 0 else None

    # Let DeepSpeed create the optimizer from ds_config (which includes the
    # "optimizer" section). Pass model_parameters for per-layer weight decay
    # and lr differentiation. DeepSpeed natively handles SGD/AdamW/Muon,
    # ensuring correct optimizer state partitioning under ZeRO stage 1/2/3.
    # for deepspeed==0.18.8, Muon optimizer is not yet compatible with ZeRO Stage 3,only support ZeRO stage 0/1/2.
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model_params_weight_decay_list,
        config=ds_config)

    # Build scheduler after DeepSpeed creates the optimizer. The scheduler
    # adjusts LR via optimizer.param_groups which DeepSpeed's optimizer
    # wrapper correctly exposes and delegates to the underlying optimizer.
    scheduler = Scheduler(config, optimizer)

    start_epoch, train_time = 1, 0
    best_loss, train_loss = 1e9, 0
    # Resume from DeepSpeed checkpoint (tag="" saves directly in checkpoint_dir)
    if os.path.exists(resume_model):
        _, client_state = model_engine.load_checkpoint(checkpoint_dir, tag="")
        if client_state is not None:
            saved_epoch = client_state['epoch']
            start_epoch += saved_epoch
            used_time = client_state['time']
            train_time += used_time

            best_loss = client_state['best_loss']
            train_loss = client_state['train_loss']
            scheduler.load_state_dict(client_state['scheduler_state_dict'])

            log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch:0>3d}, used_time: {used_time:.3f} hours, best_loss: {best_loss:.4f}, lr: {scheduler.current_lr:.6f}'
            logger.info(
                log_info) if local_rank == 0 and total_rank == 0 else None

    for epoch in range(start_epoch, config.epochs + 1):
        per_epoch_start_time = time.time()

        log_info = f'epoch {epoch:0>3d} lr: {scheduler.current_lr:.6f}'
        logger.info(log_info) if local_rank == 0 and total_rank == 0 else None

        torch.cuda.empty_cache()

        train_sampler.set_epoch(epoch)
        train_loss = train_fsq_model_deepspeed(train_loader, model_engine,
                                               train_criterion, optimizer,
                                               scheduler, epoch, logger,
                                               config)
        log_info = f'train: epoch {epoch:0>3d}, train_loss: {train_loss:.4f}'
        logger.info(log_info) if local_rank == 0 and total_rank == 0 else None

        torch.cuda.empty_cache()

        train_time += (time.time() - per_epoch_start_time) / 3600

        # train_loss is consistent across all ranks (all_reduced in
        # train_fsq_model_deepspeed), so is_best is identical on every rank.
        is_best = train_loss < best_loss
        if is_best:
            best_loss = train_loss

        if epoch in config.save_epochs or epoch == config.epochs:
            if config.deepspeed_zero_stage == 3:
                # ZeRO-3: all ranks must participate in GatheredParameters
                save_model = get_model_state_dict(model_engine, config)
                if local_rank == 0 and total_rank == 0 and save_model is not None:
                    torch.save(
                        save_model,
                        os.path.join(checkpoint_dir,
                                     f'epoch_{epoch}_model.pth'))
            else:
                # ZeRO-0/1/2: only global rank 0 needs to call state_dict
                if local_rank == 0 and total_rank == 0:
                    save_model = get_model_state_dict(model_engine, config)
                    torch.save(
                        save_model,
                        os.path.join(checkpoint_dir,
                                     f'epoch_{epoch}_model.pth'))

        if is_best:
            if config.deepspeed_zero_stage == 3:
                save_best_model = get_model_state_dict(model_engine, config)
                if local_rank == 0 and total_rank == 0 and save_best_model is not None:
                    torch.save(save_best_model,
                               os.path.join(checkpoint_dir, 'best.pth'))
            else:
                if local_rank == 0 and total_rank == 0:
                    save_best_model = get_model_state_dict(
                        model_engine, config)
                    torch.save(save_best_model,
                               os.path.join(checkpoint_dir, 'best.pth'))

        # Save DeepSpeed checkpoint for resume (all ranks participate)
        client_state = {
            'epoch': epoch,
            'time': train_time,
            'best_loss': best_loss,
            'train_loss': train_loss,
            'lr': scheduler.current_lr,
            'scheduler_state_dict': scheduler.state_dict(),
        }
        model_engine.save_checkpoint(checkpoint_dir,
                                     tag="",
                                     client_state=client_state,
                                     save_latest=False)

        log_info = f'until epoch: {epoch:0>3d}, best_loss: {best_loss:.4f}'
        logger.info(log_info) if local_rank == 0 and total_rank == 0 else None

    if local_rank == 0 and total_rank == 0:
        if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
            os.rename(
                os.path.join(checkpoint_dir, 'best.pth'),
                os.path.join(checkpoint_dir, f'total_loss{best_loss:.3f}.pth'))

    log_info = f'train done. train time: {train_time:.3f} hours, best_loss: {best_loss:.4f}'
    logger.info(log_info) if local_rank == 0 and total_rank == 0 else None

    torch.distributed.destroy_process_group()

    return


if __name__ == '__main__':
    main()
