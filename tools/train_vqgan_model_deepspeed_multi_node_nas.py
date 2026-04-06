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
import math

import torch
import deepspeed
from torch.utils.data import DataLoader

from tools.image_tokenizer_scripts import train_vqgan_model_deepspeed
from tools.utils import (get_logger, set_seed, worker_seed_init_fn)

# PatchGANDiscriminator is very small (~2.77M params, ~44 MB total memory
# per GPU including optimizer states). ZeRO partitioning brings negligible
# memory savings but adds communication overhead.  Fix it at Stage 0 (DDP).
DISCRIMINATOR_ZERO_STAGE = 0


def build_param_groups(config, model, model_type):
    """Build parameter groups for DeepSpeed optimizer.
    For AdamW: differentiate weight decay (1D params and specified layers get 0).
    For Muon: pass all trainable params; DeepSpeed native Muon handles split.
    Returns (model_params_weight_decay_list, model_layer_weight_decay_list).
    """
    assert model_type in [
        'generator_model',
        'discriminator_model',
    ]
    if model_type == 'generator_model':
        optimizer_name = config.generator_optimizer[0]
        optimizer_parameters = config.generator_optimizer[1]
    elif model_type == 'discriminator_model':
        optimizer_name = config.discriminator_optimizer[0]
        optimizer_parameters = config.discriminator_optimizer[1]

    assert optimizer_name in ['SGD', 'AdamW', 'Muon'], 'Unsupported optimizer!'

    lr = optimizer_parameters['lr']
    weight_decay = optimizer_parameters['weight_decay']

    # For Muon, DeepSpeed 0.18.9 native implementation requires each
    # parameter to have a `use_muon` attribute (True/False) so that
    # the engine can split params into Muon group (ndim>=2) and
    # AdamW fallback group (ndim<2).
    if optimizer_name == 'Muon':
        muon_param_names = []
        adamw_param_names = []
        all_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            all_params.append(param)
            # DeepSpeed 0.18.9 engine.py requires `param.use_muon`
            # attribute on every parameter. Set it based on ndim >= 2.
            if param.ndim >= 2:
                param.use_muon = True
                muon_param_names.append(name)
            else:
                param.use_muon = False
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


def build_deepspeed_config(config, model_type):
    """Build DeepSpeed config dict from training config for generator or discriminator."""
    assert model_type in [
        'generator_model',
        'discriminator_model',
    ]
    if model_type == 'generator_model':
        optimizer_name = config.generator_optimizer[0]
        optimizer_parameters = config.generator_optimizer[1]
        zero_stage = config.deepspeed_zero_stage
    elif model_type == 'discriminator_model':
        optimizer_name = config.discriminator_optimizer[0]
        optimizer_parameters = config.discriminator_optimizer[1]
        # PatchGANDiscriminator is very small, always use ZeRO Stage 0
        zero_stage = DISCRIMINATOR_ZERO_STAGE

    ds_config = {
        "train_micro_batch_size_per_gpu": config.batch_size // config.gpus_num,
        "gradient_accumulation_steps": config.accumulation_steps,
        # never print by deepspeed
        "steps_per_print": 2**31,
        "wall_clock_breakdown": False,
        "zero_optimization": {
            "stage": zero_stage,
        },
    }

    # Gradient clipping
    if model_type == 'generator_model':
        if hasattr(config, 'generator_clip_max_norm'
                   ) and config.generator_clip_max_norm > 0:
            ds_config["gradient_clipping"] = config.generator_clip_max_norm
        else:
            ds_config["gradient_clipping"] = 0.0
    elif model_type == 'discriminator_model':
        if hasattr(config, 'discriminator_clip_max_norm'
                   ) and config.discriminator_clip_max_norm > 0:
            ds_config["gradient_clipping"] = config.discriminator_clip_max_norm
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
        ds_config["fp16"] = {
            "enabled": False,
        }
        ds_config["bf16"] = {
            "enabled": False,
        }

    # ZeRO-Offload
    if hasattr(config, 'deepspeed_offload') and config.deepspeed_offload:
        if zero_stage >= 2:
            # 将优化器状态(如Adam的一阶矩m和二阶矩v)卸载到CPU内存,仅在ZeRO Stage≥2时有意义
            ds_config["zero_optimization"]["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": True,
            }
        if zero_stage == 3:
            # 将模型参数本身卸载到CPU内存,仅在ZeRO Stage=3时可用
            ds_config["zero_optimization"]["offload_param"] = {
                "device": "cpu",
                "pin_memory": True,
            }

    # ZeRO Stage 3 specific
    if zero_stage == 3:
        # ZeRO-3下每个rank只持有1/N参数分片,直接state_dict()只能拿到本rank的模型分片参数。开启此选项后,调用model_engine.save_checkpoint()时DeepSpeed 会自动执行all-gather将完整的16-bit权重收集到一起保存
        ds_config["zero_optimization"][
            "stage3_gather_16bit_weights_on_model_save"] = True

    ds_config["flops_profiler"] = {
        "enabled": True,
        "profile_step": 1,  # 在第1个step进行profiling
        "module_depth": -1,  # -1表示打印所有层级
        "top_modules": 3,  # 打印top 3模块
        "detailed": True,  # 打印详细信息
        "output_file": None,  # None表示输出到DeepSpeed log(终端/日志)
    }

    # Optimizer (let DeepSpeed create the optimizer natively for ZeRO
    # compatibility, especially for Muon which requires native support
    # under ZeRO stage 1/2/3).
    assert optimizer_name in ['SGD', 'AdamW', 'Muon'], 'Unsupported optimizer!'

    # for deepspeed==0.18.9, muon optimizer for zero stage3, reduce_scatter must be false, use allreduce instead
    if optimizer_name == 'Muon' and zero_stage == 3:
        ds_config["zero_optimization"]["reduce_scatter"] = False

    if optimizer_name == 'SGD':
        ds_config["optimizer"] = {
            "type": "SGD",
            "params": {
                "lr": optimizer_parameters['lr'],
                "momentum": optimizer_parameters.get('momentum', 0.9),
                "nesterov": optimizer_parameters.get('nesterov', False),
                "weight_decay": optimizer_parameters['weight_decay'],
            }
        }
    elif optimizer_name == 'AdamW':
        ds_config["optimizer"] = {
            "type": "AdamW",
            "params": {
                "lr":
                optimizer_parameters['lr'],
                "betas": [
                    optimizer_parameters.get('beta1', 0.9),
                    optimizer_parameters.get('beta2', 0.999)
                ],
                "eps":
                optimizer_parameters.get('eps', 1e-08),
                "weight_decay":
                optimizer_parameters['weight_decay'],
            }
        }
    elif optimizer_name == 'Muon':
        # DeepSpeed 0.18.9 engine.py _configure_basic_optimizer() only
        # recognizes these keys for Muon param groups:
        #   muon group:  ["lr", "momentum", "weight_decay", "muon_lr"]
        #   adamw group: ["lr", "betas", "eps", "weight_decay", "adam_lr"]
        # Keys like "wd", "nesterov", "ns_steps", "adamw_betas", "adamw_eps"
        # are NOT recognized and will be silently ignored.
        ds_config["optimizer"] = {
            "type": "Muon",
            "params": {
                "lr":
                optimizer_parameters['lr'],
                "weight_decay":
                optimizer_parameters['weight_decay'],
                "momentum":
                optimizer_parameters.get('momentum', 0.95),
                "betas": [
                    optimizer_parameters.get('adamw_beta1', 0.9),
                    optimizer_parameters.get('adamw_beta2', 0.999)
                ],
                "eps":
                optimizer_parameters.get('adamw_eps', 1e-08),
            }
        }

    return ds_config


def get_model_state_dict(model_engine, config, zero_stage):
    """Get full model state dict for saving.
    For ZeRO-3, all ranks must call (GatheredParameters is collective),
    but only rank 0 returns a non-None dict.
    """
    if config.use_compile:
        module = model_engine.module._orig_mod
    else:
        module = model_engine.module

    if zero_stage == 3:
        # Batch gather all parameters at once to reduce communication rounds
        all_params = list(module.parameters())
        with deepspeed.zero.GatheredParameters(all_params):
            if config.total_rank == 0 and config.local_rank == 0:
                state_dict = {
                    k: v.cpu().clone()
                    for k, v in module.state_dict().items()
                }
            else:
                state_dict = None
        return state_dict
    else:
        return module.state_dict()


class Scheduler:

    def __init__(self, config, optimizer, model_type):
        assert model_type in [
            'generator_model',
            'discriminator_model',
        ]
        if model_type == 'generator_model':
            self.scheduler_name = config.generator_scheduler[0]
            self.scheduler_parameters = config.generator_scheduler[1]
            self.optimizer_parameters = config.generator_optimizer[1]

        elif model_type == 'discriminator_model':
            self.scheduler_name = config.discriminator_scheduler[0]
            self.scheduler_parameters = config.discriminator_scheduler[1]
            self.optimizer_parameters = config.discriminator_optimizer[1]

        self.warm_up_epochs = self.scheduler_parameters['warm_up_epochs']
        self.epochs = config.epochs

        self.lr = self.optimizer_parameters['lr']
        self.current_lr = self.lr

        self.init_param_groups_lr = [
            param_group["lr"] for param_group in optimizer.param_groups
        ]

        assert self.scheduler_name in ['MultiStepLR', 'CosineLR',
                                       'PolyLR'], 'Unsupported scheduler!'
        assert self.warm_up_epochs >= 0, 'Illegal warm_up_epochs!'
        assert self.epochs > 0, 'Illegal epochs!'

    def step(self, optimizer, epoch):
        if self.scheduler_name == 'MultiStepLR':
            gamma = self.scheduler_parameters['gamma']
            milestones = self.scheduler_parameters['milestones']
        elif self.scheduler_name == 'CosineLR':
            min_lr = 0. if 'min_lr' not in self.scheduler_parameters.keys(
            ) else self.scheduler_parameters['min_lr']
        elif self.scheduler_name == 'PolyLR':
            power = self.scheduler_parameters['power']
            min_lr = 0. if 'min_lr' not in self.scheduler_parameters.keys(
            ) else self.scheduler_parameters['min_lr']

        assert len(self.init_param_groups_lr) == len(optimizer.param_groups)

        for idx, param_group in enumerate(optimizer.param_groups):
            param_group_init_lr = self.init_param_groups_lr[idx]

            if self.scheduler_name == 'MultiStepLR':
                param_group_current_lr = (
                    epoch
                ) / self.warm_up_epochs * param_group_init_lr if epoch < self.warm_up_epochs else gamma**len(
                    [m
                     for m in milestones if m <= epoch]) * param_group_init_lr
            elif self.scheduler_name == 'CosineLR':
                param_group_current_lr = (
                    epoch
                ) / self.warm_up_epochs * param_group_init_lr if epoch < self.warm_up_epochs else 0.5 * (
                    math.cos((epoch - self.warm_up_epochs) /
                             (self.epochs - self.warm_up_epochs) * math.pi) +
                    1) * (param_group_init_lr - min_lr) + min_lr
            elif self.scheduler_name == 'PolyLR':
                param_group_current_lr = (
                    epoch
                ) / self.warm_up_epochs * param_group_init_lr if epoch < self.warm_up_epochs else (
                    (1 - (epoch - self.warm_up_epochs) /
                     (self.epochs - self.warm_up_epochs))**
                    power) * (param_group_init_lr - min_lr) + min_lr

            param_group["lr"] = param_group_current_lr

        if self.scheduler_name == 'MultiStepLR':
            self.current_lr = (
                epoch
            ) / self.warm_up_epochs * self.lr if epoch < self.warm_up_epochs else gamma**len(
                [m for m in milestones if m <= epoch]) * self.lr
        elif self.scheduler_name == 'CosineLR':
            self.current_lr = (
                epoch
            ) / self.warm_up_epochs * self.lr if epoch < self.warm_up_epochs else 0.5 * (
                math.cos((epoch - self.warm_up_epochs) /
                         (self.epochs - self.warm_up_epochs) * math.pi) +
                1) * (self.lr - min_lr) + min_lr
        elif self.scheduler_name == 'PolyLR':
            self.current_lr = (
                epoch
            ) / self.warm_up_epochs * self.lr if epoch < self.warm_up_epochs else (
                (1 - (epoch - self.warm_up_epochs) /
                 (self.epochs - self.warm_up_epochs))**
                power) * (self.lr - min_lr) + min_lr

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch VQGAN Training')
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

    generator_checkpoint_dir = os.path.join(checkpoint_dir, 'generator')
    discriminator_checkpoint_dir = os.path.join(checkpoint_dir,
                                                'discriminator')
    os.makedirs(generator_checkpoint_dir, exist_ok=True)
    os.makedirs(discriminator_checkpoint_dir, exist_ok=True)

    if config.deepspeed_zero_stage == 3:
        resume_generator_model = os.path.join(
            generator_checkpoint_dir,
            'zero_pp_rank_0_mp_rank_00_model_states.pt')
    else:
        resume_generator_model = os.path.join(generator_checkpoint_dir,
                                              'mp_rank_00_model_states.pt')
    # Discriminator always uses ZeRO Stage 0, so always non-ZeRO-3 path
    resume_discriminator_model = os.path.join(discriminator_checkpoint_dir,
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

    config.gpus_num = torch.distributed.get_world_size()

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

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
                    'generator_model',
                    'discriminator_model',
                    'criterion',
            ]:
                log_info = f'{key}: {value}'
                logger.info(
                    log_info) if local_rank == 0 and total_rank == 0 else None

    generator_model = config.generator_model.cuda()
    discriminator_model = config.discriminator_model.cuda()
    criterion = config.criterion.cuda()

    # parameters needs to be updated by the optimizer
    # buffers doesn't needs to be updated by the optimizer
    log_info = f'----------------generator model parameters----------------'
    logger.info(log_info) if local_rank == 0 and total_rank == 0 else None
    for name, param in generator_model.named_parameters():
        log_info = f'name: {name}, grad: {param.requires_grad}'
        logger.info(log_info) if local_rank == 0 and total_rank == 0 else None

    log_info = f'------------------generator model buffers------------------'
    logger.info(log_info) if local_rank == 0 and total_rank == 0 else None
    for name, buffer in generator_model.named_buffers():
        log_info = f'name: {name}, grad: {buffer.requires_grad}'
        logger.info(log_info) if local_rank == 0 and total_rank == 0 else None

    # parameters needs to be updated by the optimizer
    # buffers doesn't needs to be updated by the optimizer
    log_info = f'----------------discriminator model parameters----------------'
    logger.info(log_info) if local_rank == 0 and total_rank == 0 else None
    for name, param in discriminator_model.named_parameters():
        log_info = f'name: {name}, grad: {param.requires_grad}'
        logger.info(log_info) if local_rank == 0 and total_rank == 0 else None

    log_info = f'------------------discriminator model buffers------------------'
    logger.info(log_info) if local_rank == 0 and total_rank == 0 else None
    for name, buffer in discriminator_model.named_buffers():
        log_info = f'name: {name}, grad: {buffer.requires_grad}'
        logger.info(log_info) if local_rank == 0 and total_rank == 0 else None

    generator_model_params_weight_decay_list, generator_model_layer_weight_decay_list = build_param_groups(
        config, generator_model, model_type='generator_model')

    log_info = f'-------------generator model layers weight decay---------------'
    logger.info(log_info) if local_rank == 0 and total_rank == 0 else None
    for per_layer_list in generator_model_layer_weight_decay_list:
        layer_name_list, layer_lr, layer_weight_decay = per_layer_list[
            'name'], per_layer_list['lr'], per_layer_list['weight_decay']

        lr_scale = 'not setting!'
        if 'lr_scale' in per_layer_list.keys():
            lr_scale = per_layer_list['lr_scale']

        for name in layer_name_list:
            log_info = f'name: {name}, lr: {layer_lr}, weight_decay: {layer_weight_decay}, lr_scale: {lr_scale}'
            logger.info(
                log_info) if local_rank == 0 and total_rank == 0 else None

    discriminator_model_params_weight_decay_list, discriminator_model_layer_weight_decay_list = build_param_groups(
        config, discriminator_model, model_type='discriminator_model')

    log_info = f'-------------discriminator model layers weight decay---------------'
    logger.info(log_info) if local_rank == 0 and total_rank == 0 else None
    for per_layer_list in discriminator_model_layer_weight_decay_list:
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
        generator_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
            generator_model)
        discriminator_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
            discriminator_model)

    if config.use_compile:
        # _orig_mod
        generator_model = torch.compile(generator_model,
                                        **config.compile_params)
        discriminator_model = torch.compile(discriminator_model,
                                            **config.compile_params)

    # Build DeepSpeed config and initialize engines
    generator_ds_config = build_deepspeed_config(config,
                                                 model_type='generator_model')
    log_info = f'Generator DeepSpeed config: {generator_ds_config}'
    logger.info(log_info) if local_rank == 0 and total_rank == 0 else None

    discriminator_ds_config = build_deepspeed_config(
        config, model_type='discriminator_model')
    log_info = f'Discriminator DeepSpeed config: {discriminator_ds_config}'
    logger.info(log_info) if local_rank == 0 and total_rank == 0 else None

    # Let DeepSpeed create the optimizer from ds_config (which includes the
    # "optimizer" section). Pass model_parameters for per-layer weight decay
    # and lr differentiation. DeepSpeed natively handles SGD/AdamW/Muon,
    # ensuring correct optimizer state partitioning under ZeRO stage 1/2/3.
    # for deepspeed==0.18.8, Muon optimizer is not yet compatible with ZeRO Stage 3,only support ZeRO stage 0/1/2.
    generator_model_engine, generator_optimizer, _, _ = deepspeed.initialize(
        model=generator_model,
        model_parameters=generator_model_params_weight_decay_list,
        config=generator_ds_config)

    discriminator_model_engine, discriminator_optimizer, _, _ = deepspeed.initialize(
        model=discriminator_model,
        model_parameters=discriminator_model_params_weight_decay_list,
        config=discriminator_ds_config)

    # Build scheduler after DeepSpeed creates the optimizer. The scheduler
    # adjusts LR via optimizer.param_groups which DeepSpeed's optimizer
    # wrapper correctly exposes and delegates to the underlying optimizer.
    generator_scheduler = Scheduler(config,
                                    generator_optimizer,
                                    model_type='generator_model')
    discriminator_scheduler = Scheduler(config,
                                        discriminator_optimizer,
                                        model_type='discriminator_model')

    start_epoch, train_time = 1, 0
    best_loss, train_loss = 1e9, 0
    # Resume from DeepSpeed checkpoint (generator uses tag="generator", discriminator uses tag="discriminator")
    if os.path.exists(resume_generator_model) and os.path.exists(
            resume_discriminator_model):
        _, client_state = generator_model_engine.load_checkpoint(
            generator_checkpoint_dir, tag="")
        discriminator_model_engine.load_checkpoint(
            discriminator_checkpoint_dir, tag="")
        if client_state is not None:
            saved_epoch = client_state['epoch']
            start_epoch += saved_epoch
            used_time = client_state['time']
            train_time += used_time

            best_loss = client_state['best_loss']
            train_loss = client_state['train_loss']
            generator_scheduler.load_state_dict(
                client_state['generator_scheduler_state_dict'])
            discriminator_scheduler.load_state_dict(
                client_state['discriminator_scheduler_state_dict'])

            log_info = f'resuming model from {resume_generator_model} and {resume_discriminator_model}. resume_epoch: {saved_epoch:0>3d}, used_time: {used_time:.3f} hours, best_loss: {best_loss:.4f}, generator_lr: {generator_scheduler.current_lr:.6f}, discriminator_lr: {discriminator_scheduler.current_lr:.6f}'
            logger.info(
                log_info) if local_rank == 0 and total_rank == 0 else None

    for epoch in range(start_epoch, config.epochs + 1):
        per_epoch_start_time = time.time()

        log_info = f'epoch {epoch:0>3d} generator_lr: {generator_scheduler.current_lr:.6f}'
        logger.info(log_info) if local_rank == 0 and total_rank == 0 else None
        log_info = f'epoch {epoch:0>3d} discriminator_lr: {discriminator_scheduler.current_lr:.6f}'
        logger.info(log_info) if local_rank == 0 and total_rank == 0 else None

        torch.cuda.empty_cache()

        train_sampler.set_epoch(epoch)
        generator_loss, discriminator_loss, total_loss = train_vqgan_model_deepspeed(
            train_loader, generator_model_engine, discriminator_model_engine,
            criterion, generator_optimizer, discriminator_optimizer,
            generator_scheduler, discriminator_scheduler, epoch, logger,
            config)
        log_info = f'train: epoch {epoch:0>3d}, generator_loss: {generator_loss:.4f}, discriminator_loss: {discriminator_loss:.4f}, total_loss: {total_loss:.4f}'
        logger.info(log_info) if local_rank == 0 and total_rank == 0 else None

        train_loss = generator_loss

        torch.cuda.empty_cache()

        train_time += (time.time() - per_epoch_start_time) / 3600

        # train_loss is consistent across all ranks (all_reduced in
        # train_vqgan_model_deepspeed), so is_best is identical on every rank.
        is_best = train_loss < best_loss
        if is_best:
            best_loss = train_loss

        # Merge save_interval and is_best saving to avoid redundant
        # GatheredParameters calls under ZeRO-3 (each call triggers
        # all-gather across all ranks for all parameters).
        need_save_epoch = (epoch % config.save_interval == 0)
        need_save_best = is_best

        if need_save_epoch or need_save_best:
            # Generator: save depends on config.deepspeed_zero_stage
            if config.deepspeed_zero_stage == 3:
                # ZeRO-3: all ranks must participate in GatheredParameters
                save_generator_model = get_model_state_dict(
                    generator_model_engine, config,
                    config.deepspeed_zero_stage)
                if local_rank == 0 and total_rank == 0 and save_generator_model is not None:
                    if need_save_epoch:
                        torch.save(
                            save_generator_model,
                            os.path.join(checkpoint_dir,
                                         f'epoch_{epoch}_generator_model.pth'))
                    if need_save_best:
                        torch.save(
                            save_generator_model,
                            os.path.join(checkpoint_dir,
                                         'best_generator_model.pth'))
            else:
                # ZeRO-0/1/2: only global rank 0 needs to call state_dict
                if local_rank == 0 and total_rank == 0:
                    save_generator_model = get_model_state_dict(
                        generator_model_engine, config,
                        config.deepspeed_zero_stage)
                    if need_save_epoch:
                        torch.save(
                            save_generator_model,
                            os.path.join(checkpoint_dir,
                                         f'epoch_{epoch}_generator_model.pth'))
                    if need_save_best:
                        torch.save(
                            save_generator_model,
                            os.path.join(checkpoint_dir,
                                         'best_generator_model.pth'))
            # Discriminator: always ZeRO Stage 0, only rank 0 saves
            if local_rank == 0 and total_rank == 0:
                save_discriminator_model = get_model_state_dict(
                    discriminator_model_engine, config,
                    DISCRIMINATOR_ZERO_STAGE)
                if need_save_epoch:
                    torch.save(
                        save_discriminator_model,
                        os.path.join(checkpoint_dir,
                                     f'epoch_{epoch}_discriminator_model.pth'))
                if need_save_best:
                    torch.save(
                        save_discriminator_model,
                        os.path.join(checkpoint_dir,
                                     'best_discriminator_model.pth'))

        # Save DeepSpeed checkpoint for resume (all ranks participate)
        # client_state is saved with generator engine only
        client_state = {
            'epoch':
            epoch,
            'time':
            train_time,
            'best_loss':
            best_loss,
            'train_loss':
            train_loss,
            'generator_lr':
            generator_scheduler.current_lr,
            'discriminator_lr':
            discriminator_scheduler.current_lr,
            'generator_scheduler_state_dict':
            generator_scheduler.state_dict(),
            'discriminator_scheduler_state_dict':
            discriminator_scheduler.state_dict(),
        }
        generator_model_engine.save_checkpoint(generator_checkpoint_dir,
                                               tag="",
                                               client_state=client_state,
                                               save_latest=False)
        discriminator_model_engine.save_checkpoint(
            discriminator_checkpoint_dir,
            tag="",
            client_state=None,
            save_latest=False)

        log_info = f'until epoch: {epoch:0>3d}, best_loss: {best_loss:.4f}'
        logger.info(log_info) if local_rank == 0 and total_rank == 0 else None

    if local_rank == 0 and total_rank == 0:
        if os.path.exists(
                os.path.join(checkpoint_dir, 'best_generator_model.pth')):
            os.rename(
                os.path.join(checkpoint_dir, 'best_generator_model.pth'),
                os.path.join(checkpoint_dir,
                             f'total_loss{best_loss:.3f}_generator_model.pth'))
        if os.path.exists(
                os.path.join(checkpoint_dir, 'best_discriminator_model.pth')):
            os.rename(
                os.path.join(checkpoint_dir, 'best_discriminator_model.pth'),
                os.path.join(
                    checkpoint_dir,
                    f'total_loss{best_loss:.3f}_discriminator_model.pth'))

    log_info = f'train done. train time: {train_time:.3f} hours, best_loss: {best_loss:.4f}'
    logger.info(log_info) if local_rank == 0 and total_rank == 0 else None

    torch.distributed.destroy_process_group()

    return


if __name__ == "__main__":
    main()
