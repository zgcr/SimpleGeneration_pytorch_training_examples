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
from torch.utils.data import DataLoader

from tools.autoencoder_scripts import train_flux_autoencoder_model
from tools.utils import (get_logger, set_seed, worker_seed_init_fn,
                         build_training_mode)

from tools.muon_optimizer import MuonAdamW, MuonSGD


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
            param_group['lr'] for param_group in optimizer.param_groups
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

            # Warmup phase: only when warm_up_epochs > 0 and epoch < warm_up_epochs
            if self.warm_up_epochs > 0 and epoch < self.warm_up_epochs:
                param_group_current_lr = (
                    epoch / self.warm_up_epochs) * param_group_init_lr
            else:
                if self.scheduler_name == 'MultiStepLR':
                    param_group_current_lr = gamma**len([
                        m for m in milestones if m <= epoch
                    ]) * param_group_init_lr
                elif self.scheduler_name == 'CosineLR':
                    param_group_current_lr = 0.5 * (math.cos(
                        (epoch - self.warm_up_epochs) /
                        (self.epochs - self.warm_up_epochs) * math.pi) + 1) * (
                            param_group_init_lr - min_lr) + min_lr
                elif self.scheduler_name == 'PolyLR':
                    param_group_current_lr = (
                        (1 - (epoch - self.warm_up_epochs) /
                         (self.epochs - self.warm_up_epochs))**
                        power) * (param_group_init_lr - min_lr) + min_lr

            param_group['lr'] = param_group_current_lr

        # Update self.current_lr for logging (using the global base lr)
        if self.warm_up_epochs > 0 and epoch < self.warm_up_epochs:
            self.current_lr = (epoch / self.warm_up_epochs) * self.lr
        else:
            if self.scheduler_name == 'MultiStepLR':
                self.current_lr = gamma**len(
                    [m for m in milestones if m <= epoch]) * self.lr
            elif self.scheduler_name == 'CosineLR':
                self.current_lr = 0.5 * (math.cos(
                    (epoch - self.warm_up_epochs) /
                    (self.epochs - self.warm_up_epochs) * math.pi) +
                                         1) * (self.lr - min_lr) + min_lr
            elif self.scheduler_name == 'PolyLR':
                self.current_lr = ((1 - (epoch - self.warm_up_epochs) /
                                    (self.epochs - self.warm_up_epochs))**
                                   power) * (self.lr - min_lr) + min_lr

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


def build_optimizer(config, model, model_type):
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

    assert optimizer_name in ['SGD', 'AdamW', 'MuonAdamW',
                              'MuonSGD'], 'Unsupported optimizer!'

    lr = optimizer_parameters['lr']
    weight_decay = optimizer_parameters['weight_decay']

    # if global_weight_decay = False,set 1d parms weight decay = 0.
    global_weight_decay = True if 'global_weight_decay' not in optimizer_parameters.keys(
    ) else optimizer_parameters['global_weight_decay']

    # if global_weight_decay = True,no_weight_decay_layer_name_list can't be set.
    no_weight_decay_layer_name_list = []
    if 'no_weight_decay_layer_name_list' in optimizer_parameters.keys(
    ) and isinstance(optimizer_parameters['no_weight_decay_layer_name_list'],
                     list):
        no_weight_decay_layer_name_list = optimizer_parameters[
            'no_weight_decay_layer_name_list']

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
                        optimizer_parameters['sub_layer_weight_decay'], dict):
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

    assert len(param_layer_name_list) == len(param_layer_weight_dict) == len(
        param_layer_decay_dict) == len(param_layer_lr_dict)

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

    if optimizer_name == 'SGD':
        momentum = 0.9 if 'momentum' not in optimizer_parameters.keys(
        ) else optimizer_parameters['momentum']
        nesterov = False if 'nesterov' not in optimizer_parameters.keys(
        ) else optimizer_parameters['nesterov']
        return torch.optim.SGD(
            model_params_weight_decay_list,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov), model_layer_weight_decay_list

    elif optimizer_name == 'MuonSGD':
        # Note: MuonSGD uses unified lr and wd for all parameters.
        # Per-layer lr/wd settings from optimizer_parameters are not applied.
        # MuonSGD optimizer don't support global_weight_decay
        # MuonSGD optimizer don't support no_weight_decay_layer_name_list
        # MuonSGD optimizer don't support sub_layer_lr/sub_layer_weight_decay

        exclude_muon_layer_name_list = [
            'position_encoding',
            'cls_token',
            'patch_embedding',
        ]
        if 'exclude_muon_layer_name_list' in optimizer_parameters.keys(
        ) and isinstance(optimizer_parameters['exclude_muon_layer_name_list'],
                         list):
            exclude_muon_layer_name_list = exclude_muon_layer_name_list + optimizer_parameters[
                'exclude_muon_layer_name_list']

        # Separate parameters into muon_params and sgd_params
        muon_param_list, muon_param_names = [], []
        sgd_param_list, sgd_param_names = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Muon is used for 2D parameters that are not in exclude list
            use_muon = (
                param.ndim >= 2
                and not any(exclude_name in name
                            for exclude_name in exclude_muon_layer_name_list))

            if use_muon:
                muon_param_list.append(param)
                muon_param_names.append(name)
            else:
                sgd_param_list.append(param)
                sgd_param_names.append(name)

        # Create summary for model_layer_weight_decay_list
        model_layer_weight_decay_list = []
        if len(muon_param_names) > 0:
            model_layer_weight_decay_list.append({
                'name': muon_param_names,
                'optimizer': 'MuonSGD(Muon)',
                'lr': lr,
                'weight_decay': weight_decay,
            })
        if len(sgd_param_names) > 0:
            model_layer_weight_decay_list.append({
                'name': sgd_param_names,
                'optimizer': 'MuonSGD(SGD)',
                'lr': lr,
                'weight_decay': weight_decay,
            })

        momentum = 0.95 if 'momentum' not in optimizer_parameters.keys(
        ) else optimizer_parameters['momentum']
        nesterov = True if 'nesterov' not in optimizer_parameters.keys(
        ) else optimizer_parameters['nesterov']
        ns_steps = 5 if 'ns_steps' not in optimizer_parameters.keys(
        ) else optimizer_parameters['ns_steps']

        sgd_momentum = 0.9 if 'sgd_momentum' not in optimizer_parameters.keys(
        ) else optimizer_parameters['sgd_momentum']
        sgd_nesterov = False if 'sgd_nesterov' not in optimizer_parameters.keys(
        ) else optimizer_parameters['sgd_nesterov']

        return MuonSGD(
            lr=lr,
            wd=weight_decay,
            muon_params=muon_param_list,
            sgd_params=sgd_param_list,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            sgd_momentum=sgd_momentum,
            sgd_nesterov=sgd_nesterov), model_layer_weight_decay_list

    elif optimizer_name == 'AdamW':
        beta1 = 0.9 if 'beta1' not in optimizer_parameters.keys(
        ) else optimizer_parameters['beta1']
        beta2 = 0.999 if 'beta2' not in optimizer_parameters.keys(
        ) else optimizer_parameters['beta2']
        eps = 1e-08 if 'eps' not in optimizer_parameters.keys(
        ) else optimizer_parameters['eps']
        return torch.optim.AdamW(model_params_weight_decay_list,
                                 lr=lr,
                                 betas=(beta1, beta2),
                                 eps=eps), model_layer_weight_decay_list

    elif optimizer_name == 'MuonAdamW':
        # Note: MuonAdamW uses unified lr and wd for all parameters.
        # Per-layer lr/wd settings from optimizer_parameters are not applied.
        # MuonAdamW optimizer don't support global_weight_decay
        # MuonAdamW optimizer don't support no_weight_decay_layer_name_list
        # MuonAdamW optimizer don't support sub_layer_lr/sub_layer_weight_decay

        exclude_muon_layer_name_list = [
            'position_encoding',
            'cls_token',
            'patch_embedding',
        ]
        if 'exclude_muon_layer_name_list' in optimizer_parameters.keys(
        ) and isinstance(optimizer_parameters['exclude_muon_layer_name_list'],
                         list):
            exclude_muon_layer_name_list = exclude_muon_layer_name_list + optimizer_parameters[
                'exclude_muon_layer_name_list']

        # Separate parameters into muon_params and adamw_params
        muon_param_list, muon_param_names = [], []
        adamw_param_list, adamw_param_names = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Muon is used for 2D parameters that are not in exclude list
            use_muon = (
                param.ndim >= 2
                and not any(exclude_name in name
                            for exclude_name in exclude_muon_layer_name_list))

            if use_muon:
                muon_param_list.append(param)
                muon_param_names.append(name)
            else:
                adamw_param_list.append(param)
                adamw_param_names.append(name)

        # Create summary for model_layer_weight_decay_list
        model_layer_weight_decay_list = []
        if len(muon_param_names) > 0:
            model_layer_weight_decay_list.append({
                'name': muon_param_names,
                'optimizer': 'MuonAdamW(Muon)',
                'lr': lr,
                'weight_decay': weight_decay,
            })
        if len(adamw_param_names) > 0:
            model_layer_weight_decay_list.append({
                'name': adamw_param_names,
                'optimizer': 'MuonAdamW(AdamW)',
                'lr': lr,
                'weight_decay': weight_decay,
            })

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

        return MuonAdamW(lr=lr,
                         wd=weight_decay,
                         muon_params=muon_param_list,
                         adamw_params=adamw_param_list,
                         momentum=momentum,
                         nesterov=nesterov,
                         ns_steps=ns_steps,
                         adamw_betas=(adamw_beta1, adamw_beta2),
                         adamw_eps=adamw_eps), model_layer_weight_decay_list


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch VQGAN Training')
    parser.add_argument(
        '--work-dir',
        type=str,
        help='path for get training config and saving log/models')

    return parser.parse_args()


def main():
    assert torch.cuda.is_available(), 'need gpu to train network!'
    torch.cuda.empty_cache()

    args = parse_args()
    sys.path.append(args.work_dir)
    from train_config import config
    log_dir = os.path.join(args.work_dir, 'log')
    checkpoint_dir = os.path.join(args.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    config.checkpoint_dir = checkpoint_dir
    config.gpus_type = torch.cuda.get_device_name()
    config.gpus_num = torch.cuda.device_count()

    set_seed(config.seed)

    local_rank = int(os.environ['LOCAL_RANK'])
    config.local_rank = local_rank
    # start init process
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

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

    generator_optimizer, generator_model_layer_weight_decay_list = build_optimizer(
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

    discriminator_optimizer, discriminator_model_layer_weight_decay_list = build_optimizer(
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

    generator_scheduler = Scheduler(config,
                                    generator_optimizer,
                                    model_type='generator_model')
    discriminator_scheduler = Scheduler(config,
                                        discriminator_optimizer,
                                        model_type='discriminator_model')

    generator_model, config.generator_ema_model, config.generator_scaler = build_training_mode(
        config, generator_model)
    discriminator_model, config.discriminator_ema_model, config.discriminator_scaler = build_training_mode(
        config, discriminator_model)

    start_epoch, train_time = 1, 0
    best_loss, train_loss = 1e9, 0
    if os.path.exists(resume_model):
        checkpoint = torch.load(resume_model,
                                map_location=torch.device('cpu'),
                                weights_only=True)
        generator_model.load_state_dict(
            checkpoint['generator_model_state_dict'])
        discriminator_model.load_state_dict(
            checkpoint['discriminator_model_state_dict'])
        generator_optimizer.load_state_dict(
            checkpoint['generator_optimizer_state_dict'])
        discriminator_optimizer.load_state_dict(
            checkpoint['discriminator_optimizer_state_dict'])
        generator_scheduler.load_state_dict(
            checkpoint['generator_scheduler_state_dict'])
        discriminator_scheduler.load_state_dict(
            checkpoint['discriminator_scheduler_state_dict'])

        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        used_time = checkpoint['time']
        train_time += used_time

        best_loss, train_loss, generator_lr, discriminator_lr = checkpoint[
            'best_loss'], checkpoint['train_loss'], checkpoint[
                'generator_lr'], checkpoint['discriminator_lr']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch:0>3d}, used_time: {used_time:.3f} hours, best_loss: {best_loss:.4f}, generator_lr: {generator_lr:.6f}, discriminator_lr: {discriminator_lr:.6f}'
        logger.info(log_info) if local_rank == 0 and total_rank == 0 else None

        if 'generator_ema_model_state_dict' in checkpoint.keys():
            config.generator_ema_model.ema_model.load_state_dict(
                checkpoint['generator_ema_model_state_dict'])
            config.generator_ema_model.updates = checkpoint[
                'generator_ema_model_updates']

        if 'discriminator_ema_model_state_dict' in checkpoint.keys():
            config.discriminator_ema_model.ema_model.load_state_dict(
                checkpoint['discriminator_ema_model_state_dict'])
            config.discriminator_ema_model.updates = checkpoint[
                'discriminator_ema_model_updates']

    # use torch 2.0 compile function
    config.compile_support = False
    log_info = f'using torch version:{torch.__version__}'
    logger.info(log_info) if local_rank == 0 and total_rank == 0 else None
    if re.match(r'2\.\d+\.\d+', torch.__version__):
        config.compile_support = True
        log_info = f'this torch version support torch.compile function.'
        logger.info(log_info) if local_rank == 0 and total_rank == 0 else None
    elif re.match(r'1\.\d+\.\d+', torch.__version__):
        log_info = f'this torch version unsupport torch.compile function.'
        logger.info(log_info) if local_rank == 0 and total_rank == 0 else None
    else:
        log_info = f'unsupport torch version:{torch.__version__}'
        logger.info(log_info) if local_rank == 0 and total_rank == 0 else None
        return

    config.use_compile = (config.compile_support and config.use_compile)
    if config.use_compile:
        # _orig_mod
        generator_model = torch.compile(generator_model,
                                        **config.compile_params)
        discriminator_model = torch.compile(discriminator_model,
                                            **config.compile_params)

    for epoch in range(start_epoch, config.epochs + 1):
        per_epoch_start_time = time.time()

        log_info = f'epoch {epoch:0>3d} generator_lr: {generator_scheduler.current_lr:.6f}'
        logger.info(log_info) if local_rank == 0 and total_rank == 0 else None
        log_info = f'epoch {epoch:0>3d} discriminator_lr: {discriminator_scheduler.current_lr:.6f}'
        logger.info(log_info) if local_rank == 0 and total_rank == 0 else None

        torch.cuda.empty_cache()

        train_sampler.set_epoch(epoch)
        generator_loss, discriminator_loss, total_loss = train_flux_autoencoder_model(
            train_loader, generator_model, discriminator_model, criterion,
            generator_optimizer, discriminator_optimizer, generator_scheduler,
            discriminator_scheduler, epoch, logger, config)
        log_info = f'train: epoch {epoch:0>3d}, generator_loss: {generator_loss:.4f}, discriminator_loss: {discriminator_loss:.4f}, total_loss: {total_loss:.4f}'
        logger.info(log_info) if local_rank == 0 and total_rank == 0 else None

        train_loss = generator_loss

        torch.cuda.empty_cache()

        train_time += (time.time() - per_epoch_start_time) / 3600

        if epoch % config.save_interval == 0 or epoch == config.epochs:
            if local_rank == 0 and total_rank == 0:
                if config.use_ema_model:
                    save_generator_model = config.generator_ema_model.ema_model.module.state_dict(
                    )
                    save_discriminator_model = config.discriminator_ema_model.ema_model.module.state_dict(
                    )
                elif config.use_compile:
                    save_generator_model = generator_model._orig_mod.module.state_dict(
                    )
                    save_discriminator_model = discriminator_model._orig_mod.module.state_dict(
                    )
                else:
                    save_generator_model = generator_model.module.state_dict()
                    save_discriminator_model = discriminator_model.module.state_dict(
                    )

                torch.save(
                    save_generator_model,
                    os.path.join(checkpoint_dir,
                                 f'epoch_{epoch}_generator_model.pth'))
                torch.save(
                    save_discriminator_model,
                    os.path.join(checkpoint_dir,
                                 f'epoch_{epoch}_discriminator_model.pth'))

        if local_rank == 0 and total_rank == 0:
            # save best loss model and each epoch checkpoint
            if train_loss < best_loss:
                best_loss = train_loss
                if config.use_ema_model:
                    save_best_generator_model = config.generator_ema_model.ema_model.module.state_dict(
                    )
                    save_best_discriminator_model = config.discriminator_ema_model.ema_model.module.state_dict(
                    )
                elif config.use_compile:
                    save_best_generator_model = generator_model._orig_mod.module.state_dict(
                    )
                    save_best_discriminator_model = discriminator_model._orig_mod.module.state_dict(
                    )
                else:
                    save_best_generator_model = generator_model.module.state_dict(
                    )
                    save_best_discriminator_model = discriminator_model.module.state_dict(
                    )

                torch.save(
                    save_best_generator_model,
                    os.path.join(checkpoint_dir, 'best_generator_model.pth'))
                torch.save(
                    save_best_discriminator_model,
                    os.path.join(checkpoint_dir,
                                 'best_discriminator_model.pth'))

            if config.use_compile:
                save_checkpoint_generator_model = generator_model._orig_mod.state_dict(
                )
                save_checkpoint_discriminator_model = discriminator_model._orig_mod.state_dict(
                )
            else:
                save_checkpoint_generator_model = generator_model.state_dict()
                save_checkpoint_discriminator_model = discriminator_model.state_dict(
                )

            if config.use_ema_model:
                torch.save(
                    {
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
                        'generator_model_state_dict':
                        save_checkpoint_generator_model,
                        'discriminator_model_state_dict':
                        save_checkpoint_discriminator_model,
                        'generator_ema_model_state_dict':
                        config.generator_ema_model.ema_model.state_dict(),
                        'discriminator_ema_model_state_dict':
                        config.discriminator_ema_model.ema_model.state_dict(),
                        'generator_ema_model_updates':
                        config.generator_ema_model.updates,
                        'discriminator_ema_model_updates':
                        config.discriminator_ema_model.updates,
                        'generator_optimizer_state_dict':
                        generator_optimizer.state_dict(),
                        'discriminator_optimizer_state_dict':
                        discriminator_optimizer.state_dict(),
                        'generator_scheduler_state_dict':
                        generator_scheduler.state_dict(),
                        'discriminator_scheduler_state_dict':
                        discriminator_scheduler.state_dict(),
                    }, os.path.join(checkpoint_dir, 'latest.pth'))
            else:
                torch.save(
                    {
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
                        'generator_model_state_dict':
                        save_checkpoint_generator_model,
                        'discriminator_model_state_dict':
                        save_checkpoint_discriminator_model,
                        'generator_optimizer_state_dict':
                        generator_optimizer.state_dict(),
                        'discriminator_optimizer_state_dict':
                        discriminator_optimizer.state_dict(),
                        'generator_scheduler_state_dict':
                        generator_scheduler.state_dict(),
                        'discriminator_scheduler_state_dict':
                        discriminator_scheduler.state_dict(),
                    }, os.path.join(checkpoint_dir, 'latest.pth'))

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


if __name__ == '__main__':
    main()
