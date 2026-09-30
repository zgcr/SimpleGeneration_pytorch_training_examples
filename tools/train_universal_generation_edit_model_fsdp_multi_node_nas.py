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

import numpy as np

import torch
from torch.utils.data import DataLoader

from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict, get_optimizer_state_dict, set_optimizer_state_dict, StateDictOptions
from torch.distributed.device_mesh import init_device_mesh

from SimpleGeneration.universal_generation_edit.t2i_common import T2IBucketBatchSampler
from SimpleGeneration.universal_generation_edit.ti2i_common import TI2IBucketBatchSampler
from SimpleGeneration.universal_generation_edit.mix_common import MixBucketBatchSampler

from tools.universal_generation_edit_model_fsdp_scripts import train_universal_generation_edit_model_fsdp
from tools.utils import get_logger, set_seed, worker_seed_init_fn, Scheduler
from tools.muon_optimizer_fsdp import MuonAdamWFSDP, MuonSGDFSDP


def build_fsdp2_training_mode(config, model):
    device_mesh = init_device_mesh('cuda', (config.gpus_num, ))

    # The AE is the only module moved by hand, since `fully_shard` never sees
    # it and would otherwise leave it on the CPU.
    model.ae = model.ae.cuda()

    # All-gather every shard as `torch.bfloat16` for the matmuls, reduce the
    # gradients in float32, and keep the float32 master weights the optimizer
    # updates. The compute dtype is imported rather than written again here
    # because the model casts `ctx` / `x_t` / `pixel_values` to that very value by
    # hand; two independent literals could silently drift apart.
    FSDP2_MP_POLICY = MixedPrecisionPolicy(param_dtype=torch.bfloat16,
                                           reduce_dtype=torch.float32)

    fsdp_kwargs = {
        'mesh': device_mesh,
        'mp_policy': FSDP2_MP_POLICY,
        'reshard_after_forward': config.fsdp_reshard_after_forward,
    }

    # ---- denoise DiT: one group per block ----
    denoise_model = model.denoise_model
    for per_stack_name in [
            'noise_refiner',
            'ref_image_refiner',
            'context_refiner',
            'single_blocks',
            'double_blocks',
            'mix_blocks',
    ]:
        # The three denoise-DiT variants (SingleStream / DoubleStream /
        # MixStream) each carry a different subset of these stacks, so the
        # ones a given variant does not define are simply skipped.
        if not hasattr(denoise_model, per_stack_name):
            continue
        for per_block in getattr(denoise_model, per_stack_name):
            fully_shard(per_block, **fsdp_kwargs)

    # ---- VLM: one group per text decoder layer and per vision block ----
    # Reached through the peft wrapper: `get_peft_model` leaves the original
    # module tree in place under `base_model.model`, so these are the very
    # layers `encode_condition` runs.
    vlm_model = model.vlm.base_model.model.model
    for per_layer in vlm_model.language_model.layers:
        fully_shard(per_layer, **fsdp_kwargs)
    for per_block in vlm_model.visual.blocks:
        fully_shard(per_block, **fsdp_kwargs)

    # ---- one root group per float32 subtree ----
    # Each one claims what its block groups above did not: for the DiT the
    # img_in / txt_in / time_in / final_layer, for the VLM the embed_tokens /
    # lm_head tied weight, the vision patch embed and the mergers. Sharding
    # these two subtrees rather than `model` is what keeps the bf16 AE out of
    # every group.
    fully_shard(denoise_model, **fsdp_kwargs)
    fully_shard(vlm_model, **fsdp_kwargs)

    # ---- restore the embedding / lm_head weight tie ----
    # `fully_shard` rebinds `embed_tokens.weight` to a sharded DTensor, which
    # drops the tie `tie_word_embeddings=True` had set up and strands a dense
    # float32 copy on `lm_head.weight`. Pointing lm_head back at the shard
    # frees that copy and keeps both keys of the saved state_dict sharing one
    # tensor, exactly as the DDP path produces.
    model.vlm.base_model.model.lm_head.weight = vlm_model.language_model.embed_tokens.weight

    return model


def build_fsdp2_optimizer(config, model):
    optimizer_name = config.optimizer[0]
    optimizer_parameters = config.optimizer[1]
    assert optimizer_name in ['SGD', 'AdamW', 'MuonAdamWFSDP',
                              'MuonSGDFSDP'], 'Unsupported optimizer!'

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
            if param.ndim <= 1 or any(no_weight_decay_layer_name in name
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

    elif optimizer_name == 'MuonAdamWFSDP':
        # Note: MuonAdamWFSDP uses unified lr for all parameters.
        # Per-layer lr settings from optimizer_parameters are not applied.
        # MuonAdamWFSDP optimizer don't support sub_layer_lr/sub_layer_weight_decay

        exclude_muon_layer_name_list = [
            'position_encoding',
            'cls_token',
            'patch_embedding',
            'embed',
            'lm_head',
            'merger',
        ]
        if 'exclude_muon_layer_name_list' in optimizer_parameters.keys(
        ) and isinstance(optimizer_parameters['exclude_muon_layer_name_list'],
                         list):
            exclude_muon_layer_name_list = exclude_muon_layer_name_list + optimizer_parameters[
                'exclude_muon_layer_name_list']

        # Separate parameters into muon_params, adamw_params and adamw_nowd_params
        muon_param_list, muon_param_names = [], []
        adamw_param_list, adamw_param_names = [], []
        adamw_nowd_param_list, adamw_nowd_param_names = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Muon is used for 2D parameters that are not in exclude list.
            # `param.ndim` reads the GLOBAL rank of a DTensor, not the rank of
            # the local shard, so this split is the same one the non-sharded
            # build_optimizer produces.
            use_muon = (
                param.ndim >= 2
                and not any(exclude_name in name.lower()
                            for exclude_name in exclude_muon_layer_name_list))

            if use_muon:
                muon_param_list.append(param)
                muon_param_names.append(name)
            elif global_weight_decay is False and (param.ndim <= 1 or any(
                    no_weight_decay_layer_name in name
                    for no_weight_decay_layer_name in
                    no_weight_decay_layer_name_list)):
                adamw_nowd_param_list.append(param)
                adamw_nowd_param_names.append(name)
            else:
                adamw_param_list.append(param)
                adamw_param_names.append(name)

        # Create summary for model_layer_weight_decay_list
        model_layer_weight_decay_list = []
        if len(muon_param_names) > 0:
            model_layer_weight_decay_list.append({
                'name': muon_param_names,
                'optimizer': 'MuonAdamWFSDP(Muon)',
                'lr': lr,
                'weight_decay': weight_decay,
            })
        if len(adamw_param_names) > 0:
            model_layer_weight_decay_list.append({
                'name': adamw_param_names,
                'optimizer': 'MuonAdamWFSDP(AdamW)',
                'lr': lr,
                'weight_decay': weight_decay,
            })
        if len(adamw_nowd_param_names) > 0:
            model_layer_weight_decay_list.append({
                'name': adamw_nowd_param_names,
                'optimizer': 'MuonAdamWFSDP(AdamW)',
                'lr': lr,
                'weight_decay': 0.,
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

        return MuonAdamWFSDP(
            lr=lr,
            wd=weight_decay,
            muon_params=muon_param_list,
            adamw_params=adamw_param_list,
            adamw_nowd_params=adamw_nowd_param_list,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=(adamw_beta1, adamw_beta2),
            adamw_eps=adamw_eps), model_layer_weight_decay_list

    elif optimizer_name == 'MuonSGDFSDP':
        # Note: MuonSGDFSDP uses unified lr for all parameters.
        # Per-layer lr settings from optimizer_parameters are not applied.
        # MuonSGDFSDP optimizer don't support sub_layer_lr/sub_layer_weight_decay

        exclude_muon_layer_name_list = [
            'position_encoding',
            'cls_token',
            'patch_embedding',
            'embed',
            'lm_head',
            'merger',
        ]
        if 'exclude_muon_layer_name_list' in optimizer_parameters.keys(
        ) and isinstance(optimizer_parameters['exclude_muon_layer_name_list'],
                         list):
            exclude_muon_layer_name_list = exclude_muon_layer_name_list + optimizer_parameters[
                'exclude_muon_layer_name_list']

        # Separate parameters into muon_params, sgd_params and sgd_nowd_params
        muon_param_list, muon_param_names = [], []
        sgd_param_list, sgd_param_names = [], []
        sgd_nowd_param_list, sgd_nowd_param_names = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Muon is used for 2D parameters that are not in exclude list.
            # `param.ndim` reads the GLOBAL rank of a DTensor, not the rank of
            # the local shard, so this split is the same one the non-sharded
            # build_optimizer produces.
            use_muon = (
                param.ndim >= 2
                and not any(exclude_name in name.lower()
                            for exclude_name in exclude_muon_layer_name_list))

            if use_muon:
                muon_param_list.append(param)
                muon_param_names.append(name)
            elif global_weight_decay is False and (param.ndim <= 1 or any(
                    no_weight_decay_layer_name in name
                    for no_weight_decay_layer_name in
                    no_weight_decay_layer_name_list)):
                sgd_nowd_param_list.append(param)
                sgd_nowd_param_names.append(name)
            else:
                sgd_param_list.append(param)
                sgd_param_names.append(name)

        # Create summary for model_layer_weight_decay_list
        model_layer_weight_decay_list = []
        if len(muon_param_names) > 0:
            model_layer_weight_decay_list.append({
                'name': muon_param_names,
                'optimizer': 'MuonSGDFSDP(Muon)',
                'lr': lr,
                'weight_decay': weight_decay,
            })
        if len(sgd_param_names) > 0:
            model_layer_weight_decay_list.append({
                'name': sgd_param_names,
                'optimizer': 'MuonSGDFSDP(SGD)',
                'lr': lr,
                'weight_decay': weight_decay,
            })
        if len(sgd_nowd_param_names) > 0:
            model_layer_weight_decay_list.append({
                'name': sgd_nowd_param_names,
                'optimizer': 'MuonSGDFSDP(SGD)',
                'lr': lr,
                'weight_decay': 0.,
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

        return MuonSGDFSDP(
            lr=lr,
            wd=weight_decay,
            muon_params=muon_param_list,
            sgd_params=sgd_param_list,
            sgd_nowd_params=sgd_nowd_param_list,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            sgd_momentum=sgd_momentum,
            sgd_nesterov=sgd_nesterov), model_layer_weight_decay_list


def save_fsdp2_checkpoint(model, optimizer, save_dir, save_meta_dict):
    state_dict = {
        'model':
        get_model_state_dict(model),
        'optimizer':
        get_optimizer_state_dict(
            model,
            optimizer,
            options=StateDictOptions(flatten_optimizer_state_dict=True)),
    }
    torch.distributed.checkpoint.save(state_dict, checkpoint_id=save_dir)

    if torch.distributed.get_rank() == 0:
        torch.save(save_meta_dict, f'{save_dir}_meta.pth')

    return


def load_fsdp2_checkpoint(model, optimizer, save_dir):
    state_dict = {
        'model':
        get_model_state_dict(model),
        'optimizer':
        get_optimizer_state_dict(
            model,
            optimizer,
            options=StateDictOptions(flatten_optimizer_state_dict=True)),
    }
    torch.distributed.checkpoint.load(state_dict, checkpoint_id=save_dir)

    set_model_state_dict(model, state_dict['model'])
    set_optimizer_state_dict(
        model,
        optimizer,
        state_dict['optimizer'],
        options=StateDictOptions(flatten_optimizer_state_dict=True))

    save_meta_dict = torch.load(f'{save_dir}_meta.pth',
                                map_location=torch.device('cpu'),
                                weights_only=True)

    return save_meta_dict


def save_fsdp2_model_state_dict(model, save_path, config):
    state_dict = get_model_state_dict(model,
                                      options=StateDictOptions(
                                          full_state_dict=True,
                                          cpu_offload=True))

    if config.local_rank == 0 and config.total_rank == 0:
        torch.save(state_dict, save_path)

    return


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch Universal Generation Edit FSDP2 Training')
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
    resume_model = os.path.join(checkpoint_dir, 'latest')
    config.checkpoint_dir = checkpoint_dir
    config.gpus_type = torch.cuda.get_device_name()
    config.gpus_num = torch.cuda.device_count()

    local_rank = int(os.environ['LOCAL_RANK'])
    config.local_rank = local_rank
    # start init process
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # 获取total_rank
    total_rank = torch.distributed.get_rank()
    config.total_rank = total_rank

    set_seed(config.seed + total_rank)

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
                                global_rank=total_rank,
                                seed=config.seed)
    # image_bucket_list columns:
    #   T2I : [bucket_ratio, bucket_index]
    #   TI2I: [bucket_ratio, bucket_index, reference_image_num]
    #   MIX : t2i_image_bucket_list and ti2i_image_bucket_list are kept apart,
    #         each one keeping its own column layout listed above
    assert config.task_type in ['T2I', 'TI2I', 'MIX']
    if config.task_type == 'T2I':
        image_bucket_list = config.train_dataset.image_bucket_list
        train_batch_sampler = T2IBucketBatchSampler(
            image_bucket_list[:, 1].astype(np.int32),
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
            seed=config.seed)
    elif config.task_type == 'TI2I':
        image_bucket_list = config.train_dataset.image_bucket_list
        train_batch_sampler = TI2IBucketBatchSampler(
            image_bucket_list[:, 1].astype(np.int32),
            image_bucket_list[:, 2].astype(np.int32),
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
            seed=config.seed)
    elif config.task_type == 'MIX':
        t2i_image_bucket_list = config.train_dataset.t2i_image_bucket_list
        ti2i_image_bucket_list = config.train_dataset.ti2i_image_bucket_list
        train_batch_sampler = MixBucketBatchSampler(
            t2i_image_bucket_list[:, 1].astype(np.int32),
            ti2i_image_bucket_list[:, 1].astype(np.int32),
            ti2i_image_bucket_list[:, 2].astype(np.int32),
            batch_size=batch_size,
            choose_batch_task_type_prob=config.choose_batch_task_type_prob,
            drop_last=True,
            shuffle=True,
            seed=config.seed)
    train_loader = DataLoader(config.train_dataset,
                              batch_sampler=train_batch_sampler,
                              pin_memory=True,
                              num_workers=num_workers,
                              collate_fn=config.train_collater,
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

    # The model is NOT moved to the GPU here, unlike the DDP launcher:
    # `fully_shard` copies each parameter to the device as it shards it, so
    # leaving the module on the CPU keeps the peak at (already-sharded weights
    # + one unsharded block) instead of the whole unsharded model.
    model = config.model
    train_criterion = config.train_criterion.cuda()

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

    # Sharding comes BEFORE the optimizer is built, the reverse of the DDP
    # launcher's order: `fully_shard` REPLACES every nn.Parameter with a
    # sharded DTensor, so an optimizer built first would hold references to the
    # dense tensors that no longer belong to the model and would update nothing.
    model = build_fsdp2_training_mode(config, model)

    optimizer, model_layer_weight_decay_list = build_fsdp2_optimizer(
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

    scheduler = Scheduler(config, optimizer)

    start_epoch, train_time = 1, 0
    best_loss, train_loss = 1e9, 0
    if os.path.exists(resume_model):
        checkpoint = load_fsdp2_checkpoint(model, optimizer, resume_model)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        used_time = checkpoint['time']
        train_time += used_time

        best_loss, train_loss, lr = checkpoint['best_loss'], checkpoint[
            'train_loss'], checkpoint['lr']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch:0>3d}, used_time: {used_time:.3f} hours, best_loss: {best_loss:.4f}, lr: {lr:.6f}'
        logger.info(log_info) if local_rank == 0 and total_rank == 0 else None

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
        # Compiling AFTER sharding is what FSDP2 supports: each sharded block
        # is compiled on its own, so no `_orig_mod` indirection is introduced
        # at the root and the save / `model.ae` accesses stay unchanged.
        model = torch.compile(model, **config.compile_params)

    for epoch in range(start_epoch, config.epochs + 1):
        per_epoch_start_time = time.time()

        log_info = f'epoch {epoch:0>3d} lr: {scheduler.current_lr:.6f}'
        logger.info(log_info) if local_rank == 0 and total_rank == 0 else None

        torch.cuda.empty_cache()

        train_batch_sampler.set_epoch(epoch)
        train_loss = train_universal_generation_edit_model_fsdp(
            train_loader, model, train_criterion, optimizer, scheduler, epoch,
            logger, config)
        log_info = f'train: epoch {epoch:0>3d}, train_loss: {train_loss:.4f}'
        logger.info(log_info) if local_rank == 0 and total_rank == 0 else None

        torch.cuda.empty_cache()

        train_time += (time.time() - per_epoch_start_time) / 3600

        if epoch % config.save_interval == 0 or epoch == config.epochs:
            save_fsdp2_model_state_dict(
                model, os.path.join(checkpoint_dir,
                                    f'epoch_{epoch}_model.pth'), config)

        # save best loss model and each epoch checkpoint
        if train_loss < best_loss:
            best_loss = train_loss
            save_fsdp2_model_state_dict(
                model, os.path.join(checkpoint_dir, 'best.pth'), config)

        save_fsdp2_checkpoint(
            model, optimizer, resume_model, {
                'epoch': epoch,
                'time': train_time,
                'best_loss': best_loss,
                'train_loss': train_loss,
                'lr': scheduler.current_lr,
                'scheduler_state_dict': scheduler.state_dict(),
            })

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
