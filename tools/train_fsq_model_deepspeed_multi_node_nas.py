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
from tools.utils import (get_logger, set_seed, worker_seed_init_fn, Scheduler,
                         DeepSpeedEmaModel)


def build_param_groups(config, model):
    """Build parameter groups for DeepSpeed optimizer.
    For AdamW: differentiate weight decay (1D params and specified layers get 0).
    For Muon: pass all trainable params; DeepSpeed native Muon handles split.
    Returns (model_params_weight_decay_list, model_layer_weight_decay_list).
    """
    optimizer_name = config.optimizer[0]
    optimizer_parameters = config.optimizer[1]
    assert optimizer_name in ['SGD', 'AdamW', 'Muon'], 'Unsupported optimizer!'

    lr = optimizer_parameters['lr']
    weight_decay = optimizer_parameters['weight_decay']

    # For Muon, DeepSpeed 0.19.3 native implementation requires each
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
            # DeepSpeed 0.19.3 engine.py requires `param.use_muon`
            # attribute on every parameter. Must match the logic in
            # deepspeed.set_optimizer_flags() (called inside
            # deepspeed.initialize()) which excludes params whose name
            # contains "embed" or "lm_head".
            if param.ndim >= 2 and not any(
                    keyword in name.lower()
                    for keyword in ("embed", "lm_head")):
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
        ds_config["fp16"] = {
            "enabled": False,
        }
        ds_config["bf16"] = {
            "enabled": False,
        }

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

    # For deepspeed==0.19.3, Muon optimizer requires reduce_scatter=False for ZeRO stage 1/2/3.
    # Muon's Newton-Schulz orthogonalization is a whole-matrix operation that needs the full
    # reduced gradient. With reduce_scatter=True (default), each rank only receives its own
    # partition slice after reduce-scatter, causing cross-partition parameters to get incorrect
    # orthogonalized updates (rank-divergent). ZeRO-3 already has a hard ValueError guard;
    # ZeRO-1/2 silently produces wrong results (see: https://github.com/deepspeedai/DeepSpeed/pull/8090).
    # This applies regardless of ns_method ("standard" or "gram").
    # ZeRO stage 0 does not use the ZeRO optimizer wrapper, so reduce_scatter is irrelevant.
    if optimizer_name == 'Muon' and config.deepspeed_zero_stage in [1, 2, 3]:
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
        # DeepSpeed 0.19.3 engine.py _configure_basic_optimizer() only
        # recognizes these keys for Muon param groups:
        #   muon group:  ["lr", "momentum", "weight_decay", "muon_lr", "ns_method"]
        #   adamw group: ["lr", "betas", "eps", "weight_decay", "adam_lr"]
        # Keys like "wd", "nesterov", "ns_steps", "adamw_betas", "adamw_eps"
        # are NOT recognized and will be silently ignored.
        # ns_method: Newton-Schulz orthogonalization method.
        #   "gram"     -> Gram Newton-Schulz (default), ~2x faster on rectangular
        #                 matrices, uses fp16.
        #   "standard" -> original Newton-Schulz quintic iteration, uses bf16.
        ds_config["optimizer"] = {
            "type": "Muon",
            "params": {
                "lr":
                optimizer_parameters['lr'],
                "weight_decay":
                optimizer_parameters['weight_decay'],
                "momentum":
                optimizer_parameters.get('momentum', 0.95),
                "ns_method":
                optimizer_parameters.get('ns_method', 'standard'),
                "betas": [
                    optimizer_parameters.get('adamw_beta1', 0.9),
                    optimizer_parameters.get('adamw_beta2', 0.999)
                ],
                "eps":
                optimizer_parameters.get('adamw_eps', 1e-08),
            }
        }

    return ds_config


def get_model_state_dict(model_engine, config):
    """Get full model state dict for saving.
    For ZeRO-3, all ranks must call (GatheredParameters is collective),
    but only rank 0 returns a non-None dict.
    """
    if config.use_compile:
        module = model_engine.module._orig_mod
    else:
        module = model_engine.module

    if config.deepspeed_zero_stage == 3:
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
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model_params_weight_decay_list,
        config=ds_config)

    # Build scheduler after DeepSpeed creates the optimizer. The scheduler
    # adjusts LR via optimizer.param_groups which DeepSpeed's optimizer
    # wrapper correctly exposes and delegates to the underlying optimizer.
    scheduler = Scheduler(config, optimizer)

    # Create EMA model after deepspeed.initialize().
    # For ZeRO-3, param.ds_tensor (local shard) is only available after init.
    # For ZeRO-0/1/2, params are still full after init.
    if config.use_ema_model:
        ema_model = DeepSpeedEmaModel(model_engine,
                                      config,
                                      decay=config.ema_model_decay,
                                      tau=config.ema_model_tau)
        config.ema_model = ema_model
        log_info = f'EMA model created with decay={config.ema_model_decay}, tau={config.ema_model_tau}, zero_stage={config.deepspeed_zero_stage}'
        logger.info(log_info) if local_rank == 0 and total_rank == 0 else None

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

            if config.use_ema_model:
                if 'ema_model_state' in client_state:
                    config.ema_model.load_state_dict(
                        client_state['ema_model_state'])

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

        # Merge save_interval and is_best saving to avoid redundant
        # GatheredParameters calls under ZeRO-3 (each call triggers
        # all-gather across all ranks for all parameters).
        need_save_epoch = (epoch % config.save_interval == 0
                           or epoch == config.epochs)
        need_save_best = is_best

        if need_save_epoch or need_save_best:
            if config.use_ema_model:
                # EMA enabled: only save EMA model, skip training model
                # For ZeRO-3, get_ema_model_state_dict uses all_gather
                # (collective op), so all ranks must call it.
                save_model = config.ema_model.get_ema_model_state_dict(
                    model_engine)
                if local_rank == 0 and total_rank == 0:
                    if need_save_epoch:
                        torch.save(
                            save_model,
                            os.path.join(checkpoint_dir,
                                         f'epoch_{epoch}_model.pth'))
                    if need_save_best:
                        torch.save(save_model,
                                   os.path.join(checkpoint_dir, 'best.pth'))
            else:
                # EMA disabled: save training model
                if config.deepspeed_zero_stage == 3:
                    # ZeRO-3: all ranks must participate in GatheredParameters
                    save_model = get_model_state_dict(model_engine, config)
                    if local_rank == 0 and total_rank == 0 and save_model is not None:
                        if need_save_epoch:
                            torch.save(
                                save_model,
                                os.path.join(checkpoint_dir,
                                             f'epoch_{epoch}_model.pth'))
                        if need_save_best:
                            torch.save(
                                save_model,
                                os.path.join(checkpoint_dir, 'best.pth'))
                else:
                    # ZeRO-0/1/2: only global rank 0 needs to call state_dict
                    if local_rank == 0 and total_rank == 0:
                        save_model = get_model_state_dict(model_engine, config)
                        if need_save_epoch:
                            torch.save(
                                save_model,
                                os.path.join(checkpoint_dir,
                                             f'epoch_{epoch}_model.pth'))
                        if need_save_best:
                            torch.save(
                                save_model,
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
        # Include EMA state in client_state for resume
        if config.use_ema_model:
            client_state['ema_model_state'] = config.ema_model.state_dict()

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
