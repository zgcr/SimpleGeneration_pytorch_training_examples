import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import ILSVRC2012_path

from SimpleGeneration.image_tokenizer.models.fsq import FSQ_8_codebook_16K
from SimpleGeneration.image_tokenizer import losses
from SimpleGeneration.image_tokenizer.datasets.ilsvrc2012dataset import ILSVRC2012Dataset
from SimpleGeneration.image_tokenizer.common import Opencv2PIL, RandomCropForVQGAN, TorchRandomHorizontalFlip, TorchMeanStdNormalize, ClassificationCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    # 256 or 512
    network = 'FSQ_8_codebook_16K'
    input_image_size = 256

    model = FSQ_8_codebook_16K(**{
        'use_gradient_checkpoint': True,
    })
    # load total pretrained model or not
    trained_model_path = ''
    load_state_dict(trained_model_path, model)

    loss_list = [
        'ReconstructionL1Loss',
        'PerceptualLoss',
    ]
    loss_ratio = {
        'ReconstructionL1Loss': 1.0,
        'PerceptualLoss': 1.0,
    }
    lpips_pretrained_path = '/root/code/SimpleGeneration_pytorch_training_examples/SimpleGeneration/image_tokenizer/models/cache/vgg.pth'
    train_criterion = {}
    for loss_name in loss_list:
        if loss_name == 'PerceptualLoss':
            train_criterion[loss_name] = losses.__dict__[loss_name](
                **{
                    'lpips_pretrained_path': lpips_pretrained_path,
                })
        else:
            train_criterion[loss_name] = losses.__dict__[loss_name]()

    train_dataset = ILSVRC2012Dataset(
        root_dir=ILSVRC2012_path,
        set_name='train',
        transform=transforms.Compose([
            Opencv2PIL(),
            RandomCropForVQGAN(resize=input_image_size,
                               crop_fraction_range=[0.8, 1.0]),
            TorchRandomHorizontalFlip(prob=0.5),
            TorchMeanStdNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]))
    train_collater = ClassificationCollater()

    seed = 0
    # batch_size is total size
    batch_size = 64
    # num_workers is total workers
    num_workers = 8
    accumulation_steps = 1

    optimizer = (
        'AdamW',
        {
            'lr': 1e-4,
            'global_weight_decay': False,
            # if global_weight_decay = False
            # all bias, bn and other 1d params weight set to 0 weight decay
            'weight_decay': 0,
            'no_weight_decay_layer_name_list': [],
        },
    )

    scheduler = (
        'MultiStepLR',
        {
            'warm_up_epochs': 0,
            'gamma': 0.1,
            'milestones': [100],
        },
    )

    epochs = 40
    print_interval = 100
    save_interval = 10

    use_step_save_interval = False
    step_save_interval = 10000

    # torch.float16 or torch.bfloat16
    amp_type = torch.bfloat16

    sync_bn = False
    use_amp = True
    use_compile = False
    compile_params = {
        # 'default': optimizes for large models, low compile-time and no extra memory usage.
        # 'reduce-overhead': optimizes to reduce the framework overhead and uses some extra memory, helps speed up small models, model update may not correct.
        # 'max-autotune': optimizes to produce the fastest model, but takes a very long time to compile and may failed.
        'mode': 'default',
    }

    # ZeRO stage: 0 (equivalent to DDP), 1, 2, 3
    deepspeed_zero_stage = 0
    # ZeRO-Offload: offload optimizer states (and params for stage 3) to CPU
    deepspeed_offload = False
