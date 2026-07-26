import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import ILSVRC2012_path

from SimpleGeneration.image_tokenizer.models.vqgan import VQ_8_codebook_16384
from SimpleGeneration.image_tokenizer.models.patchgandiscriminator import PatchGANDiscriminator
from SimpleGeneration.image_tokenizer.losses import VQGANLoss
from SimpleGeneration.image_tokenizer.datasets.ilsvrc2012dataset import ILSVRC2012Dataset
from SimpleGeneration.image_tokenizer.common import Opencv2PIL, RandomCropForVQGAN, TorchRandomHorizontalFlip, TorchMeanStdNormalize, ClassificationCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    # 256 or 512
    network = 'VQ_8_codebook_16384'
    input_image_size = 256

    generator_model = VQ_8_codebook_16384(**{
        'use_gradient_checkpoint': True,
    })
    # load total pretrained model or not
    trained_generator_model_path = ''
    load_state_dict(trained_generator_model_path, generator_model)

    discriminator_model = PatchGANDiscriminator(inplanes=3,
                                                layer_nums=3,
                                                planes=64)
    # load total pretrained model or not
    trained_discriminator_model_path = ''
    load_state_dict(trained_discriminator_model_path, discriminator_model)

    lpips_pretrained_path = '/root/code/SimpleGeneration_pytorch_training_examples/SimpleGeneration/image_tokenizer/models/cache/vgg.pth'
    discriminator_loss_start_epoch = 3
    criterion = VQGANLoss(discriminator_model=discriminator_model,
                          lpips_pretrained_path=lpips_pretrained_path,
                          reconstruction_weight=1.0,
                          perceptual_weight=1.0,
                          generator_adversarial_weight=0.5,
                          discriminator_weight=0.5)

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

    generator_optimizer = (
        'Muon',
        {
            'lr': 1e-4,
            'weight_decay': 0,
            'exclude_muon_layer_name_list': [],
        },
    )

    generator_scheduler = (
        'CosineLR',
        {
            'warm_up_epochs': 0,
            'min_lr': 5e-6,
        },
    )

    discriminator_optimizer = (
        'Muon',
        {
            'lr': 1e-4,
            'weight_decay': 0,
            'exclude_muon_layer_name_list': [],
        },
    )

    discriminator_scheduler = (
        'CosineLR',
        {
            'warm_up_epochs': 0,
            'min_lr': 5e-6,
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

    use_ema_model = True
    generator_ema_model_decay = 0.9999
    generator_ema_model_tau = 2000
    discriminator_ema_model_decay = 0.9999
    discriminator_ema_model_tau = 2000

    # ZeRO stage: 0 (equivalent to DDP), 1, 2, 3
    deepspeed_zero_stage = 2
    # ZeRO-Offload: offload optimizer states (and params for stage 3) to CPU
    deepspeed_offload = False
