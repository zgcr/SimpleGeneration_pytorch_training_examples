import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import ILSVRC2012_path

from SimpleGeneration.flux_autoencoder.models.flux1_autoencoder import AutoEncoder
from SimpleGeneration.flux_autoencoder.models.discriminator import NLayerDiscriminator
from SimpleGeneration.flux_autoencoder.losses import FLUX1AELoss
from SimpleGeneration.flux_autoencoder.datasets.ilsvrc2012dataset import ILSVRC2012Dataset
from SimpleGeneration.flux_autoencoder.common import Opencv2PIL, FLUXResize, FLUXRandomPad, TorchRandomHorizontalFlip, TorchMeanStdNormalize, FLUXTrainCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    # 256 or 512
    network = 'flux1_ae'
    input_image_size = 256

    generator_model = AutoEncoder(inplanes=3,
                                  planes=128,
                                  planes_mult=[1, 2, 4, 4],
                                  res_block_nums=2,
                                  z_planes=16,
                                  out_planes=3,
                                  scale_factor=0.3611,
                                  shift_factor=0.1159,
                                  logvar_init=0.0,
                                  sample_z=True,
                                  use_gradient_checkpoint=True)

    # load total pretrained model or not
    trained_generator_model_path = '/root/autodl-tmp/pretrained_models/flux1_convert_from_pytorch_official_weights/FLUX.1-dev-ae_convert_from_pytorch_official_weight.pth'
    load_state_dict(trained_generator_model_path, generator_model)

    discriminator_model = NLayerDiscriminator(inplanes=3,
                                              layer_nums=3,
                                              planes=64)
    # load total pretrained model or not
    trained_discriminator_model_path = ''
    load_state_dict(trained_discriminator_model_path, discriminator_model)

    lpips_pretrained_path = '/root/code/SimpleGeneration_pytorch_training_examples/SimpleGeneration/flux_autoencoder/models/cache/vgg.pth'
    generator_loss_start_step = 0
    discriminator_loss_start_step = 10008
    criterion = FLUX1AELoss(discriminator_model=discriminator_model,
                            lpips_pretrained_path=lpips_pretrained_path,
                            kl_weight=0.000001,
                            reconstruction_weight=1.0,
                            perceptual_weight=1.0,
                            discriminator_weight=0.5)

    train_dataset = ILSVRC2012Dataset(
        root_dir=ILSVRC2012_path,
        set_name='train',
        transform=transforms.Compose([
            Opencv2PIL(),
            FLUXResize(resize=input_image_size),
            FLUXRandomPad(resize=input_image_size),
            TorchRandomHorizontalFlip(prob=0.5),
            TorchMeanStdNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]))
    train_collater = FLUXTrainCollater(resize=input_image_size)

    seed = 0
    # batch_size is total size
    batch_size = 64
    # num_workers is total workers
    num_workers = 8
    accumulation_steps = 1

    generator_optimizer = (
        'MuonAdamW',
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
        'MuonAdamW',
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
    ema_model_decay = 0.9999
    ema_model_tau = 2000
