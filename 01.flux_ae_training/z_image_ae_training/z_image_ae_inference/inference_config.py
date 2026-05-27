import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from SimpleGeneration.flux_autoencoder.models.z_image_autoencoder import AutoEncoder
from SimpleGeneration.flux_autoencoder.common import load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'z_image_ae'

    model = AutoEncoder(inplanes=3,
                        planes=128,
                        planes_mult=[1, 2, 4, 4],
                        res_block_nums=2,
                        z_planes=16,
                        out_planes=3,
                        scale_factor=0.3611,
                        shift_factor=0.1159,
                        logvar_init=0.0,
                        sample_z=False,
                        use_gradient_checkpoint=False)
    # load total pretrained model or not
    trained_model_path = '/root/autodl-tmp/pretrained_models/z_image_convert_from_pytorch_official_weights/z_image_vae_convert_from_pytorch_official_weight.pth'
    load_state_dict(trained_model_path, model)

    seed = 0

    use_amp = False
    # torch.float16 or torch.bfloat16
    amp_type = torch.float16

    inference_image_dir = './test_images'
    save_image_dir = './test_images_result'
