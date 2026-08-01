import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from SimpleGeneration.flux_autoencoder.models.flux2_autoencoder import AutoEncoder
from SimpleGeneration.flux_autoencoder.common import load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'flux2_ae'

    model = AutoEncoder(inplanes=3,
                        planes=128,
                        planes_mult=[1, 2, 4, 4],
                        res_block_nums=2,
                        z_planes=32,
                        out_planes=3,
                        logvar_init=0.0,
                        sample_z=False,
                        use_gradient_checkpoint=False)
    # load total pretrained model or not
    trained_model_path = '/root/autodl-tmp/pretrained_models/flux2_convert_from_pytorch_official_weights/FLUX.2-dev-ae_convert_from_pytorch_official_weight.pth'
    load_state_dict(trained_model_path, model)

    seed = 0

    use_amp = False
    # torch.float16 or torch.bfloat16
    amp_type = torch.bfloat16

    inference_image_dir = './test_images'
    save_image_dir = './test_images_result'
