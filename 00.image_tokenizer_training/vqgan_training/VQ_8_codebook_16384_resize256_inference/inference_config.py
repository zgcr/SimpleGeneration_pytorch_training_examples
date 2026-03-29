import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from SimpleGeneration.image_tokenizer.models.vqgan import VQ_8_codebook_16384
from SimpleGeneration.image_tokenizer.common import load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    # 256 or 512
    network = 'VQ_8_codebook_16384'
    input_image_size = 256

    generator_model = VQ_8_codebook_16384(**{})
    # load total pretrained model or not
    trained_generator_model_path = ''
    load_state_dict(trained_generator_model_path, generator_model)

    seed = 0

    use_amp = False
    # torch.float16 or torch.bfloat16
    amp_type = torch.float16

    inference_image_dir = './test_images'
    save_image_dir = './test_images_result'
