import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import ILSVRC2012_path

from SimpleGeneration.image_tokenizer.models.vqgan import VQ_8_codebook_16384
from SimpleGeneration.image_tokenizer.datasets.ilsvrc2012dataset import ILSVRC2012Dataset
from SimpleGeneration.image_tokenizer.common import Opencv2PIL, CenterCropForVQGAN, TorchMeanStdNormalize, ClassificationCollater, load_state_dict

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

    test_dataset = ILSVRC2012Dataset(
        root_dir=ILSVRC2012_path,
        set_name='val',
        transform=transforms.Compose([
            Opencv2PIL(),
            CenterCropForVQGAN(resize=input_image_size),
            TorchMeanStdNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]))
    test_collater = ClassificationCollater()

    seed = 0
    # batch_size is total size
    batch_size = 64
    # num_workers is total workers
    num_workers = 4

    use_amp = False
    # torch.float16 or torch.bfloat16
    amp_type = torch.float16
