import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import ILSVRC2012_path

from SimpleGeneration.flux_autoencoder.models.z_image_autoencoder import AutoEncoder
from SimpleGeneration.flux_autoencoder.datasets.ilsvrc2012dataset import ILSVRC2012Dataset
from SimpleGeneration.flux_autoencoder.common import Opencv2PIL, FLUXResize, TorchMeanStdNormalize, FLUXTestCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    # 256 or 512
    network = 'z_image_ae'
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
                                  sample_z=False,
                                  use_gradient_checkpoint=False)
    # load total pretrained model or not
    trained_generator_model_path = '/root/autodl-tmp/pretrained_models/z_image_convert_from_pytorch_official_weights/z_image_vae_convert_from_pytorch_official_weight.pth'
    load_state_dict(trained_generator_model_path, generator_model)

    test_dataset = ILSVRC2012Dataset(root_dir=ILSVRC2012_path,
                                     set_name='val',
                                     transform=transforms.Compose([
                                         Opencv2PIL(),
                                         FLUXResize(resize=input_image_size),
                                         TorchMeanStdNormalize(
                                             mean=[0.5, 0.5, 0.5],
                                             std=[0.5, 0.5, 0.5]),
                                     ]))
    test_collater = FLUXTestCollater(resize=input_image_size)

    seed = 0
    # batch_size is total size
    batch_size = 64
    # num_workers is total workers
    num_workers = 8

    use_amp = False
    # torch.float16 or torch.bfloat16
    amp_type = torch.bfloat16
