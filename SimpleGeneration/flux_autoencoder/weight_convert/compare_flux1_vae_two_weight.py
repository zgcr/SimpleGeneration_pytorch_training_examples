import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from SimpleGeneration.flux_autoencoder.models.flux1_autoencoder import AutoEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == '__main__':
    # /root/autodl-tmp/pretrained_models/flux1_convert_from_pytorch_official_weights/FLUX.1-dev-ae_convert_from_pytorch_official_weight.pth
    # /root/autodl-tmp/pretrained_models/flux1_convert_from_pytorch_official_weights/FLUX.1-schnell-ae_convert_from_pytorch_official_weight.pth
    # /root/autodl-tmp/pretrained_models/flux1_convert_from_pytorch_official_weights/FLUX.1-Fill-dev-ae_convert_from_pytorch_official_weight.pth
    # /root/autodl-tmp/pretrained_models/flux1_convert_from_pytorch_official_weights/FLUX.1-Kontext-dev-ae_convert_from_pytorch_official_weight.pth

    saved_model_path1 = '/root/autodl-tmp/pretrained_models/flux1_convert_from_pytorch_official_weights/FLUX.1-dev-ae_convert_from_pytorch_official_weight.pth'
    saved_model_path2 = '/root/autodl-tmp/pretrained_models/flux1_convert_from_pytorch_official_weights/FLUX.1-Kontext-dev-ae_convert_from_pytorch_official_weight.pth'

    saved_state_dict1 = torch.load(saved_model_path1,
                                   map_location=torch.device('cpu'),
                                   weights_only=True)
    saved_state_dict2 = torch.load(saved_model_path2,
                                   map_location=torch.device('cpu'),
                                   weights_only=True)

    print('1111', len(saved_state_dict1), len(saved_state_dict2))

    in_count = 0
    for key1, value1 in saved_state_dict1.items():
        if key1 in saved_state_dict2.keys():
            value2 = saved_state_dict2[key1]
            if value1.shape == value2.shape:
                diff = torch.max(torch.abs(value1 - value2))
                if diff == 0:
                    in_count += 1
                else:
                    print('4444', key1)
            else:
                print('3333', key1)
        else:
            print('2222', key1)
    print('4444', in_count)
