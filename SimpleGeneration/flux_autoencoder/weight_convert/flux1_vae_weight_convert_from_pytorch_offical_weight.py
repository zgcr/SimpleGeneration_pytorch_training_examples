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

from safetensors.torch import load_file

if __name__ == '__main__':
    model = AutoEncoder(inplanes=3,
                        planes=128,
                        planes_mult=[1, 2, 4, 4],
                        res_block_nums=2,
                        z_planes=16,
                        out_planes=3,
                        scale_factor=0.3611,
                        shift_factor=0.1159,
                        sample_z=False,
                        use_gradient_checkpoint=False)

    model_name_list = []
    for name, weight in model.state_dict().items():
        model_name_list.append([name, weight.shape])

    print('1111', len(model_name_list))
    # for name, weight_shape in model_name_list:
    #     print('1111', name, weight_shape)

    # /root/autodl-tmp/pretrained_models/flux1_pytorch_official_weights/FLUX.1-dev-ae.safetensors
    # /root/autodl-tmp/pretrained_models/flux1_pytorch_official_weights/FLUX.1-Fill-dev-ae.safetensors
    # /root/autodl-tmp/pretrained_models/flux1_pytorch_official_weights/FLUX.1-Kontext-dev-ae.safetensors
    # /root/autodl-tmp/pretrained_models/flux1_pytorch_official_weights/FLUX.1-schnell-ae.safetensors

    saved_model_path = '/root/autodl-tmp/pretrained_models/flux1_pytorch_official_weights/FLUX.1-schnell-ae.safetensors'
    saved_state_dict = load_file(saved_model_path)

    save_name_list = []
    for name, weight in saved_state_dict.items():
        save_name_list.append([name, weight.shape])

    print('2222', len(save_name_list))
    # for name, weight_shape in save_name_list:
    #     print('2222', name, weight_shape)

    convert_dict = {}
    for key, value in saved_state_dict.items():
        if key in model.state_dict().keys():
            convert_dict[key] = value
        else:
            print('2323', key)

    convert_name_list = []
    for name, weight in convert_dict.items():
        convert_name_list.append([name, weight.shape])

    print('3333', len(convert_name_list))
    # for name, weight_shape in convert_name_list:
    #     print('3333', name, weight_shape)

    in_count = 0
    for key, value in convert_dict.items():
        if key in model.state_dict().keys():
            if value.shape == model.state_dict()[key].shape:
                in_count += 1
        else:
            print(key)
    print('4444', in_count)

    save_model_name = saved_model_path.split('/')[-1].split('.safetensors')[0]
    torch.save(
        convert_dict,
        f'/root/autodl-tmp/pretrained_models/flux1_convert_from_pytorch_official_weights/{save_model_name}_convert_from_pytorch_official_weight.pth'
    )
