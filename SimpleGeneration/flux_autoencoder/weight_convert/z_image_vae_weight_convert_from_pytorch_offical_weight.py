import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from SimpleGeneration.flux_autoencoder.models.z_image_autoencoder import AutoEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F

from safetensors.torch import load_file


def convert_diffusers_key_to_flux_key(diffusers_key):
    """
    Convert a diffusers-format key name to a flux-format key name.
    Z-Image VAE uses diffusers format (AutoencoderKL from diffusers).
    Our z_image_autoencoder.py uses flux-style key names.

    Key mapping rules:
    Encoder:
      encoder.down_blocks.{i}.resnets.{j}.xxx  -> encoder.down.{i}.block.{j}.xxx
      encoder.down_blocks.{i}.resnets.{j}.conv_shortcut -> encoder.down.{i}.block.{j}.nin_shortcut
      encoder.down_blocks.{i}.downsamplers.0.conv -> encoder.down.{i}.downsample.conv
      encoder.mid_block.resnets.0.xxx -> encoder.mid.block_1.xxx
      encoder.mid_block.resnets.1.xxx -> encoder.mid.block_2.xxx
      encoder.mid_block.attentions.0.group_norm -> encoder.mid.attn_1.norm
      encoder.mid_block.attentions.0.to_q -> encoder.mid.attn_1.q  (Linear -> Conv2d reshape needed)
      encoder.mid_block.attentions.0.to_k -> encoder.mid.attn_1.k
      encoder.mid_block.attentions.0.to_v -> encoder.mid.attn_1.v
      encoder.mid_block.attentions.0.to_out.0 -> encoder.mid.attn_1.proj_out
      encoder.conv_norm_out -> encoder.norm_out
      encoder.conv_in -> encoder.conv_in  (unchanged)
      encoder.conv_out -> encoder.conv_out  (unchanged)

    Decoder:
      decoder.up_blocks.{i}.resnets.{j}.xxx -> decoder.up.{3-i}.block.{j}.xxx
      decoder.up_blocks.{i}.resnets.{j}.conv_shortcut -> decoder.up.{3-i}.block.{j}.nin_shortcut
      decoder.up_blocks.{i}.upsamplers.0.conv -> decoder.up.{3-i}.upsample.conv
      decoder.mid_block.resnets.0.xxx -> decoder.mid.block_1.xxx
      decoder.mid_block.resnets.1.xxx -> decoder.mid.block_2.xxx
      decoder.mid_block.attentions.0.group_norm -> decoder.mid.attn_1.norm
      decoder.mid_block.attentions.0.to_q -> decoder.mid.attn_1.q
      decoder.mid_block.attentions.0.to_k -> decoder.mid.attn_1.k
      decoder.mid_block.attentions.0.to_v -> decoder.mid.attn_1.v
      decoder.mid_block.attentions.0.to_out.0 -> decoder.mid.attn_1.proj_out
      decoder.conv_norm_out -> decoder.norm_out
      decoder.conv_in -> decoder.conv_in  (unchanged)
      decoder.conv_out -> decoder.conv_out  (unchanged)

    Note: up_blocks index is reversed: diffusers up_blocks[i] -> flux up[stage_num-1-i]
          For block_out_channels=[128,256,512,512] (stage_num=4), mapping is i -> 3-i

    Attention layers use nn.Linear in diffusers but nn.Conv2d(1x1) in flux.
    Linear weight shape: (out_features, in_features)
    Conv2d(1x1) weight shape: (out_channels, in_channels, 1, 1)
    So Linear weight needs unsqueeze(-1).unsqueeze(-1) to become Conv2d weight.
    Bias shape is the same for both: (out_features,) / (out_channels,).
    """
    key = diffusers_key
    needs_reshape = False  # flag for attention Linear -> Conv2d weight reshape
    stage_num = 4  # len(block_out_channels) = 4

    # --- Encoder ---
    # down_blocks
    if 'encoder.down_blocks.' in key:
        # conv_shortcut -> nin_shortcut
        key = key.replace('.conv_shortcut.', '.nin_shortcut.')
        # resnets -> block
        key = key.replace('.resnets.', '.block.')
        # downsamplers.0.conv -> downsample.conv
        key = key.replace('.downsamplers.0.conv.', '.downsample.conv.')
        # down_blocks -> down
        key = key.replace('encoder.down_blocks.', 'encoder.down.')

    # encoder mid_block
    elif 'encoder.mid_block.' in key:
        if 'encoder.mid_block.resnets.0.' in key:
            key = key.replace('encoder.mid_block.resnets.0.',
                              'encoder.mid.block_1.')
        elif 'encoder.mid_block.resnets.1.' in key:
            key = key.replace('encoder.mid_block.resnets.1.',
                              'encoder.mid.block_2.')
        elif 'encoder.mid_block.attentions.0.group_norm.' in key:
            key = key.replace('encoder.mid_block.attentions.0.group_norm.',
                              'encoder.mid.attn_1.norm.')
        elif 'encoder.mid_block.attentions.0.to_q.' in key:
            key = key.replace('encoder.mid_block.attentions.0.to_q.',
                              'encoder.mid.attn_1.q.')
            if '.weight' in key:
                needs_reshape = True
        elif 'encoder.mid_block.attentions.0.to_k.' in key:
            key = key.replace('encoder.mid_block.attentions.0.to_k.',
                              'encoder.mid.attn_1.k.')
            if '.weight' in key:
                needs_reshape = True
        elif 'encoder.mid_block.attentions.0.to_v.' in key:
            key = key.replace('encoder.mid_block.attentions.0.to_v.',
                              'encoder.mid.attn_1.v.')
            if '.weight' in key:
                needs_reshape = True
        elif 'encoder.mid_block.attentions.0.to_out.0.' in key:
            key = key.replace('encoder.mid_block.attentions.0.to_out.0.',
                              'encoder.mid.attn_1.proj_out.')
            if '.weight' in key:
                needs_reshape = True

    # encoder conv_norm_out -> norm_out
    elif 'encoder.conv_norm_out.' in key:
        key = key.replace('encoder.conv_norm_out.', 'encoder.norm_out.')

    # --- Decoder ---
    # up_blocks (need index reversal)
    elif 'decoder.up_blocks.' in key:
        # Extract the up_blocks index
        parts = key.split('.')
        up_block_idx = int(parts[2])
        reversed_idx = stage_num - 1 - up_block_idx
        parts[2] = str(reversed_idx)

        # conv_shortcut -> nin_shortcut
        key = '.'.join(parts)
        key = key.replace('.conv_shortcut.', '.nin_shortcut.')
        # resnets -> block
        key = key.replace('.resnets.', '.block.')
        # upsamplers.0.conv -> upsample.conv
        key = key.replace('.upsamplers.0.conv.', '.upsample.conv.')
        # up_blocks -> up
        key = key.replace('decoder.up_blocks.', 'decoder.up.')

    # decoder mid_block
    elif 'decoder.mid_block.' in key:
        if 'decoder.mid_block.resnets.0.' in key:
            key = key.replace('decoder.mid_block.resnets.0.',
                              'decoder.mid.block_1.')
        elif 'decoder.mid_block.resnets.1.' in key:
            key = key.replace('decoder.mid_block.resnets.1.',
                              'decoder.mid.block_2.')
        elif 'decoder.mid_block.attentions.0.group_norm.' in key:
            key = key.replace('decoder.mid_block.attentions.0.group_norm.',
                              'decoder.mid.attn_1.norm.')
        elif 'decoder.mid_block.attentions.0.to_q.' in key:
            key = key.replace('decoder.mid_block.attentions.0.to_q.',
                              'decoder.mid.attn_1.q.')
            if '.weight' in key:
                needs_reshape = True
        elif 'decoder.mid_block.attentions.0.to_k.' in key:
            key = key.replace('decoder.mid_block.attentions.0.to_k.',
                              'decoder.mid.attn_1.k.')
            if '.weight' in key:
                needs_reshape = True
        elif 'decoder.mid_block.attentions.0.to_v.' in key:
            key = key.replace('decoder.mid_block.attentions.0.to_v.',
                              'decoder.mid.attn_1.v.')
            if '.weight' in key:
                needs_reshape = True
        elif 'decoder.mid_block.attentions.0.to_out.0.' in key:
            key = key.replace('decoder.mid_block.attentions.0.to_out.0.',
                              'decoder.mid.attn_1.proj_out.')
            if '.weight' in key:
                needs_reshape = True

    # decoder conv_norm_out -> norm_out
    elif 'decoder.conv_norm_out.' in key:
        key = key.replace('decoder.conv_norm_out.', 'decoder.norm_out.')

    # encoder.conv_in, encoder.conv_out, decoder.conv_in, decoder.conv_out
    # These keys are the same in both formats, no conversion needed.

    return key, needs_reshape


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

    # /root/autodl-tmp/pretrained_models/z_image_pytorch_official_weights/vae/diffusion_pytorch_model.safetensors

    saved_model_path = '/root/autodl-tmp/pretrained_models/z_image_pytorch_official_weights/vae/diffusion_pytorch_model.safetensors'
    saved_state_dict = load_file(saved_model_path)

    save_name_list = []
    for name, weight in saved_state_dict.items():
        save_name_list.append([name, weight.shape])

    print('2222', len(save_name_list))
    # for name, weight_shape in save_name_list:
    #     print('2222', name, weight_shape)

    # Convert diffusers keys to flux keys
    convert_dict = {}
    not_converted_keys = []
    for diffusers_key, value in saved_state_dict.items():
        flux_key, needs_reshape = convert_diffusers_key_to_flux_key(
            diffusers_key)

        # Reshape Linear weight to Conv2d(1x1) weight if needed
        if needs_reshape and value.dim() == 2:
            value = value.unsqueeze(-1).unsqueeze(-1)

        if flux_key in model.state_dict().keys():
            if value.shape == model.state_dict()[flux_key].shape:
                convert_dict[flux_key] = value
            else:
                print(
                    f'Shape mismatch: {diffusers_key} -> {flux_key}, '
                    f'saved: {value.shape}, model: {model.state_dict()[flux_key].shape}'
                )
                not_converted_keys.append(diffusers_key)
        else:
            print(f'Key not found in model: {diffusers_key} -> {flux_key}')
            not_converted_keys.append(diffusers_key)

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

    if not_converted_keys:
        print('Not converted keys:')
        for key in not_converted_keys:
            print(f'  {key}')

    # Check for model keys that are not loaded
    not_loaded_model_keys = []
    for key in model.state_dict().keys():
        if key not in convert_dict:
            not_loaded_model_keys.append(key)
    if not_loaded_model_keys:
        print('Model keys not loaded (expected for logvar, reg):')
        for key in not_loaded_model_keys:
            print(f'  {key}')

    save_model_name = 'z_image_vae'
    torch.save(
        convert_dict,
        f'/root/autodl-tmp/pretrained_models/z_image_convert_from_pytorch_official_weights/{save_model_name}_convert_from_pytorch_official_weight.pth'
    )
