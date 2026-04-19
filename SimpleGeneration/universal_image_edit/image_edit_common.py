import math
import numpy as np

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode


class Opencv2PIL:

    def __init__(self):
        pass

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'source_image'、'edited_image'、'text' keys.
        '''
        source_image, edited_image, text = sample['source_image'], sample[
            'edited_image'], sample['text']

        source_image = Image.fromarray(np.uint8(source_image))
        edited_image = Image.fromarray(np.uint8(edited_image))

        sample['source_image'], sample['edited_image'], sample[
            'text'] = source_image, edited_image, text

        return sample


class PIL2Opencv:

    def __init__(self):
        pass

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'source_image'、'edited_image'、'text' keys.
        '''
        source_image, edited_image, text = sample['source_image'], sample[
            'edited_image'], sample['text']

        source_image = np.asarray(source_image).astype(np.float32)
        edited_image = np.asarray(edited_image).astype(np.float32)

        sample['source_image'], sample['edited_image'], sample[
            'text'] = source_image, edited_image, text

        return sample


class TorchResize:

    def __init__(self, resize=224, interpolation=InterpolationMode.BICUBIC):
        self.Resize = transforms.Resize((int(resize), int(resize)),
                                        interpolation=interpolation)

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'source_image'、'edited_image'、'text' keys.
        '''
        source_image, edited_image, text = sample['source_image'], sample[
            'edited_image'], sample['text']

        source_image = self.Resize(source_image)
        edited_image = self.Resize(edited_image)

        sample['source_image'], sample['edited_image'], sample[
            'text'] = source_image, edited_image, text

        return sample


class TorchMeanStdNormalize:

    def __init__(self,
                 mean=[0.48145466, 0.4578275, 0.40821073],
                 std=[0.26862954, 0.26130258, 0.27577711]):
        self.to_tensor = transforms.ToTensor()
        self.Normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'source_image'、'edited_image'、'text' keys.
        '''
        source_image, edited_image, text = sample['source_image'], sample[
            'edited_image'], sample['text']

        source_image = self.to_tensor(source_image)
        source_image = self.Normalize(source_image)
        # 3 H W ->H W 3
        source_image = source_image.permute(1, 2, 0)
        source_image = source_image.numpy()

        edited_image = self.to_tensor(edited_image)
        edited_image = self.Normalize(edited_image)
        # 3 H W ->H W 3
        edited_image = edited_image.permute(1, 2, 0)
        edited_image = edited_image.numpy()

        sample['source_image'], sample['edited_image'], sample[
            'text'] = source_image, edited_image, text

        return sample


class ImageEditPairCollater:

    def __init__(self):
        pass

    def __call__(self, data):
        source_images = [s['source_image'] for s in data]
        edited_images = [s['edited_image'] for s in data]
        texts = [s['text'] for s in data]

        source_images = np.array(source_images).astype(np.float32)
        source_images = torch.from_numpy(source_images).float()
        # B H W 3 ->B 3 H W
        source_images = source_images.permute(0, 3, 1, 2)

        edited_images = np.array(edited_images).astype(np.float32)
        edited_images = torch.from_numpy(edited_images).float()
        # B H W 3 ->B 3 H W
        edited_images = edited_images.permute(0, 3, 1, 2)

        return {
            'source_image': source_images,
            'edited_image': edited_images,
            'text': texts,
        }


class AverageMeter:
    '''Computes and stores the average and current value'''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_state_dict(saved_model_path, model, excluded_layer_name=()):
    '''
    saved_model_path: a saved model.state_dict() .pth file path
    model: a new defined model
    excluded_layer_name: layer names that doesn't want to load parameters
    loading_new_input_size_position_encoding_weight: default False, for vit net, loading a position encoding layer with new input size, set True
    only load layer parameters which has same layer name and same layer weight shape
    '''
    if not saved_model_path:
        print('No pretrained model file!')
        return

    saved_state_dict = torch.load(saved_model_path,
                                  map_location=torch.device('cpu'),
                                  weights_only=True)

    not_loaded_save_state_dict = []
    filtered_state_dict = {}
    model_state_dict = model.state_dict()
    for name, weight in saved_state_dict.items():
        if name in model_state_dict and not any(
                excluded_name in name for excluded_name in excluded_layer_name
        ) and weight.shape == model_state_dict[name].shape:
            filtered_state_dict[name] = weight
        else:
            not_loaded_save_state_dict.append(name)

    if len(filtered_state_dict) == 0:
        print('No pretrained parameters to load!')
    else:
        print(
            f'load/model weight nums:{len(filtered_state_dict)}/{len(model.state_dict())}'
        )
        print(f'not loaded save layer weight:\n{not_loaded_save_state_dict}')
        model.load_state_dict(filtered_state_dict, strict=False)

    return
