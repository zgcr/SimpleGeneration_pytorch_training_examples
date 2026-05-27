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
        sample must be a dict,contains 'image'、'text' keys.
        '''
        image, text = sample['image'], sample['text']

        image = Image.fromarray(np.uint8(image))

        sample['image'], sample['text'] = image, text

        return sample


class PIL2Opencv:

    def __init__(self):
        pass

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'text' keys.
        '''
        image, text = sample['image'], sample['text']

        image = np.asarray(image).astype(np.float32)

        sample['image'], sample['text'] = image, text

        return sample


class TorchResize:

    def __init__(self, resize=224, interpolation=InterpolationMode.BICUBIC):
        self.Resize = transforms.Resize((int(resize), int(resize)),
                                        interpolation=interpolation)

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'text' keys.
        '''
        image, text = sample['image'], sample['text']

        image = self.Resize(image)

        sample['image'], sample['text'] = image, text

        return sample


class TorchMeanStdNormalize:

    def __init__(self,
                 mean=[0.48145466, 0.4578275, 0.40821073],
                 std=[0.26862954, 0.26130258, 0.27577711]):
        self.to_tensor = transforms.ToTensor()
        self.Normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'text' keys.
        '''
        image, text = sample['image'], sample['text']

        image = self.to_tensor(image)
        image = self.Normalize(image)
        # 3 H W ->H W 3
        image = image.permute(1, 2, 0)
        image = image.numpy()

        sample['image'], sample['text'] = image, text

        return sample


class ImageTextPairCollater:

    def __init__(self):
        pass

    def __call__(self, data):
        images = [s['image'] for s in data]
        texts = [s['text'] for s in data]

        images = np.array(images).astype(np.float32)
        images = torch.from_numpy(images).float()
        # B H W 3 ->B 3 H W
        images = images.permute(0, 3, 1, 2)

        return {
            'image': images,
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
