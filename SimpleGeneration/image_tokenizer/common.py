import cv2
import math
import numpy as np

from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


class RandomCropForVQGAN:

    def __init__(self, resize=256, crop_fraction_range=[0.8, 1.0]):
        self.resize = int(resize)
        self.crop_fraction_range = crop_fraction_range

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'label' keys.
        '''
        image, label = sample['image'], sample['label']

        min_size = math.ceil(self.resize / self.crop_fraction_range[1])
        max_size = math.ceil(self.resize / self.crop_fraction_range[0])
        random_size = np.random.randint(min_size, max_size + 1)

        while min(*image.size) >= 2 * random_size:
            image = image.resize(tuple(x // 2 for x in image.size),
                                 resample=Image.BOX)

        scale = random_size / min(*image.size)
        image = image.resize(tuple(round(x * scale) for x in image.size),
                             resample=Image.BICUBIC)

        image = np.array(image)

        crop_y = np.random.randint(0, image.shape[0] - self.resize + 1)
        crop_x = np.random.randint(0, image.shape[1] - self.resize + 1)
        image = image[crop_y:crop_y + self.resize, crop_x:crop_x + self.resize]

        image = Image.fromarray(image)

        sample['image'], sample['label'] = image, label

        return sample


class CenterCropForVQGAN:

    def __init__(self, resize=256):
        self.resize = int(resize)

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'label' keys.
        '''
        image, label = sample['image'], sample['label']

        while min(*image.size) >= 2 * self.resize:
            image = image.resize(tuple(x // 2 for x in image.size),
                                 resample=Image.BOX)

        scale = self.resize / min(*image.size)
        image = image.resize(tuple(round(x * scale) for x in image.size),
                             resample=Image.BICUBIC)

        image = np.array(image)
        # 中心裁剪
        crop_y = (image.shape[0] - self.resize) // 2
        crop_x = (image.shape[1] - self.resize) // 2
        image = image[crop_y:crop_y + self.resize, crop_x:crop_x + self.resize]

        image = Image.fromarray(image)

        sample['image'], sample['label'] = image, label

        return sample


class Opencv2PIL:

    def __init__(self):
        pass

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'label' keys.
        '''
        image, label = sample['image'], sample['label']

        image = Image.fromarray(np.uint8(image))

        sample['image'], sample['label'] = image, label

        return sample


class PIL2Opencv:

    def __init__(self):
        pass

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'label' keys.
        '''
        image, label = sample['image'], sample['label']

        image = np.asarray(image).astype(np.float32)

        sample['image'], sample['label'] = image, label

        return sample


class TorchRandomHorizontalFlip:

    def __init__(self, prob=0.5):
        self.RandomHorizontalFlip = transforms.RandomHorizontalFlip(prob)

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'label' keys.
        '''
        image, label = sample['image'], sample['label']

        image = self.RandomHorizontalFlip(image)

        sample['image'], sample['label'] = image, label

        return sample


class TorchResize:

    def __init__(self, resize=224):
        self.Resize = transforms.Resize(int(resize))

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'label' keys.
        '''
        image, label = sample['image'], sample['label']

        image = self.Resize(image)

        sample['image'], sample['label'] = image, label

        return sample


class TorchMeanStdNormalize:

    def __init__(self, mean, std):
        self.to_tensor = transforms.ToTensor()
        self.Normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'label' keys.
        '''
        image, label = sample['image'], sample['label']

        image = self.to_tensor(image)
        image = self.Normalize(image)
        # 3 H W ->H W 3
        image = image.permute(1, 2, 0)
        image = image.numpy()

        sample['image'], sample['label'] = image, label

        return sample


class ClassificationCollater:

    def __init__(self):
        pass

    def __call__(self, data):
        images = [s['image'] for s in data]
        labels = [s['label'] for s in data]

        images = np.array(images).astype(np.float32)
        labels = np.array(labels).astype(np.float32)

        images = torch.from_numpy(images).float()
        # B H W 3 ->B 3 H W
        images = images.permute(0, 3, 1, 2)
        labels = torch.from_numpy(labels).long()

        return {
            'image': images,
            'label': labels,
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


def load_state_dict(saved_model_path,
                    model,
                    excluded_layer_name=(),
                    loading_new_input_size_position_encoding_weight=False):
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
    for name, weight in saved_state_dict.items():
        if name in model.state_dict() and not any(
                excluded_name in name for excluded_name in excluded_layer_name
        ) and weight.shape == model.state_dict()[name].shape:
            filtered_state_dict[name] = weight
        else:
            not_loaded_save_state_dict.append(name)

    position_encoding_already_loaded = False
    if 'pos_embed' in filtered_state_dict.keys():
        position_encoding_already_loaded = True

    # for vit net, loading a position encoding layer with new input size
    if loading_new_input_size_position_encoding_weight and not position_encoding_already_loaded:
        # assert position_encoding_layer name are unchanged for model and saved_model
        # assert class_token num are unchanged for model and saved_model
        # assert embedding_planes are unchanged for model and saved_model
        if hasattr(model, 'cls_token') and hasattr(model, 'pos_embed'):
            model_num_cls_token = model.cls_token.shape[1]
            model_embedding_planes = model.pos_embed.shape[2]
            model_encoding_shape = int(
                (model.pos_embed.shape[1] - model_num_cls_token)**0.5)
            encoding_layer_name, encoding_layer_weight = None, None
            for name, weight in saved_state_dict.items():
                if 'pos_embed' in name:
                    encoding_layer_name = name
                    encoding_layer_weight = weight
                    break
            save_model_encoding_shape = int(
                (encoding_layer_weight.shape[1] - model_num_cls_token)**0.5)

            save_model_cls_token_weight = encoding_layer_weight[:, 0:
                                                                model_num_cls_token, :]
            save_model_position_weight = encoding_layer_weight[:,
                                                               model_num_cls_token:, :]
            save_model_position_weight = save_model_position_weight.reshape(
                -1, save_model_encoding_shape, save_model_encoding_shape,
                model_embedding_planes).permute(0, 3, 1, 2)
            save_model_position_weight = F.interpolate(
                save_model_position_weight,
                size=(model_encoding_shape, model_encoding_shape),
                mode='bicubic')
            save_model_position_weight = save_model_position_weight.permute(
                0, 2, 3, 1).flatten(1, 2)
            model_encoding_layer_weight = torch.cat(
                (save_model_cls_token_weight, save_model_position_weight),
                dim=1)

            filtered_state_dict[
                encoding_layer_name] = model_encoding_layer_weight
            not_loaded_save_state_dict.remove('pos_embed')

    if len(filtered_state_dict) == 0:
        print('No pretrained parameters to load!')
    else:
        print(
            f'load/model weight nums:{len(filtered_state_dict)}/{len(model.state_dict())}'
        )
        print(f'not loaded save layer weight:\n{not_loaded_save_state_dict}')
        model.load_state_dict(filtered_state_dict, strict=False)

    return
