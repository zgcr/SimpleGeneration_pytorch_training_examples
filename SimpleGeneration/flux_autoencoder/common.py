import numpy as np

from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


class Opencv2PIL:

    def __init__(self):
        pass

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image' key.
        '''
        image = sample['image']

        image = Image.fromarray(np.uint8(image))

        sample['image'] = image

        return sample


class PIL2Opencv:

    def __init__(self):
        pass

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image' key.
        '''
        image = sample['image']

        image = np.asarray(image).astype(np.float32)

        sample['image'] = image

        return sample


class FLUXResize:

    def __init__(self, resize=1024):
        assert resize % 16 == 0, f'resize must be a multiple of 16, got {resize}'
        self.resize = resize

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image' key.
        '''
        image = sample['image']

        width, height = image.size

        max_side = max(width, height)
        scale = self.resize / max_side

        new_width = int(16 * round(width * scale / 16))
        new_height = int(16 * round(height * scale / 16))

        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        sample['image'] = image

        return sample


class FLUXRandomPad:

    def __init__(self, resize=1024, fill_value=(0, 0, 0)):
        self.resize = resize
        self.fill_value = fill_value

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image' key.
        '''
        image = sample['image']

        width, height = image.size

        if width == self.resize and height == self.resize:

            return sample

        new_image = Image.new('RGB', (self.resize, self.resize),
                              self.fill_value)

        if width < self.resize:
            max_offset_x = self.resize - width
            offset_x = np.random.randint(0, max_offset_x + 1)
        else:
            offset_x = 0

        if height < self.resize:
            max_offset_y = self.resize - height
            offset_y = np.random.randint(0, max_offset_y + 1)
        else:
            offset_y = 0

        new_image.paste(image, (offset_x, offset_y))

        sample['image'] = new_image

        return sample


class TorchRandomHorizontalFlip:

    def __init__(self, prob=0.5):
        self.RandomHorizontalFlip = transforms.RandomHorizontalFlip(prob)

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image' key.
        '''
        image = sample['image']

        image = self.RandomHorizontalFlip(image)

        sample['image'] = image

        return sample


class TorchMeanStdNormalize:

    def __init__(self, mean, std):
        self.to_tensor = transforms.ToTensor()
        self.Normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image' key.
        '''
        image = sample['image']

        image = self.to_tensor(image)
        image = self.Normalize(image)
        # 3 H W ->H W 3
        image = image.permute(1, 2, 0)
        image = image.numpy()

        sample['image'] = image

        return sample


class FLUXTrainCollater:

    def __init__(self, resize=1024):
        self.resize = resize

    def __call__(self, data):
        images = [s['image'] for s in data]

        input_images = np.zeros((len(images), self.resize, self.resize, 3),
                                dtype=np.float32)
        for i, image in enumerate(images):
            input_images[i, 0:image.shape[0], 0:image.shape[1], :] = image
        input_images = torch.from_numpy(input_images)
        # B H W 3 ->B 3 H W
        input_images = input_images.permute(0, 3, 1, 2)
        input_images = input_images.float()

        return {
            'image': input_images,
        }


class FLUXTestCollater:

    def __init__(self, resize=1024):
        self.resize = resize

    def __call__(self, data):
        images = [s['image'] for s in data]

        input_images = np.zeros((len(images), self.resize, self.resize, 3),
                                dtype=np.float32)
        sizes = []
        for i, image in enumerate(images):
            per_image_height, per_image_width = image.shape[0], image.shape[1]
            input_images[i, 0:image.shape[0], 0:image.shape[1], :] = image
            sizes.append([per_image_height, per_image_width])
        input_images = torch.from_numpy(input_images)
        # B H W 3 ->B 3 H W
        input_images = input_images.permute(0, 3, 1, 2)
        input_images = input_images.float()

        return {
            'image': input_images,
            'size': sizes,
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

    if len(filtered_state_dict) == 0:
        print('No pretrained parameters to load!')
    else:
        print(
            f'load/model weight nums:{len(filtered_state_dict)}/{len(model.state_dict())}'
        )
        print(f'not loaded save layer weight:\n{not_loaded_save_state_dict}')
        model.load_state_dict(filtered_state_dict, strict=False)

    return
