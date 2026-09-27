import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import cv2
import math
import collections
import numpy as np

from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

from torch.utils.data import Sampler

from SimpleGeneration.universal_generation_edit.aspect_ratio_buckets import get_bucket_resolution


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


class TorchAspectRatioBucketResize:

    def __init__(self, base_resize=1024):
        assert base_resize % 16 == 0, f'base_resize must be a multiple of 16, got {base_resize}'
        self.base_resize = base_resize

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'bucket' keys.
        '''
        image, bucket = sample['image'], sample['bucket']

        bucket_ratio = float(bucket[0])

        target_height, target_width = get_bucket_resolution(
            bucket_ratio, self.base_resize)

        image = transforms.Resize(
            (target_height, target_width),
            interpolation=InterpolationMode.BICUBIC)(image)

        sample['image'] = image

        return sample


class TorchMeanStdNormalize:

    def __init__(self, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
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


class T2ICollator:

    def __init__(self):
        pass

    def __call__(self, data):
        images = [s['image'] for s in data]
        captions = [s['caption'] for s in data]

        images = np.array(images).astype(np.float32)
        images = torch.from_numpy(images).float()
        # B H W 3 ->B 3 H W
        images = images.permute(0, 3, 1, 2)

        return {
            'image': images,
            'caption': captions,
        }


class T2IBucketBatchSampler(Sampler):

    def __init__(self,
                 bucket_index_list,
                 batch_size,
                 drop_last=False,
                 shuffle=True,
                 seed=0):
        self.bucket_index_list = bucket_index_list
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

        self.num_replicas = dist.get_world_size()
        self.rank = dist.get_rank()
        self.epoch = 0

        self.bucket_to_sample_indexes = collections.OrderedDict()
        for sample_index, bucket_index in enumerate(bucket_index_list):
            self.bucket_to_sample_indexes.setdefault(int(bucket_index),
                                                     []).append(sample_index)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def build_all_batches(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        all_batches = []
        for bucket_index, sample_indexes in self.bucket_to_sample_indexes.items(
        ):
            sample_indexes = list(sample_indexes)
            if self.shuffle:
                perm = torch.randperm(len(sample_indexes),
                                      generator=generator).tolist()
                sample_indexes = [sample_indexes[p] for p in perm]
            for i in range(0, len(sample_indexes), self.batch_size):
                batch = sample_indexes[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                all_batches.append(batch)

        if self.shuffle:
            perm = torch.randperm(len(all_batches),
                                  generator=generator).tolist()
            all_batches = [all_batches[p] for p in perm]

        return all_batches

    def __iter__(self):
        all_batches = self.build_all_batches()

        total = (len(all_batches) // self.num_replicas) * self.num_replicas
        all_batches = all_batches[:total]

        for batch in all_batches[self.rank::self.num_replicas]:
            yield batch

    def __len__(self):
        total_batches = 0
        for sample_indexes in self.bucket_to_sample_indexes.values():
            if self.drop_last:
                total_batches += len(sample_indexes) // self.batch_size
            else:
                total_batches += math.ceil(
                    len(sample_indexes) / self.batch_size)

        total = (total_batches // self.num_replicas) * self.num_replicas

        return total // self.num_replicas


class AverageMeter:

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


def wrap_caption_text(caption, font, max_pixel_width):
    line_list = []
    for per_paragraph in caption.split('\n'):
        if per_paragraph == '':
            line_list.append('')
            continue

        # split paragraph into token list
        # a cjk char is a single token,continuous ascii chars are a single token
        token_list, ascii_buffer = [], ''
        for per_char in per_paragraph:
            if ord(per_char) > 0x2E7F or per_char == ' ':
                if ascii_buffer != '':
                    token_list.append(ascii_buffer)
                    ascii_buffer = ''
                token_list.append(per_char)
            else:
                ascii_buffer += per_char

        if ascii_buffer != '':
            token_list.append(ascii_buffer)

        current_line = ''
        for per_token in token_list:
            # font.getbbox returns (left,top,right,bottom),right is text pixel width
            if font.getbbox(current_line + per_token)[2] <= max_pixel_width:
                current_line += per_token
                continue

            if current_line.strip() != '':
                line_list.append(current_line)
            current_line = '' if per_token == ' ' else per_token

            # a single token is still too long,wrap it per char
            while font.getbbox(current_line)[2] > max_pixel_width and len(
                    current_line) > 1:
                cut_index = len(current_line) - 1
                while cut_index > 1 and font.getbbox(
                        current_line[:cut_index])[2] > max_pixel_width:
                    cut_index -= 1
                line_list.append(current_line[:cut_index])
                current_line = current_line[cut_index:]

        if current_line.strip() != '':
            line_list.append(current_line)

    if len(line_list) == 0:
        line_list = ['']

    return line_list


def draw_caption_on_image(image,
                          caption,
                          font_path,
                          font_size=None,
                          max_lines=6,
                          text_color=[0, 0, 0],
                          background_color=[255, 255, 255],
                          margin=8):
    image = np.ascontiguousarray(image, dtype=np.uint8)
    image_height, image_width = image.shape[0], image.shape[1]

    caption = '' if caption is None else str(caption)

    # font size is adaptive to image width when font_size is not given
    if font_size is None:
        font_size = max(14, int(round(image_width / 48.)))

    font = ImageFont.truetype(font_path, font_size)

    max_pixel_width = max(1, image_width - 2 * margin)
    line_list = wrap_caption_text(caption, font, max_pixel_width)

    if len(line_list) > max_lines:
        line_list = line_list[:max_lines]
        line_list[-1] = line_list[-1] + ' ...'

    line_height = int(round(font_size * 1.3))
    text_block_height = line_height * len(line_list) + 2 * margin

    # BGR ->RGB
    text_rgb_color = tuple(text_color[::-1])

    # overlay mode keeps image size,text is drawn on a translucent background box
    overlay_height = min(image_height, text_block_height)
    overlay_region = image[0:overlay_height, :, :].astype(np.float32)
    background_region = np.full_like(overlay_region,
                                     background_color,
                                     dtype=np.float32)
    image[0:overlay_height, :, :] = np.round(overlay_region * 0.3 +
                                             background_region * 0.7).clip(
                                                 0, 255).astype(np.uint8)

    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    for per_line_index, per_line in enumerate(line_list):
        per_line_y = margin + per_line_index * line_height
        # drop the lines that exceed the translucent background box
        if per_line_y + line_height > overlay_height:
            break
        draw.text((margin, per_line_y),
                  per_line,
                  font=font,
                  fill=text_rgb_color)

    numpy_image = np.ascontiguousarray(cv2.cvtColor(np.asarray(pil_image),
                                                    cv2.COLOR_RGB2BGR),
                                       dtype=np.uint8)

    return numpy_image
