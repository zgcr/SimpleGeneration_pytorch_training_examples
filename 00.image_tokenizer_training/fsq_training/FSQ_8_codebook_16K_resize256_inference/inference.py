import os
import sys
import warnings

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import argparse
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast

from tools.utils import set_seed


def center_crop_arr(image, image_size):
    while min(*image.size) >= 2 * image_size:
        image = image.resize(tuple(x // 2 for x in image.size),
                             resample=Image.BOX)

    scale = image_size / min(*image.size)
    image = image.resize(tuple(round(x * scale) for x in image.size),
                         resample=Image.BICUBIC)

    image = np.array(image)
    # 中心裁剪
    crop_y = (image.shape[0] - image_size) // 2
    crop_x = (image.shape[1] - image_size) // 2
    image = image[crop_y:crop_y + image_size, crop_x:crop_x + image_size]

    image = Image.fromarray(image)

    return image


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch VQGAN Inference')
    parser.add_argument('--work-dir',
                        type=str,
                        help='path for get testing config')

    return parser.parse_args()


def main():
    assert torch.cuda.is_available(), 'need gpu to train network!'
    torch.cuda.empty_cache()

    args = parse_args()
    sys.path.append(args.work_dir)
    from inference_config import config

    set_seed(config.seed)
    model = config.model.cuda()
    model.eval()

    os.makedirs(config.save_image_dir, exist_ok=True)

    all_test_images_path_list = []
    for per_image_name in sorted(os.listdir(config.inference_image_dir)):
        if per_image_name.endswith('.jpg'):
            per_image_path = os.path.join(config.inference_image_dir,
                                          per_image_name)
            image_name_prefix = per_image_name.split('/')[-1].split('.')[0]
            all_test_images_path_list.append(
                [image_name_prefix, per_image_path])

    print('1111', len(all_test_images_path_list), all_test_images_path_list[0])

    for image_name_prefix, per_image_path in tqdm(all_test_images_path_list):
        per_image = cv2.imdecode(np.fromfile(per_image_path, dtype=np.uint8),
                                 cv2.IMREAD_COLOR)
        per_image = cv2.cvtColor(per_image, cv2.COLOR_BGR2RGB)
        per_image = Image.fromarray(per_image)

        per_image = center_crop_arr(per_image, config.input_image_size)
        per_image = np.array(per_image) / 255.
        x = (per_image - 0.5) / 0.5
        x = torch.tensor(x)
        x = x.unsqueeze(dim=0)
        x = torch.einsum('nhwc->nchw', x)
        x_input = x.float().cuda()

        with torch.no_grad():
            if config.use_amp:
                with autocast(device_type="cuda", dtype=config.amp_type):
                    per_output, _, _ = model(x_input)
            else:
                per_output, _, _ = model(x_input)

        print('1111', per_output.shape)

        per_output = per_output.permute(0, 2, 3, 1)[0]

        print('2222', per_output.shape, torch.max(per_output),
              torch.min(per_output))

        per_output = (per_output * 0.5 + 0.5) * 255.
        per_output = torch.clamp(per_output, min=0, max=255)
        per_output = per_output.cpu().numpy().astype(np.uint8)
        per_output = cv2.cvtColor(per_output, cv2.COLOR_RGB2BGR)

        print('3333', per_output.shape, np.max(per_output), np.min(per_output))

        per_output_name = f'{image_name_prefix}_result.jpg'
        per_output_path = os.path.join(config.save_image_dir, per_output_name)
        cv2.imencode('.jpg', per_output)[1].tofile(per_output_path)


if __name__ == '__main__':
    main()
