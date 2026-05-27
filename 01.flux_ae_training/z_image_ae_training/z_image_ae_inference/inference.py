import os
import sys
import warnings

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast

from tools.utils import set_seed


def resize_image(image, min_mp=0.5, max_mp=2.0):
    """
    调整图像大小，使其百万像素数在指定范围内，且尺寸为32的倍数
    
    参数:
        image: PIL图像对象
        min_mp: 最小百万像素数，默认0.5
        max_mp: 最大百万像素数，默认2.0
    """
    # 获取图像的宽度和高度（单位：像素）
    width, height = image.size

    # 计算当前图像的百万像素数（总像素数除以100万）
    mp = (width * height) / 4000000

    # 检查当前百万像素数是否在指定的范围内
    if min_mp <= mp <= max_mp:
        # 即使MP在范围内，也需要确保宽高都是32的倍数
        # 将宽度调整为最接近的32的倍数（先除以32，四舍五入，再乘以32）
        new_width = int(32 * round(width / 32))
        # 将高度调整为最接近的32的倍数
        new_height = int(32 * round(height / 32))

        # 如果调整后的尺寸与原尺寸不同，需要重新缩放图像
        if new_width != width or new_height != height:
            # 使用LANCZOS重采样算法调整图像大小（高质量的下采样/上采样方法）
            image = image.resize((new_width, new_height),
                                 Image.Resampling.LANCZOS)
            # 返回调整后的图像
            return image

        # 如果尺寸已经是32的倍数，直接返回原图像
        return image

    # 如果MP不在范围内，需要计算缩放因子
    # 当前MP小于最小值时，需要放大图像
    if mp < min_mp:
        # 缩放因子 = sqrt(目标MP / 当前MP)，开平方是因为宽高都要缩放
        scale = (min_mp / mp)**0.5
    else:
        # 当前MP大于最大值时，需要缩小图像
        scale = (max_mp / mp)**0.5

    # 根据缩放因子计算新宽度，并确保是32的倍数
    new_width = int(32 * round(width * scale / 32))
    # 根据缩放因子计算新高度，并确保是32的倍数
    new_height = int(32 * round(height * scale / 32))

    # 使用LANCZOS重采样算法调整图像到新尺寸
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

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
        per_image = Image.open(per_image_path).convert('RGB')
        per_image_origin_width, per_image_origin_height = per_image.size

        print('1212', per_image_origin_width, per_image_origin_height)

        per_image = resize_image(per_image)

        print('1313', per_image.size)

        per_image = np.array(per_image) / 255.
        x = (per_image - 0.5) / 0.5
        x = torch.tensor(x)
        x = x.unsqueeze(dim=0)
        x = torch.einsum('nhwc->nchw', x)
        x_input = x.float().cuda()

        with torch.no_grad():
            if config.use_amp:
                with autocast(device_type="cuda", dtype=config.amp_type):
                    per_output = model(x_input)
            else:
                per_output = model(x_input)

        per_output = per_output.permute(0, 2, 3, 1)[0]

        print('1414', per_output.shape, torch.max(per_output),
              torch.min(per_output))

        per_output = (per_output * 0.5 + 0.5) * 255.
        per_output = torch.clamp(per_output, min=0, max=255)
        per_output = per_output.cpu().numpy().astype(np.uint8)

        per_output = Image.fromarray(per_output)
        per_output = per_output.resize(
            (per_image_origin_width, per_image_origin_height),
            Image.Resampling.LANCZOS)

        print('1515', per_output.size)

        per_output_name = f'{image_name_prefix}_result.jpg'
        per_output_path = os.path.join(config.save_image_dir, per_output_name)
        per_output.save(per_output_path, format='JPEG', quality=95)


if __name__ == '__main__':
    main()
