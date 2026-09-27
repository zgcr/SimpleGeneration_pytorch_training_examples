import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import cv2
import json
import numpy as np

from multiprocessing import Pool

from tqdm import tqdm

from torch.utils.data import Dataset

from SimpleGeneration.universal_generation_edit.aspect_ratio_buckets import ASPECT_RATIO_BUCKETS, get_closest_bucket, get_bucket_resolution


class T2IDataset(Dataset):

    def __init__(self,
                 root_dir,
                 dataset_name=[
                     'SACap-1M',
                 ],
                 set_name={
                     'SACap-1M': [
                         'train_000',
                     ],
                 },
                 min_image_long_side=64,
                 max_image_long_side=4096,
                 min_aspect_ratio=0.25,
                 max_aspect_ratio=4.0,
                 min_t2i_caption_length=4,
                 max_t2i_caption_length=1536,
                 transform=None):
        folder_task_list = self.get_all_folder_task(
            root_dir, dataset_name, set_name, min_image_long_side,
            max_image_long_side, min_aspect_ratio, max_aspect_ratio,
            min_t2i_caption_length, max_t2i_caption_length)

        self.image_name_list = []
        self.image_path_list = []
        self.image_caption_list = []
        image_bucket_array_list = []

        with Pool(processes=8) as pool:
            for per_folder_result in tqdm(pool.imap(
                    self.load_single_folder_annotation,
                    folder_task_list,
                    chunksize=1),
                                          total=len(folder_task_list)):
                per_folder_image_dir, per_folder_image_name_list, per_folder_image_caption_list, per_folder_image_bucket_array = per_folder_result

                self.image_name_list.extend(per_folder_image_name_list)
                self.image_path_list.extend(
                    os.path.join(per_folder_image_dir, per_image_name)
                    for per_image_name in per_folder_image_name_list)
                self.image_caption_list.extend(per_folder_image_caption_list)
                image_bucket_array_list.append(per_folder_image_bucket_array)

        if len(image_bucket_array_list) > 0:
            self.image_bucket_list = np.concatenate(image_bucket_array_list,
                                                    axis=0)
        else:
            self.image_bucket_list = np.array([], dtype=np.float64)

        assert len(self.image_name_list) == len(self.image_path_list) == len(
            self.image_caption_list) == len(self.image_bucket_list)

        self.transform = transform

        print(f'Dataset Size:{len(self.image_name_list)}')

    @staticmethod
    def get_all_folder_task(root_dir, dataset_name, set_name,
                            min_image_long_side, max_image_long_side,
                            min_aspect_ratio, max_aspect_ratio,
                            min_t2i_caption_length, max_t2i_caption_length):
        folder_task_list = []
        for per_dataset_name in tqdm(dataset_name):
            for per_set_name in set_name[per_dataset_name]:
                per_set_dir = os.path.join(root_dir, per_dataset_name,
                                           per_set_name)
                per_set_folder_name_list = []
                with os.scandir(per_set_dir) as per_scandir_iterator:
                    for per_dir_entry in per_scandir_iterator:
                        if not per_dir_entry.is_dir():
                            continue

                        per_set_folder_name_list.append(per_dir_entry.name)

                for per_folder_name in sorted(per_set_folder_name_list):
                    folder_task_list.append([
                        os.path.join(per_set_dir, per_folder_name),
                        os.path.join(per_set_dir, f'{per_folder_name}.json'),
                        min_image_long_side,
                        max_image_long_side,
                        min_aspect_ratio,
                        max_aspect_ratio,
                        min_t2i_caption_length,
                        max_t2i_caption_length,
                    ])

        return folder_task_list

    @staticmethod
    def load_single_folder_annotation(per_folder_task):
        (per_folder_image_dir, per_folder_text_json_path, min_image_long_side,
         max_image_long_side, min_aspect_ratio, max_aspect_ratio,
         min_t2i_caption_length, max_t2i_caption_length) = per_folder_task

        with open(per_folder_text_json_path, encoding='utf-8') as f:
            per_folder_text_dict = json.load(f)

        per_folder_image_name_list = []
        per_folder_image_caption_list = []
        per_folder_image_bucket_list = []

        for per_image_name in sorted(per_folder_text_dict.keys()):
            if not per_image_name.endswith('.jpg'):
                continue

            per_annotation = per_folder_text_dict[per_image_name]

            if not isinstance(per_annotation, dict):
                continue

            per_image_w = per_annotation['width']
            per_image_h = per_annotation['height']

            per_image_long_side = max(per_image_h, per_image_w)

            if per_image_long_side < min_image_long_side:
                continue
            if per_image_long_side > max_image_long_side:
                continue

            per_image_aspect_ratio = per_image_w / per_image_h

            if per_image_aspect_ratio < min_aspect_ratio:
                continue
            if per_image_aspect_ratio > max_aspect_ratio:
                continue

            per_t2i_caption_length = per_annotation['t2i_caption_length']

            if per_t2i_caption_length < min_t2i_caption_length:
                continue
            if per_t2i_caption_length > max_t2i_caption_length:
                continue

            per_bucket_ratio, per_bucket_index = get_closest_bucket(
                per_image_w, per_image_h, ASPECT_RATIO_BUCKETS)

            per_folder_image_name_list.append(per_image_name)
            per_folder_image_caption_list.append(per_annotation['t2i_caption'])
            per_folder_image_bucket_list.append([
                per_bucket_ratio,
                per_bucket_index,
            ])

        per_folder_image_bucket_array = np.array(per_folder_image_bucket_list,
                                                 dtype=np.float64).reshape(
                                                     -1, 2)

        return [
            per_folder_image_dir,
            per_folder_image_name_list,
            per_folder_image_caption_list,
            per_folder_image_bucket_array,
        ]

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        path = self.image_path_list[idx]

        image = self.load_image(idx)
        caption = self.load_caption(idx)

        bucket = self.image_bucket_list[idx]

        sample = {
            'image_path': path,
            'image': image,
            'caption': caption,
            'bucket': bucket,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, idx):
        image = cv2.imdecode(
            np.fromfile(self.image_path_list[idx], dtype=np.uint8),
            cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def load_caption(self, idx):
        caption = self.image_caption_list[idx]

        return caption


if __name__ == '__main__':
    import os
    import random
    import numpy as np
    import torch
    seed = 0
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    import os
    import sys

    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.append(BASE_DIR)

    from tools.path import t2i_dataset_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from SimpleGeneration.universal_generation_edit.t2i_common import Opencv2PIL, TorchAspectRatioBucketResize, TorchMeanStdNormalize, draw_caption_on_image

    # https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/SubsetOTF/SC/NotoSansSC-Regular.otf
    base_resize = 1024
    font_path = '/root/code/SimpleGeneration_pytorch_training_examples/SimpleGeneration/universal_generation_edit/datasets/NotoSansSC-Regular.otf'

    t2i_dataset = T2IDataset(
        root_dir=t2i_dataset_path,
        dataset_name=[
            'BM-6M',
            'FLUX-Reason-6M',
            'FaceID-6M',
            'GPIC',
            'MegaStyle-8M',
            'SACap-1M',
            'UNO-1M',
            'fine-t2i',
        ],
        set_name={
            'BM-6M': [
                'subset_1', 'subset_2', 'subset_3', 'subset_4', 'subset_5',
                'subset_6', 'subset_7', 'subset_8', 'subset_9'
            ],
            'FLUX-Reason-6M': [
                'aesthetics-part01_000', 'aesthetics-part01_001',
                'aesthetics-part01_002', 'aesthetics-part02_000',
                'aesthetics-part02_001', 'aesthetics-part02_002',
                'imaginative_000', 'text_000'
            ],
            'FaceID-6M': [
                'laion_512_000', 'laion_512_001', 'laion_512_002',
                'laion_512_003', 'laion_512_004'
            ],
            'GPIC': [
                'test', 'train_000', 'train_001', 'train_002', 'train_003',
                'train_004', 'train_005', 'train_006', 'train_007',
                'train_008', 'train_009', 'train_010', 'train_011',
                'train_012', 'train_013', 'train_014', 'train_015',
                'train_016', 'train_017', 'train_018', 'train_019',
                'train_020', 'train_021', 'train_022', 'train_023',
                'train_024', 'train_025', 'train_026', 'train_027',
                'train_028', 'train_029', 'train_030', 'train_031',
                'train_032', 'train_033', 'train_034', 'train_035',
                'train_036', 'train_037', 'train_038', 'train_039',
                'train_040', 'train_041', 'train_042', 'train_043',
                'train_044', 'train_045', 'train_046', 'train_047',
                'train_048', 'train_049', 'train_050', 'train_051',
                'train_052', 'train_053', 'train_054', 'train_055',
                'train_056', 'train_057', 'train_058', 'train_059',
                'train_060', 'train_061', 'train_062', 'train_063',
                'train_064', 'train_065', 'train_066', 'train_067',
                'train_068', 'train_069', 'train_070', 'train_071',
                'train_072', 'train_073', 'train_074', 'train_075',
                'train_076', 'train_077', 'train_078', 'train_079',
                'train_080', 'train_081', 'train_082', 'train_083',
                'train_084', 'train_085', 'train_086', 'train_087',
                'train_088', 'train_089', 'train_090', 'train_091',
                'train_092', 'train_093', 'train_094', 'train_095',
                'train_096', 'train_097', 'train_098', 'val'
            ],
            'MegaStyle-8M': [
                'train_000', 'train_001', 'train_002', 'train_003',
                'train_004', 'train_005', 'train_006', 'train_007'
            ],
            'SACap-1M': [
                'train_000',
            ],
            'UNO-1M': [
                'class_generation',
                'object365',
                'scene_prompt_object_object_v1',
                'scene_prompt_object_person_v1',
            ],
            'fine-t2i': [
                'curated_000',
                'synthetic_enhanced_prompt_random_resolution_000',
                'synthetic_enhanced_prompt_random_resolution_001',
                'synthetic_enhanced_prompt_square_resolution_000',
                'synthetic_enhanced_prompt_square_resolution_001',
                'synthetic_original_prompt_random_resolution_000',
                'synthetic_original_prompt_random_resolution_001',
                'synthetic_original_prompt_square_resolution_000',
                'synthetic_original_prompt_square_resolution_001',
            ],
        },
        min_image_long_side=64,
        max_image_long_side=4096,
        min_aspect_ratio=0.25,
        max_aspect_ratio=4.0,
        min_t2i_caption_length=4,
        max_t2i_caption_length=1536,
        transform=transforms.Compose([
            Opencv2PIL(),
            TorchAspectRatioBucketResize(base_resize=base_resize),
            TorchMeanStdNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]))

    count = 0
    for per_sample in tqdm(t2i_dataset):
        print('1111', per_sample['image_path'], per_sample['image'].shape,
              per_sample['image'].dtype)
        print('2222', per_sample['image'].min(), per_sample['image'].max())
        print('3333', per_sample['bucket'])

        per_bucket_ratio, per_bucket_index = float(
            per_sample['bucket'][0]), int(per_sample['bucket'][1])
        per_bucket_h, per_bucket_w = get_bucket_resolution(
            per_bucket_ratio, base_resize)

        assert per_sample['image'].shape[0] == per_bucket_h and per_sample[
            'image'].shape[1] == per_bucket_w

        temp_dir = './temp1'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        image = np.ascontiguousarray(np.round(
            (per_sample['image'] + 1.0) / 2.0 * 255.0).clip(0, 255),
                                     dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        caption = per_sample['caption']
        print('4444', caption)

        image = draw_caption_on_image(image, caption, font_path)

        cv2.imencode('.jpg', image)[1].tofile(
            os.path.join(
                temp_dir,
                f'idx_{count}_bucket_{per_bucket_index}_{per_bucket_h}x{per_bucket_w}.jpg'
            ))

        if count < 10:
            count += 1
        else:
            break
