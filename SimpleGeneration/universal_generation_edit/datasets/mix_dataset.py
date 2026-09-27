import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import cv2
import json
import random
import numpy as np

from PIL import Image, ImageOps

from multiprocessing import Pool

from tqdm import tqdm

from torch.utils.data import Dataset

from SimpleGeneration.universal_generation_edit.aspect_ratio_buckets import ASPECT_RATIO_BUCKETS, get_closest_bucket, get_bucket_resolution


class MixDataset(Dataset):

    def __init__(self,
                 t2i_root_dir,
                 ti2i_root_dir,
                 t2i_dataset_name=[
                     'SACap-1M',
                 ],
                 t2i_set_name={
                     'SACap-1M': [
                         'train_000',
                     ],
                 },
                 ti2i_dataset_name=[
                     'BM-6M',
                 ],
                 ti2i_set_name={
                     'BM-6M': [
                         'non_rigid_motions',
                     ],
                 },
                 min_image_long_side=64,
                 max_image_long_side=4096,
                 min_aspect_ratio=0.25,
                 max_aspect_ratio=4.0,
                 min_t2i_caption_length=4,
                 max_t2i_caption_length=1536,
                 min_ti2i_caption_length=4,
                 max_ti2i_caption_length=1024,
                 choose_ti2i_caption_type_prob={
                     'ConceptEdit-12M': {
                         'short_english_ti2i_caption': 0.25,
                         'short_chinese_ti2i_caption': 0.25,
                         'detailed_english_ti2i_caption': 0.25,
                         'detailed_chinese_ti2i_caption': 0.25,
                     },
                     'UnicEdit': {
                         'english_ti2i_caption': 0.5,
                         'chinese_ti2i_caption': 0.5,
                     },
                 },
                 min_reference_image_num=1,
                 max_reference_image_num=1,
                 transform=None):
        for per_dataset_name, per_choose_ti2i_caption_type_prob in choose_ti2i_caption_type_prob.items(
        ):
            assert abs(sum(per_choose_ti2i_caption_type_prob.values()) -
                       1.) < 1e-6

        t2i_folder_task_list = self.get_all_t2i_folder_task(
            t2i_root_dir, t2i_dataset_name, t2i_set_name, min_image_long_side,
            max_image_long_side, min_aspect_ratio, max_aspect_ratio,
            min_t2i_caption_length, max_t2i_caption_length)

        self.t2i_image_name_list = []
        self.t2i_image_path_list = []
        self.t2i_image_caption_list = []
        t2i_image_bucket_array_list = []

        with Pool(processes=8) as pool:
            for per_folder_result in tqdm(pool.imap(
                    self.load_single_t2i_folder_annotation,
                    t2i_folder_task_list,
                    chunksize=1),
                                          total=len(t2i_folder_task_list)):
                per_folder_image_dir, per_folder_image_name_list, per_folder_image_caption_list, per_folder_image_bucket_array = per_folder_result

                self.t2i_image_name_list.extend(per_folder_image_name_list)
                self.t2i_image_path_list.extend(
                    os.path.join(per_folder_image_dir, per_image_name)
                    for per_image_name in per_folder_image_name_list)
                self.t2i_image_caption_list.extend(
                    per_folder_image_caption_list)
                t2i_image_bucket_array_list.append(
                    per_folder_image_bucket_array)

        if len(t2i_image_bucket_array_list) > 0:
            self.t2i_image_bucket_list = np.concatenate(
                t2i_image_bucket_array_list, axis=0)
        else:
            self.t2i_image_bucket_list = np.array([], dtype=np.float64)

        assert len(self.t2i_image_name_list) == len(
            self.t2i_image_path_list) == len(
                self.t2i_image_caption_list) == len(self.t2i_image_bucket_list)

        ti2i_folder_task_list = self.get_all_ti2i_folder_task(
            ti2i_root_dir, ti2i_dataset_name, ti2i_set_name,
            min_image_long_side, max_image_long_side, min_aspect_ratio,
            max_aspect_ratio, min_ti2i_caption_length, max_ti2i_caption_length,
            choose_ti2i_caption_type_prob, min_reference_image_num,
            max_reference_image_num)

        self.ti2i_edited_image_name_list = []
        self.ti2i_edited_image_path_list = []
        self.ti2i_reference_image_path_list = []
        self.ti2i_image_caption_list = []
        self.ti2i_image_caption_type_prob_list = []
        ti2i_image_bucket_array_list = []

        with Pool(processes=8) as pool:
            for per_folder_result in tqdm(pool.imap(
                    self.load_single_ti2i_folder_annotation,
                    ti2i_folder_task_list,
                    chunksize=1),
                                          total=len(ti2i_folder_task_list)):
                per_folder_image_dir, per_folder_edited_image_name_list, per_folder_reference_image_name_list, per_folder_image_caption_list, per_folder_image_caption_type_prob_list, per_folder_image_bucket_array = per_folder_result

                self.ti2i_edited_image_name_list.extend(
                    per_folder_edited_image_name_list)
                for per_edited_image_name, per_reference_image_name_list in zip(
                        per_folder_edited_image_name_list,
                        per_folder_reference_image_name_list):
                    per_edit_pair_image_dir = os.path.join(
                        per_folder_image_dir,
                        per_edited_image_name[:-len('.jpg')])

                    self.ti2i_edited_image_path_list.append(
                        os.path.join(per_edit_pair_image_dir,
                                     per_edited_image_name))
                    self.ti2i_reference_image_path_list.append([
                        os.path.join(per_edit_pair_image_dir,
                                     per_reference_image_name)
                        for per_reference_image_name in
                        per_reference_image_name_list
                    ])
                self.ti2i_image_caption_list.extend(
                    per_folder_image_caption_list)
                self.ti2i_image_caption_type_prob_list.extend(
                    per_folder_image_caption_type_prob_list)
                ti2i_image_bucket_array_list.append(
                    per_folder_image_bucket_array)

        if len(ti2i_image_bucket_array_list) > 0:
            self.ti2i_image_bucket_list = np.concatenate(
                ti2i_image_bucket_array_list, axis=0)
        else:
            self.ti2i_image_bucket_list = np.array([], dtype=np.float64)

        assert len(self.ti2i_edited_image_name_list) == len(
            self.ti2i_edited_image_path_list) == len(
                self.ti2i_reference_image_path_list) == len(
                    self.ti2i_image_caption_list) == len(
                        self.ti2i_image_caption_type_prob_list) == len(
                            self.ti2i_image_bucket_list)

        self.transform = transform

        print(f'T2I Dataset Size:{len(self.t2i_image_name_list)}')
        print(f'TI2I Dataset Size:{len(self.ti2i_edited_image_name_list)}')

    @staticmethod
    def get_all_t2i_folder_task(root_dir, dataset_name, set_name,
                                min_image_long_side, max_image_long_side,
                                min_aspect_ratio, max_aspect_ratio,
                                min_t2i_caption_length,
                                max_t2i_caption_length):
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
    def get_all_ti2i_folder_task(
            root_dir, dataset_name, set_name, min_image_long_side,
            max_image_long_side, min_aspect_ratio, max_aspect_ratio,
            min_ti2i_caption_length, max_ti2i_caption_length,
            choose_ti2i_caption_type_prob, min_reference_image_num,
            max_reference_image_num):
        folder_task_list = []
        for per_dataset_name in tqdm(dataset_name):
            per_dataset_choose_ti2i_caption_type_prob = choose_ti2i_caption_type_prob.get(
                per_dataset_name, None)

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
                        min_ti2i_caption_length,
                        max_ti2i_caption_length,
                        per_dataset_choose_ti2i_caption_type_prob,
                        min_reference_image_num,
                        max_reference_image_num,
                    ])

        return folder_task_list

    @staticmethod
    def load_single_t2i_folder_annotation(per_folder_task):
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

    @staticmethod
    def load_single_ti2i_folder_annotation(per_folder_task):
        (per_folder_image_dir, per_folder_text_json_path, min_image_long_side,
         max_image_long_side, min_aspect_ratio, max_aspect_ratio,
         min_ti2i_caption_length, max_ti2i_caption_length,
         per_dataset_choose_ti2i_caption_type_prob, min_reference_image_num,
         max_reference_image_num) = per_folder_task

        with open(per_folder_text_json_path, encoding='utf-8') as f:
            per_folder_text_dict = json.load(f)

        per_folder_edited_image_name_list = []
        per_folder_reference_image_name_list = []
        per_folder_image_caption_list = []
        per_folder_image_caption_type_prob_list = []
        per_folder_image_bucket_list = []

        for per_edited_image_name in sorted(per_folder_text_dict.keys()):
            if not per_edited_image_name.endswith('.jpg'):
                continue

            per_annotation = per_folder_text_dict[per_edited_image_name]

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

            per_ti2i_caption_length = per_annotation['ti2i_caption_length']

            if per_dataset_choose_ti2i_caption_type_prob:
                per_ti2i_caption_length_list = [
                    per_ti2i_caption_length[f'{per_ti2i_caption_type}_length']
                    for per_ti2i_caption_type in
                    per_dataset_choose_ti2i_caption_type_prob.keys()
                ]
            else:
                per_ti2i_caption_length_list = [
                    per_ti2i_caption_length,
                ]

            if min(per_ti2i_caption_length_list) < min_ti2i_caption_length:
                continue
            if max(per_ti2i_caption_length_list) > max_ti2i_caption_length:
                continue

            per_reference_image_num = per_annotation['reference_image_num']

            if per_reference_image_num < 1:
                continue

            if per_reference_image_num < min_reference_image_num:
                continue
            if per_reference_image_num > max_reference_image_num:
                continue

            per_bucket_ratio, per_bucket_index = get_closest_bucket(
                per_image_w, per_image_h, ASPECT_RATIO_BUCKETS)

            per_folder_edited_image_name_list.append(per_edited_image_name)
            per_folder_reference_image_name_list.append(
                list(per_annotation['reference_image']))
            per_folder_image_caption_list.append(
                per_annotation['ti2i_caption'])
            per_folder_image_caption_type_prob_list.append(
                per_dataset_choose_ti2i_caption_type_prob)
            per_folder_image_bucket_list.append([
                per_bucket_ratio,
                per_bucket_index,
                per_reference_image_num,
            ])

        per_folder_image_bucket_array = np.array(per_folder_image_bucket_list,
                                                 dtype=np.float64).reshape(
                                                     -1, 3)

        return [
            per_folder_image_dir,
            per_folder_edited_image_name_list,
            per_folder_reference_image_name_list,
            per_folder_image_caption_list,
            per_folder_image_caption_type_prob_list,
            per_folder_image_bucket_array,
        ]

    def __len__(self):
        return 2 * len(self.ti2i_edited_image_name_list)

    def __getitem__(self, idx):
        sample_index, batch_task_type = idx

        assert batch_task_type in ['T2I', 'TI2I']

        if batch_task_type == 'T2I':
            t2i_image_idx = sample_index

            t2i_image_path = self.t2i_image_path_list[t2i_image_idx]

            t2i_image = self.load_t2i_image(t2i_image_idx)
            t2i_caption = self.load_t2i_caption(t2i_image_idx)

            t2i_bucket = self.t2i_image_bucket_list[t2i_image_idx]

            sample = {
                'image_path': t2i_image_path,
                'image': t2i_image,
                'caption': t2i_caption,
                'bucket': t2i_bucket,
                'task_type': 'T2I',
            }

        elif batch_task_type == 'TI2I':
            ti2i_image_idx = sample_index

            ti2i_image_path = self.ti2i_edited_image_path_list[ti2i_image_idx]
            ti2i_reference_image_path = self.ti2i_reference_image_path_list[
                ti2i_image_idx]

            ti2i_image = self.load_ti2i_image(ti2i_image_idx)
            ti2i_reference_image = self.load_ti2i_reference_image(
                ti2i_image_idx)
            ti2i_reference_pil_image = self.load_ti2i_reference_pil_image(
                ti2i_image_idx)
            ti2i_caption = self.load_ti2i_caption(ti2i_image_idx)

            ti2i_bucket = self.ti2i_image_bucket_list[ti2i_image_idx]

            sample = {
                'image_path': ti2i_image_path,
                'reference_image_path': ti2i_reference_image_path,
                'image': ti2i_image,
                'reference_image': ti2i_reference_image,
                'reference_pil_image': ti2i_reference_pil_image,
                'caption': ti2i_caption,
                'bucket': ti2i_bucket,
                'task_type': 'TI2I',
            }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_t2i_image(self, idx):
        image = cv2.imdecode(
            np.fromfile(self.t2i_image_path_list[idx], dtype=np.uint8),
            cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def load_t2i_caption(self, idx):
        caption = self.t2i_image_caption_list[idx]

        return caption

    def load_ti2i_image(self, idx):
        image = cv2.imdecode(
            np.fromfile(self.ti2i_edited_image_path_list[idx], dtype=np.uint8),
            cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def load_ti2i_reference_image(self, idx):
        reference_image = []
        for per_reference_image_path in self.ti2i_reference_image_path_list[
                idx]:
            per_reference_image = cv2.imdecode(
                np.fromfile(per_reference_image_path, dtype=np.uint8),
                cv2.IMREAD_COLOR)
            per_reference_image = cv2.cvtColor(per_reference_image,
                                               cv2.COLOR_BGR2RGB)
            reference_image.append(per_reference_image.astype(np.float32))

        return reference_image

    def load_ti2i_reference_pil_image(self, idx):
        reference_pil_image = []
        for per_reference_image_path in self.ti2i_reference_image_path_list[
                idx]:
            with Image.open(per_reference_image_path) as per_reference_image:
                per_reference_pil_image = ImageOps.exif_transpose(
                    per_reference_image).convert('RGB')
            reference_pil_image.append(per_reference_pil_image)

        return reference_pil_image

    def load_ti2i_caption(self, idx):
        caption = self.ti2i_image_caption_list[idx]
        per_dataset_choose_ti2i_caption_type_prob = self.ti2i_image_caption_type_prob_list[
            idx]

        if per_dataset_choose_ti2i_caption_type_prob:
            random_prob, accumulate_prob = random.uniform(0., 1.), 0.
            for per_ti2i_caption_type, per_ti2i_caption_type_prob in per_dataset_choose_ti2i_caption_type_prob.items(
            ):
                accumulate_prob += per_ti2i_caption_type_prob
                if random_prob < accumulate_prob:
                    break

            caption = caption[per_ti2i_caption_type]

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

    from tools.path import t2i_dataset_path, ti2i_dataset_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from SimpleGeneration.universal_generation_edit.mix_common import Opencv2PIL, TorchAspectRatioBucketResize, TorchMeanStdNormalize, MixCollator, MixBucketBatchSampler, draw_caption_on_image

    # https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/SubsetOTF/SC/NotoSansSC-Regular.otf
    base_resize = 1024
    font_path = '/root/code/SimpleGeneration_pytorch_training_examples/SimpleGeneration/universal_generation_edit/datasets/NotoSansSC-Regular.otf'

    mix_dataset = MixDataset(
        t2i_root_dir=t2i_dataset_path,
        ti2i_root_dir=ti2i_dataset_path,
        t2i_dataset_name=[
            'BM-6M',
            'FLUX-Reason-6M',
            'FaceID-6M',
            'GPIC',
            'MegaStyle-8M',
            'SACap-1M',
            'UNO-1M',
            'fine-t2i',
        ],
        t2i_set_name={
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
        ti2i_dataset_name=[
            'BM-6M',
            'ConceptEdit-12M',
            'CrispEdit-2M',
            'FoundIR',
            'ImgEdit',
            'InterEdit',
            # 'InterEdit-Mask',
            'ScaleEdit',
            'UnicEdit',
            'VINS120K',
            'X2Edit',
        ],
        ti2i_set_name={
            'BM-6M': [
                'non_rigid_motions',
            ],
            'ConceptEdit-12M': [
                'acg_and_entertainment_enhanced_prompt_random_resolution',
                'acg_and_entertainment_enhanced_prompt_square_resolution',
                'acg_and_entertainment_original_prompt_random_resolution',
                'acg_and_entertainment_original_prompt_square_resolution',
                'add_or_remove_object_enhanced_prompt_random_resolution',
                'add_or_remove_object_enhanced_prompt_square_resolution',
                'add_or_remove_object_original_prompt_random_resolution',
                'add_or_remove_object_original_prompt_square_resolution',
                'background_management_enhanced_prompt_random_resolution',
                'background_management_enhanced_prompt_square_resolution',
                'background_management_original_prompt_random_resolution',
                'background_management_original_prompt_square_resolution',
                'beauty_retouching_enhanced_prompt_random_resolution',
                'beauty_retouching_enhanced_prompt_square_resolution',
                'beauty_retouching_original_prompt_random_resolution',
                'beauty_retouching_original_prompt_square_resolution',
                'body_reshaping_enhanced_prompt_random_resolution',
                'body_reshaping_enhanced_prompt_square_resolution',
                'body_reshaping_original_prompt_random_resolution',
                'body_reshaping_original_prompt_square_resolution',
                'color_and_material_change_enhanced_prompt_random_resolution',
                'color_and_material_change_enhanced_prompt_square_resolution',
                'color_and_material_change_original_prompt_random_resolution',
                'color_and_material_change_original_prompt_square_resolution',
                'complex_instruction_following_enhanced_prompt_random_resolution',
                'complex_instruction_following_enhanced_prompt_square_resolution',
                'complex_instruction_following_original_prompt_random_resolution',
                'complex_instruction_following_original_prompt_square_resolution',
                'crop_and_composition_enhanced_prompt_random_resolution',
                'crop_and_composition_enhanced_prompt_square_resolution',
                'crop_and_composition_original_prompt_random_resolution',
                'crop_and_composition_original_prompt_square_resolution',
                'denoise_and_deblur_enhanced_prompt_random_resolution',
                'denoise_and_deblur_enhanced_prompt_square_resolution',
                'denoise_and_deblur_original_prompt_random_resolution',
                'denoise_and_deblur_original_prompt_square_resolution',
                'detail_restoration_enhanced_prompt_random_resolution',
                'detail_restoration_enhanced_prompt_square_resolution',
                'detail_restoration_original_prompt_random_resolution',
                'detail_restoration_original_prompt_square_resolution',
                'document_and_education_enhanced_prompt_random_resolution',
                'document_and_education_enhanced_prompt_square_resolution',
                'document_and_education_original_prompt_random_resolution',
                'document_and_education_original_prompt_square_resolution',
                'ecommerce_and_marketing_enhanced_prompt_random_resolution',
                'ecommerce_and_marketing_enhanced_prompt_square_resolution',
                'ecommerce_and_marketing_original_prompt_random_resolution',
                'ecommerce_and_marketing_original_prompt_square_resolution',
                'emotion_and_expression_control_enhanced_prompt_random_resolution',
                'emotion_and_expression_control_enhanced_prompt_square_resolution',
                'emotion_and_expression_control_original_prompt_random_resolution',
                'emotion_and_expression_control_original_prompt_square_resolution',
                'environment_and_style_enhanced_prompt_random_resolution',
                'environment_and_style_enhanced_prompt_square_resolution',
                'environment_and_style_original_prompt_random_resolution',
                'environment_and_style_original_prompt_square_resolution',
                'environment_and_weather_simulation_enhanced_prompt_random_resolution',
                'environment_and_weather_simulation_enhanced_prompt_square_resolution',
                'environment_and_weather_simulation_original_prompt_random_resolution',
                'environment_and_weather_simulation_original_prompt_square_resolution',
                'exposure_and_color_correction_enhanced_prompt_random_resolution',
                'exposure_and_color_correction_enhanced_prompt_square_resolution',
                'exposure_and_color_correction_original_prompt_random_resolution',
                'exposure_and_color_correction_original_prompt_square_resolution',
                'facial_attribute_editing_enhanced_prompt_random_resolution',
                'facial_attribute_editing_enhanced_prompt_square_resolution',
                'facial_attribute_editing_original_prompt_random_resolution',
                'facial_attribute_editing_original_prompt_square_resolution',
                'font_and_style_enhanced_prompt_random_resolution',
                'font_and_style_enhanced_prompt_square_resolution',
                'font_and_style_original_prompt_random_resolution',
                'font_and_style_original_prompt_square_resolution',
                'gaze_and_expression_repair_enhanced_prompt_random_resolution',
                'gaze_and_expression_repair_enhanced_prompt_square_resolution',
                'gaze_and_expression_repair_original_prompt_random_resolution',
                'gaze_and_expression_repair_original_prompt_square_resolution',
                'global_lighting_control_enhanced_prompt_random_resolution',
                'global_lighting_control_enhanced_prompt_square_resolution',
                'global_lighting_control_original_prompt_random_resolution',
                'global_lighting_control_original_prompt_square_resolution',
                'group_photo_composition_enhanced_prompt_random_resolution',
                'group_photo_composition_enhanced_prompt_square_resolution',
                'group_photo_composition_original_prompt_random_resolution',
                'group_photo_composition_original_prompt_square_resolution',
                'hair_editing_enhanced_prompt_random_resolution',
                'hair_editing_enhanced_prompt_square_resolution',
                'hair_editing_original_prompt_random_resolution',
                'hair_editing_original_prompt_square_resolution',
                'image_outpainting_enhanced_prompt_random_resolution',
                'image_outpainting_enhanced_prompt_square_resolution',
                'image_outpainting_original_prompt_random_resolution',
                'image_outpainting_original_prompt_square_resolution',
                'layout_and_logo_enhanced_prompt_random_resolution',
                'layout_and_logo_enhanced_prompt_square_resolution',
                'layout_and_logo_original_prompt_random_resolution',
                'layout_and_logo_original_prompt_square_resolution',
                'logical_reasoning_generation_enhanced_prompt_random_resolution',
                'logical_reasoning_generation_enhanced_prompt_square_resolution',
                'logical_reasoning_generation_original_prompt_random_resolution',
                'logical_reasoning_generation_original_prompt_square_resolution',
                'matting_and_layer_enhanced_prompt_random_resolution',
                'matting_and_layer_enhanced_prompt_square_resolution',
                'matting_and_layer_original_prompt_random_resolution',
                'matting_and_layer_original_prompt_square_resolution',
                'multi_image_composition_enhanced_prompt_random_resolution',
                'multi_image_composition_enhanced_prompt_square_resolution',
                'multi_image_composition_original_prompt_random_resolution',
                'multi_image_composition_original_prompt_square_resolution',
                'pose_and_action_driven_enhanced_prompt_random_resolution',
                'pose_and_action_driven_enhanced_prompt_square_resolution',
                'pose_and_action_driven_original_prompt_random_resolution',
                'pose_and_action_driven_original_prompt_square_resolution',
                'reference_image_driven_enhanced_prompt_random_resolution',
                'reference_image_driven_enhanced_prompt_square_resolution',
                'reference_image_driven_original_prompt_random_resolution',
                'reference_image_driven_original_prompt_square_resolution',
                'replace_object_enhanced_prompt_random_resolution',
                'replace_object_enhanced_prompt_square_resolution',
                'replace_object_original_prompt_random_resolution',
                'replace_object_original_prompt_square_resolution',
                'sketch_and_scribble_control_enhanced_prompt_random_resolution',
                'sketch_and_scribble_control_enhanced_prompt_square_resolution',
                'sketch_and_scribble_control_original_prompt_random_resolution',
                'sketch_and_scribble_control_original_prompt_square_resolution',
                'spatial_and_geometric_transform_enhanced_prompt_random_resolution',
                'spatial_and_geometric_transform_enhanced_prompt_square_resolution',
                'spatial_and_geometric_transform_original_prompt_random_resolution',
                'spatial_and_geometric_transform_original_prompt_square_resolution',
                'style_transfer_enhanced_prompt_random_resolution',
                'style_transfer_enhanced_prompt_square_resolution',
                'style_transfer_original_prompt_random_resolution',
                'style_transfer_original_prompt_square_resolution',
                'super_resolution_enhanced_prompt_random_resolution',
                'super_resolution_enhanced_prompt_square_resolution',
                'super_resolution_original_prompt_random_resolution',
                'super_resolution_original_prompt_square_resolution',
                'text_modification_and_generation_enhanced_prompt_random_resolution',
                'text_modification_and_generation_enhanced_prompt_square_resolution',
                'text_modification_and_generation_original_prompt_random_resolution',
                'text_modification_and_generation_original_prompt_square_resolution',
                'text_removal_and_dewatermark_enhanced_prompt_random_resolution',
                'text_removal_and_dewatermark_enhanced_prompt_square_resolution',
                'text_removal_and_dewatermark_original_prompt_random_resolution',
                'text_removal_and_dewatermark_original_prompt_square_resolution',
                'viewpoint_transformation_enhanced_prompt_random_resolution',
                'viewpoint_transformation_enhanced_prompt_square_resolution',
                'viewpoint_transformation_original_prompt_random_resolution',
                'viewpoint_transformation_original_prompt_square_resolution',
                'virtual_try_on_enhanced_prompt_random_resolution',
                'virtual_try_on_enhanced_prompt_square_resolution',
                'virtual_try_on_original_prompt_random_resolution',
                'virtual_try_on_original_prompt_square_resolution',
            ],
            'CrispEdit-2M': [
                'add', 'background_change', 'color', 'remove', 'replace',
                'style'
            ],
            'FoundIR': [
                'blur', 'blur_jpeg', 'blur_noise', 'blur_noise_jpeg', 'haze',
                'jpeg', 'lowlight', 'lowlight_blur', 'lowlight_haze',
                'lowlight_jpeg', 'lowlight_noise', 'night_rain', 'noise',
                'noise_jpeg', 'rain', 'rain_haze', 'raindrop'
            ],
            'ImgEdit': [
                'action', 'add', 'adjust_canny', 'background', 'hybrid',
                'reference_replace', 'remove', 'replace', 'style_transfer'
            ],
            'InterEdit': ['add', 'local', 'remove', 'texture'],
            'InterEdit-Mask': ['add', 'local', 'remove', 'texture'],
            'ScaleEdit': [
                'action_editing', 'background_replacement',
                'building_surface_text_editing', 'color_change',
                'compositional_editing', 'count_change',
                'gui_interface_text_editing', 'material_change',
                'movie_poster_text_editing', 'object_addition',
                'object_removal', 'object_replacement',
                'object_surface_text_editing', 'perceptual_reasoning',
                'scientific_reasoning', 'size_change', 'social_reasoning',
                'style_transfer', 'symbolic_reasoning', 'tone_adjustment',
                'viewpoint_transformation', 'visual_beautification'
            ],
            'UnicEdit': [
                'background_change', 'color_alteration',
                'compound_operation_edits', 'counting_change',
                'material_modification', 'motion_change',
                'multi-object_coordination', 'object_extraction',
                'portrait_editing', 'relation_change', 'shape-size_alteration',
                'spatial_reasoning_edits', 'style_transfer',
                'subject_addition', 'subject_removal', 'subject_replacement',
                'text_modification', 'texture_editing', 'tone_transformation',
                'viewpoint_transformation'
            ],
            'VINS120K': [
                'action_change', 'background_change', 'camera_movement',
                'color_change', 'material_change', 'object_movement',
                'personalized_generation', 'style_change', 'subject_addition',
                'subject_deletion', 'subject_replacement', 'text_change',
                'tone_transform'
            ],
            'X2Edit': [
                'action_change', 'background_change', 'camera_movement',
                'color_change', 'material_change', 'reasoning', 'style_change',
                'subject_addition', 'subject_deletion', 'subject_replacement',
                'text_change', 'tone_transform'
            ],
        },
        min_image_long_side=64,
        max_image_long_side=4096,
        min_aspect_ratio=0.25,
        max_aspect_ratio=4.0,
        min_t2i_caption_length=4,
        max_t2i_caption_length=1536,
        min_ti2i_caption_length=4,
        max_ti2i_caption_length=1024,
        choose_ti2i_caption_type_prob={
            'ConceptEdit-12M': {
                'short_english_ti2i_caption': 0.25,
                'short_chinese_ti2i_caption': 0.25,
                'detailed_english_ti2i_caption': 0.25,
                'detailed_chinese_ti2i_caption': 0.25,
            },
            'UnicEdit': {
                'english_ti2i_caption': 0.5,
                'chinese_ti2i_caption': 0.5,
            },
        },
        min_reference_image_num=1,
        max_reference_image_num=1,
        transform=transforms.Compose([
            Opencv2PIL(),
            TorchAspectRatioBucketResize(base_resize=base_resize),
            TorchMeanStdNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]))

    count = 0
    for per_sample_index in tqdm(range(len(mix_dataset))):
        per_t2i_sample = mix_dataset[(per_sample_index, 'T2I')]
        per_ti2i_sample = mix_dataset[(per_sample_index, 'TI2I')]

        print('0000', per_t2i_sample['task_type'],
              per_ti2i_sample['task_type'])

        print('1111', per_t2i_sample['image_path'],
              per_t2i_sample['image'].shape, per_t2i_sample['image'].dtype)
        print('2222', per_t2i_sample['image'].min(),
              per_t2i_sample['image'].max())
        print('3333', per_t2i_sample['bucket'])

        print('4444', per_ti2i_sample['image_path'],
              per_ti2i_sample['image'].shape, per_ti2i_sample['image'].dtype)
        print('5555', per_ti2i_sample['reference_image_path'], [
            per_reference_image.shape
            for per_reference_image in per_ti2i_sample['reference_image']
        ])
        print('6666', [
            per_reference_pil_image.size for per_reference_pil_image in
            per_ti2i_sample['reference_pil_image']
        ])
        print('7777', per_ti2i_sample['image'].min(),
              per_ti2i_sample['image'].max())
        print('8888', per_ti2i_sample['bucket'])

        per_t2i_bucket_ratio, per_t2i_bucket_index = float(
            per_t2i_sample['bucket'][0]), int(per_t2i_sample['bucket'][1])
        per_t2i_bucket_h, per_t2i_bucket_w = get_bucket_resolution(
            per_t2i_bucket_ratio, base_resize)

        assert per_t2i_sample['image'].shape[
            0] == per_t2i_bucket_h and per_t2i_sample['image'].shape[
                1] == per_t2i_bucket_w

        per_ti2i_bucket_ratio, per_ti2i_bucket_index, per_reference_image_num = float(
            per_ti2i_sample['bucket'][0]), int(
                per_ti2i_sample['bucket'][1]), int(
                    per_ti2i_sample['bucket'][2])
        per_ti2i_bucket_h, per_ti2i_bucket_w = get_bucket_resolution(
            per_ti2i_bucket_ratio, base_resize)

        assert per_reference_image_num == len(
            per_ti2i_sample['reference_image'])
        assert per_ti2i_sample['image'].shape[
            0] == per_ti2i_bucket_h and per_ti2i_sample['image'].shape[
                1] == per_ti2i_bucket_w

        temp_dir = './temp3'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        t2i_image = np.ascontiguousarray(np.round(
            (per_t2i_sample['image'] + 1.0) / 2.0 * 255.0).clip(0, 255),
                                         dtype=np.uint8)
        t2i_image = cv2.cvtColor(t2i_image, cv2.COLOR_RGB2BGR)
        t2i_caption = per_t2i_sample['caption']
        print('aaaa', t2i_caption)

        t2i_image = draw_caption_on_image(t2i_image, t2i_caption, font_path)

        cv2.imencode('.jpg', t2i_image)[1].tofile(
            os.path.join(
                temp_dir,
                f'idx_{count}_bucket_{per_t2i_bucket_index}_{per_t2i_bucket_h}x{per_t2i_bucket_w}_t2i.jpg'
            ))

        ti2i_image = np.ascontiguousarray(np.round(
            (per_ti2i_sample['image'] + 1.0) / 2.0 * 255.0).clip(0, 255),
                                          dtype=np.uint8)
        ti2i_image = cv2.cvtColor(ti2i_image, cv2.COLOR_RGB2BGR)
        ti2i_caption = per_ti2i_sample['caption']
        print('bbbb', ti2i_caption)

        ti2i_image = draw_caption_on_image(ti2i_image, ti2i_caption, font_path)

        cv2.imencode('.jpg', ti2i_image)[1].tofile(
            os.path.join(
                temp_dir,
                f'idx_{count}_bucket_{per_ti2i_bucket_index}_{per_ti2i_bucket_h}x{per_ti2i_bucket_w}_edited.jpg'
            ))

        for per_reference_image_index, per_reference_image in enumerate(
                per_ti2i_sample['reference_image']):
            per_reference_image = np.ascontiguousarray(np.round(
                (per_reference_image + 1.0) / 2.0 * 255.0).clip(0, 255),
                                                       dtype=np.uint8)
            per_reference_image = cv2.cvtColor(per_reference_image,
                                               cv2.COLOR_RGB2BGR)

            # only draw caption on the first reference image
            if per_reference_image_index == 0:
                per_reference_image = draw_caption_on_image(
                    per_reference_image, ti2i_caption, font_path)

            cv2.imencode('.jpg', per_reference_image)[1].tofile(
                os.path.join(
                    temp_dir,
                    f'idx_{count}_bucket_{per_ti2i_bucket_index}_reference_{per_reference_image_index}.jpg'
                ))

        if count < 10:
            count += 1
        else:
            break
