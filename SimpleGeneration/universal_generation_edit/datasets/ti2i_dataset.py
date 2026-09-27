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


class TI2IDataset(Dataset):

    def __init__(self,
                 root_dir,
                 dataset_name=[
                     'BM-6M',
                 ],
                 set_name={
                     'BM-6M': [
                         'non_rigid_motions',
                     ],
                 },
                 min_image_long_side=64,
                 max_image_long_side=4096,
                 min_aspect_ratio=0.25,
                 max_aspect_ratio=4.0,
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

        folder_task_list = self.get_all_folder_task(
            root_dir, dataset_name, set_name, min_image_long_side,
            max_image_long_side, min_aspect_ratio, max_aspect_ratio,
            min_ti2i_caption_length, max_ti2i_caption_length,
            choose_ti2i_caption_type_prob, min_reference_image_num,
            max_reference_image_num)

        self.edited_image_name_list = []
        self.edited_image_path_list = []
        self.reference_image_path_list = []
        self.image_caption_list = []
        self.image_caption_type_prob_list = []
        image_bucket_array_list = []

        with Pool(processes=8) as pool:
            for per_folder_result in tqdm(pool.imap(
                    self.load_single_folder_annotation,
                    folder_task_list,
                    chunksize=1),
                                          total=len(folder_task_list)):
                per_folder_image_dir, per_folder_edited_image_name_list, per_folder_reference_image_name_list, per_folder_image_caption_list, per_folder_image_caption_type_prob_list, per_folder_image_bucket_array = per_folder_result

                self.edited_image_name_list.extend(
                    per_folder_edited_image_name_list)
                for per_edited_image_name, per_reference_image_name_list in zip(
                        per_folder_edited_image_name_list,
                        per_folder_reference_image_name_list):
                    per_edit_pair_image_dir = os.path.join(
                        per_folder_image_dir,
                        per_edited_image_name[:-len('.jpg')])

                    self.edited_image_path_list.append(
                        os.path.join(per_edit_pair_image_dir,
                                     per_edited_image_name))
                    self.reference_image_path_list.append([
                        os.path.join(per_edit_pair_image_dir,
                                     per_reference_image_name)
                        for per_reference_image_name in
                        per_reference_image_name_list
                    ])
                self.image_caption_list.extend(per_folder_image_caption_list)
                self.image_caption_type_prob_list.extend(
                    per_folder_image_caption_type_prob_list)
                image_bucket_array_list.append(per_folder_image_bucket_array)

        if len(image_bucket_array_list) > 0:
            self.image_bucket_list = np.concatenate(image_bucket_array_list,
                                                    axis=0)
        else:
            self.image_bucket_list = np.array([], dtype=np.float64)

        assert len(self.edited_image_name_list) == len(
            self.edited_image_path_list) == len(
                self.reference_image_path_list) == len(
                    self.image_caption_list) == len(
                        self.image_caption_type_prob_list) == len(
                            self.image_bucket_list)

        self.transform = transform

        print(f'Dataset Size:{len(self.edited_image_name_list)}')

    @staticmethod
    def get_all_folder_task(root_dir, dataset_name, set_name,
                            min_image_long_side, max_image_long_side,
                            min_aspect_ratio, max_aspect_ratio,
                            min_ti2i_caption_length, max_ti2i_caption_length,
                            choose_ti2i_caption_type_prob,
                            min_reference_image_num, max_reference_image_num):
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
    def load_single_folder_annotation(per_folder_task):
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
        return len(self.edited_image_name_list)

    def __getitem__(self, idx):
        path = self.edited_image_path_list[idx]
        reference_path = self.reference_image_path_list[idx]

        image = self.load_image(idx)
        reference_image = self.load_reference_image(idx)
        reference_pil_image = self.load_reference_pil_image(idx)
        caption = self.load_caption(idx)

        bucket = self.image_bucket_list[idx]

        sample = {
            'image_path': path,
            'reference_image_path': reference_path,
            'image': image,
            'reference_image': reference_image,
            'reference_pil_image': reference_pil_image,
            'caption': caption,
            'bucket': bucket,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, idx):
        image = cv2.imdecode(
            np.fromfile(self.edited_image_path_list[idx], dtype=np.uint8),
            cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def load_reference_image(self, idx):
        reference_image = []
        for per_reference_image_path in self.reference_image_path_list[idx]:
            per_reference_image = cv2.imdecode(
                np.fromfile(per_reference_image_path, dtype=np.uint8),
                cv2.IMREAD_COLOR)
            per_reference_image = cv2.cvtColor(per_reference_image,
                                               cv2.COLOR_BGR2RGB)
            reference_image.append(per_reference_image.astype(np.float32))

        return reference_image

    def load_reference_pil_image(self, idx):
        reference_pil_image = []
        for per_reference_image_path in self.reference_image_path_list[idx]:
            with Image.open(per_reference_image_path) as per_reference_image:
                per_reference_pil_image = ImageOps.exif_transpose(
                    per_reference_image).convert('RGB')
            reference_pil_image.append(per_reference_pil_image)

        return reference_pil_image

    def load_caption(self, idx):
        caption = self.image_caption_list[idx]
        per_dataset_choose_ti2i_caption_type_prob = self.image_caption_type_prob_list[
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

    from tools.path import ti2i_dataset_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from SimpleGeneration.universal_generation_edit.ti2i_common import Opencv2PIL, TorchAspectRatioBucketResize, TorchMeanStdNormalize, TI2ICollator, TI2IBucketBatchSampler, draw_caption_on_image

    # https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/SubsetOTF/SC/NotoSansSC-Regular.otf
    base_resize = 1024
    font_path = '/root/code/SimpleGeneration_pytorch_training_examples/SimpleGeneration/universal_generation_edit/datasets/NotoSansSC-Regular.otf'

    ti2i_dataset = TI2IDataset(
        root_dir=ti2i_dataset_path,
        dataset_name=[
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
        set_name={
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
    for per_sample in tqdm(ti2i_dataset):
        print('1111', per_sample['image_path'], per_sample['image'].shape,
              per_sample['image'].dtype)
        print('2222', per_sample['reference_image_path'], [
            per_reference_image.shape
            for per_reference_image in per_sample['reference_image']
        ])
        print('3333', [
            per_reference_pil_image.size
            for per_reference_pil_image in per_sample['reference_pil_image']
        ])
        print('4444', per_sample['image'].min(), per_sample['image'].max())
        print('5555', per_sample['bucket'])

        per_bucket_ratio, per_bucket_index, per_reference_image_num = float(
            per_sample['bucket'][0]), int(per_sample['bucket'][1]), int(
                per_sample['bucket'][2])
        per_bucket_h, per_bucket_w = get_bucket_resolution(
            per_bucket_ratio, base_resize)

        assert per_reference_image_num == len(per_sample['reference_image'])
        assert per_sample['image'].shape[0] == per_bucket_h and per_sample[
            'image'].shape[1] == per_bucket_w

        temp_dir = './temp2'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        image = np.ascontiguousarray(np.round(
            (per_sample['image'] + 1.0) / 2.0 * 255.0).clip(0, 255),
                                     dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        caption = per_sample['caption']
        print('6666', caption)

        image = draw_caption_on_image(image, caption, font_path)

        cv2.imencode('.jpg', image)[1].tofile(
            os.path.join(
                temp_dir,
                f'idx_{count}_bucket_{per_bucket_index}_{per_bucket_h}x{per_bucket_w}_edited.jpg'
            ))

        for per_reference_image_index, per_reference_image in enumerate(
                per_sample['reference_image']):
            per_reference_image = np.ascontiguousarray(np.round(
                (per_reference_image + 1.0) / 2.0 * 255.0).clip(0, 255),
                                                       dtype=np.uint8)
            per_reference_image = cv2.cvtColor(per_reference_image,
                                               cv2.COLOR_RGB2BGR)

            # only draw caption on the first reference image
            if per_reference_image_index == 0:
                per_reference_image = draw_caption_on_image(
                    per_reference_image, caption, font_path)

            cv2.imencode('.jpg', per_reference_image)[1].tofile(
                os.path.join(
                    temp_dir,
                    f'idx_{count}_bucket_{per_bucket_index}_reference_{per_reference_image_index}.jpg'
                ))

        if count < 10:
            count += 1
        else:
            break
