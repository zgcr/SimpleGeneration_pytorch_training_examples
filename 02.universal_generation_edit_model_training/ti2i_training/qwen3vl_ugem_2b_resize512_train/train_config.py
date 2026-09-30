import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import ti2i_dataset_path

from SimpleGeneration.universal_generation_edit.losses import SigmaAwareClippedMSELoss
from SimpleGeneration.universal_generation_edit.datasets.ti2i_dataset import TI2IDataset
from SimpleGeneration.universal_generation_edit.models.qwen3vl_universal_generation_edit_model_train import QWEN3VLUniversalGenerationEditModel
from SimpleGeneration.universal_generation_edit.models.scheduler import FlowMatchingResolutionShiftTimestepSampler
from SimpleGeneration.universal_generation_edit.models.tokenizer import Qwen3VLGenerationTokenizer
from SimpleGeneration.universal_generation_edit.ti2i_common import Opencv2PIL, TorchAspectRatioBucketResize, TorchMeanStdNormalize, TI2ICollator, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'qwen3vl_universal_generation_edit_model'
    denoise_model_type = 'SingleStreamMMDiT_2B'
    task_type = 'TI2I'
    base_resize = 512

    vlm_model_path = 'Qwen/Qwen3-VL-4B-Instruct'
    tokenizer = Qwen3VLGenerationTokenizer(vlm_model_path)

    model = QWEN3VLUniversalGenerationEditModel(
        **{
            'denoise_model_type': denoise_model_type,
            'vlm_model_path': vlm_model_path,
            'deepstack_layers': (9, 18, 36),
            'max_ref_images': 5,
            'ref_time_coord_scale': 20,
            'cfg_dropout_prob': 0.1,
            'use_gradient_checkpoint': True,
            'attention_backend': 'flash_varlen',
        })

    trained_ae_model_path = '/root/autodl-tmp/pretrained_models/flux2_convert_from_pytorch_official_weights/FLUX.2-dev-ae_convert_from_pytorch_official_weight.pth'
    load_state_dict(trained_ae_model_path, model.ae)

    # load total pretrained model or not
    trained_model_path = ''
    load_state_dict(trained_model_path, model)

    train_timestep_sampler = FlowMatchingResolutionShiftTimestepSampler(
        num_train_timesteps=1000, snr_sigma=1.0, eps=1e-4)

    train_criterion = SigmaAwareClippedMSELoss(weighting_scheme='cosmap',
                                               threshold=50.0)

    train_dataset = TI2IDataset(
        root_dir=ti2i_dataset_path,
        dataset_name=[
            'BM-6M',
            'ConceptEdit-12M',
            'CrispEdit-2M',
            'FoundIR',
            'ImgEdit',
            'InterEdit',
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
    train_collater = TI2ICollator(pil_resize=512)

    seed = 0

    # batch_size is total size
    batch_size = 4
    # num_workers is total workers
    num_workers = 32
    accumulation_steps = 1

    optimizer = (
        'MuonAdamW',
        {
            'lr': 1e-4,
            'weight_decay': 0,
            'global_weight_decay': False,
            # Muon orthogonalizes whole 2D weight matrices, which is wrong for
            # the LoRA adapters (low-rank factors) and for the vision->LLM
            # merger projections, so the VLM side falls back to AdamW. The AE
            # is frozen and never reaches the optimizer at all.
            'exclude_muon_layer_name_list': [
                'vlm',
                'lora',
                'merger',
            ],
        },
    )

    scheduler = (
        'CosineLR',
        {
            'warm_up_epochs': 0,
            'min_lr': 5e-6,
        },
    )

    epochs = 10
    print_interval = 100
    save_interval = 1

    use_step_save_interval = False
    step_save_interval = 10000

    # torch.float16 or torch.bfloat16
    amp_type = torch.bfloat16

    sync_bn = False
    use_amp = True
    use_compile = False
    compile_params = {
        # 'default': optimizes for large models, low compile-time and no extra memory usage.
        # 'reduce-overhead': optimizes to reduce the framework overhead and uses some extra memory, helps speed up small models, model update may not correct.
        # 'max-autotune': optimizes to produce the fastest model, but takes a very long time to compile and may failed.
        'mode': 'default',
    }

    find_unused_parameters = True
