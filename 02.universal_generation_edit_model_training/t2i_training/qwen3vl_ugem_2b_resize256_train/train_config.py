import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import t2i_dataset_path

from SimpleGeneration.universal_generation_edit.losses import SigmaAwareClippedMSELoss
from SimpleGeneration.universal_generation_edit.datasets.t2i_dataset import T2IDataset
from SimpleGeneration.universal_generation_edit.models.qwen3vl_universal_generation_edit_model_train import QWEN3VLUniversalGenerationEditModel
from SimpleGeneration.universal_generation_edit.models.scheduler import FlowMatchingResolutionShiftTimestepSampler
from SimpleGeneration.universal_generation_edit.models.tokenizer import Qwen3VLGenerationTokenizer
from SimpleGeneration.universal_generation_edit.t2i_common import Opencv2PIL, TorchAspectRatioBucketResize, TorchMeanStdNormalize, T2ICollator, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'qwen3vl_universal_generation_edit_model'
    denoise_model_type = 'SingleStreamMMDiT_2B'
    task_type = 'T2I'
    base_resize = 256

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

    train_dataset = T2IDataset(
        root_dir=t2i_dataset_path,
        dataset_name=[
            'BM-6M',
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
            'FaceID-6M': [
                'laion_512_000', 'laion_512_001', 'laion_512_002',
                'laion_512_003', 'laion_512_004'
            ],
            'GPIC': [
                'train_000', 'train_001', 'train_002', 'train_003',
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
                'train_096', 'train_097', 'train_098'
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
        max_t2i_caption_length=1024,
        transform=transforms.Compose([
            Opencv2PIL(),
            TorchAspectRatioBucketResize(base_resize=base_resize),
            TorchMeanStdNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]))
    train_collater = T2ICollator()

    seed = 0

    # batch_size is total size
    batch_size = 16
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
