import random

import torch
from transformers import AutoProcessor

# System prompts for the text-to-image (T2I) task. The system prompt guides the VLM to encode
# the prompt in an image-description oriented way so that the produced hidden states are more visually grounded.
T2I_SYSTEM_PROMPTS = [
    "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:",
]

# System prompts for the text-image-to-image (TI2I) task. The system prompt
# guides the VLM to first understand the input reference images and then
# reason about how the user instruction should alter the image.
TI2I_SYSTEM_PROMPTS = [
    "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.",
]


class Qwen3VLGenerationTokenizer:

    def __init__(self, vlm_model_path):
        self.processor = AutoProcessor.from_pretrained(vlm_model_path,
                                                       trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer

        self.user_header_ids = self.tokenizer.encode("<|im_start|>user\n",
                                                     add_special_tokens=False)

        self.vocab_size = len(self.tokenizer)

        print(f'vocab_size: {self.vocab_size}')

    def build_chat_messages(self, system_text, user_text, pil_images=None):
        """Build Qwen3VL chat messages format for the T2I/TI2I encoder.

        Args:
            system_text: str, the system prompt (task specific).
            user_text: str, the user prompt.
            pil_images: list of PIL.Image or None. For the T2I task this is
                None (text only). For the TI2I task this is the list of
                reference images, which are inserted before the text as
                interleaved vision placeholders.

        Returns:
            list of message dicts for processor.apply_chat_template
        """
        user_content = []
        if pil_images is not None:
            for image_idx, pil_image in enumerate(pil_images):
                user_content.append({
                    "type": "text",
                    "text": f"Picture {image_idx + 1}: ",
                })
                user_content.append({
                    "type": "image",
                    "image": pil_image,
                })
        user_content.append({
            "type": "text",
            "text": user_text,
        })

        messages = [
            {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": system_text,
                }],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        return messages

    def encode(self, prompt_texts, sample_type, pil_images_list=None):
        """Encode a batch of raw prompts into VLM-ready tensors.

        build_chat_messages -> tokenize -> compute prompt_start_idx.

        Args:
            prompt_texts: list of str, raw user prompts.
            sample_type: str, "T2I" or "TI2I".
            pil_images_list: list (one per sample) of list of PIL.Image, or
                None. For the T2I task this is None (no image is fed to the
                VLM). For the TI2I task this is the reference images per sample.

        Returns:
            dict with keys:
                input_ids: [B, L] padded token ids
                attention_mask: [B, L] attention mask
                pixel_values: tensor from VLM image processor, or None
                image_grid_thw: tensor from VLM image processor, or None
                mm_token_type_ids: [B, L] multimodal token type ids, or None
                prompt_start_idx: [B] user-content start index per sample
        """
        assert sample_type in ['T2I',
                               'TI2I'], f"Invalid sample_type: {sample_type}"

        B = len(prompt_texts)
        if pil_images_list is None:
            pil_images_list = [None] * B

        if sample_type == 'T2I':
            system_prompt_list = T2I_SYSTEM_PROMPTS
        elif sample_type == 'TI2I':
            system_prompt_list = TI2I_SYSTEM_PROMPTS

        all_input_ids = []
        all_prompt_start_idx = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_mm_token_type_ids = []
        for i in range(B):
            # Sample one system prompt string per sample (str, not list).
            system_text = random.choice(system_prompt_list)
            messages = self.build_chat_messages(system_text, prompt_texts[i],
                                                pil_images_list[i])

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt")

            # input_ids: [L]
            input_ids = inputs['input_ids'].squeeze(0)
            all_input_ids.append(input_ids)
            all_prompt_start_idx.append(
                self.compute_prompt_start_idx(input_ids))

            # Collect pixel_values and image_grid_thw (TI2I task only)
            if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
                all_pixel_values.append(inputs['pixel_values'])
            if 'image_grid_thw' in inputs and inputs[
                    'image_grid_thw'] is not None:
                all_image_grid_thw.append(inputs['image_grid_thw'])

            # Collect mm_token_type_ids (required by Qwen3VL for M-RoPE)
            if 'mm_token_type_ids' in inputs and inputs[
                    'mm_token_type_ids'] is not None:
                all_mm_token_type_ids.append(
                    inputs['mm_token_type_ids'].squeeze(0))

        # Pad sequences to max length in batch
        max_len = max(ids.shape[0] for ids in all_input_ids)
        pad_token_id = self.tokenizer.pad_token_id or 0

        padded_input_ids = torch.full((B, max_len),
                                      pad_token_id,
                                      dtype=torch.long)
        padded_attention_mask = torch.zeros((B, max_len), dtype=torch.long)

        # mm_token_type_ids: pad with 0 (text type) for padding positions
        padded_mm_token_type_ids = None
        if len(all_mm_token_type_ids) > 0:
            assert len(all_mm_token_type_ids) == B, (
                "TI2I: every sample must contain at least one reference image "
                f"(got {len(all_mm_token_type_ids)} mm_token_type_ids for "
                f"batch size {B})")
            padded_mm_token_type_ids = torch.zeros((B, max_len),
                                                   dtype=torch.long)

        for i in range(B):
            L = all_input_ids[i].shape[0]
            padded_input_ids[i, :L] = all_input_ids[i]
            padded_attention_mask[i, :L] = 1
            if padded_mm_token_type_ids is not None and i < len(
                    all_mm_token_type_ids):
                padded_mm_token_type_ids[i, :L] = all_mm_token_type_ids[i]

        # Stack pixel_values and image_grid_thw
        pixel_values = None
        image_grid_thw = None
        if len(all_pixel_values) > 0:
            pixel_values = torch.cat(all_pixel_values, dim=0)
        if len(all_image_grid_thw) > 0:
            image_grid_thw = torch.cat(all_image_grid_thw, dim=0)

        prompt_start_idx = torch.tensor(all_prompt_start_idx, dtype=torch.long)

        return {
            'input_ids': padded_input_ids,
            'attention_mask': padded_attention_mask,
            'pixel_values': pixel_values,
            'image_grid_thw': image_grid_thw,
            'mm_token_type_ids': padded_mm_token_type_ids,
            'prompt_start_idx': prompt_start_idx,
        }

    def decode(self, token_ids, skip_special_tokens=False):
        """Decode a batch of token ids back to text.

        Args:
            token_ids: [B, L] tensor of token ids
            skip_special_tokens: bool, whether to skip special tokens in output

        Returns:
            list of str, decoded text for each sample in the batch
        """
        decoded_texts = self.tokenizer.batch_decode(
            token_ids, skip_special_tokens=skip_special_tokens)

        return decoded_texts

    def compute_prompt_start_idx(self, input_ids):
        """Locate the first token position of the user content.

        Everything before the user header (i.e. the system prompt) is dropped
        when extracting condition features, so the returned index marks where
        the user image/text tokens start.

        Args:
            input_ids: [L] tensor of token ids

        Returns:
            prompt_start_idx: int, position right after "<|im_start|>user\\n"
        """
        input_list = input_ids.tolist()
        header_len = len(self.user_header_ids)

        prompt_start_idx = None
        for i in range(len(input_list) - header_len + 1):
            if input_list[i:i + header_len] == self.user_header_ids:
                prompt_start_idx = i + header_len
                break

        assert prompt_start_idx is not None, (
            "user header <|im_start|>user\\n "
            f"(ids={self.user_header_ids}) not found in input_ids; the chat "
            "template or tokenizer special tokens may have changed")

        return prompt_start_idx
