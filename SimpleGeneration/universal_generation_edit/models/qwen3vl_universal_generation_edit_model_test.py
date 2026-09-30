import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from einops import rearrange

import torch
import torch.nn as nn

from peft import LoraConfig, get_peft_model
from transformers import Qwen3VLForConditionalGeneration

from SimpleGeneration.flux_autoencoder.models.flux2_autoencoder import AutoEncoder
from SimpleGeneration.universal_generation_edit import models
from SimpleGeneration.universal_generation_edit.models.scheduler import FlowMatchingEulerScheduler, FlowMatchingResolutionAdaptiveEulerScheduler

__all__ = [
    'QWEN3VLUniversalGenerationEditModelTest',
]


class QWEN3VLUniversalGenerationEditModelTest(nn.Module):

    def __init__(self,
                 denoise_model_type='DoubleStreamMMDiT_1B',
                 vlm_model_path=None,
                 deepstack_layers=(9, 18, 36),
                 in_channels=128,
                 axes_dim=[8, 40, 40, 40],
                 theta=10000,
                 num_refiner_layers=2,
                 adaln_embed_dim=256,
                 max_ref_images=5,
                 ref_time_coord_scale=20,
                 attention_backend='sdpa'):
        """
        Inference-only universal generation / edit model with traditional CFG
        support. The matching training-only model lives in
        `qwen3vl_universal_generation_edit_model_train.py`; this file carries the same
        module topology (so a training checkpoint loads into it verbatim) minus
        everything only a backward pass needs: no `forward`, no CFG dropout and
        no gradient checkpointing.

        Args:
            denoise_model_type: registry name of the denoise DiT factory, e.g.
                'DoubleStreamMMDiT_1B' / 'SingleStreamMMDiT_1B' /
                'MixStreamMMDiT_1B'. All three variants support both the T2I
                and the reference-image (TI2I) task.
            vlm_model_path: HuggingFace path of the Qwen3-VL text/image encoder
                (required).
            deepstack_layers: 1-based LM layer numbers whose hidden states are
                concatenated as the condition feature, i.e. layer k means the
                output of the k-th decoder layer. The LM stack is truncated to
                max(deepstack_layers) layers, so the deepest tap always reads
                the final layer, which HuggingFace ties to
                `last_hidden_state` -- that tap is therefore taken AFTER the
                LM's final RMSNorm while the shallower ones are raw layer
                outputs. Their norms differ by an order of magnitude, which
                costs nothing here because every tap is RMS-normalized on its
                own (see `deepstack_tap_rms_normalize`). It MUST match the
                value the checkpoint was trained with, since it sets
                `context_in_dim` of the denoise DiT.
            in_channels: VAE latent channels after 2x2 patchify (z_planes=32 x 2 x 2 = 128).
            axes_dim / theta: 4D RoPE config forwarded to the denoise DiT.
            num_refiner_layers / adaln_embed_dim / max_ref_images: forwarded to
                the denoise DiT. The GQA K/V head count `num_kv_heads` is NOT
                an argument here: it belongs to the size spec of each
                denoise-DiT factory (e.g. `DoubleStreamMMDiT_1B`).
            ref_time_coord_scale: temporal-coordinate spacing for reference
                images in the 4D RoPE: ref image j gets 
                t = ref_time_coord_scale * (j + 1).
            attention_backend: 'sdpa' or 'flash_varlen' for the denoise DiT.
        """
        super(QWEN3VLUniversalGenerationEditModelTest, self).__init__()

        assert vlm_model_path is not None, "vlm_model_path must be provided for the VLM text/image encoder"

        self.in_channels = in_channels
        self.max_ref_images = max_ref_images
        self.ref_time_coord_scale = ref_time_coord_scale
        self.deepstack_layers = tuple(deepstack_layers)
        assert min(self.deepstack_layers) >= 1
        self.deepstack_capture_indexes = [
            per_layer - 1 for per_layer in self.deepstack_layers
        ]

        # ---- LoRA-tuned VLM text/image encoder (Qwen3-VL) ----
        # The VLM is the text/image condition encoder. Native DeepStack (ViT
        # multi-level features injected into the first few LM layers) is
        # handled inside the standard VLM forward; on top of that the LM
        # hidden states of `deepstack_layers` are concatenated into `ctx`.
        # The LoRA adapters are still injected here even though nothing is
        # tuned any more: they are part of the trained checkpoint's module
        # tree, so the topology has to match it exactly.
        self.vlm_dtype = torch.bfloat16
        self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            vlm_model_path,
            dtype=self.vlm_dtype,
            attn_implementation='flash_attention_2')
        vlm_hidden_size = self.vlm.config.text_config.hidden_size

        # Only the LM layers up to the deepest deepstack layer contribute to
        # `ctx`, so the stack is cut down to that depth here, before the
        # adapters are injected. With the default taps the deepest one IS the
        # last layer, so nothing is actually dropped; the slice only bites
        # when max(deepstack_layers) is set below the checkpoint's depth. The
        # config is updated alongside (it is the very object the LM reads its
        # depth from) so that anything sizing itself from it (KV cache,
        # checkpoint reload) stays consistent.
        last_used_layer = max(self.deepstack_layers)
        language_model = self.vlm.model.language_model
        language_model.layers = language_model.layers[:last_used_layer]
        language_model.config.num_hidden_layers = last_used_layer

        vlm_lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            exclude_modules=r".*visual.*",
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.vlm = get_peft_model(self.vlm, vlm_lora_config)

        # The whole VLM is pinned to bf16, the modules the training model keeps
        # in fp32 (the LoRA adapters, which `get_peft_model` creates in fp32,
        # plus `visual.merger` / `visual.deepstack_merger_list`) included.
        # fp32 only ever bought those modules a representable optimizer
        # update, which no inference pass performs, so nothing here has to pay
        # for it: one uniform dtype instead halves their footprint, removes
        # the fp32 input-cast hook the training model needs on the mergers and
        # keeps every VLM matmul on the same kernel. Casting AFTER
        # `get_peft_model` is what covers the adapters; a fp32 checkpoint
        # loaded on top is copied into these bf16 tensors in place.
        self.vlm = self.vlm.to(self.vlm_dtype)

        context_in_dim = len(self.deepstack_layers) * vlm_hidden_size
        self.context_in_dim = context_in_dim

        # The AE is pinned to bf16, for both the T2I and the TI2I task:
        # at 1024x1024 fp32 costs ~1.4 GB more on the encode and ~1.6 GB more
        # on the decode per batch element, while the bf16 latent stays within
        # 1.6% of z.std -- far below the loss the DiT converges to. It is also
        # the dtype the model was trained against, so encode / decode here
        # reproduce the training latents exactly.
        # `use_gradient_checkpoint` stays False: it only ever trades compute
        # for activation memory on a backward pass.
        # Casting BEFORE the config loads the fp32 AE checkpoint is safe:
        # `load_state_dict` copies into the existing bf16 tensors in place.
        # No `.eval()` is forced here: `AutoEncoder.normalize` /
        # `inv_normalize` call `bn.eval()` themselves on every invocation, so
        # no running statistic can move whichever mode the module sits in, the
        # AE carries no Dropout and its GroupNorms are mode independent.
        self.ae = AutoEncoder(inplanes=3,
                              planes=128,
                              planes_mult=[1, 2, 4, 4],
                              res_block_nums=2,
                              z_planes=32,
                              out_planes=3,
                              logvar_init=0.0,
                              sample_z=False,
                              use_gradient_checkpoint=False)
        self.ae_dtype = torch.bfloat16
        self.ae = self.ae.to(self.ae_dtype)

        self.denoise_model = models.__dict__[denoise_model_type](
            in_channels=in_channels,
            context_in_dim=context_in_dim,
            axes_dim=axes_dim,
            theta=theta,
            num_refiner_layers=num_refiner_layers,
            adaln_embed_dim=adaln_embed_dim,
            max_ref_images=max_ref_images,
            use_gradient_checkpoint=False,
            attention_backend=attention_backend,
        )
        self.out_channels = self.denoise_model.out_channels

        self.requires_grad_(False)

    def deepstack_tap_rms_normalize(self, tap_hidden_state):
        """
        Parameter-free RMS normalization of ONE deepstack tap.

        The mean square is accumulated in float32 and the result is cast back,
        same as `RMSNorm.forward` in the denoise DiT, so the tap keeps the bf16
        the VLM emits.

        Args:
            tap_hidden_state: [B, S, vlm_hidden_size] one LM hidden state.

        Returns:
            [B, S, vlm_hidden_size] the tap with unit RMS along the last axis.
        """
        rrms = torch.rsqrt(
            tap_hidden_state.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        tap_hidden_state = tap_hidden_state * rrms.to(tap_hidden_state.dtype)

        return tap_hidden_state

    def encode_condition(self,
                         input_ids,
                         attention_mask,
                         prompt_start_idx,
                         pixel_values=None,
                         image_grid_thw=None,
                         mm_token_type_ids=None):
        """
        Encode text (T2I) or text + reference image(s) (TI2I) with the VLM.

        The VLM standard forward automatically applies native DeepStack; here
        we further concatenate the LM hidden states of `deepstack_layers`, each
        RMS-normalized on its own (see `deepstack_tap_rms_normalize`).
        The leading system-prompt tokens are dropped per sample and the valid
        tokens are padded to the batch max length to form `ctx`.

        Returns:
            ctx: [B, L_ctx, context_in_dim]
            ctx_ids: [B, L_ctx, 4] position ids (t=0, h=0, w=0, l=index)
            ctx_mask: [B, L_ctx] bool, True for valid (non-padding) tokens
        """
        device = input_ids.device

        vlm_kwargs = dict(input_ids=input_ids,
                          attention_mask=attention_mask,
                          use_cache=False,
                          output_hidden_states=self.deepstack_capture_indexes)
        if pixel_values is not None:
            vlm_kwargs['pixel_values'] = pixel_values.to(self.vlm_dtype)
            vlm_kwargs['image_grid_thw'] = image_grid_thw
        if mm_token_type_ids is not None:
            vlm_kwargs['mm_token_type_ids'] = mm_token_type_ids

        # Call the inner Qwen3VLModel directly to skip the lm_head: only the
        # per-layer hidden states are needed, so the [B, L, vocab~=151k]
        # projection and its activation memory are avoided. LoRA adapters stay
        # active and native DeepStack is still handled inside the forward.
        #
        # The autocast is opted out of unconditionally, under all three amp
        # modes (fp32 / bf16 / fp16). The whole VLM is bf16, so it computes in
        # bf16 whatever precision the caller asked for, and `ctx` stops
        # depending on the caller's amp mode.
        device_type = input_ids.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            outputs = self.vlm.base_model.model.model(**vlm_kwargs)

            # tuple length = num_layers, and only the requested
            # `deepstack_capture_indexes` entries hold a tensor (every other
            # one is None), so the states of the layers that feed no tap are
            # freed as soon as the next layer consumes them.
            hidden_states = outputs.hidden_states
            # [B, S, len(layers) * D]
            stacked = torch.cat([
                self.deepstack_tap_rms_normalize(hidden_states[k])
                for k in self.deepstack_capture_indexes
            ],
                                dim=-1)

        B, L = input_ids.shape
        # Drop the system-prompt prefix and left-pack the valid tokens into
        # `ctx`, fully on the GPU: `cumsum` hands every valid token its
        # destination slot directly, so only the single `max_len` readout
        # costs a GPU->CPU sync.
        positions = torch.arange(L, device=device)[None, :]
        valid_mask = attention_mask.bool() & (positions
                                              >= prompt_start_idx[:, None])
        valid_lens = valid_mask.sum(dim=1)
        # `max(1, ...)` floors the batch at one context slot, the same floor
        # the reference stream keeps in `generate`: a zero-length `ctx` would
        # hand the DiT a different topology than the one it was trained with.
        max_len = max(1, int(valid_lens.max()))

        # destination slot of every valid token inside its own sample
        dst_index = torch.cumsum(valid_mask.long(), dim=1) - 1
        batch_index = torch.arange(B, device=device)[:, None].expand(B, L)

        ctx = stacked.new_zeros(B, max_len, stacked.shape[-1])
        ctx[batch_index[valid_mask],
            dst_index[valid_mask]] = stacked[valid_mask]
        # `clamp(min=1)` keeps exactly ONE valid slot for a sample whose user
        # content came back empty: `ctx_mask` is the key-padding mask of the
        # DiT's context refiner, so an all-False row would make its softmax
        # all-masked and produce NaN that leaks into the image stream through
        # the joint attention. That slot holds the zero vector `ctx` was built
        # with, which is the same unconditional signal the CFG branch of
        # `generate` feeds, so the fallback needs no separate convention.
        ctx_mask = torch.arange(
            max_len, device=device)[None, :] < valid_lens.clamp(min=1)[:, None]

        # sequential position ids along the l axis (t=h=w=0)
        ctx_ids = torch.zeros(B, max_len, 4, device=device)
        ctx_ids[..., 3] = torch.arange(max_len, device=device)[None, :]

        return ctx, ctx_ids, ctx_mask

    @torch.no_grad()
    def encode_image_latent(self, image, t_coord=0.0):
        """
        VAE-encode an image tensor to flattened latent tokens + 4D ids.

        Args:
            image: [B, 3, H, W] tensor normalized to [-1, 1]
            t_coord: temporal coordinate for the 4D RoPE (0 for the noise
                target, 10/20/... for reference images).

        Returns:
            tokens: [B, h*w, in_channels]
            ids: [B, h*w, 4] with (t=t_coord, h, w, l=0)
        """
        device = image.device
        # The AE weights are bf16 (see __init__) and the autocast is disabled
        # around them: freezing a module does not stop autocast, so without
        # this opt-out an outer fp16 amp run would drag the AE to fp16 and the
        # latent would depend on the caller's amp mode. Pinning both keeps
        # encode() producing the same latent under fp32 / bf16 / fp16 amp.
        with torch.autocast(device_type=device.type, enabled=False):
            # [B, 128, h, w]
            z = self.ae.encode(image.to(self.ae_dtype))
        _, _, grid_h, grid_w = z.shape

        tokens = rearrange(z, "B C h w -> B (h w) C")

        img_ids = torch.zeros(grid_h, grid_w, 4, device=device)
        img_ids[..., 0] = t_coord
        img_ids[..., 1] = torch.arange(grid_h, device=device)[:, None]
        img_ids[..., 2] = torch.arange(grid_w, device=device)[None, :]
        ids = rearrange(img_ids,
                        "h w c -> (h w) c")[None].repeat(image.shape[0], 1, 1)

        return tokens, ids

    @torch.no_grad()
    def generate(self,
                 input_ids,
                 attention_mask,
                 prompt_start_idx,
                 target_height,
                 target_width,
                 scheduler,
                 guidance_scale=3.5,
                 pixel_values=None,
                 image_grid_thw=None,
                 mm_token_type_ids=None,
                 reference_images=None,
                 seed=None,
                 cfg_renorm=False,
                 cfg_renorm_min=0.0):
        """
        Generate an image from text (T2I) or text + reference images (TI2I).

        Complete inference pipeline: sample initial noise → encode VLM
        condition → encode reference image latents (if any) → multi-step ODE
        denoising with traditional Classifier-Free Guidance (CFG) →
        VAE decode to pixel image.

        The task (T2I vs TI2I) is inferred from the inputs: T2I passes no
        `pixel_values` / `reference_images`, while TI2I passes both.

        Guidance uses **traditional CFG** with the formula
        ``v = v_uncond + guidance_scale * (v_cond - v_uncond)``. The
        unconditional branch feeds a single zero condition token (one zero
        text token, plus one zero reference token for TI2I), matching the CFG
        dropout used during training. Without reference images the two
        branches share a sequence length and are fused into one [2B, ...]
        batch (Batch-CFG); with them they run as two forwards. When
        ``guidance_scale == 1.0`` CFG is skipped entirely.

        The ODE solver is configurable via the ``scheduler`` argument, which
        accepts either scheduler type defined in ``scheduler.py``:

        * ``FlowMatchingEulerScheduler`` — 1st-order Euler with optional
          time-shift.
        * ``FlowMatchingResolutionAdaptiveEulerScheduler`` — Euler with
          FLUX.2-style resolution-adaptive shift (mu computed from
          image_seq_len). The only scheduler that consumes image_seq_len.

        Optionally, **CFG Renorm** rescales the guided velocity so that its
        per-token L2 norm does not exceed the conditional velocity's norm,
        preventing colour over-saturation at high guidance scales.

        Precision: the denoise DiT follows the caller's autocast, so a float32
        / bf16 / fp16 inference pass really runs it at that precision (its
        attention kernels stay pinned to bf16 by design, except that an fp16
        autocast pulls the two SDPA paths down to fp16). The VLM condition
        encoder and the AE both opt out of the autocast entirely and are
        pinned to bf16, so the same ``seed`` yields the identical condition
        feature and the identical latent under all three amp modes. The
        initial noise, the latent ``x_t``, the timesteps, the position ids, the
        CFG formula / CFG Renorm, the ODE solver step and the returned image
        all stay in float32.

        Args:
            input_ids: [B, L] padded token ids from the tokenizer.
            attention_mask: [B, L] attention mask from the tokenizer.
            prompt_start_idx: [B] user-content start index per sample.
            target_height: int, target image height in pixels (must be
                divisible by 16, since AE downsamples 8x and patchify 2x).
            target_width: int, target image width in pixels (same constraint).
            scheduler: an instance of one of the two supported scheduler
                classes. Controls the timestep schedule and ODE solver.
            guidance_scale: float, CFG guidance strength (default 3.5).
                When 1.0, CFG is disabled (single forward per step).
            pixel_values: VLM image inputs for TI2I (from tokenizer), or None.
            image_grid_thw: VLM image grid info for TI2I, or None.
            mm_token_type_ids: VLM multimodal token type ids for TI2I, or None.
            reference_images: list (len B) of [N_i, 3, Hr, Wr] tensors in
                [-1, 1], or None for T2I.
            seed: int or None, random seed for reproducible noise sampling.
            cfg_renorm: bool, whether to enable CFG Renorm (default False).
                Prevents colour over-saturation by rescaling the guided
                velocity norm back to the conditional velocity's norm.
            cfg_renorm_min: float, minimum allowed renorm scale factor
                (default 0.0).

        Returns:
            images: [B, 3, target_height, target_width] generated images in
                [-1, 1] (same range as the training target images).
        """
        assert target_height % 16 == 0 and target_width % 16 == 0

        self.eval()

        device = input_ids.device
        img_in_dtype = self.denoise_model.img_in.weight.dtype
        B = input_ids.shape[0]

        # Latent grid dimensions after AE encode + patchify.
        grid_h = target_height // 16
        grid_w = target_width // 16

        # ---- reproducible noise ----
        # Sampled in float32 regardless of the DiT weight dtype: x_t is
        # carried in float32 through the whole ODE loop anyway, and sampling
        # in the weight dtype would make the same `seed` start a different
        # trajectory under one precision than under another.
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            noise = torch.randn(B,
                                grid_h * grid_w,
                                self.in_channels,
                                device=device,
                                dtype=torch.float32,
                                generator=generator)
        else:
            noise = torch.randn(B,
                                grid_h * grid_w,
                                self.in_channels,
                                device=device,
                                dtype=torch.float32)

        # ---- encode VLM condition (text, or text + reference images) ----
        ctx, ctx_ids, ctx_mask = self.encode_condition(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_start_idx=prompt_start_idx,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids)
        # `ctx` is a feature value feeding the DiT's txt_in nn.Linear, so it is
        # aligned to the DiT's runtime weight dtype. The VLM always emits bf16,
        # so without this cast an fp32 DiT would hit a dtype mismatch on the
        # paths that run outside an outer autocast. `ctx_ids` is a position
        # index feeding only rope(), so it is pinned to float32
        # unconditionally: bf16 is integer-exact only up to 256, above which
        # neighbouring token positions collapse and corrupt the RoPE phase.
        ctx = ctx.to(img_in_dtype)
        ctx_ids = ctx_ids.float()

        # ---- build noise-target position ids (t_coord=0, same as training) ----
        img_ids = torch.zeros(grid_h, grid_w, 4, device=device)
        img_ids[..., 0] = 0.0
        img_ids[..., 1] = torch.arange(grid_h, device=device)[:, None]
        img_ids[..., 2] = torch.arange(grid_w, device=device)[None, :]
        img_ids = rearrange(img_ids,
                            "h w c -> (h w) c")[None].repeat(B, 1, 1).float()

        # ---- encode reference image latents (TI2I task) ----
        ref_tokens = None
        ref_ids = None
        ref_valid = None
        ref_index = None
        if reference_images is not None:
            ref_tokens_list = []
            ref_ids_list = []
            ref_index_list = []
            for i, per_sample_refs in enumerate(reference_images):
                if per_sample_refs is None or len(per_sample_refs) == 0:
                    ref_tokens_list.append(noise.new_zeros(
                        0, self.in_channels))
                    ref_ids_list.append(noise.new_zeros(0, 4))
                    ref_index_list.append(
                        torch.zeros(0, dtype=torch.long, device=device))
                    continue
                assert len(per_sample_refs) <= self.max_ref_images
                per_tokens = []
                per_ids = []
                per_index = []
                for j in range(len(per_sample_refs)):
                    ref_height, ref_width = per_sample_refs[j].shape[
                        -2], per_sample_refs[j].shape[-1]
                    assert ref_height % 16 == 0 and ref_width % 16 == 0
                    ref_img = per_sample_refs[j][None].to(device)
                    t_off = self.ref_time_coord_scale * (j + 1)
                    r_tokens, r_ids = self.encode_image_latent(
                        ref_img, t_coord=float(t_off))
                    per_tokens.append(r_tokens[0])
                    per_ids.append(r_ids[0])
                    per_index.append(
                        torch.full((r_tokens.shape[1], ),
                                   j,
                                   dtype=torch.long,
                                   device=device))

                ref_tokens_list.append(torch.cat(per_tokens, dim=0))
                ref_ids_list.append(torch.cat(per_ids, dim=0))
                ref_index_list.append(torch.cat(per_index, dim=0))

            # Same floor as the training forward: keep at least one reference
            # slot so the DiT sees the topology it was trained with, even when
            # every sample of this batch turned out reference-free.
            max_ref_len = max(
                1,
                max(per_sample_tokens.shape[0]
                    for per_sample_tokens in ref_tokens_list))
            ref_tokens = noise.new_zeros(B, max_ref_len, self.in_channels)
            ref_ids = img_ids.new_zeros(B, max_ref_len, 4)
            ref_valid = torch.zeros(B,
                                    max_ref_len,
                                    dtype=torch.bool,
                                    device=device)
            ref_index = torch.zeros(B,
                                    max_ref_len,
                                    dtype=torch.long,
                                    device=device)
            for i in range(B):
                n = ref_tokens_list[i].shape[0]
                if n > 0:
                    ref_tokens[i, :n] = ref_tokens_list[i].to(noise.dtype)
                    ref_ids[i, :n] = ref_ids_list[i].float()
                    ref_valid[i, :n] = True
                    ref_index[i, :n] = ref_index_list[i]
                else:
                    # Same single zero reference token as the training
                    # forward, on the reference time coordinate.
                    ref_ids[i, 0, 0] = self.ref_time_coord_scale
                    ref_valid[i, 0] = True

        has_ref = ref_tokens is not None

        # ---- prepare unconditional branch for Batch-CFG ----
        do_cfg = guidance_scale != 1.0
        if do_cfg:
            # Unconditional ctx: zeros with only the first token valid, the
            # same convention the CFG dropout uses during training (an
            # all-False mask would make the context refiner's softmax
            # all-masked and produce NaN).
            ctx_uncond = torch.zeros_like(ctx)
            # The uncond branch reuses the cond position ids: they are
            # identical and only ever read by rope(), so no copy is needed.
            ctx_ids_uncond = ctx_ids
            ctx_mask_uncond = torch.zeros_like(ctx_mask)
            ctx_mask_uncond[:, 0] = True

            # Unconditional reference stream: one zero reference token (zero
            # latent on the reference time coordinate, identity index 0), what
            # a CFG-dropped sample sees during training. The cond branch keeps
            # its L_ref reference tokens, so the two branches differ in
            # sequence length and run as two separate forwards below.
            if has_ref:
                ref_tokens_uncond = noise.new_zeros(B, 1, self.in_channels)
                ref_ids_uncond = img_ids.new_zeros(B, 1, 4)
                ref_ids_uncond[..., 0] = self.ref_time_coord_scale
                ref_valid_uncond = torch.ones(B,
                                              1,
                                              dtype=torch.bool,
                                              device=device)
                ref_index_uncond = torch.zeros(B,
                                               1,
                                               dtype=torch.long,
                                               device=device)
            else:
                # No refs: fuse cond + uncond into [2B, ...] batch
                ctx_double = torch.cat([ctx, ctx_uncond], dim=0)
                ctx_ids_double = torch.cat([ctx_ids, ctx_ids_uncond], dim=0)
                ctx_mask_double = torch.cat([ctx_mask, ctx_mask_uncond], dim=0)
                img_ids_double = torch.cat([img_ids, img_ids], dim=0)

        # ---- build timestep schedule (float32) ----
        # FlowMatchingEulerScheduler builds its schedule from num_steps /
        # shift alone; FlowMatchingResolutionAdaptiveEulerScheduler
        # additionally needs image_seq_len for the empirical mu.
        if isinstance(scheduler, FlowMatchingEulerScheduler):
            schedule = scheduler.get_schedule(device=device)
        elif isinstance(scheduler,
                        FlowMatchingResolutionAdaptiveEulerScheduler):
            image_seq_len = grid_h * grid_w
            schedule = scheduler.get_schedule(image_seq_len=image_seq_len,
                                              device=device)
        else:
            raise ValueError(f'Unsupported scheduler type: {type(scheduler)}')

        # ---- ODE denoising loop ----
        # x_t is carried in float32 and only cast to the DiT dtype right
        # before each forward: over 20-50 steps the per-step increment
        # |v * dt| is close to the bf16 resolution, so a bf16 accumulator
        # would lose part of every update.
        x_t = noise
        # Read the whole schedule out once, instead of paying a `.item()`
        # GPU->CPU sync per step.
        schedule_value_list = schedule.tolist()

        # No autocast is opened around the denoise DiT: it follows the
        # caller's precision. Every predicted velocity is cast back to float32
        # right away, so the CFG formula, the CFG Renorm and the ODE solver
        # step stay in float32.
        for i in range(len(schedule) - 1):
            # The timestep stays float32: timestep_embedding multiplies t by
            # 1000, and a bf16 t would quantise neighbouring steps onto the
            # same embedding and not match the float32 training timesteps.
            t_batch = torch.full((B, ),
                                 schedule_value_list[i],
                                 device=device,
                                 dtype=torch.float32)

            x_t_model = x_t.to(img_in_dtype)

            if not do_cfg:
                # No CFG: single forward pass
                v_pred = self.denoise_model(x_t_model,
                                            img_ids,
                                            t_batch,
                                            ctx,
                                            ctx_ids,
                                            ctx_mask,
                                            ref_tokens=ref_tokens,
                                            ref_ids=ref_ids,
                                            ref_valid=ref_valid,
                                            ref_index=ref_index).float()
            elif has_ref:
                # CFG with reference images: two separate forwards, since the
                # cond branch has L_ref reference tokens while the uncond one
                # has a single zero token, and the differing sequence lengths
                # prevent batch fusion.
                v_cond = self.denoise_model(x_t_model,
                                            img_ids,
                                            t_batch,
                                            ctx,
                                            ctx_ids,
                                            ctx_mask,
                                            ref_tokens=ref_tokens,
                                            ref_ids=ref_ids,
                                            ref_valid=ref_valid,
                                            ref_index=ref_index).float()
                v_uncond = self.denoise_model(
                    x_t_model,
                    img_ids,
                    t_batch,
                    ctx_uncond,
                    ctx_ids_uncond,
                    ctx_mask_uncond,
                    ref_tokens=ref_tokens_uncond,
                    ref_ids=ref_ids_uncond,
                    ref_valid=ref_valid_uncond,
                    ref_index=ref_index_uncond).float()
                # Traditional CFG formula
                v_pred = v_uncond + guidance_scale * (v_cond - v_uncond)

                # CFG Renorm: rescale the guided velocity norm to the
                # conditional velocity norm.
                if cfg_renorm:
                    cond_norm = torch.norm(v_cond, dim=-1, keepdim=True)
                    pred_norm = torch.norm(v_pred, dim=-1, keepdim=True)
                    scale = (cond_norm / (pred_norm + 1e-8)).clamp(
                        min=cfg_renorm_min, max=1.0)
                    v_pred = v_pred * scale
            else:
                # CFG without reference images: Batch-CFG optimization.
                # Fuse cond + uncond into one [2B, ...] batch.
                x_t_double = torch.cat([x_t_model, x_t_model], dim=0)
                t_double = torch.cat([t_batch, t_batch], dim=0)

                v_double = self.denoise_model(x_t_double,
                                              img_ids_double,
                                              t_double,
                                              ctx_double,
                                              ctx_ids_double,
                                              ctx_mask_double,
                                              ref_tokens=None,
                                              ref_ids=None,
                                              ref_valid=None,
                                              ref_index=None)
                v_cond, v_uncond = v_double.float().chunk(2, dim=0)
                # Traditional CFG formula
                v_pred = v_uncond + guidance_scale * (v_cond - v_uncond)

                # CFG Renorm
                if cfg_renorm:
                    cond_norm = torch.norm(v_cond, dim=-1, keepdim=True)
                    pred_norm = torch.norm(v_pred, dim=-1, keepdim=True)
                    scale = (cond_norm / (pred_norm + 1e-8)).clamp(
                        min=cfg_renorm_min, max=1.0)
                    v_pred = v_pred * scale

            # ---- ODE solver step ----
            # One uniform call for every scheduler.
            x_t = scheduler.step(x_t, v_pred, schedule, i)

        # ---- VAE decode: latent tokens → pixel image ----
        # Same bf16 pinning and autocast opt-out as `encode_image_latent`, so
        # decode stays the exact inverse of the encode the model was trained
        # against. Only the returned image is promoted back to float32, which
        # keeps the output contract independent of the AE's internal dtype.
        z = rearrange(x_t, "B (h w) C -> B C h w", h=grid_h, w=grid_w)
        with torch.autocast(device_type=device.type, enabled=False):
            images = self.ae.decode(z.to(self.ae_dtype)).float()

        return images


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

    from PIL import Image
    from tokenizer import Qwen3VLGenerationTokenizer

    ################################################################################################################
    ################################################################################################################
    ################################################################################################################
    # flux2 AE geometry (SimpleGeneration/flux_autoencoder/models/flux2_autoencoder.py):
    #   encoder conv downsample 8x  (planes_mult=[1, 2, 4, 4] -> 2^3 stages)
    #   x 2x2 patchify inside encode()
    #   = 16x total, and z_planes(32) * 2 * 2 = 128 latent channels.
    batch_size = 1
    ae_z_planes = 32
    ae_patch_size = 2
    ae_conv_downsample_ratio = 8
    ae_downsample_ratio = ae_conv_downsample_ratio * ae_patch_size
    in_channels = ae_z_planes * ae_patch_size * ae_patch_size
    # reference image j gets the temporal RoPE coordinate scale * (j + 1)
    ref_time_coord_scale = 10
    max_ref_images = 5
    # LM layer taps concatenated into ctx; context_in_dim follows from them.
    deepstack_layers = (9, 18, 36)

    qwen3vl_model_path = "Qwen/Qwen3-VL-4B-Instruct"
    gen_tokenizer = Qwen3VLGenerationTokenizer(qwen3vl_model_path)

    print(f"batch_size: {batch_size}")
    print(f"ae_z_planes: {ae_z_planes}")
    print(f"ae_patch_size: {ae_patch_size}")
    print(f"ae_conv_downsample_ratio: {ae_conv_downsample_ratio}")
    print(f"ae_downsample_ratio: {ae_downsample_ratio}")
    print(f"in_channels: {in_channels}")
    print(f"ref_time_coord_scale: {ref_time_coord_scale}")
    print(f"max_ref_images: {max_ref_images}")
    print(f"deepstack_layers: {deepstack_layers}")

    model = QWEN3VLUniversalGenerationEditModelTest(
        denoise_model_type='DoubleStreamMMDiT_1B',
        vlm_model_path=qwen3vl_model_path,
        deepstack_layers=deepstack_layers,
        in_channels=in_channels,
        max_ref_images=max_ref_images,
        ref_time_coord_scale=ref_time_coord_scale,
        attention_backend='sdpa')
    model = model.cuda()

    print(f"context_in_dim: {model.context_in_dim}")
    print(f"out_channels: {model.out_channels}")
    ################################################################################################################
    ################################################################################################################
    ################################################################################################################

    # The AE and the whole VLM (frozen base weights, LoRA adapters and the two
    # merger modules alike) are bf16 while the denoise DiT keeps float32
    # weights and follows the caller's autocast. Nothing is trainable.
    parameter_dtype_dict = {}
    for per_parameter_name, per_parameter in model.named_parameters():
        if per_parameter_name.startswith('ae.'):
            per_group_name = 'ae'
        elif per_parameter_name.startswith('denoise_model.'):
            per_group_name = 'denoise_model'
        elif 'lora_' in per_parameter_name:
            per_group_name = 'vlm_lora'
        elif 'merger' in per_parameter_name:
            per_group_name = 'vlm_merger'
        else:
            per_group_name = 'vlm_frozen'
        per_group_key = (per_group_name, str(per_parameter.dtype),
                         per_parameter.requires_grad)
        parameter_dtype_dict[per_group_key] = parameter_dtype_dict.get(
            per_group_key, 0) + per_parameter.numel()
        assert not per_parameter.requires_grad
        if per_group_name == 'denoise_model':
            assert per_parameter.dtype == torch.float32
        else:
            assert per_parameter.dtype == torch.bfloat16

    for per_group_key in sorted(parameter_dtype_dict, key=str):
        print(f"group: {per_group_key[0]}, dtype: {per_group_key[1]}, "
              f"requires_grad: {per_group_key[2]}, "
              f"parameter_nums: {parameter_dtype_dict[per_group_key]}")

    input_resolution_list = [[256, 256], [512, 512]]
    input_reference_image_flag_list = [False, True]
    input_amp_type_list = [None, torch.bfloat16, torch.float16]
    # 3.5 exercises CFG (Batch-CFG for T2I, two forwards for TI2I), 1.0 the
    # single-forward path that skips it.
    input_guidance_scale_list = [3.5, 1.0]
    for image_height, image_width in input_resolution_list:
        for use_reference_image in input_reference_image_flag_list:
            task_name = 'TI2I' if use_reference_image else 'T2I'
            print(
                f"{task_name}, image_height: {image_height}, image_width: {image_width}"
            )

            # ---- image resolution -> AE latent grid -> noise-target tokens ----
            assert image_height % ae_downsample_ratio == 0
            assert image_width % ae_downsample_ratio == 0
            latent_height = image_height // ae_downsample_ratio
            latent_width = image_width // ae_downsample_ratio
            img_seq_len = latent_height * latent_width
            print(
                f"{task_name}, latent_height: {latent_height}, latent_width: {latent_width}, img_seq_len: {img_seq_len}"
            )

            # ---- prompt / reference image -> VLM condition input ----
            # T2I feeds no image to the VLM, so pixel_values / image_grid_thw /
            # mm_token_type_ids come back as None and must stay None.
            if use_reference_image:
                prompt_texts = ["change the background to a city street"]
                pil_images_list = [[
                    Image.fromarray(
                        np.uint8(
                            np.random.rand(image_height, image_width, 3) *
                            255))
                ] for _ in range(batch_size)]
            else:
                prompt_texts = [
                    "a photo of a black-and-white Chinese rural dog"
                ]
                pil_images_list = None

            tokenized = gen_tokenizer.encode(prompt_texts=prompt_texts,
                                             sample_type=task_name,
                                             pil_images_list=pil_images_list)

            input_ids = tokenized['input_ids'].cuda()
            attention_mask = tokenized['attention_mask'].cuda()
            prompt_start_idx = tokenized['prompt_start_idx'].cuda()
            print(f"{task_name}, input_ids shape: {tuple(input_ids.shape)}")
            print(
                f"{task_name}, attention_mask shape: {tuple(attention_mask.shape)}"
            )
            print(
                f"{task_name}, prompt_start_idx: {prompt_start_idx.tolist()}")

            pixel_values, image_grid_thw, mm_token_type_ids = None, None, None
            if use_reference_image:
                pixel_values = tokenized['pixel_values'].cuda()
                image_grid_thw = tokenized['image_grid_thw'].cuda()
                mm_token_type_ids = tokenized['mm_token_type_ids'].cuda()
                print(
                    f"{task_name}, pixel_values shape: {tuple(pixel_values.shape)}"
                )
                print(
                    f"{task_name}, image_grid_thw: {image_grid_thw.tolist()}")
                print(
                    f"{task_name}, mm_token_type_ids shape: {tuple(mm_token_type_ids.shape)}"
                )

            # ---- AE-side inputs: reference images ----
            # reference_images is a list (len batch_size) of per-sample lists,
            # since each sample may carry a different number of reference
            # images and each one keeps its own resolution.
            if use_reference_image:
                # One reference image at the same resolution as the target.
                reference_image_height, reference_image_width = image_height, image_width
                ref_latent_height = reference_image_height // ae_downsample_ratio
                ref_latent_width = reference_image_width // ae_downsample_ratio
                ref_seq_len = ref_latent_height * ref_latent_width
                reference_images = [[
                    torch.randn(3, reference_image_height,
                                reference_image_width).cuda()
                ] for _ in range(batch_size)]
                print(
                    f"{task_name}, ref_latent_height: {ref_latent_height}, ref_latent_width: {ref_latent_width}"
                )
                print(f"{task_name}, ref_seq_len: {ref_seq_len}")
            else:
                reference_images = None

            for amp_type in input_amp_type_list:
                # amp_type None is the pure float32 path: the autocast is opened
                # with enabled=False so that all three modes go through the same
                # code and only the precision differs.
                if amp_type is None:
                    amp_context = torch.autocast(device_type='cuda',
                                                 enabled=False)
                else:
                    amp_context = torch.autocast(device_type='cuda',
                                                 dtype=amp_type)

                # ---- inference path: multi-step ODE denoise + AE decode ----
                # Both schedulers are exercised: only the resolution-adaptive
                # one consumes image_seq_len. The AE is kept frozen by
                # `requires_grad_(False)` plus the `bn.eval()` inside
                # normalize / inv_normalize, so no statistic moves here.
                model.eval()
                for scheduler in [
                        FlowMatchingEulerScheduler(num_steps=2, shift=3.0),
                        FlowMatchingResolutionAdaptiveEulerScheduler(
                            num_steps=2, snr_sigma=1.0),
                ]:
                    for guidance_scale in input_guidance_scale_list:
                        with amp_context:
                            images = model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                prompt_start_idx=prompt_start_idx,
                                target_height=image_height,
                                target_width=image_width,
                                scheduler=scheduler,
                                guidance_scale=guidance_scale,
                                pixel_values=pixel_values,
                                image_grid_thw=image_grid_thw,
                                mm_token_type_ids=mm_token_type_ids,
                                reference_images=reference_images,
                                seed=seed,
                                cfg_renorm=True,
                                cfg_renorm_min=0.0)

                        print(
                            f"{task_name}, amp_type: {amp_type}, scheduler: {type(scheduler).__name__}, "
                            f"guidance_scale: {guidance_scale}, images shape: {tuple(images.shape)}, "
                            f"images dtype: {images.dtype}")
                        assert images.shape == (batch_size, 3, image_height,
                                                image_width)
                        assert images.dtype == torch.float32
                        assert not torch.any(torch.isnan(images))

                torch.cuda.empty_cache()
            ########################################################################################################
            ########################################################################################################
            ########################################################################################################

    del model
    torch.cuda.empty_cache()
