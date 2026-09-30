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

__all__ = [
    'QWEN3VLUniversalGenerationEditModelFSDP',
]


class QWEN3VLUniversalGenerationEditModelFSDP(nn.Module):

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
                 cfg_dropout_prob=0.1,
                 use_gradient_checkpoint=False,
                 attention_backend='sdpa'):
        """
        FSDP2-only universal generation / edit model. The DDP/DeepSpeed variant
        lives in `qwen3vl_universal_generation_edit_model_train.py` and the
        inference-only model in `qwen3vl_universal_generation_edit_model_test.py`.

        The ONLY structural difference from the DDP variant is the parameter
        dtype layout, and it exists because FSDP2 shards per parameter:
        `_FSDPParamGroup._init_mp_dtypes` asserts that every parameter inside
        one sharded group carries the same original dtype, so the DDP variant's
        mixed layout (bf16 frozen VLM base weights next to fp32 LoRA adapters
        inside the very same decoder layer) cannot be sharded at all.

        Here the WHOLE VLM is loaded in float32 and nothing is cast afterwards,
        so the VLM and the denoise DiT are uniformly float32 and both can be
        sharded. Dropping to `torch.bfloat16` for the actual matmuls is
        then the job of FSDP2's `MixedPrecisionPolicy`: it all-gathers the
        shards as that dtype, keeps the float32 master weights for the
        optimizer update, and reduces gradients in float32. 

        The float32 master weight the optimizer needs is no longer something
        this model has to arrange by hand; FSDP2 holds it by construction.

        The AE is the one module left out of the sharding.

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
                own (see `deepstack_tap_rms_normalize`). The default
                (9, 18, 36) taps the shallow / middle / final thirds of the
                36-layer stack shared by Qwen3-VL-4B-Instruct and
                Qwen3-VL-8B-Instruct, and leaves that stack at its full depth:
                the shallow tap carries the spatial / texture / grounding
                signal the TI2I task needs, the final tap the compositional /
                counting / text-rendering signal the T2I task needs.
            in_channels: VAE latent channels after 2x2 patchify (z_planes=32 x 2 x 2 = 128).
            axes_dim / theta: 4D RoPE config forwarded to the denoise DiT.
            num_refiner_layers / adaln_embed_dim / max_ref_images: forwarded to
                the denoise DiT. The GQA K/V head count `num_kv_heads` is NOT
                an argument here: it belongs to the size spec of each
                denoise-DiT factory (e.g. `DoubleStreamMMDiT_1B`).
            ref_time_coord_scale: temporal-coordinate spacing for reference
                images in the 4D RoPE: ref image j gets 
                t = ref_time_coord_scale * (j + 1).
            cfg_dropout_prob: probability of replacing all conditions (text +
                reference images) of a sample by a single zero condition token
                during training, which is what enables traditional
                Classifier-Free Guidance (CFG) in the inference model. Set to
                0.0 to disable CFG dropout.
            use_gradient_checkpoint: gradient checkpointing for the VLM + DiT.
            attention_backend: 'sdpa' or 'flash_varlen' for the denoise DiT.
        """
        super(QWEN3VLUniversalGenerationEditModelFSDP, self).__init__()

        assert vlm_model_path is not None, "vlm_model_path must be provided for the VLM text/image encoder"

        self.in_channels = in_channels
        self.max_ref_images = max_ref_images
        self.ref_time_coord_scale = ref_time_coord_scale
        self.cfg_dropout_prob = cfg_dropout_prob
        self.use_gradient_checkpoint = use_gradient_checkpoint
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
        # Trainable: the LoRA adapters on the LM projections plus the two
        # vision->LLM projection modules (`visual.merger` and
        # `visual.deepstack_merger_list`). Everything else stays frozen.
        # Gradients must reach them, so `encode_condition` runs with grad.
        self.vlm_dtype = torch.bfloat16
        self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            vlm_model_path,
            dtype=torch.float32,
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

        if self.use_gradient_checkpoint:
            self.vlm.enable_input_require_grads()
            self.vlm.gradient_checkpointing_enable({"use_reentrant": False})

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

        visual = self.vlm.base_model.model.model.visual
        visual.requires_grad_(False)
        visual.merger.requires_grad_(True)
        visual.deepstack_merger_list.requires_grad_(True)

        context_in_dim = len(self.deepstack_layers) * vlm_hidden_size
        self.context_in_dim = context_in_dim

        # The AE is frozen (no weight and no BatchNorm statistic ever moves)
        # and pinned to bf16, for both the T2I and the TI2I task:
        # at 1024x1024 fp32 costs ~1.4 GB more on the encode and ~1.6 GB more
        # on the decode per batch element, while the bf16 latent stays within
        # 1.6% of z.std -- far below the loss the DiT converges to.
        # `use_gradient_checkpoint` stays False: the AE only runs under
        # no_grad, where `checkpoint()` degrades into a plain call.
        # Casting BEFORE the config loads the fp32 AE checkpoint is safe:
        # `load_state_dict` copies into the existing bf16 tensors in place.
        # `requires_grad_(False)` alone is enough to freeze it; no `.eval()` is
        # forced here, so an outer `model.train()` may flip the AE submodule to
        # train mode without any effect: `AutoEncoder.normalize` /
        # `inv_normalize` call `bn.eval()` themselves on every invocation, so no
        # running statistic can move whichever mode the module sits in, the AE
        # carries no Dropout and its GroupNorms are mode independent.
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
        self.ae.requires_grad_(False)

        self.denoise_model = models.__dict__[denoise_model_type](
            in_channels=in_channels,
            context_in_dim=context_in_dim,
            axes_dim=axes_dim,
            theta=theta,
            num_refiner_layers=num_refiner_layers,
            adaln_embed_dim=adaln_embed_dim,
            max_ref_images=max_ref_images,
            use_gradient_checkpoint=use_gradient_checkpoint,
            attention_backend=attention_backend,
        )
        self.out_channels = self.denoise_model.out_channels

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
        # The autocast is opted out of unconditionally. Precision here is
        # decided entirely by FSDP2's `MixedPrecisionPolicy`, which gathers
        # every VLM shard as `torch.bfloat16`, so an autocast on top would
        # only add a second, redundant casting rule -- and leaving it on would
        # let the caller's amp mode silently override the policy the whole
        # model was sharded with.
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
        # the reference stream keeps in `forward`: a zero-length `ctx` would
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
        # with, which is the same unconditional signal the CFG dropout in
        # `forward` feeds, so the fallback needs no separate convention.
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

    def forward(self,
                target_image,
                timesteps,
                input_ids,
                attention_mask,
                prompt_start_idx,
                noise=None,
                pixel_values=None,
                image_grid_thw=None,
                mm_token_type_ids=None,
                reference_images=None):
        """
        End-to-end forward: VLM encode + VAE encode + flow-matching denoise.

        During training, CFG dropout is applied: with probability
        ``cfg_dropout_prob``, the condition of each sample is independently
        replaced by a single zero condition token (one zero text token and one
        zero reference token). This teaches the model unconditional generation,
        enabling traditional Classifier-Free Guidance (CFG) at inference time.

        The task (T2I vs TI2I) is inferred directly from the inputs: T2I passes
        no `pixel_values` / `reference_images`, while TI2I passes both the VLM
        image inputs and the VAE-side `reference_images`.

        Args:
            target_image: [B, 3, H, W] clean target image in [-1, 1]. The VAE
                encodes it to x0; a noised latent x_t is built via rectified
                flow interpolation with `timesteps` and `noise`.
            timesteps: [B] flow-matching timesteps in [0, 1].
            input_ids/attention_mask/prompt_start_idx: from the tokenizer.
            noise: [B, L_img, in_channels] or None. If None, sampled inside.
            pixel_values/image_grid_thw/mm_token_type_ids: VLM image inputs for
                the TI2I task (from the tokenizer), or None for T2I.
            reference_images: list (len B) of [N_i, 3, Hr, Wr] tensors in
                [-1, 1], or None. Reference latents are appended to the image
                stream with distinct temporal coordinates. Supported by all three
                denoise DiT variants (DoubleStream, SingleStream, MixStream).

        Returns:
            model_pred: [B, L_img, in_channels] predicted velocity.
            target: [B, L_img, in_channels] flow-matching target (noise - x0).
        """
        target_height, target_width = target_image.shape[
            2], target_image.shape[3]
        assert target_height % 16 == 0 and target_width % 16 == 0

        img_in_dtype = torch.bfloat16

        # ---- condition feature from the VLM ----
        ctx, ctx_ids, ctx_mask = self.encode_condition(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_start_idx=prompt_start_idx,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids)
        # `ctx` is a feature value feeding the DiT's txt_in nn.Linear, so it is
        # aligned to the DiT's runtime dtype.
        ctx = ctx.to(img_in_dtype)
        # `ctx_ids` is a position index feeding only rope(), so it is pinned to
        # float32 unconditionally: bf16 is integer-exact only up to 256, above
        # which neighbouring token positions collapse and corrupt the RoPE
        # phase. The `pos.float()` inside rope() cannot undo that rounding.
        ctx_ids = ctx_ids.float()

        # ---- CFG dropout: zero out conditions for random samples ----
        # A dropped condition keeps exactly ONE valid token instead of an
        # all-False mask: `ctx_mask` is the key-padding mask of the DiT's
        # context refiner, so an all-masked softmax row would produce NaN that
        # leaks into the image stream through the joint attention. One zero
        # token is also what the inference model's `generate()` feeds its
        # unconditional branch, so the training and inference unconditional
        # signals match. The reference stream is dropped further down, once its
        # padded tensors exist, so `drop_mask` is carried over to there.
        drop_mask = None
        if self.training and self.cfg_dropout_prob > 0.0:
            B_cond, L_cond = ctx.shape[0], ctx.shape[1]
            drop_mask = torch.rand(B_cond,
                                   device=ctx.device) < self.cfg_dropout_prob
            # Out of place on purpose: `ctx.to(img_in_dtype)` above is a no-op
            # when the dtype already matches, so `ctx` still aliases the tensor
            # built inside `encode_condition` and an in-place write would
            # corrupt the VLM's autograd graph.
            ctx = torch.where(drop_mask[:, None, None], torch.zeros_like(ctx),
                              ctx)
            # Keep exactly ONE valid token (index 0) for the dropped samples.
            first_token_mask = torch.arange(L_cond,
                                            device=ctx.device)[None, :] == 0
            ctx_mask = torch.where(drop_mask[:, None], first_token_mask,
                                   ctx_mask)

        # ---- target latent (x0) and rectified-flow noised latent (x_t) ----
        x0, img_ids = self.encode_image_latent(target_image, t_coord=0.0)
        # The AE emits bf16; x0 / noise / target are kept in float32 because
        # `target` is the regression label of the flow-matching objective, and
        # any rounding there becomes an irreducible loss floor. Only `x_t`,
        # which feeds the DiT's img_in nn.Linear, drops to the DiT dtype.
        x0 = x0.float()
        # `img_ids` is an index, pinned to float32 for the same reason as
        # `ctx_ids`. It also seeds `ref_ids` below (new_zeros inherits the
        # dtype), so this one cast keeps the whole position-id chain in fp32.
        img_ids = img_ids.float()

        if noise is None:
            noise = torch.randn_like(x0)
        noise = noise.float()

        t = timesteps.float()[:, None, None]
        x_t = ((1.0 - t) * x0 + t * noise).to(img_in_dtype)
        target = noise - x0

        B = x_t.shape[0]
        device = x_t.device

        # ---- reference image latents (TI2I task) ----
        ref_tokens = None
        ref_ids = None
        ref_valid = None
        ref_index = None
        max_ref_len = 0
        if reference_images is not None:
            ref_tokens_list = []
            ref_ids_list = []
            ref_index_list = []
            for i, per_sample_refs in enumerate(reference_images):
                if per_sample_refs is None or len(per_sample_refs) == 0:
                    ref_tokens_list.append(x_t.new_zeros(0, self.in_channels))
                    ref_ids_list.append(x_t.new_zeros(0, 4))
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
                    ref_img = per_sample_refs[j][None].to(target_image.device)
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

            # A TI2I batch always keeps at least one reference slot, even when
            # every sample was CFG-dropped: collapsing the reference stream
            # would hand the DiT a different topology (no vec_zero, no
            # is_source, no reference refiner) than the inference model's.
            max_ref_len = max(
                1,
                max(per_sample_tokens.shape[0]
                    for per_sample_tokens in ref_tokens_list))
            ref_tokens = x_t.new_zeros(B, max_ref_len, self.in_channels)
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
                    ref_tokens[i, :n] = ref_tokens_list[i].to(x_t.dtype)
                    ref_ids[i, :n] = ref_ids_list[i].to(img_ids.dtype)
                    ref_valid[i, :n] = True
                    ref_index[i, :n] = ref_index_list[i]
                else:
                    # Reference-free sample (CFG-dropped, or a T2I sample in a
                    # TI2I batch): keep one valid slot (zero latent, identity
                    # index 0), since an all-False `ref_valid` row would make
                    # the reference refiner's softmax all-masked and produce
                    # NaN. Its id sits on the reference time coordinate so the
                    # 4D RoPE keeps it apart from the noise-target token.
                    ref_ids[i, 0, 0] = self.ref_time_coord_scale
                    ref_valid[i, 0] = True

            # ---- CFG dropout on the reference stream ----
            # Applied to the padded tensors rather than to `reference_images`,
            # so a dropped sample is resolved by a GPU-side mask instead of by
            # the `drop_mask.tolist()` readout that used to cost one GPU->CPU
            # sync per step. Out of place, so the batch produced by the
            # collator stays intact. A dropped sample collapses to the single
            # zero reference token the reference-free branch above builds,
            # which is also what the inference model's `generate()` feeds its
            # unconditional branch.
            if drop_mask is not None:
                keep_mask = ~drop_mask[:, None]
                drop_ids = ref_ids.new_zeros(1, max_ref_len, 4)
                drop_ids[0, 0, 0] = self.ref_time_coord_scale
                drop_valid = torch.arange(max_ref_len,
                                          device=device)[None, :] == 0
                ref_tokens = torch.where(keep_mask[..., None], ref_tokens,
                                         torch.zeros_like(ref_tokens))
                ref_ids = torch.where(keep_mask[..., None], ref_ids, drop_ids)
                ref_valid = torch.where(keep_mask, ref_valid, drop_valid)
                ref_index = torch.where(keep_mask, ref_index,
                                        torch.zeros_like(ref_index))

        # ---- denoise DiT (velocity prediction on the noise-target tokens) ----
        model_pred = self.denoise_model(x_t,
                                        img_ids,
                                        timesteps,
                                        ctx,
                                        ctx_ids,
                                        ctx_mask,
                                        ref_tokens=ref_tokens,
                                        ref_ids=ref_ids,
                                        ref_valid=ref_valid,
                                        ref_index=ref_index)

        return model_pred, target
