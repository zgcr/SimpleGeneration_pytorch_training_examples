# SeFi-Image vs FLUX2 去噪模型架构对比分析

> **分析范围**：仅聚焦于去噪模型（Denoising Model / DiT）部分的网络结构差异与创新点
>
> **SeFi-Image 代码路径**：`/opt/nas/p/zhugechaoran/download/code/SeFi-Image/sefi/modeling/flux2_sefi_transformer.py` + `sefi/runner.py` + `sefi/builder.py`
>
> **FLUX2 代码路径**：`/opt/nas/p/zhugechaoran/download/code/flux2/src/flux2/model.py` + `flux2/src/flux2/sampling.py`

---

## 一、整体架构概览

### FLUX2 去噪模型架构

```
输入: x (latent tokens), ctx (text embeddings), timestep, guidance
     │
     ▼
time_in(timestep_emb) + guidance_in(guidance_emb) → vec
     │
     ├─→ double_stream_modulation_img(vec) → mod_img（全层共享）
     ├─→ double_stream_modulation_txt(vec) → mod_txt（全层共享）
     └─→ single_stream_modulation(vec) → mod_single（全层共享）
     │
     ├─→ img_in(x) → img_tokens
     └─→ txt_in(ctx) → txt_tokens
     │
     ▼
DoubleStreamBlock × depth (默认8层)
  - img/txt 各自独立 LayerNorm + QKV + MLP
  - Joint Attention: cat(txt_q, img_q), cat(txt_k, img_k), cat(txt_v, img_v)
  - AdaLN-Zero 调制
     │
cat(txt, img) → 拼接
     │
     ▼
SingleStreamBlock × depth_single_blocks (默认48层)
  - 统一处理拼接后的 tokens
  - linear1 → [QKV, MLP_input] → Attention + MLP → linear2
  - AdaLN-Zero 调制
     │
去掉 txt tokens
     │
     ▼
final_layer (LastLayer): AdaLN + Linear(hidden_size → out_channels)
     │
     ▼
输出: velocity prediction [B, L_img, 128]
```

### SeFi-Image 去噪模型架构

```
输入: hidden_states (semantic+texture latent), encoder_hidden_states (text),
      timestep_sem, timestep_tex
     │
     ▼
SEFIDualTimestepEmbeddings:
  timestep_sem×1000 → time_proj → semantic_embedder → sem_emb (half_dim)
  timestep_tex×1000 → time_proj → texture_embedder → tex_emb (half_dim)
  temb = concat([sem_emb, tex_emb])  ← 替代 FLUX2 的 time_in + guidance_in
     │
     ├─→ double_stream_modulation_img(temb) → mod_img（全层共享）
     ├─→ double_stream_modulation_txt(temb) → mod_txt（全层共享）
     └─→ single_stream_modulation(temb) → mod_single（全层共享）
     │
     ├─→ x_embedder(hidden_states) → img_tokens
     └─→ context_embedder(encoder_hidden_states) → txt_tokens
     │
     ▼
双流 Transformer Blocks × num_layers (来自 Flux2Transformer2DModel)
  - 与 FLUX2 相同的 DoubleStreamBlock 结构
  - 使用相同的全层共享调制参数
     │
cat(txt, img) → 拼接
     │
     ▼
单流 Transformer Blocks × num_single_layers
  - 与 FLUX2 相同的 SingleStreamBlock 结构
     │
去掉 txt tokens
     │
     ▼
norm_out(hidden_states, temb) + proj_out(hidden_states)
     │
     ▼
输出: velocity prediction [B, L_img, total_channels]
     → 分离为 vel_sem 和 vel_tex，分别用独立 dt 更新
```

---

## 二、去噪模型网络结构变化创新点详细分析

### 创新点 1：双时间步嵌入模块 (SEFIDualTimestepEmbeddings) 替代单时间步+Guidance嵌入

**FLUX2 的做法：**
```python
# FLUX2: model.py
self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)  # 单时间步
self.guidance_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)  # guidance嵌入

# forward:
vec = self.time_in(timestep_embedding(timesteps, 256))
if self.use_guidance_embed:
    vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
# vec: [B, hidden_size]，用于所有调制
```

FLUX2 使用一个 `MLPEmbedder`（Linear→SiLU→Linear，输入256维，输出 hidden_size=6144 维）将单一时间步编码为条件向量 `vec`，可选地加上 guidance 嵌入。两者共享同一隐藏维度 `hidden_size`。

**SeFi-Image 的做法：**
```python
# SeFi-Image: flux2_sefi_transformer.py
class SEFIDualTimestepEmbeddings(nn.Module):
    def __init__(self, in_channels, embedding_dim, bias=False):
        half_dim = embedding_dim // 2
        self.time_proj = Timesteps(num_channels=in_channels, ...)  # 正弦位置编码（共享）
        self.semantic_embedder = TimestepEmbedding(in_channels, time_embed_dim=half_dim)
        self.texture_embedder = TimestepEmbedding(in_channels, time_embed_dim=half_dim)

    def forward(self, timestep_sem, timestep_tex):
        sem_proj = self.time_proj(timestep_sem)
        tex_proj = self.time_proj(timestep_tex)
        sem_emb = self.semantic_embedder(sem_proj)  # → [B, half_dim]
        tex_emb = self.texture_embedder(tex_proj)   # → [B, half_dim]
        return torch.cat([sem_emb, tex_emb], dim=-1)  # → [B, embedding_dim]
```

SeFi-Image 使用两个独立的 `TimestepEmbedding`（来自 diffusers，内部也是 Linear→SiLU→Linear），分别编码语义时间步和纹理时间步。二者共享前端的正弦 `time_proj`（Timesteps），但各自有独立的 MLP 映射。最终通过 **concat** 而非相加组合，各占 `hidden_size/2` 维度。

**区别与分析：**

| 对比维度 | FLUX2 | SeFi-Image |
|---------|-------|-----------|
| 时间步数量 | 1个全局时间步 | 2个独立时间步（语义+纹理） |
| 组合方式 | `time_emb + guidance_emb`（相加） | `concat([sem_emb, tex_emb])`（拼接） |
| 输出维度 | `hidden_size` | `hidden_size`（各占一半） |
| Guidance 嵌入 | 有，通过 `guidance_in` 单独编码 | 无，被移除（`time_guidance_embed = nn.Identity()`） |
| 正弦编码共享 | 单一 `timestep_embedding` 函数 | `time_proj` 共享给语义和纹理 |

**优点判断：这是生成效果维度的优化**。通过为语义和纹理通道提供独立的时间步条件信息，网络能感知两个通道处于不同的去噪阶段（因为 delta_t 偏移导致语义去噪领先于纹理），从而为每个通道生成更精准的速度预测。使用 concat 而非相加也保留了两个时间步嵌入的独立信息，避免信息混淆。

---

### 创新点 2：移除 Guidance Embedding 模块

**FLUX2 的做法：**
```python
# FLUX2: model.py
self.use_guidance_embed = params.use_guidance_embed  # True for FLUX2 dev
if self.use_guidance_embed:
    self.guidance_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, disable_bias=True)

# forward:
if self.use_guidance_embed:
    guidance_emb = timestep_embedding(guidance, 256)
    vec = vec + self.guidance_in(guidance_emb)
```

FLUX2 使用一个独立的 `MLPEmbedder`（`guidance_in`）将 guidance scale 值编码为 256→6144 维向量，加到时间步嵌入上。这个模块是蒸馏模型独有的设计，允许模型在单次前向传播中感知 guidance 强度，无需做两次前向的 CFG。

**SeFi-Image 的做法：**
```python
# SeFi-Image: flux2_sefi_transformer.py
self.backbone.time_guidance_embed = nn.Identity()  # 替换为恒等映射

# builder.py:
transformer_cfg["guidance_embeds"] = False  # 构建配置时直接禁用
```

SeFi-Image 在构建 Flux2Transformer2DModel backbone 时就设置 `guidance_embeds=False`，并在初始化后将 `time_guidance_embed` 替换为 `nn.Identity()`，彻底移除 guidance embedding 模块。

**区别与分析：**

| 对比维度 | FLUX2 | SeFi-Image |
|---------|-------|-----------|
| Guidance 编码方式 | `guidance_in` MLP嵌入到 vec 中 | 不在 DiT 内部编码 guidance |
| Guidance 实现位置 | DiT 网络内部 | 推理循环外部（CFG 或 AutoGuidance） |
| 参数量影响 | 额外一个 MLPEmbedder（256→6144→6144） | 节省了这部分参数 |

**优点判断：这不是直接的生成效果优化，而是架构简化优化**。SeFi-Image 将 guidance 的实现从网络内部移到了推理逻辑层（通过 CFG 或 AutoGuidance），使 DiT 网络本身更简洁。双时间步嵌入已经承载了足够丰富的条件信息，不再需要额外的 guidance embedding。这使得模型架构更清晰，且支持更灵活的 guidance 策略（如 AutoGuidance），而非绑定在网络权重中。

---

### 创新点 3：语义-纹理双通道 Latent 输入设计

**FLUX2 的做法：**
```python
# FLUX2: model.py
self.in_channels = params.in_channels  # 128
self.out_channels = params.in_channels  # 128
self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=False)  # 128→6144
```

FLUX2 的输入是单一的 128 通道 latent（由 VAE patch 操作产生），所有通道承载混合的语义和纹理信息。`img_in` 将 128 维映射到 hidden_size。

**SeFi-Image 的做法：**
```python
# SeFi-Image: builder.py
semantic_channels = _derive_semantic_channels(config)  # 从配置读取
texture_channels = int(texture_codec.texture_channels)  # latent_channels × 4
total_channels = int(semantic_channels + texture_channels)
transformer_cfg["in_channels"] = int(total_channels)
transformer_cfg["out_channels"] = int(total_channels)

# x_embedder 的输入维度变为 total_channels (semantic_channels + texture_channels)
```

SeFi-Image 的 latent 由两部分组成：
- **语义通道**（`semantic_channels`）：不经过 VAE，在纯噪声空间中操作
- **纹理通道**（`texture_channels = latent_channels × 4`）：由 VAE 编解码

两者拼接后作为一个整体输入到 DiT 中。`x_embedder`（即 `img_in`）的输入维度变为 `total_channels`。

```python
# runner.py - Euler 更新时分离
vel_sem = velocity[:, :self.semantic_channels]
vel_tex = velocity[:, self.semantic_channels:]
lat_sem = latents[:, :self.semantic_channels]
lat_tex = latents[:, self.semantic_channels:]
dt_sem = sigmas_sem_next - sigmas_sem_cur
dt_tex = sigmas_tex_next - sigmas_tex_cur
lat_sem = lat_sem + dt_sem * vel_sem
lat_tex = lat_tex + dt_tex * vel_tex
```

**区别与分析：**

| 对比维度 | FLUX2 | SeFi-Image |
|---------|-------|-----------|
| Latent 通道含义 | 128ch 混合语义+纹理 | semantic_channels + texture_channels 分离 |
| 输入投影 | `img_in`: 128→hidden_size | `x_embedder`: total_channels→hidden_size |
| 输出投影 | `proj_out`: hidden_size→128 | `proj_out`: hidden_size→total_channels |
| Euler 步进 | 统一 dt 更新所有通道 | 语义和纹理使用独立 dt |
| 解码 | 解码整个 128ch latent | 仅解码纹理通道，丢弃语义通道 |

**优点判断：这是生成效果维度的优化**。通过将 latent 空间显式拆分为语义和纹理两部分，模型可以对两者施加不同的去噪步长（dt_sem ≠ dt_tex），使语义结构先被确定，纹理细节在结构锚点上生成。这种分离设计的核心收益是：

1. 语义通道不需要经过 VAE，可以在更自由的表示空间中编码高层结构信息
2. 纹理通道保留 VAE 的高保真重建能力
3. 通过 delta_t 偏移实现「语义先行」的生成策略，有效减少结构不一致的问题

---

### 创新点 4：分离的 Euler 步长更新（Semantic-Texture Independent dt）

**FLUX2 的做法：**
```python
# FLUX2: sampling.py
img = img + (t_prev - t_curr) * pred  # 所有通道使用统一的 dt
```

FLUX2 使用统一的时间步差值 `(t_prev - t_curr)` 乘以速度预测，更新所有 128 个 latent 通道。

**SeFi-Image 的做法：**
```python
# SeFi-Image: runner.py
# 语义和纹理时间步独立计算
u_tex_cur = torch.clamp(u_sem_raw_cur - self.delta_t, min=0.0, max=1.0)
u_sem_cur = torch.clamp(u_sem_raw_cur, max=1.0)

timesteps_sem_cur, sigmas_sem_cur = self._timesteps_and_sigmas(u_sem_cur, ...)
timesteps_tex_cur, sigmas_tex_cur = self._timesteps_and_sigmas(u_tex_cur, ...)
_, sigmas_sem_next = self._timesteps_and_sigmas(u_sem_next, ...)
_, sigmas_tex_next = self._timesteps_and_sigmas(u_tex_next, ...)

# 独立步长
dt_sem = sigmas_sem_next - sigmas_sem_cur
dt_tex = sigmas_tex_next - sigmas_tex_cur

lat_sem = lat_sem + dt_sem * vel_sem
lat_tex = lat_tex + dt_tex * vel_tex
```

SeFi-Image 根据语义优先调度（delta_t 偏移），为语义和纹理通道计算不同的 sigma 和 dt。语义时间步始终领先于纹理时间步（`u_sem >= u_tex`），因此 `dt_sem` 和 `dt_tex` 在每一步通常不同。

**区别与分析：**

| 对比维度 | FLUX2 | SeFi-Image |
|---------|-------|-----------|
| 步长计算 | `dt = t_prev - t_curr`，所有通道统一 | `dt_sem` 和 `dt_tex` 独立计算 |
| sigma 调度 | 全局统一的 sigma schedule | 语义和纹理各自有独立的 sigma schedule |
| 语义-纹理关系 | 所有通道同步去噪 | 语义领先 delta_t 去噪 |

**优点判断：这是生成效果维度的优化**。与创新点 3 紧密相关，独立步长是语义优先策略的执行机制。在去噪早期，语义通道以更大的步幅快速确定结构，而纹理通道在语义锚点指引下逐步精细化。这种异步去噪策略可以减少生成图像中"结构未定但纹理已固化"导致的不连贯问题。

---

### 创新点 5：语义优先去噪调度（Semantic-First Denoising Schedule）

**FLUX2 的做法：**
```python
# FLUX2: sampling.py
def get_schedule(num_steps, image_seq_len):
    mu = compute_empirical_mu(image_seq_len, num_steps)
    timesteps = torch.linspace(1, 0, num_steps + 1)
    timesteps = generalized_time_snr_shift(timesteps, mu, 1.0)
    return timesteps.tolist()
```

FLUX2 使用基于 `generalized_time_snr_shift` 的经验性 mu 计算时间步调度。时间步从 1 到 0 线性分布后经过 SNR shift 变换。所有通道共享同一调度。

**SeFi-Image 的做法：**
```python
# SeFi-Image: runner.py
u_base_unit = torch.linspace(0.0, 1.0, steps=num_inference_steps + 1)
u_shifted_unit = _apply_timestep_shift_unit_interval(u_base_unit, self.timestep_shift_alpha)
# alpha shift: t' = alpha * t / (1 + (alpha-1) * t)

u_sem_raw_schedule = u_shifted_unit * (1.0 + self.delta_t)
# 每一步:
u_tex_cur = clamp(u_sem_raw_cur - delta_t, 0, 1)
u_sem_cur = clamp(u_sem_raw_cur, max=1)
```

SeFi-Image 使用一个参数化的时间步偏移公式 `t' = alpha * t / (1 + (alpha-1) * t)`，然后通过 `delta_t` 参数在语义和纹理之间创建时间偏移。delta_t 的存在保证了在整个去噪过程中 `u_sem >= u_tex`，即语义去噪始终领先。

**区别与分析：**

| 对比维度 | FLUX2 | SeFi-Image |
|---------|-------|-----------|
| 调度类型 | SNR-based mu shift | alpha 参数化偏移 + delta_t 双通道分离 |
| 调度公式 | `exp(mu) / (exp(mu) + (1/t - 1)^sigma)` | `alpha * t / (1 + (alpha-1) * t)` |
| 通道差异 | 无 | 语义始终领先 delta_t |
| 可配置参数 | 隐式由图像序列长度决定的 mu | 显式 `timestep_shift_alpha` 和 `delta_t` |

**优点判断：这是生成效果维度的优化**。语义优先调度是 SeFi-Image 的核心创新理念在网络输入层面的体现。通过让语义通道先完成去噪，纹理通道可以在已确定的语义结构上生成细节，避免了「先画细节后改结构」的冲突。这种调度策略更符合人类绘画的「先构图后上色」的认知过程。

---

### 创新点 6：仅解码纹理通道的输出策略

**FLUX2 的做法：**
```python
# FLUX2: sampling.py + autoencoder.py
# 去噪完成后，整个 latent 送入 VAE Decoder
img = ae.decode(latent)  # 全部 128ch latent → RGB 图像
```

FLUX2 对去噪完成后的完整 128 通道 latent 进行 VAE 解码。

**SeFi-Image 的做法：**
```python
# SeFi-Image: runner.py
texture_latents = latents[:, self.semantic_channels:]  # 只取纹理通道
decoded = self.texture_codec.decode_texture(texture_latents, pipeline_cls=self.pipeline_cls)
# texture_codec 内部: denormalize → unpatchify → VAE.decode()
```

SeFi-Image 在去噪完成后，**丢弃语义通道**，仅将纹理通道送入 VAE Decoder 解码。语义通道在整个去噪过程中仅作为结构锚点引导纹理生成，本身不参与最终图像重建。

**区别与分析：**

| 对比维度 | FLUX2 | SeFi-Image |
|---------|-------|-----------|
| 解码输入 | 全部 latent 通道 | 仅纹理通道 |
| 语义通道 | 不存在独立语义通道 | 作为辅助引导，最终丢弃 |
| VAE 兼容性 | 完整匹配 | 纹理通道维度 = latent_channels × 4，与 VAE 兼容 |

**优点判断：这是生成效果维度的优化**。语义通道的角色是「隐式结构锚点」，它不直接映射到像素空间，而是在 DiT 内部通过联合注意力和共享调制参数影响纹理通道的生成。这种设计让语义通道可以在一个不受 VAE 重建约束的自由空间中编码高层语义信息，而纹理通道专注于高保真像素重建。两者的分工使各自的表示空间更专注、更高效。

---

### 创新点 7：移除参考图像/KV缓存机制，专注文生图

**FLUX2 的做法：**
```python
# FLUX2: model.py
def forward_kv_extract(self, x, x_ids, timesteps, ctx, ctx_ids, guidance,
                       x_seq_concat, x_seq_concat_ids, ref_fixed_timestep=0.0):
    # 拼接参考图 tokens
    x = torch.cat([x_seq_concat, x], dim=1)
    # 为参考图计算固定时间步的 modulation
    ref_vec = self.time_in(timestep_embedding(torch.full_like(timesteps, ref_fixed_timestep), 256))
    # 混合 ref 和 img 的 modulation
    double_block_mod_img = _blend_double_mods(double_block_mod_img, ref_double_mod, num_ref_tokens, L_img)
    # 提取 KV cache
    cache = {"k_ref": k[:, :, ref_start:ref_end, :].clone(),
             "v_ref": v[:, :, ref_start:ref_end, :].clone()}

def forward_kv_cached(self, x, x_ids, timesteps, ctx, ctx_ids, guidance, kv_cache):
    # 后续步骤复用缓存的 KV
```

FLUX2 的 DiT 内置了完整的参考图像处理机制：
- `forward_kv_extract`：第一步完整前向，提取参考图的 KV 缓存
- `forward_kv_cached`：后续步骤复用缓存
- `_blend_double_mods` / `_blend_single_mods`：按位置混合参考图和生成图的调制参数
- `causal_attn_fn`：因果注意力，参考图只能自注意力，生成图可看到所有

**SeFi-Image 的做法：**
```python
# SeFi-Image: flux2_sefi_transformer.py
def forward(self, hidden_states, timestep_sem, timestep_tex,
            encoder_hidden_states, txt_ids, img_ids, joint_attention_kwargs=None):
    # 无参考图相关参数
    # 无 KV cache 机制
    # 无因果注意力
    # 纯文生图前向
```

SeFi-Image 的 DiT 没有任何参考图像输入接口，不支持 `forward_kv_extract` 或 `forward_kv_cached`，也不包含因果注意力函数。模型专注于文生图任务。

**区别与分析：**

| 对比维度 | FLUX2 | SeFi-Image |
|---------|-------|-----------|
| 参考图支持 | ✅ 支持单/多参考图 | ❌ 不支持 |
| KV Cache | ✅ 有，加速参考图推理 | ❌ 无 |
| 因果注意力 | ✅ `causal_attn_fn` | ❌ 无（标准全注意力） |
| Modulation 混合 | ✅ `_blend_double_mods` / `_blend_single_mods` | ❌ 无 |
| forward 变体 | 3个（forward / forward_kv_extract / forward_kv_cached） | 1个（forward） |

**优点判断：这不是生成效果的优化，而是架构简化和功能聚焦的优化**。SeFi-Image 通过移除参考图像机制，大幅简化了 DiT 的代码复杂度和推理逻辑。模型专注于文生图这一核心任务，减少了不必要的计算开销和代码维护成本。如果不需要图像编辑功能，这种简化使模型更易训练和部署。但代价是失去了图像编辑能力。

---

### 创新点 8：AutoGuidance 机制（使用独立小模型作为无条件基线）

**FLUX2 的做法：**
```python
# FLUX2: sampling.py
# 方式1: 内嵌 guidance embedding（蒸馏模型）
vec = vec + self.guidance_in(timestep_embedding(guidance, 256))

# 方式2: 标准 CFG（denoise_cfg）
pred_uncond, pred_cond = pred.chunk(2)
pred = pred_uncond + guidance * (pred_cond - pred_uncond)
```

FLUX2 提供两种 guidance 方式：内嵌 guidance embedding（蒸馏模型单次前向）或标准 CFG（双次前向）。

**SeFi-Image 的做法：**
```python
# SeFi-Image: runner.py
if self.autoguidance_enabled:
    # 使用独立的小模型做"无条件"预测
    pred_base = self._predict_velocity(
        self.autoguidance_transformer,  # 可以是不同规模的 SeFi 模型
        packed_latents=packed_latents,
        timesteps_sem=timesteps_sem_cur,
        timesteps_tex=timesteps_tex_cur,
        encoder_hidden_states=autoguidance_prompt_embeds,  # 可以是不同文本编码器的输出
        txt_ids=autoguidance_text_ids,
        img_ids=latent_ids,
    )
    velocity = _combine_guided_velocity(pred_base, pred_cond, guidance_scale)
    # = pred_base + guidance_scale * (pred_cond - pred_base)
```

SeFi-Image 支持 AutoGuidance，使用一个**独立的更小的 transformer 模型**作为基线预测（替代标准 CFG 中的空文本预测）。这个小模型可以有不同的规模、不同的配置，甚至使用不同的文本编码器。

**区别与分析：**

| 对比维度 | FLUX2 | SeFi-Image |
|---------|-------|-----------|
| Guidance 方式 | 内嵌embedding / 标准CFG | 标准CFG + AutoGuidance |
| 无条件基线 | 空文本的同一模型预测 | 独立小模型的条件预测 |
| 计算效率 | CFG 需要 2× 大模型前向 | AutoGuidance: 1× 大模型 + 1× 小模型 |
| 灵活性 | 固定 | 小模型可独立配置和训练 |

**优点判断：这是生成效果和推理效率两个维度的优化**。AutoGuidance 相比标准 CFG 有两个优势：(1) 用小模型替代大模型的空文本前向，减少计算量；(2) 小模型提供的基线可能比简单的空文本预测更有信息量，从而产生更好的 guidance 效果。这在推理效率和生成质量之间取得了更好的平衡。

---

### 创新点 9：Guidance Interval（引导区间限制）

**FLUX2 的做法：**
```python
# FLUX2: sampling.py - denoise / denoise_cfg
# guidance 在所有去噪步骤中始终生效
for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
    pred = model(...)
    # guidance 始终应用
    img = img + (t_prev - t_curr) * pred
```

FLUX2 的 guidance（无论是内嵌 embedding 还是 CFG）在所有去噪步骤中都生效。

**SeFi-Image 的做法：**
```python
# SeFi-Image: runner.py
guidance_active = _guidance_interval_is_active(
    base_sigmas_schedule[step],
    self.guidance_interval_sigma_lo,
    self.guidance_interval_sigma_hi,
)

if not guidance_active:
    velocity = pred_cond  # 直接使用条件预测，不应用 guidance
elif self.autoguidance_enabled:
    # 应用 AutoGuidance
elif guidance_scale > 1.0:
    # 应用标准 CFG
```

```python
def _guidance_interval_is_active(sigma, sigma_lo, sigma_hi):
    return float(sigma_lo) < sigma_value <= float(sigma_hi)
```

SeFi-Image 支持限制 guidance 仅在特定 sigma 区间内生效。当 sigma 不在 `(sigma_lo, sigma_hi]` 范围内时，直接使用条件预测，跳过 guidance 计算。

**区别与分析：**

| 对比维度 | FLUX2 | SeFi-Image |
|---------|-------|-----------|
| Guidance 生效范围 | 全程 | 可配置 sigma 区间 |
| 参数 | 无 | `guidance_interval_sigma_lo`, `guidance_interval_sigma_hi` |
| 计算节省 | 无 | 在区间外跳过 guidance 计算，节省 ~50% 计算 |

**优点判断：这是推理效率和生成效果两个维度的优化**。研究表明 guidance 在去噪过程的不同阶段有不同的影响：在早期（高 sigma）阶段 guidance 对确定整体结构至关重要，但在晚期（低 sigma）阶段过强的 guidance 可能导致过饱和或伪影。通过限制 guidance 区间，可以在不影响生成质量的前提下节省计算量，甚至在某些情况下提升生成质量。

---

### 创新点 10：可配置的时间步偏移公式

**FLUX2 的做法：**
```python
# FLUX2: sampling.py
def generalized_time_snr_shift(t, mu, sigma):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

def compute_empirical_mu(image_seq_len, num_steps):
    # 基于经验拟合的分段线性函数
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    ...
```

FLUX2 使用基于 SNR 的 shift 函数，mu 由图像序列长度和步数通过经验公式隐式计算。

**SeFi-Image 的做法：**
```python
# SeFi-Image: runner.py
def _apply_timestep_shift_unit_interval(u_unit, alpha):
    """Apply t' = alpha*t / (1 + (alpha-1)*t) on unit coordinate u in [0, 1]."""
    denominator = 1.0 + (alpha - 1.0) * u_unit
    return (alpha * u_unit) / denominator
```

SeFi-Image 使用更简洁的参数化偏移公式，通过单一 `alpha` 参数控制偏移程度。公式 `t' = alpha*t / (1 + (alpha-1)*t)` 是一个 Möbius 变换，当 `alpha > 1` 时将时间步向早期集中（更多步数用于去噪早期），当 `alpha < 1` 时反之。

**区别与分析：**

| 对比维度 | FLUX2 | SeFi-Image |
|---------|-------|-----------|
| 偏移公式 | `exp(mu) / (exp(mu) + (1/t - 1)^sigma)` | `alpha * t / (1 + (alpha-1) * t)` |
| 参数 | 隐式 mu（由图像大小和步数决定） | 显式 `timestep_shift_alpha` |
| 可控性 | 低（硬编码的经验公式） | 高（直接可调的单一参数） |
| 默认值 | 依赖图像尺寸 | Base/RL: alpha=0.3, Turbo: alpha=1.0 |

**优点判断：这是模型可调性和灵活性维度的优化**。简洁的参数化公式使用户和研究者更容易理解和调整时间步偏移策略。不同的 alpha 值可以适配不同的模型家族（Base/RL vs Turbo），提供更好的超参搜索体验。

---

### 创新点 11：丰富的模型规模预设体系

**FLUX2 的做法：**
```python
# FLUX2: model.py - 3种规模
@dataclass
class Flux2Params:  # 32B
    hidden_size: int = 6144; num_heads: int = 48; depth: int = 8; depth_single_blocks: int = 48

@dataclass
class Klein9BParams:  # 9B
    hidden_size: int = 4096; num_heads: int = 32; depth: int = 8; depth_single_blocks: int = 24

@dataclass
class Klein4BParams:  # 4B
    hidden_size: int = 3072; num_heads: int = 24; depth: int = 5; depth_single_blocks: int = 20
```

FLUX2 提供 3 种固定规模的模型参数配置。

**SeFi-Image 的做法：**
```python
# SeFi-Image: builder.py - 9种预设 + 自定义
SEFI_SCALE_PRESETS = {
    "0p5b": {"attention_head_dim": 128, "num_attention_heads": 12, "num_layers": 3,  "num_single_layers": 10, "joint_attention_dim": 6144},
    "1b":   {"attention_head_dim": 128, "num_attention_heads": 16, "num_layers": 4,  "num_single_layers": 12, "joint_attention_dim": 6144},
    "2b":   {"attention_head_dim": 128, "num_attention_heads": 20, "num_layers": 4,  "num_single_layers": 16, "joint_attention_dim": 6144},
    "3b":   {"attention_head_dim": 128, "num_attention_heads": 22, "num_layers": 5,  "num_single_layers": 18, "joint_attention_dim": 7680},
    "4b":   {"attention_head_dim": 128, "num_attention_heads": 24, "num_layers": 5,  "num_single_layers": 20, "joint_attention_dim": 7680},
    "5b":   {"attention_head_dim": 128, "num_attention_heads": 26, "num_layers": 6,  "num_single_layers": 21, "joint_attention_dim": 7680},
    "6b":   {"attention_head_dim": 128, "num_attention_heads": 28, "num_layers": 6,  "num_single_layers": 22, "joint_attention_dim": 7680},
    "8b":   {"attention_head_dim": 128, "num_attention_heads": 30, "num_layers": 7,  "num_single_layers": 24, "joint_attention_dim": 7680},
    "9b":   {"attention_head_dim": 128, "num_attention_heads": 32, "num_layers": 8,  "num_single_layers": 24, "joint_attention_dim": 12288},
}
# 还支持 transformer_scale: "custom" + transformer_overrides 自定义配置
```

SeFi-Image 提供 9 种从 0.5B 到 9B 的细粒度预设，并支持通过 `custom` + `transformer_overrides` 完全自定义模型规模。

**区别与分析：**

| 对比维度 | FLUX2 | SeFi-Image |
|---------|-------|-----------|
| 预设规模数 | 3种 (4B/9B/32B) | 9种 (0.5B~9B) + 自定义 |
| 规模粒度 | 粗粒度 | 细粒度（每 1-2B 一个预设） |
| 自定义支持 | 无 | ✅ `custom` + `transformer_overrides` |
| 最小规模 | 4B | 0.5B |

**优点判断：这是部署灵活性和资源适配维度的优化**。更细粒度的规模预设使用户可以根据硬件资源精确选择最合适的模型大小，而非被迫在过大和过小之间选择。0.5B 的最小规模使模型可以在消费级 GPU 甚至移动端上运行。自定义规模支持为研究实验提供了最大灵活性。

---

### 创新点 12：文本编码器选择差异（Qwen3-VL 替代 Mistral-Small-24B）

**FLUX2 的做法：**
```python
# FLUX2: text_encoder.py
class Mistral3SmallEmbedder(nn.Module):  # 24B 参数的多模态 LLM
    OUTPUT_LAYERS_MISTRAL = [10, 20, 30]
    # hidden_size = 5120, output_dim = 5120 × 3 = 15360
    # 支持图文多模态理解 + Prompt Upsampling + 内容安全过滤

class Qwen3Embedder(nn.Module):  # Klein 系列使用
    OUTPUT_LAYERS_QWEN3 = [9, 18, 27]
    # Qwen3-4B: output_dim = 2560 × 3 = 7680
    # Qwen3-8B: output_dim = 4096 × 3 = 12288
    # 纯文本 LLM（Qwen3，非 VL 版本）
```

**SeFi-Image 的做法：**
```python
# SeFi-Image: qwen3vl_text_encoder.py
class Qwen3VLTextEncoder(nn.Module):  # 基于 Qwen3-VL（视觉语言模型）
    QWEN3VL_MODEL_PATHS = {
        "qwen3vl_2b": "Qwen3-VL-2B-Instruct",   # hidden_size=2048, output_dim=6144
        "qwen3vl_4b": "Qwen3-VL-4B-Instruct",   # hidden_size=2560, output_dim=7680
        "qwen3vl_8b": "Qwen3-VL-8B-Instruct",   # hidden_size=4096, output_dim=12288
    }
    # 使用 Qwen3VLForConditionalGeneration，但删除视觉部分
    # del self.model.model.visual
    hidden_layers = (9, 18, 27)
```

**区别与分析：**

| 对比维度 | FLUX2 (dev) | FLUX2 (klein) | SeFi-Image |
|---------|-------------|---------------|-----------|
| 文本编码器 | Mistral-Small-3.2-24B | Qwen3-4B/8B (纯文本) | Qwen3-VL-2B/4B/8B (VL模型去视觉) |
| 参数量 | 24B | 4B/8B | 2B/4B/8B |
| 模型类型 | 多模态 LLM | 纯文本 LLM | VL LLM（去掉视觉部分） |
| 提取层 | 10/20/30 | 9/18/27 | 9/18/27 |
| 输出维度 | 15360 | 7680/12288 | 6144/7680/12288 |

**优点判断：这是部署效率维度的优化，但可能牺牲部分文本理解能力**。使用更小的文本编码器（最小 2B vs FLUX2 的 24B）显著减少了显存占用和推理延迟。选择 Qwen3-VL 而非纯 Qwen3 的基底模型可能保留了部分视觉-语言对齐的先验知识，但删除视觉模块后这种优势有多大存疑。较小的文本编码器在复杂文本理解能力上可能弱于 FLUX2 的 24B Mistral。

---

### 创新点 13：DiT 骨干基于 diffusers 的 Flux2Transformer2DModel 封装

**FLUX2 的做法：**
```python
# FLUX2: model.py
class Flux2(nn.Module):
    # 完全自定义实现，所有组件（DoubleStreamBlock, SingleStreamBlock, 
    # SelfAttention, Modulation, LastLayer 等）都在同一文件中定义
    # 包含 forward / forward_kv_extract / forward_kv_cached 三种前向方法
```

FLUX2 使用完全自定义实现的 DiT 架构，所有组件定义在 `model.py` 中。

**SeFi-Image 的做法：**
```python
# SeFi-Image: flux2_sefi_transformer.py
class Flux2SEFITransformer2DModel(nn.Module):
    def __init__(self, backbone_config, text_input_dim):
        self.backbone = Flux2Transformer2DModel.from_config(backbone_config)
        # 复用 diffusers 库的 Flux2Transformer2DModel
        self.backbone.time_guidance_embed = nn.Identity()  # 修改 guidance
        self.dual_time_embed = SEFIDualTimestepEmbeddings(...)  # 新增双时间步
```

SeFi-Image 的 DiT 是对 diffusers 库中 `Flux2Transformer2DModel` 的**封装（wrapper）**。它复用了 Flux2 的 backbone（包括所有 transformer blocks、modulation 层、位置编码等），仅在外层修改了：
1. 时间步嵌入（替换为双时间步）
2. Guidance 嵌入（移除）
3. Forward 逻辑（适配双时间步）

**区别与分析：**

| 对比维度 | FLUX2 | SeFi-Image |
|---------|-------|-----------|
| 实现方式 | 完全自定义 | 封装 diffusers 的 Flux2Transformer2DModel |
| 内部 blocks | 自定义 DoubleStreamBlock / SingleStreamBlock | 复用 diffusers 的实现 |
| 修改范围 | N/A | 仅修改时间步嵌入、guidance、forward 逻辑 |
| 代码量 | ~830行（含所有组件） | ~247行（封装层） |
| 可维护性 | 需自行维护所有组件 | 享受 diffusers 库的更新和优化 |

**优点判断：这是工程效率和可维护性维度的优化**。通过复用 diffusers 的成熟实现，SeFi-Image 大幅减少了需要维护的代码量，同时可以享受 diffusers 库的持续优化（如更高效的 attention 实现、内存优化等）。封装模式也使得核心创新点（双时间步、语义优先调度）更加突出和清晰。但代价是对 backbone 内部结构的控制力较弱。

---

### 创新点 14：forward 中的参数自适应兼容（动态签名解析）

**FLUX2 的做法：**
```python
# FLUX2: model.py
# 固定的 forward 签名，直接调用内部组件
for block in self.double_blocks:
    img, txt, _ = block.forward_kv_extract(img, txt, pe_x, pe_ctx, 
                                            double_block_mod_img, double_block_mod_txt, ...)
```

FLUX2 的 forward 直接调用自定义的内部组件方法。

**SeFi-Image 的做法：**
```python
# SeFi-Image: flux2_sefi_transformer.py
def _resolve_double_stream_modulation_kwargs(self):
    params = inspect.signature(self.backbone.transformer_blocks[0].forward).parameters
    if "temb_mod_img" in params and "temb_mod_txt" in params:
        return "temb_mod_img", "temb_mod_txt"
    if "temb_mod_params_img" in params and "temb_mod_params_txt" in params:
        return "temb_mod_params_img", "temb_mod_params_txt"

def _resolve_single_stream_modulation_kwarg(self):
    params = inspect.signature(self.backbone.single_transformer_blocks[0].forward).parameters
    if "temb_mod" in params:
        return "temb_mod"
    if "temb_mod_params" in params:
        return "temb_mod_params"

# forward 中使用动态关键字参数
encoder_hidden_states, hidden_states = block(
    hidden_states=hidden_states,
    encoder_hidden_states=encoder_hidden_states,
    **{self._double_mod_img_kwarg: double_stream_mod_img,
       self._double_mod_txt_kwarg: double_stream_mod_txt},
    image_rotary_emb=concat_rotary_emb,
)
```

SeFi-Image 使用 `inspect.signature` 动态解析 backbone blocks 的 forward 签名，自动适配不同版本的 diffusers 库中 `Flux2TransformerBlock` 的参数命名（`temb_mod_img` vs `temb_mod_params_img`）。

**区别与分析：**

| 对比维度 | FLUX2 | SeFi-Image |
|---------|-------|-----------|
| 接口调用 | 固定签名，直接调用 | 动态签名解析，自适应调用 |
| diffusers 版本兼容 | 不依赖 diffusers | 兼容多个 diffusers 版本 |
| 稳健性 | 强（自包含） | 强（动态适配） |

**优点判断：这是工程兼容性维度的优化**。通过动态签名解析，SeFi-Image 可以兼容 diffusers 库不同版本中可能存在的 API 变化（如参数名从 `temb_mod_img` 改为 `temb_mod_params_img`），减少因上游库更新导致的兼容性问题。

---

## 三、创新点总结表

| 编号 | 创新点 | FLUX2 做法 | SeFi-Image 做法 | 优化维度 |
|------|--------|-----------|----------------|---------|
| 1 | 双时间步嵌入模块 | 单时间步 `time_in` + 可选 `guidance_in`，相加组合 | 双时间步 `semantic_embedder` + `texture_embedder`，concat 组合 | **生成效果** |
| 2 | 移除 Guidance Embedding | 网络内嵌 `guidance_in` MLPEmbedder | 移除，替换为 `nn.Identity()`，guidance 在推理循环外部实现 | **架构简化** |
| 3 | 语义-纹理双通道 Latent | 128ch 混合 latent | `semantic_channels + texture_channels` 分离，语义不经 VAE | **生成效果** |
| 4 | 分离的 Euler 步长 | 统一 `dt = t_prev - t_curr` | `dt_sem` 和 `dt_tex` 独立计算 | **生成效果** |
| 5 | 语义优先去噪调度 | SNR-based mu shift，所有通道同步 | alpha 偏移 + delta_t 使语义领先纹理 | **生成效果** |
| 6 | 仅解码纹理通道 | 解码全部 latent | 丢弃语义通道，仅解码纹理 | **生成效果** |
| 7 | 移除参考图/KV缓存 | 支持参考图 + KV cache + 因果注意力 | 全部移除，专注文生图 | **架构简化**（牺牲编辑能力） |
| 8 | AutoGuidance 机制 | 内嵌 guidance / 标准 CFG | 独立小模型作为基线预测 | **生成效果 + 推理效率** |
| 9 | Guidance Interval | guidance 全程生效 | 可限制 sigma 区间，区间外跳过 | **推理效率 + 生成效果** |
| 10 | 可配置时间步偏移 | 隐式经验 mu 公式 | 显式 alpha 参数化 Möbius 变换 | **模型可调性** |
| 11 | 丰富模型规模预设 | 3种固定规模 | 9种预设 + 自定义 | **部署灵活性** |
| 12 | 文本编码器替换 | Mistral-Small-24B / Qwen3 (纯文本) | Qwen3-VL-2B/4B/8B（去视觉部分） | **部署效率**（可能牺牲文本理解） |
| 13 | DiT 骨干封装复用 | 完全自定义实现 (~830行) | 封装 diffusers Flux2Transformer2DModel (~247行) | **工程效率** |
| 14 | 动态签名自适应 | 固定接口直接调用 | `inspect.signature` 动态解析 | **工程兼容性** |

---

## 四、核心创新理念总结

SeFi-Image 去噪模型相比 FLUX2 的**最核心创新**是 **「语义-纹理分离 + 语义优先去噪」** 范式（创新点 1, 3, 4, 5, 6 共同构成）。这一范式的核心思想是：

1. **分离**：将 latent 空间显式拆分为语义通道（编码高层结构）和纹理通道（编码低层细节），两者在不同的表示空间中操作
2. **先行**：通过 delta_t 偏移，让语义通道的去噪进度始终领先于纹理通道
3. **锚定**：语义通道在去噪过程中为纹理通道提供结构锚点，但自身不参与最终图像重建
4. **感知**：通过双时间步嵌入，让 DiT 网络显式感知两个通道处于不同的去噪阶段

这种设计将图像生成的「结构确定」和「细节填充」两个过程解耦，使模型可以先稳定地确定整体构图，然后在结构锚点上精细地生成纹理细节，从而潜在地提升生成图像的结构一致性和整体质量。

而**辅助创新**包括 AutoGuidance 机制（创新点 8）、Guidance Interval（创新点 9）等推理策略优化，以及丰富的规模预设（创新点 11）、文本编码器替换（创新点 12）等工程和部署优化。