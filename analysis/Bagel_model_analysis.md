# Bagel 模型代码全面分析报告

> 本报告基于 `Bagel/` 目录下的全部源代码实现进行深入分析，所有结论均由代码逻辑推导得出。

> **代码实现根目录**：`/root/code/Bagel/`

> **模型全称**：BAGEL — Unified Model for Multimodal Understanding and Generation（字节跳动 Seed 团队）

> **论文**：[Emerging Properties in Unified Multimodal Pretraining (arXiv:2505.14683)](https://arxiv.org/abs/2505.14683)

---

## 目录

1. [问题1：是否使用了VAE？VAE结构类型](#问题1是否使用了vaevae结构类型)
2. [问题2：是否使用了flow_matching的DIT模型？单流还是双流？](#问题2是否使用了flow_matching的dit模型单流还是双流)
3. [问题3：具体网络结构和子网络](#问题3具体网络结构和子网络)
4. [问题4：模型网络结构图](#问题4模型网络结构图)
5. [问题5：是否支持文生图和图像编辑](#问题5是否支持文生图和图像编辑)
6. [问题6：文生图流程图](#问题6文生图流程图)
7. [问题7：图像+文本提示图像编辑流程图](#问题7图像文本提示图像编辑流程图)
8. [问题8：相比FLUX.2模型的创新点](#问题8相比flux2模型的创新点)

---

## 问题1：是否使用了VAE？VAE结构类型

### 结论：是的，Bagel 使用了 VAE。其 VAE 结构与 FLUX.1 模型的 VAE **完全一致**，不是 FLUX.2 的 VAE 结构。

### 代码分析依据

**1. VAE 参数定义（`modeling/autoencoder.py` 第339-351行）**

```python
def load_ae(local_path: str) -> AutoEncoder:
    ae_params = AutoEncoderParams(
        resolution=256,
        in_channels=3,
        downsample=8,
        ch=128,
        out_ch=3,
        ch_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        z_channels=16,
        scale_factor=0.3611,
        shift_factor=0.1159,
    )
```

关键参数：
- `z_channels=16`：latent 通道数为 16
- `downsample=8`：空间下采样倍数为 8x
- `ch=128`, `ch_mult=[1,2,4,4]`, `num_res_blocks=2`：与 FLUX.1 VAE 参数完全一致
- `scale_factor=0.3611`, `shift_factor=0.1159`：使用简单的缩放和偏移因子进行 latent 归一化

**2. Encoder 结构（第122-193行）**

Encoder 由以下模块组成：
- `conv_in`：3×3 卷积，将 3 通道输入映射到 `ch=128` 通道
- 4 个下采样层级（由 `ch_mult=[1,2,4,4]` 控制），每个层级包含：
  - 2 个 `ResnetBlock`（GroupNorm → Swish → Conv3x3 → GroupNorm → Swish → Conv3x3 + shortcut）
  - `Downsample`（stride=2 的 3×3 卷积），最后一个层级除外
- `mid` 中间层：`ResnetBlock` → `AttnBlock` → `ResnetBlock`
- `norm_out` → Swish → `conv_out`：输出 `2*z_channels=32` 通道（均值和对数方差）

**3. Decoder 结构（第196-272行）**

Decoder 是 Encoder 的镜像结构：
- `conv_in`：将 `z_channels=16` 通道映射到 `block_in`
- `mid` 中间层：`ResnetBlock` → `AttnBlock` → `ResnetBlock`
- 4 个上采样层级（反序），每个层级包含：
  - 3 个 `ResnetBlock`（`num_res_blocks + 1`）
  - `Upsample`（最近邻插值 + 3×3 卷积），第一个层级除外
- `norm_out` → Swish → `conv_out`：输出 3 通道 RGB 图像

**4. DiagonalGaussian 采样（第275-287行）**

```python
class DiagonalGaussian(nn.Module):
    def forward(self, z: Tensor) -> Tensor:
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(mean)
        else:
            return mean
```

使用标准的重参数化技巧（reparameterization trick）进行采样。

**5. Encode/Decode 方法（第315-322行）**

```python
def encode(self, x: Tensor) -> Tensor:
    z = self.reg(self.encoder(x))
    z = self.scale_factor * (z - self.shift_factor)
    return z

def decode(self, z: Tensor) -> Tensor:
    z = z / self.scale_factor + self.shift_factor
    return self.decoder(z)
```

使用简单的线性缩放进行 latent 归一化，这与 FLUX.1 完全一致。

### 与 FLUX.1 和 FLUX.2 VAE 的对比

| 特征 | Bagel VAE | FLUX.1 VAE | FLUX.2 VAE |
|------|-----------|-----------|-----------|
| 基础结构 | 卷积 Encoder-Decoder | 卷积 Encoder-Decoder | 卷积 Encoder-Decoder |
| z_channels | **16** | **16** | 32 |
| 有效下采样倍数 | **8x** | **8x** | 16x（8x + 2x Patch） |
| Latent 通道数 | **16** | **16** | 128（32×2×2） |
| Latent 归一化 | **scale_factor + shift_factor** | **scale_factor + shift_factor** | BatchNorm |
| scale_factor | **0.3611** | **0.3611** | 无 |
| shift_factor | **0.1159** | **0.1159** | 无 |
| ch_mult | **[1,2,4,4]** | **[1,2,4,4]** | [1,2,4,4] |
| Patch 重排 | **无** | **无** | 有（ps=[2,2]） |
| quant_conv | **无** | 有 | 有 |

**结论**：Bagel 的 VAE 与 FLUX.1 的 VAE 结构参数完全一致，唯一的微小差异是 Bagel 没有 `quant_conv`/`post_quant_conv` 层（FLUX.1 的部分实现中也可省略），在核心架构上完全是 FLUX.1 类型的 VAE。

---

## 问题2：是否使用了flow_matching的DIT模型？单流还是双流？

### 结论：Bagel 使用了 flow matching 进行图像生成，但 **没有使用传统的独立 DIT 模型（也不是单流 DIT 或双流 MMDIT）**。它的创新之处在于：**直接使用 LLM（Qwen2 大语言模型）作为 flow matching 的去噪网络**。

### 代码分析依据

**1. Flow Matching 训练（`modeling/bagel/bagel.py` 第181-228行）**

在 `Bagel.forward()` 方法中可以清晰看到 flow matching 的训练过程：

```python
# 1. 从 VAE 编码获得 clean latent
packed_latent_clean = torch.cat(packed_latent, dim=0)

# 2. 生成随机噪声
noise = torch.randn_like(packed_latent_clean)

# 3. 按照 flow matching 公式进行插值
packed_timesteps = torch.sigmoid(packed_timesteps)
packed_timesteps = self.timestep_shift * packed_timesteps / (1 + (self.timestep_shift - 1) * packed_timesteps)
packed_latent = (1 - packed_timesteps[:, None]) * packed_latent_clean + packed_timesteps[:, None] * noise

# 4. 将 noisy latent 通过 vae2llm 线性层投影 + timestep embedding + position embedding
packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + latent_token_pos_emb

# 5. 将处理后的 latent token 放入 packed_sequence 中对应位置
packed_sequence[packed_vae_token_indexes] = packed_latent

# 6. 送入 LLM（Qwen2）获得 last_hidden_state
last_hidden_state = self.language_model(
    packed_sequence=packed_sequence,
    sample_lens=sample_lens,
    attention_mask=attention_mask,
    packed_position_ids=packed_position_ids,
    **extra_inputs,
)

# 7. 通过 llm2vae 线性层得到速度预测
packed_mse_preds = self.llm2vae(last_hidden_state[mse_loss_indexes])

# 8. 计算 flow matching 的 velocity loss
target = noise - packed_latent_clean  # v_t = dx_t/dt = x_1 - x_0
mse = (packed_mse_preds - target[has_mse]) ** 2
```

**2. Flow Matching 推理（`modeling/bagel/bagel.py` 第643-754行）**

在 `generate_image()` 方法中使用欧拉步进进行去噪：

```python
def generate_image(self, ...):
    x_t = packed_init_noises  # 从纯噪声开始

    timesteps = torch.linspace(1, 0, num_timesteps, device=x_t.device)
    timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
    dts = timesteps[:-1] - timesteps[1:]
    timesteps = timesteps[:-1]

    for i, t in enumerate(timesteps):
        v_t = self._forward_flow(x_t=x_t, timestep=timestep, ...)
        x_t = x_t - v_t * dts[i]  # 欧拉步进：velocity 指向 data → noise

    return unpacked_latent
```

**3. 去噪网络就是 LLM（`_forward_flow` 方法，第756-907行）**

```python
def _forward_flow(self, x_t, timestep, ...):
    # 文本 embedding
    packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
    packed_sequence[packed_text_indexes] = packed_text_embedding

    # noisy latent embedding（投影 + timestep + position）
    x_t = self.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed
    packed_sequence[packed_vae_token_indexes] = x_t

    # 送入 LLM 进行 forward
    output = self.language_model.forward_inference(
        packed_query_sequence=packed_sequence,
        ...
    )

    # 从 LLM 输出映射回 latent 空间
    v_t = self.llm2vae(output.packed_query_sequence)
    v_t = v_t[packed_vae_token_indexes]
    return v_t
```

**4. LLM 架构：不是传统 DIT**

Bagel 的 `language_model` 是一个修改过的 Qwen2（`Qwen2ForCausalLM`），在 `qwen2_navit.py` 中定义。它有三种 Decoder Layer 变体：

- **`Qwen2DecoderLayer`**：标准 Transformer Decoder Layer（单路径）
- **`Qwen2MoEDecoderLayer`**：Attention 共享但 MLP 分路的 MoE 变体
- **`Qwen2MoTDecoderLayer`**（默认使用）：**Attention 和 MLP 都分路的 Mixture-of-Transformer (MoT) 变体**

在 MoT 变体中，理解任务（und）和生成任务（gen）使用**不同的参数**：

```python
class Qwen2MoTDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        # 理解路径（und）
        self.self_attn = PackedAttentionMoT(config, layer_idx)  # 包含 q/k/v/o_proj + q/k/v/o_proj_moe_gen
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(...)
        self.post_attention_layernorm = Qwen2RMSNorm(...)

        # 生成路径（gen）
        self.mlp_moe_gen = Qwen2MLP(config)
        self.input_layernorm_moe_gen = Qwen2RMSNorm(...)
        self.post_attention_layernorm_moe_gen = Qwen2RMSNorm(...)
```

`PackedAttentionMoT` 中，理解 token 和生成 token 使用**不同的 Q/K/V/O 投影**：

```python
# 理解 token 使用标准投影
packed_query_states[packed_und_token_indexes] = self.q_proj(packed_sequence_und)
# 生成 token 使用 MoE 投影
packed_query_states[packed_gen_token_indexes] = self.q_proj_moe_gen(packed_sequence_gen)
```

但在 Attention 计算时，**所有 token 共享同一个 attention 空间**（联合注意力），只是 QKV 投影和 MLP 参数不同。

### 总结

| 特征 | 说明 |
|------|------|
| 是否 Flow Matching | ✅ 是，使用 rectified flow 的欧拉步进 |
| 去噪网络类型 | **LLM（Qwen2 + MoT）**，不是独立 DIT |
| 是否单流 DIT | ❌ 不是传统单流 DIT |
| 是否双流 MMDIT | ❌ 不是传统双流 MMDIT |
| 架构本质 | Causal LLM 中文本 token 和 VAE latent token 混合在同一序列中联合处理 |
| 理解/生成参数分离 | ✅ MoT 架构，Attention 投影和 MLP 各有理解和生成两套参数 |
| 注意力模式 | 生成时使用非因果（bidirectional）注意力；理解时使用因果（causal）注意力 |

---

## 问题3：具体网络结构和子网络

### Bagel 模型包含以下子网络结构：

#### 子网络 1：SigLIP Vision Transformer（ViT）— 图像理解编码器

**文件**：`modeling/bagel/siglip_navit.py`

- **模型类**：`SiglipVisionModel` → `SiglipVisionTransformer`
- **功能**：将输入图像编码为**语义级别**的视觉特征，用于图像理解任务
- **结构**：
  - `SiglipVisionEmbeddings`：Patch 嵌入（Conv2d 或 Linear）
  - `SiglipEncoder`：多层 `SiglipEncoderLayer`（LayerNorm → Flash Attention → MLP）
  - `post_layernorm`：最终 LayerNorm
  - 支持 2D RoPE 位置编码（`RotaryEmbedding2D`）
  - 使用 `flash_attn_varlen_func` 进行可变长度 Flash Attention
- **参数**（默认配置）：
  - `hidden_size`：ViT 隐藏维度
  - `patch_size=14`：每个 patch 14×14 像素
  - `max_num_patch_per_side=70`：最大 70 个 patch/边

#### 子网络 2：MLP Connector — ViT 到 LLM 的桥接

**文件**：`modeling/bagel/modeling_utils.py` 第113-124行

```python
class MLPconnector(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_act):
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.activation_fn = ACT2FN[hidden_act]  # gelu_pytorch_tanh
        self.fc2 = nn.Linear(out_dim, out_dim)
```

- **功能**：将 ViT 输出维度映射到 LLM 隐藏维度
- **结构**：两层 MLP（Linear → GELU → Linear）

#### 子网络 3：Qwen2 LLM（MoT 变体）— 核心 Transformer

**文件**：`modeling/bagel/qwen2_navit.py`

- **模型类**：`Qwen2ForCausalLM` → `Qwen2Model`
- **功能**：
  - 作为语言模型处理文本理解任务（CE loss）
  - 作为去噪网络处理图像生成任务（MSE flow matching loss）
- **结构**：
  - `embed_tokens`：词嵌入层
  - `layers`：N 层 `Qwen2MoTDecoderLayer`（默认）
    - 每层包含：
      - `PackedAttentionMoT`：注意力层，理解/生成使用不同的 Q/K/V/O 投影
      - `Qwen2MLP` × 2：理解 MLP 和生成 MLP
      - `Qwen2RMSNorm` × 4：理解和生成各两个 LayerNorm
  - `norm` / `norm_moe_gen`：最终 RMSNorm
  - `rotary_emb`：RoPE 位置编码
  - `lm_head`：语言建模输出层
- **特色**：支持三种 Layer 模式：
  - `Qwen2DecoderLayer`：标准层
  - `Qwen2MoEDecoderLayer`：仅 MLP 分路
  - `Qwen2MoTDecoderLayer`：Attention + MLP 全部分路（**默认使用**）
- **参数**：7B 激活参数 / 14B 总参数

#### 子网络 4：VAE Encoder — 像素级图像编码器

**文件**：`modeling/autoencoder.py`

- **功能**：将 RGB 图像编码为低维 latent 表示
- **结构**：
  ```
  输入 [B, 3, H, W]
    → conv_in (3→128)
    → 4个下采样层级 (ResnetBlock×2 + Downsample)
    → mid (ResnetBlock → AttnBlock → ResnetBlock)
    → norm_out → Swish → conv_out (→32通道)
    → DiagonalGaussian (32→16通道, 取均值或重参数化采样)
    → scale_factor * (z - shift_factor)
  输出 latent [B, 16, H/8, W/8]
  ```

#### 子网络 5：VAE Decoder — 图像重建解码器

**文件**：`modeling/autoencoder.py`

- **功能**：将 latent 表示解码回 RGB 图像
- **结构**：
  ```
  输入 latent [B, 16, H/8, W/8]
    → z / scale_factor + shift_factor
    → conv_in (16→block_in)
    → mid (ResnetBlock → AttnBlock → ResnetBlock)
    → 4个上采样层级 (ResnetBlock×3 + Upsample)
    → norm_out → Swish → conv_out (→3通道)
  输出 [B, 3, H, W]
  ```

#### 子网络 6：TimestepEmbedder — 时间步嵌入

**文件**：`modeling/bagel/modeling_utils.py` 第74-110行

```python
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
```

- **功能**：将标量时间步编码为高维向量
- **方法**：先用正弦余弦频率编码，再通过两层 MLP

#### 子网络 7：PositionEmbedding — 2D 位置编码

**文件**：`modeling/bagel/modeling_utils.py` 第127-144行

```python
class PositionEmbedding(nn.Module):
    def __init__(self, max_num_patch_per_side, hidden_size):
        self.pos_embed = nn.Parameter(
            torch.zeros(max_num_patch_per_side ** 2, hidden_size),
            requires_grad=False
        )
```

- **功能**：为 latent token 和 ViT token 提供 2D 正弦余弦位置编码
- **特点**：参数冻结（`requires_grad=False`），预计算的 sincos 编码

#### 子网络 8：vae2llm / llm2vae — 线性投影层

**文件**：`modeling/bagel/bagel.py` 第76-77行

```python
self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)
self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)
```

- `vae2llm`：将 VAE latent patch token（`patch_size² × z_channels = 2² × 16 = 64` 维）投影到 LLM 隐藏维度
- `llm2vae`：将 LLM 隐藏维度投影回 latent patch token 空间
- `llm2vae` 初始化为零权重和零偏置（第98-99行），确保训练初期不影响 LLM

---

## 问题4：模型网络结构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                     BAGEL 完整模型架构                                │
│                                                                     │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│  │  输入图像     │    │  文本 Prompt      │    │  输入图像（编辑） │   │
│  │  (理解/编辑)  │    │  (字符串)         │    │  (编辑模式才有)   │   │
│  └──────┬───────┘    └────────┬─────────┘    └────────┬─────────┘   │
│         │                     │                       │             │
│    ┌────┴────┐                │                       │             │
│    ▼         ▼                │                       ▼             │
│ ┌──────┐ ┌──────┐            │               ┌──────────────┐      │
│ │ViT   │ │ VAE  │            │               │  VAE Encoder │      │
│ │编码器│ │Encoder│            │               │  (像素级)    │      │
│ │(语义)│ │(像素)│            │               └──────┬───────┘      │
│ └──┬───┘ └──┬───┘            │                      │              │
│    │        │                │                      ▼              │
│    ▼        │                │               patchify + vae2llm   │
│ ┌──────┐   │                │               + timestep_embed     │
│ │ MLP  │   │                │               + position_embed     │
│ │连接器│   │                │                      │              │
│ └──┬───┘   │                │                      │              │
│    │       │                │                      │              │
│    ▼       ▼                ▼                      ▼              │
│  vit_tokens  vae_tokens  text_tokens         vae_cond_tokens     │
│  + pos_emb   + t_emb     (embed_tokens)      (编辑条件)          │
│  + pos_emb   + pos_emb                                           │
│    │         │              │                      │              │
│    └─────────┴──────────────┴──────────────────────┘              │
│                             │                                     │
│                    packed_sequence                                │
│                    (所有token拼接在统一序列中)                       │
│                             │                                     │
│                             ▼                                     │
│    ┌────────────────────────────────────────────────────┐         │
│    │          Qwen2 LLM (MoT 变体)                      │         │
│    │                                                    │         │
│    │  ┌──────────────────────────────────────────────┐  │         │
│    │  │  Qwen2MoTDecoderLayer × N                    │  │         │
│    │  │                                              │  │         │
│    │  │  Input LayerNorm (und) / LayerNorm (gen)     │  │         │
│    │  │          │                                   │  │         │
│    │  │  PackedAttentionMoT:                         │  │         │
│    │  │    und tokens → q/k/v_proj → QKV             │  │         │
│    │  │    gen tokens → q/k/v_proj_moe_gen → QKV     │  │         │
│    │  │    QK Norm (und/gen 分别 normalize)           │  │         │
│    │  │    RoPE 位置编码                               │  │         │
│    │  │    联合 Flash Attention (所有token一起)         │  │         │
│    │  │    und tokens → o_proj                       │  │         │
│    │  │    gen tokens → o_proj_moe_gen               │  │         │
│    │  │          │                                   │  │         │
│    │  │  + Residual                                  │  │         │
│    │  │          │                                   │  │         │
│    │  │  Post LayerNorm (und) / LayerNorm (gen)      │  │         │
│    │  │    und tokens → MLP (und)                    │  │         │
│    │  │    gen tokens → MLP_moe_gen                  │  │         │
│    │  │          │                                   │  │         │
│    │  │  + Residual                                  │  │         │
│    │  └──────────────────────────────────────────────┘  │         │
│    │                                                    │         │
│    │  Final RMSNorm (und) / RMSNorm_moe_gen            │         │
│    └────────────────────────┬───────────────────────────┘         │
│                             │                                     │
│              last_hidden_state                                    │
│                             │                                     │
│              ┌──────────────┼──────────────┐                      │
│              ▼              │              ▼                      │
│    ┌──────────────┐         │    ┌──────────────┐                 │
│    │   lm_head    │         │    │   llm2vae    │                 │
│    │  (理解输出)   │         │    │  (生成输出)   │                 │
│    │ → text logits│         │    │ → v_pred     │                 │
│    └──────┬───────┘         │    └──────┬───────┘                 │
│           │                 │           │                         │
│           ▼                 │           ▼                         │
│     CE Loss                │    MSE Loss (flow matching)         │
│     (理解任务)              │    (生成任务)                        │
│                             │           │                         │
│                             │    欧拉步进去噪 (推理时)              │
│                             │           │                         │
│                             │    unpatchify                       │
│                             │           │                         │
│                             │           ▼                         │
│                             │    ┌──────────────┐                 │
│                             │    │ VAE Decoder  │                 │
│                             │    │ (图像重建)    │                 │
│                             │    └──────┬───────┘                 │
│                             │           │                         │
│                             │           ▼                         │
│                             │    输出图像 [3, H, W]                │
│                             │                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 子网络连接关系简图

```
输入图像 ──→ ViT ──→ MLP Connector ──→ ┐
                                        ├──→ packed_sequence ──→ Qwen2 LLM (MoT) ──→ lm_head ──→ 文本输出
输入图像 ──→ VAE Encoder ──→ vae2llm ──→┤                                         └──→ llm2vae ──→ v_pred
                                        │                                                           │
文本     ──→ embed_tokens ─────────────→┘                                         欧拉去噪 ←────────┘
                                                                                      │
时间步   ──→ TimestepEmbedder ──→ (加到 vae token 上)                                  ▼
                                                                               VAE Decoder ──→ 输出图像
位置     ──→ PositionEmbedding ──→ (加到 vae/vit token 上)
```

---

## 问题5：是否支持文生图和图像编辑

### 结论

| 功能 | 是否支持 | 代码依据 |
|------|---------|---------|
| **文生图（Text-to-Image）** | ✅ 支持 | `inferencer.py` 中只输入 text，不输入 image |
| **图像+文本提示编辑** | ✅ 支持 | `inferencer.py` 中同时输入 image 和 text |
| **图像理解（VLM）** | ✅ 支持 | `inferencer.py` 中 `understanding_output=True` |

### 代码依据

**1. 统一的推理接口（`inferencer.py` 第288-313行）**

```python
def __call__(self, image=None, text=None, **kargs):
    input_list = []
    if image is not None:
        input_list.append(image)
    if text is not None:
        input_list.append(text)
    output_list = self.interleave_inference(input_list, **kargs)
```

**2. 文生图（`inferencer.py` 第207-286行）**

当 `understanding_output=False` 且只有 text 输入时：
```python
# 只更新文本上下文
gen_context = self.update_context_text(input_term, gen_context)
# 然后生成图像
img = self.gen_image(image_shapes, gen_context, ...)
```

**3. 图像编辑（同一方法）**

当同时输入 image 和 text 时：
```python
# 先用 VAE 编码图像（像素级条件）
gen_context = self.update_context_image(input_term, gen_context, vae=True)
# 再用文本更新上下文
gen_context = self.update_context_text(input_term, gen_context)
# 最后生成编辑后的图像
img = self.gen_image(image_shapes, gen_context, ...)
```

编辑时图像同时通过 VAE Encoder 和 ViT 编码（`vae=not understanding_output`），提供像素级和语义级的双重条件。

**4. 图像理解（`understanding_output=True`）**

```python
if understanding_output:
    gen_text = self.gen_text(gen_context, ...)
    output_list.append(gen_text)
```

此时图像只通过 ViT 编码（不通过 VAE），LLM 生成文本回答。

**5. Gradio UI 验证（`app.py`）**

`app.py` 中定义了三个 Tab，进一步确认了三种功能：
- `📝 Text to Image`：文生图
- `🖌️ Image Edit`：图像编辑
- `🖼️ Image Understanding`：图像理解

---

## 问题6：文生图流程图

```
输入数据:
  ├── 文本 Prompt (字符串, 如 "A cat sitting on a windowsill")
  ├── 图像尺寸 image_shapes (如 (1024, 1024))
  └── 采样参数 (num_timesteps=50, timestep_shift=3.0, cfg_text_scale=4.0, seed)

═══════════════════════════════════════════════════════════════

步骤1: 初始化上下文 (init_gen_context)
  ┌─────────────────────┐
  │ 创建空的 KV Cache    │
  │ kv_lens = [0]        │
  │ ropes = [0]          │
  │ past_key_values =    │
  │   NaiveCache(N_layers)│
  └──────────┬──────────┘
             │
  同时创建 cfg_text_context 和 cfg_img_context (深拷贝)

═══════════════════════════════════════════════════════════════

步骤2: （可选）系统提示 / Think 模式
  ┌─────────────────────────────────┐
  │ system_prompt = "You should     │
  │ first think about the planning  │
  │ process..."                     │
  └──────────┬──────────────────────┘
             │
             ▼
  update_context_text(system_prompt, gen_context)
    → Tokenize → embed_tokens → LLM forward
    → 更新 KV Cache

═══════════════════════════════════════════════════════════════

步骤3: 文本编码 (update_context_text)
  ┌─────────────────────┐
  │ 文本 Prompt          │
  │ "A cat sitting..."  │
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────────────────────────┐
  │ prepare_prompts()                       │
  │   1. tokenizer.encode(prompt)           │
  │   2. 加 BOS/EOS token                   │
  │   3. 计算 position_ids                  │
  └──────────┬──────────────────────────────┘
             │
             ▼
  ┌─────────────────────────────────────────┐
  │ forward_cache_update_text()             │
  │                                         │
  │   1. embed_tokens(text_ids) → text_emb │
  │   2. LLM.forward_inference(            │
  │        packed_query_sequence=text_emb,  │
  │        update_past_key_values=True,     │
  │        is_causal=True                   │
  │      )                                  │
  │   3. 更新 KV Cache                     │
  └──────────┬──────────────────────────────┘
             │
  gen_context 的 KV Cache 已包含文本信息
  cfg_text_context = deepcopy(gen_context 在文本之前)  ← 无文本条件
  cfg_img_context 更新文本

═══════════════════════════════════════════════════════════════

步骤4: （可选）Think / CoT 生成
  ┌─────────────────────────────────────────┐
  │ gen_text(gen_context)                   │
  │   1. prepare_start_tokens()             │
  │   2. generate_text() 自回归生成          │
  │      while step < max_length:           │
  │        embed → LLM forward → lm_head    │
  │        → argmax/sample → next token     │
  │   3. 输出思维链文本                      │
  └──────────┬──────────────────────────────┘
             │
  gen_context = update_context_text(think_text, gen_context)

═══════════════════════════════════════════════════════════════

步骤5: 准备 latent 生成空间 (prepare_vae_latent)
  ┌─────────────────────────────────────────┐
  │ prepare_vae_latent()                    │
  │                                         │
  │ 1. 计算 latent 尺寸:                    │
  │    h = H / latent_downsample            │
  │    w = W / latent_downsample            │
  │    (latent_downsample = 8 * 2 = 16)     │
  │    → h×w 个 latent tokens               │
  │                                         │
  │ 2. 生成初始噪声:                         │
  │    packed_init_noises = torch.randn(    │
  │      h*w, 16*2*2=64                     │
  │    )                                    │
  │                                         │
  │ 3. 计算 2D position_ids                 │
  │                                         │
  │ 4. 添加 start_of_image / end_of_image  │
  │    特殊 token 包裹 latent tokens         │
  └──────────┬──────────────────────────────┘

═══════════════════════════════════════════════════════════════

步骤6: Flow Matching 去噪循环 (generate_image)
  timesteps = linspace(1, 0, 50) 经 timestep_shift 调整
  x_t = packed_init_noises  (纯噪声)

  for t in timesteps:  (从 t=1 到 t≈0)
    ┌─────────────────────────────────────────┐
    │ _forward_flow()                         │
    │                                         │
    │ 1. text_emb = embed_tokens(text_ids)   │
    │    packed_sequence[text_idx] = text_emb │
    │                                         │
    │ 2. timestep_embed = TimestepEmbedder(t) │
    │    pos_embed = PositionEmbedding(pos)   │
    │    x_t_proj = vae2llm(x_t)             │
    │           + timestep_embed              │
    │           + pos_embed                   │
    │    packed_sequence[vae_idx] = x_t_proj  │
    │                                         │
    │ 3. LLM.forward_inference(              │
    │      packed_query_sequence,             │
    │      past_key_values=KV_Cache,         │
    │      update_past_key_values=False,      │  ← 不更新 KV Cache
    │      is_causal=False,                   │  ← 非因果注意力
    │      mode="gen"                         │  ← 使用生成路径
    │    )                                    │
    │                                         │
    │ 4. v_t = llm2vae(output)[vae_idx]      │
    │    → 速度场预测 [h*w, 64]               │
    └──────────────────┬──────────────────────┘
                       │
    ┌──────────────────┴──────────────────────┐
    │ CFG (Classifier-Free Guidance)          │
    │                                         │
    │ if cfg_text_scale > 1.0:               │
    │   cfg_text_v_t = LLM.forward(          │
    │     same x_t, 但用 cfg_text KV_Cache   │
    │     (无文本条件的 KV Cache)              │
    │   )                                    │
    │   v_t = cfg_text_v_t +                 │
    │     cfg_text_scale * (v_t - cfg_text_v_t)│
    │                                         │
    │ if cfg_img_scale > 1.0:                │
    │   cfg_img_v_t = LLM.forward(           │
    │     same x_t, 但用 cfg_img KV_Cache    │
    │     (无图像条件的 KV Cache)              │
    │   )                                    │
    │   v_t 进一步 CFG 调整                   │
    │                                         │
    │ CFG-Renorm: 归一化防止过强 guidance       │
    └──────────────────┬──────────────────────┘
                       │
    x_t = x_t - v_t * dt    (欧拉步进, velocity 指向 data→noise)

  ═══════ 重复 num_timesteps 步 ═══════

═══════════════════════════════════════════════════════════════

步骤7: Latent → 图像 (decode_image)
  ┌─────────────────────────────────────────┐
  │ decode_image(unpacked_latent)            │
  │                                         │
  │ 1. unpatchify:                          │
  │    [h*w, 64] → [1, 16, h*2, w*2]       │
  │    (h*2, w*2 因为 latent_patch_size=2)  │
  │                                         │
  │ 2. VAE Decoder:                         │
  │    vae_model.decode(latent)             │
  │    → z / scale_factor + shift_factor    │
  │    → conv_in → mid → up (4级上采样)      │
  │    → norm_out → swish → conv_out        │
  │                                         │
  │ 3. 后处理:                               │
  │    image = (image * 0.5 + 0.5).clamp(0,1)│
  │    → PIL Image                          │
  │                                         │
  │ 输出: RGB 图像 [H, W, 3]                │
  └─────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════

最终输出:
  ├── 由 VAE Decoder 子网络输出的 RGB 图像
  └── （可选）Think 文本（由 lm_head 子网络输出）
```

---

## 问题7：图像+文本提示图像编辑流程图

```
输入数据:
  ├── 输入图像 (PIL Image, 如 women.jpg)
  ├── 编辑指令 Prompt (如 "She boards a modern subway, wearing the same clothes")
  └── 采样参数 (num_timesteps=50, timestep_shift=3.0,
  │   cfg_text_scale=4.0, cfg_img_scale=2.0, seed)

═══════════════════════════════════════════════════════════════

步骤1: 初始化上下文 (与文生图相同)
  gen_context, cfg_text_context, cfg_img_context = init

═══════════════════════════════════════════════════════════════

步骤2: （可选）系统提示 + Think
  与文生图相同

═══════════════════════════════════════════════════════════════

步骤3: 输入图像编码 (update_context_image)

  ┌──────────────────────────────────────────┐
  │ 输入图像                                  │
  │ (PIL Image, resize 到合适尺寸)            │
  └──────────┬───────────────────────────────┘
             │
      ┌──────┴──────┐
      ▼             ▼
  ┌────────┐   ┌────────┐
  │ VAE    │   │ ViT    │
  │ 编码   │   │ 编码   │
  └──┬─────┘   └──┬─────┘
     │            │
     ▼            ▼

【步骤3a: VAE 编码 (update_context_image, vae=True)】

  ┌─────────────────────────────────────────────────┐
  │ prepare_vae_images()                            │
  │                                                 │
  │ 1. vae_transform(image) → image_tensor          │
  │ 2. 计算 patchified_vae_latent_shapes:           │
  │    h = H / latent_downsample                    │
  │    w = W / latent_downsample                    │
  │ 3. 计算 vae_position_ids (2D sincos)            │
  │ 4. 添加 start_of_image / end_of_image tokens    │
  └──────────┬──────────────────────────────────────┘
             │
             ▼
  ┌─────────────────────────────────────────────────┐
  │ forward_cache_update_vae()                      │
  │                                                 │
  │ 1. text_emb = embed_tokens(special_tokens)     │
  │                                                 │
  │ 2. padded_latent = vae_model.encode(images)    │
  │    → VAE Encoder: 图像 → latent [16, H/8, W/8] │
  │                                                 │
  │ 3. Patchify latent:                            │
  │    [16, h*2, w*2] → [h*w, 64]                 │
  │    (patch_size=2, 每个patch = 2×2×16 = 64维)    │
  │                                                 │
  │ 4. packed_latent = vae2llm(latent)             │
  │    + TimestepEmbedder(timestep=0)  ← 干净图像t=0│
  │    + PositionEmbedding(pos_ids)                │
  │                                                 │
  │ 5. packed_sequence[vae_idx] = packed_latent    │
  │    packed_sequence[text_idx] = text_emb        │
  │                                                 │
  │ 6. LLM.forward_inference(                      │
  │      packed_sequence,                          │
  │      mode="gen",  ← 使用生成路径                 │
  │      update_past_key_values=True,               │
  │      is_causal=False                            │
  │    )                                            │
  │    → 更新 KV Cache                              │
  └──────────┬──────────────────────────────────────┘

【步骤3b: ViT 编码 (update_context_image, vit=True)】

  ┌─────────────────────────────────────────────────┐
  │ prepare_vit_images()                            │
  │                                                 │
  │ 1. vit_transform(image) → image_tensor          │
  │ 2. patchify(image, patch_size=14)               │
  │    → [num_patches, 14*14*3] 个 patch tokens     │
  │ 3. 计算 vit_position_ids                        │
  │ 4. 添加 start_of_image / end_of_image tokens    │
  └──────────┬──────────────────────────────────────┘
             │
             ▼
  ┌─────────────────────────────────────────────────┐
  │ forward_cache_update_vit()                      │
  │                                                 │
  │ 1. text_emb = embed_tokens(special_tokens)     │
  │                                                 │
  │ 2. vit_token_embed = vit_model(                │
  │      packed_pixel_values,                       │
  │      packed_flattened_position_ids,             │
  │      cu_seqlens, max_seqlen                    │
  │    )                                            │
  │    → SigLIP ViT: patch tokens → 语义特征        │
  │                                                 │
  │ 3. vit_token_embed = MLP_connector(vit_embed)  │
  │    → 映射到 LLM 隐藏维度                        │
  │                                                 │
  │ 4. vit_token_embed += PositionEmbedding(pos)   │
  │                                                 │
  │ 5. packed_sequence[vit_idx] = vit_token_embed  │
  │    packed_sequence[text_idx] = text_emb        │
  │                                                 │
  │ 6. LLM.forward_inference(                      │
  │      packed_sequence,                          │
  │      mode="und",  ← 使用理解路径                 │
  │      update_past_key_values=True,               │
  │      is_causal=False                            │
  │    )                                            │
  │    → 更新 KV Cache                              │
  └──────────┬──────────────────────────────────────┘

  此时 KV Cache 中已包含:
    gen_context: 图像的 VAE + ViT 编码信息

═══════════════════════════════════════════════════════════════

步骤4: 文本编辑指令编码 (update_context_text)

  ┌─────────────────────────────────────────────────┐
  │ "She boards a modern subway..."                 │
  └──────────┬──────────────────────────────────────┘
             │
             ▼
  cfg_text_context = deepcopy(gen_context)  ← 此时有图像但无文本
  gen_context = update_context_text(prompt, gen_context)
  cfg_img_context = update_context_text(prompt, cfg_img_context)  ← 有文本但无图像

═══════════════════════════════════════════════════════════════

步骤5: Flow Matching 去噪循环 (generate_image)

  与文生图步骤6基本相同，但有以下关键差异：

  1. KV Cache 中已包含输入图像的 VAE 和 ViT 编码信息
     → 每步去噪时，LLM 可以通过注意力机制"看到"输入图像

  2. 双 CFG 引导：
     - cfg_text_scale: 基于 cfg_text_context (有图像无文本)
       → 控制文本编辑指令的遵循程度
     - cfg_img_scale: 基于 cfg_img_context (有文本无图像)
       → 控制输入图像细节的保留程度

  3. CFG 计算公式:
     v_t_text = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
     v_t_final = cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)

  4. CFG-Renorm 防止过强 guidance:
     - "global": 全局归一化 (文生图默认)
     - "text_channel": 只对文本条件按通道归一化 (编辑推荐)
     - "channel": 按通道归一化

═══════════════════════════════════════════════════════════════

步骤6: Latent → 图像 (decode_image)

  与文生图步骤7完全相同：
  unpacked_latent → unpatchify → VAE Decoder → RGB 图像

═══════════════════════════════════════════════════════════════

整体数据流对比（文生图 vs 图像编辑）:

         文生图                              图像编辑
  ────────────────              ────────────────────────────
  text → KV Cache               image → VAE Enc → KV Cache
                                image → ViT → MLP → KV Cache
                                text → KV Cache
       ↓                              ↓
  noise → LLM → v_pred          noise → LLM (可看到图像) → v_pred
       ↓                              ↓
  欧拉去噪 × N步                 欧拉去噪 × N步 (双CFG)
       ↓                              ↓
  VAE Decoder → 图像              VAE Decoder → 编辑后图像

═══════════════════════════════════════════════════════════════

最终输出:
  ├── 由 VAE Decoder 子网络输出的编辑后 RGB 图像
  └── （可选）Think 文本（由 lm_head 子网络输出）

═══════════════════════════════════════════════════════════════

编辑模式的注意力可见性机制:

  KV Cache 中的 token 序列布局 (gen_context):
  [... system_prompt ... | vae_image_tokens | vit_image_tokens | text_prompt ...]
       (因果注意力)         (非因果, gen)       (非因果, und)      (因果注意力)

  去噪时新生成的 latent tokens:
  [start_of_image | noisy_latent_tokens | end_of_image]
       全部使用非因果注意力，可以看到 KV Cache 中的所有 token
       → 可以同时利用：
         1. VAE 编码的像素级信息（保留图像细节）
         2. ViT 编码的语义级信息（理解图像内容）
         3. 文本编辑指令（指导编辑方向）
```

---

## 问题8：相比FLUX.2模型的创新点

### 基于代码实现分析，Bagel 相比 FLUX.2 具有以下创新点和改进：

#### 1. 统一的理解+生成+编辑模型（最核心创新）

FLUX.2 只支持图像**生成和编辑**，不具备图像理解能力。Bagel 在**一个模型**中同时支持三种能力：

- **图像理解（VLM）**：输入图像+文本问题 → 输出文本回答
- **文生图（T2I）**：输入文本 → 输出图像
- **图像编辑（Image Editing）**：输入图像+编辑指令 → 输出编辑后图像

```python
# inferencer.py - 统一推理接口
def __call__(self, image=None, text=None, **kargs):
    # image + text + understanding_output=True → 图像理解
    # text only → 文生图
    # image + text → 图像编辑
```

#### 2. LLM 即 DIT：用大语言模型替代独立 DIT

FLUX.2 使用**独立的 DIT 模型**（DoubleStreamBlock + SingleStreamBlock）作为去噪网络，文本编码器是单独的 Mistral/Qwen3。

Bagel 则完全不同：**LLM 本身就是去噪网络**。文本 token 和 VAE latent token 在同一个 Transformer 序列中联合处理，通过同一个注意力机制进行交互。

```python
# Bagel: LLM 同时处理文本和 latent
packed_sequence[packed_text_indexes] = text_embedding
packed_sequence[packed_vae_token_indexes] = latent_embedding
output = self.language_model(packed_sequence)  # 一次 forward 处理所有
v_t = self.llm2vae(output[vae_indexes])  # 从 LLM 输出中提取速度预测
```

```python
# FLUX.2: 独立 DIT + 独立文本编码器
ctx = text_encoder(prompt)      # 单独的文本编码
img = dit_model(x, ctx, t)      # 单独的 DIT 去噪
```

这意味着 Bagel 不需要独立的大型文本编码器（如 Mistral-24B 或 Qwen3-8B），LLM 自身承担了文本编码和条件注入的角色。

#### 3. Mixture-of-Transformer-Experts (MoT) 架构

FLUX.2 的 DIT 中所有 token 共享相同的参数。Bagel 引入了 MoT 架构，为理解和生成任务使用**不同的专家参数**：

```python
# Qwen2MoTDecoderLayer - 每层有两套独立参数
class Qwen2MoTDecoderLayer:
    # 理解路径 (und)
    self.self_attn.q_proj, k_proj, v_proj, o_proj
    self.mlp
    self.input_layernorm, self.post_attention_layernorm

    # 生成路径 (gen)
    self.self_attn.q_proj_moe_gen, k_proj_moe_gen, v_proj_moe_gen, o_proj_moe_gen
    self.mlp_moe_gen
    self.input_layernorm_moe_gen, self.post_attention_layernorm_moe_gen
```

这使得 7B 激活参数/14B 总参数的模型能够同时在理解和生成任务上达到高性能，不同任务激活不同的参数子集。

#### 4. 双编码器设计（ViT + VAE）

FLUX.2 在编辑模式下只使用 VAE 编码参考图像（像素级信息）。Bagel 同时使用**两个编码器**：

- **SigLIP ViT**：提取**语义级别**的视觉特征（用于理解图像内容、场景、物体关系）
- **VAE Encoder**：提取**像素级别**的 latent 表示（用于保留图像细节、颜色、纹理）

```python
# inferencer.py - 编辑模式同时使用两种编码
gen_context = self.update_context_image(input_term, gen_context, vae=True)  # VAE 编码
# 在 update_context_image 内部还会调用 ViT 编码
```

这种双编码器设计使模型能够同时理解图像语义和保留像素细节。

#### 5. 无需独立文本编码器

FLUX.2 需要一个庞大的独立文本编码器：
- **FLUX.2 [dev]**：Mistral-Small-3.2-24B（24B 参数！）
- **FLUX.2 [klein]**：Qwen3-4B/8B

Bagel 的 LLM 自身承担了文本编码的角色，不需要额外的文本编码器。这大幅减少了总参数量和推理时的内存占用。

#### 6. Next Group of Token Prediction 范式

FLUX.2 只有一种训练目标：flow matching 的速度预测（MSE loss）。

Bagel 在同一个模型中同时优化**两种损失**：

```python
# bagel.py forward() - 同时计算两种损失
# 1. 图像生成：Flow Matching MSE Loss
target = noise - packed_latent_clean  # v_t = x_1 - x_0
mse = (packed_mse_preds - target) ** 2

# 2. 文本生成：Cross-Entropy Loss
packed_ce_preds = self.language_model.lm_head(last_hidden_state[ce_loss_indexes])
ce = F.cross_entropy(packed_ce_preds, packed_label_ids, reduction="none")

return dict(mse=mse, ce=ce)
```

训练循环中通过权重平衡两种损失：
```python
# pretrain_unified_navit.py
loss = ce * training_args.ce_weight + mse * training_args.mse_weight
```

#### 7. 思维链（Think/CoT）支持

FLUX.2 没有思维链能力。Bagel 在图像生成和编辑之前可以先进行**推理规划**：

```python
# inferencer.py
GEN_THINK_SYSTEM_PROMPT = '''You should first think about the planning process 
in the mind and then generate the image. The planning process is enclosed 
within <think> </think> tags...'''

if think:
    gen_text = self.gen_text(gen_context, ...)  # 先生成思维链
    gen_context = self.update_context_text(gen_text, gen_context)  # 再据此生成图像
```

README 中显示，启用 CoT 后在 GenEval 上从 0.82 提升到 0.88，在 WISE 上从 0.52 提升到 0.70，效果显著。

#### 8. 双 CFG 支持（Text CFG + Image CFG）

FLUX.2 只支持单一的 guidance 参数。Bagel 支持**两个独立的 CFG 维度**：

```python
# 文本 CFG：控制文本条件的遵循程度
cfg_text_scale=4.0  # 1.0 表示不使用文本 CFG
# 图像 CFG：控制输入图像细节的保留程度（编辑模式）
cfg_img_scale=2.0   # 1.0 表示不使用图像 CFG
```

通过三个不同的 KV Cache 实现：
- `gen_context`：完整条件（有图像 + 有文本）
- `cfg_text_context`：无文本条件（有图像 + 无文本）
- `cfg_img_context`：无图像条件（无图像 + 有文本）

#### 9. CFG-Renorm 多种策略

FLUX.2 没有 CFG-Renorm 机制。Bagel 提供三种 CFG-Renorm 策略来防止 guidance 过强导致的伪影：

```python
# bagel.py _forward_flow()
if cfg_renorm_type == "global":
    norm_v_t = torch.norm(v_t)        # 全局归一化（文生图默认）
elif cfg_renorm_type == "channel":
    norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)  # 逐通道归一化
elif cfg_renorm_type == "text_channel":
    # 仅对文本 CFG 部分进行通道归一化（编辑推荐）
    ...
```

#### 10. Latent Patchify 机制

FLUX.2 的 latent token 是像素级的（128 通道，H/16 × W/16 空间分辨率）。Bagel 使用 **latent_patch_size=2** 将 VAE 的 latent 进一步分组：

```python
# bagel.py
self.latent_patch_size = config.latent_patch_size  # 2
self.patch_latent_dim = self.latent_patch_size ** 2 * self.latent_channel  # 2*2*16 = 64

# patchify: [16, h*2, w*2] → [h*w, 64]
latent = latent.reshape(self.latent_channel, h, p, w, p)
latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
```

这将每个 2×2 的 latent 空间块合并为一个 64 维 token，减少了 LLM 需要处理的 token 数量（减少 4 倍），有效降低了计算成本。

#### 11. KV Cache 机制用于上下文管理

FLUX.2 在编辑模式下将参考图 tokens 直接拼接到去噪序列中（或使用 KV Cache 加速）。Bagel 采用了更优雅的**分阶段 KV Cache 更新**机制：

```python
# 阶段 1：编码输入图像，更新 KV Cache
past_key_values = forward_cache_update_vae(vae_model, past_key_values, ...)
past_key_values = forward_cache_update_vit(past_key_values, ...)

# 阶段 2：编码文本，更新 KV Cache
past_key_values = forward_cache_update_text(past_key_values, ...)

# 阶段 3：去噪循环，不更新 KV Cache（只读）
output = language_model.forward_inference(
    ..., past_key_values=past_key_values,
    update_past_key_values=False,  # ← 不更新
)
```

这样在去噪循环的每一步都不需要重新计算条件信息的 KV，只需要计算新的 latent query 对已有 KV 的注意力。

#### 12. 条件 Dropout 训练策略

FLUX.2 的代码中没有明确的条件 dropout 机制。Bagel 在训练时对三种条件独立进行 dropout：

```python
# pretrain_unified_navit.py
text_cond_dropout_prob: float = 0.1    # 10% 概率丢弃文本条件
vae_cond_dropout_prob: float = 0.3     # 30% 概率丢弃 VAE 条件
vit_cond_dropout_prob: float = 0.3     # 30% 概率丢弃 ViT 条件
```

这使得模型可以在推理时灵活使用不同的 CFG 组合（text CFG、image CFG 或两者结合）。

#### 13. 冻结理解参数训练（freeze_und）

Bagel 支持在训练生成能力时**冻结理解路径的参数**：

```python
# Qwen2Config
freeze_und: bool = False  # 可选冻结理解参数

# Qwen2MoTDecoderLayer.forward_train()
if self.freeze_und:
    packed_sequence_[packed_und_token_indexes] = \
        packed_sequence_[packed_und_token_indexes].detach()
```

这允许在微调生成能力时保持已训练好的理解能力不退化。

#### 14. TaylorSeer 缓存加速推理

Bagel 集成了 TaylorSeer 技术来加速推理，通过 Taylor 展开近似中间层的输出，跳过部分计算：

```python
# bagel.py generate_image()
if enable_taylorseer:
    self.language_model.model.enable_taylorseer = True
    model_pred_cache_dic, model_pred_current = cache_init(self, num_timesteps)
```

```python
# qwen2_navit.py Qwen2MoTDecoderLayer.forward_inference()
if enable_taylorseer:
    if self.current['type'] == 'full':
        # 完整计算并缓存
        derivative_approximation(cache_dic, current, feature)
    elif self.current['type'] == 'Taylor':
        # 用 Taylor 公式近似，跳过完整计算
        packed_query_sequence = taylor_formula(cache_dic, current)
```

FLUX.2 的 KV Cache 只缓存参考图的 K/V，而 TaylorSeer 直接近似整层输出，加速更激进。

#### 15. NaViT 风格的可变分辨率支持

Bagel 使用 NaViT（Native Resolution ViT）风格的序列打包（packing），支持在同一 batch 中处理不同分辨率的图像：

```python
# 使用 flash_attn_varlen_func 处理可变长度序列
packed_attn_output = flash_attn_varlen_func(
    q=packed_query_states,
    k=merged_key_states,
    v=merged_value_states,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=max(query_lens),
    max_seqlen_k=max(key_values_lens),
)
```

FLUX.2 也支持可变分辨率，但 Bagel 的实现与 LLM 的 packed sequence 机制更深度集成。

### 创新点总结表

| # | 创新点 | Bagel | FLUX.2 |
|---|--------|-------|--------|
| 1 | 统一理解+生成 | ✅ 同一模型 | ❌ 仅生成/编辑 |
| 2 | LLM即DIT | ✅ LLM作为去噪网络 | ❌ 独立DIT |
| 3 | MoT架构 | ✅ 理解/生成双路径 | ❌ 单一路径 |
| 4 | 双编码器(ViT+VAE) | ✅ 语义+像素级 | ❌ 仅VAE |
| 5 | 无需独立文本编码器 | ✅ LLM自编码 | ❌ 需Mistral-24B/Qwen3 |
| 6 | 双损失(CE+MSE) | ✅ 同时优化 | ❌ 仅MSE |
| 7 | 思维链(CoT) | ✅ Think模式 | ❌ 不支持 |
| 8 | 双CFG(text+image) | ✅ 独立控制 | ❌ 单一guidance |
| 9 | CFG-Renorm | ✅ 3种策略 | ❌ 无 |
| 10 | Latent Patchify | ✅ 减少4x tokens | ❌ 直接flatten |
| 11 | 分阶段KV Cache | ✅ 条件信息预缓存 | ✅ KV Cache(类似) |
| 12 | 条件Dropout | ✅ text/vae/vit独立 | ❌ 不明确 |
| 13 | 冻结理解参数 | ✅ freeze_und | ❌ 不适用 |
| 14 | TaylorSeer加速 | ✅ Taylor展开近似 | ❌ 无 |
| 15 | NaViT可变分辨率 | ✅ packed sequence | ✅ 支持(方式不同) |
