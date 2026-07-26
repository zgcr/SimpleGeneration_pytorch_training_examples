# Lance 模型全面分析报告

> **分析对象**：[Lance](https://github.com/bytedance/Lance) — Unified Multimodal Modeling by Multi-Task Synergy
>
> **模型规模**：3B 活跃参数
>
> **开发团队**：ByteDance Research
>
> **分析依据**：基于 Lance 代码仓库中所有核心代码文件的逐行分析
>
> **代码实现根目录**：`/root/code/Lance/`

---

## 问题 1：这个模型是否使用了 VAE？若使用了，VAE 结构属于什么类型？

### 结论

**Lance 模型使用了 VAE，但既不是 FLUX1 VAE 也不是 FLUX2 VAE，而是 Wan2.2 Video VAE（万相视频模型 VAE）。**

### 分析依据

#### 1.1 VAE 的初始化与使用

在推理入口 `inference_lance.py` 第 479 行：

```python
vae_model = WanVideoVAE()
vae_config: AutoEncoderParams = deepcopy(vae_model.vae_config)
```

在 `modeling/vae/wan/model.py` 第 35-57 行，`WanVideoVAE` 类初始化了 VAE 配置：

```python
class WanVideoVAE(object):
    __version__ = "v2.2"
    __name__ = "WanVideoVAE"

    def __init__(self, config_path: str = "", **kwargs) -> None:
        ...
        self.vae_config = AutoEncoderParams(
            downsample_spatial=16,
            downsample_temporal=4,
            z_channels=48,
        )
```

#### 1.2 VAE 的底层结构

在 `modeling/vae/wan/vae2_2.py` 中，VAE 的核心实现为 `Wan2_2_VAE` 类（第 861 行），其底层结构为 `WanVAE_` 类（第 710 行）：

```python
class WanVAE_(nn.Module):
    def __init__(self, dim=160, dec_dim=256, z_dim=16, dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2, attn_scales=[], temperal_downsample=[True, True, False], dropout=0.0):
        self.encoder = Encoder3d(dim, z_dim * 2, ...)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dec_dim, z_dim, ...)
```

Wan2.2 VAE 的实例化参数为：

```python
Wan2_2_VAE(z_dim=48, c_dim=160, dim_mult=[1, 2, 4, 4],
           temperal_downsample=[False, True, True])
```

#### 1.3 关键特征对比

| 特征 | Wan2.2 VAE (Lance 使用) | FLUX1 VAE | FLUX2 VAE |
|------|------------------------|-----------|-----------|
| 维度 | **3D** (视频+图像) | 2D (仅图像) | 2D (仅图像) |
| 卷积类型 | **CausalConv3d** | Conv2d | Conv2d |
| z_channels | **48** | 16 | 128 |
| 空间下采样倍数 | **16** | 8 | 16 |
| 时间下采样倍数 | **4** | N/A | N/A |
| 输入通道 | **12** (patchify 3×2×2) | 3 | 3 |
| Encoder 结构 | 3D ResBlock + AttentionBlock | 2D ResBlock + AttnBlock | 2D ResBlock + AttnBlock |
| Decoder 结构 | 3D ResBlock + AttentionBlock | 2D ResBlock + AttnBlock | 2D ResBlock + AttnBlock |
| Normalization | **RMS_norm** | GroupNorm | GroupNorm |

**结论：Lance 使用的 VAE 是 Wan2.2 3D Causal VAE，专门用于视频/图像的统一编解码，与 FLUX1/FLUX2 的 2D VAE 完全不同。**

---

## 问题 2：这个模型是否使用了 flow_matching 的 DIT 模型？

### 结论

**Lance 使用了 Flow Matching 训练算法，但其去噪骨干网络不是传统的 DIT 或双流 MMDIT，而是基于 Qwen2 大语言模型（LLM）的 Transformer Decoder。** 这是一种 "LLM as Denoiser" 的全新范式。

### 分析依据

#### 2.1 Flow Matching 训练

在 `modeling/lance/lance.py` 第 256-298 行的 `forward()` 方法中，可以清楚看到 Flow Matching 的训练过程：

```python
# Flow Matching 加噪过程
noise = torch.randn_like(packed_latent_clean)
packed_timesteps = torch.sigmoid(packed_timesteps)
packed_timesteps = self.timestep_shift * packed_timesteps / (1 + (self.timestep_shift - 1) * packed_timesteps)
packed_latent = (1 - packed_timesteps[:, None]) * packed_latent_clean + packed_timesteps[:, None] * noise

# 目标：velocity = noise - clean
target = noise - packed_latent_clean
mse = (packed_mse_preds - target[has_mse]) ** 2
```

#### 2.2 去噪骨干网络

去噪骨干网络不是独立的 DIT/MMDIT 模型，而是 Qwen2 LLM：

```python
# lance.py 第 284 行
last_hidden_state = self.language_model(
    packed_sequence=packed_sequence,
    sample_lens=sample_lens,
    attention_mask=attention_mask,
    packed_position_ids=packed_position_ids,
    **extra_inputs,
)
```

其中 `self.language_model` 是 `Qwen2ForCausalLM`，定义在 `qwen2_navit.py` 第 976 行：

```python
class Qwen2ForCausalLM(Qwen2PreTrainedModel):
    def __init__(self, config: Qwen2Config):
        self.model: Qwen2Model = Qwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
```

#### 2.3 与传统 DIT/MMDIT 的区别

| 特征 | 传统 DIT | FLUX2 MMDIT | Lance (LLM as Denoiser) |
|------|---------|-------------|------------------------|
| 骨干网络 | ViT-style Transformer | Double-Stream + Single-Stream | **Qwen2 Causal Decoder** |
| 注意力类型 | 双向 Self-Attention | 双向 Joint Attention | **灵活的 Sparse/Causal Attention** |
| 文本条件注入 | Cross-Attention / AdaLN | Joint Attention + AdaLN | **直接 token 拼接到统一序列** |
| 时间步条件 | AdaLN / DiT Block | AdaLN Modulation | **TimestepEmbedder + 加到 latent embedding** |
| 位置编码 | 2D/3D Sincos | 4D RoPE 矩阵旋转 | **Qwen2 RoPE + 3D Sincos** |
| 是否可做理解 | ❌ | ❌ | **✅ 同时支持理解和生成** |

**因此，Lance 不是传统的单流 DIT 模型，也不是双流 MMDIT 模型。它是一种基于 LLM Decoder 的 Flow Matching 去噪模型。**

---

## 问题 3：这个模型的具体网络结构是怎样的？有哪些子网络结构？

### 结论

Lance 模型由以下 **6 个核心子网络结构** 组成：

### 3.1 子网络 1：Qwen2 LLM（核心骨干网络）

- **代码位置**：`modeling/lance/qwen2_navit.py`
- **结构**：`Qwen2ForCausalLM` → `Qwen2Model` → `Qwen2DecoderLayer` × N 层
- **功能**：统一的骨干 Transformer Decoder，同时用于文本理解和视觉生成的去噪
- **关键组件**：
  - `embed_tokens`：文本 token 嵌入层
  - `layers`：N 层 Decoder Layer（支持三种变体）
  - `norm` / `norm_moe_gen`：最终 RMSNorm 层
  - `lm_head`：语言模型输出头（用于文本生成）
  - `rotary_emb`：旋转位置编码（RoPE）
- **层变体**：
  - `Qwen2DecoderLayer`：标准 Decoder 层，共享 QKV/MLP
  - `Qwen2MoEDecoderLayer`：MoE FFN 变体，注意力共享但 FFN 分为理解/生成两套
  - `Qwen2MoTDecoderLayer`：**MoT（Mixture of Transformers）变体**，注意力 QKV/O 和 FFN 都分为理解/生成两套独立参数（`PackedAttentionMoT`）

### 3.2 子网络 2：Qwen2.5-VL ViT（视觉理解编码器）

- **代码位置**：`modeling/vit/qwen2_5_vl_vit.py`
- **结构**：`Qwen2_5_VisionTransformerPretrainedModel`
- **功能**：将输入图像/视频帧编码为视觉特征，用于图像理解和编辑任务中的参考图像理解
- **关键组件**：
  - `patch_embed`：3D PatchEmbed（`Conv3d`，kernel=[temporal_patch_size, patch_size, patch_size]）
  - `rotary_pos_emb`：2D 视觉旋转位置编码
  - `blocks`：N 层 `Qwen2_5_VLVisionBlock`（含 Attention + MLP）
  - `merger`：`Qwen2_5_VLPatchMerger`（spatial_merge_size=2，将 2×2 patch 合并为 1 个 token）
- **状态**：**冻结**（`vit_model.eval()`，不参与训练）

### 3.3 子网络 3：MLP Connector（ViT → LLM 连接器）

- **代码位置**：`modeling/lance/modeling_utils.py` 第 149-160 行
- **结构**：`MLPconnector`
- **功能**：将 ViT 输出维度投影到 LLM 隐藏维度
- **实现**：
  ```python
  class MLPconnector(nn.Module):
      def __init__(self, in_dim, out_dim, hidden_act):
          self.fc1 = nn.Linear(in_dim, out_dim)
          self.activation_fn = ACT2FN[hidden_act]  # gelu_pytorch_tanh
          self.fc2 = nn.Linear(out_dim, out_dim)
  ```

### 3.4 子网络 4：Wan2.2 VAE（视频/图像编解码器）

- **代码位置**：`modeling/vae/wan/model.py` + `modeling/vae/wan/vae2_2.py`
- **结构**：`WanVideoVAE` → `Wan2_2_VAE` → `WanVAE_`（包含 `Encoder3d` + `Decoder3d`）
- **功能**：
  - **Encoder**：将像素空间的图像/视频编码为潜在表示（latent）
  - **Decoder**：将潜在表示解码回像素空间
- **关键参数**：`z_channels=48, downsample_spatial=16, downsample_temporal=4`
- **状态**：**冻结**

### 3.5 子网络 5：生成任务辅助模块

- **代码位置**：`modeling/lance/lance.py` + `modeling/lance/modeling_utils.py`
- **包含组件**：
  - `TimestepEmbedder`：时间步嵌入（Sinusoidal → MLP[256 → hidden_size → hidden_size]）
  - `vae2llm`：`nn.Linear(patch_latent_dim, hidden_size)` — VAE latent → LLM 维度的线性投影
  - `llm2vae`：`nn.Linear(hidden_size, patch_latent_dim)` — LLM 维度 → VAE latent 的线性投影
  - `PositionEmbedding3D`：3D 正弦余弦位置编码（时间 × 高度 × 宽度）
- **功能**：将 VAE 潜在空间与 LLM 隐藏空间之间进行映射

### 3.6 子网络 6：文本 Tokenizer

- **代码位置**：`modeling/qwen2/tokenization_qwen2.py`
- **结构**：`Qwen2Tokenizer`（BPE tokenizer）
- **功能**：文本分词和 token 化

---

## 问题 4：由各个子网络结构组成的模型网络结构图

### 整体模型结构图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Lance 统一多模态模型                                │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                         输入处理层                                       │   │
│  │                                                                          │   │
│  │  文本输入 ──→ [Qwen2 Tokenizer] ──→ text_ids                            │   │
│  │                      │                                                   │   │
│  │                      ▼                                                   │   │
│  │              [embed_tokens] ──→ text_embedding                           │   │
│  │                                      │                                   │   │
│  │  参考图像/视频 ──→ [ViT (冻结)] ──→ [MLP Connector] ──→ vit_embedding    │   │
│  │                                                             │            │   │
│  │  目标图像/视频 ──→ [VAE Encoder (冻结)] ──→ latent                       │   │
│  │                                              │                           │   │
│  │                    时间步 t ──→ [TimestepEmbedder] ──→ timestep_embed     │   │
│  │                                                           │              │   │
│  │                    latent ──→ [vae2llm] + timestep_embed                 │   │
│  │                              + [3D PositionEmbed] ──→ vae_embedding      │   │
│  │                                                           │              │   │
│  │              ┌────────────────────────────────────────────┐│              │   │
│  │              │    packed_sequence (统一序列)               ││              │   │
│  │              │  = text_embed + vit_embed + vae_embed      ││              │   │
│  │              └────────────────────────────────────────────┘│              │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                          │
│                                      ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                    Qwen2 LLM 骨干网络 (MoT Decoder)                      │   │
│  │                                                                          │   │
│  │  packed_sequence ──→ [RoPE Position Encoding]                            │   │
│  │                          │                                               │   │
│  │                          ▼                                               │   │
│  │              ┌─────────────────────────────────┐                         │   │
│  │              │  Qwen2MoTDecoderLayer × N 层     │                         │   │
│  │              │                                  │                         │   │
│  │              │  理解 tokens ──→ [Attn_und] ───┐ │                         │   │
│  │              │  生成 tokens ──→ [Attn_gen] ─┐ │ │                         │   │
│  │              │                              ▼ ▼ │                         │   │
│  │              │              [Joint Attention]    │                         │   │
│  │              │                     │             │                         │   │
│  │              │  理解 tokens ──→ [MLP_und] ──→ + │                         │   │
│  │              │  生成 tokens ──→ [MLP_gen] ──→ + │                         │   │
│  │              └─────────────────────────────────┘                         │   │
│  │                          │                                               │   │
│  │                          ▼                                               │   │
│  │              [RMSNorm / RMSNorm_moe_gen]                                 │   │
│  │                          │                                               │   │
│  │                  last_hidden_state                                        │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                          │
│                                      ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                         输出处理层                                       │   │
│  │                                                                          │   │
│  │  生成任务: last_hidden_state[mse_indexes]                                │   │
│  │              ──→ [llm2vae] ──→ velocity_pred                             │   │
│  │              ──→ MSE Loss (vs noise - clean)                             │   │
│  │                                                                          │   │
│  │  理解任务: last_hidden_state[ce_indexes]                                 │   │
│  │              ──→ [lm_head] ──→ logits                                    │   │
│  │              ──→ CE Loss                                                 │   │
│  │                                                                          │   │
│  │  推理解码: velocity_pred ──→ ODE Euler Step ──→ 去噪 latent              │   │
│  │              ──→ [VAE Decoder (冻结)] ──→ 输出图像/视频                   │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 简化版箭头示意图

```
文本 ──→ [Tokenizer] ──→ [embed_tokens] ──→ ┐
                                              │
参考图 ──→ [ViT] ──→ [MLP Connector] ──→ ────┤
                                              │──→ packed_sequence ──→ [Qwen2 LLM (MoT)] ──→ last_hidden_state
目标图 ──→ [VAE Enc] ──→ [vae2llm] ──→ ──────┤                                                    │
                          + [TimeEmb]         │                                                    ├──→ [llm2vae] ──→ velocity ──→ [VAE Dec] ──→ 输出图像/视频
                          + [3D PosEmb]       ┘                                                    │
                                                                                                   └──→ [lm_head] ──→ 文本输出
```

---

## 问题 5：这个模型能否实现文生图？能否实现图像+文本提示进行图像编辑？

### 结论

**✅ Lance 模型能实现文生图（T2I）。**

**✅ Lance 模型能实现图像+文本提示进行图像编辑（Image Edit）。**

### 分析依据

#### 5.1 文生图支持

在 `inference_lance.py` 第 63-74 行，明确定义了支持的任务：

```python
TASK_T2I = "t2i"
GENERATION_TASKS = {TASK_T2V, TASK_T2I, TASK_I2V, TASK_IMAGE_EDIT, TASK_VIDEO_EDIT}
```

在 `README.md` 中给出了文生图推理命令：

```bash
bash inference_lance.sh \
  --TASK_NAME t2i \
  --MODEL_PATH downloads/Lance_3B \
  --RESOLUTION image_768res
```

#### 5.2 图像编辑支持

同样在任务定义中：

```python
TASK_IMAGE_EDIT = "image_edit"
```

推理命令：

```bash
bash inference_lance.sh \
  --TASK_NAME image_edit \
  --MODEL_PATH downloads/Lance_3B \
  --RESOLUTION image_768res
```

#### 5.3 完整任务支持列表

| 任务类型 | 任务名称 | 类别 |
|---------|---------|------|
| 文生图 | `t2i` | 生成 |
| 文生视频 | `t2v` | 生成 |
| 图生视频 | `i2v` | 生成 |
| 图像编辑 | `image_edit` | 生成 |
| 视频编辑 | `video_edit` | 生成 |
| 图像理解 | `x2t_image` | 理解 |
| 视频理解 | `x2t_video` | 理解 |

---

## 问题 6：文生图流程图

### 文生图（T2I）完整流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            文生图 (T2I) 推理流程                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────── 输入数据 ───────────────────┐                         │
│  │                                                │                         │
│  │  1. 文本 prompt (字符串)                        │                         │
│  │  2. 目标图像尺寸 (H, W)                        │                         │
│  │                                                │                         │
│  └────────────────────────────────────────────────┘                         │
│                          │                                                   │
│                          ▼                                                   │
│  ┌─────────────────── Step 1: 文本编码 ───────────────────┐                 │
│  │                                                         │                 │
│  │  文本 prompt                                            │                 │
│  │      │                                                  │                 │
│  │      ▼                                                  │                 │
│  │  [Qwen2 Tokenizer] ──→ text_ids (1D token序列)         │                 │
│  │      │                                                  │                 │
│  │      ▼                                                  │                 │
│  │  [embed_tokens] ──→ text_embedding (seq_len, hidden_dim)│                 │
│  │                                                         │                 │
│  │  ★ 子网络: Qwen2 Tokenizer + embed_tokens              │                 │
│  └─────────────────────────────────────────────────────────┘                 │
│                          │                                                   │
│                          ▼                                                   │
│  ┌─────────────────── Step 2: 初始化噪声 ─────────────────┐                 │
│  │                                                         │                 │
│  │  根据目标尺寸 (H, W) 计算 latent 尺寸:                  │                 │
│  │    t = 1 (单帧图像)                                     │                 │
│  │    h = H / (downsample_spatial × patch_h)               │                 │
│  │    w = W / (downsample_spatial × patch_w)               │                 │
│  │                                                         │                 │
│  │  x_T ~ N(0, I), shape = (t*h*w, patch_latent_dim)      │                 │
│  │  其中 patch_latent_dim = pt × ph × pw × z_channels     │                 │
│  │                       = 1 × 2 × 2 × 48 = 192           │                 │
│  └─────────────────────────────────────────────────────────┘                 │
│                          │                                                   │
│                          ▼                                                   │
│  ┌─────────────────── Step 3: 构建统一序列 ───────────────┐                  │
│  │                                                         │                 │
│  │  packed_sequence[text_indexes] = text_embedding          │                 │
│  │                                                         │                 │
│  │  ★ 无 ViT 输入（纯文生图不需要参考图像）                │                 │
│  └─────────────────────────────────────────────────────────┘                 │
│                          │                                                   │
│                          ▼                                                   │
│  ┌─────── Step 4: 去噪循环 (num_timesteps 步) ───────────┐                  │
│  │                                                         │                 │
│  │  timesteps: [1.0, ..., 0.0] (等间距, timestep_shift)    │                 │
│  │                                                         │                 │
│  │  for each timestep t_i:                                 │                 │
│  │    │                                                    │                 │
│  │    ▼                                                    │                 │
│  │  ┌── Step 4a: VAE token 编码 ──┐                       │                 │
│  │  │                              │                       │                 │
│  │  │  timestep_embed = [TimestepEmbedder](t_i)            │                 │
│  │  │  latent_pos_embed = [3D PositionEmbed](pos_ids)      │                 │
│  │  │  vae_embed = [vae2llm](x_t) + timestep_embed        │                 │
│  │  │              + latent_pos_embed                      │                 │
│  │  │                              │                       │                 │
│  │  │  packed_sequence[vae_indexes] = vae_embed            │                 │
│  │  │                              │                       │                 │
│  │  │  ★ 子网络: TimestepEmbedder, vae2llm, 3D PosEmbed   │                 │
│  │  └──────────────────────────────┘                       │                 │
│  │    │                                                    │                 │
│  │    ▼                                                    │                 │
│  │  ┌── Step 4b: LLM Forward ──┐                          │                 │
│  │  │                            │                         │                 │
│  │  │  last_hidden_state =                                 │                 │
│  │  │    [Qwen2 LLM](packed_sequence,                      │                 │
│  │  │                 attention_mask,                       │                 │
│  │  │                 position_ids)                         │                 │
│  │  │                            │                         │                 │
│  │  │  ★ 子网络: Qwen2 LLM (MoT Decoder)                  │                 │
│  │  └────────────────────────────┘                         │                 │
│  │    │                                                    │                 │
│  │    ▼                                                    │                 │
│  │  ┌── Step 4c: 预测 velocity ──┐                        │                 │
│  │  │                              │                       │                 │
│  │  │  v_t = [llm2vae](                                    │                 │
│  │  │          last_hidden_state[mse_indexes])              │                 │
│  │  │                              │                       │                 │
│  │  │  ★ 子网络: llm2vae                                   │                 │
│  │  └──────────────────────────────┘                       │                 │
│  │    │                                                    │                 │
│  │    ▼                                                    │                 │
│  │  ┌── Step 4d: CFG (可选) ──┐                           │                 │
│  │  │                          │                           │                 │
│  │  │  if cfg_text_scale > 1:                              │                 │
│  │  │    v_uncond = uncond_forward(...)                     │                 │
│  │  │    v_t = v_uncond + scale*(v_t - v_uncond)           │                 │
│  │  │    + cfg_renorm                                      │                 │
│  │  └──────────────────────────┘                           │                 │
│  │    │                                                    │                 │
│  │    ▼                                                    │                 │
│  │  ┌── Step 4e: Euler ODE Step ──┐                       │                 │
│  │  │                               │                      │                 │
│  │  │  x_t = x_t - v_t * dt                               │                 │
│  │  └───────────────────────────────┘                      │                 │
│  │                                                         │                 │
│  └─────────────────────────────────────────────────────────┘                 │
│                          │                                                   │
│                          ▼                                                   │
│  ┌─────────────────── Step 5: VAE 解码 ───────────────────┐                 │
│  │                                                         │                 │
│  │  x_0 (去噪后 latent)                                    │                 │
│  │      │                                                  │                 │
│  │      ▼                                                  │                 │
│  │  unpatchify: (t*h*w, pt*ph*pw*c) → (T, H', W', C)      │                 │
│  │      │                                                  │                 │
│  │      ▼                                                  │                 │
│  │  [VAE Decoder (冻结)] ──→ 输出图像 (3, H, W)            │                 │
│  │                                                         │                 │
│  │  ★ 子网络: Wan2.2 VAE Decoder                           │                 │
│  └─────────────────────────────────────────────────────────┘                 │
│                          │                                                   │
│                          ▼                                                   │
│  ┌─────────────────── 输出数据 ───────────────────┐                         │
│  │                                                │                         │
│  │  生成的图像 (RGB, H × W)                        │                         │
│  │  (最终由 VAE Decoder 子网络输出)                 │                         │
│  └────────────────────────────────────────────────┘                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 问题 7：图像+文本提示进行图像编辑流程图

### 图像编辑（Image Edit）完整流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        图像编辑 (Image Edit) 推理流程                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────── 输入数据 ───────────────────┐                         │
│  │                                                │                         │
│  │  1. 源图像 (原始图像，需要被编辑)               │                         │
│  │  2. 文本 prompt (编辑指令)                      │                         │
│  │                                                │                         │
│  └────────────────────────────────────────────────┘                         │
│                    │                    │                                     │
│                    ▼                    ▼                                     │
│  ┌─── Step 1a: 源图像 ViT 编码 ───┐  ┌─── Step 1b: 源图像 VAE 编码 ──┐     │
│  │                                 │  │                                │     │
│  │  源图像                         │  │  源图像                        │     │
│  │    │                            │  │    │                           │     │
│  │    ▼                            │  │    ▼                           │     │
│  │  预处理为 ViT 输入 tokens       │  │  [VAE Encoder (冻结)]         │     │
│  │    │                            │  │    │                           │     │
│  │    ▼                            │  │    ▼                           │     │
│  │  [Qwen2.5-VL ViT (冻结)]       │  │  source_latent (t,h,w,c)      │     │
│  │    │                            │  │    │                           │     │
│  │    ▼                            │  │    ▼                           │     │
│  │  [MLP Connector]               │  │  patchify ──→ source_tokens    │     │
│  │    │                            │  │  (clean, 不加噪)              │     │
│  │    ▼                            │  │                                │     │
│  │  vit_embedding                  │  │  ★ 子网络: VAE Encoder         │     │
│  │                                 │  └────────────────────────────────┘     │
│  │  ★ 子网络: ViT + Connector     │                │                        │
│  └─────────────────────────────────┘                │                        │
│           │                                          │                        │
│           ▼                                          ▼                        │
│  ┌─── Step 2: 文本编码 ───┐                                                 │
│  │                         │                                                 │
│  │  文本 prompt            │                                                 │
│  │    │                    │                                                 │
│  │    ▼                    │                                                 │
│  │  [Tokenizer]            │                                                 │
│  │    │                    │                                                 │
│  │    ▼                    │                                                 │
│  │  [embed_tokens]         │                                                 │
│  │    │                    │                                                 │
│  │    ▼                    │                                                 │
│  │  text_embedding         │                                                 │
│  └─────────────────────────┘                                                 │
│           │                                                                   │
│           ▼                                                                   │
│  ┌─── Step 3: 构建统一序列（含 source + target） ───────────────────┐        │
│  │                                                                   │        │
│  │  序列结构 (interleave format):                                    │        │
│  │                                                                   │        │
│  │  [text_tokens | vit_tokens(源图) | source_vae_tokens | target_vae_tokens]  │
│  │       ↑              ↑                   ↑                  ↑      │        │
│  │    causal         full attn          noise(clean)      noise(noisy)│        │
│  │                                                                   │        │
│  │  source_vae_tokens: 源图 VAE latent (clean, timestep=0)          │        │
│  │  target_vae_tokens: 目标噪声 (初始 x_T ~ N(0,I))                │        │
│  │                                                                   │        │
│  │  packed_sequence = 组合所有 embedding                              │        │
│  │  packed_sequence[text_idx] = text_embedding                        │        │
│  │  packed_sequence[vit_idx] = vit_embedding                          │        │
│  │  packed_sequence[vae_idx] = vae_embedding (source + target)        │        │
│  └───────────────────────────────────────────────────────────────────┘        │
│                          │                                                    │
│                          ▼                                                    │
│  ┌─── Step 4: 去噪循环 (num_timesteps 步) ──────────────────────────┐        │
│  │                                                                    │        │
│  │  for each timestep t_i:                                            │        │
│  │    │                                                               │        │
│  │    ▼                                                               │        │
│  │  ┌── Step 4a: VAE token 编码 ──┐                                  │        │
│  │  │                              │                                  │        │
│  │  │  source_vae_embed:                                              │        │
│  │  │    [vae2llm](source_tokens) + [TimeEmb](0) + [3D PosEmb]       │        │
│  │  │    (timestep=0, 源图保持 clean)                                 │        │
│  │  │                                                                 │        │
│  │  │  target_vae_embed:                                              │        │
│  │  │    [vae2llm](x_t) + [TimeEmb](t_i) + [3D PosEmb]              │        │
│  │  │    (timestep=t_i, 目标图在去噪)                                 │        │
│  │  │                                                                 │        │
│  │  │  packed_sequence[vae_idx] = source_embed + target_embed         │        │
│  │  └──────────────────────────────┘                                  │        │
│  │    │                                                               │        │
│  │    ▼                                                               │        │
│  │  ┌── Step 4b: LLM Forward ──┐                                     │        │
│  │  │                            │                                    │        │
│  │  │  Sparse Attention Mask:                                         │        │
│  │  │    - text: causal                                               │        │
│  │  │    - vit: full (双向)                                           │        │
│  │  │    - source_vae: full_noise (与text+vit+target双向)             │        │
│  │  │    - target_vae: noise (与text+vit+source双向)                  │        │
│  │  │                                                                 │        │
│  │  │  last_hidden_state =                                            │        │
│  │  │    [Qwen2 LLM (MoT)](packed_sequence, mask, pos)                │        │
│  │  │                                                                 │        │
│  │  │  ★ 子网络: Qwen2 LLM                                           │        │
│  │  └────────────────────────────┘                                    │        │
│  │    │                                                               │        │
│  │    ▼                                                               │        │
│  │  ┌── Step 4c: 预测 velocity (仅目标部分) ──┐                      │        │
│  │  │                                          │                      │        │
│  │  │  v_t = [llm2vae](                                               │        │
│  │  │          last_hidden_state[mse_indexes])                         │        │
│  │  │                                                                 │        │
│  │  │  注: mse_indexes 仅对应 target tokens                           │        │
│  │  │      (不对 source tokens 计算 loss)                              │        │
│  │  │                                                                 │        │
│  │  │  ★ 子网络: llm2vae                                              │        │
│  │  └──────────────────────────────┘                                  │        │
│  │    │                                                               │        │
│  │    ▼                                                               │        │
│  │  ┌── Step 4d: CFG (可选, text + vit 两级) ──┐                     │        │
│  │  │                                            │                    │        │
│  │  │  if cfg_text_scale > 1:                                         │        │
│  │  │    v_uncond_text = forward(去掉文本条件)                         │        │
│  │  │                                                                 │        │
│  │  │  if cfg_vit_scale > 1:                                          │        │
│  │  │    v_uncond_vit = forward(去掉文本+ViT条件)                     │        │
│  │  │                                                                 │        │
│  │  │  v_t = v_uncond_vit                                             │        │
│  │  │      + cfg_text * (v_t - v_uncond_text)                         │        │
│  │  │      + cfg_vit * (v_uncond_text - v_uncond_vit)                 │        │
│  │  │    + cfg_renorm                                                 │        │
│  │  └────────────────────────────────┘                                │        │
│  │    │                                                               │        │
│  │    ▼                                                               │        │
│  │  ┌── Step 4e: Euler ODE Step ──┐                                  │        │
│  │  │                               │                                 │        │
│  │  │  x_t[target_idx] -= v_t * dt                                    │        │
│  │  │  (仅更新目标 tokens)                                            │        │
│  │  └───────────────────────────────┘                                 │        │
│  │                                                                    │        │
│  └────────────────────────────────────────────────────────────────────┘        │
│                          │                                                     │
│                          ▼                                                     │
│  ┌─────────────────── Step 5: VAE 解码 ───────────────────┐                   │
│  │                                                         │                   │
│  │  x_0_target (去噪后的目标 latent)                       │                   │
│  │      │                                                  │                   │
│  │      ▼                                                  │                   │
│  │  unpatchify: (t*h*w, pt*ph*pw*c) → (T, H', W', C)      │                   │
│  │      │                                                  │                   │
│  │      ▼                                                  │                   │
│  │  [VAE Decoder (冻结)] ──→ 编辑后的图像 (3, H, W)        │                   │
│  │                                                         │                   │
│  │  ★ 子网络: Wan2.2 VAE Decoder                           │                   │
│  └─────────────────────────────────────────────────────────┘                   │
│                          │                                                     │
│                          ▼                                                     │
│  ┌─────────────────── 输出数据 ───────────────────┐                           │
│  │                                                │                           │
│  │  编辑后的图像 (RGB, H × W)                      │                           │
│  │  (最终由 VAE Decoder 子网络输出)                 │                           │
│  └────────────────────────────────────────────────┘                           │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 图像编辑与文生图的关键区别

| 特征 | 文生图 (T2I) | 图像编辑 (Image Edit) |
|------|-------------|---------------------|
| ViT 输入 | ❌ 无 | ✅ 源图像经 ViT 编码 |
| 源图 VAE latent | ❌ 无 | ✅ 源图像经 VAE 编码，作为 clean condition |
| 目标噪声初始化 | 纯噪声 x_T | 纯噪声 x_T（目标位置） |
| Attention Mask | 简单 causal + noise | 复杂 sparse mask（text causal + vit full + source full_noise + target noise）|
| CFG 类型 | text CFG | text CFG + vit CFG（两级 CFG）|
| MSE Loss 位置 | 所有 VAE tokens | 仅目标 VAE tokens |

---

## 问题 8：相比于 FLUX2 模型，Lance 模型具有哪些创新点或改进点？

### 全面对比与创新点列表

#### 创新点 1：LLM as Denoiser — 用大语言模型替代独立 DIT/MMDIT

- **FLUX2**：使用独立的 MMDIT（Double-Stream 8层 + Single-Stream 48层，~12B参数）作为去噪网络，文本编码由独立的 text encoder 提供
- **Lance**：**直接使用 Qwen2 LLM Causal Decoder 作为去噪骨干网络**，将 Flow Matching 的 velocity prediction 任务融入到 LLM 的 forward pass 中
- **意义**：这是一种全新范式，将视觉生成和语言理解完全统一到同一个 Transformer 中

#### 创新点 2：统一多模态模型 — 一个模型同时支持 7 种任务

- **FLUX2**：仅支持图像生成和图像编辑（通过参考图 KV-Cache）
- **Lance**：**单个 3B 模型同时支持文生图、文生视频、图生视频、图像编辑、视频编辑、图像理解、视频理解** 共 7 种任务
- **实现方式**：通过统一的 packed sequence + sparse attention mask 机制，不同任务使用不同的 attention pattern

#### 创新点 3：MoT（Mixture of Transformers）架构

- **FLUX2**：所有 token 共享同一套 QKV/MLP 参数
- **Lance**：引入 `Qwen2MoTDecoderLayer`，**为理解任务和生成任务分别维护独立的 QKV projection、Output projection、LayerNorm 和 MLP**
  - 理解 tokens → `q_proj`, `k_proj`, `v_proj`, `o_proj`, `mlp`, `input_layernorm`, `post_attention_layernorm`
  - 生成 tokens → `q_proj_moe_gen`, `k_proj_moe_gen`, `v_proj_moe_gen`, `o_proj_moe_gen`, `mlp_moe_gen`, `input_layernorm_moe_gen`, `post_attention_layernorm_moe_gen`
  - **但注意力计算本身是 joint 的**（QKV 分别投影后合并做 attention）
- **意义**：在保持统一 attention 的同时，让理解和生成任务在参数空间上解耦，减少任务冲突

#### 创新点 4：3D Video VAE 支持

- **FLUX2**：使用 2D VAE（`in_channels=128`），仅支持静态图像
- **Lance**：使用 **Wan2.2 3D Causal VAE**（`z_channels=48, downsample_spatial=16, downsample_temporal=4`），原生支持视频和图像的统一编解码
- **意义**：通过 3D VAE，Lance 可以自然处理视频数据，实现文生视频和视频编辑

#### 创新点 5：无独立 Text Encoder — LLM 自身处理文本

- **FLUX2**：需要独立的 text encoder（context_in_dim=15360，来自外部大模型如 T5/CLIP）
- **Lance**：**LLM 自身的 embed_tokens 直接处理文本 token**，不需要额外的 text encoder
- **意义**：简化了模型架构，减少了推理时的模型数量和显存占用

#### 创新点 6：ViT 视觉理解编码器集成

- **FLUX2**：没有视觉理解能力，图像编辑通过 KV-Cache 机制实现
- **Lance**：集成了 **Qwen2.5-VL ViT** 作为视觉理解编码器，通过 MLP Connector 接入 LLM
- **意义**：使模型具备了真正的视觉理解能力（VQA、Image Captioning 等），并且可以利用视觉理解信息辅助图像/视频编辑

#### 创新点 7：灵活的 Sparse Attention Mask 机制

- **FLUX2**：使用简单的 causal attention + reference token self-attention 机制
- **Lance**：设计了**复杂的 Sparse Attention Mask**，支持多种 attention 模式：
  - `causal`：因果注意力（文本区域）
  - `full`：双向注意力（ViT 区域）
  - `noise`：目标噪声区域的注意力
  - `full_noise`：源图 clean latent 的注意力
  - `full_noise_target`：另一种变体
- **意义**：允许不同类型的 token（文本、ViT 特征、源图 latent、目标噪声）以不同的方式互相关注

#### 创新点 8：Packed Sequence 多样本高效训练

- **FLUX2**：推理时每次处理单个样本
- **Lance**：使用 **Packed Sequence** 技术，将多个不同长度的样本打包到一个序列中并行处理，结合 `flex_attention` 和 `flash_attn_varlen_func` 实现高效计算
- **意义**：显著提高训练效率，减少 padding 浪费

#### 创新点 9：Timestep 条件注入方式不同

- **FLUX2**：通过 **AdaLN Modulation** 注入时间步条件（`Modulation(SiLU(vec) → Linear → shift/scale/gate)`），每个 block 的 norm 层都被 timestep 调制
- **Lance**：通过 **直接加法** 注入时间步条件（`vae_embed = vae2llm(latent) + timestep_embed + pos_embed`），时间步信息在输入嵌入层就融合进去，不修改 LLM 内部结构
- **意义**：Lance 的方式对 LLM 骨干的侵入性最小，允许直接复用预训练的 LLM 权重

#### 创新点 10：Latent Patchify 策略

- **FLUX2**：`in_channels=128`，直接使用 VAE 输出（已经 patchified）
- **Lance**：使用 `latent_patch_size=(1, 2, 2)`，将 VAE 输出的 `(T, H, W, 48)` latent 进一步 patchify 为 `(T*H/2*W/2, 1*2*2*48=192)` 的 token 序列
- **意义**：减少了序列长度（4倍），提高了计算效率

#### 创新点 11：两级 CFG（Text + ViT）

- **FLUX2**：支持单级 text CFG（+ optional guidance embed）
- **Lance**：支持**两级 Classifier-Free Guidance**：
  - `cfg_text_scale`：文本条件的 CFG
  - `cfg_vit_scale`：ViT 视觉条件的 CFG
  - 公式：`v = v_uncond_vit + cfg_text * (v_cond - v_uncond_text) + cfg_vit * (v_uncond_text - v_uncond_vit)`
- **意义**：在编辑任务中可以独立控制文本和视觉条件的引导强度

#### 创新点 12：CFG Renormalization

- **FLUX2**：无 CFG renormalization 机制
- **Lance**：实现了 **CFG Renormalization**，支持 `global` 和 `channel` 两种模式：
  ```python
  scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
  v_t = v_t_ * scale
  ```
- **意义**：防止 CFG 过度放大导致的色彩失真和伪影

#### 创新点 13：3D 正弦余弦位置编码

- **FLUX2**：使用 4D RoPE（t, h, w, l），通过矩阵旋转形式
- **Lance**：使用 **3D 正弦余弦固定位置编码**（`PositionEmbedding3D`，时间 × 高度 × 宽度），加到 VAE token 上；同时 LLM 内部使用 **Qwen2 RoPE**（或 Qwen2.5-VL 的 M-RoPE）
- **意义**：双重位置编码机制——固定 sincos 提供绝对位置信息，RoPE 提供相对位置信息

#### 创新点 14：多模态 RoPE 支持（M-RoPE）

- **FLUX2**：仅使用标准 RoPE
- **Lance**：可选启用 **Qwen2.5-VL 的 Multimodal RoPE（M-RoPE）**，为 temporal/height/width 三个维度分别编码旋转位置
- **意义**：更好地建模视频数据的时空结构

#### 创新点 15：模型规模与效率

- **FLUX2**：12B 参数（仅生成功能）
- **Lance**：**3B 活跃参数**，同时支持理解+生成+编辑 7 种任务
- **意义**：在参数量仅为 FLUX2 的 1/4 的情况下，实现了更多功能

#### 创新点 16：可冻结理解侧参数的训练策略

- **FLUX2**：无此设计
- **Lance**：支持 `freeze_und` 模式，**可以冻结理解侧参数仅训练生成侧参数**，实现解耦训练
  ```python
  if self.config.freeze_und:
      packed_sequence[packed_und_token_indexes] = packed_sequence[packed_und_token_indexes].detach()
  ```
- **意义**：在多任务训练中，可以分阶段训练，避免理解和生成任务之间的互相干扰

---

## 总结

Lance 是一个极具创新性的统一多模态模型，其核心创新在于使用 LLM 直接作为 Flow Matching 的去噪网络，打破了传统视觉生成模型必须使用独立 DIT/MMDIT 的范式。通过 MoT 架构、Sparse Attention、Packed Sequence 等技术，Lance 在仅 3B 参数的规模下实现了图像/视频的理解、生成和编辑的统一。

| 维度 | FLUX2 | Lance |
|------|-------|-------|
| 参数量 | 12B | **3B** |
| 去噪网络 | MMDIT (DS+SS) | **Qwen2 LLM (MoT)** |
| VAE | 2D (128ch) | **3D Wan2.2 (48ch)** |
| Text Encoder | 外部大模型 | **LLM 自身 embed_tokens** |
| 视觉理解 | ❌ | **✅ Qwen2.5-VL ViT** |
| 支持视频 | ❌ | **✅** |
| 支持理解 | ❌ | **✅** |
| 支持编辑 | ✅ (KV-Cache) | **✅ (Source concat + Sparse Mask)** |
| 任务数量 | 2 (生成+编辑) | **7 (T2I/T2V/I2V/ImgEdit/VidEdit/ImgUnd/VidUnd)** |
| CFG 类型 | 单级 | **两级 (Text + ViT) + Renorm** |
| 训练策略 | - | **MoT解耦 + 可冻结理解侧** |
