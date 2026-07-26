# JoyAI-Image 模型全面分析报告

> 本报告基于 JoyAI-Image 代码仓库的完整源代码分析得出，涵盖模型架构、子网络结构、数据流、以及与 FLUX2 模型的对比。

> **代码实现根目录**：`/root/code/JoyAI-Image/`

---

## 目录

1. [VAE 分析](#1-vae-分析)
2. [Flow Matching DIT 分析](#2-flow-matching-dit-分析)
3. [子网络结构总览](#3-子网络结构总览)
4. [模型网络结构图](#4-模型网络结构图)
5. [文生图与图像编辑能力](#5-文生图与图像编辑能力)
6. [文生图流程图](#6-文生图流程图)
7. [图像编辑流程图](#7-图像编辑流程图)
8. [相比 FLUX2 模型的创新点与改进点](#8-相比-flux2-模型的创新点与改进点)

---

## 1. VAE 分析

### 结论：JoyAI-Image 使用了 VAE，且是 Wan2.1 风格的 3D 因果卷积 VAE（WanVAE），与 FLUX1 和 FLUX2 的 VAE 结构均不同。

### 详细分析

**代码证据：**

在 `src/modules/models/__init__.py` 的 `load_pipeline()` 函数中：
```python
vae = build_from_config(cfg.vae_arch_config, **factory_kwargs)
```

VAE 的实现位于 `src/modules/models/mmdit/vae/wanvae.py`，核心类为 `WanxVAE`，它封装了 `WanVAE_` 模型。

**WanVAE 的结构特点：**

| 特性 | WanVAE (JoyAI-Image) | FLUX1 VAE | FLUX2 VAE |
|------|---------------------|-----------|-----------|
| 维度 | **3D（视频/图像统一）** | 2D | 2D |
| 卷积类型 | `CausalConv3d`（因果3D卷积） | `Conv2d` | `Conv2d` |
| latent channels | **16** | 4 | **32**（使用2×2 patchify后为128） |
| 空间下采样因子 | **8** | 8 | **8**（但latent经过2×2 patchify，有效为16） |
| 时间下采样因子 | **4** | N/A | N/A |
| 归一化方式 | `RMS_norm` | `GroupNorm` | `GroupNorm` |
| 注意力块 | 2D `AttentionBlock`（单头） | `AttnBlock` | `AttnBlock` |
| 编码器结构 | `Encoder3d` | `Encoder` (2D) | `Encoder` (2D) |
| 解码器结构 | `Decoder3d` | `Decoder` (2D) | `Decoder` (2D) |
| 重采样 | 混合3D/2D（`downsample3d`/`downsample2d`） | 2D `Downsample` | 2D `Downsample` |
| 归一化方法 | 均值/标准差手动归一化 | N/A | `BatchNorm2d` |

**WanVAE 的具体配置（来自代码）：**
```python
cfg = dict(
    dim=96,
    z_dim=16,  # 16维latent通道
    dim_mult=[1, 2, 4, 4],
    num_res_blocks=2,
    attn_scales=[],
    temperal_downsample=[False, True, True],  # 时间维度：不降采样、降采样、降采样
    dropout=0.0
)
```

**WanVAE 的编码过程：**
- 输入 `(B, 3, T, H, W)` 的RGB视频/图像
- 通过 `CausalConv3d` 初始卷积
- 经过多层 `ResidualBlock`（含 `CausalConv3d`）和 `Resample`（含时间维度下采样）
- 中间层包含残差块 + 2D注意力块
- 输出 `(B, z_dim*2, T', H', W')`，split 为 mu 和 log_var
- 使用重参数化技巧采样 latent
- 使用预设的 mean/std 进行归一化

**WanVAE 的解码过程：**
- 输入归一化的 latent `(B, 16, T', H/8, W/8)`
- 反归一化
- 通过 `Decoder3d`（含 `CausalConv3d`、残差块、上采样）
- 输出 `(B, 3, T, H, W)` 的RGB视频/图像

**结论：WanVAE 是一种独特的3D因果卷积VAE，来源于阿里巴巴的Wan2.1视频生成项目，既不是FLUX1的2D VAE结构，也不是FLUX2的带patchify和BatchNorm的2D VAE结构，而是一种专为视频/图像统一处理设计的3D VAE。**

---

## 2. Flow Matching DIT 分析

### 结论：JoyAI-Image 使用了 Flow Matching 的 DIT 模型，且使用的是**纯双流 MMDiT 模型**（无单流块）。

### 详细分析

**Flow Matching 证据：**

调度器位于 `src/modules/models/scheduler.py`，类名为 `FlowMatchDiscreteScheduler`：
```python
class FlowMatchDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """Euler scheduler."""
```

其核心特征：
- 使用 `sigmas = torch.linspace(1, 0, num_inference_steps + 1)` 线性sigma调度
- 使用 `sd3_time_shift` 函数进行时间位移：`(shift * t) / (1 + (shift - 1) * t)`
- step 方法使用 Euler 求解器：`prev_sample = sample + model_output * dt`
- 这正是标准的 **Flow Matching** 框架

**双流 MMDiT 证据：**

DIT 模型位于 `src/modules/models/mmdit/dit/models.py`，核心类为 `Transformer3DModel`：
```python
class Transformer3DModel(ModelMixin, ConfigMixin):
    # ...
    self.double_blocks = nn.ModuleList(
        [
            MMDoubleStreamBlock(...)  # 仅有双流块
            for _ in range(mm_double_blocks_depth)
        ]
    )
```

**关键：JoyAI-Image 的 DIT 只包含 `double_blocks`（双流块），不包含任何 `single_blocks`（单流块）。**

**MMDoubleStreamBlock 的结构：**

```
MMDoubleStreamBlock:
├── img_mod (ModulateWan): 图像流调制
├── img_norm1 (LayerNorm): 图像流归一化
├── img_attn_qkv (Linear): 图像流 QKV 投影
├── img_attn_q_norm (RMSNorm): Q 归一化
├── img_attn_k_norm (RMSNorm): K 归一化
├── img_attn_proj (Linear): 图像流注意力输出投影
├── img_norm2 (LayerNorm): 图像流 FFN 归一化
├── img_mlp (FeedForward): 图像流 MLP
├── txt_mod (ModulateWan): 文本流调制
├── txt_norm1 (LayerNorm): 文本流归一化
├── txt_attn_qkv (Linear): 文本流 QKV 投影
├── txt_attn_q_norm (RMSNorm): Q 归一化
├── txt_attn_k_norm (RMSNorm): K 归一化
├── txt_attn_proj (Linear): 文本流注意力输出投影
├── txt_norm2 (LayerNorm): 文本流 FFN 归一化
└── txt_mlp (FeedForward): 文本流 MLP
```

**双流块的工作流程：**
1. 图像和文本分别通过各自的调制、归一化和QKV投影
2. 对图像Q/K应用3D RoPE位置编码
3. 将图像和文本的Q、K、V concatenate后做**联合注意力**
4. 注意力输出拆分回图像和文本
5. 分别通过各自的投影、残差连接、FFN等

---

## 3. 子网络结构总览

### JoyAI-Image 包含以下6个主要子网络结构：

| 编号 | 子网络 | 类名 | 描述 | 参数量级 |
|------|--------|------|------|---------|
| 1 | **MLLM 理解模型** | `Qwen3VLForConditionalGeneration` | 8B 多模态大语言模型，用于图像理解任务 | ~8B |
| 2 | **Text Encoder（文本/视觉编码器）** | `Qwen3VLForConditionalGeneration` | 提取文本+图像的条件嵌入，送入DIT | ~8B |
| 3 | **VAE Encoder** | `Encoder3d`（WanVAE内部） | 3D因果卷积编码器，将RGB图像/视频编码为latent | ~数百M |
| 4 | **VAE Decoder** | `Decoder3d`（WanVAE内部） | 3D因果卷积解码器，将latent解码为RGB图像/视频 | ~数百M |
| 5 | **MMDiT（扩散Transformer）** | `Transformer3DModel` | 16B 双流MM-DiT，核心去噪网络 | ~16B |
| 6 | **Scheduler（调度器）** | `FlowMatchDiscreteScheduler` | Flow Matching 调度器（无参数） | 0 |

### 各子网络详细结构：

#### 3.1 MLLM 理解模型（JoyAI-Image-Und）
- **基础架构**：Qwen3VL（Qwen3 Vision-Language Model）
- **功能**：接收图像和文本输入，输出自然语言描述/回答
- **加载方式**：`Qwen3VLForConditionalGeneration.from_pretrained()`
- **使用场景**：独立的图像理解推理（`inference_und.py`）
- **处理器**：`AutoProcessor`（处理图像和文本的tokenization）

#### 3.2 Text Encoder
- **基础架构**：同样使用 Qwen3VL
- **加载代码**：
  ```python
  from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer
  model = Qwen3VLForConditionalGeneration.from_pretrained(...)
  tokenizer = AutoTokenizer.from_pretrained(...)
  ```
- **功能**：提取文本 prompt（或文本+图像）的隐藏状态作为条件嵌入
- **输出**：最后一层的 hidden_states，形状为 `(B, seq_len, 4096)`
- **特殊处理**：
  - 使用 prompt template 包装输入
  - 去掉 template 前缀的 token（`drop_idx`）
  - 对图像编辑任务，使用 `AutoProcessor` 处理图像 token

#### 3.3 VAE Encoder（Encoder3d）
- **输入**：`(B, 3, T, H, W)` RGB图像/视频
- **结构**：初始卷积 → 多级下采样（ResidualBlock + Resample） → 中间层（ResidualBlock + AttentionBlock） → 输出卷积
- **输出**：`(B, 32, T', H/8, W/8)`（mu + log_var 各16通道）
- **下采样配置**：`dim_mult=[1, 2, 4, 4]`，通道数为 `[96, 192, 384, 384]`

#### 3.4 VAE Decoder（Decoder3d）
- **输入**：`(B, 16, T', H/8, W/8)` latent
- **结构**：初始卷积 → 中间层 → 多级上采样（ResidualBlock + Resample） → 输出卷积
- **输出**：`(B, 3, T, H, W)` RGB图像/视频

#### 3.5 MMDiT（Transformer3DModel）
- **参数配置**：
  - `hidden_size=3072`（默认）
  - `heads_num=24`
  - `mm_double_blocks_depth=20`（20层双流块）
  - `patch_size=[1, 2, 2]`
  - `in_channels=4`（VAE latent_channels，但可通过weight inflation扩展）
  - `text_states_dim=4096`（Qwen3VL hidden_size）
  - `rope_dim_list=[16, 56, 56]`（3D RoPE维度分配：时间16，高度56，宽度56）
- **内部组件**：
  - `img_in`：Conv3d，将latent patchify为序列
  - `condition_embedder`：时间步嵌入 + 文本投影（WanTimeTextImageEmbedding）
  - `double_blocks`：20层 MMDoubleStreamBlock
  - `norm_out` + `proj_out`：最终输出层

#### 3.6 Scheduler（FlowMatchDiscreteScheduler）
- **类型**：Flow Matching 离散调度器
- **求解器**：Euler
- **时间调度**：线性sigma + SD3风格时间位移
- **训练时间步**：1000

---

## 4. 模型网络结构图

### 4.1 整体架构（生成/编辑模式）

```
┌─────────────────────────────────────────────────────────────────────┐
│                        JoyAI-Image 整体架构                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐     ┌──────────────────────────────────────────┐  │
│  │  输入图像     │────▶│  Text Encoder (Qwen3VL)                  │  │
│  │  (可选)       │     │  - 接收文本prompt + 可选的参考图像        │  │
│  └──────────────┘     │  - 输出: prompt_embeds (B, L, 4096)      │  │
│                        │  - 输出: prompt_embeds_mask (B, L)        │  │
│  ┌──────────────┐     └──────────────┬───────────────────────────┘  │
│  │  文本Prompt   │────▶              │                              │
│  └──────────────┘                    │                              │
│                                      ▼                              │
│  ┌──────────────┐     ┌──────────────────────────────────────────┐  │
│  │  输入图像     │────▶│  VAE Encoder (Encoder3d)                  │  │
│  │  (编辑模式)   │     │  - 将参考图像编码为latent                 │  │
│  └──────────────┘     │  - 输出: ref_latent (B, 16, 1, H/8, W/8) │  │
│                        └──────────────┬───────────────────────────┘  │
│                                       │                              │
│         ┌─────────────────────────────┤                              │
│         │                             │                              │
│         ▼                             ▼                              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Denoising Loop (FlowMatchDiscreteScheduler)                 │   │
│  │  ┌────────────────────────────────────────────────────────┐  │   │
│  │  │  MMDiT (Transformer3DModel, 16B)                       │  │   │
│  │  │  - hidden_states: latent_model_input (含噪声latent)     │  │   │
│  │  │  - timestep: 当前时间步                                 │  │   │
│  │  │  - encoder_hidden_states: prompt_embeds                 │  │   │
│  │  │  - 20层 MMDoubleStreamBlock                             │  │   │
│  │  │  - 输出: noise_pred (预测噪声)                          │  │   │
│  │  └────────────────────────────────────────────────────────┘  │   │
│  │  CFG: noise = uncond + scale * (cond - uncond)               │   │
│  │  Euler Step: latents = latents + noise_pred * dt             │   │
│  └──────────────────────────┬───────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  VAE Decoder (Decoder3d)                                     │   │
│  │  - 将去噪后的latent解码为RGB图像                             │   │
│  │  - 输出: (B, 3, T, H, W) RGB图像                            │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│                        输出图像 (PIL Image)                          │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 子网络连接关系

```
文本Prompt ──────────────────────────────────────────┐
                                                      │
输入图像（编辑模式）──┬──────────────────────────────┐ │
                      │                              │ │
                      ▼                              ▼ ▼
              ┌──────────────┐              ┌──────────────────┐
              │  VAE Encoder │              │  Text Encoder    │
              │  (Encoder3d) │              │  (Qwen3VL)       │
              └──────┬───────┘              └────────┬─────────┘
                     │                               │
                     │ ref_latent                     │ prompt_embeds
                     ▼                               ▼
              ┌──────────────────────────────────────────────┐
              │            MMDiT (Transformer3DModel)         │
              │  ┌────────────────────────────────────────┐  │
              │  │ img_in(Conv3d) ──▶ 20×MMDoubleStreamBlock │
              │  │         ▲ timestep_emb  ▲ txt_embeds    │  │
              │  │         │               │               │  │
              │  │   WanTimeTextImage      │               │  │
              │  │    Embedding             │               │  │
              │  └────────────────────────────────────────┘  │
              └──────────────┬───────────────────────────────┘
                             │ denoised_latent
                             ▼
              ┌──────────────────────────────┐
              │       VAE Decoder            │
              │       (Decoder3d)            │
              └──────────────┬───────────────┘
                             │
                             ▼
                        输出图像
```

---

## 5. 文生图与图像编辑能力

### 结论：JoyAI-Image 既能实现文生图，也能实现图像+文本提示进行图像编辑。

### 5.1 文生图能力

**代码证据（`inference.py`）：**

```python
parser.add_argument('--image', help='Optional input image path for image editing.')
# image参数是可选的，不提供时为文生图模式
```

**`model.py` 中的 `infer()` 方法：**
```python
if params.image is None:
    # 文生图模式
    prompts = [params.prompt]
    negative_prompt = [params.neg_prompt]
    images = None
    height = params.height
    width = params.width
```

当不提供输入图像时，pipeline接收 `images=None`，此时：
- 不进行 VAE 编码
- 从纯高斯噪声开始去噪
- 只使用文本 prompt 作为条件

README中也确认了这一功能：
> `--image` | str | None | Input image path (required for editing, omit for T2I)

### 5.2 图像+文本编辑能力

**代码证据（`model.py`）：**
```python
else:
    # 图像编辑模式
    processed = _dynamic_resize_from_bucket(params.image, basesize=params.basesize)
    width, height = processed.size
    image_tokens = '<image>\n'
    prompts = [f"<|im_start|>user\n{image_tokens}{params.prompt}<|im_end|>\n"]
    negative_prompt = [f"<|im_start|>user\n{image_tokens}{params.neg_prompt}<|im_end|>\n"]
    images = [processed]
```

编辑模式下：
- 输入图像被resize到bucket尺寸
- 文本prompt中插入`<image>`标记，让Text Encoder处理图像
- 输入图像通过VAE编码为参考latent
- 参考latent与噪声latent拼接（multi-item机制）

---

## 6. 文生图流程图

```
═══════════════════════════════════════════════════════════════════
                    文生图（Text-to-Image）完整流程
═══════════════════════════════════════════════════════════════════

【输入数据】
  ├── 文本Prompt: str (如 "A beautiful sunset over mountains")
  ├── 负面Prompt: str (可选)
  ├── 目标高度: int (如 1024)
  ├── 目标宽度: int (如 1024)
  ├── 推理步数: int (如 50)
  ├── Guidance Scale: float (如 5.0)
  └── 随机种子: int

                              │
                              ▼
═══════════════ 第1步：文本编码 ═══════════════

  文本Prompt ──▶ Prompt Template包装
                    │
                    │ "<|im_start|>system\n...Describe the image...<|im_end|>\n
                    │  <|im_start|>user\n{prompt}<|im_end|>\n
                    │  <|im_start|>assistant\n"
                    ▼
            ┌──────────────────────┐
            │  Tokenizer (Qwen3VL) │
            │  - 文本tokenize       │
            │  - 填充/截断          │
            └──────────┬───────────┘
                       ▼
            ┌──────────────────────┐
            │  Text Encoder        │
            │  (Qwen3VL)           │
            │  - forward()         │
            │  - 提取最后一层       │
            │    hidden_states     │
            └──────────┬───────────┘
                       │
                       ▼
              prompt_embeds: (1, L, 4096)
              prompt_embeds_mask: (1, L)

  [CFG: 同样编码负面prompt，得到 negative_prompt_embeds]
  [拼接: prompt_embeds = cat(neg_embeds, pos_embeds)]

                              │
                              ▼
═══════════════ 第2步：准备噪声Latent ═══════════════

            ┌──────────────────────────┐
            │  随机噪声采样             │
            │  shape = (1, 1, 16,      │
            │    (H/8), (W/8))         │
            │  = (1, 1, 16, 1, 128,128)│
            └──────────┬───────────────┘
                       │
                       ▼
              latents: (1, 1, 16, 1, H/8, W/8)

                              │
                              ▼
═══════════════ 第3步：去噪循环 ═══════════════

  for t in timesteps (如50步):
    │
    │  latents reshape ──▶ (B, C, T, H, W) = (1, 16, 1, 128, 128)
    │
    │  [CFG: latent_model_input = cat(latents, latents)]
    │
    ├──▶ ┌──────────────────────────────────────────┐
    │    │  MMDiT (Transformer3DModel, 16B)          │
    │    │                                            │
    │    │  1. img_in(Conv3d): latent → img tokens    │
    │    │     (B, C, T, H, W) → (B, T*H*W, hidden)  │
    │    │                                            │
    │    │  2. condition_embedder:                     │
    │    │     timestep → temb, vec                    │
    │    │     prompt_embeds → txt                     │
    │    │                                            │
    │    │  3. 3D RoPE位置编码:                        │
    │    │     vis_freqs_cis = get_rotary_pos_embed()  │
    │    │                                            │
    │    │  4. 20× MMDoubleStreamBlock:                │
    │    │     img, txt = block(img, txt, vec,         │
    │    │                      vis_freqs_cis)         │
    │    │     - 图像流: modulate → QKV → RoPE         │
    │    │     - 文本流: modulate → QKV                 │
    │    │     - 联合注意力: cat(img_q, txt_q) ...      │
    │    │     - 拆分 → 残差 → FFN                      │
    │    │                                            │
    │    │  5. proj_out(norm_out(img)) → noise_pred    │
    │    │     unpatchify → (B, C, T, H, W)            │
    │    └──────────────────┬───────────────────────────┘
    │                       │
    │                       ▼
    │    noise_pred: (2, 1, 16, 1, H/8, W/8)
    │
    │  [CFG: noise = uncond + scale * (cond - uncond)]
    │  [Norm rescaling: noise *= (cond_norm / noise_norm)]
    │
    │  [Euler Step: latents = latents + noise_pred * dt]
    │
    └──▶ 继续下一步...

                              │
                              ▼
═══════════════ 第4步：VAE解码 ═══════════════

  denoised_latents: (1, 1, 16, 1, H/8, W/8)
          │
          │  reshape → (1, 16, 1, H/8, W/8)
          ▼
  ┌──────────────────────────┐
  │  VAE Decoder (Decoder3d) │
  │  - 反归一化               │
  │  - conv2 → 解码器主体     │
  │  - 上采样 + 残差块        │
  │  - 输出卷积               │
  └──────────┬───────────────┘
             │
             ▼
  image: (1, 3, 1, H, W)  →  归一化到[0,1]

                              │
                              ▼
═══════════════ 第5步：输出 ═══════════════

【输出数据】
  └── PIL.Image: (H, W, 3) RGB图像
      - 由 VAE Decoder 子网络输出
      - 经过 (image / 2 + 0.5).clamp(0, 1) 归一化
      - 转为 uint8 保存为PNG
```

---

## 7. 图像编辑流程图

```
═══════════════════════════════════════════════════════════════════
              图像+文本提示编辑（Image Editing）完整流程
═══════════════════════════════════════════════════════════════════

【输入数据】
  ├── 输入图像: PIL.Image (如 test_images/test_1.jpg)
  ├── 编辑Prompt: str (如 "Turn the plate blue")
  ├── 负面Prompt: str (可选)
  ├── basesize: int (如 1024，用于图像resize)
  ├── 推理步数: int (如 30)
  ├── Guidance Scale: float (如 4.0)
  └── 随机种子: int

                              │
                              ▼
═══════════════ 第1步：图像预处理 ═══════════════

  输入图像 ──▶ _dynamic_resize_from_bucket()
               │
               │  根据 basesize 找到最佳 bucket 尺寸
               │  等比缩放 + 中心裁剪到目标尺寸
               ▼
          processed_image: PIL.Image (target_H × target_W)

                              │
                              ▼
═══════════════ 第2步：构建编辑Prompt ═══════════════

  prompt = "<|im_start|>user\n<image>\n{edit_prompt}<|im_end|>\n"
  neg_prompt = "<|im_start|>user\n<image>\n{neg_prompt}<|im_end|>\n"
  images = [processed_image]

                              │
                              ▼
═══════════════ 第3步：文本+图像联合编码 ═══════════════

  prompt ──▶ 替换 <image>\n 为
             <|vision_start|><|image_pad|><|vision_end|>
          │
          ▼
  ┌──────────────────────────────────────────┐
  │  Qwen Processor (AutoProcessor)           │
  │  - 处理文本tokenization                    │
  │  - 处理图像pixel值                         │
  │  - 生成 input_ids, attention_mask,         │
  │    pixel_values, image_grid_thw            │
  └──────────────┬───────────────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────────────┐
  │  Text Encoder (Qwen3VL)                   │
  │  - forward(input_ids, attention_mask,      │
  │            pixel_values, ...)              │
  │  - Qwen3VL内部:                            │
  │    * ViT处理图像 → 视觉token嵌入            │
  │    * LLM处理文本+视觉token → hidden_states  │
  │  - 提取最后一层hidden_states               │
  │  - 去掉template前缀 (drop_idx=34)          │
  └──────────────┬───────────────────────────┘
                 │
                 ▼
  prompt_embeds: (1, L, 4096)  -- 包含文本+视觉信息
  prompt_embeds_mask: (1, L)

  [CFG: 同样编码负面prompt（同样包含图像token）]
  [拼接: prompt_embeds = cat(neg_embeds, pos_embeds)]

                              │
                              ▼
═══════════════ 第4步：参考图像VAE编码 + 噪声准备 ═══════════════

  processed_image ──▶ 转为tensor, 归一化到[-1, 1]
                      reshape为 (1, 3, 1, H, W)
          │
          ▼
  ┌──────────────────────────┐
  │  VAE Encoder (Encoder3d) │
  │  - encode()               │
  │  - 重参数化采样            │
  │  - mean/std归一化          │
  └──────────┬───────────────┘
             │
             ▼
  ref_latent: (1, 1, 16, 1, H/8, W/8)  [num_items=2: 1个参考+1个目标]

  随机噪声: (1, 1, 16, 1, H/8, W/8)

  latents = cat(ref_latent, noise) → (1, 2, 16, 1, H/8, W/8)
  [ref_latents保存副本用于每步恢复]

                              │
                              ▼
═══════════════ 第5步：去噪循环 ═══════════════

  for t in timesteps (如30步):
    │
    │  恢复参考latent: latents[:, :1] = ref_latents
    │
    │  latents reshape:
    │    (B, n, C, T, H, W) → (B, C, n*T, H, W) = (1, 16, 2, H/8, W/8)
    │    [注意：最后一个item(目标)被移到前面]
    │
    │  [CFG: latent_model_input = cat(latents, latents)]
    │
    ├──▶ ┌──────────────────────────────────────────────┐
    │    │  MMDiT (Transformer3DModel, 16B)              │
    │    │                                                │
    │    │  1. img_in(Conv3d): latent → img tokens        │
    │    │     (B, C, 2, H/8, W/8)                        │
    │    │     → (B, 2*H/8/2*W/8/2, hidden_size)          │
    │    │     [patch_size=[1,2,2], 空间2×2 patchify]      │
    │    │                                                │
    │    │  2. condition_embedder:                         │
    │    │     timestep → temb, vec (6*hidden_size)        │
    │    │     prompt_embeds → txt (4096 → hidden_size)    │
    │    │                                                │
    │    │  3. 3D RoPE位置编码:                            │
    │    │     vis_freqs_cis 包含时间(2)+空间维度信息       │
    │    │                                                │
    │    │  4. 20× MMDoubleStreamBlock:                    │
    │    │     [图像流和文本流分别处理]                      │
    │    │     img_q/k: 应用3D RoPE                        │
    │    │     联合注意力: cat(img, txt)做attention         │
    │    │     拆分 → 门控残差连接 → FFN                     │
    │    │                                                │
    │    │  5. proj_out → unpatchify → noise_pred          │
    │    │     恢复 multi-item 顺序                         │
    │    └──────────────────┬───────────────────────────────┘
    │                       │
    │                       ▼
    │    noise_pred: (2, 2, 16, 1, H/8, W/8)
    │
    │  [CFG: noise = uncond + scale * (cond - uncond)]
    │  [Norm rescaling]
    │
    │  [Euler Step: latents = latents + noise_pred * dt]
    │  [注意: 只更新目标item的latent, 参考item每步恢复]
    │
    └──▶ 继续下一步...

                              │
                              ▼
═══════════════ 第6步：VAE解码 ═══════════════

  denoised_latents: (1, 2, 16, 1, H/8, W/8)
          │
          │  reshape → (2, 16, 1, H/8, W/8)
          │  [2个item分别解码]
          ▼
  ┌──────────────────────────┐
  │  VAE Decoder (Decoder3d) │
  │  - 反归一化               │
  │  - 解码主体               │
  │  - 上采样 + 残差块        │
  │  - 输出卷积               │
  └──────────┬───────────────┘
             │
             ▼
  images: (1, 2, 3, 1, H, W)  [item 0=参考, item 1=编辑结果]
                              │
                              │  取最后一个item的最后一帧:
                              │  output[0, -1, 0] = 编辑后的图像
                              ▼
═══════════════ 第7步：输出 ═══════════════

【输出数据】
  └── PIL.Image: (H, W, 3) RGB编辑后图像
      - 由 VAE Decoder 子网络输出
      - 取 multi-item 的最后一个item（目标item）
      - 经过归一化和uint8转换
      - 保存为PNG
```

---

## 8. 相比 FLUX2 模型的创新点与改进点

### 8.1 VAE 架构：3D 因果卷积 VAE vs 2D VAE

| 对比维度 | FLUX2 | JoyAI-Image |
|---------|-------|-------------|
| VAE类型 | 2D VAE（标准SD风格） | **3D因果卷积VAE（WanVAE）** |
| 卷积类型 | `Conv2d` | **`CausalConv3d`** |
| 时间维度 | 不支持 | **支持（时间下采样因子4）** |
| latent通道 | 32（patchify后128） | **16** |
| 归一化 | `GroupNorm` + `BatchNorm2d` patchify | **`RMS_norm` + 均值/标准差归一化** |
| 适用范围 | 仅2D图像 | **统一处理图像和视频** |

**创新意义**：JoyAI-Image使用3D因果卷积VAE，天然支持视频帧的时间维度编解码，为未来的视频编辑扩展奠定基础。FLUX2的2D VAE只能处理单帧图像。

### 8.2 DIT 架构：纯双流 MMDiT vs 双流+单流混合

| 对比维度 | FLUX2 | JoyAI-Image |
|---------|-------|-------------|
| 双流块数量 | 8层 `DoubleStreamBlock` | **20层 `MMDoubleStreamBlock`** |
| 单流块数量 | 48层 `SingleStreamBlock` | **0层（无单流块）** |
| 总深度 | 56层（8+48） | **20层（纯双流）** |
| 架构风格 | FLUX风格（双流→单流） | **WanX风格（纯双流）** |

**创新意义**：JoyAI-Image采用纯双流架构，图像和文本在所有层都保持独立的表示空间，仅在注意力计算时交互。相比FLUX2的单流块（将图像和文本concat为统一序列），纯双流设计可能更有利于保持各模态特征的独立性。

### 8.3 文本编码器：多模态VLM vs 纯文本LLM

| 对比维度 | FLUX2 | JoyAI-Image |
|---------|-------|-------------|
| 文本编码器 | Mistral-Small-3.2-24B / Qwen3-FP8（纯文本LLM） | **Qwen3VL（视觉-语言多模态模型）** |
| 图像理解 | 不支持（编码器不处理图像） | **支持（Qwen3VL的ViT处理参考图像）** |
| 隐藏层提取 | 多层concat：`[layer10, layer20, layer30]` → `(B, L, 3*D)` | **最后一层：`hidden_states[-1]`** |
| 编辑条件 | 通过KV-cache注入参考图像的latent | **通过VLM直接理解参考图像内容** |

**创新意义**：JoyAI-Image使用Qwen3VL作为文本编码器，可以在语义层面理解参考图像的内容、空间关系和结构，然后将这些高级语义信息作为条件送入DIT。FLUX2的编码器只处理文本，参考图像仅以latent形式通过KV-cache注入，缺乏语义理解。

### 8.4 调制机制：WanX可学习参数表 vs MLP投影

| 对比维度 | FLUX2 | JoyAI-Image |
|---------|-------|-------------|
| 调制方式 | `Modulation`类：`Linear(dim, 6*dim)` MLP投影 | **`ModulateWan`类：可学习参数表** |
| 调制计算 | `out = Linear(SiLU(vec))` | **`out = modulate_table + x`** |
| 参数存储 | 每层独立的调制MLP | **全局共享的调制MLP + 每块可学习偏移表** |
| 设计理念 | 动态生成调制参数 | **基准偏移 + 条件调整** |

**创新意义**：WanX的调制机制使用可学习的参数表作为基准偏移，时间步条件通过加法叠加，而非FLUX2的MLP全量生成。这种设计更轻量且参数效率更高。

### 8.5 位置编码：3D RoPE vs 4D RoPE

| 对比维度 | FLUX2 | JoyAI-Image |
|---------|-------|-------------|
| RoPE维度 | 4维：`axes_dim=[32,32,32,32]`（总128维） | **3维：`rope_dim_list=[16,56,56]`（总128维）** |
| 轴含义 | 时间+3D空间 | **时间+高度+宽度** |
| theta | 2000 | **256** |
| 文本RoPE | 文本也使用4D RoPE（pe_ctx） | **文本不使用RoPE（可选mrope模式）** |

**创新意义**：JoyAI-Image的RoPE将更多维度分配给空间（56+56=112维），时间仅16维，更适合以图像编辑为主的任务。同时使用更小的theta=256，提供更细粒度的位置编码。

### 8.6 图像编辑机制：Multi-Item vs KV-Cache参考

| 对比维度 | FLUX2 | JoyAI-Image |
|---------|-------|-------------|
| 参考图像处理 | KV-cache：第一步提取参考图像KV，后续步复用 | **Multi-item：参考latent全程参与** |
| 注意力模式 | 因果注意力（参考token只自注意） | **全注意力（参考和目标互相注意）** |
| 参考图像编码 | 直接encode为latent，无语义处理 | **双路径：1) VLM语义理解 2) VAE latent** |
| 时间步处理 | 参考token使用固定timestep=0 | **参考latent每步恢复原始值** |

**创新意义**：JoyAI-Image的Multi-item机制让参考图像的latent在每个去噪步都完整参与注意力计算（而非仅用KV-cache），提供更丰富的参考信息。同时，VLM编码器从语义层面理解参考图像，两条路径互补。

### 8.7 统一的理解-生成-编辑闭环

| 对比维度 | FLUX2 | JoyAI-Image |
|---------|-------|-------------|
| 图像理解 | 不支持 | **✅ 独立的MLLM理解模块** |
| 文生图 | ✅ | ✅ |
| 图像编辑 | ✅（通过参考图像KV注入） | **✅（通过VLM语义+multi-item）** |
| 空间编辑 | 不支持 | **✅ 物体移动、旋转、相机控制** |
| 闭环协同 | 无 | **理解↔生成↔编辑三方协同** |

**创新意义**：JoyAI-Image提出了理解-生成-编辑的闭环协同框架，空间理解能力增强生成质量，生成的多视角图像又反过来增强空间推理能力。

### 8.8 空间智能（Spatial Intelligence）

| 对比维度 | FLUX2 | JoyAI-Image |
|---------|-------|-------------|
| 空间理解 | 不支持 | **✅ 空间关系推理** |
| 物体移动 | 不支持 | **✅ 支持（红框引导）** |
| 物体旋转 | 不支持 | **✅ 支持（8个视角方向）** |
| 相机控制 | 不支持 | **✅ 支持（Yaw/Pitch/Zoom）** |
| 多视图生成 | 不支持 | **✅ 支持** |
| 3D一致性 | 不涉及 | **✅ 多视图3D一致性** |

**创新意义**：这是JoyAI-Image最核心的创新点。通过SpatialEdit数据和训练策略，模型获得了精确的空间编辑能力，包括物体移动、旋转和相机控制，这些是FLUX2完全不具备的。

### 8.9 CFG噪声归一化

| 对比维度 | FLUX2 | JoyAI-Image |
|---------|-------|-------------|
| CFG策略 | 标准CFG：`uncond + scale * (cond - uncond)` | **CFG + 条件范数归一化** |

**代码证据（`pipeline.py`）：**
```python
noise_pred = noise_pred_uncond + self.guidance_scale * (
    noise_pred_text - noise_pred_uncond
)
cond_norm = torch.norm(noise_pred_text, dim=2, keepdim=True)
noise_norm = torch.norm(noise_pred, dim=2, keepdim=True)
noise_pred = noise_pred * (cond_norm / noise_norm)  # 归一化
```

**创新意义**：在标准CFG基础上添加了条件范数归一化，将CFG后的噪声预测缩放到与条件预测相同的范数，避免过度放大导致的图像质量下降。

### 8.10 其他技术差异

| 对比维度 | FLUX2 | JoyAI-Image |
|---------|-------|-------------|
| 模型框架 | 自定义框架 | **基于diffusers + transformers** |
| 注意力实现 | PyTorch `scaled_dot_product_attention` | **Flash Attention（`flash_attn_varlen_func`）** |
| QK归一化 | `QKNorm`（可学习RMSNorm） | **`RMSNorm`（可学习，但实现略有不同）** |
| MLP激活 | `SiLUActivation`（SiLU门控） | **GELU-approximate** |
| 输入投影 | `nn.Linear`（1D） | **`nn.Conv3d`（3D卷积patchify）** |
| FSDP支持 | 不涉及 | **✅ 完整的FSDP2推理支持** |
| 多GPU推理 | 不涉及 | **✅ HSDP分片推理** |
| Prompt重写 | 内置upsampling | **可选的LLM prompt重写** |
| guidance embed | ✅（独立guidance嵌入） | **❌（不使用）** |

---

## 总结

JoyAI-Image是一个**统一的多模态基础模型**，将图像理解（8B MLLM）、文本到图像生成和指令引导的图像编辑整合在一个框架中。其核心创新在于：

1. **3D因果卷积VAE（WanVAE）**：统一处理图像和视频
2. **纯双流MMDiT架构**：20层双流块，无单流块
3. **Qwen3VL多模态文本编码器**：语义级图像理解作为编辑条件
4. **Multi-item编辑机制**：参考图像全程参与注意力计算
5. **WanX调制机制**：可学习参数表的轻量调制
6. **空间智能**：物体移动、旋转、相机控制等精确空间编辑
7. **理解-生成-编辑闭环**：三个任务互相增强
8. **CFG噪声归一化**：提升生成质量
9. **3D RoPE位置编码**：适配图像编辑的维度分配

相比FLUX2，JoyAI-Image在空间编辑能力、多模态理解、视频兼容性等方面具有显著优势，代表了图像生成和编辑领域的一个重要进展。
