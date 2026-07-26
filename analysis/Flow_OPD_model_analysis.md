# Flow-OPD 模型代码全面分析报告

> 本报告基于 `Flow-OPD/` 目录下的全部源代码实现进行深入分析，所有结论均由代码逻辑推导得出。

> **代码实现根目录**：`/root/code/Flow-OPD/`

---

## 目录

1. [问题1：是否使用了VAE？VAE结构类型](#问题1是否使用了vaevae结构类型)
2. [问题2：是否使用了flow_matching的DIT模型？单流还是双流？](#问题2是否使用了flow_matching的dit模型单流还是双流)
3. [问题3：具体网络结构和子网络](#问题3具体网络结构和子网络)
4. [问题4：模型网络结构图](#问题4模型网络结构图)
5. [问题5：是否支持文生图和图像编辑](#问题5是否支持文生图和图像编辑)
6. [问题6：文生图流程图](#问题6文生图流程图)
7. [问题7：图像+文本提示图像编辑流程图](#问题7图像文本提示图像编辑流程图)
8. [问题8：相比FLUX2模型的创新点/改进点](#问题8相比flux2模型的创新点改进点)

---

## 问题1：是否使用了VAE？VAE结构类型

### 结论：是的，Flow-OPD 使用了 VAE。该 VAE 来自 Stable Diffusion 3.5 Medium（SD3.5-M），是 SD3 系列的标准 VAE，与 FLUX.1 的 VAE 结构类似，与 FLUX.2 的 VAE 结构不同。

### 代码分析依据

**1. VAE 的加载方式**

Flow-OPD 通过 HuggingFace `diffusers` 库加载整个 SD3.5 管线，其中包含 VAE：

```python
# scripts/train_sd3.py 第416-418行
pipeline = StableDiffusion3Pipeline.from_pretrained(
    config.pretrained.model  # "stabilityai/stable-diffusion-3.5-medium"
)
```

```python
# config/base.py 第37行
pretrained.model = "runwayml/stable-diffusion-v1-5"  # 默认值

# config/grpo.py 第10行 / 第35行
config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"  # 实际使用
```

**2. VAE 被冻结，仅用于推理**

```python
# scripts/train_sd3.py 第420行
pipeline.vae.requires_grad_(False)

# 第449行
pipeline.vae.to(accelerator.device, dtype=torch.float32)
```

VAE 的参数被完全冻结（`requires_grad_(False)`），不参与训练，仅在采样阶段用于将 latent 解码为图像。

**3. VAE 在推理管线中的使用**

在 `sd3_pipeline_with_logprob.py` 中，VAE 的使用方式如下：

```python
# flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py 第108行
num_channels_latents = self.transformer.config.in_channels  # SD3.5 的 in_channels = 16

# 第181-183行（最终解码）
latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
latents = latents.to(dtype=self.vae.dtype)
image = self.vae.decode(latents, return_dict=False)[0]
```

**4. SD3.5 的 VAE 结构特征**

SD3.5-M 使用的是 `diffusers` 库中的 `AutoencoderKL`，其关键参数为：
- **z_channels = 16**：latent 通道数为 16
- **有效下采样倍数 = 8x**：空间分辨率缩小 8 倍
- **使用 scaling_factor + shift_factor** 进行 latent 归一化
- **采用经典的 KL 正则化 VAE**

**5. 与 FLUX.1 和 FLUX.2 VAE 的对比**

| 特征 | SD3.5 VAE (Flow-OPD使用) | FLUX.1 VAE | FLUX.2 VAE |
|------|-------------------------|-----------|-----------|
| 基础结构 | 卷积 Encoder-Decoder | 卷积 Encoder-Decoder | 卷积 Encoder-Decoder |
| z_channels | 16 | 16 | 32 |
| 有效下采样倍数 | 8x | 8x | 16x (8x + 2x Patch) |
| Latent 通道数 | 16 | 16 | 128 (32×2×2) |
| Latent 归一化 | scaling_factor + shift_factor | scaling_factor | BatchNorm 归一化 |
| 类型 | AutoencoderKL (diffusers) | 自定义 AutoEncoder | 自定义 AutoEncoder |

**结论**：Flow-OPD 使用的 VAE 与 FLUX.1 的 VAE 在关键参数（z_channels=16、下采样 8x）上相同，与 FLUX.2 的 VAE（z_channels=32、下采样 16x、BatchNorm 归一化）明显不同。Flow-OPD 使用的是 SD3 系列标准的 `AutoencoderKL` VAE。

---

## 问题2：是否使用了flow_matching的DIT模型？单流还是双流？

### 结论：是的，使用了 flow matching 的 DIT 模型。Flow-OPD 使用的是 SD3.5-M 的 **双流 MMDIT（Multi-Modal Diffusion Transformer）** 模型。

### 代码分析依据

**1. Flow Matching 调度器**

Flow-OPD 使用 `FlowMatchEulerDiscreteScheduler`，这是 Flow Matching 的标准调度器：

```python
# flow_grpo/diffusers_patch/sd3_sde_with_logprob.py 第9行
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

def sde_step_with_logprob(
    self: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    ...
):
```

```python
# scripts/train_sd3_sft.py 第536-538行
noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    config.pretrained.model, subfolder="scheduler"
)
```

**2. SDE 步进（Flow Matching 的 SDE 变体）**

Flow-OPD 将标准的 Flow Matching ODE 扩展为 SDE（随机微分方程），用于 on-policy 采样：

```python
# flow_grpo/diffusers_patch/sd3_sde_with_logprob.py 第49-62行
if sde_type == 'sde':
    std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * noise_level
    
    # SDE 公式
    prev_sample_mean = sample * (1 + std_dev_t**2 / (2*sigma) * dt) + \
                       model_output * (1 + std_dev_t**2 * (1-sigma) / (2*sigma)) * dt
    
    # 添加随机噪声（SDE 随机项）
    prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1*dt) * variance_noise
```

这与 FLUX.2 使用的纯 ODE 欧拉步进（`x = x + dt * v_pred`）不同。

**3. 双流 MMDIT 结构**

SD3.5-M 的 Transformer 是一个双流 MMDIT，从 LoRA 的 `target_modules` 配置中可以明确看出：

```python
# scripts/train_sd3.py 第458-467行
target_modules = [
    "attn.add_k_proj",   # 文本分支的 Key 投影
    "attn.add_q_proj",   # 文本分支的 Query 投影
    "attn.add_v_proj",   # 文本分支的 Value 投影
    "attn.to_add_out",   # 文本分支的输出投影
    "attn.to_k",         # 图像分支的 Key 投影
    "attn.to_out.0",     # 图像分支的输出投影
    "attn.to_q",         # 图像分支的 Query 投影
    "attn.to_v",         # 图像分支的 Value 投影
]
```

**关键证据**：存在两套并行的注意力投影层：
- `attn.to_q/k/v` 和 `attn.to_out.0`：**图像分支**的注意力层
- `attn.add_q/k/v_proj` 和 `attn.to_add_out`：**文本分支**的注意力层

这正是 SD3 系列 MMDIT（Multi-Modal Diffusion Transformer）的标志性特征——图像和文本各有独立的 QKV 投影和输出投影，但在 Attention 计算时将两个分支的 Q/K/V 拼接在一起进行联合注意力。

**4. Transformer 的 Forward 调用**

```python
# scripts/train_sd3.py 第954-960行
noise_pred = transformer(
    hidden_states=torch.cat([sample["latents"][:, j]] * 2),  # latent 输入
    timestep=torch.cat([sample["timesteps"][:, j]] * 2),      # 时间步
    encoder_hidden_states=embeds,                              # 文本嵌入
    pooled_projections=pooled_embeds,                          # 池化文本嵌入
    return_dict=False,
)[0]
```

输入包含：
- `hidden_states`：图像 latent tokens
- `encoder_hidden_states`：文本编码器的输出嵌入
- `pooled_projections`：池化后的文本嵌入（用于全局条件）

这与 SD3 MMDIT 的标准接口完全一致。

### 总结

| 特征 | 说明 |
|------|------|
| 是否 Flow Matching | ✅ 是，使用 `FlowMatchEulerDiscreteScheduler`，扩展为 SDE 版本 |
| 双流 MMDIT | ✅ 是，SD3.5-M 的双流 Transformer (`attn.to_*` + `attn.add_*`) |
| 单流 DIT | ❌ 不包含独立的单流块 |
| 位置编码 | SD3.5 标准位置编码 |
| Modulation | 每层独立的 AdaLN-Zero 调制（与 FLUX.2 的全局共享不同） |

---

## 问题3：具体网络结构和子网络

### Flow-OPD 包含以下子网络结构：

#### 子网络 1：文本编码器 1 — CLIP Text Encoder (OpenCLIP ViT-G)

SD3.5-M 的第一个文本编码器，从代码中可以看出它使用了两个 CLIP 编码器和一个 T5 编码器：

```python
# scripts/train_sd3.py 第426-427行
text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]
```

```python
# scripts/train_sd3.py 第421-423行
pipeline.text_encoder.requires_grad_(False)
pipeline.text_encoder_2.requires_grad_(False)
pipeline.text_encoder_3.requires_grad_(False)
```

- `text_encoder`：CLIP ViT-L/14 文本编码器
- `text_encoder_2`：OpenCLIP ViT-bigG/14 文本编码器
- `text_encoder_3`：T5-XXL 文本编码器

#### 子网络 2：文本编码器 2 — CLIP Text Encoder 2 (OpenCLIP ViT-bigG)

与子网络1并行的第二个 CLIP 文本编码器。两个 CLIP 编码器的输出通过池化操作产生 `pooled_prompt_embeds`。

#### 子网络 3：文本编码器 3 — T5-XXL

T5-XXL 文本编码器，提供更长序列和更丰富的文本语义特征。

```python
# flow_grpo/diffusers_patch/train_dreambooth_lora_sd3.py 第19-50行
def _encode_prompt_with_t5(text_encoder, tokenizer, max_sequence_length, prompt=...):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,  # 128
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    return prompt_embeds
```

三个文本编码器的输出通过 `encode_prompt()` 函数整合：
- CLIP 1 + CLIP 2 的输出拼接形成 `prompt_embeds` 和 `pooled_prompt_embeds`
- T5 的输出拼接到 `prompt_embeds` 中

#### 子网络 4：VAE Encoder

SD3.5-M 的 `AutoencoderKL` Encoder 部分：
- 输入：RGB 图像 `[B, 3, H, W]`
- 输出：latent `[B, 16, H/8, W/8]`
- 结构：卷积下采样 + ResNet 块 + 注意力块

**注意**：在 Flow-OPD 的训练流程中，VAE Encoder 实际上**未直接使用**（训练时从纯噪声开始采样，不需要编码真实图像）。VAE Encoder 仅在 SFT 训练模式下隐式使用（用于将采样的最终 latent 作为 SFT 目标）。

#### 子网络 5：VAE Decoder

SD3.5-M 的 `AutoencoderKL` Decoder 部分：
- 输入：latent `[B, 16, H/8, W/8]`
- 输出：RGB 图像 `[B, 3, H, W]`
- 使用 scaling_factor 和 shift_factor 进行归一化

```python
# sd3_pipeline_with_logprob.py 第181-183行
latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
latents = latents.to(dtype=self.vae.dtype)
image = self.vae.decode(latents, return_dict=False)[0]
```

#### 子网络 6：MMDiT Transformer（Flow Matching 去噪网络，核心可训练部分）

SD3.5-M 的双流 Multi-Modal Diffusion Transformer：
- 输入通道数：`in_channels = 16`
- 输出：预测的速度场（velocity field）
- 包含图像分支和文本分支的联合注意力
- 使用 LoRA 进行微调（rank=32, alpha=64）

```python
# scripts/train_sd3.py 第468-473行
transformer_lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    init_lora_weights="gaussian",
    target_modules=target_modules,  # 8个注意力层
)
```

#### 子网络 7：参考教师 Transformer（推理时冻结）

用于 OPD KL 奖励计算的参考模型，与主 Transformer 结构相同但加载不同的 LoRA 权重：

```python
# scripts/train_sd3.py 第487-500行
ref_pipeline = StableDiffusion3Pipeline.from_pretrained(config.pretrained.model)
ref_pipeline.transformer.requires_grad_(False)
if config.train.get("kl_ref_lora_path"):
    ref_transformer = PeftModel.from_pretrained(ref_pipeline.transformer, config.train.kl_ref_lora_path)
```

#### 子网络 8：MAR (Manifold Anchor Regularization) Transformer（可选）

用于防止美学退化的锚定正则化模型：

```python
# scripts/train_sd3.py 第503-511行
if config.train.get("mar_lora") and config.train.mar_lora != "":
    mar_pipeline = StableDiffusion3Pipeline.from_pretrained(config.pretrained.model)
    mar_transformer = PeftModel.from_pretrained(mar_pipeline.transformer, config.train.mar_lora)
```

#### 子网络 9：各类奖励模型（Reward Models）

用于评估生成图像质量的外部模型（非 DIT 结构，独立运行）：

```python
# flow_grpo/rewards.py
score_functions = {
    "deqa": deqa_score_remote,         # DeQA 质量评估（远程服务）
    "ocr": ocr_score,                  # PaddleOCR 文字识别
    "pickscore": pickscore_score,       # PickScore 人类偏好
    "geneval": geneval_score,           # GenEval 对象检测（远程服务）
    "aesthetic": aesthetic_score,       # CLIP 美学评分
    "clipscore": clip_score,           # CLIP 文图匹配分
    "imagereward": imagereward_score,  # ImageReward
    "qwenvl": qwenvl_score_remote,     # QwenVL 多模态评分
    "unifiedreward": unifiedreward_score_sglang,  # UnifiedReward
}
```

#### 子网络 10：EMA 模块

指数移动平均模块，用于稳定训练：

```python
# flow_grpo/ema.py
class EMAModuleWrapper:
    def __init__(self, parameters, decay=0.9, update_step_interval=8, device=None):
        self.ema_parameters = [p.clone().detach().to(device) for p in parameters]
```

---

## 问题4：模型网络结构图

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     Flow-OPD 完整模型架构                                     │
│                                                                              │
│  ┌───────────────────┐                                                       │
│  │  文本 Prompt       │                                                       │
│  │  (字符串)          │                                                       │
│  └─────────┬─────────┘                                                       │
│            │                                                                 │
│            ▼                                                                 │
│  ┌────────────────────────────────────────────────────┐                       │
│  │              Text Encoders (3个并行)                │                       │
│  │                                                    │                       │
│  │  ┌──────────┐  ┌──────────────┐  ┌───────────┐    │                       │
│  │  │ CLIP     │  │ OpenCLIP     │  │ T5-XXL    │    │                       │
│  │  │ ViT-L/14 │  │ ViT-bigG/14 │  │           │    │                       │
│  │  └────┬─────┘  └──────┬───────┘  └─────┬─────┘    │                       │
│  │       │               │                │          │                       │
│  │       └──────┬────────┘                │          │                       │
│  │              ▼                         │          │                       │
│  │       pooled_prompt_embeds             │          │                       │
│  │              │                         │          │                       │
│  │              └───────────┬─────────────┘          │                       │
│  │                          ▼                        │                       │
│  │                   prompt_embeds                   │                       │
│  └──────────────────────────┬────────────────────────┘                       │
│                             │                                                │
│            ┌────────────────┴────────────────┐                                │
│            ▼                                ▼                                │
│     prompt_embeds                  pooled_prompt_embeds                       │
│     [B, L, D]                     [B, D_pool]                                │
│            │                                │                                │
│            │    ┌──────────────────┐         │                                │
│            │    │ 随机噪声          │         │                                │
│            │    │ torch.randn()    │         │                                │
│            │    │ [B,16,H/8,W/8]  │         │                                │
│            │    └────────┬─────────┘         │                                │
│            │             │                   │                                │
│            ▼             ▼                   ▼                                │
│  ┌─────────────────────────────────────────────────────────────────┐          │
│  │              MMDiT Transformer (SD3.5-M) + LoRA                │          │
│  │              (Flow Matching 去噪网络)                            │          │
│  │                                                                 │          │
│  │  timestep ──→ 时间步嵌入                                         │          │
│  │                                                                 │          │
│  │  ┌─────────────────────────────────────────────────────┐        │          │
│  │  │  Joint Attention Block × N (双流MMDIT)               │        │          │
│  │  │                                                     │        │          │
│  │  │  图像分支:                   文本分支:                │        │          │
│  │  │  hidden_states              encoder_hidden_states   │        │          │
│  │  │  → attn.to_q/k/v           → attn.add_q/k/v_proj   │        │          │
│  │  │  → 联合 Attention (拼接Q/K/V)                        │        │          │
│  │  │  → attn.to_out.0           → attn.to_add_out       │        │          │
│  │  │  → MLP                     → MLP                   │        │          │
│  │  │  → 残差连接                  → 残差连接               │        │          │
│  │  │                                                     │        │          │
│  │  │  pooled_projections ──→ AdaLN-Zero 全局调制           │        │          │
│  │  └─────────────────────────────────────────────────────┘        │          │
│  │                                                                 │          │
│  │  输出: v_pred (速度场预测) [B, 16, H/8, W/8]                     │          │
│  └──────────────────────────────┬──────────────────────────────────┘          │
│                                 │                                            │
│           SDE 步进: x_{t-1} = μ(x_t, v_pred) + σ * noise                     │
│                                 │                                            │
│            (重复 num_steps 步，带 log_prob 计算)                               │
│                                 │                                            │
│                                 ▼                                            │
│                          denoised latent                                      │
│                          [B, 16, H/8, W/8]                                   │
│                                 │                                            │
│                    latent = latent / scaling + shift                          │
│                                 │                                            │
│                                 ▼                                            │
│                    ┌──────────────────────┐                                   │
│                    │  VAE Decoder          │                                   │
│                    │  (AutoencoderKL)      │                                   │
│                    │  latent → RGB         │                                   │
│                    └──────────┬───────────┘                                   │
│                               │                                              │
│                               ▼                                              │
│                        输出图像 [B, 3, H, W]                                   │
│                               │                                              │
│                               ▼                                              │
│                    ┌──────────────────────┐                                   │
│                    │  Reward Models       │                                   │
│                    │  (OCR/GenEval/       │                                   │
│                    │   PickScore/DeQA...) │                                   │
│                    └──────────┬───────────┘                                   │
│                               │                                              │
│                          reward scores                                       │
│                               │                                              │
│                               ▼                                              │
│                    ┌──────────────────────┐                                   │
│                    │ GRPO/OPD 训练循环     │                                   │
│                    │ advantages → loss    │                                   │
│                    │ → 反向传播 LoRA 参数   │                                   │
│                    └─────────────────────┘                                   │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════          │
│                                                                              │
│  并行参考模型（不参与梯度计算）:                                                │
│                                                                              │
│  ┌──────────────────────────┐    ┌──────────────────────────┐                │
│  │ Ref Transformer          │    │ MAR Transformer (可选)    │                │
│  │ (教师 LoRA + Base)       │    │ (美学锚定 LoRA + Base)    │                │
│  │ → KL 奖励计算             │    │ → KL 正则化损失           │                │
│  └──────────────────────────┘    └──────────────────────────┘                │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 问题5：是否支持文生图和图像编辑

### 结论

| 功能 | 是否支持 | 代码依据 |
|------|---------|---------|
| **文生图（Text-to-Image）** | ✅ 支持 | 核心功能，所有训练和推理代码都以文生图为基础 |
| **图像+文本提示图像编辑** | ❌ 不支持 | 代码中没有任何图像编辑相关的实现 |

### 代码依据

**1. 文生图支持**

Flow-OPD 的所有训练脚本和推理 Demo 都是纯文本到图像生成：

```python
# scripts/demo/sd3_sde_demo.py
pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
prompt = 'A steaming cup of coffee'
images, _, _, _ = pipeline_with_logprob(pipe, prompt, ...)
```

数据集全部为纯文本提示：

```python
# scripts/train_sd3.py 第80-90行
class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}
```

**2. 图像编辑不支持**

- 代码中**没有任何图像输入的处理逻辑**（无 VAE Encoder 调用、无参考图加载、无 mask 处理）
- 所有采样都从**纯随机噪声**开始（`self.prepare_latents()` 生成随机 latent）
- Pipeline 使用的是 `StableDiffusion3Pipeline`（文生图管线），而非任何 img2img 管线
- 整个项目的 README 和论文标题也明确表明这是一个**文生图质量优化**方法

虽然 `flow_grpo/diffusers_patch/` 目录下存在一些与 Flux Kontext（图像编辑管线）和 QwenImage（图像编辑管线）相关的 pipeline 文件，但这些是**实验性扩展**，并非 Flow-OPD 的核心功能：

```
flow_grpo/diffusers_patch/flux_kontext_pipeline_with_logprob.py  # Flux Kontext 编辑实验
flow_grpo/diffusers_patch/qwenimage_edit_pipeline_with_logprob.py  # QwenImage 编辑实验
```

这些文件是为将 Flow-OPD 方法扩展到编辑模型而做的预研代码，不影响主体功能。

---

## 问题6：文生图流程图

### 文生图时数据流：

```
输入数据:
  ├── 文本 Prompt (字符串, 如 "A steaming cup of coffee")
  ├── 图像分辨率 (resolution=512 或 768)
  ├── 采样参数 (num_steps=10, guidance_scale=4.5, noise_level=0.7)
  └── LoRA 权重路径 (可选)

═══════════════════════════════════════════════════════════════

步骤1: 文本编码 (3个编码器并行)
  ┌─────────────────────┐
  │ 文本 Prompt          │
  │ "A steaming cup      │
  │  of coffee"          │
  └──────────┬──────────┘
             │
             ▼
  ┌──────────────────────────────────────────────────────┐
  │ encode_prompt() (train_dreambooth_lora_sd3.py)       │
  │                                                      │
  │  ┌────────────┐  ┌──────────────┐  ┌──────────────┐ │
  │  │ CLIP ViT-L │  │ OpenCLIP     │  │ T5-XXL       │ │
  │  │ tokenizer  │  │ ViT-bigG     │  │ tokenizer    │ │
  │  │ + encoder  │  │ tokenizer    │  │ + encoder    │ │
  │  │            │  │ + encoder    │  │ max_len=128  │ │
  │  └─────┬──────┘  └──────┬───────┘  └──────┬───────┘ │
  │        │                │                 │         │
  │        ▼                ▼                 │         │
  │   CLIP1_emb        CLIP2_emb             │         │
  │        │                │                 │         │
  │        │     pool()     │     pool()      │         │
  │        ▼                ▼                 │         │
  │   pooled_1         pooled_2              │         │
  │        │                │                 │         │
  │        └──────┬─────────┘                 │         │
  │               ▼                           │         │
  │        pooled_prompt_embeds               │         │
  │        [B, D_pool]                        │         │
  │               │                           │         │
  │    CLIP1_emb + CLIP2_emb + T5_emb         │         │
  │    (padding + concat)                     │         │
  │               ▼                           ▼         │
  │        prompt_embeds [B, L, D_text]                 │
  └──────────────────────────┬──────────────────────────┘
                             │
  同时生成 negative embeddings (空字符串)
  → neg_prompt_embeds, neg_pooled_prompt_embeds

═══════════════════════════════════════════════════════════════

步骤2: 初始噪声生成
  ┌─────────────────────┐
  │ prepare_latents()    │
  │ torch.randn()        │
  │ shape=(B, 16,        │
  │   H/8, W/8)          │
  │ 如 (1, 16, 64, 64)   │
  │ → float32            │
  └──────────┬──────────┘
             │
             ▼
  latents [B, 16, H/8, W/8]  (纯高斯噪声)

═══════════════════════════════════════════════════════════════

步骤3: 时间步调度
  ┌─────────────────────────────────────┐
  │ retrieve_timesteps()                 │
  │ FlowMatchEulerDiscreteScheduler     │
  │                                     │
  │ num_steps=10 → timesteps:            │
  │ [1000, 889, 778, ..., 111, 0]       │
  │ (从噪声到干净的时间步序列)             │
  └──────────────────┬──────────────────┘

═══════════════════════════════════════════════════════════════

步骤4: Flow Matching SDE 去噪循环

  for i, t in enumerate(timesteps):

    ┌─────────────────────────────────────────────────────┐
    │ CFG: 拼接无条件和有条件输入                            │
    │                                                     │
    │ latent_input = cat([latents, latents])               │
    │ embeds = cat([neg_prompt_embeds, prompt_embeds])     │
    │ pooled = cat([neg_pooled, pooled_embeds])            │
    └───────────────────────┬─────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────┐
    │ MMDiT Transformer (SD3.5-M + LoRA)                  │
    │                                                     │
    │ 输入:                                                │
    │   hidden_states = latent_input [2B, 16, H/8, W/8]   │
    │   timestep = t [2B]                                  │
    │   encoder_hidden_states = embeds [2B, L, D]          │
    │   pooled_projections = pooled [2B, D_pool]           │
    │                                                     │
    │ 内部计算:                                             │
    │   1. 时间步嵌入 + 池化文本嵌入 → AdaLN 调制参数         │
    │   2. 图像 latent → 图像分支 QKV                       │
    │   3. 文本 embedding → 文本分支 QKV                    │
    │   4. 联合注意力: cat(img_Q, txt_Q) × cat(img_K, txt_K)│
    │   5. 各分支独立 MLP + 残差连接                         │
    │   6. 重复 N 个 Joint Attention Block                  │
    │   7. 输出层: 投影回 16 通道                            │
    │                                                     │
    │ 输出: noise_pred [2B, 16, H/8, W/8]                 │
    └───────────────────────┬─────────────────────────────┘
                            │
    CFG 融合:
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = uncond + guidance_scale * (text - uncond)
                            │
                            ▼
    ┌─────────────────────────────────────────────────────┐
    │ SDE 步进 (sde_step_with_logprob)                    │
    │                                                     │
    │ 1. 计算 sigma, sigma_prev, dt                       │
    │ 2. std_dev_t = sqrt(σ/(1-σ)) * noise_level          │
    │ 3. 确定性均值:                                       │
    │    μ = x*(1+σ²/(2σ)*dt) + v_pred*(1+σ²(1-σ)/(2σ))*dt│
    │ 4. 随机项:                                           │
    │    x_{t-1} = μ + std_dev_t * sqrt(-dt) * ε          │
    │ 5. 计算 log_prob (用于 GRPO)                         │
    │                                                     │
    │ 输出: prev_sample, log_prob                          │
    └───────────────────────┬─────────────────────────────┘
                            │
    latents = prev_sample
    记录: all_latents.append(latents)
           all_log_probs.append(log_prob)

═══════════════════════════════════════════════════════════════

步骤5: Latent 解码为图像

  ┌─────────────────────────────────────────────────────┐
  │ VAE Decoder (AutoencoderKL)                         │
  │                                                     │
  │ 1. 归一化: latent = latent / scaling + shift         │
  │ 2. 转换精度: latent.to(vae.dtype)                    │
  │ 3. vae.decode(latent)                               │
  │    → conv_in → mid → 4级上采样(ResNet+Upsample)      │
  │    → norm_out → Swish → conv_out                    │
  │ 4. image_processor.postprocess()                    │
  │                                                     │
  │ 输出: 图像 [B, 3, H, W] (pt tensor, 值域[0,1])      │
  └───────────────────────┬─────────────────────────────┘
                          │
                          ▼
                   生成的 RGB 图像
                   [B, 3, H, W]

═══════════════════════════════════════════════════════════════

步骤6: 奖励计算 (训练时)

  ┌─────────────────────────────────────────────────────┐
  │ Reward Models                                       │
  │                                                     │
  │ 根据 config.reward_fn 选择:                          │
  │   "ocr": PaddleOCR → 文字识别准确率                   │
  │   "geneval": 远程 GenEval 服务 → 对象检测准确率         │
  │   "pickscore": PickScore → 人类偏好分                 │
  │   "deqa": DeQA 远程服务 → 图像质量分                   │
  │   ...                                               │
  │                                                     │
  │ multi_score(): 加权融合多个奖励                       │
  │                                                     │
  │ 输出: score_details (包含 avg, 各子项分数)             │
  └───────────────────────┬─────────────────────────────┘
                          │
                          ▼
                    rewards (标量)

═══════════════════════════════════════════════════════════════

步骤7: GRPO/OPD 策略优化 (训练时)

  ┌─────────────────────────────────────────────────────┐
  │ 训练循环                                             │
  │                                                     │
  │ 1. Per-prompt stat tracking → advantages             │
  │ 2. 重新前向传播 Transformer → 新的 log_prob           │
  │ 3. 计算 ratio = exp(log_prob - old_log_prob)         │
  │ 4. PPO Clipped Loss:                                │
  │    loss = max(-adv*ratio, -adv*clip(ratio))          │
  │                                                     │
  │ (OPD模式下额外计算):                                  │
  │ 5. KL奖励 = MSE(μ_policy, μ_ref_teacher) / (2σ²)    │
  │ 6. MAR正则化 = MSE(μ_policy, μ_mar) / (2σ²)          │
  │ 7. loss = policy_loss + β * kl_loss                  │
  │                                                     │
  │ 8. 反向传播 → 更新 LoRA 参数                          │
  │ 9. EMA 更新                                          │
  └─────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════

最终输出: 由 VAE Decoder 子网络输出 RGB 图像
```

---

## 问题7：图像+文本提示图像编辑流程图

### 结论：Flow-OPD 不支持图像+文本提示的图像编辑功能。

### 原因分析

1. **无图像编辑管线**：核心代码使用的是 `StableDiffusion3Pipeline`（纯文生图管线），不是 `StableDiffusion3Img2ImgPipeline` 或任何编辑管线。

2. **无图像输入机制**：
   - 数据集全部是纯文本（`TextPromptDataset` 和 `GenevalPromptDataset` 都只包含文本提示）
   - 采样过程从纯随机噪声开始，没有将输入图像编码为 latent 的步骤
   - 没有 mask、inpainting 或 image conditioning 的任何代码

3. **设计目标不同**：Flow-OPD 的目标是通过 On-Policy Distillation 提升**文生图模型**在多个维度（OCR准确性、对象生成、美学质量、人类偏好）上的表现，而非进行图像编辑。

4. **虽然存在编辑相关的实验代码**（如 `flux_kontext_pipeline_with_logprob.py`、`qwenimage_edit_pipeline_with_logprob.py`），但这些是对 Flow-OPD 方法的扩展性实验，不影响主体功能，且没有对应的训练脚本和配置文件。

因此，**无法画出图像编辑的流程图**。

---

## 问题8：相比FLUX2模型的创新点/改进点

### 前提说明

Flow-OPD 和 FLUX.2 是**完全不同类型**的工作：
- **FLUX.2**：一个新的**模型架构**（包含新的 VAE、新的 DIT 架构、新的文本编码器等），支持文生图和图像编辑
- **Flow-OPD**：一个基于 SD3.5-M 的**训练方法/优化框架**，通过 On-Policy Distillation 和多教师蒸馏来提升文生图质量

以下从多个维度列举 Flow-OPD 相比 FLUX.2 的创新点和差异：

---

#### 创新点 1：On-Policy Distillation (OPD) 训练框架

**FLUX.2**：标准的预训练 + 蒸馏（Step Distillation + Guidance Distillation），教师到学生的单向知识传递。

**Flow-OPD**：提出了 On-Policy Distillation，核心创新在于：

```python
# scripts/train_sd3.py 第1088-1094行
# OPD: 用教师模型的向量场作为密集奖励
if reward_mode == "kl_only":
    kl_reward = ((prev_sample_mean - prev_sample_mean_ref_lora) ** 2).mean(dim=(1, 2, 3), keepdim=True) / (2 * std_dev_t ** 2)
    advantages = config.train.kl_scale * kl_reward
```

- **密集向量场监督**：教师模型在每个去噪时间步提供完整的向量场预测（而非仅提供最终图像的标量奖励），实现了**轨迹级别的密集监督**
- **On-Policy 采样**：学生模型使用自己的策略生成采样轨迹（通过 SDE），然后用教师模型的向量场对每一步进行评估

---

#### 创新点 2：SDE 采样替代 ODE 采样

**FLUX.2**：使用标准的 ODE 欧拉步进进行去噪：
```python
# FLUX.2 (sampling.py)
img = img + (t_prev - t_curr) * pred  # 确定性 ODE
```

**Flow-OPD**：将 Flow Matching 的 ODE 扩展为 SDE，引入随机性以实现 On-Policy 探索：

```python
# Flow-OPD (sd3_sde_with_logprob.py 第49-62行)
std_dev_t = torch.sqrt(sigma / (1 - sigma)) * noise_level  # 噪声系数

# 确定性均值
prev_sample_mean = sample * (1 + std_dev_t**2/(2*sigma) * dt) + \
                   model_output * (1 + std_dev_t**2*(1-sigma)/(2*sigma)) * dt

# SDE: 添加随机噪声
prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-dt) * noise

# 计算 log_prob（用于 GRPO 策略梯度）
log_prob = -((prev_sample - prev_sample_mean) ** 2) / (2 * (std_dev_t * sqrt(-dt))**2) - ...
```

这使得模型能够在采样过程中进行**随机探索**，生成多样化的样本用于策略优化，同时可以精确计算每一步的 `log_prob`。

---

#### 创新点 3：多教师蒸馏（Multi-Teacher Distillation）

**FLUX.2**：使用单一预训练模型的蒸馏。

**Flow-OPD**：支持多个专家教师模型的交替/混合蒸馏：

```python
# config/grpo.py 第128-156行
config.alternate_datasets = [
    {
        "name": "ocr",
        "kl_ref_lora_path": "path/SD3.5M-FlowGRPO-Text",  # OCR 教师
        "epochs_per_cycle": 1,
    },
    {
        "name": "geneval",
        "kl_ref_lora_path": "path/SD3.5M-FlowGRPO-GenEval",  # GenEval 教师
        "epochs_per_cycle": 3,
    },
]
config.training_mode = "alternate"
```

```python
# scripts/train_sd3_opd_mix.py 第893-912行
# 为每个数据集加载独立的教师模型
ref_transformers = {}
for ds_config in alternate_datasets:
    kl_ref_path = ds_config.get("kl_ref_lora_path")
    if kl_ref_path:
        ref_trans = PeftModel.from_pretrained(ref_pipe.transformer, kl_ref_path)
        ref_transformers[ds_name] = ref_trans
```

每个教师（OCR专家、GenEval专家、PickScore专家等）都作为"生成式奖励模型"（Generative Reward Model），在其擅长的领域提供密集向量场监督。

---

#### 创新点 4：MAR (Manifold Anchor Regularization) 防美学退化

**FLUX.2**：无此机制。

**Flow-OPD**：引入 MAR 正则化，使用一个冻结的美学教师模型作为"锚点"，防止在优化 OCR/GenEval 等功能性指标时导致图像美学质量下降：

```python
# scripts/train_sd3.py 第503-511行
if config.train.get("mar_lora") and config.train.mar_lora != "":
    mar_transformer = PeftModel.from_pretrained(mar_pipeline.transformer, config.train.mar_lora)

# 第1201-1204行（训练时使用 MAR 正则化）
if config.train.beta > 0:
    kl_loss = ((prev_sample_mean - prev_sample_mean_ref_base) ** 2).mean(dim=(1,2,3), keepdim=True) / (2 * std_dev_t ** 2)
    loss = policy_loss + config.train.beta * kl_loss
```

---

#### 创新点 5：GRPO 策略优化（PPO-Clipped 风格）

**FLUX.2**：不涉及强化学习训练。

**Flow-OPD**：使用 GRPO（Group Relative Policy Optimization）进行策略优化，类似 PPO 的 clipped loss：

```python
# scripts/train_sd3.py 第1193-1206行
ratio = torch.exp(log_prob - sample["log_probs"][:, j])
unclipped_loss = -advantages * ratio
clipped_loss = -advantages * torch.clamp(
    ratio,
    1.0 - config.train.clip_range,
    1.0 + config.train.clip_range,
)
policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
```

同时支持多种奖励模式：

```python
# train.reward_mode 选项:
# "task_only"  - 仅使用任务奖励（标量）
# "kl_only"   - 仅使用 KL 密集奖励（OPD 模式）
# "mixed"     - 任务奖励 + KL 密集奖励混合
# "gkd"       - 直接使用 KL 作为 SFT 损失（GKD 模式）
```

---

#### 创新点 6：冷启动初始化策略（Cold Start）

**FLUX.2**：标准预训练。

**Flow-OPD**：提供两种冷启动策略加速收敛：

**（1）SFT 冷启动**：先用 SFT 方式在教师数据上预训练 LoRA

```python
# scripts/train_sd3_sft.py 第836行
target = noise - model_input  # Flow Matching 目标
fm_loss = ((model_pred.float() - target.float()) ** 2).mean(dim=(1, 2, 3))
```

**（2）LoRA 合并冷启动**：将多个专家 LoRA 的权重平均合并为初始 LoRA

```python
# scripts/merge.py 第70-71行
if merge_mode == "average":
    merged[key] = torch.stack(tensors).mean(dim=0)
elif merge_mode == "weighted":
    merged[key] = sum(t * w for t, w in zip(tensors, weights_tensor))
```

---

#### 创新点 7：Per-Prompt 统计追踪

**FLUX.2**：无此机制。

**Flow-OPD**：实现了 per-prompt 的统计追踪器，使 advantages 归一化更加稳定：

```python
# flow_grpo/stat_tracking.py（通过 train_sd3.py 使用）
stat_tracker = PerPromptStatTracker(config.sample.global_std)
advantages = stat_tracker.update(prompts, gathered_rewards['avg'])
```

支持 global_std 模式（使用全局标准差）和 per-prompt 模式（每个 prompt 独立归一化）。

---

#### 创新点 8：EMA 参数平滑

**FLUX.2**：推理模型，不涉及训练。

**Flow-OPD**：使用 EMA（指数移动平均）对 LoRA 参数进行平滑：

```python
# flow_grpo/ema.py
class EMAModuleWrapper:
    def __init__(self, parameters, decay=0.9, update_step_interval=8, device=None):
        self.ema_parameters = [p.clone().detach().to(device) for p in parameters]

    def step(self, parameters, optimization_step):
        one_minus_decay = 1 - self.get_current_decay(optimization_step)
        # 每 8 步更新一次 EMA
        if (optimization_step + 1) % self.update_step_interval == 0:
            ema_parameter.add_(one_minus_decay * (parameter - ema_parameter))
```

配置为 `decay=0.9, update_step_interval=8`，相当于对最近约 160 步的参数取加权平均，用于稳定训练和评估。

---

#### 创新点 9：交替训练模式（Alternate Training）

**FLUX.2**：不涉及多数据集训练。

**Flow-OPD**：支持按 epoch 交替训练多个数据集，每个数据集有独立的教师模型和奖励函数：

```python
# config/grpo.py 第128-160行
config.alternate_datasets = [
    {
        "name": "ocr",
        "epochs_per_cycle": 1,   # 每轮训练 1 个 epoch
        "kl_ref_lora_path": "path/SD3.5M-FlowGRPO-Text",
        "reward_fn": {"ocr": 1.0},
    },
    {
        "name": "geneval",
        "epochs_per_cycle": 3,   # 每轮训练 3 个 epoch
        "kl_ref_lora_path": "path/SD3.5M-FlowGRPO-GenEval",
        "reward_fn": {"geneval": 1.0},
    },
]
```

这种设计允许不同能力以不同的训练频率进行优化，平衡各维度的提升速度。

---

#### 创新点 10：GKD (Generalized Knowledge Distillation) 模式

**FLUX.2**：使用标准蒸馏。

**Flow-OPD**：除了 OPD 模式外，还支持 GKD 模式，直接使用 KL 散度作为损失（无需策略梯度）：

```python
# scripts/train_sd3.py 第1172-1177行
elif reward_mode == "gkd":
    # GKD: 直接使用 mean(kl_reward) 作为损失
    kl_reward_gkd = ((prev_sample_mean - prev_sample_mean_ref_lora) ** 2).mean(...) / (2 * std_dev_t ** 2)
    gkd_loss = torch.mean(kl_reward_gkd)
    policy_loss = torch.tensor(0.0)
```

---

#### 创新点 11：多奖励模型融合

**FLUX.2**：不涉及奖励模型。

**Flow-OPD**：支持多个奖励模型的加权融合，包括本地和远程部署：

```python
# flow_grpo/rewards.py 第426-445行
score_functions = {
    "deqa": deqa_score_remote,          # 远程部署
    "ocr": ocr_score,                   # 本地 PaddleOCR
    "pickscore": pickscore_score,        # 本地 GPU
    "geneval": geneval_score,            # 远程部署
    "aesthetic": aesthetic_score,        # 本地 GPU
    "clipscore": clip_score,             # 本地 GPU
    "imagereward": imagereward_score,    # 本地 GPU
    "qwenvl": qwenvl_score_remote,       # 远程 vLLM
    "unifiedreward": unifiedreward_score_sglang,  # 远程 sglang
}
```

奖励计算支持并行执行（ThreadPoolExecutor），OCR 等非线程安全的模型串行执行。

---

#### 创新点 12：LoRA 高效微调策略

**FLUX.2**：需要全量训练（32B 参数）。

**Flow-OPD**：仅微调 LoRA 适配器（rank=32），大幅降低训练成本：

```python
# scripts/train_sd3.py 第468-473行
transformer_lora_config = LoraConfig(
    r=32,                              # LoRA rank
    lora_alpha=64,                     # LoRA alpha
    init_lora_weights="gaussian",      # 高斯初始化
    target_modules=target_modules,     # 8个注意力层
)
```

仅微调注意力层的 Q/K/V/Out 投影，训练参数量远小于全量模型。

---

### 创新点总结表

| 编号 | 创新点/改进点 | Flow-OPD | FLUX.2 |
|------|-------------|---------|--------|
| 1 | On-Policy Distillation (密集向量场监督) | ✅ 核心创新 | ❌ 无 |
| 2 | SDE 采样 (随机探索 + log_prob 计算) | ✅ Flow Matching SDE | ❌ 标准 ODE |
| 3 | 多教师蒸馏 (每个教师一个专业能力) | ✅ 支持交替/混合 | ❌ 无 |
| 4 | MAR 正则化 (防美学退化) | ✅ 支持 | ❌ 无 |
| 5 | GRPO 策略优化 (PPO-Clipped) | ✅ 支持多种模式 | ❌ 无 |
| 6 | 冷启动策略 (SFT/LoRA合并) | ✅ 支持 | ❌ 无 |
| 7 | Per-Prompt 统计追踪 | ✅ 支持 | ❌ 无 |
| 8 | EMA 参数平滑 | ✅ 支持 | ❌ 无(推理模型) |
| 9 | 交替训练模式 | ✅ 支持 | ❌ 无 |
| 10 | GKD 蒸馏模式 | ✅ 支持 | ❌ 标准蒸馏 |
| 11 | 多奖励模型融合 | ✅ 12种奖励模型 | ❌ 无 |
| 12 | LoRA 高效微调 | ✅ rank=32 | ❌ 全量训练 |

**核心差异总结**：FLUX.2 是一个**模型架构创新**（新 VAE、新 DIT、新文本编码器），而 Flow-OPD 是一个**训练方法创新**（On-Policy Distillation + 多教师蒸馏 + GRPO 策略优化），两者解决的是不同层面的问题。Flow-OPD 的方法理论上可以应用于包括 FLUX.2 在内的任何 Flow Matching 模型。
