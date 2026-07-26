# Z-Image Universal Image Edit 模型代码全面分析报告

> 本报告基于 `SimpleGeneration/universal_image_edit/` 目录下的全部源代码实现进行深入分析，所有结论均由代码逻辑推导得出。

> **代码实现根目录**：`/root/code/SimpleGeneration_pytorch_training_examples/`

---

## 目录

1. [问题1：是否使用了VAE？VAE结构类型](#问题1是否使用了vaevae结构类型)
2. [问题2：是否使用了flow_matching的DIT模型？单流还是双流？](#问题2是否使用了flow_matching的dit模型单流还是双流)
3. [问题3：具体网络结构和子网络](#问题3具体网络结构和子网络)
4. [问题4：模型网络结构图](#问题4模型网络结构图)
5. [问题5：是否支持文生图和图像编辑](#问题5是否支持文生图和图像编辑)
6. [问题6：文生图流程图](#问题6文生图流程图)
7. [问题7：图像+文本提示图像编辑流程图](#问题7图像文本提示图像编辑流程图)
8. [问题8：相比FLUX2模型的创新点和改进点](#问题8相比flux2模型的创新点和改进点)

---

## 问题1：是否使用了VAE？VAE结构类型

### 结论：是的，本模型使用了VAE。使用的是 **FLUX2 的 VAE 结构**（冻结，不参与训练），与 FLUX2 模型的 VAE 完全一致。

### 代码分析依据

**1. 设计文档明确指定 VAE 来源**

`design_doc.md` 第24行明确写道：

```
- **VAE (冻结)**: flux2_autoencoder.py 不做任何改动
```

第366行：

```
4. **VAE不变**: flux2_autoencoder完全冻结，五档模型共享同一VAE
```

**2. VAE 具体结构（`flux2_autoencoder.py`）**

VAE 定义在 `SimpleGeneration/flux_autoencoder/models/flux2_autoencoder.py` 的 `AutoEncoder` 类中，默认参数为：

```python
# flux2_autoencoder.py 第416-423行
# FLUX.2-dev
# inplanes=3,
# planes=128,
# planes_mult=[1, 2, 4, 4],
# res_block_nums=2,
# out_planes=3,
# z_planes=32,
```

这些参数与 FLUX2 原版 VAE **完全一致**：`ch=128`、`ch_mult=[1,2,4,4]`、`num_res_blocks=2`、`z_channels=32`。

**3. Encoder 结构**

```python
# flux2_autoencoder.py 第160-288行
class Encoder(nn.Module):
    def __init__(self, inplanes=3, planes=128, planes_mult=[1, 2, 4, 4],
                 res_block_nums=2, z_planes=32, ...):
```

Encoder 由以下模块组成：
- `conv_in`：3×3 卷积，将 3 通道 RGB 输入映射到 `planes=128` 通道
- 4个下采样层级（由 `planes_mult=[1,2,4,4]` 控制），每个层级包含：
  - 2个 `ResnetBlock`（GroupNorm32 → SiLU → Conv3x3 → GroupNorm32 → SiLU → Conv3x3 + shortcut）
  - `Downsample`（stride=2 的 3×3 卷积），最后一个层级除外
- `mid` 中间层：`ResnetBlock` → `AttnBlock`（使用 `F.scaled_dot_product_attention`） → `ResnetBlock`
- `norm_out` → SiLU → `conv_out`：输出 `2*z_planes=64` 通道（均值和方差）
- `quant_conv`：1×1 卷积（64→64通道）

**4. Decoder 结构**

```python
# flux2_autoencoder.py 第291-413行
class Decoder(nn.Module):
    def __init__(self, z_planes=32, planes=128, planes_mult=[1, 2, 4, 4],
                 res_block_nums=2, out_planes=3, ...):
```

Decoder 是 Encoder 的镜像结构：
- `post_quant_conv`：1×1 卷积（32→32通道）
- `conv_in`：将 `z_planes=32` 通道映射到 `block_in=512`
- `mid` 中间层：`ResnetBlock` → `AttnBlock` → `ResnetBlock`
- 4个上采样层级（反序），每个层级包含：
  - 3个 `ResnetBlock`（`res_block_nums + 1`）
  - `Upsample`（最近邻插值 + 3×3 卷积），第一个层级除外
- `norm_out` → SiLU → `conv_out`：输出 3 通道 RGB 图像

**5. FLUX2 VAE 的关键特性 —— BatchNorm 归一化 + Patch 重排**

```python
# flux2_autoencoder.py 第454-462行
self.ps = [2, 2]
self.bn = nn.BatchNorm2d(math.prod(self.ps) * z_planes,  # 2*2*32 = 128 通道
                         eps=self.bn_eps, momentum=self.bn_momentum,
                         affine=False, track_running_stats=True)
```

编码时的 Patch 重排操作：
```python
# flux2_autoencoder.py 第476-486行
def encode(self, x):
    moments = self.encoder(x)
    mean = torch.chunk(moments, 2, dim=1)[0]  # 只取均值
    z = rearrange(mean, "... c (i pi) (j pj) -> ... (c pi pj) i j", pi=2, pj=2)
    z = self.normalize(z)  # BatchNorm 归一化
    return z
```

这意味着：
- Encoder 硬件下采样 **8x**（3次 stride-2 卷积）
- Patch 操作进一步 **2x**（将 32 通道 × 2×2 patch 折叠为 128 通道）
- **有效总下采样倍数为 16x**
- 输出 latent 形状：`[B, 128, H/16, W/16]`

**6. 与 FLUX1 VAE 的对比**

| 特征 | FLUX1 VAE | FLUX2 VAE（本模型使用） |
|------|-----------|----------------------|
| 基础结构 | 卷积 Encoder-Decoder | 卷积 Encoder-Decoder（相同） |
| z_channels | 16 | **32** |
| 有效下采样倍数 | 8x | **16x**（8x + 2x Patch） |
| Latent 通道数 | 16 | **128**（32×2×2） |
| Latent 归一化 | Scaling factor (scale=0.3611, shift=0.1159) | **BatchNorm 归一化** |
| Encoder 输出 | 取均值 | 取均值（相同） |
| 参数 ch_mult | [1,2,4,4] | [1,2,4,4]（相同） |
| quant_conv | 无 | **有**（1×1 卷积） |

### 最终结论

本模型使用的 VAE **与 FLUX2 模型的 VAE 结构完全一致**，不是 FLUX1 的 VAE，也不是其他类型的 VAE。关键区别在于 FLUX2 VAE 使用了更大的 z_channels=32、BatchNorm 归一化、以及 2×2 Patch 空间重排，使得有效下采样倍数为 16x，latent 通道数为 128。

---

## 问题2：是否使用了flow_matching的DIT模型？单流还是双流？

### 结论：是的，本模型使用了 Flow Matching 的 DIT 模型。本模型提供了**两种 DIT 架构变体**：

1. **双流 MMDIT（Double-Stream + Single-Stream）**：类似 FLUX2 的架构，先经过双流块再经过单流块
2. **纯单流 MMDIT（Full Single-Stream）**：所有层都使用 SingleStreamBlock，参考自 Z-Image 的设计

### 代码分析依据

**1. Flow Matching 训练算法**

`losses.py` 中定义了标准的 Flow Matching 前向加噪和损失计算：

```python
# losses.py 第120-142行
def add_noise_flow_matching(clean_latents, noise, timesteps):
    """z_t = (1 - t) * x_0 + t * epsilon"""
    noisy_latents = (1.0 - t) * clean_latents + t * noise
    return noisy_latents

def compute_flow_matching_loss(model_output, noise, clean_latents, sigmas, ...):
    """target = noise - x_0，损失 = MSE(v_pred, target)"""
    target = noise - clean_latents
    loss = F.mse_loss(model_output.float(), target.float(), reduction="none")
    return loss.mean()
```

这是标准的 **velocity prediction** Flow Matching 算法。

**2. 双流 MMDIT 模型（`double_stream_mmdit.py`）**

```python
# double_stream_mmdit.py 第360-479行
class MMDITModel(nn.Module):
    def __init__(self, in_channels=128, hidden_size=3072, num_heads=24,
                 depth=8, depth_single_blocks=32, ...):
        # Double-stream blocks
        self.double_blocks = nn.ModuleList([
            DoubleStreamBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        # Single-stream blocks
        self.single_blocks = nn.ModuleList([
            SingleStreamBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth_single_blocks)
        ])
```

前向传播流程：
```python
def forward(self, x, x_ids, timesteps, ctx, ctx_ids):
    # 1. 双流块处理
    for block in self.double_blocks:
        img, txt = block(img, txt, vec, pe_img, pe_txt)
    
    # 2. 拼接 txt + img 进入单流块
    img = torch.cat((txt, img), dim=1)
    
    # 3. 单流块处理
    for block in self.single_blocks:
        img = block(img, pe, vec)
    
    # 4. 去掉txt tokens，只保留img tokens
    img = img[:, num_txt_tokens:, ...]
    
    # 5. 最终输出层
    img = self.final_layer(img, vec)
    return img
```

**3. DoubleStreamBlock 的 Joint Attention 机制**

```python
# double_stream_mmdit.py 第245-256行
# Joint attention: concat txt + img
q = torch.cat((txt_q, img_q), dim=2)
k = torch.cat((txt_k, img_k), dim=2)
v = torch.cat((txt_v, img_v), dim=2)

# Apply RoPE
pe = torch.cat((pe_txt, pe_img), dim=2)
q, k = apply_rope(q, k, pe)

# F.scaled_dot_product_attention (auto flash attn with BF16)
attn = F.scaled_dot_product_attention(q, k, v)
```

双流块中，img 和 txt 各自独立做 Modulation/Norm/QKV 投影，但在 Attention 计算时将 Q/K/V **拼接在一起**进行 **Joint Attention**，然后将注意力输出分回各自的残差连接和 MLP。

**4. 纯单流 MMDIT 模型（`single_stream_mmdit.py`）**

```python
# single_stream_mmdit.py 第248-346行
class FullSingleStreamMMDITModel(nn.Module):
    def __init__(self, in_channels=128, hidden_size=3072, num_heads=24,
                 depth=40, ...):
        # 仅使用 Single-stream blocks，无 Double-stream blocks
        self.single_blocks = nn.ModuleList([
            SingleStreamBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
```

前向传播流程：
```python
def forward(self, x, x_ids, timesteps, ctx, ctx_ids):
    # 1. 输入阶段即拼接 txt + img
    x = torch.cat((txt, img), dim=1)
    pe = torch.cat((pe_txt, pe_img), dim=2)
    
    # 2. 所有层都是单流块
    for block in self.single_blocks:
        x = block(x, pe, vec)
    
    # 3. 去掉txt tokens
    x = x[:, num_txt_tokens:, ...]
    
    # 4. 最终输出层
    x = self.final_layer(x, vec)
    return x
```

**5. 五档模型配置**

双流+单流 MMDIT 配置（`double_stream_mmdit.py`）：

| 配置 | hidden_size | num_heads | depth(双流) | depth_single(单流) | mlp_ratio | 参数量 |
|------|------------|-----------|------------|-------------------|-----------|--------|
| 1B | 1536 | 24 | 4 | 16 | 4.0 | ~1.14B |
| 2B | 2048 | 16 | 5 | 18 | 3.0 | ~2.02B |
| 4B | 2560 | 20 | 6 | 24 | 3.0 | ~4.04B |
| 6B | 3072 | 24 | 7 | 25 | 3.0 | ~6.32B |
| 8B | 3456 | 27 | 8 | 26 | 3.0 | ~8.65B |

纯单流 MMDIT 配置（`single_stream_mmdit.py`）：

| 配置 | hidden_size | num_heads | depth(全单流) | mlp_ratio | 参数量 |
|------|------------|-----------|-------------|-----------|--------|
| 1B | 1536 | 24 | 22 | 4.0 | ~1.00B |
| 2B | 2048 | 16 | 30 | 3.0 | ~2.03B |
| 4B | 2560 | 20 | 38 | 3.0 | ~4.01B |
| 6B | 3072 | 24 | 40 | 3.0 | ~6.08B |
| 8B | 3456 | 27 | 42 | 3.0 | ~8.07B |

**6. Flow Matching Euler 调度器**

```python
# scheduler.py 第20-74行
class FlowMatchEulerScheduler:
    def step(self, model_output, timestep, sample):
        """Euler step: x_{t-1} = x_t + (sigma_{t+1} - sigma_t) * v_pred"""
        dt = sigma_next - sigma
        prev_sample = sample + dt * model_output.to(torch.float32)
        return (prev_sample.to(model_output.dtype), )
```

### 总结

| 特征 | 说明 |
|------|------|
| 是否 Flow Matching | ✅ 是，使用 velocity prediction + Euler ODE solver |
| 双流 MMDIT 变体 | ✅ `DoubleStreamBlock × D` → `SingleStreamBlock × S` |
| 纯单流 MMDIT 变体 | ✅ `SingleStreamBlock × depth`（全部单流） |
| 位置编码 | **2D RoPE (h, w)**，矩阵旋转形式，theta=2000 |
| Modulation | **每层独立** AdaLN (6/3参数)，SiLU(vec)→Linear |
| Attention | `F.scaled_dot_product_attention`，QKV 合并 Linear |
| QK-Norm | RMSNorm per head |
| FFN | SiLU-Gated，mlp_ratio=3.0/4.0 |

---

## 问题3：具体网络结构和子网络

### 本模型共包含以下4个子网络结构：

#### 子网络 1：VLM 文本编码器（Qwen3.5-4B，冻结）

**来源**：设计文档指定使用 Qwen3.5-4B

```
# design_doc.md 第23行
- **VLM (冻结)**: Qwen3.5-4B 取最后隐藏层作为文本条件
```

- **模型**：Qwen3.5-4B，一个 4B 参数的大语言模型
- **输出维度**：`hidden_size = 2560`（对应 `context_in_dim=2560`）
- **使用方式**：取最后隐藏层（`hidden_states[-1]`）作为文本 embedding
- **训练时状态**：完全冻结，不参与梯度计算

```python
# train_t2i.py 第76-79行（注释中的实际使用方式）
# vlm_model = AutoModelForCausalLM.from_pretrained(vlm_model_name, ...)
# vlm_out = vlm_model(**inputs, output_hidden_states=True)
# ctx = vlm_out.hidden_states[-1]  # [B, seq, 2560]
```

#### 子网络 2：VAE Encoder（FLUX2 Autoencoder Encoder，冻结）

```
输入图像 [B, 3, H, W]
  → conv_in (3→128, 3×3)
  → Stage 0: ResnetBlock×2 (128→128) + Downsample ↓2
  → Stage 1: ResnetBlock×2 (128→256) + Downsample ↓2
  → Stage 2: ResnetBlock×2 (256→512) + Downsample ↓2
  → Stage 3: ResnetBlock×2 (512→512) [无Downsample]
  → mid: ResnetBlock(512) → AttnBlock(512) → ResnetBlock(512)
  → norm_out(GN32) → SiLU → conv_out(512→64, 3×3)
  → quant_conv(64→64, 1×1)
  → 取均值 chunk(64→32 通道)
  → Patch重排: (32, H/8, W/8) → (128, H/16, W/16)
  → BatchNorm 归一化
输出 latent [B, 128, H/16, W/16]
```

- 有效下采样倍数：**16x**
- 输出通道数：**128**

#### 子网络 3：VAE Decoder（FLUX2 Autoencoder Decoder，冻结）

```
输入 latent [B, 128, H/16, W/16]
  → BatchNorm 反归一化
  → Unpatch重排: (128, H/16, W/16) → (32, H/8, W/8)
  → post_quant_conv(32→32, 1×1)
  → conv_in(32→512, 3×3)
  → mid: ResnetBlock(512) → AttnBlock(512) → ResnetBlock(512)
  → Stage 3: ResnetBlock×3 (512→512) [无Upsample]
  → Stage 2: ResnetBlock×3 (512→512) + Upsample ↑2
  → Stage 1: ResnetBlock×3 (512→256) + Upsample ↑2
  → Stage 0: ResnetBlock×3 (256→128) + Upsample ↑2
  → norm_out(GN32) → SiLU → conv_out(128→3, 3×3)
输出图像 [B, 3, H, W]
```

#### 子网络 4：MMDIT DIT 模型（可训练，核心网络）

**有两种变体：**

**变体A：双流+单流 MMDIT（`MMDITModel`）**

| 组件 | 说明 |
|------|------|
| `pe_embedder` | `EmbedND`：2D RoPE 位置编码（h,w 两轴） |
| `img_in` | `nn.Linear(128, hidden_size, bias=False)`：latent token 投影 |
| `txt_in` | `nn.Linear(2560, hidden_size, bias=False)`：文本 embedding 投影 |
| `time_in` | `MLPTimeStepEmbedder(256, hidden_size)`：时间步嵌入 |
| `double_blocks` | `DoubleStreamBlock × depth`：双流块 |
| `single_blocks` | `SingleStreamBlock × depth_single`：单流块 |
| `final_layer` | `LastLayer`：AdaLN + Linear 输出 |

**变体B：纯单流 MMDIT（`FullSingleStreamMMDITModel`）**

| 组件 | 说明 |
|------|------|
| `pe_embedder` | `EmbedND`：2D RoPE 位置编码 |
| `img_in` | `nn.Linear(128, hidden_size, bias=False)` |
| `txt_in` | `nn.Linear(2560, hidden_size, bias=False)` |
| `time_in` | `MLPTimeStepEmbedder(256, hidden_size)` |
| `single_blocks` | `SingleStreamBlock × depth`：全部单流块 |
| `final_layer` | `LastLayer`：AdaLN + Linear 输出 |

---

## 问题4：模型网络结构图

### 变体A：双流+单流 MMDIT 架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Universal Image Edit 完整模型架构                     │
│                      （双流+单流 MMDIT 变体）                             │
│                                                                         │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────────┐   │
│  │  输入图像     │    │  文本 Prompt      │    │  参考图像（编辑模式） │   │
│  │  [3,H,W]     │    │  (字符串)         │    │  [3,H',W'] × N      │   │
│  └──────┬───────┘    └────────┬─────────┘    └────────┬─────────────┘   │
│         │                     │                       │                 │
│         ▼                     ▼                       ▼                 │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────────┐   │
│  │  子网络2:     │    │  子网络1:         │    │  子网络2:             │   │
│  │  VAE Encoder │    │  VLM             │    │  VAE Encoder         │   │
│  │  (FLUX2,冻结) │    │  (Qwen3.5-4B,   │    │  (FLUX2,冻结,共享)    │   │
│  │              │    │   冻结)           │    │                      │   │
│  └──────┬───────┘    └────────┬─────────┘    └────────┬─────────────┘   │
│         │                     │                       │                 │
│         ▼                     ▼                       ▼                 │
│   latent               text embedding           ref latent             │
│   [128,H/16,W/16]      [B,N_txt,2560]          [128,H'/16,W'/16]×N   │
│         │                     │                       │                 │
│   flatten→tokens         txt_in投影              flatten→tokens         │
│   [B,h*w,128]           [B,N_txt,hidden]         [B,N_ref,128]         │
│         │                     │                       │                 │
│   加噪(仅训练时)              │                  保持干净                │
│   z_t=(1-t)x_0+t·ε          │                       │                 │
│         │                     │                       │                 │
│    img_in投影                 │              img_in投影(编辑时拼接到img) │
│         └─────────┬───────────┘───────────────────────┘                 │
│                   │                                                     │
│                   ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │              子网络4: MMDIT DIT（可训练）                         │    │
│  │                                                                 │    │
│  │  timestep t ──→ timestep_embedding(256d) ──→ time_in ──→ vec    │    │
│  │                                                                 │    │
│  │  img_ids ──→ pe_embedder ──→ pe_img (2D RoPE)                   │    │
│  │  txt_ids ──→ pe_embedder ──→ pe_txt (2D RoPE)                   │    │
│  │                                                                 │    │
│  │  ┌───────────────────────────────────┐                          │    │
│  │  │  DoubleStreamBlock × D            │                          │    │
│  │  │  ┌─────────────┐ ┌─────────────┐ │                          │    │
│  │  │  │ img 分支:    │ │ txt 分支:    │ │                          │    │
│  │  │  │ AdaLN调制    │ │ AdaLN调制    │ │                          │    │
│  │  │  │ LayerNorm   │ │ LayerNorm   │ │                          │    │
│  │  │  │ QKV投影     │ │ QKV投影     │ │                          │    │
│  │  │  │ QK-RMSNorm  │ │ QK-RMSNorm  │ │                          │    │
│  │  │  └──────┬──────┘ └──────┬──────┘ │                          │    │
│  │  │         └──── Joint Attention ────┘                          │    │
│  │  │              (concat QKV → SDPA                              │    │
│  │  │               → split → proj)                                │    │
│  │  │  img: + gate*attn + gate*MLP                                 │    │
│  │  │  txt: + gate*attn + gate*MLP                                 │    │
│  │  └───────────────────────────────────┘                          │    │
│  │                     │                                           │    │
│  │            cat(txt, img) 拼接                                    │    │
│  │                     ▼                                           │    │
│  │  ┌───────────────────────────────────┐                          │    │
│  │  │  SingleStreamBlock × S            │                          │    │
│  │  │  AdaLN调制 → LayerNorm             │                          │    │
│  │  │  → linear1 → split(QKV, MLP_in)   │                          │    │
│  │  │  → QK-RMSNorm → RoPE → SDPA       │                          │    │
│  │  │  → cat(attn, SiLU_gate(MLP_in))   │                          │    │
│  │  │  → linear2 → x + gate*output      │                          │    │
│  │  └───────────────────────────────────┘                          │    │
│  │                     │                                           │    │
│  │         去掉txt tokens，保留img tokens                           │    │
│  │                     ▼                                           │    │
│  │  ┌───────────────────────────────────┐                          │    │
│  │  │  LastLayer (final_layer)          │                          │    │
│  │  │  AdaLN调制(shift,scale)            │                          │    │
│  │  │  → LayerNorm → Linear(hidden→128) │                          │    │
│  │  └───────────────────────────────────┘                          │    │
│  │                     │                                           │    │
│  └─────────────────────┼───────────────────────────────────────────┘    │
│                        │                                                │
│                   v_pred [B, N_img, 128]                                │
│                   (velocity prediction)                                 │
│                        │                                                │
│       训练时: loss = MSE(v_pred, noise - x_0)                           │
│       推理时: Euler step: x_{t-dt} = x_t + dt * v_pred                 │
│              (重复 num_steps 步)                                        │
│                        │                                                │
│                        ▼                                                │
│              reshape: [B,h*w,128] → [B,128,h,w]                        │
│                        │                                                │
│                        ▼                                                │
│              ┌──────────────────────┐                                   │
│              │  子网络3:             │                                   │
│              │  VAE Decoder         │                                   │
│              │  (FLUX2,冻结)        │                                   │
│              └────────┬─────────────┘                                   │
│                       │                                                 │
│                       ▼                                                 │
│                 输出图像 [B, 3, H, W]                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 变体B：纯单流 MMDIT 架构

```
┌─────────────────────────────────────────────────────┐
│         Universal Image Edit 完整模型架构             │
│          （纯单流 MMDIT 变体）                        │
│                                                      │
│  输入图像 ──→ [VAE Encoder(冻结)] ──→ img tokens     │
│  文本Prompt ──→ [VLM(冻结)] ──→ txt embedding        │
│  参考图(可选) ──→ [VAE Encoder(冻结)] ──→ ref tokens │
│                                                      │
│  timestep t ──→ [Timestep Embedding] ──→ vec         │
│                                                      │
│         cat(txt, img [+ ref]) 直接拼接                │
│                     │                                │
│                     ▼                                │
│         [SingleStreamBlock × depth]                  │
│         (所有层统一处理拼接后的序列)                    │
│                     │                                │
│                     ▼                                │
│         去掉txt tokens                               │
│                     │                                │
│                     ▼                                │
│         [LastLayer (AdaLN + Linear)]                 │
│                     │                                │
│                     ▼                                │
│         v_pred → Euler去噪                           │
│                     │                                │
│                     ▼                                │
│         [VAE Decoder(冻结)] ──→ 输出图像              │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## 问题5：是否支持文生图和图像编辑

### 结论

| 功能 | 是否支持 | 代码依据 |
|------|---------|---------|
| **文生图（Text-to-Image）** | ✅ 支持 | `pipeline/train_t2i.py` + `pipeline/inference_t2i.py` |
| **图像+文本提示编辑（Image Editing）** | ✅ 支持 | `pipeline/train_edit.py` + `pipeline/inference_edit.py` |

### 代码依据

**1. 文生图支持**

训练Pipeline（`train_t2i.py`）：
```python
# train_t2i.py 第45-206行
def train_t2i(model_size="1B", ...):
    """文生图训练Pipeline
    流程：
    1. 加载冻结的VLM (Qwen3.5-4B)
    2. 加载冻结的VAE (flux2 autoencoder)
    3. 构建可训练的MMDIT模型
    4. 训练循环
    """
```

推理Pipeline（`inference_t2i.py`）：
```python
# inference_t2i.py 第24-185行
@torch.no_grad()
def inference_t2i(model, vae, vlm_model, vlm_tokenizer, prompt, ...):
    """文生图推理Pipeline"""
    # 1. VLM编码文本
    # 2. 初始化纯噪声
    # 3. Euler ODE去噪循环 (支持CFG)
    # 4. VAE解码
```

**2. 图像编辑支持**

训练Pipeline（`train_edit.py`）：
```python
# train_edit.py 第65-235行
def train_edit(model_size="1B", ...):
    """扩展任务(编辑)训练Pipeline
    关键设计：
    - 参考图通过VAE编码后拼接到img token序列
    - 位置ID中参考图使用独立的空间坐标（不与目标图重叠）
    - 仅对目标图token计算loss
    """
```

推理Pipeline（`inference_edit.py`）：
```python
# inference_edit.py 第26-217行
@torch.no_grad()
def inference_edit(model, vae, vlm_model, vlm_tokenizer, prompt,
                   source_images, ...):
    """图像编辑推理Pipeline
    支持：inpainting、局部改动、风格迁移、多图参考输入编辑等
    """
```

**3. 统一多任务设计（`design_doc.md`）**

```
### 多任务统一
- **Text-to-Image**: 无参考图，等价于纯文生图
- **Inpainting**: 参考图 = 原图，文本描述修改区域（无需mask）
- **Style Transfer**: 参考图 = 风格图，文本描述风格指令
- **Multi-ref Edit**: 多张参考图拼接，模型通过attention自动融合
```

关键在于：**同一个 MMDIT 模型同时支持文生图和编辑**，不需要修改模型结构。区别仅在于是否在 img token 序列中拼接参考图的 latent tokens。

---

## 问题6：文生图流程图

### 文生图时的完整数据流：

```
═══════════════════════════════════════════════════════════════════
输入数据:
  ├── 文本 Prompt (字符串，如 "A beautiful photo of scenery")
  ├── 图像尺寸 (width=512, height=512)
  └── 采样参数 (num_steps=30, guidance_scale=7.5, seed)
═══════════════════════════════════════════════════════════════════

步骤1: 文本编码 → 子网络1: VLM (Qwen3.5-4B, 冻结)
  ┌───────────────────────┐
  │ 文本 Prompt            │
  │ "A beautiful photo..." │
  └──────────┬────────────┘
             │
             ▼
  ┌─────────────────────────────────────┐
  │ VLM: Qwen3.5-4B (冻结)              │
  │                                     │
  │ 1. Tokenize 文本                     │
  │ 2. Forward pass (无梯度)             │
  │ 3. 提取最后隐藏层 hidden_states[-1]   │
  │                                     │
  │ 输出: ctx [B, N_txt, 2560]          │
  └──────────┬──────────────────────────┘
             │
             ▼
  ctx [B, 64, 2560]（文本 embedding）
  ctx_ids [B, 64, 2]（文本位置 ID）

═══════════════════════════════════════════════════════════════════

步骤2: 训练时 — VAE编码图像 → 子网络2: VAE Encoder (FLUX2, 冻结)
  ┌───────────────────────┐
  │ 训练图像               │
  │ [B, 3, 512, 512]      │
  └──────────┬────────────┘
             │
             ▼
  ┌──────────────────────────────────────┐
  │ VAE Encoder (FLUX2, 冻结)             │
  │                                      │
  │ 1. Encoder: conv→down×3→mid          │
  │    → [B, 32, 64, 64]                 │
  │ 2. quant_conv: [B, 64, 64, 64]       │
  │ 3. 取均值: [B, 32, 64, 64]           │
  │ 4. Patch重排: [B, 128, 32, 32]       │
  │ 5. BatchNorm归一化                    │
  │                                      │
  │ 输出: latent [B, 128, 32, 32]        │
  └──────────┬───────────────────────────┘
             │
             ▼
  flatten: latent [B, 1024, 128]（h*w = 32*32 = 1024 tokens）
  img_ids [B, 1024, 2]（图像位置 ID）

  推理时: 直接生成纯高斯噪声 z_1 ~ N(0,I)，shape=[B, 1024, 128]

═══════════════════════════════════════════════════════════════════

步骤3: 训练时 — 采样时间步 + 加噪
  ┌──────────────────────────────────────┐
  │ 1. t ~ logit_normal(0, 1)           │
  │    t = sigmoid(N(0,1)), t ∈ (0,1)   │
  │                                      │
  │ 2. noise ~ N(0, I), 同shape          │
  │                                      │
  │ 3. z_t = (1-t) * x_0 + t * noise    │
  │    (Flow Matching 线性插值加噪)       │
  │                                      │
  │ 4. target = noise - x_0             │
  │    (velocity prediction 目标)        │
  └──────────┬───────────────────────────┘
             │
             ▼
  noisy_latents [B, 1024, 128]

═══════════════════════════════════════════════════════════════════

步骤4: MMDIT 前向 → 子网络4: MMDIT DIT (可训练)
  ┌──────────────────────────────────────────────────────┐
  │ MMDITModel.forward()                                │
  │                                                      │
  │ 输入:                                                 │
  │   x = noisy_latents [B, 1024, 128]                  │
  │   x_ids = img位置ID [B, 1024, 2]                    │
  │   timesteps = t [B] (0~1)                           │
  │   ctx = text_emb [B, 64, 2560]                     │
  │   ctx_ids = txt位置ID [B, 64, 2]                    │
  │                                                      │
  │ 内部计算:                                             │
  │   1. vec = time_in(timestep_embedding(t, 256))      │
  │   2. img = img_in(x)  → [B, 1024, hidden]          │
  │   3. txt = txt_in(ctx) → [B, 64, hidden]           │
  │   4. pe_img = pe_embedder(x_ids) [2D RoPE]         │
  │   5. pe_txt = pe_embedder(ctx_ids) [2D RoPE]       │
  │   6. D × DoubleStreamBlock(img, txt, vec, pe)       │
  │   7. cat(txt, img) → S × SingleStreamBlock          │
  │   8. 去掉 txt tokens                                │
  │   9. final_layer(img, vec) → v_pred                 │
  │                                                      │
  │ 输出: v_pred [B, 1024, 128]                         │
  └──────────┬───────────────────────────────────────────┘
             │
             ▼

═══════════════════════════════════════════════════════════════════

步骤5: 训练时 — 计算 Loss + 反向传播
  loss = MSE(v_pred, noise - x_0)  [标量]
  → optimizer.zero_grad()
  → loss.backward()
  → clip_grad_norm_(1.0)
  → optimizer.step()

  推理时 — Euler ODE 去噪循环 (CFG 可选):
  for t in schedule:
    v_pred = model(z_t, ids, t, ctx, ctx_ids)
    if CFG:
      v = v_uncond + scale * (v_cond - v_uncond)
    z_{t-dt} = z_t + dt * v_pred
  (重复 num_steps 步)

═══════════════════════════════════════════════════════════════════

步骤6: 推理时 — VAE解码 → 子网络3: VAE Decoder (FLUX2, 冻结)
  ┌──────────────────────────────────────┐
  │ 去噪完成后的 latent                    │
  │ [B, 1024, 128]                       │
  │                                      │
  │ reshape: [B, 128, 32, 32]            │
  └──────────┬───────────────────────────┘
             │
             ▼
  ┌──────────────────────────────────────┐
  │ VAE Decoder (FLUX2, 冻结)             │
  │                                      │
  │ 1. inv_normalize (BatchNorm反归一化)  │
  │ 2. unpatch: (128,32,32)→(32,64,64)  │
  │ 3. post_quant_conv(32→32)            │
  │ 4. conv_in → mid → up×3             │
  │ 5. norm_out → SiLU → conv_out       │
  │                                      │
  │ 输出: 图像 [B, 3, 512, 512]          │
  └──────────┬───────────────────────────┘
             │
             ▼
  ┌──────────────────────────────────────┐
  │ 后处理                                │
  │ 1. clamp(0, 1)                       │
  │ 2. 转换为 PIL Image                   │
  │ 3. 保存为 PNG                         │
  └──────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════

最终输出: 由 **子网络3: VAE Decoder** 输出 RGB 图像 [B, 3, H, W]
```

---

## 问题7：图像+文本提示图像编辑流程图

### 图像编辑时的完整数据流：

```
═══════════════════════════════════════════════════════════════════
输入数据:
  ├── 文本 Prompt (编辑指令，如 "change the style to watercolor")
  ├── 参考图像列表 (1~N张 PIL Image)
  ├── 图像尺寸 (width=512, height=512)
  └── 采样参数 (num_steps=30, guidance_scale=7.5, seed)
═══════════════════════════════════════════════════════════════════

步骤1: 文本编码 → 子网络1: VLM (Qwen3.5-4B, 冻结)
  ┌─────────────────────────────────┐
  │ 编辑指令 + 可选参考图上下文        │
  │ "change the style to watercolor" │
  └──────────┬──────────────────────┘
             │
             ▼
  ┌─────────────────────────────────────┐
  │ VLM: Qwen3.5-4B (冻结)              │
  │                                     │
  │ 1. Tokenize (文本+可选图片token)     │
  │ 2. Forward (无梯度)                  │
  │ 3. 提取 hidden_states[-1]           │
  │                                     │
  │ 输出: ctx [B, 128, 2560]            │
  │ (图文混合输入序列更长, 128 tokens)    │
  └──────────┬──────────────────────────┘
             │
             ▼
  ctx [B, 128, 2560]
  ctx_ids [B, 128, 2]

═══════════════════════════════════════════════════════════════════

步骤2: 目标图 VAE 编码(训练时) / 初始化噪声(推理时)

  训练时: → 子网络2: VAE Encoder (FLUX2, 冻结)
  ┌───────────────────────┐
  │ 目标图 (ground truth)  │
  │ [B, 3, 512, 512]      │
  └──────────┬────────────┘
             │
             ▼
  ┌──────────────────────────────────────┐
  │ VAE Encoder → Patch → BatchNorm      │
  │ 输出: target_latents [B, 1024, 128] │
  └──────────┬───────────────────────────┘
             │
             ▼
  目标图 tokens [B, 1024, 128]

  推理时: 直接初始化纯噪声
  target_latents = randn(B, 1024, 128)

═══════════════════════════════════════════════════════════════════

步骤3: 参考图 VAE 编码 → 子网络2: VAE Encoder (FLUX2, 冻结, 共享)
  ┌──────────────────────────────────────┐
  │ 参考图列表                             │
  │ [source_img1, source_img2, ...]      │
  │ 共 N 张图                             │
  └──────────┬───────────────────────────┘
             │
             ▼
  ┌──────────────────────────────────────────────┐
  │ 对每张参考图:                                   │
  │   VAE Encoder → Patch → BatchNorm             │
  │   → ref_latent [1, 128, 32, 32]              │
  │   → flatten → [1024, 128] per image          │
  │                                               │
  │ 拼接所有参考图:                                 │
  │   ref_latents [B, N×1024, 128]               │
  │                                               │
  │ 参考图位置ID (使用偏移坐标，不与目标图重叠):      │
  │   ref_ids: 每张图的 h_offset = (i+1)*h + 10   │
  │   例: 第1张图 h ∈ [42,73], 第2张图 h ∈ [84,115]│
  └──────────┬───────────────────────────────────┘
             │
             ▼
  ref_latents [B, N×1024, 128]（干净的参考图tokens）
  ref_ids [B, N×1024, 2]（偏移后的位置ID）

═══════════════════════════════════════════════════════════════════

步骤4: 构建输入序列 + 加噪

  训练时:
  ┌──────────────────────────────────────────────┐
  │ 1. 仅对目标图加噪 (参考图保持干净!)             │
  │    noise ~ N(0, I), t ~ logit_normal         │
  │    noisy_target = (1-t)*target + t*noise      │
  │                                               │
  │ 2. 拼接: x = [noisy_target, clean_ref]       │
  │    x_input [B, 1024+N×1024, 128]             │
  │                                               │
  │ 3. 拼接位置ID:                                 │
  │    img_ids = [target_ids, ref_ids]            │
  │    [B, 1024+N×1024, 2]                       │
  └──────────┬───────────────────────────────────┘

  推理时:
  ┌──────────────────────────────────────────────┐
  │ 1. 目标区域初始化纯噪声: z_1 ~ N(0, I)        │
  │                                               │
  │ 2. 每步拼接: x = [target_noisy, ref_clean]   │
  │    (参考图在每步都保持干净不变)                  │
  └──────────┬───────────────────────────────────┘
             │
             ▼

═══════════════════════════════════════════════════════════════════

步骤5: MMDIT 前向 → 子网络4: MMDIT DIT (可训练)
  ┌──────────────────────────────────────────────────────┐
  │ MMDITModel.forward()                                │
  │                                                      │
  │ 输入:                                                 │
  │   x = [noisy_target, clean_ref]                     │
  │       [B, (1+N)×1024, 128]                          │
  │   x_ids = [target_ids, ref_ids]                     │
  │       [B, (1+N)×1024, 2]                            │
  │   timesteps = t [B]                                 │
  │   ctx = text_emb [B, 128, 2560]                    │
  │   ctx_ids [B, 128, 2]                               │
  │                                                      │
  │ 注意: 目标图tokens和参考图tokens在同一序列中            │
  │ 通过joint attention自然融合参考图信息                   │
  │ (无显式mask，无因果注意力)                             │
  │                                                      │
  │ 输出: v_pred [B, (1+N)×1024, 128]                   │
  └──────────┬───────────────────────────────────────────┘
             │
             ▼

═══════════════════════════════════════════════════════════════════

步骤6: 训练时 — 仅对目标图计算 Loss

  ┌──────────────────────────────────────────────┐
  │ 截取目标图部分的预测:                           │
  │   v_pred_target = v_pred[:, :1024, :]        │
  │   (丢弃参考图部分的预测，不计算loss)             │
  │                                               │
  │ target = noise - clean_target                 │
  │ loss = MSE(v_pred_target, target) [标量]      │
  │                                               │
  │ → backward() → clip_grad_norm_(1.0) → step() │
  └──────────────────────────────────────────────┘

  推理时 — Euler ODE 去噪循环 (仅更新目标图):

  for t in schedule:
    ┌──────────────────────────────────────────┐
    │ 1. 拼接: x = [target_noisy, ref_clean]  │
    │ 2. v_pred = model(x, ids, t, ctx, ...)  │
    │ 3. 截取: v_target = v_pred[:, :1024, :] │
    │ 4. CFG(可选):                            │
    │    v = v_uncond + scale*(v_cond-v_uncond)│
    │ 5. Euler step (仅更新target):            │
    │    target = target + dt * v_target       │
    │ (参考图tokens不更新，下一步重新拼接)       │
    └──────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════

步骤7: 推理时 — VAE解码 → 子网络3: VAE Decoder (FLUX2, 冻结)
  ┌──────────────────────────────────────┐
  │ 去噪完成后的目标 latent               │
  │ target_latents [B, 1024, 128]       │
  │                                      │
  │ reshape: [B, 128, 32, 32]            │
  └──────────┬───────────────────────────┘
             │
             ▼
  ┌──────────────────────────────────────┐
  │ VAE Decoder (FLUX2, 冻结)             │
  │                                      │
  │ 1. inv_normalize                     │
  │ 2. unpatch: (128,32,32)→(32,64,64)  │
  │ 3. Decoder: conv→mid→up×3→conv_out  │
  │                                      │
  │ 输出: 编辑后图像 [B, 3, 512, 512]    │
  └──────────┬───────────────────────────┘
             │
             ▼
  ┌──────────────────────────────────────┐
  │ 后处理 → PIL Image → 保存            │
  └──────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════

最终输出: 由 **子网络3: VAE Decoder** 输出编辑后的 RGB 图像 [B, 3, H, W]

═══════════════════════════════════════════════════════════════════

与文生图的关键区别总结:
  ┌────────────────────────────────────────────────────────────┐
  │ 1. 参考图编码: source images → VAE.encode → ref_tokens    │
  │ 2. 位置ID偏移: 参考图使用独立空间坐标，不与目标图重叠        │
  │ 3. 序列拼接: x = [noisy_target, clean_ref]               │
  │ 4. Loss范围: 仅对目标图tokens计算loss                      │
  │ 5. 去噪更新: 仅更新目标图，每步重新拼接干净参考图           │
  │ 6. 不使用mask: inpaint等通过文本指令隐式指定编辑区域       │
  └────────────────────────────────────────────────────────────┘
```

---

## 问题8：相比FLUX2模型的创新点和改进点

### 基于代码实现分析，本模型相比 FLUX2 具有以下创新点和改进：

#### 1. VLM 文本编码器简化：Qwen3.5-4B 替代 Mistral-Small-3.2-24B

| 特征 | FLUX2 | 本模型 |
|------|-------|--------|
| 文本编码器 | Mistral-Small-3.2-24B（24B 参数，多模态 LLM） | **Qwen3.5-4B**（4B 参数，LLM） |
| 输出维度 | 15360（3层 × 5120 拼接） | **2560**（最后1层 hidden_states） |
| 多层特征提取 | ✅ 第10/20/30层拼接 | ❌ **仅取最后隐藏层** |
| 参数量 | ~24B | **~4B**（大幅减少） |

**代码依据**：
```python
# design_doc.md
# VLM (冻结): Qwen3.5-4B 取最后隐藏层作为文本条件

# double_stream_mmdit.py 第379行
context_in_dim=2560  # Qwen3.5-4B 的 hidden_size = 2560
```

**改进意义**：VLM 参数量从 24B 降至 4B，显著降低了推理成本和显存需求。通过 `txt_in` 线性投影层即可适配不同 VLM，未来可随时升级。

#### 2. 2D RoPE 替代 4D RoPE（位置编码简化）

| 特征 | FLUX2 | 本模型 |
|------|-------|--------|
| RoPE 维度 | 4D (t, h, w, l) | **2D (h, w)** |
| axes_dim | [32, 32, 32, 32] | **[64, 64]**（或 [32, 32]） |
| theta | 2000 | 2000（相同） |
| 实现方式 | 矩阵旋转 | 矩阵旋转（相同） |

**代码依据**：
```python
# design_doc.md 第86-91行
# 推断依据: 纯2D图像任务不需要时间轴。
# 结论: 2D RoPE (h,w)，矩阵旋转形式，theta=2000。

# double_stream_mmdit.py 第378行
axes_dim=[64, 64]  # 仅 h, w 两个轴
```

**改进意义**：纯 2D 图像生成/编辑不需要时间轴和序列轴，简化为 2D RoPE 减少了计算开销，同时使 head_dim 更高效分配（全部用于空间位置编码）。

#### 3. 每层独立 Modulation 替代全局共享 Modulation

| 特征 | FLUX2 | 本模型 |
|------|-------|--------|
| Modulation 方式 | **全局共享** 3 个 Modulation | **每层独立** Modulation |
| 双流块 | 所有层共享 `double_stream_modulation_img/txt` | **每个块有自己的** `img_mod` 和 `txt_mod` |
| 单流块 | 所有层共享 `single_stream_modulation` | **每个块有自己的** `mod` |

**代码依据**：
```python
# double_stream_mmdit.py 第191-205行
class DoubleStreamBlock(nn.Module):
    def __init__(self, ...):
        self.img_mod = Modulation(hidden_size, double=True)  # 每个块独立
        self.txt_mod = Modulation(hidden_size, double=True)  # 每个块独立

# single_stream_mmdit.py 第192行
class SingleStreamBlock(nn.Module):
    def __init__(self, ...):
        self.mod = Modulation(hidden_size)  # 每个块独立
```

**改进意义**：每层独立的 Modulation 使不同深度的层可以有不同的调制行为，理论上表达能力更强。FLUX2 的全局共享方案虽然减少参数，但限制了各层的调制灵活性。

#### 4. 无因果注意力、无显式 Mask 的简化参考图处理

| 特征 | FLUX2 | 本模型 |
|------|-------|--------|
| 参考图处理 | 因果注意力（causal_attn_fn） | **无显式 mask，全注意力** |
| Attention 可见性 | ref 只能看自己，img+txt 可看所有 | **所有 tokens 互相可见** |
| 实现复杂度 | 需要特殊注意力 mask 逻辑 | **标准 SDPA，无额外逻辑** |

**代码依据**：
```python
# double_stream_mmdit.py 第254-255行
# F.scaled_dot_product_attention (auto flash attn with BF16)
attn = F.scaled_dot_product_attention(q, k, v)  # 无 mask 参数

# design_doc.md 第137-138行
# | Mask | 无显式mask | 无显式mask | 无 | **无显式mask** |
```

**改进意义**：去掉因果注意力简化了实现，同时允许参考图 tokens 之间互相关注，可能在多参考图场景下获得更好的跨参考图信息融合。代价是参考图可能被去噪过程的信号"污染"，但设计文档指出通过位置偏移已经自然区分。

#### 5. 参考图无固定时间步（统一 Timestep 调制）

| 特征 | FLUX2 | 本模型 |
|------|-------|--------|
| 参考图时间步 | **固定 t=0.0**（分别计算 modulation） | **使用当前时间步**（统一 modulation） |
| Modulation 混合 | `_blend_double_mods` 按位置混合 | **无混合，统一 vec** |
| 实现复杂度 | 需要分别计算 ref/img 的 modulation 并混合 | **单一 vec 通过所有层** |

**代码依据**：
```python
# double_stream_mmdit.py 第436-439行
def forward(self, x, x_ids, timesteps, ctx, ctx_ids):
    # Timestep embedding → conditioning vector (统一的 vec)
    t_emb = timestep_embedding(timesteps, self.time_embed_dim)
    vec = self.time_in(t_emb)
    # 所有 tokens (包括参考图) 使用同一个 vec
```

**改进意义**：统一时间步简化了实现，避免了 FLUX2 中复杂的 modulation 混合逻辑。参考图虽然使用了当前去噪时间步的调制，但其内容本身是干净的（未加噪），因此调制参数的"语义"信息仍然主要来自时间步。

#### 6. 无 KV Cache（简化推理流程）

| 特征 | FLUX2 | 本模型 |
|------|-------|--------|
| KV Cache | ✅ 支持（`forward_kv_extract` + `forward_kv_cached`） | ❌ **无 KV Cache** |
| 推理加速 | 参考图 KV 缓存复用，减少重复计算 | **每步完整前向传播** |

**代码依据**：
```python
# inference_edit.py 第180-191行
# 每步都完整拼接并前向：
x_input = torch.cat([latent_input.to(torch.bfloat16), ref_latents_cfg], dim=1)
v_pred = model(x=x_input, x_ids=all_img_ids, timesteps=t_input,
               ctx=prompt_embeds, ctx_ids=txt_ids)
```

**改进意义**：去掉 KV Cache 简化了代码实现和维护成本，适合研究和快速迭代。在中小模型（1B~8B）上，每步完整前向传播的额外开销相对可接受。

#### 7. 无 Guidance Embedding（使用标准 CFG）

| 特征 | FLUX2 | 本模型 |
|------|-------|--------|
| Guidance 注入 | `guidance_in`（MLP嵌入，注入到 vec） | **无 guidance_in，使用标准 CFG** |
| CFG 实现 | Guidance Embedding（蒸馏版） + 标准 CFG（Base版） | **仅标准 CFG** |

**代码依据**：
```python
# double_stream_mmdit.py 第370-381行
class MMDITModel(nn.Module):
    def __init__(self, ...):
        # 无 self.guidance_in
        self.time_in = MLPTimeStepEmbedder(in_dim=time_embed_dim, hidden_dim=hidden_size)
        # 仅 timestep embedding，无 guidance embedding

# inference_t2i.py 第157-159行
if do_cfg:
    v_uncond, v_cond = v_pred.chunk(2)
    v_pred = v_uncond + guidance_scale * (v_cond - v_uncond)  # 标准 CFG
```

**改进意义**：标准 CFG 实现更简洁，且不需要 guidance distillation 训练。适合从零开始训练的场景。

#### 8. 提供双流+单流和纯单流两种架构变体

| 特征 | FLUX2 | 本模型 |
|------|-------|--------|
| 架构选择 | 仅双流+单流 | **双流+单流 + 纯单流两种变体** |
| 纯单流模型 | ❌ 无 | ✅ `FullSingleStreamMMDITModel` |

**代码依据**：
```python
# single_stream_mmdit.py 第248行
class FullSingleStreamMMDITModel(nn.Module):
    """
    与 double_stream_mmdit.py 的区别：
    - 全部使用 SingleStreamBlock，无 DoubleStreamBlock
    - txt 和 img 在输入阶段即拼接为统一序列
    - 结构更简单、参数效率更高、前向更快
    - 参考来源：Z-Image 纯单流 + Flux2 SingleStreamBlock 设计
    """
```

**改进意义**：提供纯单流变体，在同等参数预算下可以使用更深的网络（Scaling Laws 表明深度通常比宽度更重要），前向速度更快。

#### 9. 五档参数规模配置（1B/2B/4B/6B/8B）

| 特征 | FLUX2 | 本模型 |
|------|-------|--------|
| 模型规模档位 | 3档（Klein-4B, Klein-9B, Dev-32B） | **5档**（1B, 2B, 4B, 6B, 8B） |
| 最小模型 | Klein-4B | **1B**（更轻量） |
| 最大模型 | Dev-32B | 8B（更紧凑） |

**代码依据**：
```python
# double_stream_mmdit.py 第523-610行
def mmdit_1b(**kwargs): ...   # 1.14B
def mmdit_2b(**kwargs): ...   # 2.02B
def mmdit_4b(**kwargs): ...   # 4.04B
def mmdit_6b(**kwargs): ...   # 6.32B
def mmdit_8b(**kwargs): ...   # 8.65B

# single_stream_mmdit.py 第391-473行
def mmdit_single_1b(**kwargs): ...   # 1.00B
def mmdit_single_2b(**kwargs): ...   # 2.03B
def mmdit_single_4b(**kwargs): ...   # 4.01B
def mmdit_single_6b(**kwargs): ...   # 6.08B
def mmdit_single_8b(**kwargs): ...   # 8.07B
```

**改进意义**：提供更细粒度的模型规模选择，从 1B 到 8B 覆盖从快速实验到生产部署的各种需求。每一档模型的宽度和深度均严格大于前一档。

#### 10. 所有 Linear 层统一 bias=False

| 特征 | FLUX2 | 本模型 |
|------|-------|--------|
| Bias 设计 | **几乎全部 bias=False** | **全部 bias=False** |

**代码依据**：
```python
# double_stream_mmdit.py 中所有 Linear 层
self.in_layer = nn.Linear(in_dim, hidden_dim, bias=False)   # TimeStepEmbedder
self.lin = nn.Linear(dim, self.multiplier * dim, bias=False) # Modulation
self.qkv = nn.Linear(dim, dim * 3, bias=False)              # SelfAttention
self.proj = nn.Linear(dim, dim, bias=False)                  # SelfAttention
self.img_in = nn.Linear(in_channels, hidden_size, bias=False) # 输入投影
self.txt_in = nn.Linear(context_in_dim, hidden_size, bias=False) # 输入投影
# MLP 中的 Linear 也全部 bias=False
```

**改进意义**：与 FLUX2 一致的设计选择，减少参数量并提高训练稳定性。本模型严格执行全部 bias=False，确保一致性。

#### 11. 1B 模型使用 mlp_ratio=4.0（自适应 FFN 比率）

| 特征 | FLUX2 | 本模型 |
|------|-------|--------|
| mlp_ratio | 全系列 3.0 | **1B: 4.0，2B~8B: 3.0** |

**代码依据**：
```python
# design_doc.md 第100-104行
# 推断依据: SiLU-Gated与AdaLN的SiLU一致。
# mlp_ratio=3.0与Flux2全系列一致。
# 1B模型因head_dim=64较小，保持mlp_ratio=4.0以保证MLP表达力。
```

**改进意义**：对 1B 小模型做了针对性优化 — 因为 head_dim=64 较小（2B+ 为 128），使用更大的 mlp_ratio=4.0 来弥补 MLP 的表达能力不足。

#### 12. 支持多种 Loss Weighting 方案

| 特征 | FLUX2 | 本模型 |
|------|-------|--------|
| Loss Weighting | 简单 MSE | **3种方案可选** |

**代码依据**：
```python
# losses.py 第68-117行
def compute_flow_matching_loss(..., weighting_scheme="none"):
    if weighting_scheme == "sigma_sqrt":
        w = 1.0 / (sigmas + 1e-6)  # 高噪声降权
    elif weighting_scheme == "min_snr_5":
        # Min-SNR-gamma(γ=5)，防止梯度爆炸
        snr = (1.0 - sigmas) / (sigmas + 1e-6)
        w = torch.minimum(snr, torch.full_like(snr, gamma)) / snr
```

**改进意义**：提供 `none`（均匀权重）、`sigma_sqrt`（高噪声降权）、`min_snr_5`（Min-SNR-gamma）三种 weighting 方案，为不同训练阶段和数据集提供灵活选择。

#### 13. 支持多种时间步采样策略

| 特征 | FLUX2 | 本模型 |
|------|-------|--------|
| 时间步采样 | 自定义 schedule | **uniform + logit_normal** |

**代码依据**：
```python
# losses.py 第42-65行
def sample_timesteps_uniform(batch_size, device):
    """均匀采样 t ∈ (0, 1)"""
    return torch.rand(batch_size, device=device)

def sample_timesteps_logit_normal(batch_size, device, mean=0.0, std=1.0):
    """Logit-Normal 分布采样（中间时间步采样更多）"""
    u = torch.randn(batch_size, device=device) * std + mean
    t = torch.sigmoid(u).clamp(1e-5, 1.0 - 1e-5)
    return t
```

**改进意义**：提供两种标准化的时间步采样策略。Logit-Normal 在中间时间步附近采样更多，这些时间步的梯度信号更有价值（来自 SD3 论文的结论）。

#### 14. 参考图位置编码使用简单偏移坐标

| 特征 | FLUX2 | 本模型 |
|------|-------|--------|
| 参考图位置区分 | **4D RoPE 中的时间坐标**（t=10, 20, 30...） | **2D 空间坐标偏移**（h+offset） |

**代码依据**：
```python
# train_edit.py 第167-179行
for ref_idx in range(num_refs):
    offset_h = (ref_idx + 1) * latent_h + 10  # 简单的高度偏移
    h_ids = torch.arange(offset_h, offset_h + latent_h, ...)
    w_ids = torch.arange(latent_w, ...)
    grid = torch.meshgrid(h_ids, w_ids, indexing="ij")
    ref_ids = torch.stack(grid, dim=-1).reshape(-1, 2)
```

**改进意义**：使用简单的空间偏移坐标来区分不同参考图，无需引入额外的时间维度。这与 2D RoPE 的设计一致，避免了 4D RoPE 的复杂性。

#### 15. 无内容安全过滤、无 Prompt Upsampling

| 特征 | FLUX2 | 本模型 |
|------|-------|--------|
| NSFW 过滤 | ✅ 内置 | ❌ 无 |
| 版权过滤 | ✅ 内置 | ❌ 无 |
| Prompt Upsampling | ✅ 本地+API | ❌ 无 |

**改进意义**：作为研究/训练框架，去掉了商业化的安全过滤和 prompt 增强功能，保持代码简洁，专注于核心模型架构。

---

### 创新点和改进总结表

| # | 维度 | FLUX2 | 本模型 | 改进方向 |
|---|------|-------|--------|---------|
| 1 | 文本编码器 | Mistral-Small-24B (多层拼接→15360d) | **Qwen3.5-4B (最后层→2560d)** | 轻量化 |
| 2 | 位置编码 | 4D RoPE (t,h,w,l) | **2D RoPE (h,w)** | 简化 |
| 3 | Modulation | 全局共享 3 个 Modulation | **每层独立 Modulation** | 更灵活 |
| 4 | 参考图注意力 | 因果注意力（ref 自注意力） | **标准全注意力（无 mask）** | 简化 |
| 5 | 参考图时间步 | 固定 t=0.0 + Modulation 混合 | **统一当前时间步** | 简化 |
| 6 | KV Cache | ✅ 支持 | ❌ 无（每步完整前向） | 简化 |
| 7 | Guidance | Guidance Embedding + CFG | **仅标准 CFG** | 简化 |
| 8 | 架构变体 | 仅双流+单流 | **双流+单流 + 纯单流** | 更多选择 |
| 9 | 模型规模 | 3档 (4B/9B/32B) | **5档 (1B/2B/4B/6B/8B)** | 更细粒度 |
| 10 | bias | 几乎全部 False | **严格全部 False** | 一致性 |
| 11 | mlp_ratio | 全系列 3.0 | **1B: 4.0, 其余 3.0** | 自适应 |
| 12 | Loss Weighting | 简单 MSE | **3种方案可选** | 更灵活 |
| 13 | 时间步采样 | 自定义 | **uniform + logit_normal** | 更标准化 |
| 14 | 参考图位置 | 4D RoPE 时间坐标 | **2D 空间坐标偏移** | 简化 |
| 15 | 商业功能 | NSFW过滤 + Prompt Upsampling | **无（纯研究框架）** | 简洁 |

**总体设计哲学**：相比 FLUX2 的商业化完整实现，本模型采取了**"简化但不简陋"**的设计策略。在保持核心架构有效性的前提下，去掉了因果注意力、KV Cache、Guidance Embedding、全局共享 Modulation 等复杂机制，转而使用更简洁的标准实现。同时提供了纯单流变体和五档模型配置，增加了灵活性和可扩展性。这种设计特别适合**从零开始训练**的研究场景和中小规模部署。

---

> **报告完成时间**: 基于 `SimpleGeneration/universal_image_edit/` 目录全部源代码分析
>
> **分析的文件列表**:
> - `SimpleGeneration/universal_image_edit/design_doc.md` — 完整设计文档
> - `SimpleGeneration/universal_image_edit/models/double_stream_mmdit.py` — 双流+单流 MMDIT 模型
> - `SimpleGeneration/universal_image_edit/models/single_stream_mmdit.py` — 纯单流 MMDIT 模型
> - `SimpleGeneration/universal_image_edit/losses.py` — Flow Matching 损失函数
> - `SimpleGeneration/universal_image_edit/pipeline/train_t2i.py` — 文生图训练 Pipeline
> - `SimpleGeneration/universal_image_edit/pipeline/train_edit.py` — 图像编辑训练 Pipeline
> - `SimpleGeneration/universal_image_edit/pipeline/inference_t2i.py` — 文生图推理 Pipeline
> - `SimpleGeneration/universal_image_edit/pipeline/inference_edit.py` — 图像编辑推理 Pipeline
> - `SimpleGeneration/universal_image_edit/pipeline/scheduler.py` — Flow Match Euler 调度器
> - `SimpleGeneration/flux_autoencoder/models/flux2_autoencoder.py` — FLUX2 VAE 实现
> - `SimpleGeneration/flux_autoencoder/models/flux1_autoencoder.py` — FLUX1 VAE 实现（对比）
> - `analysis/FLUX2_model_analysis.md` — FLUX2 模型分析报告（对比参考）
