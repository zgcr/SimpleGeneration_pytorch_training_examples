# FLUX.2 模型代码全面分析报告

> 本报告基于 `flux2/` 目录下的全部源代码实现进行深入分析，所有结论均由代码逻辑推导得出。

> **代码实现根目录**：`/root/code/flux2/`

---

## 目录

1. [问题1：是否使用了VAE？VAE结构类型](#问题1是否使用了vaevae结构类型)
2. [问题2：是否使用了flow_matching的DIT模型？单流还是双流？](#问题2是否使用了flow_matching的dit模型单流还是双流)
3. [问题3：具体网络结构和子网络](#问题3具体网络结构和子网络)
4. [问题4：模型网络结构图](#问题4模型网络结构图)
5. [问题5：是否支持文生图和图像编辑](#问题5是否支持文生图和图像编辑)
6. [问题6：文生图流程图](#问题6文生图流程图)
7. [问题7：图像+文本提示图像编辑流程图](#问题7图像文本提示图像编辑流程图)
8. [问题8：相比FLUX.1模型的创新点](#问题8相比flux1模型的创新点)

---

## 问题1：是否使用了VAE？VAE结构类型

### 结论：是的，FLUX.2 使用了 VAE。其VAE结构基础与FLUX.1相同（经典卷积式Encoder-Decoder），但增加了关键改进。

### 代码分析依据

**1. VAE 基础结构（`autoencoder.py`）**

FLUX.2 的 VAE 定义在 `AutoEncoder` 类中，包含 `Encoder` 和 `Decoder` 两个子模块：

```python
# autoencoder.py 第271-291行
class AutoEncoder(nn.Module):
    def __init__(self, params: AutoEncoderParams):
        self.encoder = Encoder(...)
        self.decoder = Decoder(...)
```

默认参数为：
```python
@dataclass
class AutoEncoderParams:
    resolution: int = 256
    in_channels: int = 3
    ch: int = 128
    out_ch: int = 3
    ch_mult: list[int] = [1, 2, 4, 4]
    num_res_blocks: int = 2
    z_channels: int = 32
```

这些参数（`ch=128`、`ch_mult=[1,2,4,4]`、`num_res_blocks=2`、`z_channels=32`）与 FLUX.1 的 VAE 参数**完全一致**。

**2. Encoder 结构**

Encoder 由以下模块组成（与FLUX.1 VAE相同）：
- `conv_in`：3×3 卷积，将 3 通道输入映射到 `ch=128` 通道
- 4个下采样层级（由 `ch_mult=[1,2,4,4]` 控制），每个层级包含：
  - 2个 `ResnetBlock`（GroupNorm → Swish → Conv3x3 → GroupNorm → Swish → Conv3x3 + shortcut）
  - `Downsample`（stride=2 的 3×3 卷积），最后一个层级除外
- `mid` 中间层：`ResnetBlock` → `AttnBlock` → `ResnetBlock`
- `norm_out` → Swish → `conv_out`：输出 `2*z_channels=64` 通道（均值和方差）
- `quant_conv`：1×1 卷积

**3. Decoder 结构**

Decoder 是 Encoder 的镜像结构：
- `post_quant_conv`：1×1 卷积
- `conv_in`：将 `z_channels=32` 通道映射到 `block_in`
- `mid` 中间层：`ResnetBlock` → `AttnBlock` → `ResnetBlock`
- 4个上采样层级（反序），每个层级包含：
  - 3个 `ResnetBlock`（`num_res_blocks + 1`）
  - `Upsample`（最近邻插值 + 3×3 卷积），第一个层级除外
- `norm_out` → Swish → `conv_out`：输出 3 通道 RGB 图像

**4. FLUX.2 VAE 的关键改进**

与 FLUX.1 VAE 相比，FLUX.2 的 VAE 增加了两个重要改进：

**（1）BatchNorm 归一化/反归一化机制**

```python
# autoencoder.py 第293-312行
self.bn = torch.nn.BatchNorm2d(
    math.prod(self.ps) * params.z_channels,  # 2*2*32 = 128 通道
    eps=1e-4, momentum=0.1, affine=False, track_running_stats=True,
)

def normalize(self, z):
    self.bn.eval()
    return self.bn(z)

def inv_normalize(self, z):
    self.bn.eval()
    s = torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + self.bn_eps)
    m = self.bn.running_mean.view(1, -1, 1, 1)
    return z * s + m
```

这个 BatchNorm 使用 `affine=False`（无可学习参数），仅使用 `running_mean` 和 `running_var` 来标准化 latent 空间。这比 FLUX.1 的简单 scaling factor 更加精细。

**（2）Patch 空间重排操作**

```python
# autoencoder.py 第295行
self.ps = [2, 2]

# encode 方法
def encode(self, x):
    moments = self.encoder(x)
    mean = torch.chunk(moments, 2, dim=1)[0]  # 只取均值，不取方差
    z = rearrange(mean, "... c (i pi) (j pj) -> ... (c pi pj) i j", pi=2, pj=2)
    z = self.normalize(z)
    return z

# decode 方法
def decode(self, z):
    z = self.inv_normalize(z)
    z = rearrange(z, "... (c pi pj) i j -> ... c (i pi) (j pj)", pi=2, pj=2)
    dec = self.decoder(z)
    return dec
```

Patch 操作将 `z_channels=32` 通道、空间尺寸为 `(H/8, W/8)` 的 latent，重排为 `128` 通道、空间尺寸为 `(H/16, W/16)` 的表示。这意味着 FLUX.2 的 **有效下采样倍数为 16x**（Encoder 硬件下采样 8x + Patch 操作进一步 2x）。

**注意**：Encode 时只使用了均值（`mean`），没有使用重参数化技巧采样，因此在**推理时**这个 VAE 的行为更像一个**确定性自编码器（AE）**而非概率 VAE。

### 总结

| 特征 | FLUX.1 VAE | FLUX.2 VAE |
|------|-----------|-----------|
| 基础结构 | 卷积 Encoder-Decoder | 卷积 Encoder-Decoder（相同） |
| z_channels | 16 | 32 |
| 有效下采样倍数 | 8x | 16x（8x + 2x Patch） |
| Latent 通道数 | 16 | 128（32×2×2） |
| Latent 归一化 | Scaling factor | BatchNorm 归一化 |
| Encoder 输出 | 取均值 | 取均值（相同） |
| 参数 ch_mult | [1,2,4,4] | [1,2,4,4]（相同） |

---

## 问题2：是否使用了flow_matching的DIT模型？单流还是双流？

### 结论：是的，使用了 flow matching 的 DIT 模型。FLUX.2 **同时使用了双流 MMDIT（DoubleStreamBlock）和单流 DIT（SingleStreamBlock）**，先经过双流块再经过单流块。

### 代码分析依据

**1. 模型参数定义（`model.py`）**

```python
@dataclass
class Flux2Params:
    in_channels: int = 128          # latent 通道数
    context_in_dim: int = 15360     # 文本编码器输出维度（Mistral多层拼接）
    hidden_size: int = 6144         # 隐藏层维度
    num_heads: int = 48             # 注意力头数
    depth: int = 8                  # 双流块层数
    depth_single_blocks: int = 48   # 单流块层数
    axes_dim: list[int] = [32, 32, 32, 32]  # 4D RoPE 维度分配
    theta: int = 2000               # RoPE theta
    mlp_ratio: float = 3.0          # MLP 扩展比率
    use_guidance_embed: bool = True  # 是否使用 guidance embedding
```

**2. 双流 MMDIT 块（`DoubleStreamBlock`）**

```python
# model.py 第524-681行
class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio):
        # 图像分支
        self.img_norm1 = nn.LayerNorm(hidden_size, ...)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads)
        self.img_norm2 = nn.LayerNorm(hidden_size, ...)
        self.img_mlp = nn.Sequential(Linear → SiLUActivation → Linear)

        # 文本分支
        self.txt_norm1 = nn.LayerNorm(hidden_size, ...)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads)
        self.txt_norm2 = nn.LayerNorm(hidden_size, ...)
        self.txt_mlp = nn.Sequential(Linear → SiLUActivation → Linear)
```

双流块中，图像和文本各有**独立的 LayerNorm、Attention QKV 投影、MLP**，但在 Attention 计算时，Q/K/V 被 **拼接（concatenate）** 在一起进行**联合注意力**：

```python
q = torch.cat((txt_q, img_q), dim=2)
k = torch.cat((txt_k, img_k), dim=2)
v = torch.cat((txt_v, img_v), dim=2)
```

然后将注意力输出再分开，分别送入各自的残差连接和 MLP。

**3. 单流 DIT 块（`SingleStreamBlock`）**

```python
# model.py 第437-521行
class SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio):
        self.linear1 = nn.Linear(hidden_size, hidden_size*3 + mlp_hidden_dim*2)
        self.linear2 = nn.Linear(hidden_size + mlp_hidden_dim, hidden_size)
        self.norm = QKNorm(head_dim)
        self.pre_norm = nn.LayerNorm(hidden_size, ...)
        self.mlp_act = SiLUActivation()
```

单流块中，图像和文本 token 已经**拼接在一起**，使用同一组参数进行处理。`linear1` 同时产生 QKV（用于 attention）和 MLP 的输入，最后通过 `linear2` 合并输出。

**4. 前向传播流程（`Flux2.forward()`）**

```python
def forward(self, x, x_ids, timesteps, ctx, ctx_ids, guidance):
    # 1. 时间步嵌入
    vec = self.time_in(timestep_embedding(timesteps, 256))
    if self.use_guidance_embed:
        vec = vec + self.guidance_in(timestep_embedding(guidance, 256))

    # 2. 全局 Modulation（共享）
    double_block_mod_img = self.double_stream_modulation_img(vec)
    double_block_mod_txt = self.double_stream_modulation_txt(vec)
    single_block_mod, _ = self.single_stream_modulation(vec)

    # 3. 输入投影
    img = self.img_in(x)       # [B, L_img, hidden_size]
    txt = self.txt_in(ctx)     # [B, L_txt, hidden_size]

    # 4. 位置编码
    pe_x = self.pe_embedder(x_ids)
    pe_ctx = self.pe_embedder(ctx_ids)

    # 5. 双流块处理
    for block in self.double_blocks:
        img, txt, _ = block.forward_kv_extract(img, txt, pe_x, pe_ctx, ...)

    # 6. 拼接进入单流块
    img = torch.cat((txt, img), dim=1)
    pe = torch.cat((pe_ctx, pe_x), dim=2)

    for block in self.single_blocks:
        img, _ = block.forward_kv_extract(img, pe, ...)

    # 7. 去掉文本token，只保留图像token
    img = img[:, num_txt_tokens:, ...]

    # 8. 最终输出层
    img = self.final_layer(img, vec)
    return img
```

**5. Flow Matching 去噪过程（`sampling.py`）**

```python
def denoise(model, img, img_ids, txt, txt_ids, timesteps, guidance, ...):
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, ...)
        pred = model(x=img_input, x_ids=img_input_ids, timesteps=t_vec,
                     ctx=txt, ctx_ids=txt_ids, guidance=guidance_vec)
        img = img + (t_prev - t_curr) * pred   # 欧拉步进
    return img
```

使用标准的 flow matching 欧拉步进 `x_{t-1} = x_t + (t_prev - t_curr) * v_pred`，其中模型预测速度场 `v`。

### 总结

| 特征 | 说明 |
|------|------|
| 是否 Flow Matching | ✅ 是，使用欧拉步进的 rectified flow |
| 双流 MMDIT | ✅ `DoubleStreamBlock` × 8 层 |
| 单流 DIT | ✅ `SingleStreamBlock` × 48 层 |
| 架构顺序 | 先双流后单流 |
| 位置编码 | 4D RoPE（t, h, w, l） |
| Modulation | 全局共享的 AdaLN-Zero 调制 |

---

## 问题3：具体网络结构和子网络

### FLUX.2 包含以下子网络结构：

#### 子网络 1：文本编码器（Text Encoder）

FLUX.2 支持两种文本编码器：

**（1）Mistral-Small-3.2-24B-Instruct（用于 FLUX.2 [dev] 32B 模型）**

```python
# text_encoder.py
class Mistral3SmallEmbedder(nn.Module):
    def __init__(self):
        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            "mistralai/Mistral-Small-3.2-24B-Instruct-2506", ...
        )
        self.processor = AutoProcessor.from_pretrained(...)
```

- **模型**：Mistral-Small-3.2-24B，一个 24B 参数的**多模态大语言模型**
- **输出**：提取第 10、20、30 层的隐藏状态，拼接为 `(B, L, 3×5120) = (B, L, 15360)` 维度的特征
- **最大序列长度**：512 tokens
- **附加功能**：Prompt Upsampling（本地或 API）、内容安全过滤（NSFW检测 + 版权检测）

```python
OUTPUT_LAYERS_MISTRAL = [10, 20, 30]

def forward(self, txt):
    output = self.model(input_ids=..., output_hidden_states=True)
    out = torch.stack([output.hidden_states[k] for k in OUTPUT_LAYERS_MISTRAL], dim=1)
    return rearrange(out, "b c l d -> b l (c d)")  # (B, 512, 15360)
```

**（2）Qwen3（用于 FLUX.2 [klein] 系列小模型）**

```python
# text_encoder.py
class Qwen3Embedder(nn.Module):
    def __init__(self, model_spec):
        self.model = AutoModelForCausalLM.from_pretrained(model_spec, ...)
        self.tokenizer = AutoTokenizer.from_pretrained(model_spec)
```

- **模型**：Qwen3-4B-FP8 或 Qwen3-8B-FP8，纯文本 LLM
- **输出**：提取第 9、18、27 层的隐藏状态
  - Klein 4B 使用 Qwen3-4B：输出维度 `(B, 512, 3×2560) = (B, 512, 7680)`
  - Klein 9B 使用 Qwen3-8B：输出维度 `(B, 512, 3×4096) = (B, 512, 12288)`

#### 子网络 2：VAE Encoder

```
输入图像 [B, 3, H, W]
    → conv_in (3→128)
    → 4个下采样层级 (ResnetBlock×2 + Downsample) → 空间下采样8x
    → mid (ResnetBlock → AttnBlock → ResnetBlock)
    → norm_out → Swish → conv_out (→64通道)
    → quant_conv (64→64)
    → 取均值 chunk (64→32通道)
    → Patch重排 (32通道, H/8, W/8) → (128通道, H/16, W/16)
    → BatchNorm 归一化
输出 latent [B, 128, H/16, W/16]
```

#### 子网络 3：VAE Decoder

```
输入 latent [B, 128, H/16, W/16]
    → BatchNorm 反归一化
    → Unpatch重排 (128通道, H/16, W/16) → (32通道, H/8, W/8)
    → post_quant_conv (32→32)
    → conv_in (32→block_in)
    → mid (ResnetBlock → AttnBlock → ResnetBlock)
    → 4个上采样层级 (ResnetBlock×3 + Upsample) → 空间上采样8x
    → norm_out → Swish → conv_out (→3通道)
输出图像 [B, 3, H, W]
```

#### 子网络 4：Flow Matching DIT（Flux2 主模型）

**FLUX.2 [dev] 32B 参数配置：**

| 组件 | 参数 |
|------|------|
| `in_channels` | 128 |
| `context_in_dim` | 15360（Mistral 3层×5120） |
| `hidden_size` | 6144 |
| `num_heads` | 48 |
| `depth`（双流块） | 8 |
| `depth_single_blocks`（单流块） | 48 |
| `axes_dim` | [32, 32, 32, 32]（4D RoPE） |
| `mlp_ratio` | 3.0 |
| `use_guidance_embed` | True |

**内部组件：**

1. **`img_in`**：`nn.Linear(128, 6144)`，将 latent token 投影到隐藏空间
2. **`txt_in`**：`nn.Linear(15360, 6144)`，将文本 embedding 投影到隐藏空间
3. **`time_in`**：`MLPEmbedder(256, 6144)`，时间步嵌入
4. **`guidance_in`**：`MLPEmbedder(256, 6144)`，guidance 嵌入（可选）
5. **`pe_embedder`**：`EmbedND`，4D RoPE 位置编码
6. **`double_stream_modulation_img/txt`**：全局 Modulation，产生 AdaLN 调制参数
7. **`single_stream_modulation`**：全局 Modulation
8. **`double_blocks`**：8 个 `DoubleStreamBlock`
9. **`single_blocks`**：48 个 `SingleStreamBlock`
10. **`final_layer`**：`LastLayer`，AdaLN + Linear 输出

**Klein 系列变体：**

| 变体 | hidden_size | num_heads | depth | single_blocks | text_encoder |
|------|------------|-----------|-------|---------------|-------------|
| Klein 4B | 3072 | 24 | 5 | 20 | Qwen3-4B |
| Klein 9B | 4096 | 32 | 8 | 24 | Qwen3-8B |
| Dev 32B | 6144 | 48 | 8 | 48 | Mistral-Small-3.2-24B |

---

## 问题4：模型网络结构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FLUX.2 完整模型架构                           │
│                                                                     │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│  │  输入图像     │    │  文本 Prompt      │    │  参考图像（可选） │   │
│  │  [3,H,W]     │    │  (字符串)         │    │  [3,H',W']      │   │
│  └──────┬───────┘    └────────┬─────────┘    └────────┬─────────┘   │
│         │                     │                       │             │
│         ▼                     ▼                       ▼             │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│  │  VAE Encoder │    │  Text Encoder    │    │  VAE Encoder     │   │
│  │  (卷积式AE)  │    │  (Mistral-Small  │    │  (同左，共享权重) │   │
│  │              │    │   或 Qwen3)      │    │                  │   │
│  └──────┬───────┘    └────────┬─────────┘    └────────┬─────────┘   │
│         │                     │                       │             │
│         ▼                     ▼                       ▼             │
│    latent noise          text_emb                 ref_tokens       │
│    [128,H/16,W/16]       [L,hidden]               [128,H'/16,W'/16]│
│         │                     │                       │             │
│    flatten + prc_img     prc_txt                  prc_img          │
│         │                     │                       │             │
│         ▼                     ▼                       ▼             │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              Flow Matching DIT (Flux2)                      │    │
│  │                                                             │    │
│  │  timestep ──→ time_in ──→ vec                               │    │
│  │  guidance ──→ guidance_in ──→ vec (累加)                     │    │
│  │                                                             │    │
│  │  vec ──→ double_stream_modulation_img ──→ mod_img           │    │
│  │  vec ──→ double_stream_modulation_txt ──→ mod_txt           │    │
│  │  vec ──→ single_stream_modulation ──→ mod_single            │    │
│  │                                                             │    │
│  │  x ──→ img_in ──→ img_tokens                               │    │
│  │  ctx ──→ txt_in ──→ txt_tokens                              │    │
│  │                                                             │    │
│  │  ┌───────────────────────────────────┐                      │    │
│  │  │  DoubleStreamBlock × 8            │                      │    │
│  │  │  (img分支 + txt分支，联合注意力)    │                      │    │
│  │  │  img: Norm→QKV→Attn→Proj→MLP      │                      │    │
│  │  │  txt: Norm→QKV→Attn→Proj→MLP      │                      │    │
│  │  │  AdaLN-Zero 调制 (共享mod)         │                      │    │
│  │  └───────────────────────────────────┘                      │    │
│  │                     │                                       │    │
│  │            cat(txt, img)                                    │    │
│  │                     ▼                                       │    │
│  │  ┌───────────────────────────────────┐                      │    │
│  │  │  SingleStreamBlock × 48           │                      │    │
│  │  │  (拼接后统一处理)                   │                      │    │
│  │  │  Norm→Linear1→[QKV,MLP]           │                      │    │
│  │  │  →Attn→cat(attn,mlp)→Linear2      │                      │    │
│  │  │  AdaLN-Zero 调制 (共享mod)         │                      │    │
│  │  └───────────────────────────────────┘                      │    │
│  │                     │                                       │    │
│  │         去掉txt tokens                                      │    │
│  │                     ▼                                       │    │
│  │  ┌───────────────────────────────────┐                      │    │
│  │  │  LastLayer (final_layer)          │                      │    │
│  │  │  AdaLN → Linear(6144→128)         │                      │    │
│  │  └───────────────────────────────────┘                      │    │
│  │                     │                                       │    │
│  └─────────────────────┼───────────────────────────────────────┘    │
│                        │                                            │
│                   v_pred (速度场预测)                                │
│                        │                                            │
│              欧拉步进: x = x + (t_prev - t_curr) * v_pred           │
│                        │                                            │
│              (重复 num_steps 步)                                     │
│                        │                                            │
│                        ▼                                            │
│                  scatter_ids (重排)                                  │
│                        │                                            │
│                        ▼                                            │
│              ┌──────────────────┐                                   │
│              │  VAE Decoder     │                                   │
│              │  (卷积式解码器)   │                                   │
│              └────────┬─────────┘                                   │
│                       │                                             │
│                       ▼                                             │
│                 输出图像 [3, H, W]                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 问题5：是否支持文生图和图像编辑

### 结论

| 功能 | 是否支持 | 代码依据 |
|------|---------|---------|
| **文生图（Text-to-Image）** | ✅ 支持 | `cli.py` 中不提供 `input_images` 时自动进入文生图模式 |
| **图像+文本提示编辑** | ✅ 支持 | `cli.py` 中提供 `input_images` 时进入编辑模式 |
| **单参考图编辑** | ✅ 支持 | `input_images` 传入 1 张图片 |
| **多参考图编辑** | ✅ 支持 | `input_images` 传入多张图片 |

### 代码依据

```python
# cli.py 第420-443行
img_ctx = [Image.open(input_image) for input_image in cfg.input_images]

with torch.no_grad():
    ref_tokens, ref_ids = encode_image_refs(ae, img_ctx)
    # ref_tokens 为 None 时 → 文生图
    # ref_tokens 不为 None 时 → 图像编辑
```

```python
# sampling.py 第52-90行
def encode_image_refs(ae, img_ctx):
    if not img_ctx:
        return None, None  # 没有参考图 → 文生图模式

    # 编码参考图为 latent tokens
    for img in img_ctx_prep:
        encoded = ae.encode(img[None].cuda())[0]
        encoded_refs.append(encoded)

    # 为每张参考图分配不同的时间坐标 t_off
    t_off = [scale + scale * t for t in torch.arange(0, len(encoded_refs))]
    ref_tokens, ref_ids = listed_prc_img(encoded_refs, t_coord=t_off)
    return ref_tokens, ref_ids
```

---

## 问题6：文生图流程图

### 文生图时数据流：

```
输入数据:
  ├── 文本 Prompt (字符串)
  ├── 图像尺寸 (width, height)
  ├── 采样参数 (num_steps, guidance, seed)
  └── 无参考图像 (input_images=[])

═══════════════════════════════════════════════════════════════

步骤1: 文本编码
  ┌─────────────────────┐
  │ 文本 Prompt          │
  │ "a photo of a cat"  │
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────────────────┐
  │ Text Encoder                    │
  │ (Mistral-Small-3.2-24B          │
  │  或 Qwen3-4B/8B)               │
  │                                 │
  │ 1. Tokenize + Chat Template     │
  │ 2. Forward pass                 │
  │ 3. 提取第10/20/30层隐藏状态      │
  │ 4. Stack + Rearrange            │
  │    → (B, 512, 15360)            │
  └──────────┬──────────────────────┘
             │
             ▼
  ctx [B, 512, 15360]
             │
  batched_prc_txt()
  → ctx [B, 512, 15360], ctx_ids [B, 512, 4]

═══════════════════════════════════════════════════════════════

步骤2: 噪声生成
  ┌─────────────────────┐
  │ torch.randn()        │
  │ shape=(1, 128,       │
  │   H/16, W/16)        │
  │ seed=指定种子         │
  └──────────┬──────────┘
             │
  batched_prc_img()
  → x [B, H/16*W/16, 128], x_ids [B, H/16*W/16, 4]

═══════════════════════════════════════════════════════════════

步骤3: 参考图编码
  encode_image_refs(ae, [])
  → ref_tokens = None, ref_ids = None
  （无参考图，纯文生图）

═══════════════════════════════════════════════════════════════

步骤4: Flow Matching 去噪循环 (denoise函数)
  timesteps = get_schedule(num_steps, image_seq_len)
  例如: [1.0, 0.98, 0.96, ..., 0.02, 0.0]

  for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
    ┌─────────────────────────────────────────┐
    │ Flux2.forward()                         │
    │                                         │
    │ 输入:                                    │
    │   x = noisy_latent [B, L_img, 128]      │
    │   x_ids = position_ids [B, L_img, 4]    │
    │   timesteps = t_curr (标量)              │
    │   ctx = text_emb [B, 512, 15360]        │
    │   ctx_ids = [B, 512, 4]                 │
    │   guidance = 4.0 (标量)                  │
    │                                         │
    │ 内部计算:                                │
    │   1. vec = time_in(t) + guidance_in(g)  │
    │   2. mod_img, mod_txt, mod_single       │
    │      = modulation(vec) [全局共享]         │
    │   3. img = img_in(x)                    │
    │   4. txt = txt_in(ctx)                  │
    │   5. pe = pe_embedder(ids)              │
    │   6. 8x DoubleStreamBlock               │
    │   7. cat(txt, img) → 48x SingleStream   │
    │   8. 取img部分 → final_layer            │
    │                                         │
    │ 输出: v_pred [B, L_img, 128]            │
    └──────────────────┬──────────────────────┘
                       │
    x = x + (t_prev - t_curr) * v_pred  (欧拉步进)

═══════════════════════════════════════════════════════════════

步骤5: Latent → 图像
  ┌─────────────────────────────────────────┐
  │ scatter_ids(x, x_ids)                   │
  │ 将 token 序列重排为空间 latent            │
  │ → [B, 128, H/16, W/16]                 │
  └──────────────────┬──────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────┐
  │ VAE Decoder (ae.decode)                 │
  │                                         │
  │ 1. inv_normalize (BatchNorm反归一化)     │
  │ 2. unpatch: (128,H/16,W/16)             │
  │    → (32,H/8,W/8)                       │
  │ 3. post_quant_conv                      │
  │ 4. conv_in → mid → up (4级上采样)        │
  │ 5. norm_out → swish → conv_out          │
  │                                         │
  │ 输出: 图像 [B, 3, H, W]                 │
  └──────────────────┬──────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────┐
  │ 后处理                                   │
  │ 1. clamp(-1, 1)                         │
  │ 2. 转换为 PIL Image                      │
  │ 3. 保存为 PNG                            │
  └─────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════

最终输出: 由 VAE Decoder 子网络输出 RGB 图像
```

---

## 问题7：图像+文本提示图像编辑流程图

### 图像编辑时数据流：

```
输入数据:
  ├── 文本 Prompt (编辑指令, 如 "change the hat to red")
  ├── 参考图像 (1张或多张 PIL Image)
  ├── 图像尺寸 (width, height)
  └── 采样参数 (num_steps, guidance, seed)

═══════════════════════════════════════════════════════════════

步骤1: 文本编码 (与文生图相同)
  ┌─────────────────────┐
  │ 编辑指令 Prompt       │
  └──────────┬──────────┘
             ▼
  ┌─────────────────────────────────┐
  │ Text Encoder                    │
  │ (Mistral-Small 或 Qwen3)       │
  │ → ctx [B, 512, 15360]          │
  └──────────┬──────────────────────┘
             │
  batched_prc_txt()
  → ctx, ctx_ids

═══════════════════════════════════════════════════════════════

步骤2: 参考图像编码
  ┌──────────────────────────────────┐
  │ 参考图像列表                       │
  │ [img1.jpg, img2.jpg, ...]        │
  └──────────┬───────────────────────┘
             │
             ▼
  ┌─────────────────────────────────────────┐
  │ encode_image_refs(ae, img_ctx)          │
  │                                         │
  │ 对每张参考图:                              │
  │   1. default_prep: RGB转换→尺寸限制        │
  │      →裁剪为16的倍数→归一化到[-1,1]          │
  │   2. VAE Encoder: ae.encode(img)         │
  │      → latent [128, H'/16, W'/16]       │
  │   3. prc_img: flatten + 生成位置ID         │
  │      → tokens [H'W'/256, 128]            │
  │      → ids [H'W'/256, 4]                 │
  │      (每张图的时间坐标 t 不同:              │
  │       t=10, 20, 30... 用于区分)            │
  │                                          │
  │ 拼接所有参考图的 tokens 和 ids:              │
  │   ref_tokens [1, total_ref_L, 128]       │
  │   ref_ids [1, total_ref_L, 4]            │
  └──────────┬──────────────────────────────┘
             │
             ▼
  ref_tokens, ref_ids (参考图latent tokens)

═══════════════════════════════════════════════════════════════

步骤3: 噪声生成 (与文生图相同)
  torch.randn(1, 128, H/16, W/16)
  → x [B, L_img, 128], x_ids [B, L_img, 4]

═══════════════════════════════════════════════════════════════

步骤4: Flow Matching 去噪循环

【方式A: 标准去噪 (denoise函数)】
  当 ref_tokens 不为 None 时:
  img_input = cat(x, ref_tokens)          ← 拼接噪声和参考图tokens
  img_input_ids = cat(x_ids, ref_ids)     ← 拼接位置ID

  for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
    ┌─────────────────────────────────────────┐
    │ Flux2.forward()                         │
    │                                         │
    │ 输入:                                    │
    │   x = cat(noisy_img, ref_tokens)        │
    │       [B, L_img+L_ref, 128]             │
    │   ctx = text_emb [B, 512, 15360]        │
    │   timesteps, guidance                    │
    │                                         │
    │ 注意: 参考图tokens与噪声tokens一起         │
    │ 进入DIT，共同参与注意力计算                  │
    │                                         │
    │ 输出: v_pred [B, L_img+L_ref, 128]      │
    │ → 只取前 L_img 部分作为预测                │
    └──────────────────┬──────────────────────┘
                       │
    x = x + (t_prev - t_curr) * v_pred[:, :L_img]

【方式B: KV Cache 加速去噪 (denoise_cached函数, Klein-9B-KV)】
  Step 0:
    ┌─────────────────────────────────────────┐
    │ Flux2.forward_kv_extract()              │
    │                                         │
    │ 输入: x(噪声), x_seq_concat(参考图)       │
    │ 1. 拼接 [ref_tokens, img_tokens]        │
    │ 2. 为ref生成固定timestep (t=0.0)         │
    │ 3. 分别计算ref和img的modulation           │
    │ 4. DoubleStreamBlock: 提取ref的KV cache   │
    │ 5. SingleStreamBlock: 提取ref的KV cache   │
    │ 6. 因果注意力:                            │
    │    - txt+img: 可以注意到所有tokens          │
    │    - ref: 只能注意到自己 (自注意力)          │
    │                                         │
    │ 输出: v_pred, kv_cache                   │
    │   kv_cache = {                           │
    │     "double_blocks": [每层的k_ref, v_ref], │
    │     "single_blocks": [每层的k_ref, v_ref], │
    │     "num_ref_tokens": N                  │
    │   }                                     │
    └──────────────────┬──────────────────────┘

  Steps 1+:
    ┌─────────────────────────────────────────┐
    │ Flux2.forward_kv_cached()               │
    │                                         │
    │ 输入: x(噪声, 不含ref), kv_cache         │
    │ 1. 只输入 img_tokens (无ref在序列中)       │
    │ 2. 从kv_cache中注入ref的K/V              │
    │ 3. 注意力计算:                            │
    │    k_all = cat(k_txt, k_ref_cached, k_img)│
    │    v_all = cat(v_txt, v_ref_cached, v_img)│
    │ 4. 大幅减少计算量                         │
    │                                         │
    │ 输出: v_pred                             │
    └──────────────────┬──────────────────────┘

═══════════════════════════════════════════════════════════════

步骤5: Latent → 图像 (与文生图相同)
  scatter_ids → VAE Decoder → 输出图像

═══════════════════════════════════════════════════════════════

最终输出: 由 VAE Decoder 子网络输出 编辑后的 RGB 图像

═══════════════════════════════════════════════════════════════

因果注意力机制详解 (causal_attn_fn):

  序列布局: [txt_tokens, ref_tokens, img_tokens]

  注意力可见性矩阵:
  ┌────────┬─────────┬────────────┬────────────┐
  │        │ txt (K) │ ref (K)    │ img (K)    │
  ├────────┼─────────┼────────────┼────────────┤
  │txt (Q) │   ✅    │    ✅      │    ✅      │
  │ref (Q) │   ❌    │    ✅      │    ❌      │
  │img (Q) │   ✅    │    ✅      │    ✅      │
  └────────┴─────────┴────────────┴────────────┘

  ref_tokens 只能看到自己 → 防止信息泄漏
  img_tokens 可以看到所有 → 允许从ref中提取信息
```

---

## 问题8：相比FLUX.1模型的创新点

### 基于代码实现分析，FLUX.2 相比 FLUX.1 具有以下创新点和改进：

#### 1. 统一的生成与编辑模型（最核心创新）

FLUX.1 只支持文生图（text-to-image）。FLUX.2 在**一个模型**中同时支持：
- 文生图（Text-to-Image）
- 单参考图编辑（Single-ref Editing）
- 多参考图编辑（Multi-ref Editing）

**无需微调或额外训练**，通过在去噪过程中将参考图的 latent tokens 拼接到输入序列中实现。

```python
# sampling.py - denoise函数中
if img_cond_seq is not None:
    img_input = torch.cat((img_input, img_cond_seq), dim=1)
    img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)
```

#### 2. 改进的 VAE（AutoEncoder）

- **更大的 latent 通道数**：`z_channels=32`（FLUX.1 为 16），经 patch 后有效通道数 128
- **Patch 空间重排**：使用 `ps=[2,2]` 将空间维度折叠到通道维度，有效下采样 16x（FLUX.1 为 8x）
- **BatchNorm 归一化**：使用 `BatchNorm2d`（`affine=False`）对 latent 进行标准化，替代 FLUX.1 的简单缩放因子
- README 明确声明"FLUX.2 autoencoder has considerably improved over the FLUX.1 autoencoder"

```python
# FLUX.2 的 BatchNorm 归一化
self.bn = torch.nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=False, track_running_stats=True)
```

#### 3. 文本编码器升级：使用多模态大语言模型

FLUX.1 使用 **CLIP + T5-XXL** 双文本编码器。FLUX.2 完全替换为：

- **FLUX.2 [dev]**：Mistral-Small-3.2-24B-Instruct（**多模态 LLM**，支持图文理解）
- **FLUX.2 [klein]**：Qwen3-4B/8B（纯文本 LLM）

关键改进：
- 从多个中间层（第10/20/30层）提取隐藏状态并拼接，获得**多尺度语义特征**
- 输出维度从 T5 的 4096 提升到 15360（Mistral）或 7680/12288（Qwen3）
- 多模态LLM可同时理解图像和文本，支持 Prompt Upsampling

```python
OUTPUT_LAYERS_MISTRAL = [10, 20, 30]
out = torch.stack([output.hidden_states[k] for k in OUTPUT_LAYERS_MISTRAL], dim=1)
return rearrange(out, "b c l d -> b l (c d)")  # 拼接多层特征
```

#### 4. 因果注意力机制（Causal Attention for Reference Images）

FLUX.2 引入了**因果注意力**（`causal_attn_fn`）用于处理参考图像：

- 参考图 tokens **只能注意到自己**（self-attention only）
- 生成图 tokens 和文本 tokens **可以注意到所有** tokens（包括参考图）
- 这防止了参考图被去噪过程"污染"，同时允许生成图从参考图中提取信息

```python
# 参考图只能自注意力
attn_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref)
# 文本+生成图可以看到所有
attn_txt_img = F.scaled_dot_product_attention(q_txt_img, k_all, v_all)
```

#### 5. 参考图固定时间步（Reference Fixed Timestep）

在编辑模式下，参考图使用**固定时间步 t=0.0**（即"完全干净"的状态），而生成图使用当前去噪时间步。这通过分别生成不同的 modulation 参数实现：

```python
ref_vec = self.time_in(timestep_embedding(torch.full_like(timesteps, ref_fixed_timestep), 256))
# ref_fixed_timestep = 0.0
```

然后通过 `_blend_double_mods` 和 `_blend_single_mods` 将参考图和生成图的 modulation 参数按位置混合：

```python
# [ref_mod, img_mod] 位置混合
def _blend_mod_triple(img_m, ref_m, num_ref, seq_len):
    blended.append(
        torch.cat([rm.expand(B, num_ref, -1), im.expand(B, seq_len, -1)[:, num_ref:, :]], dim=1)
    )
```

#### 6. KV Cache 加速推理

FLUX.2 引入了 **KV Cache 机制**用于加速参考图的推理：

- 第一步（`forward_kv_extract`）：完整前向传播，同时提取参考图的 K/V 缓存
- 后续步骤（`forward_kv_cached`）：复用缓存的 K/V，无需再次计算参考图

```python
# 第一步：提取KV cache
cache = {
    "k_ref": k[:, :, ref_start:ref_end, :].clone(),
    "v_ref": v[:, :, ref_start:ref_end, :].clone(),
}

# 后续步骤：注入缓存
k_all = torch.cat([k_txt, k_ref_cached, k_img], dim=2)
v_all = torch.cat([v_txt, v_ref_cached, v_img], dim=2)
```

这使得 Klein-9B-KV 在多参考图编辑时比不使用 KV cache 的版本**快数倍**。

#### 7. 全局共享 Modulation（Global Shared Modulation）

FLUX.1 中每个 Transformer Block 有自己独立的 Modulation 层。FLUX.2 改为**全局共享**：

- `double_stream_modulation_img`：所有双流块共享一个 img modulation
- `double_stream_modulation_txt`：所有双流块共享一个 txt modulation
- `single_stream_modulation`：所有单流块共享一个 modulation

```python
# 全局 modulation（在 Flux2.__init__ 中定义一次）
self.double_stream_modulation_img = Modulation(self.hidden_size, double=True)
self.double_stream_modulation_txt = Modulation(self.hidden_size, double=True)
self.single_stream_modulation = Modulation(self.hidden_size, double=False)

# 在 forward 中计算一次，所有块共享
double_block_mod_img = self.double_stream_modulation_img(vec)
double_block_mod_txt = self.double_stream_modulation_txt(vec)
single_block_mod, _ = self.single_stream_modulation(vec)
```

这显著减少了参数量（从 depth×Modulation 减少为 3 个 Modulation）。

#### 8. 4D RoPE 位置编码

FLUX.1 使用 3D RoPE（对应 `axes_dim=[16, 56, 56]`，维度分配给 t, h, w）。FLUX.2 改为 **4D RoPE**：

```python
axes_dim: list[int] = [32, 32, 32, 32]  # 4个维度：t, h, w, l
```

- `t`：时间维度（用于区分参考图和生成图的时间坐标）
- `h`：高度
- `w`：宽度
- `l`：序列/层级维度（文本 token 的序列位置）

4D 位置编码使模型能够更好地区分不同类型的 token（文本 vs 图像 vs 参考图）。

#### 9. SiLU Gated Activation（SiLUActivation）

FLUX.1 使用标准 GELU 激活函数。FLUX.2 改为 **SiLU Gated Activation**：

```python
class SiLUActivation(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return self.gate_fn(x1) * x2  # SiLU gating
```

MLP 使用 `hidden_dim * 2` 的中间维度，然后 chunk 成两半做 gated 乘法，与 LLaMA/Mistral 等模型的 SwiGLU 类似。

#### 10. 无 Bias 设计

FLUX.2 中几乎所有的线性层都设置 `bias=False`：

```python
self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=False)
self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size, bias=False)
# Modulation
self.lin = nn.Linear(dim, self.multiplier * dim, bias=not disable_bias)  # disable_bias=True
# SelfAttention
self.qkv = nn.Linear(dim, dim * 3, bias=False)
self.proj = nn.Linear(dim, dim, bias=False)
```

这是现代 Transformer 的趋势，减少参数量并提高训练稳定性。

#### 11. Klein 系列蒸馏小模型

FLUX.2 引入了 Klein 系列模型（4B/9B），通过以下方式实现高效推理：

- **Step Distillation**（步骤蒸馏）：将 50 步推理压缩到 4 步
- **Guidance Distillation**（引导蒸馏）：将 CFG 的计算量减半
- **更小的模型尺寸**：4B/9B vs 32B

```python
# Klein 4B: 5个双流块 + 20个单流块
Klein4BParams: depth=5, depth_single_blocks=20, hidden_size=3072

# Klein 9B: 8个双流块 + 24个单流块
Klein9BParams: depth=8, depth_single_blocks=24, hidden_size=4096
```

#### 12. Prompt Upsampling 功能

FLUX.2 内置了 **Prompt Upsampling**（提示词增强）功能：

- **本地模式（local）**：使用同一个 Mistral-Small 模型的生成能力来增强用户的简短提示词
- **API 模式（openrouter）**：通过 OpenRouter API 调用外部 LLM（如 Pixtral-Large）来增强
- 区分文生图（T2I）和图编辑（I2I）的不同 upsampling 策略

```python
SYSTEM_MESSAGE_UPSAMPLING_T2I = """You are an expert prompt engineer for FLUX.2...
Rewrite user prompts to be more descriptive while strictly preserving their core subject and intent."""

SYSTEM_MESSAGE_UPSAMPLING_I2I = """You are FLUX.2..., an image-editing expert.
You convert editing requests into one concise instruction (50-80 words)."""
```

#### 13. 内容安全过滤机制

FLUX.2 集成了多层安全过滤：

- **NSFW 图像检测**：使用 `Falconsai/nsfw_image_detection` 分类器
- **版权和公众人物检测**：使用 Mistral-Small 的多模态理解能力
- **文本提示过滤**：检查提示词中的版权内容和公众人物提及
- **输出过滤**：对生成结果进行安全检查

```python
# 文本过滤
if mod_and_upsampling_model.test_txt(updates["prompt"]):
    print("Your prompt has been flagged...")

# 图像过滤（输入和输出）
if mod_and_upsampling_model.test_image(img):
    print("Your output has been flagged...")
```

#### 14. CFG（Classifier-Free Guidance）支持

FLUX.2 的 Base 模型（非蒸馏版）支持标准的 **Classifier-Free Guidance**：

```python
def denoise_cfg(model, img, img_ids, txt, txt_ids, timesteps, guidance, ...):
    img = torch.cat([img, img], dim=0)  # 复制输入
    # txt 已经是 cat([txt_empty, txt_prompt])
    ...
    pred_uncond, pred_cond = pred.chunk(2)
    pred = pred_uncond + guidance * (pred_cond - pred_uncond)  # CFG 公式
```

蒸馏版本则通过 `guidance_embed` 直接注入 guidance 值，无需双倍计算。

#### 15. 多参考图的时间坐标编码

FLUX.2 通过**不同的时间坐标**来区分多张参考图：

```python
# 每张参考图的时间坐标以 scale=10 为间隔
t_off = [scale + scale * t for t in torch.arange(0, len(encoded_refs))]
# 例如: 第1张图 t=10, 第2张图 t=20, 第3张图 t=30
# 生成图的 t=0（默认）
```

这允许模型通过 4D RoPE 位置编码来区分不同的参考图，并在注意力中正确处理它们的空间关系。

---

### 创新点总结表

| # | 创新点 | FLUX.1 | FLUX.2 |
|---|--------|--------|--------|
| 1 | 统一生成与编辑 | ❌ 仅文生图 | ✅ 文生图 + 单/多参考图编辑 |
| 2 | VAE 改进 | z=16ch, 8x下采样, scaling | z=32ch(128 patched), 16x下采样, BatchNorm |
| 3 | 文本编码器 | CLIP + T5-XXL | Mistral-Small-3.2-24B (多模态LLM) 或 Qwen3 |
| 4 | 因果注意力 | ❌ 无 | ✅ 参考图自注意力 + 生成图全注意力 |
| 5 | 参考图固定时间步 | ❌ 无 | ✅ ref_fixed_timestep=0.0 |
| 6 | KV Cache 加速 | ❌ 无 | ✅ 参考图 KV 缓存复用 |
| 7 | 全局共享 Modulation | ❌ 每层独立 | ✅ 全局共享3个Modulation |
| 8 | 4D RoPE | 3D RoPE (t,h,w) | 4D RoPE (t,h,w,l) |
| 9 | SiLU Gated Activation | GELU | SiLU Gated (SwiGLU) |
| 10 | 无 Bias 设计 | 部分有 bias | 几乎全部 bias=False |
| 11 | Klein 蒸馏小模型 | ❌ 无 | ✅ 4B/9B 蒸馏模型 |
| 12 | Prompt Upsampling | ❌ 无 | ✅ 本地 + API 两种模式 |
| 13 | 内容安全过滤 | ❌ 无内置 | ✅ NSFW + 版权 + 公众人物过滤 |
| 14 | CFG 支持 | Guidance distilled | ✅ CFG + Guidance distilled 两种模式 |
| 15 | 多参考图时间坐标 | ❌ 无 | ✅ 不同时间坐标区分多参考图 |

---

> **报告完成时间**: 基于 flux2/ 目录全部源代码分析
>
> **分析的文件列表**:
> - `flux2/src/flux2/model.py` - DIT 模型定义
> - `flux2/src/flux2/autoencoder.py` - VAE 定义
> - `flux2/src/flux2/text_encoder.py` - 文本编码器
> - `flux2/src/flux2/sampling.py` - 采样和去噪逻辑
> - `flux2/src/flux2/util.py` - 模型加载和配置
> - `flux2/src/flux2/system_messages.py` - 系统提示词
> - `flux2/scripts/cli.py` - 命令行推理入口
> - `flux2/README.md` - 项目说明文档
> - `flux2/model_cards/FLUX.2-dev.md` - 模型卡片
