# Krea-2 (K2) 模型代码全面分析报告

> 本报告基于 `krea-2/` 目录下的全部源代码实现进行深入分析，所有结论均由代码逻辑推导得出。

> **代码实现根目录**：`/root/code/krea-2/`

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

### 结论：是的，Krea-2 使用了 VAE。使用的是 **Qwen-Image VAE**（`AutoencoderKLQwenImage`），既不同于 FLUX.1 的 VAE 结构，也不同于 FLUX.2 的 VAE 结构，属于第三方 Qwen 系列的 VAE。

### 代码分析依据

**1. VAE 定义（`autoencoder.py`）**

```python
# autoencoder.py 第6-22行
class QwenAutoencoder(nn.Module):
    """qwen-ae-f8-16c: the Qwen-Image VAE (f8, 16 latent channels)."""

    def __init__(self):
        super().__init__()
        from diffusers import AutoencoderKLQwenImage

        self.ae = AutoencoderKLQwenImage.from_pretrained("Qwen/Qwen-Image", subfolder="vae")
        self.compression = 8
        self.channels = 16
        self.register_buffer("latents_mean", torch.tensor(self.ae.latents_mean).view(1, -1, 1, 1, 1))
        self.register_buffer("latents_std", torch.tensor(self.ae.latents_std).view(1, -1, 1, 1, 1))
```

关键特征：
- **模型来源**：`Qwen/Qwen-Image` 的 `vae` 子文件夹，使用 `diffusers` 库的 `AutoencoderKLQwenImage` 类
- **下采样倍数**：`self.compression = 8`，即 8x 空间下采样
- **Latent 通道数**：`self.channels = 16`
- **Latent 归一化方式**：使用预计算的 `latents_mean` 和 `latents_std` 进行标准化

**2. Decode 方法**

```python
# autoencoder.py 第19-22行
def decode(self, x: Tensor) -> Tensor:
    x = rearrange(x, "b c h w -> b c 1 h w")        # 添加时间维度（伪3D）
    x = (x * self.latents_std) + self.latents_mean    # 反标准化
    return rearrange(self.ae.decode(x).sample, "b c 1 h w -> b c h w")  # 解码后移除时间维度
```

注意：
- decode 时先添加一个时间维度 `1`（`b c h w → b c 1 h w`），说明 `AutoencoderKLQwenImage` 原始设计支持视频（3D），但此处仅用于图像（时间维度为1）
- 反标准化使用 `x * std + mean`，与 FLUX.2 的 BatchNorm 反归一化不同
- **仅实现了 decode 方法**，没有 encode 方法，说明推理时只需要解码器

**3. 代码中没有 Encode 方法**

`QwenAutoencoder` 类中**只定义了 `decode` 方法**，没有 `encode` 方法。这意味着该 VAE 在推理管线中只负责将 latent 解码为像素图像，不需要将图像编码为 latent（因为不支持图像编辑）。

### 与 FLUX.1 和 FLUX.2 VAE 的对比

| 特征 | FLUX.1 VAE | FLUX.2 VAE | Krea-2 VAE |
|------|-----------|-----------|-----------|
| 类型 | 自定义卷积 AE | 自定义卷积 AE + BatchNorm | Qwen-Image VAE（`AutoencoderKLQwenImage`） |
| 来源 | 自研 | 自研 | 第三方（Qwen/Qwen-Image） |
| 下采样倍数 | 8x | 16x（8x + 2x Patch） | 8x |
| z_channels | 16 | 32（有效128） | 16 |
| Latent 归一化 | Scaling factor | BatchNorm（running stats） | 均值/标准差归一化 |
| 是否支持 Encode | ✅ | ✅ | ❌（仅 Decode） |
| 是否支持3D/视频 | ❌ | ❌ | 底层支持（伪3D调用） |

---

## 问题2：是否使用了flow_matching的DIT模型？单流还是双流？

### 结论：是的，使用了 flow matching 的 DIT 模型。Krea-2 使用的是**纯单流 DIT 模型**（`SingleStreamDiT`），不含双流 MMDIT 块。

### 代码分析依据

**1. 模型参数配置（`inference.py`）**

```python
# inference.py 第12-25行
single_mmdit_large_wide = SingleMMDiTConfig(
    features=6144,      # 隐藏层维度
    tdim=256,            # 时间步嵌入维度
    txtdim=2560,         # 文本编码维度（Qwen3-VL 单层维度）
    heads=48,            # 注意力头数
    kvheads=12,          # KV 头数（GQA）
    multiplier=4,        # MLP 扩展倍数
    layers=28,           # 单流块层数
    patch=2,             # Patch 大小
    channels=16,         # Latent 通道数
    txtheads=20,         # 文本融合注意力头数
    txtkvheads=20,       # 文本融合 KV 头数
    txtlayers=12,        # 文本编码器选取的层数
)
```

**2. 单流 DIT 主模型（`mmdit.py`，`SingleStreamDiT` 类）**

```python
# mmdit.py 第321-417行
class SingleStreamDiT(nn.Module):
    def __init__(self, config: SingleMMDiTConfig):
        # ...
        self.blocks = nn.ModuleList(
            [SingleStreamBlock(...) for _ in range(config.layers)]  # 28个单流块
        )
```

整个 DIT 模型**仅由 `SingleStreamBlock` 组成**，没有任何 `DoubleStreamBlock`。尽管配置类名为 `SingleMMDiTConfig`，但实际模型是纯单流架构。

**3. 单流块结构（`SingleStreamBlock`）**

```python
# mmdit.py 第293-318行
class SingleStreamBlock(nn.Module):
    def __init__(self, features, heads, multiplier, bias=False, kvheads=None):
        self.mod = DoubleSharedModulation(features)   # 6个调制参数
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(dim=features, heads=heads, bias=bias, kvheads=kvheads)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x, vec, freqs, mask=None):
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        x = x + pregate * self.attn(
            (1 + prescale) * self.prenorm(x) + preshift, freqs, mask
        )
        x = x + postgate * self.mlp(
            (1 + postscale) * self.postnorm(x) + postshift
        )
        return x
```

每个单流块内部包含：
- `DoubleSharedModulation`：产生 6 个调制参数（pre/post 各 scale, shift, gate）
- `RMSNorm` × 2：前归一化 + 后归一化
- `Attention`：自注意力（支持 GQA，heads=48, kvheads=12）
- `SwiGLU`：SwiGLU 激活的 MLP

**4. Flow Matching 去噪过程（`sampling.py`）**

```python
# sampling.py 第122-132行
# Euler integration of the flow ODE with CFG.
img = x
for tcurr, tprev in zip(ts[:-1], ts[1:]):
    t = torch.full((len(img),), tcurr, dtype=img.dtype, device=img.device)
    cond = model(img=img, context=txt, t=t, pos=pos, mask=mask)
    if cfg:
        uncond = model(img=img, context=untxt, t=t, pos=unpos, mask=unmask)
        v = cond + guidance * (cond - uncond)
    else:
        v = cond
    img = img + (tprev - tcurr) * v
```

使用标准的 **flow matching 欧拉步进**：`x_{t-1} = x_t + (t_prev - t_curr) * v_pred`，时间步从 1 递减到 0，模型预测速度场 `v`。

**5. 前向传播中文本和图像的合并方式**

```python
# mmdit.py 第395-416行 （SingleStreamDiT.forward）
def forward(self, img, context, t, pos, mask=None):
    img = self.first(img)                    # 图像 token 投影
    t = self.tmlp(temb(t, ...))              # 时间步嵌入
    tvec = self.tproj(t)                     # 调制向量

    context = self.txtfusion(context, mask=txtmask)  # 文本多层融合
    context = self.txtmlp(context)                   # 文本投影到 hidden_size

    combined = torch.cat((context, img), dim=1)      # 拼接文本和图像
    # ...
    for block in self.blocks:
        combined = block(combined, tvec, freqs, mask)  # 所有 block 处理拼接后的序列

    output = final[:, txtlen : txtlen + imglen, :]  # 只取图像部分
    return output
```

**文本和图像 token 在进入 Transformer 之前就已经拼接在一起**，然后一起通过所有 28 个 `SingleStreamBlock` 进行处理。这是典型的**单流架构**。

### 总结

| 特征 | 说明 |
|------|------|
| 是否 Flow Matching | ✅ 是，使用欧拉步进的 rectified flow |
| 双流 MMDIT | ❌ 没有 `DoubleStreamBlock` |
| 单流 DIT | ✅ `SingleStreamBlock` × 28 层 |
| 架构类型 | 纯单流（文本+图像拼接后共同处理） |
| 位置编码 | 3D RoPE（t, h, w） |
| 注意力类型 | GQA（48 heads, 12 kv_heads） |
| Modulation | 每层独立的 `DoubleSharedModulation` |

---

## 问题3：具体网络结构和子网络

### Krea-2 包含以下 4 个子网络结构：

#### 子网络 1：文本编码器（Qwen3-VL Conditioner + TextFusionTransformer）

Krea-2 的文本编码器由两部分组成：

**（1）Qwen3-VL 多模态大语言模型（`encoder.py`，`Qwen3VLConditioner`）**

```python
# encoder.py 第19-76行
class Qwen3VLConditioner(torch.nn.Module):
    def __init__(self, version, max_length=512, select_layers=(2,5,8,11,14,17,20,23,26,29,32,35)):
        self.qwen = Qwen3VLForConditionalGeneration.from_pretrained(version)
        self.tokenizer = AutoTokenizer.from_pretrained(version)
        self.processor = Qwen2TokenizerFast.from_pretrained(version)
        self.max_length = max_length
        self.select_layers = select_layers  # 选取12个层的隐藏状态
```

- **模型**：`Qwen3-VL-4B-Instruct`，一个 4B 参数的**视觉语言模型**（VLM）
- **选取层**：从 36 层中均匀选取 12 层（第 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35 层）
- **输出维度**：`(B, L, 12, 2560)` — 12 个层的隐藏状态，每层 2560 维
- **最大序列长度**：512 tokens
- **Prompt 模板**：使用特定的系统提示模板包裹用户输入
- **仅使用文本编码能力**：虽然是 VLM，但推理代码中只传入文本，不传入图像

```python
# encoder.py 第40-76行
def forward(self, text: list[str]) -> tuple[Tensor, Tensor]:
    # 添加 prompt 模板前缀和后缀
    text = [self.prompt_template_encode_prefix + item for item in text]
    # ...
    states = self.qwen(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)
    
    # 选取 12 层隐藏状态并堆叠
    hiddens = torch.stack(
        [states.hidden_states[i] for i in self.select_layers], dim=2
    )  # (B, L+prefix, 12, 2560)
    hiddens = hiddens[:, prefix_idx:]  # 去掉 prompt 模板前缀
    mask = mask[:, prefix_idx:]
    return hiddens, mask  # (B, L, 12, 2560), (B, L)
```

**（2）TextFusionTransformer（`mmdit.py`，`TextFusionTransformer`）**

这是 Krea-2 的独特设计——一个专门的文本融合子网络，将 12 层的文本隐藏状态融合为单一表示：

```python
# mmdit.py 第251-290行
class TextFusionTransformer(torch.nn.Module):
    def __init__(self, num_txt_layers, txt_dim, heads, multiplier, bias, kvheads):
        # 2 个 layerwise blocks：跨层融合
        self.layerwise_blocks = torch.nn.ModuleList(
            [TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)]
        )
        # 线性投影：12层 → 1层
        self.projector = torch.nn.Linear(num_txt_layers, 1, bias=False)
        # 2 个 refiner blocks：序列级别精炼
        self.refiner_blocks = torch.nn.ModuleList(
            [TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)]
        )

    def forward(self, x, mask=None):
        b, l, n, d = x.shape  # (batch, seq_len, num_layers=12, dim=2560)
        x = x.reshape(b * l, n, d)  # 每个 token 位置独立处理12层
        for block in self.layerwise_blocks:
            x = block(x)              # 跨层注意力融合
        x = rearrange(x, "(b l) n d -> b l d n", b=b, l=l)
        x = self.projector(x)         # 12层投影为1层：(B, L, D, 12) → (B, L, D, 1)
        x = x.squeeze(-1)             # (B, L, D)
        for block in self.refiner_blocks:
            x = block(x, mask=mask)   # 序列级别精炼
        return x  # (B, L, 2560)
```

融合过程：
1. `layerwise_blocks` × 2：对每个 token 位置的 12 层隐藏状态做跨层注意力融合
2. `projector`：线性投影将 12 层压缩为 1 层
3. `refiner_blocks` × 2：对融合后的序列做全局注意力精炼

**TextFusionBlock 结构：**

```python
# mmdit.py 第229-248行
class TextFusionBlock(torch.nn.Module):
    def __init__(self, features, heads, multiplier, bias, kvheads):
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(dim=features, heads=heads, bias=bias, kvheads=kvheads)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x, mask=None):
        x = x + self.attn(self.prenorm(x), mask=mask)
        x = x + self.mlp(self.postnorm(x))
        return x
```

然后通过 `txtmlp` 将 2560 维文本特征投影到 6144 维隐藏空间：

```python
# mmdit.py 第367-372行
self.txtmlp = nn.Sequential(
    RMSNorm(config.txtdim),           # RMSNorm(2560)
    nn.Linear(config.txtdim, config.features),  # 2560 → 6144
    nn.GELU(approximate="tanh"),
    nn.Linear(config.features, config.features),  # 6144 → 6144
)
```

#### 子网络 2：VAE Decoder（QwenAutoencoder）

```
输入 latent [B, 16, H/8, W/8]
    → rearrange: 添加伪时间维度 → [B, 16, 1, H/8, W/8]
    → 反标准化: x * latents_std + latents_mean
    → AutoencoderKLQwenImage.decode()（Qwen-Image VAE 内部解码器）
    → rearrange: 移除伪时间维度 → [B, 3, H, W]
输出图像 [B, 3, H, W]
```

注意：此 VAE **仅包含解码器**，不包含编码器。

#### 子网络 3：单流 DIT（SingleStreamDiT）

**参数配置（large_wide）：**

| 组件 | 参数 |
|------|------|
| `features`（隐藏维度） | 6144 |
| `tdim`（时间步嵌入维度） | 256 |
| `txtdim`（文本维度） | 2560 |
| `heads`（注意力头数） | 48 |
| `kvheads`（KV 头数） | 12（GQA，4:1 比例） |
| `multiplier`（MLP 扩展） | 4 |
| `layers`（单流块数） | 28 |
| `patch`（Patch 大小） | 2 |
| `channels`（latent 通道数） | 16 |

**内部组件列表：**

1. **`posemb`**（`PositionalEncoding`）：3D RoPE 位置编码
2. **`first`**（`nn.Linear`）：`channels * patch² = 16*4 = 64 → 6144`，Patchify + 投影
3. **`blocks`**：28 个 `SingleStreamBlock`，每个包含：
   - `DoubleSharedModulation`：6 个调制参数（pre/post 的 scale, shift, gate）
   - `RMSNorm` × 2
   - `Attention`（GQA：48 heads, 12 kv_heads，带 Gated Output）
   - `SwiGLU` MLP
4. **`tmlp`**（`nn.Sequential`）：时间步嵌入 MLP，`256 → 6144 → 6144`（GELU）
5. **`tproj`**（`nn.Sequential`）：调制向量投影，`6144 → 6144*6 = 36864`（GELU）
6. **`txtfusion`**（`TextFusionTransformer`）：文本多层融合子网络
7. **`txtmlp`**（`nn.Sequential`）：文本投影，`RMSNorm → 2560 → 6144 → 6144`
8. **`last`**（`LastLayer`）：最终输出层，`RMSNorm + SimpleModulation + Linear(6144 → 64)`

#### 子网络 4：时间步嵌入网络

```python
# mmdit.py 第52-69行 (temb 函数)
# 正弦余弦嵌入，输入为标量 t，输出为 (B, 1, 256) 维

# mmdit.py 第354-358行 (tmlp)
self.tmlp = nn.Sequential(
    nn.Linear(config.tdim, config.features),   # 256 → 6144
    nn.GELU(approximate="tanh"),
    nn.Linear(config.features, config.features),  # 6144 → 6144
)

# mmdit.py 第375-377行 (tproj)
self.tproj = nn.Sequential(
    nn.GELU(approximate="tanh"),
    nn.Linear(config.features, config.features * 6)  # 6144 → 36864
)
```

时间步嵌入经过 `temb → tmlp → tproj`，产生每个 `SingleStreamBlock` 所需的 6 个调制参数。

---

## 问题4：模型网络结构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Krea-2 (K2) 完整模型架构                            │
│                                                                         │
│  ┌──────────────────┐                  ┌──────────────────┐              │
│  │  文本 Prompt       │                  │  时间步 t (标量)  │              │
│  │  (字符串)          │                  │  [0, 1]          │              │
│  └────────┬─────────┘                  └────────┬─────────┘              │
│           │                                     │                        │
│           ▼                                     ▼                        │
│  ┌──────────────────────────┐          ┌──────────────────┐              │
│  │  子网络1: 文本编码器       │          │  子网络4: 时间步嵌入│              │
│  │                          │          │                  │              │
│  │  ┌────────────────────┐  │          │  temb(t, 256)    │              │
│  │  │ Qwen3-VL-4B-Instruct│  │          │      │          │              │
│  │  │ (VLM，仅用文本编码)  │  │          │      ▼          │              │
│  │  │ 选取12层隐藏状态     │  │          │  tmlp(256→6144) │              │
│  │  │ → (B,L,12,2560)    │  │          │      │          │              │
│  │  └─────────┬──────────┘  │          │      ▼          │              │
│  │            │              │          │  tproj(→36864)  │              │
│  │            ▼              │          │  → tvec (调制向量)│              │
│  │  ┌────────────────────┐  │          └────────┬─────────┘              │
│  │  │ TextFusionTransformer│ │                   │                        │
│  │  │                    │  │                   │                        │
│  │  │ layerwise_blocks×2 │  │                   │                        │
│  │  │ (跨层注意力融合)     │  │                   │                        │
│  │  │      │             │  │                   │                        │
│  │  │ projector(12→1)    │  │                   │                        │
│  │  │      │             │  │                   │                        │
│  │  │ refiner_blocks×2   │  │                   │                        │
│  │  │ (序列精炼)          │  │                   │                        │
│  │  │ → (B,L,2560)       │  │                   │                        │
│  │  └─────────┬──────────┘  │                   │                        │
│  │            │              │                   │                        │
│  │            ▼              │                   │                        │
│  │  txtmlp(2560→6144)       │                   │                        │
│  │  → context (B,L,6144)    │                   │                        │
│  └────────────┬─────────────┘                   │                        │
│               │                                  │                        │
│               │     ┌────────────────────┐       │                        │
│               │     │  随机噪声            │       │                        │
│               │     │  (B,16,H/8,W/8)    │       │                        │
│               │     └─────────┬──────────┘       │                        │
│               │               │                  │                        │
│               │        Patchify + first           │                        │
│               │        (64→6144)                  │                        │
│               │               │                  │                        │
│               │               ▼                  │                        │
│               │        img (B,L_img,6144)        │                        │
│               │               │                  │                        │
│               ▼               ▼                  │                        │
│        ┌──────────────────────────────┐           │                        │
│        │  cat(context, img) → combined│           │                        │
│        │  (B, L_txt+L_img, 6144)      │           │                        │
│        └──────────────┬───────────────┘           │                        │
│                       │                          │                        │
│                       ▼                          ▼                        │
│  ┌───────────────────────────────────────────────────────────────────┐    │
│  │               子网络3: 单流 DIT (SingleStreamDiT)                  │    │
│  │                                                                   │    │
│  │  ┌─────────────────────────────────────────────────────────────┐  │    │
│  │  │  SingleStreamBlock × 28 层                                  │  │    │
│  │  │                                                             │  │    │
│  │  │  每层结构:                                                   │  │    │
│  │  │  tvec ──→ DoubleSharedModulation ──→ 6个调制参数              │  │    │
│  │  │           (prescale, preshift, pregate,                     │  │    │
│  │  │            postscale, postshift, postgate)                  │  │    │
│  │  │                                                             │  │    │
│  │  │  x ──→ RMSNorm ──→ AdaLN调制 ──→ Attention(GQA) ──→        │  │    │
│  │  │        gate(pregate) ──→ 残差连接                            │  │    │
│  │  │                                                             │  │    │
│  │  │  x ──→ RMSNorm ──→ AdaLN调制 ──→ SwiGLU MLP ──→            │  │    │
│  │  │        gate(postgate) ──→ 残差连接                           │  │    │
│  │  │                                                             │  │    │
│  │  │  (附: RoPE 3D 位置编码应用于 Q, K)                           │  │    │
│  │  └─────────────────────────────────────────────────────────────┘  │    │
│  │                        │                                          │    │
│  │           去掉 txt tokens，只保留 img tokens                       │    │
│  │                        │                                          │    │
│  │                        ▼                                          │    │
│  │  ┌─────────────────────────────────────────────────┐              │    │
│  │  │  LastLayer                                      │              │    │
│  │  │  t ──→ SimpleModulation ──→ (scale, shift)      │              │    │
│  │  │  x ──→ RMSNorm ──→ AdaLN ──→ Linear(6144→64)   │              │    │
│  │  └──────────────────────┬──────────────────────────┘              │    │
│  │                         │                                         │    │
│  └─────────────────────────┼─────────────────────────────────────────┘    │
│                            │                                              │
│                     v_pred (速度场预测)                                     │
│                     (B, L_img, 64)                                        │
│                            │                                              │
│             Unpatchify: (B, L_img, 64) → (B, 16, H/8, W/8)              │
│                            │                                              │
│             欧拉步进: x = x + (t_prev - t_curr) * v_pred                  │
│                            │                                              │
│             (重复 num_steps 步)                                            │
│                            │                                              │
│                            ▼                                              │
│  ┌────────────────────────────────────────────────┐                       │
│  │  子网络2: VAE Decoder (QwenAutoencoder)         │                       │
│  │                                                │                       │
│  │  latent (B,16,H/8,W/8)                         │                       │
│  │     → 添加伪时间维度 → (B,16,1,H/8,W/8)         │                       │
│  │     → 反标准化 (x * std + mean)                 │                       │
│  │     → AutoencoderKLQwenImage.decode()           │                       │
│  │     → 移除伪时间维度 → (B,3,H,W)                │                       │
│  └───────────────────────┬────────────────────────┘                       │
│                          │                                                │
│                          ▼                                                │
│                    输出图像 [B, 3, H, W]                                   │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 问题5：是否支持文生图和图像编辑

### 结论

| 功能 | 是否支持 | 代码依据 |
|------|---------|---------|
| **文生图（Text-to-Image）** | ✅ 支持 | `inference.py` 和 `sampling.py` 明确实现 |
| **图像+文本提示编辑** | ❌ 不支持 | 推理代码无参考图输入接口，VAE 无 Encoder |

### 代码依据

**1. 文生图 — 明确支持**

```python
# inference.py 第62-134行
@click.command(help="Generate images with Krea 2 (K2).")
@click.argument("prompt")    # 只有文本 prompt 参数
# 没有 --input-image 或类似的参考图参数
def main(prompt, steps, cfg, y1, y2, width, height, num_images, seed, checkpoint, output, mu):
    dit, ae, encoder = _pipeline(checkpoint=checkpoint)
    images = sample(dit, ae, encoder, [prompt] * num_images, ...)
```

`inference.py` 的 CLI 命令只接受文本 prompt，没有图像输入参数。

**2. 图像编辑 — 不支持**

不支持的证据：

- **`QwenAutoencoder` 只有 `decode` 方法**，没有 `encode` 方法 → 无法将参考图编码为 latent
- **`sampling.py` 的 `sample` 函数不接受参考图参数**：

```python
# sampling.py 第57-76行
def sample(model, ae, encoder, prompts, *,
           negative_prompts=None, device="cuda", dtype=torch.bfloat16,
           width=1024, height=1024, steps=28, guidance=4.5,
           seed=0, minres=256, maxres=1280, y1=0.5, y2=1.15, mu=None):
    # 没有 input_image 或 reference_image 参数
```

- **DIT 模型（`SingleStreamDiT.forward`）没有参考图输入**：

```python
# mmdit.py 第379-416行
def forward(self, img, context, t, pos, mask=None):
    # img: 噪声latent，context: 文本编码，t: 时间步
    # 没有参考图 token 输入
```

- **README.md 明确描述为 "image generation model"**，没有提及编辑功能
- **docs/ 目录只有 prompting.md（文生图提示指南）和 safety.md**，没有编辑相关文档

---

## 问题6：文生图流程图

### 文生图时完整数据流：

```
输入数据:
  ├── 文本 Prompt (字符串，如 "a fox walking in the snow")
  ├── 图像尺寸 (width=1024, height=1024)
  ├── 采样参数 (steps=28/52, guidance=3.5/4.5, seed=0)
  └── 检查点类型 (oss_raw 或 oss_turbo)

═══════════════════════════════════════════════════════════════

步骤1: 尺寸对齐
  ┌─────────────────────────────────────────┐
  │ width, height 对齐到 ae.compression *   │
  │ patch = 8 * 2 = 16 的倍数               │
  │ 例: 1024 已是16的倍数，保持不变           │
  └─────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════

步骤2: 文本编码 (子网络1: Qwen3-VL + TextFusionTransformer)
  ┌─────────────────────┐
  │ 文本 Prompt          │
  │ "a fox walking..."  │
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────────────────────────────────┐
  │ Qwen3VLConditioner.forward()                    │
  │                                                 │
  │ 1. 添加 Prompt 模板:                             │
  │    "<|im_start|>system\nDescribe..."             │
  │    + 用户prompt                                  │
  │    + "<|im_end|>\n<|im_start|>assistant\n"       │
  │                                                 │
  │ 2. Tokenize (max_length=512+29)                 │
  │    → input_ids, attention_mask                   │
  │                                                 │
  │ 3. Qwen3-VL-4B Forward:                         │
  │    output_hidden_states=True                     │
  │                                                 │
  │ 4. 选取12层隐藏状态:                              │
  │    第2,5,8,11,14,17,20,23,26,29,32,35层         │
  │    → stack → (B, L+34, 12, 2560)                │
  │                                                 │
  │ 5. 去掉模板前缀 (前34个token):                    │
  │    → hiddens (B, L, 12, 2560)                   │
  │    → mask (B, L)                                │
  └──────────┬──────────────────────────────────────┘
             │
             ▼
  txt (B, L, 12, 2560), txtmask (B, L)
  
  ┌──────────────────────────────────────────────────────┐
  │ 注: 此时txt尚未融合，还需经过TextFusionTransformer   │
  │     (在DIT forward内部调用)                           │
  └──────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════

步骤3: 噪声生成
  ┌─────────────────────────────────────────┐
  │ 对每个 prompt 生成独立的高斯噪声:        │
  │ torch.randn(1, 16, H/8, W/8)           │
  │   = torch.randn(1, 16, 128, 128)       │
  │ generator = seed + i                    │
  │                                         │
  │ 所有 prompt 的噪声 cat 在 batch 维度    │
  │ → noise (B, 16, 128, 128)              │
  └──────────────────┬──────────────────────┘
                     │
  ┌──────────────────▼──────────────────────┐
  │ prepare() 函数:                          │
  │                                         │
  │ 1. Patchify:                            │
  │    (B,16,128,128) → (B, 64*64, 16*2*2) │
  │    = (B, 4096, 64)                      │
  │                                         │
  │ 2. 生成图像位置 ID:                       │
  │    imgids (64, 64, 3)                   │
  │    每个位置 = (0, h_idx, w_idx)          │
  │    → imgpos (B, 4096, 3)                │
  │                                         │
  │ 3. 生成文本位置 ID:                       │
  │    txtpos = zeros(B, L_txt, 3)          │
  │                                         │
  │ 4. 拼接 mask 和 pos:                     │
  │    mask = cat(txtmask, imgmask)          │
  │    pos = cat(txtpos, imgpos)             │
  └──────────────────┬──────────────────────┘
                     │
                     ▼
  img (B, 4096, 64), pos (B, L_txt+4096, 3), mask (B, L_txt+4096)

═══════════════════════════════════════════════════════════════

步骤4: CFG 无条件编码 (如果 guidance > 0)
  ┌─────────────────────────────────────────┐
  │ 对空字符串 [""] * n 编码:                │
  │ untxt, untxtmask = encoder([""] * n)    │
  │ _, unpos, unmask = prepare(noise, ...)   │
  └─────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════

步骤5: 时间步调度
  ┌─────────────────────────────────────────┐
  │ timesteps() 函数:                        │
  │                                         │
  │ 1. 生成均匀网格: ts = linspace(1, 0, S+1)│
  │                                         │
  │ 2. 计算 mu (分辨率感知的时间步偏移):       │
  │    x1 = (minres/(8*2))² = 16²= 256      │
  │    x2 = (maxres/(8*2))² = 80²= 6400     │
  │    mu = 线性插值(seq_len, x1→y1, x2→y2) │
  │                                         │
  │ 3. 时间步变换 (SNR shift):               │
  │    ts = exp(mu) / (exp(mu) + (1/ts-1))   │
  │                                         │
  │ → [1.0, 0.98, ..., 0.02, 0.0]           │
  │   (共 steps+1 = 29 个时间点)              │
  └─────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════

步骤6: Flow Matching 去噪循环 (核心推理)
  
  for tcurr, tprev in zip(ts[:-1], ts[1:]):
    ┌─────────────────────────────────────────────────────┐
    │ SingleStreamDiT.forward()  (条件分支)                │
    │                                                     │
    │ 输入:                                                │
    │   img = noisy_latent (B, 4096, 64)                  │
    │   context = txt (B, L, 12, 2560)                    │
    │   t = tcurr (标量)                                   │
    │   pos = 位置ID (B, L_txt+4096, 3)                    │
    │   mask = 注意力掩码 (B, L_txt+4096)                  │
    │                                                     │
    │ 内部计算:                                            │
    │   1. img = first(img)          → (B, 4096, 6144)    │
    │   2. t_emb = tmlp(temb(t))     → (B, 1, 6144)      │
    │   3. tvec = tproj(t_emb)       → (B, 1, 36864)     │
    │                                                     │
    │   4. TextFusionTransformer:                          │
    │      txt (B,L,12,2560) → 跨层融合 → 投影 → 精炼      │
    │      → context (B, L, 2560)                         │
    │   5. txtmlp: context → (B, L, 6144)                 │
    │                                                     │
    │   6. combined = cat(context, img)                    │
    │      → (B, L_txt+4096, 6144)                        │
    │                                                     │
    │   7. 填充到256的倍数 (编译优化)                        │
    │   8. freqs = posemb(pos)  (3D RoPE)                  │
    │                                                     │
    │   9. 28 × SingleStreamBlock:                         │
    │      每层: tvec → mod → RMSNorm → Attention(GQA)     │
    │            → gate → residual → RMSNorm → SwiGLU     │
    │            → gate → residual                         │
    │                                                     │
    │   10. last = LastLayer:                              │
    │       t_emb → SimpleModulation → (scale, shift)     │
    │       combined → RMSNorm → AdaLN → Linear(→64)      │
    │                                                     │
    │   11. 取 img 部分: output[:, L_txt:L_txt+4096, :]   │
    │                                                     │
    │ 输出: cond (B, 4096, 64)                             │
    └─────────────────────┬───────────────────────────────┘
                          │
    ┌─────────────────────▼───────────────────────────────┐
    │ 如果 cfg > 0:                                       │
    │                                                     │
    │ SingleStreamDiT.forward()  (无条件分支)               │
    │   输入: img, untxt, t, unpos, unmask                 │
    │   → uncond (B, 4096, 64)                            │
    │                                                     │
    │ CFG 合并:                                            │
    │   v = cond + guidance * (cond - uncond)              │
    └─────────────────────┬───────────────────────────────┘
                          │
    欧拉步进: img = img + (tprev - tcurr) * v
    
    (重复 steps 次)

═══════════════════════════════════════════════════════════════

步骤7: Latent → 图像 (子网络2: VAE Decoder)
  ┌─────────────────────────────────────────┐
  │ Unpatchify:                             │
  │ img (B, 4096, 64)                       │
  │ → rearrange("b (h w) (c ph pw)         │
  │     -> b c (h ph) (w pw)")              │
  │ → (B, 16, 128, 128)                    │
  └──────────────────┬──────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────┐
  │ QwenAutoencoder.decode()                │
  │                                         │
  │ 1. rearrange → (B, 16, 1, 128, 128)    │
  │    (添加伪时间维度)                       │
  │                                         │
  │ 2. 反标准化:                             │
  │    x = x * latents_std + latents_mean   │
  │                                         │
  │ 3. AutoencoderKLQwenImage.decode(x)     │
  │    → (B, 3, 1, 1024, 1024)             │
  │                                         │
  │ 4. rearrange → (B, 3, 1024, 1024)      │
  │    (移除伪时间维度)                       │
  └──────────────────┬──────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────┐
  │ 后处理                                   │
  │ 1. clamp(-1, 1)                         │
  │ 2. 归一化到 [0, 1]: * 0.5 + 0.5         │
  │ 3. 转为 uint8: * 255                     │
  │ 4. rearrange("b c h w -> b h w c")      │
  │ 5. numpy → PIL Image                    │
  │ 6. 保存为 PNG 文件                       │
  └─────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════

最终输出: 由 VAE Decoder (QwenAutoencoder) 子网络输出 RGB 图像
         保存为 sample_0.png, sample_1.png, ...
```

---

## 问题7：图像+文本提示图像编辑流程图

### 结论：Krea-2 **不支持**图像+文本提示进行图像编辑。

### 不支持的代码依据

1. **VAE 无 Encoder**：`QwenAutoencoder` 只实现了 `decode` 方法，没有 `encode` 方法，无法将参考图像编码为 latent 表示。

2. **推理入口无图像输入**：`inference.py` 的 CLI 只接受文本 prompt，没有 `--input-image` 参数。

3. **采样函数无参考图接口**：`sampling.py` 的 `sample` 函数参数中没有参考图输入。

4. **DIT 前向传播无参考图处理**：`SingleStreamDiT.forward()` 只接受 `img`（噪声latent）、`context`（文本）、`t`（时间步），没有参考图 token 输入。

5. **无因果注意力机制**：与 FLUX.2 不同，Krea-2 没有实现 `causal_attn_fn`、`forward_kv_extract`、`forward_kv_cached` 等用于参考图编辑的机制。

6. **无参考图编码函数**：与 FLUX.2 的 `encode_image_refs` 不同，Krea-2 没有类似的参考图编码/处理函数。

7. **README 和文档**：README 描述为 "an image generation model"（图像生成模型），文档中只有文生图的 prompting 指南，没有编辑相关内容。

因此，**无法绘制图像编辑流程图**，因为该功能在代码中不存在。

---

## 问题8：相比FLUX.2模型的创新点

### 基于代码实现分析，Krea-2 相比 FLUX.2 具有以下创新点和改进：

#### 1. 纯单流架构（最显著的架构差异）

**FLUX.2**：使用 **双流（DoubleStreamBlock）× 8 + 单流（SingleStreamBlock）× 48** 的混合架构，先用双流块分别处理图像和文本，再用单流块统一处理。

**Krea-2**：使用**纯单流（SingleStreamBlock）× 28** 的架构，文本和图像在进入 Transformer 之前就拼接在一起，所有 block 使用相同的参数处理拼接后的序列。

```python
# Krea-2: 纯单流
combined = torch.cat((context, img), dim=1)
for block in self.blocks:  # 28 个 SingleStreamBlock
    combined = block(combined, tvec, freqs, mask)

# FLUX.2: 双流 + 单流混合
for block in self.double_blocks:  # 8 个 DoubleStreamBlock
    img, txt, _ = block(img, txt, ...)
img = torch.cat((txt, img), dim=1)
for block in self.single_blocks:  # 48 个 SingleStreamBlock
    img, _ = block(img, ...)
```

**优势**：纯单流架构参数量更少、实现更简洁，模型参数可以在文本和图像之间充分共享。

#### 2. 文本编码器创新：Qwen3-VL 多模态视觉语言模型

**FLUX.2**：
- Dev 版使用 Mistral-Small-3.2-24B（24B 参数 VLM），提取 3 层（第10/20/30层）
- Klein 版使用 Qwen3-4B/8B（纯文本 LLM），提取 3 层（第9/18/27层）

**Krea-2**：使用 **Qwen3-VL-4B-Instruct**（4B 参数视觉语言模型），提取 **12 层**（第2,5,8,11,14,17,20,23,26,29,32,35层）

```python
# Krea-2: 12层，每隔3层取一层
select_layers = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)

# FLUX.2 Mistral: 仅3层
OUTPUT_LAYERS_MISTRAL = [10, 20, 30]
```

**差异**：Krea-2 提取了多 4 倍的层级信息（12 层 vs 3 层），虽然文本编码器参数量更小（4B vs 24B），但通过更密集的层采样获取了更丰富的多尺度语义特征。

#### 3. TextFusionTransformer —— 独特的多层文本融合网络

这是 Krea-2 最具创新性的设计之一。FLUX.2 简单地将多层隐藏状态沿特征维度拼接：

```python
# FLUX.2: 简单拼接 3 层
out = torch.stack([output.hidden_states[k] for k in OUTPUT_LAYERS], dim=1)
return rearrange(out, "b c l d -> b l (c d)")  # 拼接维度: 3*5120=15360
```

Krea-2 使用一个**专门的 4 层 Transformer 网络**来融合 12 层文本特征：

```python
# Krea-2: TextFusionTransformer
# 1. layerwise_blocks × 2: 跨层注意力融合 (每个token位置独立处理12层)
# 2. projector: 线性投影 12 → 1
# 3. refiner_blocks × 2: 序列级别全局精炼
```

**优势**：
- 学习性地融合多层特征，而非简单拼接
- 跨层注意力可以建模层间关系
- 序列级精炼可以全局优化文本表示
- 最终输出维度仅 2560（vs FLUX.2 的 15360），更紧凑高效

#### 4. Grouped-Query Attention (GQA)

**FLUX.2**：使用标准 Multi-Head Attention（`num_heads=48`，Q/K/V 头数相同）

**Krea-2**：使用 GQA（`heads=48`, `kvheads=12`，4:1 比例）

```python
# Krea-2 Attention
self.wq = nn.Linear(dim, headdim * heads, bias=bias)      # Q: 48 heads
self.wk = nn.Linear(dim, headdim * kvheads, bias=bias)     # K: 12 heads
self.wv = nn.Linear(dim, headdim * kvheads, bias=bias)     # V: 12 heads
self.gate = nn.Linear(dim, dim, bias=bias)                 # 门控输出
```

**优势**：GQA 将 KV 参数减少为 Q 的 1/4，显著降低计算量和内存占用，同时保持接近 MHA 的性能。

#### 5. Gated Attention 输出

**FLUX.2**：注意力输出直接通过投影层 `proj`：

```python
# FLUX.2 Attention
out = attn_fn(q, k, v)
return self.proj(out)
```

**Krea-2**：注意力输出乘以一个**可学习的门控信号**：

```python
# Krea-2 Attention
q, k, v, gate = self.wq(qkv), self.wk(qkv), self.wv(qkv), self.gate(qkv)
out = self.wo(attention(q, k, v, ...) * F.sigmoid(gate))
```

`gate` 是一个与 QKV 并行计算的线性变换，通过 sigmoid 后与注意力输出逐元素相乘。这类似于 Gated Linear Units (GLU) 的思想应用到注意力中。

**优势**：门控机制允许模型动态控制每个位置注意力输出的信息流量。

#### 6. SwiGLU 替代 SiLU Gated Activation

**FLUX.2**：使用 `SiLUActivation`（SiLU gating），MLP 结构为 `Linear → chunk → SiLU(x1)*x2 → Linear`：

```python
# FLUX.2 SiLUActivation
class SiLUActivation(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return self.gate_fn(x1) * x2  # SiLU gating
```

**Krea-2**：使用 `SwiGLU`，结构为 `gate(x)*up(x) → down`，三个独立的线性层：

```python
# Krea-2 SwiGLU
class SwiGLU(nn.Module):
    def __init__(self, features, multiplier, bias, multiple=128):
        mlpdim = int(2 * features / 3) * multiplier
        mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)
        self.gate = nn.Linear(features, mlpdim, bias=bias)
        self.up = nn.Linear(features, mlpdim, bias=bias)
        self.down = nn.Linear(mlpdim, features, bias=bias)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))
```

**差异**：
- FLUX.2 的 SiLU gating 将一个宽线性层的输出 chunk 成两半做 gating
- Krea-2 的 SwiGLU 使用三个独立的线性层（gate, up, down），与 LLaMA/Qwen 等主流 LLM 的 MLP 结构一致
- Krea-2 将 MLP 中间维度对齐到 128 的倍数（`multiple=128`），优化硬件计算效率

#### 7. 每层独立 Modulation vs 全局共享 Modulation

**FLUX.2**：使用**全局共享的 Modulation**，所有双流块共享一个 modulation，所有单流块共享一个 modulation：

```python
# FLUX.2: 全局共享，在 __init__ 中定义一次
self.double_stream_modulation_img = Modulation(hidden_size, double=True)
self.single_stream_modulation = Modulation(hidden_size, double=False)
# 在 forward 中计算一次，传入所有 block
```

**Krea-2**：每个 `SingleStreamBlock` 有**独立的 `DoubleSharedModulation`**：

```python
# Krea-2: 每层独立 Modulation
class SingleStreamBlock(nn.Module):
    def __init__(self, ...):
        self.mod = DoubleSharedModulation(features)  # 每层独立
```

但注意，Krea-2 的 `DoubleSharedModulation` 结构特殊——它是一个**可学习的偏置参数**，不是从时间步条件中动态计算的：

```python
# Krea-2 DoubleSharedModulation
class DoubleSharedModulation(torch.nn.Module):
    def __init__(self, dim):
        self.lin = torch.nn.Parameter(torch.zeros(6 * dim))  # 可学习参数

    def forward(self, vec):
        out = vec + self.lin  # vec 来自 tproj(t_emb)，lin 是每层的偏置
        return out.chunk(6, dim=-1)
```

**差异**：
- FLUX.2 的 Modulation 使用 `SiLU → Linear` 从时间步条件 vec 动态生成调制参数
- Krea-2 的 DoubleSharedModulation 是 `vec + learnable_bias`，每层有独立的可学习偏置
- 这意味着 Krea-2 的调制参数 = 全局时间条件 + 每层独立微调，比 FLUX.2 更灵活

#### 8. RMSNorm 替代 LayerNorm

**FLUX.2**：使用 `LayerNorm`（`elementwise_affine=False`）：

```python
# FLUX.2
self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
```

**Krea-2**：使用 `RMSNorm`（带可学习缩放参数）：

```python
# Krea-2 RMSNorm
class RMSNorm(torch.nn.Module):
    def __init__(self, features, eps=1e-05):
        self.scale = torch.nn.Parameter(torch.zeros(features, dtype=torch.float32))

    def forward(self, x):
        t = x.float()
        t = F.rms_norm(t, (self.features,), eps=self.eps, weight=(self.scale.float() + 1.0))
        return t.to(dtype)
```

**差异**：
- RMSNorm 不需要计算均值，只计算均方根，计算效率更高
- Krea-2 的 RMSNorm 使用 `scale + 1.0` 的初始化策略（初始时 scale=0，等效于缩放因子=1）
- 与 FLUX.2 的 LayerNorm (无可学习参数) 不同，Krea-2 的 RMSNorm 有可学习的 scale 参数

#### 9. 3D RoPE 位置编码 vs 4D RoPE

**FLUX.2**：使用 **4D RoPE**（`axes_dim=[32,32,32,32]`），对应 (t, h, w, l) 四个维度：

```python
# FLUX.2: 4D RoPE
axes_dim: list[int] = [32, 32, 32, 32]  # t, h, w, l
```

**Krea-2**：使用 **3D RoPE**（对应 t, h, w 三个维度）：

```python
# Krea-2: 3D 位置编码
imgids = torch.zeros((h_, w_, 3), device=img.device)
imgids[..., 1] = torch.arange(h_)[:, None]  # h
imgids[..., 2] = torch.arange(w_)[None, :]  # w

# RoPE 维度分配
headdim = 6144 // 48 = 128
axes = [128 - 12*(128//16), 6*(128//16), 6*(128//16)]
     = [128 - 96, 48, 48]
     = [32, 48, 48]  # 不均匀分配：t=32, h=48, w=48
```

**差异**：
- FLUX.2 使用 4 维 RoPE 且均匀分配（各 32 维）
- Krea-2 使用 3 维 RoPE，空间维度分配更多（h=48, w=48 > t=32）
- Krea-2 的文本 token 位置全为零向量 `txtpos = zeros(B, L, 3)`，不使用位置编码
- FLUX.2 的第 4 维 `l` 用于文本 token 的序列位置

#### 10. Qwen-Image VAE vs 自研 VAE

**FLUX.2**：使用**自研的卷积 VAE**，结构包含 Encoder + Decoder + BatchNorm 归一化 + Patch 重排，有效下采样 16x，有效通道数 128。

**Krea-2**：使用**第三方 Qwen-Image VAE**（`AutoencoderKLQwenImage`），下采样 8x，通道数 16。

```python
# Krea-2: 使用 Qwen-Image VAE
from diffusers import AutoencoderKLQwenImage
self.ae = AutoencoderKLQwenImage.from_pretrained("Qwen/Qwen-Image", subfolder="vae")
self.compression = 8
self.channels = 16
```

**差异**：
- Krea-2 的 latent 空间更小（16 通道 vs 128 通道），空间分辨率更高（H/8 vs H/16）
- Krea-2 使用均值/标准差归一化（预计算的统计量），FLUX.2 使用 BatchNorm
- Krea-2 的 VAE 原生支持视频（3D），虽然此处仅用于图像
- Krea-2 只实现了 Decoder（不需要 Encoder），FLUX.2 完整实现了 Encoder + Decoder

#### 11. 更轻量的模型设计

**FLUX.2 [dev]**：
- 文本编码器：24B 参数（Mistral-Small-3.2-24B）
- DIT：~32B 参数（6144 hidden, 8 double + 48 single blocks）
- 输入维度：15360（3层×5120 拼接）

**Krea-2**：
- 文本编码器：4B 参数（Qwen3-VL-4B）
- DIT：单流 28 层（6144 hidden），参数量显著少于 FLUX.2
- 输入维度：2560（经 TextFusionTransformer 融合后）

```python
# Krea-2 参数对比
features=6144, heads=48, kvheads=12, layers=28
# FLUX.2 参数对比
hidden_size=6144, num_heads=48, depth=8, depth_single_blocks=48
```

总层数：Krea-2 = 28 层 vs FLUX.2 = 56 层（8+48）。加上 GQA 减少 KV 参数，Krea-2 的 DIT 参数量约为 FLUX.2 的一半。

#### 12. 去除 Guidance Embedding

**FLUX.2**：支持 **Guidance Embedding**（`use_guidance_embed=True`），将 guidance 值也编码为条件输入：

```python
# FLUX.2
if self.use_guidance_embed:
    guidance_emb = timestep_embedding(guidance, 256)
    vec = vec + self.guidance_in(guidance_emb)
```

**Krea-2**：**不使用 Guidance Embedding**。CFG 通过经典的两次前向传播实现：

```python
# Krea-2: 经典 CFG
cond = model(img=img, context=txt, t=t, pos=pos, mask=mask)
uncond = model(img=img, context=untxt, t=t, pos=unpos, mask=unmask)
v = cond + guidance * (cond - uncond)
```

**差异**：FLUX.2 将 guidance scale 作为条件嵌入输入模型（使模型可以被 guidance 蒸馏），Krea-2 使用传统的 CFG 方法（需要两次前向传播但更简单）。

#### 13. LastLayer 的实现差异

**FLUX.2**：使用 `LayerNorm + AdaLN（SiLU→Linear→chunk）+ Linear`：

```python
# FLUX.2 LastLayer
self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False)
self.linear = nn.Linear(hidden_size, out_channels, bias=False)
self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=False))
```

**Krea-2**：使用 `RMSNorm + SimpleModulation（可学习参数）+ Linear`：

```python
# Krea-2 LastLayer
self.norm = RMSNorm(features)
self.linear = nn.Linear(features, patch*patch*channels, bias=True)  # 带 bias
self.modulation = SimpleModulation(features)

# SimpleModulation: 可学习偏置
class SimpleModulation(torch.nn.Module):
    def __init__(self, dim):
        self.lin = torch.nn.Parameter(torch.zeros(2, dim))  # 2个可学习向量
    def forward(self, vec):
        out = vec + rearrange(self.lin, "two d -> 1 two d")
        scale, shift = out.chunk(2, dim=1)
        return scale, shift
```

**差异**：
- Krea-2 的 SimpleModulation 使用可学习偏置（而非 SiLU 激活的线性变换）
- Krea-2 的 `linear` 输出层带 `bias=True`（FLUX.2 为 `bias=False`）
- Krea-2 使用 RMSNorm（FLUX.2 使用 LayerNorm）

#### 14. RoPE theta 参数差异

```python
# FLUX.2
theta: int = 2000

# Krea-2
theta: float = 1e3  # = 1000
```

Krea-2 使用更小的 theta 值（1000 vs 2000），这会导致高频分量更强，可能有助于捕捉精细的空间关系。

### 创新点总结表

| # | 创新/改进点 | FLUX.2 | Krea-2 |
|---|------------|--------|--------|
| 1 | 架构类型 | 双流(8)+单流(48) | 纯单流(28) |
| 2 | 文本编码器 | Mistral-24B/Qwen3-LLM | Qwen3-VL-4B (VLM) |
| 3 | 文本层采样 | 3 层拼接 | 12 层 + TextFusionTransformer 融合 |
| 4 | 注意力类型 | 标准 MHA | GQA (4:1 比例) + 门控输出 |
| 5 | MLP 激活 | SiLU Gating | SwiGLU (三线性层) |
| 6 | Modulation | 全局共享(SiLU+Linear) | 每层独立可学习偏置 |
| 7 | 归一化 | LayerNorm (无参数) | RMSNorm (带可学习 scale) |
| 8 | 位置编码 | 4D RoPE (均匀 [32,32,32,32]) | 3D RoPE (不均匀 [32,48,48]) |
| 9 | VAE | 自研 (16x, 128ch, BN) | Qwen-Image (8x, 16ch, mean/std) |
| 10 | 模型规模 | ~32B DIT + 24B 编码器 | ~16B DIT + 4B 编码器 |
| 11 | Guidance | Guidance Embedding | 经典 CFG (两次前向) |
| 12 | 图像编辑 | ✅ 支持 (因果注意力+KV cache) | ❌ 不支持 (纯文生图) |
| 13 | RoPE theta | 2000 | 1000 |
| 14 | LastLayer | AdaLN (SiLU+Linear) | SimpleModulation (可学习偏置) |
