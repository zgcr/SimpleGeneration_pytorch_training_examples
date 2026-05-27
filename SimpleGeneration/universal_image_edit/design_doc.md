# 通用图像编辑模型设计文档

## QWEN3.5-4B VLM + MMDIT + Flow Matching

---

## 一、整体架构设计

### 推断依据

1. **Flux2**: Double-Stream(8层) → Single-Stream(48层) 架构，先独立处理文本/图像特征，再融合处理。这是当前最主流的MMDIT结构，在12B规模上验证了卓越效果。
2. **Z-Image**: 纯Single-Stream(30层) + Refiner(2+2层)，结构更简单但缺少独立处理阶段。
3. **FireRed**: 基于QwenImage Transformer，使用Qwen2.5-VL作为text encoder，source latent拼接方案。
4. **JoyAI**: MMDoubleStreamBlock × N (WanX风格)，仅双流无单流，3D支持。

### 结论

采用 **Flux2 的 Double-Stream + Single-Stream** 架构：

- **Double-Stream Blocks**: img/txt各自独立调制/QKV/MLP，joint attention融合
- **Single-Stream Blocks**: txt+img拼接后统一处理，QKV+MLP合并linear高效实现
- **VLM (冻结)**: Qwen3.5-4B 取最后隐藏层作为文本条件
- **VAE (冻结)**: flux2_autoencoder.py 不做任何改动
- **训练算法**: Flow Matching (velocity prediction)

### 架构流程图

```
输入文本 ──→ [Qwen3.5-4B VLM (冻结)] ──→ text embedding (B, N_txt, 2560)
                                              │
输入图像 ──→ [flux2 VAE (冻结)] ──→ latent ──→ pack ──→ img tokens (B, N_img, 128)
                                                          │
时间步 t ──→ [Timestep Embedding] ──→ [MLP] ──→ vec (B, hidden_size)
                                                  │
                    ┌─────────────────────────────┘
                    │
                    ▼
            [Double-Stream Blocks × D]   ← img/txt 独立调制 + joint attention
                    │
                    ▼
            txt + img 拼接
                    │
                    ▼
            [Single-Stream Blocks × S]   ← 统一处理
                    │
                    ▼
            去掉txt tokens
                    │
                    ▼
            [Final Layer (AdaLN + Linear)]
                    │
                    ▼
            velocity prediction (B, N_img, 128)
```

---

## 二、核心技术选型

### 2.1 Attention

| 对比   | Flux2                          | Z-Image                        | JoyAI                  | 本设计                             |
| ------ | ------------------------------ | ------------------------------ | ---------------------- | ---------------------------------- |
| 算子   | F.scaled_dot_product_attention | dispatch_attention (多backend) | flash_attn_varlen_func | **F.scaled_dot_product_attention** |
| QKV    | 合并Linear, bias=False         | 分离Linear, bias=False         | 合并Linear, bias=True  | **合并Linear, bias=False**         |
| Layout | B,H,L,D                        | B,L,H,D                        | B,L,H,D                | **B,H,L,D**                        |

**推断依据**: 用户要求必须使用`F.scaled_dot_product_attention`以在PyTorch 2.5.1+ BF16下自动启用Flash Attention。Flux2的(B,H,L,D) layout与该算子原生兼容。

**结论**: 采用`F.scaled_dot_product_attention`，QKV合并Linear(bias=False)，(B,H,L,D) layout。

### 2.2 Normalization

| 对比     | Flux2                | Z-Image | JoyAI                | 本设计                   |
| -------- | -------------------- | ------- | -------------------- | ------------------------ |
| 主体Norm | LayerNorm(no affine) | RMSNorm | LayerNorm(no affine) | **LayerNorm(no affine)** |
| QK-Norm  | RMSNorm              | RMSNorm | RMSNorm              | **RMSNorm**              |

**推断依据**: 四库共识使用RMSNorm做QK-Norm。主体使用LayerNorm(elementwise_affine=False)配合AdaLN调制。

**结论**: LayerNorm(no affine) + RMSNorm for QK。

### 2.3 Position Encoding (RoPE)

| 对比  | Flux2        | Z-Image        | JoyAI        | 本设计       |
| ----- | ------------ | -------------- | ------------ | ------------ |
| 维度  | 4D (t,h,w,l) | 3D (t,h,w)     | 3D (t,h,w)   | **2D (h,w)** |
| 实现  | 矩阵旋转     | complex-valued | real cos/sin | **矩阵旋转** |
| theta | 2000         | 256            | 256          | **2000**     |

**推断依据**: 纯2D图像任务不需要时间轴。Flux2的矩阵旋转形式数值最稳定，与SDPA兼容。theta=2000与Flux2一致，适配图像latent序列长度(256~4096)，比NLP标准值10000有更强的短序列位置区分能力。

**结论**: 2D RoPE (h,w)，矩阵旋转形式，theta=2000。

### 2.4 FFN

| 对比  | Flux2      | Z-Image           | JoyAI            | 本设计         |
| ----- | ---------- | ----------------- | ---------------- | -------------- |
| 类型  | SiLU-Gated | SwiGLU (w1,w2,w3) | GELU-approximate | **SiLU-Gated** |
| ratio | 3.0        | 8/3 ≈ 2.67        | 4.0              | **3.0/4.0**    |

**推断依据**: SiLU-Gated与AdaLN的SiLU一致，减少激活函数种类。mlp_ratio=3.0与Flux2全系列一致，在同等参数预算下允许更深的网络（Scaling Laws表明深度通常比宽度更重要）。1B模型因head_dim=64较小，保持mlp_ratio=4.0以保证MLP表达力。

**结论**: SiLU-Gated FFN, mlp_ratio=3.0（2B/4B/6B/8B），mlp_ratio=4.0（1B）。

### 2.5 Modulation (AdaLN)

| 对比 | Flux2                 | Z-Image            | JoyAI        | 本设计                 |
| ---- | --------------------- | ------------------ | ------------ | ---------------------- |
| 参数 | 6(shift,scale,gate)×2 | 4(scale,gate)×2    | 6(Wan table) | **6 (Flux2)**          |
| Gate | 直接乘                | tanh               | 直接乘       | **直接乘**             |
| bias | False                 | True               | 无(Parameter) | **False**              |
| 来源 | SiLU(vec) → Linear    | SiLU(vec) → Linear | 可学习表+vec | **SiLU(vec) → Linear** |

**推断依据**: Flux2的6参数方案最完整（shift+scale+gate），gate直接乘比tanh更简单有效。bias=False与Flux2一致，所有Linear层统一无bias。

**结论**: AdaLN-single, 6参数(双流)/3参数(单流), SiLU(vec)→Linear(bias=False)。

### 2.6 LayerScale

**推断依据**: 传统LayerScale是可学习标量缩放残差。在AdaLN架构中，gate已经承担了LayerScale的功能。

**结论**: 通过AdaLN的gate参数隐式实现，不额外添加。

### 2.7 QK-Norm

**推断依据**: 四库均使用QK-Norm防止注意力分数爆炸。RMSNorm比LayerNorm更轻量。

**结论**: RMSNorm on Q/K per head，带可学习scale参数。

### 2.8 参考图处理方案

| 对比 | Flux2                    | FireRed            | JoyAI        | 本设计            |
| ---- | ------------------------ | ------------------ | ------------ | ----------------- |
| 方式 | 拼接到img seq + KV-Cache | 拼接到noisy latent | temporal维度 | **拼接到img seq** |
| Mask | 无显式mask               | 无显式mask         | 无           | **无显式mask**    |

**推断依据**: Flux2/FireRed的拼接方案最通用，支持0~N张参考图，不需要修改模型结构。

**结论**: 参考图VAE编码→拼接到目标图token后方→共享attention→仅对目标图token计算loss/更新。

---

## 三、五档模型配置

### 推断依据

- 参数量估算（bias=False, SiLU-Gated FFN）：
  - mlp_ratio=3.0: Double Block ≈ 38d², Single Block ≈ 16d²
  - mlp_ratio=4.0: Double Block ≈ 44d², Single Block ≈ 19d²
- Flux2 Klein系列验证了不同规模的可行性（全系列 mlp_ratio=3.0, theta=2000）
- mlp_ratio=3.0 在同等参数预算下允许更深网络，Scaling Laws表明深度通常比宽度更重要
- theta=2000 适配图像latent序列长度(256~4096)，比NLP标准值10000有更强位置区分能力

### 结论

| 配置项         | 1B      | 2B      | 4B      | 6B      | 8B      |
| -------------- | ------- | ------- | ------- | ------- | ------- |
| hidden_size    | 1536    | 2048    | 2560    | 3072    | 3456    |
| num_heads      | 24      | 16      | 20      | 24      | 27      |
| head_dim       | 64      | 128     | 128     | 128     | 128     |
| depth (double) | 4       | 5       | 6       | 7       | 8       |
| depth_single   | 16      | 18      | 24      | 25      | 26      |
| mlp_ratio      | 4.0     | 3.0     | 3.0     | 3.0     | 3.0     |
| axes_dim       | [32,32] | [64,64] | [64,64] | [64,64] | [64,64] |
| theta          | 2000    | 2000    | 2000    | 2000    | 2000    |
| in_channels    | 128     | 128     | 128     | 128     | 128     |
| context_in_dim | 2560    | 2560    | 2560    | 2560    | 2560    |
| 总层数         | 20      | 23      | 30      | 32      | 34      |
| 精算参数量     | 1.14B   | 2.02B   | 4.04B   | 6.32B   | 8.65B   |

---

## 四、详细模块设计

### 4.1 RMSNorm

- 实现：`x * rsqrt(mean(x²) + eps) * learnable_weight`
- 用于：QK-Norm per head

### 4.2 QKNorm

- 对Q和K分别做RMSNorm
- 保证注意力分数在合理范围

### 4.3 SiLU-Gated Activation

- `SiLU(x1) * x2`，其中`x1, x2 = split(x)`
- 用于FFN和单流block的MLP

### 4.4 Modulation (AdaLN-single)

- 输入：timestep conditioning vector
- 双流：6参数 = (shift1, scale1, gate1, shift2, scale2, gate2)
- 单流：3参数 = (shift, scale, gate)
- 实现：`SiLU(vec) → Linear → chunk`

### 4.5 RoPE (2D Rotary Positional Embedding)

- 矩阵旋转形式：`[cos, -sin; sin, cos]`
- 两轴分别计算后拼接
- 在apply_rope时与Q/K相乘

### 4.6 Double-Stream Block

```
img_modulated = (1 + scale) * LayerNorm(img) + shift
txt_modulated = (1 + scale) * LayerNorm(txt) + shift
→ 各自QKV → joint attention (concat Q,K,V → SDPA → split)
→ 各自 gate * proj(attn)
→ 各自 gate * MLP(modulated_norm(x))
```

### 4.7 Single-Stream Block

```
x_mod = (1 + scale) * LayerNorm(x) + shift
→ linear1 → split(QKV, MLP_in)
→ QKV → SDPA
→ concat(attn, SiLU_gate(MLP_in)) → linear2
→ x + gate * output
```

### 4.8 Final Layer

```
mod = SiLU(vec) → Linear → (shift, scale)
x = (1 + scale) * LayerNorm(x) + shift
x = Linear(x)
```

---

## 五、损失函数

### 推断依据

- Flow Matching: `z_t = (1-t)*x_0 + t*noise`, target `v = noise - x_0`
- FireRed: weighted MSE + threshold clipping
- 标准FM: 简单MSE即可

### 结论

```python
# 加噪
noisy = (1 - t) * clean + t * noise

# 目标
target = noise - clean

# 损失
loss = MSE(v_pred, target)  # 可选sigma weighting
```

支持三种weighting方案：

1. **none**: 均匀权重
2. **sigma_sqrt**: w = 1/sigma，降低高噪声权重
3. **min_snr_5**: Min-SNR-gamma(γ=5)，防止梯度爆炸

时间步采样：

1. **uniform**: t ~ U(0, 1)
2. **logit_normal**: t = sigmoid(N(μ, σ²))，推荐用于正式训练

---

## 六、文生图训练Pipeline

### 流程

1. **VLM编码**: `text → Qwen3.5-4B(frozen) → hidden_states[-1] → ctx`
2. **VAE编码**: `image → flux2_AE.encode(frozen) → pack → latent tokens`
3. **构建IDs**: `create_image_ids(h, w)`, `create_txt_ids(seq_len)`
4. **采样+加噪**: `t ~ logit_normal`, `z_t = (1-t)*x_0 + t*noise`
5. **MMDIT前向**: `v_pred = model(z_t, ids, t, ctx, ctx_ids)`
6. **计算loss**: `loss = MSE(v_pred, noise - x_0)`
7. **反向传播**: AdamW + gradient clipping

### 关键代码见 `pipeline/train_t2i.py`

---

## 七、文生图推理Pipeline

### 流程

1. **VLM编码**: 文本prompt → embedding
2. **初始化噪声**: `z_1 ~ N(0, I)`
3. **设置scheduler**: Euler ODE solver + dynamic shifting
4. **去噪循环**:
   - 可选CFG: `v = v_uncond + scale * (v_cond - v_uncond)`
   - Euler step: `z_{t-dt} = z_t + dt * v_pred`
5. **VAE解码**: `latent → flux2_AE.decode → image`

### 关键代码见 `pipeline/inference_t2i.py`

---

## 八、编辑任务训练Pipeline

### 与文生图的区别

1. **参考图编码**: source images通过VAE编码为latent tokens
2. **位置ID**: 参考图使用偏移坐标避免与目标图重叠
3. **拼接**: `x = [noisy_target_tokens, clean_ref_tokens]`
4. **Loss**: 仅对目标图tokens计算 `v_pred[:, :N_target]`

### 多任务统一

- **Text-to-Image**: 无参考图，等价于纯文生图
- **Inpainting**: 参考图 = 原图，文本描述修改区域（无需mask）
- **Style Transfer**: 参考图 = 风格图，文本描述风格指令
- **Multi-ref Edit**: 多张参考图拼接，模型通过attention自动融合

### 关键代码见 `pipeline/train_edit.py`

---

## 九、编辑任务推理Pipeline

### 流程

1. **VLM编码**: 文本+可选参考图信息
2. **VAE编码参考图**: `ref_images → VAE.encode → ref_latent_tokens`
3. **初始化目标噪声**: `target_z1 ~ N(0, I)`
4. **去噪循环**:
   - 每步拼接: `x = [target_noisy, ref_clean]`
   - MMDIT预测: `v = model(x, ...)`
   - 截取目标: `v_target = v[:, :N_target]`
   - 可选CFG
   - Euler step仅更新target
5. **VAE解码**: target latent → image

### 关键代码见 `pipeline/inference_edit.py`

---

## 十、总结

### 设计亮点

1. **结构简洁高效**: Double-Stream → Single-Stream的两阶段设计，兼顾特征独立处理和深度融合
2. **统一多任务**: 通过token拼接方案统一处理文生图和各类编辑任务，无需修改模型结构
3. **VLM即插即用**: Qwen3.5-4B冻结，仅需一个Linear投影层，可随时升级VLM
4. **VAE不变**: flux2_autoencoder完全冻结，五档模型共享同一VAE
5. **Flash Attention兼容**: 全部使用F.scaled_dot_product_attention，BF16下自动Flash Attention
6. **五档配置**: 1B/2B/4B/6B/8B覆盖实验到生产的各种需求

### 技术选型汇总

| 维度              | 选型                                  |
| ----------------- | ------------------------------------- |
| 整体结构          | Double-Stream + Single-Stream (Flux2) |
| Attention         | F.scaled_dot_product_attention        |
| Normalization     | LayerNorm(no affine) + RMSNorm        |
| Position Encoding | 2D RoPE (矩阵旋转)                    |
| QK-Norm           | RMSNorm per head                      |
| FFN               | SiLU-Gated, ratio=3.0/4.0            |
| Modulation        | AdaLN-single (6/3参数, bias=False)    |
| LayerScale        | 通过gate隐式实现                      |
| 训练算法          | Flow Matching (velocity)              |
| 参考图处理        | Token拼接 + joint attention           |
| Scheduler         | Euler ODE + dynamic shifting          |
| VLM               | Qwen3.5-4B (冻结)                     |
| VAE               | flux2_autoencoder (冻结)              |

### 文件结构

```
SimpleGeneration/universal_edit/
├── __init__.py
├── design_doc.md              ← 本文档
├── models/
│   ├── __init__.py
│   ├── config.py              ← 五档配置 (1B/2B/4B/6B/8B)
│   ├── mmdit.py               ← 完整MMDIT模型定义
│   └── losses.py              ← Flow Matching损失函数
└── pipeline/
    ├── __init__.py
    ├── scheduler.py           ← Flow Match Euler调度器
    ├── train_t2i.py           ← 文生图训练Pipeline
    ├── train_edit.py          ← 编辑任务训练Pipeline
    ├── inference_t2i.py       ← 文生图推理Pipeline
    └── inference_edit.py      ← 编辑任务推理Pipeline
```
