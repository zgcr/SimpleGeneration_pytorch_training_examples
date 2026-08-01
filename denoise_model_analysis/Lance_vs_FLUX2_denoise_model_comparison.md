# Lance vs FLUX2 去噪模型架构对比分析

> **分析对象**：
> - **Lance**：ByteDance Research 的统一多模态模型（3B 活跃参数），去噪骨干网络为基于 Qwen2 LLM 的 Causal Decoder
> - **FLUX2**：Black Forest Labs 的图像生成/编辑模型（Dev 32B / Klein 4B/9B），去噪骨干网络为双流+单流 MMDIT
>
> **代码实现根目录**：
> - Lance: `/opt/nas/p/zhugechaoran/download/code/Lance/`
> - FLUX2: `/opt/nas/p/zhugechaoran/download/code/flux2/`
>
> **分析范围**：仅关注去噪模型（Denoiser）部分的网络结构差异，不涉及 VAE、Text Encoder 等外部组件

---

## 一、整体架构层面的变化创新点

### 创新点 1：LLM Causal Decoder 替代独立的 MMDIT 作为去噪骨干网络

**FLUX2 的做法**：
- 使用独立设计的 MMDIT 架构（`Flux2` 类，`model.py`），包含 `DoubleStreamBlock` × 8 层 + `SingleStreamBlock` × 48 层
- 图像 latent tokens 和文本 tokens 通过各自的投影层进入 MMDIT，文本由外部 Text Encoder（Mistral-Small-3.2-24B 或 Qwen3）编码后提供
- MMDIT 是一个专门为去噪任务设计的网络，不具备语言生成能力

**Lance 的做法**：
- 直接使用 Qwen2 大语言模型的 Causal Decoder（`Qwen2ForCausalLM` → `Qwen2Model`，`qwen2_navit.py` 第 976 行和第 820 行）作为去噪骨干网络
- LLM 的 `embed_tokens` 直接处理文本 token，不需要外部 Text Encoder
- 同一个 Transformer 既做 Flow Matching 速度场预测（生成任务），又做自回归 next-token prediction（理解任务）

**代码验证**：
- Lance `lance.py` 第 284-290 行：`last_hidden_state = self.language_model(packed_sequence=packed_sequence, ...)`，其中 `self.language_model` 是 `Qwen2ForCausalLM`
- Lance `lance.py` 第 294 行：`packed_mse_preds = self.llm2vae(last_hidden_state[mse_loss_indexes])`，LLM 输出直接用于速度场预测
- FLUX2 `model.py` 第 52-168 行：`Flux2` 类完全独立，有自己的 `img_in`、`txt_in`、`double_blocks`、`single_blocks`、`final_layer`

**优点分析**：
- **优点方向：模型统一性与多任务能力**。用 LLM 作为去噪骨干网络可以实现理解与生成的统一，一个模型同时支持文生图/视频、图像/视频编辑、图像/视频理解等 7 种任务，而 FLUX2 的 MMDIT 只能做生成和编辑。这属于功能维度的优化，而非直接的生成效果优化。
- **优点方向：参数效率**。LLM 的预训练知识可以直接复用，无需从头训练一个独立的 MMDIT 去噪网络，降低了训练成本。
- **潜在劣势**：Causal Decoder 的注意力是单向的（通过 sparse mask 实现双向部分），而 MMDIT 的注意力是天然双向的，后者对去噪任务可能更天然适配。

---

### 创新点 2：MoT（Mixture of Transformers）解耦理解与生成参数

**FLUX2 的做法**：
- 双流块中图像和文本各有独立的 QKV/MLP（`DoubleStreamBlock`，`model.py` 第 524-681 行），但这是按**模态**（图像 vs 文本）分的，不是按**任务**分的
- 单流块中所有 tokens 共享一套参数（`SingleStreamBlock`，`model.py` 第 437-521 行）
- 没有"理解 vs 生成"的参数解耦概念

**Lance 的做法**：
- 引入 `Qwen2MoTDecoderLayer`（`qwen2_navit.py` 第 575-707 行），在每一层中为**理解任务**和**生成任务**维护完全独立的参数：
  - 理解侧：`q_proj`, `k_proj`, `v_proj`, `o_proj`, `q_norm`, `k_norm`, `input_layernorm`, `post_attention_layernorm`, `mlp`
  - 生成侧：`q_proj_moe_gen`, `k_proj_moe_gen`, `v_proj_moe_gen`, `o_proj_moe_gen`, `q_norm_moe_gen`, `k_norm_moe_gen`, `input_layernorm_moe_gen`, `post_attention_layernorm_moe_gen`, `mlp_moe_gen`
- **注意力计算本身是联合的**（joint attention）：理解 tokens 和生成 tokens 的 QKV 分别通过各自的投影层后，合并到一起做统一的注意力计算（`qwen2_navit.py` 第 278-293 行的 `PackedAttentionMoT.forward_train`）

**代码验证**：
- `PackedAttentionMoT`（`qwen2_navit.py` 第 229-484 行）：
  ```python
  # 第 254-257 行：独立的生成侧 QKV+O 投影
  self.q_proj_moe_gen = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
  self.k_proj_moe_gen = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
  self.v_proj_moe_gen = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
  self.o_proj_moe_gen = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
  ```
- `Qwen2MoTDecoderLayer.forward_train`（第 601-644 行）：
  ```python
  # 独立的 LayerNorm
  packed_sequence_[packed_und_token_indexes] = self.input_layernorm(packed_sequence[packed_und_token_indexes])
  packed_sequence_[packed_gen_token_indexes] = self.input_layernorm_moe_gen(packed_sequence[packed_gen_token_indexes])
  # 独立的 MLP
  packed_sequence_[packed_und_token_indexes] = self.mlp(self.post_attention_layernorm(...))
  packed_sequence_[packed_gen_token_indexes] = self.mlp_moe_gen(self.post_attention_layernorm_moe_gen(...))
  ```

**优点分析**：
- **优点方向：多任务训练中的任务冲突缓解**。理解和生成是两种性质不同的任务（自回归 vs 去噪），共享参数可能导致任务间干扰。MoT 通过独立参数解耦两个任务的特征空间，同时通过联合注意力保持信息交互。这是对**多任务统一模型**训练效果的优化，间接有利于生成效果。

---

### 创新点 3：中间还提供 MoE（Mixture of Experts）FFN 变体

**FLUX2 的做法**：
- 不存在 MoE 变体

**Lance 的做法**：
- 除了 MoT 变体（注意力+MLP 都分开）外，还提供 `Qwen2MoEDecoderLayer`（`qwen2_navit.py` 第 710-810 行）：
  - 注意力层：使用共享的 `PackedAttention`（QKV 共享参数）
  - FFN 层：按理解/生成分为 `self.mlp`（理解）和 `self.mlp_moe_gen`（生成）两套独立的 `Qwen2MLP`

**代码验证**：
- `Qwen2MoEDecoderLayer`（`qwen2_navit.py` 第 710-810 行）：
  ```python
  self.self_attn = PackedAttention(config, layer_idx)  # 共享注意力
  self.mlp = Qwen2MLP(config)       # 理解侧 FFN
  self.mlp_moe_gen = Qwen2MLP(config)  # 生成侧 FFN
  ```
- 层类型通过配置选择：`Decoder_layer_dict`（第 813-817 行）支持 `"Qwen2DecoderLayer"`（全共享）、`"Qwen2MoEDecoderLayer"`（MoE FFN）、`"Qwen2MoTDecoderLayer"`（MoT 全分开）

**优点分析**：
- **优点方向：灵活的参数-效率权衡**。MoE 变体在注意力层共享参数（减少参数量）但 FFN 层分开（保留任务解耦），提供了在 MoT（全分开、参数多）和全共享（参数少但任务冲突大）之间的折中方案。这属于模型设计灵活性的优化。

---

### 创新点 4：无独立 Text Encoder，LLM 自身的 embed_tokens 直接处理文本

**FLUX2 的做法**：
- 需要一个独立的大型 Text Encoder：
  - Dev 32B 使用 Mistral-Small-3.2-24B-Instruct（24B 参数的多模态 LLM），提取第 10/20/30 层隐藏状态拼接为 `(B, 512, 15360)` 维度
  - Klein 4B 使用 Qwen3-4B，输出维度 `(B, 512, 7680)`
  - Klein 9B 使用 Qwen3-8B，输出维度 `(B, 512, 12288)`
- 文本编码通过 `txt_in = nn.Linear(context_in_dim, hidden_size, bias=False)`（`model.py` 第 70 行）投影到隐藏空间

**Lance 的做法**：
- 不需要外部 Text Encoder
- 文本 token IDs 直接通过 LLM 自身的 `embed_tokens`（`nn.Embedding`）映射为嵌入向量
- Lance `lance.py` 第 235-237 行：
  ```python
  packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
  packed_sequence = packed_text_embedding.new_zeros(size=(sequence_length, self.hidden_size))
  packed_sequence[packed_text_indexes] = packed_text_embedding[packed_text_indexes]
  ```

**代码验证**：
- FLUX2 的文本输入是已编码的 `ctx` tensor（15360 维），通过 `txt_in` 线性层投影
- Lance 的文本输入是原始 `packed_text_ids`（token IDs），通过 `embed_tokens` 查表得到嵌入

**优点分析**：
- **优点方向：推理效率与架构简洁性**。省去了一个巨大的外部 Text Encoder（FLUX2 Dev 需要 24B 的 Mistral），显著减少推理时的模型数量和显存占用。这是推理效率维度的优化。
- **潜在劣势**：FLUX2 使用大型 Text Encoder 提取的多层语义特征（15360 维拼接 3 层）可能包含更丰富的文本语义信息，对复杂文本描述的理解可能更好。

---

## 二、注意力机制层面的变化创新点

### 创新点 5：基于 flex_attention 的复杂 Sparse Attention Mask 机制

**FLUX2 的做法**：
- 在编辑模式下使用 `causal_attn_fn`（`model.py` 第 758-815 行）实现因果注意力：
  - 序列布局：`[txt, ref, img]`
  - 文本+图像 tokens 可以注意到所有 tokens（全双向）
  - 参考图 tokens 只能注意到自己（self-attention only）
  - 实现方式：手动将 Q/K/V 按区域切分，分别调用 `F.scaled_dot_product_attention`，再拼接结果
- 在纯文生图模式下，所有 tokens 之间是完全双向注意力（`is_causal=False`）

**Lance 的做法**：
- 使用 PyTorch 的 `flex_attention` + `create_block_mask` 实现灵活的稀疏注意力（`lance.py` 第 139-146 行，`data_utils.py` 第 134-161 行）
- 支持多种注意力模式的组合：
  - `causal`：因果注意力（文本区域），Q 只能看到位置 ≤ 自己的 KV
  - `full`：完全双向注意力（ViT 特征区域），区域内所有 token 互相可见
  - `noise`：目标噪声 token 的注意力，可以看到前面的所有 causal/full 区域和自己所在的 noise 区域，但不同 noise 区域之间不可见
  - `full_noise`：源图 clean latent 的注意力模式
  - `full_noise_target`：另一种变体
- 通过 `create_sparse_mask` 函数（`data_utils.py` 第 134-161 行）组合多个 mask 函数：
  ```python
  return and_masks(or_masks(causal_mask, full_and_noise_mask), remove_noise_mask, sample_mask)
  ```
  - `causal_mask`：实现因果注意力基础
  - `full_and_noise_mask`：让同一 full/noise 组内的 token 互相可见
  - `remove_noise_mask`：阻止不同 noise 区域之间的互相注意
  - `sample_mask`：保证不同样本（packed sequence 中）之间不互相注意

**代码验证**：
- Lance `lance.py` 第 240-244 行：
  ```python
  sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes_, packed_text_embedding.device)
  attention_mask = create_block_mask(sparse_mask, B=1, H=self.num_heads, Q_LEN=seqlen, KV_LEN=seqlen, ...)
  ```
- FLUX2 `model.py` 第 758-815 行：手动的三段式 attention 实现

**优点分析**：
- **优点方向：灵活性与可扩展性**。Lance 的 sparse mask 机制支持任意组合的注意力模式，可以轻松适配新的任务类型（如多轮编辑、交错图文序列等），而 FLUX2 的硬编码三段式注意力只能处理固定的 `[txt, ref, img]` 布局。这属于架构灵活性维度的优化。
- **优点方向：计算效率**。`flex_attention` 利用 block-sparse 计算，避免了 FLUX2 中先切分再分别计算再拼接的开销。

---

### 创新点 6：支持 GQA（Grouped Query Attention）

**FLUX2 的做法**：
- 使用标准的 MHA（Multi-Head Attention），`num_heads = num_key_value_heads`
- `SelfAttention`（`model.py` 第 375-387 行）：`self.qkv = nn.Linear(dim, dim * 3, bias=False)`，Q/K/V 维度相同
- 没有 GQA 支持

**Lance 的做法**：
- 支持 GQA，`num_key_value_heads` 可以与 `num_attention_heads` 不同
- `Qwen2Attention`（`modeling_qwen2.py` 第 233-268 行）：
  ```python
  self.num_key_value_heads = config.num_key_value_heads
  self.num_key_value_groups = self.num_heads // self.num_key_value_heads
  self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
  self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
  self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
  ```
- 在 `PackedAttention` 和 `PackedAttentionMoT` 中通过 `enable_gqa=True` 参数传给 `flex_attention`（`qwen2_navit.py` 第 151 行）

**代码验证**：
- Lance `qwen2_navit.py` 第 147-153 行（`PackedAttention.forward_train`）：
  ```python
  packed_attn_output = flex_attention(
      packed_query_states_.unsqueeze(0),
      packed_key_states_.unsqueeze(0),
      packed_value_states.unsqueeze(0),
      enable_gqa=True,
      block_mask=attention_mask,
  )
  ```

**优点分析**：
- **优点方向：推理效率**。GQA 通过减少 K/V 头数来降低 KV Cache 的内存占用和计算量，在不显著影响模型质量的前提下提高推理速度。这属于推理效率维度的优化。

---

### 创新点 7：QK Norm 的分离式设计（理解 QK Norm vs 生成 QK Norm）

**FLUX2 的做法**：
- 使用统一的 `QKNorm`（`model.py` 第 746-755 行），对 Q 和 K 分别做 RMSNorm
- 所有 tokens 共享同一个 QKNorm 参数
- `RMSNorm` 有可学习的 `scale` 参数

**Lance 的做法**：
- 在 MoT 变体中（`PackedAttentionMoT`，`qwen2_navit.py` 第 229-253 行），QK Norm 也按理解/生成分离：
  ```python
  # 理解侧
  self.q_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
  self.k_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
  # 生成侧
  self.q_norm_moe_gen = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
  self.k_norm_moe_gen = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
  ```
- 还支持通过 `qk_norm_und` 和 `qk_norm_gen` 配置独立控制理解/生成侧是否启用 QK Norm（`qwen2_navit.py` 第 232-241 行）

**代码验证**：
- Lance `qwen2_navit.py` 第 301-312 行（`PackedAttentionMoT.forward_train`）：
  ```python
  packed_query_states_[packed_und_token_indexes] = self.q_norm(packed_query_states[packed_und_token_indexes])
  packed_query_states_[packed_gen_token_indexes] = self.q_norm_moe_gen(packed_query_states[packed_gen_token_indexes])
  packed_key_states_[packed_und_token_indexes] = self.k_norm(packed_key_states[packed_und_token_indexes])
  packed_key_states_[packed_gen_token_indexes] = self.k_norm_moe_gen(packed_key_states[packed_gen_token_indexes])
  ```

**优点分析**：
- **优点方向：任务解耦精细度**。将 QK Norm 也按任务分开，使得理解和生成可以学习各自最适合的 query/key 归一化策略。这是 MoT 解耦思想的细化，有助于减少任务冲突，间接改善生成效果。

---

## 三、条件注入方式的变化创新点

### 创新点 8：Timestep 条件注入方式——直接加法 vs AdaLN Modulation

**FLUX2 的做法**：
- 通过 **AdaLN Modulation**（自适应层归一化）注入时间步条件：
  1. `time_in`（`MLPEmbedder`，`model.py` 第 69 行）：将 timestep 嵌入为 `vec`
  2. `guidance_in`（可选）：将 guidance 嵌入加到 `vec` 上
  3. 全局共享的 `Modulation` 层（`model.py` 第 98-108 行）：`SiLU(vec) → Linear → (shift, scale, gate) × N`
  4. 每个 block 使用 `(shift, scale, gate)` 对 LayerNorm 输出进行调制：`x_mod = (1 + scale) * LayerNorm(x) + shift`，输出通过 `gate` 控制
- 最终输出层 `LastLayer`（`model.py` 第 415-434 行）也使用 AdaLN

**Lance 的做法**：
- 通过 **直接加法** 注入时间步条件：
  1. `TimestepEmbedder`（`modeling_utils.py` 第 110-146 行）：`sinusoidal_embedding → Linear(256, hidden_size) → SiLU → Linear(hidden_size, hidden_size)`
  2. timestep embedding 直接加到 VAE latent embedding 上：
     ```python
     # lance.py 第 269-271 行
     packed_timestep_embeds = self.time_embedder(packed_timesteps)
     packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + latent_token_pos_emb
     ```
- **不修改 LLM 内部的任何 LayerNorm 或 attention 结构**，时间步信息在输入嵌入层就融合完毕

**代码验证**：
- FLUX2 `model.py` 第 469-471 行（`SingleStreamBlock._qkv`）：
  ```python
  x_mod = (1 + mod_scale) * self.pre_norm(x) + mod_shift
  ```
- FLUX2 `model.py` 第 482-484 行（`SingleStreamBlock._out`）：
  ```python
  output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
  return x + mod_gate * output
  ```
- Lance `lance.py` 第 271 行：`packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + latent_token_pos_emb`

**优点分析**：
- **优点方向：对预训练 LLM 的最小侵入性**。Lance 不修改 LLM 内部结构，可以直接复用预训练的 Qwen2 权重而无需重新适配 Modulation 层。这对于利用大规模预训练知识至关重要。属于训练效率和知识迁移维度的优化。
- **潜在劣势**：AdaLN Modulation 在每一层都根据 timestep 调制特征，理论上对时间步条件的建模更精细，可能对生成质量有帮助。Lance 的直接加法方式在深层网络中 timestep 信号可能衰减。

---

### 创新点 9：无 Modulation 层设计（全局共享 Modulation 的彻底移除）

**FLUX2 的做法**：
- 有 3 个全局共享的 Modulation 层（`model.py` 第 98-108 行）：
  - `double_stream_modulation_img`：双流块图像侧 modulation（输出 6 个调制量：2 组 (shift, scale, gate)）
  - `double_stream_modulation_txt`：双流块文本侧 modulation（同上）
  - `single_stream_modulation`：单流块 modulation（输出 3 个调制量：1 组 (shift, scale, gate)）
- Modulation 层包含一个 `Linear(hidden_size, multiplier * hidden_size, bias=False)`

**Lance 的做法**：
- **完全没有 Modulation 层**
- LLM 的 Decoder Layer 保持标准结构：`RMSNorm → Attention → Add → RMSNorm → MLP → Add`
- 没有 AdaLN-Zero 的 shift/scale/gate 机制

**代码验证**：
- FLUX2 `model.py` 第 400-412 行：`Modulation` 类定义
- Lance 的 `Qwen2DecoderLayer`（`qwen2_navit.py` 第 487-572 行）和 `Qwen2MoTDecoderLayer`（第 575-707 行）中没有任何 modulation 相关代码

**优点分析**：
- **优点方向：架构简洁性与预训练兼容性**。移除 Modulation 意味着 LLM 骨干完全保持原始结构，可以无缝加载预训练权重。同时减少了参数量和计算量。属于训练便利性和推理效率维度的优化。

---

### 创新点 10：无 Final Layer / 无 AdaLN 输出层

**FLUX2 的做法**：
- 最后有一个 `LastLayer`（`model.py` 第 415-434 行）：
  ```python
  class LastLayer(nn.Module):
      self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False)
      self.linear = nn.Linear(hidden_size, out_channels, bias=False)
      self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=False))
  ```
  - 使用 AdaLN（通过 `vec` 产生 shift 和 scale）对最终输出做一次调制后，再通过线性层投影回 `out_channels=128`

**Lance 的做法**：
- 没有 `LastLayer` / `FinalLayer`
- LLM 的最后一层输出经过 `RMSNorm`（在 MoE/MoT 模式下，生成 tokens 用独立的 `norm_moe_gen`），然后直接通过 `llm2vae`（`nn.Linear(hidden_size, patch_latent_dim)`）投影回 latent 空间

**代码验证**：
- Lance `qwen2_navit.py` 第 895-903 行（`Qwen2Model.forward_train` 结尾）：
  ```python
  if self.use_moe:
      packed_sequence_[packed_und_token_indexes] = self.norm(packed_sequence[packed_und_token_indexes])
      packed_sequence_[packed_gen_token_indexes] = self.norm_moe_gen(packed_sequence[packed_gen_token_indexes])
  else:
      return self.norm(packed_sequence)
  ```
- Lance `lance.py` 第 294 行：`packed_mse_preds = self.llm2vae(last_hidden_state[mse_loss_indexes])`

**优点分析**：
- **优点方向：简洁性**。与创新点 9 一致，保持了 LLM 骨干的原始结构。属于架构简洁性维度的优化。

---

## 四、位置编码层面的变化创新点

### 创新点 11：双重位置编码机制（3D Sincos + RoPE）

**FLUX2 的做法**：
- 使用单一的 **4D RoPE** 位置编码（`model.py` 第 67 行和 694-707 行）：
  - `EmbedND`：对 4 个维度（t, h, w, l）分别计算 RoPE，`axes_dim=[32, 32, 32, 32]`
  - 文本 tokens 使用 (0, 0, 0, l) 坐标，图像 tokens 使用 (t, h, w, 0) 坐标
  - RoPE 通过矩阵旋转形式应用：`apply_rope(q, k, pe)` 将 cos/sin 矩阵与 Q/K 做旋转（`model.py` 第 828-832 行）

**Lance 的做法**：
- 使用**双重位置编码**：
  1. **3D 固定 Sincos 位置编码**（`PositionEmbedding3D`，`modeling_utils.py` 第 183-198 行）：
     - 为 VAE latent tokens 提供**绝对位置信息**
     - 维度分配：时间 T、高度 H、宽度 W 各占 `hidden_size/3` 维（确保偶数）
     - 通过查表实现：`self.pos_embed[position_ids]`
     - 作为**加法**嵌入加到 latent embedding 上
  2. **Qwen2 RoPE**（`modeling_qwen2.py` 第 80-166 行）或 **Qwen2.5-VL M-RoPE**（可选）：
     - 为整个序列提供**相对位置信息**
     - 在 LLM 内部的每个 attention 层中通过旋转应用
     - 标准 RoPE 使用 1D position_ids
     - M-RoPE 使用 3D position_ids（temporal, height, width），每个维度独立旋转

**代码验证**：
- Lance 3D Sincos：`lance.py` 第 270-271 行：
  ```python
  latent_token_pos_emb = self.latent_pos_embed(packed_latent_position_ids)
  packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + latent_token_pos_emb
  ```
- Lance RoPE：`qwen2_navit.py` 第 864-872 行：
  ```python
  if self.apply_qwen_2_5_vl_pos_emb:
      packed_position_embeddings = self.rotary_emb(packed_sequence.unsqueeze(0), packed_position_ids)
  else:
      cos, sin = self.rotary_emb(packed_sequence, packed_position_ids.unsqueeze(0))
  ```
- FLUX2 4D RoPE：`model.py` 第 700-707 行（`EmbedND.forward`）和第 828-832 行（`apply_rope`）

**优点分析**：
- **优点方向：位置信息的多层次表达**。固定 3D Sincos 提供了稳定的绝对空间位置感知（特别是对 VAE latent tokens），RoPE 提供了灵活的相对位置关系建模。双重编码机制理论上能更好地建模图像的空间结构。对生成效果可能有积极影响。

---

### 创新点 12：支持 M-RoPE（Multimodal RoPE）

**FLUX2 的做法**：
- 使用自定义的 4D RoPE（`axes_dim=[32, 32, 32, 32]`），通过矩阵旋转实现
- 不支持 Qwen2.5-VL 的 M-RoPE

**Lance 的做法**：
- 可选启用 Qwen2.5-VL 的 **M-RoPE**（`apply_qwen_2_5_vl_pos_emb` 配置项）
- 使用 `Qwen2_5_VLRotaryEmbedding`（`qwen2_navit.py` 第 837 行）和 `apply_multimodal_rotary_pos_emb`（`qwen2_navit.py` 第 114-118 行）
- M-RoPE 使用 3D position_ids（shape `[3, B, L]`），通过 `mrope_section` 配置将 head_dim 分为三段，分别对应 temporal、height、width 三个维度的旋转

**代码验证**：
- `qwen2_navit.py` 第 111-118 行：
  ```python
  if kwargs.get("apply_qwen_2_5_vl_pos_emb"):
      packed_query_states, packed_key_states = apply_multimodal_rotary_pos_emb(
          packed_query_states, packed_key_states, packed_cos, packed_sin, self.config.rope_scaling["mrope_section"]
      )
  ```
- `qwen2_navit.py` 第 1119-1297 行：`get_rope_index` 方法为不同类型的 tokens（文本、图像、视频）计算各自的 3D 位置 ID

**优点分析**：
- **优点方向：多模态空间关系建模**。M-RoPE 为每个注意力头的不同维度分配不同的空间维度旋转，使模型能更好地建模视频数据的时空结构。相比 FLUX2 的 4D RoPE（每个维度固定分配 32 维），M-RoPE 的维度分配更灵活。对视频相关任务的生成效果有帮助。

---

## 五、输入/输出处理层面的变化创新点

### 创新点 13：VAE Latent 的 Patchify 策略不同

**FLUX2 的做法**：
- VAE 输出的 latent 通道数为 128（32 通道经 2×2 patch 重排后），空间分辨率为 `(H/16, W/16)`
- 不做进一步 patchify，直接 flatten 为 token 序列，每个 token 维度 = 128
- `img_in = nn.Linear(128, hidden_size, bias=False)`（`model.py` 第 68 行）

**Lance 的做法**：
- VAE 输出的 latent 通道数为 48，空间分辨率为 `(T, H/16, W/16)`（3D Causal VAE）
- 进一步 patchify：使用 `latent_patch_size = (1, 2, 2)`，将 latent 从 `(T, H/16, W/16, 48)` 重排为 `(T*H/32*W/32, 1*2*2*48=192)` 的 token 序列
- `vae2llm = nn.Linear(192, hidden_size)`（`lance.py` 第 104 行）

**代码验证**：
- Lance `lance.py` 第 258-261 行：
  ```python
  patches = rearrange(latent, "(t pt) (h ph) (w pw) c -> (t h w) (pt ph pw c)", t=t, pt=pt, h=h, ph=ph, w=w, pw=pw)
  ```
- FLUX2 `sampling.py` 第 141-151 行（`prc_img`）：
  ```python
  x = rearrange(x, "c h w -> (h w) c")  # 直接 flatten，不做额外 patchify
  ```

**优点分析**：
- **优点方向：序列长度压缩**。Lance 的额外 2×2 patchify 将空间维度的 token 数量减少了 4 倍，显著降低了注意力计算的二次复杂度。对于高分辨率图像生成，这带来了更好的计算效率。属于计算效率维度的优化。

---

### 创新点 14：输入/输出投影的简单线性层设计

**FLUX2 的做法**：
- 输入投影：`img_in = nn.Linear(128, 6144, bias=False)`
- 输出投影：`LastLayer`（包含 AdaLN + Linear）
- 从 latent 空间到隐藏空间的映射和反映射是不对称的

**Lance 的做法**：
- 输入投影：`vae2llm = nn.Linear(192, hidden_size)`（有 bias）
- 输出投影：`llm2vae = nn.Linear(hidden_size, 192)`（有 bias）
- 输入和输出使用对称的简单线性层

**代码验证**：
- Lance `lance.py` 第 104-105 行：
  ```python
  self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)
  self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)
  ```

**优点分析**：
- **优点方向：简洁性**。对称的线性投影设计更简洁，且保留了 bias 项以增加表达能力。属于架构简洁性维度的优化。

---

### 创新点 15：最终输出层的独立 RMSNorm（生成侧专用 norm_moe_gen）

**FLUX2 的做法**：
- 最终输出通过 `final_layer`（`LastLayer`）处理，使用 `LayerNorm`（无可学习参数）+ AdaLN modulation

**Lance 的做法**：
- 在 MoE/MoT 模式下，LLM 最后一层输出后有两个独立的 RMSNorm：
  - `self.norm`：用于理解 tokens
  - `self.norm_moe_gen`：用于生成 tokens

**代码验证**：
- `qwen2_navit.py` 第 831-833 行（`Qwen2Model.__init__`）：
  ```python
  self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
  if self.use_moe:
      self.norm_moe_gen = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
  ```
- `qwen2_navit.py` 第 895-901 行（`forward_train` 结尾）：
  ```python
  packed_sequence_[packed_und_token_indexes] = self.norm(packed_sequence[packed_und_token_indexes])
  packed_sequence_[packed_gen_token_indexes] = self.norm_moe_gen(packed_sequence[packed_gen_token_indexes])
  ```

**优点分析**：
- **优点方向：任务解耦的一致性**。与 MoT 架构中每层的独立 LayerNorm 保持一致，最终的归一化也是分开的，确保了理解和生成任务在特征归一化上的完全解耦。属于多任务训练效果维度的优化。

---

## 六、训练机制层面的变化创新点

### 创新点 16：Packed Sequence 多样本高效训练

**FLUX2 的做法**：
- 每次 forward 处理单个样本（batch_size=1 的序列）
- 不支持不同长度样本的打包

**Lance 的做法**：
- 使用 **Packed Sequence** 技术：将多个不同长度的样本打包到一个序列中，通过 `sample_lens` 记录每个样本的长度
- 配合 `flex_attention` 的 `block_mask`（包含 `sample_mask` 确保跨样本不互相注意）实现高效计算
- 同时支持 `flash_attn_varlen_func`（`qwen2_navit.py` 第 209-218 行）用于推理时的变长序列注意力

**代码验证**：
- Lance `lance.py` 第 284-289 行：
  ```python
  last_hidden_state = self.language_model(
      packed_sequence=packed_sequence,
      sample_lens=sample_lens,
      attention_mask=attention_mask,
      ...
  )
  ```
- `data_utils.py` 第 144-145 行（`sample_mask`）：
  ```python
  def sample_mask(b, h, q_idx, kv_idx):
      return document_id[q_idx] == document_id[kv_idx]
  ```

**优点分析**：
- **优点方向：训练效率**。Packed Sequence 减少了 padding 浪费，特别是当不同样本的序列长度差异较大时（如文生图 vs 视频编辑），能显著提高 GPU 利用率。属于训练效率维度的优化。

---

### 创新点 17：可冻结理解侧参数的训练策略（freeze_und）

**FLUX2 的做法**：
- 不存在理解/生成参数解耦的概念，因此不支持选择性冻结

**Lance 的做法**：
- 支持 `freeze_und` 模式（`configuration_qwen2.py` 第 250 行：`self.freeze_und = freeze_und`）
- 在 `freeze_und=True` 时：
  1. 输入层：理解 tokens 的嵌入被 detach（`qwen2_navit.py` 第 860-861 行）
  2. 每个 MoT 层：理解侧的 attention 输出和 MLP 输出被 detach（`qwen2_navit.py` 第 298-299 行和第 630-631 行）
  3. 效果：梯度不会回传到理解侧参数，只训练生成侧（`_moe_gen` 后缀的参数）

**代码验证**：
- `qwen2_navit.py` 第 298-299 行：
  ```python
  if self.config.freeze_und:
      packed_value_states[packed_und_token_indexes] = packed_value_states[packed_und_token_indexes].detach()
  ```
- `qwen2_navit.py` 第 630-631 行：
  ```python
  if self.freeze_und:
      packed_sequence_[packed_und_token_indexes] = packed_sequence_[packed_und_token_indexes].detach()
  ```

**优点分析**：
- **优点方向：分阶段训练策略**。在多任务训练中，可以先训练理解侧（利用大量理解数据），再冻结理解侧只训练生成侧（利用生成数据），避免生成训练破坏已有的理解能力。属于训练策略维度的优化。

---

## 七、CFG 与采样层面的变化创新点

### 创新点 18：两级 CFG（Text + ViT）

**FLUX2 的做法**：
- 支持单级 CFG：`denoise_cfg`（`sampling.py` 第 364-410 行），通过 `pred = pred_uncond + guidance * (pred_cond - pred_uncond)` 实现
- 或者通过 `guidance_embed` 直接注入 guidance 值（无需双倍计算）

**Lance 的做法**：
- 支持两级 CFG：
  - `cfg_text_scale`：文本条件的引导强度
  - `cfg_vit_scale`：ViT 视觉条件的引导强度
  - 公式（`lance.py` 第 612 行）：
    ```python
    v_t_ = cfg_text_vit_v_t + cfg_text_scale_ * (v_t - cfg_text_v_t) + cfg_vit_scale_ * (cfg_text_v_t - cfg_text_vit_v_t)
    ```
  - 其中 `v_t` = 条件完整预测，`cfg_text_v_t` = 去掉文本条件的预测，`cfg_text_vit_v_t` = 去掉文本+ViT条件的预测

**代码验证**：
- Lance `lance.py` 第 556-614 行：
  ```python
  if cfg_text_scale_ > 1.0:
      cfg_text_v_t = self.uncond_forward(...)
      if cfg_vit_pro:
          cfg_text_vit_v_t = self.uncond_forward(...)
          v_t_ = cfg_text_vit_v_t + cfg_text_scale_ * (v_t - cfg_text_v_t) + cfg_vit_scale_ * (cfg_text_v_t - cfg_text_vit_v_t)
  ```

**优点分析**：
- **优点方向：编辑任务的可控性**。在图像编辑场景中，可以独立控制文本指令和视觉参考图的引导强度，提供更精细的编辑控制。对编辑任务的生成效果有直接帮助。

---

### 创新点 19：CFG Renormalization

**FLUX2 的做法**：
- 没有 CFG renormalization 机制

**Lance 的做法**：
- 实现了 CFG Renormalization（`lance.py` 第 616-628 行），支持 `global` 和 `channel` 两种模式：
  ```python
  if cfg_renorm_type == "global":
      norm_v_t = torch.norm(v_t)
      norm_v_t_ = torch.norm(v_t_)
      scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
  elif cfg_renorm_type == "channel":
      norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
      norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
      scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
  v_t = v_t_ * scale
  ```

**优点分析**：
- **优点方向：生成质量**。CFG 会放大预测幅度，可能导致色彩过饱和或伪影。Renormalization 将 CFG 后的速度场范数约束在原始范数附近，防止过度放大。对生成效果有直接的正面影响。

---

### 创新点 20：Timestep Shift 采样策略不同

**FLUX2 的做法**：
- 使用 **广义时间 SNR shift**（`sampling.py` 第 240-248 行）：
  ```python
  def generalized_time_snr_shift(t, mu, sigma):
      return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
  ```
  - `mu` 根据图像序列长度和步数动态计算（`compute_empirical_mu`，第 251-266 行）
  - 使用经验公式 `a * image_seq_len + b` 拟合最优 `mu`

**Lance 的做法**：
- 使用 **线性 timestep shift**（`lance.py` 第 267 行和 510-511 行）：
  ```python
  packed_timesteps = self.timestep_shift * packed_timesteps / (1 + (self.timestep_shift - 1) * packed_timesteps)
  ```
  - `timestep_shift` 是一个固定的超参数（如 4.0）
  - timesteps 线性等间距生成：`timesteps = torch.linspace(1, 0, num_timesteps + 1)`

**代码验证**：
- Lance `lance.py` 第 510-512 行（推理）：
  ```python
  timesteps = torch.linspace(1, 0, num_timesteps + 1, device=x_t.device)
  timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
  dts = timesteps[:-1] - timesteps[1:]
  ```

**优点分析**：
- **比较说明**：两种方式各有特点。FLUX2 的自适应 shift 根据分辨率动态调整，理论上更优；Lance 的固定 shift 更简单直接。此项不构成明显优势。

---

## 八、Normalization 层面的变化创新点

### 创新点 21：RMSNorm（有可学习参数）vs LayerNorm（无可学习参数）

**FLUX2 的做法**：
- Transformer block 内部使用 `nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)`，即**无可学习参数的 LayerNorm**
- 示例：`model.py` 第 537 行：`self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)`
- Modulation 的 shift/scale 参数替代了 LayerNorm 的可学习参数

**Lance 的做法**：
- 使用 `Qwen2RMSNorm`（`modeling_qwen2.py` 第 59-76 行），即**有可学习 weight 参数的 RMSNorm**：
  ```python
  class Qwen2RMSNorm(nn.Module):
      def __init__(self, hidden_size, eps=1e-6):
          self.weight = nn.Parameter(torch.ones(hidden_size))
      def forward(self, hidden_states):
          variance = hidden_states.pow(2).mean(-1, keepdim=True)
          hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
          return self.weight * hidden_states.to(input_dtype)
  ```
- RMSNorm 只使用均方根归一化（不减均值），计算更快；同时保留可学习的 weight 参数

**优点分析**：
- **优点方向：计算效率**。RMSNorm 相比 LayerNorm 省去了均值计算，速度更快。但这更多是 LLM 的标准做法，不是 Lance 的独创。

---

### 创新点 22：线性层 bias 设计不同

**FLUX2 的做法**：
- 几乎所有线性层都设置 `bias=False`：
  - `img_in`、`txt_in`、`SelfAttention.qkv`、`SelfAttention.proj`、`MLP` 中的线性层等
  - Modulation 中的 `lin` 也是 `bias=False`（`disable_bias=True`）

**Lance 的做法**：
- QKV 投影层有 `bias=True`：
  ```python
  self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
  self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
  self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
  ```
- O 投影层 `bias=False`：`self.o_proj = nn.Linear(..., bias=False)`
- MLP 的三个线性层都是 `bias=False`：`gate_proj`、`up_proj`、`down_proj`
- `vae2llm` 和 `llm2vae` 使用默认 `bias=True`
- `TimestepEmbedder` 的 MLP 使用 `bias=True`

**优点分析**：
- **比较说明**：QKV 的 bias 可以增加表达能力，但增加了参数量。这更多是 LLM 的设计惯例（Qwen2 的原始设计），不构成明显的生成效果优势。

---

## 九、MLP / FFN 层面的变化创新点

### 创新点 23：SwiGLU（Gate-Up-Down）vs SiLU Gated（chunk-gate）

**FLUX2 的做法**：
- 使用 **SiLU Gated Activation**（`SiLUActivation`，`model.py` 第 390-397 行）：
  ```python
  class SiLUActivation(nn.Module):
      def forward(self, x):
          x1, x2 = x.chunk(2, dim=-1)
          return self.gate_fn(x1) * x2  # SiLU(x1) * x2
  ```
- MLP 结构：`Linear(hidden → 2*mlp_hidden) → SiLUActivation(chunk+gate) → Linear(mlp_hidden → hidden)`
- mlp_ratio = 3.0，即 `mlp_hidden_dim = int(hidden_size * 3.0)`

**Lance 的做法**：
- 使用 **SwiGLU（Standard Gate-Up-Down）**（`Qwen2MLP`，`modeling_qwen2.py` 第 206-217 行）：
  ```python
  class Qwen2MLP(nn.Module):
      def __init__(self, config):
          self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
          self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
          self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
          self.act_fn = ACT2FN[config.hidden_act]  # silu
      def forward(self, hidden_state):
          return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))
  ```
- 使用三个独立的线性层（gate、up、down），而非 FLUX2 的两个线性层

**优点分析**：
- **比较说明**：两者本质上都是 SiLU gated MLP 的变体，数学上等价。FLUX2 通过一个大线性层 + chunk 实现，Lance 通过两个独立线性层（gate + up）实现。后者是 LLM 领域的标准做法（LLaMA/Qwen 系列），两者在表达能力上基本等同。不构成明显优势。

---

## 十、总结表

| # | 变化创新点 | FLUX2 做法 | Lance 做法 | 优化维度 |
|---|-----------|-----------|-----------|---------|
| 1 | 去噪骨干网络 | 独立 MMDIT (DS 8层 + SS 48层) | Qwen2 LLM Causal Decoder | 多任务统一性、参数效率 |
| 2 | MoT 解耦 | 无 | 理解/生成独立 QKV+O+Norm+MLP | 多任务冲突缓解 |
| 3 | MoE FFN 变体 | 无 | 注意力共享、FFN 分开 | 参数-效率权衡灵活性 |
| 4 | Text Encoder | 外部大模型 (24B Mistral / Qwen3) | LLM 自身 embed_tokens | 推理效率、架构简洁性 |
| 5 | Sparse Attention | 硬编码三段式 causal_attn_fn | flex_attention + create_block_mask | 灵活性、可扩展性 |
| 6 | GQA 支持 | 不支持 (MHA) | 支持 GQA | 推理效率 |
| 7 | QK Norm 分离 | 统一 QKNorm | 理解/生成独立 QK Norm | 任务解耦精细度 |
| 8 | Timestep 注入 | AdaLN Modulation (每层调制) | 直接加法 (仅输入层) | 预训练兼容性、训练效率 |
| 9 | Modulation 层 | 3 个全局共享 Modulation | 完全没有 | 架构简洁性、预训练兼容性 |
| 10 | 输出层 | LastLayer (AdaLN + Linear) | 直接 RMSNorm + Linear | 架构简洁性 |
| 11 | 位置编码 | 单一 4D RoPE | 双重：3D Sincos + RoPE | 位置信息多层次表达 |
| 12 | M-RoPE 支持 | 不支持 | 可选启用 Qwen2.5-VL M-RoPE | 多模态空间关系建模 |
| 13 | Latent Patchify | 直接 flatten (128维) | 额外 (1,2,2) patchify (192维) | 序列长度压缩、计算效率 |
| 14 | 输入/输出投影 | 不对称 (有/无 AdaLN) | 对称简单 Linear | 架构简洁性 |
| 15 | 最终 Norm | LayerNorm (无可学习参数) | RMSNorm (理解/生成独立) | 任务解耦一致性 |
| 16 | 训练机制 | 单样本处理 | Packed Sequence 多样本打包 | 训练效率 |
| 17 | 可冻结训练 | 不支持 | freeze_und 冻结理解侧 | 分阶段训练策略 |
| 18 | CFG 级数 | 单级 (text only) | 两级 (text + ViT) | 编辑可控性 |
| 19 | CFG Renorm | 无 | global / channel renorm | 生成质量 |
| 20 | Timestep Schedule | 自适应 generalized SNR shift | 固定 linear shift | (各有特点) |
| 21 | Norm 类型 | LayerNorm (无参数) + Modulation | RMSNorm (有可学习 weight) | 计算效率 |
| 22 | Linear bias | 几乎全部 bias=False | QKV bias=True, MLP bias=False | (设计惯例差异) |
| 23 | MLP 结构 | SiLU chunk-gate (2 层) | SwiGLU gate-up-down (3 层) | (等效设计) |

---

> **报告完成时间**: 基于 Lance 和 FLUX2 目录全部去噪模型相关源代码的逐行分析
>
> **分析的关键文件**:
> - Lance: `modeling/lance/lance.py`, `modeling/lance/qwen2_navit.py`, `modeling/lance/modeling_utils.py`, `modeling/qwen2/modeling_qwen2.py`, `modeling/qwen2/configuration_qwen2.py`, `data/data_utils.py`
> - FLUX2: `src/flux2/model.py`, `src/flux2/sampling.py`