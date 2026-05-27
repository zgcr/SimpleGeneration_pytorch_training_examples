"""
文生图 (Text-to-Image) 完整训练Pipeline示例

推断依据：
- FireRed: VAE编码→pack→加噪→Transformer预测velocity→MSE loss
- Z-Image/JoyAI: 类似流程，VLM提取embedding→DiT去噪

结论：标准 Flow Matching 训练流程：
1. VLM (冻结) 提取文本embedding
2. VAE (冻结) 编码图像为latent
3. Pack latent为token序列
4. 采样时间步，加噪
5. MMDIT预测velocity
6. 计算FM loss，反向传播
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from einops import rearrange

# ============ 以下为示例用的mock组件 ============


class MockT2IDataset(Dataset):
    """模拟文生图数据集 (实际使用时替换为真实数据集)"""

    def __init__(self, num_samples=1000, image_size=512):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 模拟: 随机图像 + 文本prompt
        image = torch.randn(3, self.image_size, self.image_size)
        text = f"A beautiful photo of scenery {idx}"
        return {"image": image, "text": text}


# ============ 训练主函数 ============


def train_t2i(
    model_size: str = "1B",
    num_epochs: int = 100,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    image_size: int = 512,
    device: str = "cuda",
    vlm_model_name: str = "Qwen/Qwen3.5-4B",
    vae_checkpoint: str = None,
    save_dir: str = "./checkpoints_t2i",
):
    """
    文生图训练Pipeline

    流程：
    1. 加载冻结的VLM (Qwen3.5-4B)
    2. 加载冻结的VAE (flux2 autoencoder)
    3. 构建可训练的MMDIT模型
    4. 训练循环
    """
    from ..models.mmdit import (MODEL_FACTORY, create_image_ids,
                                create_txt_ids)
    from ..losses import (sample_timesteps_logit_normal,
                          add_noise_flow_matching, compute_flow_matching_loss)

    os.makedirs(save_dir, exist_ok=True)

    # ========== 1. 加载VLM (冻结) ==========
    print("[1/4] Loading VLM (frozen)...")
    # 实际使用时:
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # vlm_tokenizer = AutoTokenizer.from_pretrained(vlm_model_name)
    # vlm_model = AutoModelForCausalLM.from_pretrained(
    #     vlm_model_name, torch_dtype=torch.bfloat16)
    # vlm_model.eval().requires_grad_(False).to(device)

    # 示例用mock VLM:
    vlm_hidden_dim = 2560  # Qwen3.5-4B hidden size
    print(f"  VLM hidden_dim = {vlm_hidden_dim}")

    # ========== 2. 加载VAE (冻结) ==========
    print("[2/4] Loading VAE (frozen)...")
    # 实际使用时:
    # from SimpleGeneration.flux_autoencoder.models.flux2_autoencoder import AutoEncoder
    # vae = AutoEncoder(inplanes=3, planes=128, planes_mult=[1,2,4,4],
    #                   res_block_nums=2, z_planes=32, out_planes=3)
    # vae.load_state_dict(torch.load(vae_checkpoint))
    # vae.eval().requires_grad_(False).to(device)

    vae_z_planes = 32
    vae_downsample = 8  # 3次下采样
    vae_pack_factor = 2  # patchify 2x2
    latent_channels = vae_z_planes * vae_pack_factor * vae_pack_factor  # 128
    latent_h = image_size // vae_downsample // vae_pack_factor
    latent_w = image_size // vae_downsample // vae_pack_factor
    print(f"  Latent shape: ({latent_channels}, {latent_h}, {latent_w})")

    # ========== 3. 构建MMDIT (可训练) ==========
    print(f"[3/4] Building MMDIT model ({model_size})...")
    model_fn = MODEL_FACTORY[model_size]
    model = model_fn(use_gradient_checkpoint=True).to(device)

    num_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  MMDIT parameters: {num_params:.2f}B")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=learning_rate,
                                  weight_decay=0.01)

    # ========== 4. 训练循环 ==========
    print("[4/4] Starting training...")
    dataset = MockT2IDataset(num_samples=1000, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_steps = 0

        for batch in dataloader:
            # ---- Step A: VLM编码文本 ----
            # 实际使用时:
            # inputs = vlm_tokenizer(batch["text"], return_tensors="pt",
            #                        padding=True, truncation=True,
            #                        max_length=512).to(device)
            # with torch.no_grad():
            #     vlm_out = vlm_model(**inputs, output_hidden_states=True)
            #     ctx = vlm_out.hidden_states[-1]  # [B, seq, 2560]

            # Mock VLM output:
            B = len(batch["text"])
            txt_seq_len = 64  # 模拟文本序列长度
            ctx = torch.randn(B,
                              txt_seq_len,
                              vlm_hidden_dim,
                              device=device,
                              dtype=torch.bfloat16)

            # ---- Step B: VAE编码图像 ----
            # 实际使用时:
            # images = batch["image"].to(device)
            # with torch.no_grad():
            #     latents = vae.encode(images)  # [B, 128, h, w] (已pack)
            # latents = rearrange(latents, "b c h w -> b (h w) c")

            # Mock latent:
            latents = torch.randn(B,
                                  latent_h * latent_w,
                                  latent_channels,
                                  device=device,
                                  dtype=torch.bfloat16)

            # ---- Step C: 构建位置ID ----
            img_ids = create_image_ids(latent_h, latent_w,
                                       device).unsqueeze(0).expand(B, -1, -1)
            txt_ids = create_txt_ids(txt_seq_len,
                                     device).unsqueeze(0).expand(B, -1, -1)

            # ---- Step D: 采样时间步 + 加噪 ----
            noise = torch.randn_like(latents)
            timesteps = sample_timesteps_logit_normal(B, device)
            noisy_latents = add_noise_flow_matching(latents, noise, timesteps)

            # ---- Step E: MMDIT前向 ----
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                v_pred = model(
                    x=noisy_latents,
                    x_ids=img_ids,
                    timesteps=timesteps,
                    ctx=ctx,
                    ctx_ids=txt_ids,
                )

            # ---- Step F: 计算loss ----
            loss = compute_flow_matching_loss(
                model_output=v_pred,
                noise=noise,
                clean_latents=latents,
                sigmas=timesteps,
                weighting_scheme="none",
            )

            # ---- Step G: 反向传播 ----
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_steps += 1

        avg_loss = total_loss / max(num_steps, 1)
        print(f"  Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.6f}")

        # 保存checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(save_dir, f"mmdit_t2i_ep{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    print("Training complete!")


if __name__ == "__main__":
    train_t2i(model_size="1B", num_epochs=2, batch_size=2)
