"""
扩展任务 (Image Editing) 完整训练Pipeline示例
支持：inpainting、局部改动、风格迁移、多图参考输入编辑等

推断依据：
- FireRed: source images编码为latent→pack→拼接到noisy_latents后面
- Flux2: 参考图编码为token→拼接到img序列→共享attention
- JoyAI: Multi-item input，参考图放temporal维度

结论：采用FireRed/Flux2方案：
1. VLM (冻结) 编码文本+可选图片信息
2. VAE (冻结) 编码目标图和参考图
3. 参考图latent token拼接到噪声token后方
4. MMDIT在joint attention中自然融合参考图信息
5. 仅对目标图部分计算loss

注意：不输入mask，inpaint类任务通过文本指令隐式指定区域
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from einops import rearrange


class MockEditDataset(Dataset):
    """模拟编辑数据集"""

    def __init__(self, num_samples=1000, image_size=512, max_ref_images=3):
        self.num_samples = num_samples
        self.image_size = image_size
        self.max_ref_images = max_ref_images

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        import random
        num_refs = random.randint(1, self.max_ref_images)
        target_image = torch.randn(3, self.image_size, self.image_size)
        source_images = [
            torch.randn(3, self.image_size, self.image_size)
            for _ in range(num_refs)
        ]
        text = f"Edit the image: change the style to watercolor {idx}"
        return {
            "target_image": target_image,
            "source_images": source_images,
            "text": text,
        }


def collate_edit_fn(batch):
    """自定义collate：处理不定数量的参考图"""
    target_images = [item["target_image"] for item in batch]
    source_images_list = [item["source_images"] for item in batch]
    texts = [item["text"] for item in batch]
    return {
        "target_images": target_images,
        "source_images_list": source_images_list,
        "texts": texts,
    }


def train_edit(
    model_size: str = "1B",
    num_epochs: int = 100,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    image_size: int = 512,
    device: str = "cuda",
    vlm_model_name: str = "Qwen/Qwen3.5-4B",
    save_dir: str = "./checkpoints_edit",
):
    """
    扩展任务(编辑)训练Pipeline

    关键设计：
    - 参考图通过VAE编码后拼接到img token序列
    - 位置ID中参考图使用独立的空间坐标（不与目标图重叠）
    - 仅对目标图token计算loss
    """
    from ..models.mmdit import (MODEL_FACTORY, create_image_ids,
                                create_txt_ids)
    from ..losses import (sample_timesteps_logit_normal,
                          add_noise_flow_matching, compute_flow_matching_loss)

    os.makedirs(save_dir, exist_ok=True)

    # ========== 加载组件 ==========
    print("[1/4] Loading VLM (frozen)...")
    vlm_hidden_dim = 2560  # Qwen3.5-4B
    # 实际使用时加载真实VLM (参见train_t2i.py中的注释)

    print("[2/4] Loading VAE (frozen)...")
    vae_z_planes = 32
    vae_downsample = 8
    vae_pack_factor = 2
    latent_channels = vae_z_planes * vae_pack_factor * vae_pack_factor
    latent_h = image_size // vae_downsample // vae_pack_factor
    latent_w = image_size // vae_downsample // vae_pack_factor
    num_img_tokens = latent_h * latent_w

    print(f"[3/4] Building MMDIT model ({model_size})...")
    model_fn = MODEL_FACTORY[model_size]
    model = model_fn(use_gradient_checkpoint=True).to(device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  MMDIT parameters: {num_params:.2f}B")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=learning_rate,
                                  weight_decay=0.01)

    # ========== 训练循环 ==========
    print("[4/4] Starting training...")
    dataset = MockEditDataset(num_samples=500, image_size=image_size)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=collate_edit_fn)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_steps = 0

        for batch in dataloader:
            B = len(batch["texts"])

            # ---- Step A: VLM编码 (文本+可选图片上下文) ----
            # 实际使用时，可以将参考图也通过VLM处理(如FireRed):
            # messages = [{"role": "user", "content": [...images, text]}]
            # vlm_out = vlm_model(messages)
            # ctx = vlm_out.hidden_states[-1]

            txt_seq_len = 128
            ctx = torch.randn(B,
                              txt_seq_len,
                              vlm_hidden_dim,
                              device=device,
                              dtype=torch.bfloat16)

            # ---- Step B: VAE编码目标图 + 参考图 ----
            # 实际使用时:
            # target_latents = vae.encode(target_images)
            # ref_latents_list = [vae.encode(ref) for ref in source_images]

            target_latents = torch.randn(B,
                                         num_img_tokens,
                                         latent_channels,
                                         device=device,
                                         dtype=torch.bfloat16)

            # 参考图latent (假设batch内所有样本参考图数量相同)
            num_refs = len(batch["source_images_list"][0])
            ref_latents_per_sample = torch.randn(B,
                                                 num_refs * num_img_tokens,
                                                 latent_channels,
                                                 device=device,
                                                 dtype=torch.bfloat16)

            # ---- Step C: 构建位置ID ----
            # 目标图位置ID
            target_ids = create_image_ids(latent_h, latent_w, device)

            # 参考图位置ID：使用偏移坐标避免与目标图重叠
            ref_ids_list = []
            for ref_idx in range(num_refs):
                offset_h = (ref_idx + 1) * latent_h + 10
                h_ids = torch.arange(offset_h,
                                     offset_h + latent_h,
                                     device=device,
                                     dtype=torch.float32)
                w_ids = torch.arange(latent_w,
                                     device=device,
                                     dtype=torch.float32)
                grid = torch.meshgrid(h_ids, w_ids, indexing="ij")
                ref_ids = torch.stack(grid, dim=-1).reshape(-1, 2)
                ref_ids_list.append(ref_ids)
            all_ref_ids = torch.cat(ref_ids_list, dim=0)

            # 拼接：[target_tokens, ref_tokens]
            all_img_ids = torch.cat([target_ids, all_ref_ids],
                                    dim=0).unsqueeze(0).expand(B, -1, -1)

            txt_ids = create_txt_ids(txt_seq_len,
                                     device).unsqueeze(0).expand(B, -1, -1)

            # ---- Step D: 加噪（仅目标图加噪，参考图保持干净） ----
            noise = torch.randn_like(target_latents)
            timesteps = sample_timesteps_logit_normal(B, device)
            noisy_target = add_noise_flow_matching(target_latents, noise,
                                                   timesteps)

            # 拼接噪声目标 + 干净参考图
            x_input = torch.cat([noisy_target, ref_latents_per_sample], dim=1)

            # ---- Step E: MMDIT前向 ----
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                v_pred = model(
                    x=x_input,
                    x_ids=all_img_ids,
                    timesteps=timesteps,
                    ctx=ctx,
                    ctx_ids=txt_ids,
                )

            # ---- Step F: 仅对目标图token计算loss ----
            v_pred_target = v_pred[:, :num_img_tokens, :]
            loss = compute_flow_matching_loss(
                model_output=v_pred_target,
                noise=noise,
                clean_latents=target_latents,
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

        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(save_dir, f"mmdit_edit_ep{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    print("Training complete!")


if __name__ == "__main__":
    train_edit(model_size="1B", num_epochs=2, batch_size=2)
