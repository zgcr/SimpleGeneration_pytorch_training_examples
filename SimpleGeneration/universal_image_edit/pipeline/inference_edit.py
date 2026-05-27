"""
扩展任务 (Image Editing) 完整推理Pipeline示例
支持：inpainting、局部改动、风格迁移、多图参考输入编辑等

推断依据：
- Flux2: 参考图编码→拼接到img token→共享attention去噪
- FireRed: source latent拼接→Transformer预测→截取目标部分

结论：
1. VLM编码文本+可选参考图信息
2. VAE编码参考图为latent
3. 初始化目标图为纯噪声
4. 拼接[噪声target, 干净ref]送入MMDIT
5. Euler ODE去噪，每步仅更新target部分
6. VAE解码
"""
import torch
from torch import Tensor
from einops import rearrange
from typing import List, Optional, Union
from PIL import Image
import numpy as np


@torch.no_grad()
def inference_edit(
    model,
    vae,
    vlm_model,
    vlm_tokenizer,
    prompt: Union[str, List[str]],
    source_images: List[Image.Image],
    negative_prompt: Optional[str] = "",
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    device: str = "cuda",
) -> List[Image.Image]:
    """
    图像编辑推理Pipeline

    Args:
        model: 训练好的MMDIT模型
        vae: 冻结的VAE
        vlm_model: 冻结的VLM
        vlm_tokenizer: VLM tokenizer
        prompt: 编辑指令文本
        source_images: 参考图列表 (1~N张PIL Image)
        negative_prompt: 负面提示
        height: 输出图像高度
        width: 输出图像宽度
        num_inference_steps: 去噪步数
        guidance_scale: CFG强度
        generator: 随机数生成器
        device: 计算设备

    Returns:
        编辑后的PIL Image列表
    """
    from ..models.mmdit import create_image_ids, create_txt_ids
    from ..losses import compute_shift
    from .scheduler import FlowMatchEulerScheduler

    if isinstance(prompt, str):
        prompt = [prompt]
    batch_size = len(prompt)
    num_refs = len(source_images)
    do_cfg = guidance_scale > 1.0

    # ========== 1. VLM编码 (文本 + 参考图上下文) ==========
    # 实际使用时 (FireRed风格):
    # messages = [
    #     {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
    #     {"role": "user", "content": [
    #         *[{"type": "image", "image": img} for img in source_images],
    #         {"type": "text", "text": prompt[0]}
    #     ]}
    # ]
    # inputs = processor.apply_chat_template(messages, ...)
    # vlm_out = vlm_model(**inputs, output_hidden_states=True)
    # prompt_embeds = vlm_out.hidden_states[-1]

    vlm_hidden_dim = 2560
    txt_seq_len = 128  # 图文混合输入通常更长
    prompt_embeds = torch.randn(batch_size,
                                txt_seq_len,
                                vlm_hidden_dim,
                                device=device,
                                dtype=torch.bfloat16)
    if do_cfg:
        neg_embeds = torch.zeros(batch_size,
                                 txt_seq_len,
                                 vlm_hidden_dim,
                                 device=device,
                                 dtype=torch.bfloat16)
        prompt_embeds = torch.cat([neg_embeds, prompt_embeds], dim=0)

    # ========== 2. VAE编码参考图 ==========
    vae_downsample = 8
    vae_pack_factor = 2
    vae_z_planes = 32
    latent_channels = vae_z_planes * vae_pack_factor * vae_pack_factor
    latent_h = height // vae_downsample // vae_pack_factor
    latent_w = width // vae_downsample // vae_pack_factor
    num_img_tokens = latent_h * latent_w

    # 实际使用时:
    # ref_latents = []
    # for img in source_images:
    #     img_tensor = preprocess(img).to(device)  # [1, 3, H, W]
    #     lat = vae.encode(img_tensor)  # [1, 128, h, w]
    #     lat = rearrange(lat, "1 c h w -> (h w) c")
    #     ref_latents.append(lat)
    # ref_latents = torch.cat(ref_latents, dim=0)  # [N*h*w, 128]
    # ref_latents = ref_latents.unsqueeze(0).expand(B, -1, -1)

    # Mock:
    ref_latents = torch.randn(batch_size,
                              num_refs * num_img_tokens,
                              latent_channels,
                              device=device,
                              dtype=torch.bfloat16)

    # ========== 3. 初始化目标区域纯噪声 ==========
    target_latents = torch.randn(batch_size,
                                 num_img_tokens,
                                 latent_channels,
                                 device=device,
                                 dtype=torch.float32,
                                 generator=generator)

    # ========== 4. 构建位置ID ==========
    target_ids = create_image_ids(latent_h, latent_w, device)

    ref_ids_list = []
    for ref_idx in range(num_refs):
        offset_h = (ref_idx + 1) * latent_h + 10
        h_ids = torch.arange(offset_h,
                             offset_h + latent_h,
                             device=device,
                             dtype=torch.float32)
        w_ids = torch.arange(latent_w, device=device, dtype=torch.float32)
        grid = torch.meshgrid(h_ids, w_ids, indexing="ij")
        ref_id = torch.stack(grid, dim=-1).reshape(-1, 2)
        ref_ids_list.append(ref_id)
    all_ref_ids = torch.cat(ref_ids_list, dim=0)

    all_img_ids = torch.cat([target_ids, all_ref_ids], dim=0)
    all_img_ids = all_img_ids.unsqueeze(0).expand(batch_size, -1, -1)

    txt_ids = create_txt_ids(txt_seq_len, device)
    txt_ids = txt_ids.unsqueeze(0).expand(batch_size, -1, -1)

    if do_cfg:
        all_img_ids = torch.cat([all_img_ids, all_img_ids], dim=0)
        txt_ids = torch.cat([txt_ids, txt_ids], dim=0)
        ref_latents_cfg = torch.cat([ref_latents, ref_latents], dim=0)
    else:
        ref_latents_cfg = ref_latents

    # ========== 5. 设置scheduler ==========
    scheduler = FlowMatchEulerScheduler(num_train_timesteps=1000)
    mu = compute_shift(num_img_tokens)
    scheduler.set_timesteps(num_inference_steps, device=device, mu=mu)

    # ========== 6. 去噪循环 ==========
    model.eval()
    for i, t in enumerate(scheduler.timesteps):
        timestep = t / 1000.0

        if do_cfg:
            latent_input = torch.cat([target_latents, target_latents], dim=0)
            t_input = timestep.expand(batch_size * 2)
        else:
            latent_input = target_latents
            t_input = timestep.expand(batch_size)

        # 拼接 [target(noisy), ref(clean)]
        x_input = torch.cat([latent_input.to(torch.bfloat16), ref_latents_cfg],
                            dim=1)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            v_pred = model(
                x=x_input,
                x_ids=all_img_ids,
                timesteps=t_input,
                ctx=prompt_embeds,
                ctx_ids=txt_ids,
            )

        # 仅取目标图部分的预测
        v_pred = v_pred[:, :num_img_tokens, :]

        if do_cfg:
            v_uncond, v_cond = v_pred.chunk(2)
            v_pred = v_uncond + guidance_scale * (v_cond - v_uncond)

        # Euler step (仅更新target)
        target_latents = scheduler.step(v_pred, t, target_latents)[0]

    # ========== 7. VAE解码 ==========
    latents_2d = rearrange(target_latents,
                           "b (h w) c -> b c h w",
                           h=latent_h,
                           w=latent_w)
    # 实际: images = vae.decode(latents_2d)
    images = torch.randn(batch_size, 3, height, width, device=device)
    images = images.clamp(0, 1)

    # ========== 8. 转PIL ==========
    images_np = images.cpu().float().permute(0, 2, 3, 1).numpy()
    images_np = (images_np * 255).round().astype(np.uint8)
    pil_images = [Image.fromarray(img) for img in images_np]

    return pil_images


if __name__ == "__main__":
    print("Image Editing Inference Pipeline")
    print("Usage: inference_edit(model, vae, vlm, tokenizer, prompt, images)")
