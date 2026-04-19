"""
文生图 (Text-to-Image) 完整推理Pipeline示例

推断依据：
- Flux2: denoise loop with Euler ODE solver, CFG支持
- Z-Image: dynamic shifting scheduler, CFG truncation
- JoyAI: FlowMatchDiscreteScheduler, classifier-free guidance

结论：采用标准Flow Matching推理流程：
1. VLM (冻结) 编码文本prompt
2. 初始化纯高斯噪声
3. Euler ODE solver逐步去噪 (支持CFG)
4. VAE (冻结) 解码latent为图像
"""
import torch
from torch import Tensor
from einops import rearrange
from typing import List, Optional, Union
from PIL import Image
import numpy as np


@torch.no_grad()
def inference_t2i(
    model,
    vae,
    vlm_model,
    vlm_tokenizer,
    prompt: Union[str, List[str]],
    negative_prompt: Optional[str] = "",
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    device: str = "cuda",
) -> List[Image.Image]:
    """
    文生图推理Pipeline

    Args:
        model: 训练好的MMDIT模型
        vae: 冻结的VAE (flux2 autoencoder)
        vlm_model: 冻结的VLM (Qwen3.5-4B)
        vlm_tokenizer: VLM tokenizer
        prompt: 文本提示
        negative_prompt: 负面提示 (用于CFG)
        height: 输出图像高度
        width: 输出图像宽度
        num_inference_steps: 去噪步数
        guidance_scale: CFG强度，>1启用CFG
        generator: 随机数生成器
        device: 计算设备

    Returns:
        PIL Image列表
    """
    from ..models.mmdit import create_image_ids, create_txt_ids
    from ..losses import compute_shift
    from .scheduler import FlowMatchEulerScheduler

    if isinstance(prompt, str):
        prompt = [prompt]
    batch_size = len(prompt)

    do_cfg = guidance_scale > 1.0

    # ========== 1. VLM编码文本 ==========
    # 实际使用时:
    # formatted = [vlm_tokenizer.apply_chat_template(
    #     [{"role": "user", "content": p}],
    #     tokenize=False, add_generation_prompt=True)
    #     for p in prompt]
    # inputs = vlm_tokenizer(formatted, padding=True,
    #                         max_length=512, truncation=True,
    #                         return_tensors="pt").to(device)
    # vlm_out = vlm_model(input_ids=inputs.input_ids,
    #                      attention_mask=inputs.attention_mask,
    #                      output_hidden_states=True)
    # prompt_embeds = vlm_out.hidden_states[-1]
    # 去掉padding部分...

    # Mock (示例):
    vlm_hidden_dim = 2560
    txt_seq_len = 64
    prompt_embeds = torch.randn(batch_size,
                                txt_seq_len,
                                vlm_hidden_dim,
                                device=device,
                                dtype=torch.bfloat16)

    if do_cfg:
        # 负面prompt编码
        neg_embeds = torch.randn(batch_size,
                                 txt_seq_len,
                                 vlm_hidden_dim,
                                 device=device,
                                 dtype=torch.bfloat16)
        # 拼接: [neg, pos]
        prompt_embeds = torch.cat([neg_embeds, prompt_embeds], dim=0)

    # ========== 2. 准备latent空间参数 ==========
    vae_downsample = 8
    vae_pack_factor = 2
    vae_z_planes = 32
    latent_channels = vae_z_planes * vae_pack_factor * vae_pack_factor
    latent_h = height // vae_downsample // vae_pack_factor
    latent_w = width // vae_downsample // vae_pack_factor
    num_img_tokens = latent_h * latent_w

    # ========== 3. 初始化纯噪声 ==========
    latents = torch.randn(batch_size,
                          num_img_tokens,
                          latent_channels,
                          device=device,
                          dtype=torch.float32,
                          generator=generator)

    # ========== 4. 构建位置ID ==========
    img_ids = create_image_ids(latent_h, latent_w, device)
    img_ids = img_ids.unsqueeze(0).expand(batch_size, -1, -1)
    txt_ids = create_txt_ids(txt_seq_len, device)
    txt_ids = txt_ids.unsqueeze(0).expand(batch_size, -1, -1)

    if do_cfg:
        img_ids = torch.cat([img_ids, img_ids], dim=0)
        txt_ids = torch.cat([txt_ids, txt_ids], dim=0)

    # ========== 5. 设置scheduler ==========
    scheduler = FlowMatchEulerScheduler(num_train_timesteps=1000)
    mu = compute_shift(num_img_tokens)
    scheduler.set_timesteps(num_inference_steps, device=device, mu=mu)

    # ========== 6. 去噪循环 ==========
    model.eval()
    for i, t in enumerate(scheduler.timesteps):
        timestep = t / 1000.0  # 归一化到[0,1]

        if do_cfg:
            latent_input = torch.cat([latents, latents], dim=0)
            t_input = timestep.expand(batch_size * 2)
        else:
            latent_input = latents
            t_input = timestep.expand(batch_size)

        # MMDIT前向
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            v_pred = model(
                x=latent_input.to(torch.bfloat16),
                x_ids=img_ids,
                timesteps=t_input,
                ctx=prompt_embeds,
                ctx_ids=txt_ids,
            )

        # CFG
        if do_cfg:
            v_uncond, v_cond = v_pred.chunk(2)
            v_pred = v_uncond + guidance_scale * (v_cond - v_uncond)

        # Euler step
        latents = scheduler.step(v_pred, t, latents)[0]

    # ========== 7. VAE解码 ==========
    # latents: [B, h*w, 128] → [B, 128, h, w]
    latents_2d = rearrange(latents,
                           "b (h w) c -> b c h w",
                           h=latent_h,
                           w=latent_w)

    # 实际使用时:
    # images = vae.decode(latents_2d)
    # images = (images + 1) / 2  # [-1,1] → [0,1]
    # images = images.clamp(0, 1)

    # Mock VAE decode:
    images = torch.randn(batch_size, 3, height, width, device=device)
    images = images.clamp(0, 1)

    # ========== 8. 转为PIL Image ==========
    images_np = images.cpu().float().permute(0, 2, 3, 1).numpy()
    images_np = (images_np * 255).round().astype(np.uint8)
    pil_images = [Image.fromarray(img) for img in images_np]

    return pil_images


if __name__ == "__main__":
    print("Text-to-Image Inference Pipeline")
    print("Usage: inference_t2i(model, vae, vlm, tokenizer, prompt)")
