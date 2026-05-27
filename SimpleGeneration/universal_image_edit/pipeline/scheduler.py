"""
Flow Match Euler Discrete Scheduler

推断依据：
- Z-Image: FlowMatchEulerDiscreteScheduler, sigma-based, dynamic shifting
- JoyAI: FlowMatchDiscreteScheduler, sd3_time_shift
- Flux2: 自定义 get_schedule + generalized_time_snr_shift

结论：采用Z-Image/JoyAI的标准Euler scheduler，支持dynamic shifting。
简洁实现，不依赖diffusers库。
"""
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor


class FlowMatchEulerScheduler:
    """Flow Matching Euler ODE Solver"""

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 1.0):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.timesteps = None
        self.sigmas = None
        self._step_index = None

    def time_shift(self, mu: float, sigma: float, t: Tensor) -> Tensor:
        """Dynamic time shifting (SD3/Flux2 style)"""
        return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0)**sigma)

    def set_timesteps(self,
                      num_inference_steps: int,
                      device: Union[str, torch.device] = "cpu",
                      mu: Optional[float] = None):
        """设置推理时间步"""
        self.num_inference_steps = num_inference_steps

        sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1)

        if mu is not None:
            # Dynamic shifting
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            # Static shifting
            sigmas = (self.shift * sigmas / (1 + (self.shift - 1) * sigmas))

        self.sigmas = sigmas.to(device)
        self.timesteps = (sigmas[:-1] * self.num_train_timesteps).to(
            dtype=torch.float32, device=device)
        self._step_index = None

    def step(self, model_output: Tensor, timestep: Union[float, Tensor],
             sample: Tensor) -> Tuple[Tensor]:
        """Euler step: x_{t-1} = x_t + (sigma_{t+1} - sigma_t) * v_pred"""
        if self._step_index is None:
            if isinstance(timestep, Tensor):
                timestep = timestep.to(self.timesteps.device)
            indices = (self.timesteps == timestep).nonzero()
            pos = 1 if len(indices) > 1 else 0
            self._step_index = indices[pos].item()

        sigma = self.sigmas[self._step_index]
        sigma_next = self.sigmas[self._step_index + 1]
        dt = sigma_next - sigma

        sample = sample.to(torch.float32)
        prev_sample = sample + dt * model_output.to(torch.float32)

        self._step_index += 1

        return (prev_sample.to(model_output.dtype), )

    def get_schedule(self, num_steps: int, image_seq_len: int) -> List[float]:
        """获取时间步schedule (Flux2风格)"""
        from ..losses import compute_shift
        mu = compute_shift(image_seq_len)
        timesteps = torch.linspace(1, 0, num_steps + 1)
        timesteps = self.time_shift(mu, 1.0, timesteps)
        return timesteps.tolist()
