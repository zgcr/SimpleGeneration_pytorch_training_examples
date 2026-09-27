import math

import torch

__all__ = [
    'FlowMatchingUniformTimestepSampler',
    'FlowMatchingLogitNormalTimestepSampler',
    'FlowMatchingResolutionShiftTimestepSampler',
    'FlowMatchingEulerScheduler',
    'FlowMatchingResolutionAdaptiveEulerScheduler',
]

# ---------------------------------------------------------------------------
#  Training-inference matched pairs:
#    FlowMatchingUniformTimestepSampler         ↔ FlowMatchingEulerScheduler(shift=1.0)
#    FlowMatchingLogitNormalTimestepSampler     ↔ FlowMatchingEulerScheduler(shift>=1.0)
#    FlowMatchingResolutionShiftTimestepSampler ↔ FlowMatchingResolutionAdaptiveEulerScheduler
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
#  image_seq_len = (H / 16) * (W / 16), where the 16x comes from
#  VAE 8x downsample + 2x2 patchify. For square images:
#  ┌────────────┬──────────────────┐
#  │ Resolution │ image_seq_len    │
#  ├────────────┼──────────────────┤
#  │  256x256   │    256           │
#  │  512x512   │   1024           │
#  │ 1024x1024  │   4096           │
#  │ 2048x2048  │  16384           │
#  └────────────┴──────────────────┘
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
#  COMMON API
#
#  Training timestep samplers:
#      sample(batch_size)                -> [batch_size] timesteps
#      sample(batch_size, image_seq_len) -> [batch_size] timesteps
#          (FlowMatchingResolutionShiftTimestepSampler only)
#
#  Inference schedulers:
#      get_schedule(device=None, dtype=torch.float32)
#      get_schedule(image_seq_len, device=None, dtype=torch.float32)
#          (FlowMatchingResolutionAdaptiveEulerScheduler only)
#          -> [num_steps + 1] sigma schedule, from 1.0 (noise) to 0.0 (data)
#      step(x_t, v_pred, sigmas, step_index)
#          -> x_next
#
#  Only the two resolution-adaptive classes consume image_seq_len, which feeds
#  the dynamic-shift mu; all the other classes are resolution agnostic. The
#  step() signature is shared by both schedulers, so the caller drives the
#  denoising loop without any isinstance branch.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
#  SIGMA CONVENTION AND THE TIME-SNR SHIFT
#
#  Rectified flow with x_t = (1 - t) * x0 + t * noise, so sigma_t == t:
#  sigma = 1 is pure noise and sigma = 0 is clean data. Every schedule and
#  every sampled timestep in this file lives on that sigma axis.
#
#  The generalized time-SNR shift used by the resolution-adaptive classes,
#      sigma' = exp(mu) / (exp(mu) + (1 / sigma - 1) ** snr_sigma)
#  is, for the default snr_sigma = 1.0, exactly a shift in logit space:
#      logit(sigma') = mu + logit(sigma)
#  so exp(mu) is the effective `shift` value of the simpler rational form
#      sigma' = shift * sigma / (1 + (shift - 1) * sigma)
#  used by FlowMatchingEulerScheduler. Both push probability mass / step
#  budget towards the high-noise regime when mu > 0 (shift > 1).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
#  RESOLUTION-DEPENDENT mu, AND THE TERMINAL ANCHOR
#
#  mu comes from compute_dynamic_shift_mu(image_seq_len), which is linear in
#  image_seq_len and takes NO step budget. The training sampler and the
#  inference scheduler carry an identical copy of it, so the sigma distribution
#  a resolution is trained on is by construction the one it is sampled with.
#
#  shift_terminal is orthogonal to mu: mu decides the sigma DISTRIBUTION, the
#  terminal anchor only rescales WHERE the finitely many evaluation points of
#  one schedule land. It therefore never reintroduces a train / inference
#  mismatch, and it is what keeps the last Euler step small at low step counts.
# ---------------------------------------------------------------------------


class FlowMatchingUniformTimestepSampler:
    """Flow Matching training timestep sampler — uniform.

    Samples timesteps uniformly from U(eps, 1-eps) for the rectified-flow
    training objective. The small eps margin avoids the boundary instability
    at t=0 (pure data) and t=1 (pure noise).

    Matched inference scheduler: FlowMatchingEulerScheduler(shift=1.0)
    (standard uniform schedule, no time-shift).

    Advantage scenario:
        All timesteps are treated equally, which is suboptimal for training 
        efficiency but provides the most unbiased gradient signal.
    """

    def __init__(self, eps=1e-4):
        self.eps = eps

    def sample(self, batch_size):
        timesteps = torch.rand(batch_size)
        timesteps = timesteps * (1.0 - 2 * self.eps) + self.eps

        return timesteps


class FlowMatchingLogitNormalTimestepSampler:
    """Flow Matching training timestep sampler — logit-normal + shift.

    First draws from a normal distribution N(mu, sigma^2), then maps through
    sigmoid to obtain t in (0, 1). This produces a bell-shaped density that
    concentrates samples around the middle timesteps (the hardest region where
    signal and noise are balanced), which accelerates convergence compared to
    uniform sampling.

    An optional timestep shift is applied after the logit-normal sampling:
        t' = shift * t / (1 + (shift - 1) * t)
    When shift > 1 the distribution is pushed towards higher t (noisier),
    allocating more training budget to the high-noise regime where global
    structure is learned. This is especially beneficial for high-resolution
    images whose spatial structure is more complex.

    Matched inference scheduler:
        FlowMatchingEulerScheduler with the same or larger shift.
        In practice, training with shift=1.0 and inferring with shift=3.0
        is the most common and validated configuration.

    Advantage scenario:
        Best for fixed-resolution or progressive-resolution training.
        The logit-normal density focuses training on the most informative
        timestep region, leading to faster convergence and higher final quality
        compared to uniform sampling. The optional shift further improves
        high-resolution generation by giving more training signal to the
        global-structure phase.
    """

    def __init__(self, mu=0.0, sigma=1.0, shift=1.0, eps=1e-4):
        self.mu = mu
        self.sigma = sigma
        self.shift = shift
        self.eps = eps

    def sample(self, batch_size):
        normal_samples = (self.mu + self.sigma * torch.randn(batch_size))
        timesteps = torch.sigmoid(normal_samples)

        if self.shift != 1.0:
            timesteps = (self.shift * timesteps /
                         (1.0 + (self.shift - 1.0) * timesteps))

        timesteps = timesteps.clamp(self.eps, 1.0 - self.eps)

        return timesteps


class FlowMatchingResolutionShiftTimestepSampler:
    """Flow Matching training timestep sampler — resolution-adaptive shift.

    Draws timesteps uniformly from (0, 1), then applies a resolution-dependent
    generalized time-SNR shift so that higher-resolution images automatically
    receive a stronger shift towards the high-noise regime.

    The shift parameter mu comes from compute_dynamic_shift_mu, which is linear
    in image_seq_len and takes no step budget at all, so mu is a function of
    image_seq_len ONLY and the inference scheduler evaluates the very same
    function. That shared definition is the point: a training run has no step
    budget to feed a step-dependent formula, so any such formula forces the two
    sides to agree on a convention instead of on a function.

    The calibration is Qwen-Image-2.1's, re-anchored for native 2K generation,
    and it keeps the effective shift exp(mu) inside [1.649, 3.717] across the
    256 -> 2048 stages WITHOUT a clamp: 1.649 at 256x256, 1.714 at 512x512,
    2.001 at 1024x1024 and 3.717 at 2048x2048. Staying clamp-free is what keeps
    the adaptivity alive at high resolution — a steeper curve has to be clamped,
    and a clamp makes every resolution above its knee share one single shift,
    which is exactly where adapting to the resolution matters most.

    The generalized time-SNR shift transform applied to each sigma is:
        sigma' = exp(mu) / (exp(mu) + (1/sigma - 1)^snr_sigma)
    where snr_sigma defaults to 1.0 (matching FLUX.2).

    Matched inference scheduler:
        FlowMatchingResolutionAdaptiveEulerScheduler with the same
        snr_sigma value.

    Advantage scenario:
        Best for multi-resolution training where each stage (or even each
        mini-batch) may contain images of different sizes. The adaptive shift
        automatically adjusts the timestep distribution for every resolution,
        eliminating the need to manually tune shift per stage. This is the
        optimal strategy when the model is trained across multiple resolution
        stages and must support all resolutions at inference.
    """

    def __init__(self, num_train_timesteps=1000, snr_sigma=1.0, eps=1e-4):
        self.num_train_timesteps = num_train_timesteps
        self.snr_sigma = snr_sigma
        self.eps = eps

    def compute_dynamic_shift_mu(self,
                                 image_seq_len,
                                 base_seq_len=256,
                                 max_seq_len=8192,
                                 base_shift=0.5,
                                 max_shift=0.9):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = m * image_seq_len + b
        mu = float(mu)

        return mu

    def sample(self, batch_size, image_seq_len):
        mu = self.compute_dynamic_shift_mu(image_seq_len)

        base_sigmas = torch.rand(batch_size).clamp(self.eps, 1.0 - self.eps)
        exp_mu = math.exp(mu)
        timesteps = exp_mu / (exp_mu +
                              (1.0 / base_sigmas - 1.0)**self.snr_sigma)

        timesteps = timesteps.clamp(self.eps, 1.0 - self.eps)

        return timesteps


class FlowMatchingEulerScheduler:
    """Flow Matching inference scheduler with Euler ODE solver.

    Generates a deterministic timestep schedule from t=1 (pure noise) to t=0
    (clean data) and provides an Euler single-step update for the probability
    flow ODE  dx/dt = v_theta(x_t, t).

    Supports the time-shift that redistributes the timesteps so that
    more denoising steps are allocated to the high-noise regime:
        t' = shift * t / (1 + (shift - 1) * t)
    When shift=1.0 this reduces to a standard uniform schedule.

    shift_terminal optionally stretches the schedule so that its LAST
    evaluation point lands on that sigma instead of on shift(1 / num_steps),
    which pins the size of the final Euler step no matter how many steps are
    requested. It defaults to 0.0 (disabled) here, keeping this the plain
    baseline schedule that matches the uniform sampler exactly.

    Matched training sampler:
        FlowMatchingUniformTimestepSampler (when shift=1.0).
        FlowMatchingLogitNormalTimestepSampler (when shift>=1.0).
        Although training may use a different shift value (or shift=1.0),
        using shift > 1 at inference is a common and validated practice
        (Bagel/Lance use shift=1.0 for training and shift=3.0 for inference).
        The mild train-inference mismatch is beneficial because high-noise
        steps determine global structure and benefit from finer discretisation.

    Advantage scenario:
        Simplest and most robust inference scheduler. Works with any training
        sampler. The shift parameter provides a single knob to trade off
        structure quality (shift > 1) vs. detail quality (shift = 1).
    """

    def __init__(self, num_steps=50, shift=1.0, shift_terminal=0.0):
        self.num_steps = num_steps
        self.shift = shift
        self.shift_terminal = shift_terminal

    def stretch_schedule_to_terminal(self, timesteps):
        if self.num_steps < 2:

            return timesteps

        one_minus_sigmas = 1.0 - timesteps[:-1]
        scale_factor = one_minus_sigmas[-1] / (1.0 - self.shift_terminal)
        timesteps[:-1] = 1.0 - one_minus_sigmas / scale_factor

        return timesteps

    def get_schedule(self, device=None, dtype=torch.float32):
        timesteps = torch.linspace(1.0,
                                   0.0,
                                   self.num_steps + 1,
                                   device=device,
                                   dtype=dtype)

        if self.shift != 1.0:
            timesteps = (self.shift * timesteps /
                         (1.0 + (self.shift - 1.0) * timesteps))

        if self.shift_terminal > 0.0:
            timesteps = self.stretch_schedule_to_terminal(timesteps)

        timesteps = timesteps.clamp(0.0, 1.0)

        return timesteps

    def step(self, x_t, v_pred, sigmas, step_index):
        sigma_s = sigmas[step_index]
        sigma_t = sigmas[step_index + 1]

        x_next = x_t.float() + v_pred.float() * (sigma_t - sigma_s).float()

        return x_next


class FlowMatchingResolutionAdaptiveEulerScheduler:
    """Flow Matching inference scheduler with Euler ODE solver — resolution-adaptive shift.

    Computes a resolution-dependent shift parameter mu with
    compute_dynamic_shift_mu, then applies a generalized time-SNR shift to the
    uniform timestep grid:
        t' = exp(mu) / (exp(mu) + (1/t - 1)^snr_sigma)

    When snr_sigma=1.0 (default), this matches the FLUX.2 generalized
    time-SNR shift exactly. mu is linear in image_seq_len and takes no step
    budget, and the method is an identical copy of the training sampler's, so
    every resolution is sampled on the very sigma distribution it was trained
    on — the two sides share a function rather than a convention.

    shift_terminal then stretches the schedule so that its LAST evaluation
    point lands on that sigma. It defaults to Qwen-Image-2.1's 0.02, and it is
    what decouples the size of the final Euler step from num_steps: at 2048
    with 8 steps the last step covers dt = -0.02 instead of the -0.3468 the
    unstretched grid would leave for it. The stretch is orthogonal to mu, since
    it only moves the finitely many evaluation points of one schedule and never
    the underlying sigma distribution.

    Matched training sampler:
        FlowMatchingResolutionShiftTimestepSampler with the same snr_sigma.

    Advantage scenario:
        Best for multi-resolution inference when the model was trained
        across multiple resolution stages. The adaptive shift automatically
        gives larger images more budget in the high-noise regime without
        requiring manual per-resolution tuning, and the terminal anchor keeps
        quality stable when using fewer inference steps.
    """

    def __init__(self, num_steps=50, snr_sigma=1.0, shift_terminal=0.02):
        self.num_steps = num_steps
        self.snr_sigma = snr_sigma
        self.shift_terminal = shift_terminal

    def compute_dynamic_shift_mu(self,
                                 image_seq_len,
                                 base_seq_len=256,
                                 max_seq_len=8192,
                                 base_shift=0.5,
                                 max_shift=0.9):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = m * image_seq_len + b
        mu = float(mu)

        return mu

    def stretch_schedule_to_terminal(self, timesteps):
        if self.num_steps < 2:

            return timesteps

        one_minus_sigmas = 1.0 - timesteps[:-1]
        scale_factor = one_minus_sigmas[-1] / (1.0 - self.shift_terminal)
        timesteps[:-1] = 1.0 - one_minus_sigmas / scale_factor

        return timesteps

    def get_schedule(self, image_seq_len, device=None, dtype=torch.float32):
        mu = self.compute_dynamic_shift_mu(image_seq_len)
        timesteps = torch.linspace(1.0,
                                   0.0,
                                   self.num_steps + 1,
                                   device=device,
                                   dtype=dtype)

        exp_mu = math.exp(mu)
        nonzero = timesteps > 0
        timesteps[nonzero] = exp_mu / (
            exp_mu + (1.0 / timesteps[nonzero] - 1.0)**self.snr_sigma)

        if self.shift_terminal > 0.0:
            timesteps = self.stretch_schedule_to_terminal(timesteps)

        timesteps = timesteps.clamp(0.0, 1.0)

        return timesteps

    def step(self, x_t, v_pred, sigmas, step_index):
        sigma_s = sigmas[step_index]
        sigma_t = sigmas[step_index + 1]

        x_next = x_t.float() + v_pred.float() * (sigma_t - sigma_s).float()

        return x_next


if __name__ == '__main__':
    import os
    import random
    import numpy as np
    import torch
    seed = 0
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 1. Uniform sampler
    sampler1 = FlowMatchingUniformTimestepSampler(eps=1e-4)
    timesteps1 = sampler1.sample(batch_size=4)
    print('1111', timesteps1)

    # 2. Logit-Normal sampler (with shift=1.0)
    sampler2 = FlowMatchingLogitNormalTimestepSampler(mu=0.0,
                                                      sigma=1.0,
                                                      shift=1.0,
                                                      eps=1e-4)
    timesteps2 = sampler2.sample(batch_size=4)
    print('2222', timesteps2)

    # 3. Logit-Normal sampler (with shift=3.0)
    sampler3 = FlowMatchingLogitNormalTimestepSampler(mu=0.0,
                                                      sigma=1.0,
                                                      shift=3.0,
                                                      eps=1e-4)
    timesteps3 = sampler3.sample(batch_size=4)
    print('3333', timesteps3)

    # 4. Resolution-shift sampler (resolution-dependent dynamic-shift mu)
    sampler4 = FlowMatchingResolutionShiftTimestepSampler()
    timesteps4_256 = sampler4.sample(batch_size=4, image_seq_len=256)
    timesteps4_4096 = sampler4.sample(batch_size=4, image_seq_len=4096)
    print('4444', timesteps4_256)
    print('5555', timesteps4_4096)

    # 5. Euler scheduler (no shift)
    scheduler5 = FlowMatchingEulerScheduler(num_steps=50, shift=1.0)
    schedule5 = scheduler5.get_schedule()
    print('6666', schedule5)

    # 6. Euler scheduler (shift=3.0)
    scheduler6 = FlowMatchingEulerScheduler(num_steps=50, shift=3.0)
    schedule6 = scheduler6.get_schedule()
    print('7777', schedule6)

    # 7. Resolution-adaptive Euler scheduler (dynamic-shift mu + SNR shift)
    scheduler7 = FlowMatchingResolutionAdaptiveEulerScheduler(num_steps=50,
                                                              snr_sigma=1.0)
    schedule7_256 = scheduler7.get_schedule(image_seq_len=256)
    schedule7_4096 = scheduler7.get_schedule(image_seq_len=4096)
    print('8888', schedule7_256)
    print('9999', schedule7_4096)

    # 8. Verify Euler step
    x_t8 = torch.randn(2, 4)
    v_pred8 = torch.randn(2, 4)
    x_next8 = scheduler5.step(x_t8, v_pred8, schedule5, 0)
    expected8 = x_t8 + v_pred8 * (schedule5[1] - schedule5[0])
    print('1010', x_next8.shape,
          bool(torch.allclose(x_next8, expected8, atol=1e-6)))

    # 9. Verify resolution-adaptive Euler step
    x_t9 = torch.randn(2, 4)
    v_pred9 = torch.randn(2, 4)
    x_next9 = scheduler7.step(x_t9, v_pred9, schedule7_4096, 0)
    expected9 = x_t9 + v_pred9 * (schedule7_4096[1] - schedule7_4096[0])
    print('1111', x_next9.shape,
          bool(torch.allclose(x_next9, expected9, atol=1e-6)))

    # 10. Training sigma distribution per resolution stage
    for per_stage_base_resize, per_stage_image_seq_len in [(256, 256),
                                                           (512, 1024),
                                                           (1024, 4096),
                                                           (2048, 16384)]:
        per_stage_timesteps = sampler4.sample(
            batch_size=100000, image_seq_len=per_stage_image_seq_len)
        print(
            f'1212 stage:{per_stage_base_resize}, image_seq_len:{per_stage_image_seq_len}, '
            f'median:{per_stage_timesteps.median():.4f}, '
            f'p(sigma<0.1):{(per_stage_timesteps < 0.1).float().mean():.4f}, '
            f'p(sigma>0.9):{(per_stage_timesteps > 0.9).float().mean():.4f}')

    # 11. Adaptive shift direction: larger image -> more high-noise budget
    for per_image_seq_len in [256, 1024, 4096, 16384]:
        per_schedule = FlowMatchingResolutionAdaptiveEulerScheduler(
            num_steps=10).get_schedule(image_seq_len=per_image_seq_len)
        print(f'1313 image_seq_len:{per_image_seq_len}, '
              f'schedule:{[round(float(x), 3) for x in per_schedule]}')

    # 12. Training and inference read the SAME mu out of the SAME function, so
    # the sigma distribution a resolution is trained on is the one it is
    # sampled with. mu rises monotonically and needs no clamp.
    previous_mu = None
    for per_image_seq_len in [256, 1024, 4096, 16384]:
        train_mu = sampler4.compute_dynamic_shift_mu(per_image_seq_len)
        infer_mu = FlowMatchingResolutionAdaptiveEulerScheduler(
            num_steps=50).compute_dynamic_shift_mu(per_image_seq_len)
        assert train_mu == infer_mu
        # mu must NOT depend on the step budget either.
        assert train_mu == FlowMatchingResolutionAdaptiveEulerScheduler(
            num_steps=8).compute_dynamic_shift_mu(per_image_seq_len)
        if previous_mu is not None:
            assert train_mu > previous_mu
        previous_mu = train_mu
        print(f'1414 image_seq_len:{per_image_seq_len}, '
              f'train_mu:{train_mu:.4f}, infer_mu:{infer_mu:.4f}, '
              f'shift:{math.exp(train_mu):.3f}')

    # 13. shift_terminal pins the LAST evaluation point (and therefore the size
    # of the final Euler step) whatever the step budget, while the appended
    # terminal sigma stays exactly 0.0 so the schedule still ends on clean data.
    for per_num_steps in [8, 20, 50]:
        per_schedule = FlowMatchingResolutionAdaptiveEulerScheduler(
            num_steps=per_num_steps,
            shift_terminal=0.02).get_schedule(image_seq_len=16384)
        per_schedule_without_terminal = (
            FlowMatchingResolutionAdaptiveEulerScheduler(
                num_steps=per_num_steps,
                shift_terminal=0.0).get_schedule(image_seq_len=16384))
        assert abs(float(per_schedule[-2]) - 0.02) < 1e-6
        assert float(per_schedule[-1]) == 0.0
        assert bool((per_schedule[:-1].diff() < 0).all())
        print(
            f'1515 num_steps:{per_num_steps}, '
            f'last_dt:{float(per_schedule[-1] - per_schedule[-2]):+.4f}, '
            f'last_dt_without_terminal:'
            f'{float(per_schedule_without_terminal[-1] - per_schedule_without_terminal[-2]):+.4f}'
        )

    # 14. The 8-step 2048 schedule, which is where the terminal anchor matters
    # most: without it a single Euler step would have to cover the last 35%.
    print('1616', [
        round(float(x), 4)
        for x in FlowMatchingResolutionAdaptiveEulerScheduler(
            num_steps=8).get_schedule(image_seq_len=16384)
    ])
