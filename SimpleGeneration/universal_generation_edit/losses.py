import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'MSELoss',
    'SigmaAwareClippedMSELoss',
]


class MSELoss(nn.Module):

    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, model_pred, target):
        model_pred = model_pred.float()
        target = target.float()

        mse_loss = F.mse_loss(model_pred, target, reduction='mean')

        loss_dict = {
            'mse_loss': mse_loss,
        }

        return loss_dict


# ---------------------------------------------------------------------------
#  SigmaAwareClippedMSELoss recommended parameters for each training sampler
#  (from scheduler.py):
#
#  ┌───────────────────────────────────────────┬──────────────────┬───────────┐
#  │ Training Sampler                          │ weighting_scheme │ threshold │
#  ├───────────────────────────────────────────┼──────────────────┼───────────┤
#  │ FlowMatchingUniformTimestepSampler        │ 'cosmap'         │   50.0    │
#  │ FlowMatchingLogitNormalTimestepSampler    │ 'none'           │   50.0    │
#  │ FlowMatchingResolutionShiftTimestepSampler│ 'cosmap'         │   50.0    │
#  └───────────────────────────────────────────┴──────────────────┴───────────┘
#
#  Rationale:
#  - Uniform sampler treats all σ equally; 'cosmap' adds mid-σ emphasis
#    (hardest region where signal and noise are balanced) to compensate.
#  - LogitNormal sampler already concentrates density at mid-σ; adding
#    'cosmap' would over-emphasise the middle and starve the endpoints,
#    so 'none' keeps the logit-normal density as the sole weighting.
#  - ResolutionShift sampler adapts σ range per resolution but is still
#    ~uniform on its shifted grid; 'cosmap' adds mid-σ emphasis just
#    like the uniform case (shift handles resolution, cosmap handles
#    difficulty).
#  - 'sigma_sqrt' (σ⁻²) is NOT recommended for velocity prediction:
#    it diverges as σ→0, causing numerical instability regardless of
#    the sampler.
#  - threshold=50.0 is universally beneficial: normal |pred−target|
#    in latent space is ≪50, so this only clips true outliers and
#    prevents gradient explosions.
# ---------------------------------------------------------------------------


class SigmaAwareClippedMSELoss(nn.Module):

    def __init__(self, weighting_scheme='none', threshold=50.0):
        super(SigmaAwareClippedMSELoss, self).__init__()
        assert weighting_scheme in [
            'none', 'sigma_sqrt', 'cosmap'
        ], f"weighting_scheme must be 'none', 'sigma_sqrt' or 'cosmap'"
        self.weighting_scheme = weighting_scheme
        self.threshold = threshold

    def forward(self, model_pred, target, sigmas):
        model_pred = model_pred.float()
        target = target.float()
        sigmas = sigmas.float()

        loss = F.mse_loss(model_pred, target, reduction='none')

        mask = ((model_pred - target).abs() <= self.threshold).float()
        loss = loss * mask

        if self.weighting_scheme == 'none':
            weighting = torch.ones_like(sigmas)
        elif self.weighting_scheme == 'sigma_sqrt':
            weighting = (sigmas**-2.0).float()
        elif self.weighting_scheme == 'cosmap':
            weighting = 2.0 / (math.pi *
                               (1.0 - 2.0 * sigmas + 2.0 * sigmas**2))

        while weighting.ndim < loss.ndim:
            weighting = weighting.unsqueeze(-1)

        loss = loss * weighting
        loss = loss.mean()

        loss_dict = {
            'sigma_aware_clipped_mse_loss': loss,
        }

        return loss_dict


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

    B, L, C = 4, 256, 128
    model_pred = torch.randn(B, L, C)
    target = torch.randn(B, L, C)
    # random timesteps in (0, 1)
    sigmas = torch.rand(B)
    print('input shapes:', model_pred.shape, target.shape, sigmas.shape)

    # 1. MSE
    loss1 = MSELoss()
    out1 = loss1(model_pred, target)
    print('1111', out1)

    # 2. Sigma-aware clipped MSE (none weighting, clipping only)
    loss2 = SigmaAwareClippedMSELoss(weighting_scheme='none', threshold=50.0)
    out2 = loss2(model_pred, target, sigmas=sigmas)
    print('2222', out2)

    # 3. Sigma-aware clipped MSE (sigma_sqrt weighting)
    loss3 = SigmaAwareClippedMSELoss(weighting_scheme='sigma_sqrt',
                                     threshold=50.0)
    out3 = loss3(model_pred, target, sigmas=sigmas)
    print('3333', out3)

    # 4. Sigma-aware clipped MSE (cosmap weighting)
    loss4 = SigmaAwareClippedMSELoss(weighting_scheme='cosmap', threshold=50.0)
    out4 = loss4(model_pred, target, sigmas=sigmas)
    print('4444', out4)
