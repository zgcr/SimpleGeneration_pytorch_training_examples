"""
https://github.com/CompVis/taming-transformers/blob/master/taming/modules/discriminator/model.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NLayerDiscriminator(nn.Module):
    """
    Defines a PatchGAN discriminator as in Pix2Pix
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, inplanes=3, layer_nums=3, planes=64):
        super(NLayerDiscriminator, self).__init__()
        sequence = [
            nn.Conv2d(inplanes,
                      planes,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        planes_mult = 1
        planes_mult_prev = 1
        for n in range(1, layer_nums):
            planes_mult_prev = planes_mult
            planes_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(planes * planes_mult_prev,
                          planes * planes_mult,
                          kernel_size=4,
                          stride=2,
                          padding=1,
                          bias=False),
                nn.BatchNorm2d(planes * planes_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        planes_mult_prev = planes_mult
        planes_mult = min(2**layer_nums, 8)
        sequence += [
            nn.Conv2d(planes * planes_mult_prev,
                      planes * planes_mult,
                      kernel_size=4,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(planes * planes_mult),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        sequence += [
            nn.Conv2d(planes * planes_mult,
                      1,
                      kernel_size=4,
                      stride=1,
                      padding=1,
                      bias=True)
        ]
        self.main = nn.Sequential(*sequence)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        x = self.main(x)

        return x
