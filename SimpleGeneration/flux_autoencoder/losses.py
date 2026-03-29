"""
https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/lpips.py
https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/losses/contperceptual.py
"""
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision


class ScalingLayer(nn.Module):

    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer(
            'shift',
            torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer(
            'scale',
            torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, x):
        x = (x - self.shift) / self.scale

        return x


class NetLinLayer(nn.Module):

    def __init__(self, inplanes, planes=1, use_dropout=True):
        super(NetLinLayer, self).__init__()
        layers = [
            nn.Dropout(),
        ] if use_dropout else []
        layers += [
            nn.Conv2d(inplanes,
                      planes,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
        ]
        self.model = nn.Sequential(*layers)


class VGG16(nn.Module):

    def __init__(self):
        super(VGG16, self).__init__()
        # use vgg16 pytorch official weight
        # https://download.pytorch.org/models/vgg16-397923af.pth
        vgg_model = torchvision.models.vgg16(weights=None)
        vgg_model.load_state_dict(
            torch.load(
                '/root/autodl-tmp/pretrained_models/vgg16_pytorch_official_weights/vgg16-397923af.pth',
                map_location='cpu',
                weights_only=True))
        vgg_pretrained_features = vgg_model.features

        # vgg_pretrained_features = torchvision.models.vgg16(
        #     weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h

        vgg_outputs = namedtuple(
            "VggOutputs",
            ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3,
                          h_relu5_3)

        return out


class LPIPS(nn.Module):

    def __init__(self, lpips_pretrained_path):
        super(LPIPS, self).__init__()
        self.scaling_layer = ScalingLayer()
        # vg16 features
        self.vgg_planes = [64, 128, 256, 512, 512]
        self.net = VGG16()
        self.lin0 = NetLinLayer(self.vgg_planes[0], use_dropout=True)
        self.lin1 = NetLinLayer(self.vgg_planes[1], use_dropout=True)
        self.lin2 = NetLinLayer(self.vgg_planes[2], use_dropout=True)
        self.lin3 = NetLinLayer(self.vgg_planes[3], use_dropout=True)
        self.lin4 = NetLinLayer(self.vgg_planes[4], use_dropout=True)

        self.load_state_dict(torch.load(lpips_pretrained_path,
                                        map_location=torch.device("cpu")),
                             strict=False)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        in0_input = self.scaling_layer(input)
        in1_input = self.scaling_layer(target)
        outs0 = self.net(in0_input)
        outs1 = self.net(in1_input)

        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.vgg_planes)):
            feats0[kk] = self.normalize_tensor(outs0[kk])
            feats1[kk] = self.normalize_tensor(outs1[kk])

            diffs[kk] = (feats0[kk] - feats1[kk])**2

        res = [(lins[kk].model(diffs[kk])).mean([2, 3], keepdim=True)
               for kk in range(len(self.vgg_planes))]

        val = res[0]
        for l in range(1, len(self.vgg_planes)):
            val += res[l]

        return val

    def normalize_tensor(self, x, eps=1e-10):
        norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))

        return x / (norm_factor + eps)


class ReconstructionL1Loss(nn.Module):

    def __init__(self):
        super(ReconstructionL1Loss, self).__init__()
        self.loss = nn.L1Loss(reduction='mean')

    def forward(self, inputs, preds):
        inputs = inputs.float()
        preds = preds.float()

        loss = self.loss(inputs, preds)

        return loss


class ReconstructionL2Loss(nn.Module):

    def __init__(self):
        super(ReconstructionL2Loss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, preds):
        inputs = inputs.float()
        preds = preds.float()

        loss = self.loss(inputs, preds)

        return loss


class PerceptualLoss(nn.Module):

    def __init__(self, lpips_pretrained_path=''):
        super(PerceptualLoss, self).__init__()
        self.loss = LPIPS(lpips_pretrained_path=lpips_pretrained_path).eval()

    def forward(self, inputs, preds):
        inputs = inputs.float()
        preds = preds.float()

        loss = self.loss(inputs, preds)
        loss = loss.mean()

        return loss


class GeneratorAdversarialLoss(nn.Module):

    def __init__(self):
        super(GeneratorAdversarialLoss, self).__init__()
        pass

    def forward(self, fake_preds):
        fake_preds = fake_preds.float()

        loss = -fake_preds.mean()

        return loss


class DiscriminatorAdversarialLoss(nn.Module):

    def __init__(self):
        super(DiscriminatorAdversarialLoss, self).__init__()
        pass

    def forward(self, real_preds, fake_preds):
        real_preds = real_preds.float()
        fake_preds = fake_preds.float()

        real_loss = torch.mean(F.relu(1. - real_preds))
        fake_loss = torch.mean(F.relu(1. + fake_preds))
        loss = 0.5 * (real_loss + fake_loss)

        return loss


class FLUX1AELoss(nn.Module):

    def __init__(self,
                 discriminator_model,
                 lpips_pretrained_path,
                 kl_weight=0.000001,
                 reconstruction_weight=1.0,
                 perceptual_weight=1.0,
                 discriminator_weight=0.5):
        super(FLUX1AELoss, self).__init__()
        self.kl_weight = kl_weight
        self.reconstruction_weight = reconstruction_weight
        self.perceptual_weight = perceptual_weight
        self.discriminator_weight = discriminator_weight

        self.discriminator_model = discriminator_model

        self.reconstruction_loss = ReconstructionL1Loss()
        self.perceptual_loss = PerceptualLoss(
            lpips_pretrained_path=lpips_pretrained_path)
        self.generator_adversarial_loss = GeneratorAdversarialLoss()
        self.discriminator_adversarial_loss = DiscriminatorAdversarialLoss()

    def calculate_adaptive_weight(self, nll_loss, generator_adversarial_loss,
                                  last_layer):
        nll_grads = torch.autograd.grad(nll_loss,
                                        last_layer,
                                        retain_graph=True)[0]
        g_grads = torch.autograd.grad(generator_adversarial_loss,
                                      last_layer,
                                      retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0., 1e4).detach()
        d_weight = d_weight * self.discriminator_weight

        return d_weight

    def forward(self,
                images,
                reconstruction_images,
                logvar=None,
                kl_out=None,
                loss_type='generator_loss',
                last_layer=None):
        assert loss_type in ['generator_loss', 'discriminator_loss']

        images = images.float()
        reconstruction_images = reconstruction_images.float()

        if loss_type == 'generator_loss':
            reconstruction_loss = self.reconstruction_loss(
                images, reconstruction_images)
            reconstruction_loss = self.reconstruction_weight * reconstruction_loss

            perceptual_loss = self.perceptual_loss(images,
                                                   reconstruction_images)
            perceptual_loss = self.perceptual_weight * perceptual_loss

            nll_loss = reconstruction_loss + perceptual_loss
            nll_loss = nll_loss / torch.exp(logvar) + logvar

            kl_loss = torch.mean(kl_out)
            kl_loss = self.kl_weight * kl_loss

            generator_fake = self.discriminator_model(reconstruction_images)
            generator_adversarial_loss = self.generator_adversarial_loss(
                generator_fake)
            generator_adversarial_weight = self.calculate_adaptive_weight(
                nll_loss, generator_adversarial_loss, last_layer=last_layer)
            generator_adversarial_loss = generator_adversarial_weight * generator_adversarial_loss

            loss_dict = {
                'nll_loss': nll_loss,
                'kl_loss': kl_loss,
                'generator_adversarial_loss': generator_adversarial_loss,
                'generator_fake': generator_fake,
            }

            return loss_dict

        elif loss_type == 'discriminator_loss':
            discriminator_real = self.discriminator_model(images.detach())
            discriminator_fake = self.discriminator_model(
                reconstruction_images.detach())

            discriminator_adversarial_loss = self.discriminator_adversarial_loss(
                discriminator_real, discriminator_fake)
            discriminator_adversarial_loss = self.discriminator_weight * discriminator_adversarial_loss

            loss_dict = {
                'discriminator_adversarial_loss':
                discriminator_adversarial_loss,
                'discriminator_real': discriminator_real,
                'discriminator_fake': discriminator_fake,
            }

            return loss_dict
