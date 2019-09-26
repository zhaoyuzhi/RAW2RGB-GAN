import torch
import torch.nn as nn
from network_module import *

# ----------------------------------------
#         Initialize the networks
# ----------------------------------------
def weights_init(net, init_type = 'normal', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    In our paper, we choose the default setting: zero mean Gaussian distribution with a standard deviation of 0.02
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

# ----------------------------------------
#                Generator
# ----------------------------------------
# Generator contains 2 Auto-Encoders
class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        # The generator is U shaped
        # Encoder
        self.E1 = Conv2dLayer(opt.in_channels, opt.start_channels, 7, 1, 3, pad_type = opt.pad, norm = 'none')
        self.E2 = Conv2dLayer(opt.start_channels, opt.start_channels * 2, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E3 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E4 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 8, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        # Bottle neck
        self.BottleNeck = nn.Sequential(
            ResConv2dLayer(opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            ResConv2dLayer(opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            ResConv2dLayer(opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            ResConv2dLayer(opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        )
        # Decoder
        self.D1 = TransposeConv2dLayer(opt.start_channels * 8, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D2 = TransposeConv2dLayer(opt.start_channels * 8, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D3 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels * 1, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D4 = Conv2dLayer(opt.start_channels * 2, opt.out_channels, 7, 1, 3, pad_type = opt.pad, norm = 'none', activation = 'tanh')
        # Sal Decoder
        self.SalDecoder = SalGenerator(opt)

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        E1 = self.E1(x)                                         # out: batch * 64 * 224 * 224
        E2 = self.E2(E1)                                        # out: batch * 128 * 112 * 112
        E3 = self.E3(E2)                                        # out: batch * 256 * 56 * 56
        E4 = self.E4(E3)                                        # out: batch * 512 * 28 * 28
        # Bottle neck
        E4 = self.BottleNeck(E4)                                # out: batch * 512 * 28 * 28
        # Decode the center code
        D1 = self.D1(E4)                                        # out: batch * 256 * 56 * 56
        D1 = torch.cat((D1, E3), 1)                             # out: batch * 512 * 56 * 56
        D2 = self.D2(D1)                                        # out: batch * 128 * 112 * 112
        D2 = torch.cat((D2, E2), 1)                             # out: batch * 256 * 112 * 112
        D3 = self.D3(D2)                                        # out: batch * 64 * 224 * 224
        D3 = torch.cat((D3, E1), 1)                             # out: batch * 128 * 224 * 224
        x = self.D4(D3)                                         # out: batch * out_channel * 256 * 256
        # Sal Decode
        sal = self.SalDecoder(D1, D2, D3)

        return x, sal

class SalGenerator(nn.Module):
    def __init__(self, opt):
        super(SalGenerator, self).__init__()
        # Decoder 1
        self.D1 = nn.Sequential(
            TransposeConv2dLayer(opt.start_channels * 8, opt.latent_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2),
            Conv2dLayer(opt.latent_channels, opt.latent_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            TransposeConv2dLayer(opt.latent_channels, opt.latent_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2),
            Conv2dLayer(opt.latent_channels, opt.latent_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        )
        self.D2 = nn.Sequential(
            TransposeConv2dLayer(opt.start_channels * 4, opt.latent_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2),
            Conv2dLayer(opt.latent_channels, opt.latent_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        )
        self.D3 = Conv2dLayer(opt.start_channels * 2, opt.latent_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        # Decoder 2
        self.D4 = nn.Sequential(
            Conv2dLayer(opt.latent_channels * 3, opt.latent_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            Conv2dLayer(opt.latent_channels, opt.sal_channels, 7, 1, 3, pad_type = opt.pad, norm = 'none', activation = 'sigmoid')
        )

    def forward(self, D1, D2, D3):
        D1 = self.D1(D1)
        D2 = self.D2(D2)
        D3 = self.D3(D3)
        D4 = torch.cat((D1, D2, D3), 1)
        D4 = self.D4(D4)

        return D4

# ----------------------------------------
#               Discriminator
# ----------------------------------------
# PatchDiscriminator70: PatchGAN discriminator
# Usage: Initialize PatchGAN in training code like:
#        discriminator = PatchDiscriminator70()
# This is a kind of PatchGAN. Patch is implied in the output. This is 70 * 70 PatchGAN
class PatchDiscriminator70(nn.Module):
    def __init__(self, opt):
        super(PatchDiscriminator70, self).__init__()
        # Down sampling
        self.block1 = nn.Sequential(
            Conv2dLayer(opt.in_channels + opt.out_channels, opt.start_channels, 1, 1, 0, pad_type = opt.pad, norm = 'none'),
            Conv2dLayer(opt.start_channels, opt.start_channels, 7, 1, 3, pad_type = opt.pad, norm = opt.norm)
        )
        self.block2 = nn.Sequential(
            Conv2dLayer(opt.start_channels , opt.start_channels * 2, 4, 2, 1, pad_type = opt.pad, norm = opt.norm),
            Conv2dLayer(opt.start_channels * 2, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        )
        self.block3 = nn.Sequential(
            Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, norm = opt.norm),
            Conv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        )
        self.block4 = nn.Sequential(
            Conv2dLayer(opt.start_channels * 4, opt.start_channels * 8, 4, 2, 1, pad_type = opt.pad, norm = opt.norm),
            Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        )
        # Final output, implemention of 70 * 70 PatchGAN
        self.final1 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 4, 4, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.final2 = Conv2dLayer(opt.start_channels * 4, 1, 4, 1, 1, pad_type = opt.pad, norm = 'none', activation = 'none')

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        # img_A: input; img_B: output
        x = torch.cat((img_A, img_B), 1)                        # out: batch * 7 * 256 * 256
        block1 = self.block1(x)                                 # out: batch * 64 * 256 * 256
        block2 = self.block2(block1)                            # out: batch * 128 * 128 * 128
        block3 = self.block3(block2)                            # out: batch * 256 * 64 * 64
        x = self.block4(block3)                                 # out: batch * 512 * 32 * 32
        x = self.final1(x)                                      # out: batch * 512 * 31 * 31
        x = self.final2(x)                                      # out: batch * 1 * 30 * 30
        return block1, block2, block3, x

# ----------------------------------------
#             Perceptual Net
# ----------------------------------------
# For perceptual loss
class PerceptualNet(nn.Module):
    def __init__(self):
        super(PerceptualNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, 3, 1, 1)
        )

    def forward(self, x):
        x = self.features(x)
        return x
