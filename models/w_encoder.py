"""W-Encoder Baseline: 단일 W 벡터로 매핑하는 GAN inversion."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from encoder import ResNetEncoder
from stylegan_wrapper import StyleGAN2GeneratorADA


class WMapper(nn.Module):
    def __init__(self, encoder_channels, n_styles, style_dim=512):
        super().__init__()
        self.n_styles = n_styles
        total_ch = sum(encoder_channels)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(total_ch, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, style_dim),
        )

    def forward(self, features):
        pooled = [self.gap(f).flatten(1) for f in features]
        w = self.fc(torch.cat(pooled, dim=1))
        return w.unsqueeze(1).expand(-1, self.n_styles, -1)


class WEncoderModel(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.stylegan = StyleGAN2GeneratorADA(cfg.stylegan2_ckpt, cfg.stylegan2_ada_repo)
        self.encoder  = ResNetEncoder(pretrained=True)
        self.mapper   = WMapper(cfg.encoder_channels, cfg.n_styles, cfg.style_dim)

    def encode(self, x):
        return self.mapper(self.encoder(x))

    def decode(self, w_plus):
        return self.stylegan(w_plus)

    def forward(self, x):
        w_plus = self.encode(x)
        return self.decode(w_plus), w_plus
