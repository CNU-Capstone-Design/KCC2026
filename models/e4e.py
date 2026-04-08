"""E4E Baseline: encoder4editing 스타일 GAN inversion."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from encoder import ResNetEncoder
from stylegan_wrapper import StyleGAN2GeneratorADA


class E4EMapper(nn.Module):
    def __init__(self, encoder_channels, n_styles, style_dim=512):
        super().__init__()
        total_ch = sum(encoder_channels)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_shared = nn.Sequential(
            nn.Linear(total_ch, 512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.delta_heads = nn.ModuleList([
            nn.Linear(512, style_dim) for _ in range(n_styles)
        ])
        for h in self.delta_heads:
            nn.init.normal_(h.weight, std=0.01)
            nn.init.zeros_(h.bias)

    def forward(self, features):
        pooled = [self.gap(f).flatten(1) for f in features]
        x = self.fc_shared(torch.cat(pooled, dim=1))
        return torch.stack([h(x) for h in self.delta_heads], dim=1)


class E4EModel(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.stylegan = StyleGAN2GeneratorADA(cfg.stylegan2_ckpt, cfg.stylegan2_ada_repo)
        self.encoder  = ResNetEncoder(pretrained=True)
        self.mapper   = E4EMapper(cfg.encoder_channels, cfg.n_styles, cfg.style_dim)

    def encode(self, x):
        delta_w = self.mapper(self.encoder(x))
        B = x.shape[0]
        mean_w = self.stylegan.mean_w.unsqueeze(1).expand(B, self.cfg.n_styles, -1)
        return mean_w + delta_w, delta_w

    def decode(self, w_plus):
        return self.stylegan(w_plus)

    def forward(self, x):
        w_plus, delta_w = self.encode(x)
        return self.decode(w_plus), delta_w
