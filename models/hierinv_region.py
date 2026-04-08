"""
HierInv-Region v2: 계층적 이중 경로 얼굴 GAN 역변환 모델.

W+ = W_avg + global_delta + Σ(region_delta_k)
              ↑                    ↑
        GlobalHierMapper      RegionHierMapper × 9
        (재구성 전담)           (편집 전담, 작은 delta)

편집 (swap):
    base의 global_delta 유지 + 지정 region_delta만 donor로 교체
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import ResNetEncoder
from parser import FaceParser
from stylegan_wrapper import StyleGAN2GeneratorADA


# W+ 레이어 → encoder feature 레벨 매핑 (256px, 14 layers)
LAYER_TO_FEAT = {
    0: 3, 1: 3, 2: 3, 3: 3,    # f3 (512ch, 8×8)   → 전체 구조
    4: 2, 5: 2, 6: 2, 7: 2,    # f2 (256ch, 16×16) → 부위 위치
    8: 1, 9: 1, 10: 1,          # f1 (128ch, 32×32) → 중간 텍스처
    11: 0, 12: 0, 13: 0,        # f0 (64ch,  64×64) → 세밀한 텍스처
}


class GlobalHierMapper(nn.Module):
    """전체 feature → global W+ delta (재구성 전담)."""

    def __init__(self, encoder_channels, n_styles=14, style_dim=512):
        super().__init__()
        self.level_proj = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(ch, 512),
                nn.LeakyReLU(0.2, inplace=True),
            )
            for ch in encoder_channels
        ])
        self.layer_heads = nn.ModuleList([
            nn.Linear(512, style_dim) for _ in range(n_styles)
        ])
        for head in self.layer_heads:
            nn.init.normal_(head.weight, std=0.01)
            nn.init.zeros_(head.bias)

    def forward(self, features):
        level_feats = [proj(f) for proj, f in zip(self.level_proj, features)]
        deltas = [self.layer_heads[i](level_feats[LAYER_TO_FEAT[i]])
                  for i in range(len(self.layer_heads))]
        return torch.stack(deltas, dim=1)   # [B, n_styles, 512]


class RegionHierMapper(nn.Module):
    """마스킹 feature → region W+ delta (편집 전담, 작은 값 유지)."""

    def __init__(self, encoder_channels, n_styles=14, style_dim=512, hidden=128):
        super().__init__()
        self.level_proj = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(ch, hidden),
                nn.LeakyReLU(0.2, inplace=True),
            )
            for ch in encoder_channels
        ])
        self.layer_heads = nn.ModuleList([
            nn.Linear(hidden, style_dim) for _ in range(n_styles)
        ])
        for head in self.layer_heads:
            nn.init.normal_(head.weight, std=0.001)
            nn.init.zeros_(head.bias)

    def forward(self, masked_features):
        level_feats = [proj(f) for proj, f in zip(self.level_proj, masked_features)]
        deltas = [self.layer_heads[i](level_feats[LAYER_TO_FEAT[i]])
                  for i in range(len(self.layer_heads))]
        return torch.stack(deltas, dim=1)   # [B, n_styles, 512]


class HierInvRegionModel(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.cfg          = cfg
        self.device       = device
        self.region_names = list(cfg.REGIONS.keys())

        # Frozen
        self.parser   = FaceParser(cfg.bisenet_weights, cfg.REGIONS, device)
        self.stylegan = StyleGAN2GeneratorADA(
            ckpt_path=cfg.stylegan2_ckpt,
            ada_repo_path=cfg.stylegan2_ada_repo,
        )

        # Trainable
        self.encoder = ResNetEncoder(pretrained=True)
        self.global_mapper = GlobalHierMapper(
            encoder_channels=cfg.encoder_channels,
            n_styles=cfg.n_styles,
            style_dim=cfg.style_dim,
        )
        self.region_mappers = nn.ModuleDict({
            name: RegionHierMapper(
                encoder_channels=cfg.encoder_channels,
                n_styles=cfg.n_styles,
                style_dim=cfg.style_dim,
            )
            for name in self.region_names
        })

    def _mask_features(self, features, mask):
        return [f * F.interpolate(mask, size=f.shape[2:], mode='nearest')
                for f in features]

    def encode(self, x):
        features = self.encoder(x)
        masks    = self.parser(x)
        global_delta  = self.global_mapper(features)
        region_deltas = {}
        region_sum    = None
        for name in self.region_names:
            masked = self._mask_features(features, masks[name])
            rd = self.region_mappers[name](masked)
            region_deltas[name] = rd
            region_sum = rd if region_sum is None else region_sum + rd
        total_delta = global_delta + region_sum
        return global_delta, region_deltas, total_delta

    def decode(self, total_delta):
        B      = total_delta.shape[0]
        mean_w = self.stylegan.mean_w.unsqueeze(1).expand(B, self.cfg.n_styles, -1)
        return self.stylegan(mean_w + total_delta)

    def forward(self, x):
        global_delta, region_deltas, total_delta = self.encode(x)
        return self.decode(total_delta), global_delta, region_deltas

    @torch.no_grad()
    def swap(self, x_base, x_donor, swap_regions):
        """
        base의 swap_regions를 donor로 교체.
        global_delta는 base 유지 → 얼굴형·정체성 보존.
        """
        g_base, rd_base, _ = self.encode(x_base)
        _, rd_donor, _     = self.encode(x_donor)
        region_sum = None
        for name in self.region_names:
            rd = rd_donor[name] if name in swap_regions else rd_base[name]
            region_sum = rd if region_sum is None else region_sum + rd
        return self.decode(g_base + region_sum)
