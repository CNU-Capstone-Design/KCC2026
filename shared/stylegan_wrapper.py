"""
StyleGAN2 Generator / Discriminator 래퍼.

두 포맷 지원:
  - rosinality .pt  : StyleGAN2Generator (legacy)
  - NVIDIA ADA .pkl : StyleGAN2GeneratorADA (현재 사용)
"""

import sys
import torch
import torch.nn as nn


ADA_REPO = None  # init_ada_repo()로 설정


def init_ada_repo(repo_path):
    global ADA_REPO
    ADA_REPO = repo_path
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


class StyleGAN2GeneratorADA(nn.Module):
    """NVIDIA StyleGAN2-ADA .pkl 로드 래퍼. rosinality 래퍼와 동일 인터페이스."""

    def __init__(self, ckpt_path, ada_repo_path, n_mean_latent=4096):
        super().__init__()
        init_ada_repo(ada_repo_path)

        import pickle
        with open(ckpt_path, 'rb') as f:
            data = pickle.load(f)

        self.g = data['G_ema'].float()

        # mean W 계산 (mapping network 평균)
        with torch.no_grad():
            z = torch.randn(n_mean_latent, 512)
            ws = self.g.mapping(z, None)   # [N, num_ws, 512]
            mean_w = ws[:, 0, :].mean(0, keepdim=True)  # [1, 512]
        self.register_buffer('mean_w', mean_w)

        # 완전 동결
        for p in self.g.parameters():
            p.requires_grad = False

    def forward(self, w_plus):
        """
        Args:
            w_plus: [B, n_styles, 512]
        Returns:
            img: [B, 3, H, W]  range [-1, 1]
        """
        img = self.g.synthesis(w_plus, noise_mode='const')
        return img


class StyleGAN2DiscriminatorADA(nn.Module):
    """NVIDIA StyleGAN2-ADA .pkl 에서 Discriminator 로드."""

    def __init__(self, ckpt_path, ada_repo_path):
        super().__init__()
        init_ada_repo(ada_repo_path)

        import pickle
        with open(ckpt_path, 'rb') as f:
            data = pickle.load(f)

        self.d = data['D'].float()

    def forward(self, x):
        return self.d(x, None)


# legacy rosinality 래퍼 (하위 호환)
class StyleGAN2Generator(nn.Module):
    def __init__(self, ckpt_path, size, repo_path,
                 channel_multiplier=1, finetune_layers=0,
                 n_mean_latent=4096):
        super().__init__()
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        from model import Generator

        self.g = Generator(size=size, style_dim=512, n_mlp=8,
                           channel_multiplier=channel_multiplier)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state = ckpt.get('g_ema', ckpt.get('g', ckpt))
        self.g.load_state_dict(state, strict=False)

        with torch.no_grad():
            mean_w = self.g.mean_latent(n_mean_latent)
        self.register_buffer('mean_w', mean_w)

        for p in self.g.parameters():
            p.requires_grad = False
        if finetune_layers > 0:
            trainable = list(self.g.convs)[-finetune_layers:] + \
                        list(self.g.to_rgbs)[-finetune_layers:]
            for module in trainable:
                for p in module.parameters():
                    p.requires_grad = True

    def forward(self, w_plus):
        img, _ = self.g([w_plus], input_is_latent=True, randomize_noise=False)
        return img


class StyleGAN2Discriminator(nn.Module):
    def __init__(self, size, repo_path, channel_multiplier=1):
        super().__init__()
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        from model import Discriminator
        self.d = Discriminator(size=size, channel_multiplier=channel_multiplier)

    def forward(self, x):
        return self.d(x)
