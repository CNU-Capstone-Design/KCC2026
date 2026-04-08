"""
SF-GAN Loss Functions.

E4E 기반 손실 구성:
  L_total = λ_l2 * L2 + λ_lpips * LPIPS + λ_id * ID + λ_w * W-norm + λ_adv * Adv

- L2:     pixel-level reconstruction
- LPIPS:  perceptual similarity (VGG16)
- ID:     ArcFace identity preservation  ← E4E 핵심
- W-norm: ||W+ - W_avg||² regularization ← E4E 핵심 (editability 보존)
- Adv:    non-saturating GAN loss
"""

import torch
import torch.nn as nn
import torchvision.models as tvm


# ── Perceptual (LPIPS-style, VGG16) ─────────────────────────────────────────

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1)
        self.feat = nn.Sequential(*list(vgg.features)[:16]).eval()
        for p in self.feat.parameters():
            p.requires_grad = False

    def forward(self, x, y):
        # x, y: [-1,1] → 정규화
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
        x = ((x + 1) / 2 - mean) / std
        y = ((y + 1) / 2 - mean) / std
        return torch.mean(torch.abs(self.feat(x) - self.feat(y)))


# ── Reconstruction Loss ──────────────────────────────────────────────────────

class ReconLoss(nn.Module):
    def __init__(self, lambda_l2=1.0, lambda_lpips=0.8):
        super().__init__()
        self.lambda_l2    = lambda_l2
        self.lambda_lpips = lambda_lpips
        self.percep       = PerceptualLoss()

    def forward(self, pred, target):
        l2   = torch.mean((pred - target) ** 2)
        lpip = self.percep(pred, target)
        return self.lambda_l2 * l2 + self.lambda_lpips * lpip, l2, lpip


# ── W-norm Regularization ────────────────────────────────────────────────────

def w_norm_loss(total_w_plus):
    """
    total_w_plus: [B, n_styles, 512] — W_avg 대비 offset의 norm
    ||total_w_plus||² 최소화 → W_avg 가까이 유지 → editability 보존 (E4E)
    """
    return torch.mean(total_w_plus.pow(2))


# ── GAN Losses ───────────────────────────────────────────────────────────────

class StyleGANLoss(nn.Module):
    def g_loss(self, fake_pred):
        return torch.nn.functional.softplus(-fake_pred).mean()

    def d_loss(self, real_pred, fake_pred):
        return (torch.nn.functional.softplus(fake_pred) +
                torch.nn.functional.softplus(-real_pred)).mean()


def r1_penalty(real_pred, real_img):
    grad, = torch.autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    return grad.pow(2).reshape(grad.shape[0], -1).sum(1).mean()
