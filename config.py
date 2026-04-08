"""
HierInv-Region v2 공통 설정.
모든 학습/평가 스크립트에서 공유.
"""

import os

_ROOT = os.path.dirname(os.path.abspath(__file__))


class Config:
    # ── 데이터 ──────────────────────────────────────────────
    data_root    = os.path.join(_ROOT, 'data', 'celebahq_aligned')   # CelebA-HQ aligned PNG
    img_size     = 256
    batch_size   = 8
    num_workers  = 4

    # ── Region 정의 (BiSeNet 19-class → 9 region) ───────────
    REGIONS = {
        'skin':       [1],
        'brow':       [2, 3],
        'eye':        [4, 5],
        'nose':       [10],
        'mouth':      [11, 12, 13],
        'hair':       [17],
        'ear':        [7, 8],
        'neck':       [14],
        'background': [0, 6, 9, 15, 16, 18],
    }

    # ── Encoder ─────────────────────────────────────────────
    encoder_channels = [64, 128, 256, 512]   # ResNet34 layer1~4

    # ── StyleGAN2 (NVIDIA ADA .pkl) ─────────────────────────
    # pretrained/ 폴더에 배치 필요 (README 참고)
    stylegan2_ckpt     = os.path.join(_ROOT, 'pretrained', 'stylegan2-celebahq-256x256.pkl')
    stylegan2_ada_repo = os.path.join(_ROOT, 'stylegan2-ada-pytorch')
    stylegan2_size     = 256
    n_styles           = 14      # 256px: 14 W 레이어
    style_dim          = 512

    # ── Pretrained weights ──────────────────────────────────
    bisenet_weights = os.path.join(_ROOT, 'pretrained', 'bisenet.pth')
    ir_se50_weights = os.path.join(_ROOT, 'pretrained', 'model_ir_se50.pth')

    # ── 손실 가중치 ─────────────────────────────────────────
    lambda_l2      = 1.0
    lambda_lpips   = 0.8
    lambda_id      = 0.1
    lambda_w_norm  = 0.005

    # ── 학습 ────────────────────────────────────────────────
    lr_G     = 2e-4
    beta1    = 0.0
    beta2    = 0.99
    n_epochs = 100
