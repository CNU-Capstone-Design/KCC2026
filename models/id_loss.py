"""
ArcFace Identity Loss.
encoder4editing의 IR-SE50 기반 구현.
얼굴 identity 보존을 위해 생성 이미지와 원본의 face feature cosine similarity 최대화.
"""

import torch
import torch.nn as nn
import sys
import os

from .irse.model_irse import Backbone


class IDLoss(nn.Module):
    def __init__(self, ir_se50_path):
        super().__init__()
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(ir_se50_path, map_location='cpu'))
        self.facenet.eval()
        for p in self.facenet.parameters():
            p.requires_grad = False
        self.face_pool = nn.AdaptiveAvgPool2d((112, 112))

    def _extract(self, x):
        # x: [-1,1], [B,3,256,256] → crop 얼굴 중심 → 112x112 → ArcFace feature
        x = x[:, :, 35:221, 32:220]
        x = self.face_pool(x)
        return self.facenet(x)

    def forward(self, y_hat, x):
        """
        y_hat: 생성 이미지 [-1,1]
        x:     원본 이미지 [-1,1]
        return: 1 - cosine_similarity (낮을수록 좋음)
        """
        f_real = self._extract(x).detach()
        f_fake = self._extract(y_hat)
        # cosine similarity per sample, 평균
        loss = 1 - torch.cosine_similarity(f_fake, f_real, dim=1).mean()
        return loss
