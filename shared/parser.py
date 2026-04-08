"""BiSeNet face parser (frozen)."""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

BISENET_REPO = '/home/jungbug/coding/capstone/face-parsing.PyTorch'


class FaceParser(nn.Module):
    def __init__(self, weight_path, regions, device):
        super().__init__()
        self.regions = regions

        import importlib.util
        resnet_spec = importlib.util.spec_from_file_location(
            "bisenet_resnet", f"{BISENET_REPO}/resnet.py")
        resnet_mod = importlib.util.module_from_spec(resnet_spec)
        sys.modules['resnet'] = resnet_mod
        resnet_spec.loader.exec_module(resnet_mod)

        model_spec = importlib.util.spec_from_file_location(
            "bisenet_model", f"{BISENET_REPO}/model.py")
        model_mod = importlib.util.module_from_spec(model_spec)
        model_spec.loader.exec_module(model_mod)

        self.net = model_mod.BiSeNet(n_classes=19)
        self.net.load_state_dict(torch.load(weight_path, map_location=device))
        self.net.to(device).eval()
        for p in self.net.parameters():
            p.requires_grad = False

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x):
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        x_norm = (x * 0.5 + 0.5 - mean) / std

        h, w = x.shape[2], x.shape[3]
        x_512 = F.interpolate(x_norm, size=(512, 512),
                               mode='bilinear', align_corners=False)
        out = self.net(x_512)[0]
        label = out.argmax(dim=1, keepdim=True)
        label = F.interpolate(label.float(), size=(h, w), mode='nearest').long()

        masks = {}
        for name, label_ids in self.regions.items():
            mask = torch.zeros_like(label, dtype=torch.float32)
            for lid in label_ids:
                mask += (label == lid).float()
            masks[name] = mask.clamp(0, 1)
        return masks
