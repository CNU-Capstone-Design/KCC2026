"""
HierInv-Region v2 학습 스크립트.

실행:
    python train/train_hierinv.py
    python train/train_hierinv.py --resume checkpoints/hierinv/ckpt_epoch0050.pth
"""

import os, sys, argparse, warnings
warnings.filterwarnings('ignore')

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'shared'))

import torch
import torchvision.utils as vutils
from torch.optim import Adam

from config import Config
from dataset import get_dataloader
from losses import ReconLoss, w_norm_loss
from id_loss import IDLoss
from models.hierinv_region import HierInvRegionModel

CKPT_DIR   = os.path.join(_ROOT, 'checkpoints', 'hierinv')
SAMPLE_DIR = os.path.join(_ROOT, 'samples', 'hierinv')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    cfg    = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    print(f'Device:  {device}')
    print(f'Data:    {cfg.data_root}')
    print(f'Epochs:  {cfg.n_epochs}')

    loader = get_dataloader(cfg.data_root, cfg.img_size, cfg.batch_size,
                            cfg.num_workers, train=True)
    print(f'Dataset: {len(loader.dataset)} images\n')

    model = HierInvRegionModel(cfg, device).to(device)

    params  = list(model.encoder.parameters())
    params += list(model.global_mapper.parameters())
    for m in model.region_mappers.values():
        params += list(m.parameters())
    opt = Adam(params, lr=cfg.lr_G, betas=(cfg.beta1, cfg.beta2))
    print(f'Trainable params: {sum(p.numel() for p in params):,}\n')

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.encoder.load_state_dict(ckpt['encoder'])
        model.global_mapper.load_state_dict(ckpt['global_mapper'])
        model.region_mappers.load_state_dict(ckpt['region_mappers'])
        opt.load_state_dict(ckpt['opt'])
        start_epoch = ckpt['epoch'] + 1
        print(f'Resumed from epoch {ckpt["epoch"]}')

    recon_loss = ReconLoss(cfg.lambda_l2, cfg.lambda_lpips).to(device)
    id_loss    = IDLoss(cfg.ir_se50_weights).to(device)

    for epoch in range(start_epoch, cfg.n_epochs):
        model.train()
        for step, real in enumerate(loader):
            real = real.to(device)
            gen, global_delta, region_deltas = model(real)

            l_recon, l2, lpips = recon_loss(gen, real)
            l_id     = id_loss(gen, real) * cfg.lambda_id
            l_wnorm  = w_norm_loss(global_delta) * cfg.lambda_w_norm
            l_rnorm  = sum(w_norm_loss(rd) for rd in region_deltas.values()) \
                       * cfg.lambda_w_norm * 0.1
            loss = l_recon + l_id + l_wnorm + l_rnorm

            opt.zero_grad(); loss.backward(); opt.step()

            if step % 100 == 0:
                print(f'[{epoch}/{cfg.n_epochs-1}] step {step}/{len(loader)} '
                      f'| l2: {l2.item():.4f}  lpips: {lpips.item():.4f}  '
                      f'id: {l_id.item():.4f}  wnorm: {l_wnorm.item():.4f}  '
                      f'rnorm: {l_rnorm.item():.4f}', flush=True)

        ckpt_path = os.path.join(CKPT_DIR, f'ckpt_epoch{epoch:04d}.pth')
        torch.save({
            'epoch': epoch,
            'encoder': model.encoder.state_dict(),
            'global_mapper': model.global_mapper.state_dict(),
            'region_mappers': model.region_mappers.state_dict(),
            'opt': opt.state_dict(),
        }, ckpt_path)
        print(f'  → saved {ckpt_path}', flush=True)

        model.eval()
        with torch.no_grad():
            s = real[:4]
            gen_s, _, _ = model(s)
            grid = vutils.make_grid(torch.cat([s, gen_s], dim=0),
                                    nrow=4, normalize=True, value_range=(-1, 1))
            vutils.save_image(grid, os.path.join(SAMPLE_DIR, f'epoch{epoch:04d}.png'))


if __name__ == '__main__':
    main()
