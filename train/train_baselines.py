"""
PSP / E4E / W-Encoder Baseline 학습 스크립트.

실행:
    python train/train_baselines.py --model psp
    python train/train_baselines.py --model e4e
    python train/train_baselines.py --model wenc
    python train/train_baselines.py --model e4e --resume checkpoints/e4e/ckpt_epoch0050.pth
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


def load_model(name, cfg, device):
    if name == 'psp':
        from models.psp import PSPModel
        return PSPModel(cfg, device).to(device)
    elif name == 'e4e':
        from models.e4e import E4EModel
        return E4EModel(cfg, device).to(device)
    elif name == 'wenc':
        from models.w_encoder import WEncoderModel
        return WEncoderModel(cfg, device).to(device)
    else:
        raise ValueError(f'Unknown model: {name}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',  type=str, required=True, choices=['psp', 'e4e', 'wenc'])
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    cfg    = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_dir   = os.path.join(_ROOT, 'checkpoints', args.model)
    sample_dir = os.path.join(_ROOT, 'samples', args.model)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    print(f'Model:   {args.model.upper()}')
    print(f'Device:  {device}')
    print(f'Data:    {cfg.data_root}')

    loader = get_dataloader(cfg.data_root, cfg.img_size, cfg.batch_size,
                            cfg.num_workers, train=True)
    print(f'Dataset: {len(loader.dataset)} images\n')

    model = load_model(args.model, cfg, device)
    params = list(model.encoder.parameters()) + list(model.mapper.parameters())
    opt    = Adam(params, lr=cfg.lr_G, betas=(cfg.beta1, cfg.beta2))
    print(f'Trainable params: {sum(p.numel() for p in params):,}\n')

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.encoder.load_state_dict(ckpt['encoder'])
        model.mapper.load_state_dict(ckpt['mapper'])
        opt.load_state_dict(ckpt['opt'])
        start_epoch = ckpt['epoch'] + 1
        print(f'Resumed from epoch {ckpt["epoch"]}')

    recon_loss = ReconLoss(cfg.lambda_l2, cfg.lambda_lpips).to(device)
    id_loss    = IDLoss(cfg.ir_se50_weights).to(device) if args.model == 'e4e' else None

    for epoch in range(start_epoch, cfg.n_epochs):
        model.train()
        for step, real in enumerate(loader):
            real = real.to(device)
            out  = model(real)
            gen  = out[0]
            delta = out[1] if len(out) > 1 else None

            loss, l2, lpips = recon_loss(gen, real)

            if args.model == 'e4e' and id_loss is not None:
                loss += id_loss(gen, real) * cfg.lambda_id
            if args.model == 'e4e' and delta is not None:
                loss += w_norm_loss(delta) * cfg.lambda_w_norm

            opt.zero_grad(); loss.backward(); opt.step()

            if step % 100 == 0:
                print(f'[{epoch}/{cfg.n_epochs-1}] step {step}/{len(loader)} '
                      f'| loss: {loss.item():.4f}  l2: {l2.item():.4f}  '
                      f'lpips: {lpips.item():.4f}', flush=True)

        ckpt_path = os.path.join(ckpt_dir, f'ckpt_epoch{epoch:04d}.pth')
        torch.save({
            'epoch': epoch,
            'encoder': model.encoder.state_dict(),
            'mapper': model.mapper.state_dict(),
            'opt': opt.state_dict(),
        }, ckpt_path)
        print(f'  → saved {ckpt_path}', flush=True)

        model.eval()
        with torch.no_grad():
            s = real[:4]
            gen_s = model(s)[0]
            grid  = vutils.make_grid(torch.cat([s, gen_s], dim=0),
                                     nrow=4, normalize=True, value_range=(-1, 1))
            vutils.save_image(grid, os.path.join(sample_dir, f'epoch{epoch:04d}.png'))


if __name__ == '__main__':
    main()
