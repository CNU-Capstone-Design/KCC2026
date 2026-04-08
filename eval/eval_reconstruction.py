"""
재구성 품질 평가 (LPIPS / PSNR / SSIM).
HierInv-Region v2, PSP, E4E, W-Encoder 전체 비교.

실행:
    python eval/eval_reconstruction.py
    python eval/eval_reconstruction.py --test_dir data/test_images --n 200
"""

import os, sys, json, argparse, warnings
warnings.filterwarnings('ignore')

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'shared'))

import numpy as np
import torch
import torchvision.transforms as T
from pathlib import Path
from PIL import Image

from config import Config

to_tensor = T.Compose([
    T.Resize((256, 256)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3),
])
denorm = T.Normalize([-1]*3, [2]*3)


def latest_ckpt(d):
    ckpts = sorted(Path(d).glob('ckpt_epoch*.pth'))
    if not ckpts: raise FileNotFoundError(f'No checkpoints in {d}')
    return ckpts[-1]


@torch.no_grad()
def run_inference(model, test_paths, out_dir, device, batch_size=8):
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(list(out_dir.glob('*.png'))) >= len(test_paths):
        print(f'  [skip] already exists'); return
    model.eval()
    to_pil = T.ToPILImage()
    for i in range(0, len(test_paths), batch_size):
        batch = test_paths[i:i+batch_size]
        imgs  = torch.stack([to_tensor(Image.open(p).convert('RGB')) for p in batch]).to(device)
        out   = model(imgs)
        gen   = out[0] if isinstance(out, tuple) else out
        for p, g in zip(batch, gen):
            to_pil(denorm(g.cpu()).clamp(0, 1)).save(out_dir / p.name)
        print(f'  {i+len(batch)}/{len(test_paths)}', end='\r')
    print()


def compute_metrics(gt_paths, pred_dir, lpips_fn, device):
    from skimage.metrics import structural_similarity as ssim_fn
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    norm = T.Compose([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    lpips_s, psnr_s, ssim_s = [], [], []
    for p in gt_paths:
        pp = pred_dir / p.name
        if not pp.exists(): continue
        gt   = Image.open(p).convert('RGB').resize((256, 256))
        pred = Image.open(pp).convert('RGB').resize((256, 256))
        gt_np, pred_np = np.array(gt), np.array(pred)
        psnr_s.append(psnr_fn(gt_np, pred_np, data_range=255))
        ssim_s.append(ssim_fn(gt_np, pred_np, data_range=255, channel_axis=2))
        with torch.no_grad():
            lpips_s.append(lpips_fn(
                norm(gt).unsqueeze(0).to(device),
                norm(pred).unsqueeze(0).to(device)
            ).item())
    return {
        'LPIPS': round(float(np.mean(lpips_s)), 4),
        'PSNR':  round(float(np.mean(psnr_s)),  2),
        'SSIM':  round(float(np.mean(ssim_s)),  4),
        'n':     len(psnr_s),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str,
                        default=os.path.join(_ROOT, 'data', 'test_images'))
    parser.add_argument('--n', type=int, default=200, help='평가 이미지 수')
    args = parser.parse_args()

    import lpips as lpips_lib
    cfg    = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    test_paths = sorted(Path(args.test_dir).glob('*.png'))[:args.n]
    print(f'Test images: {len(test_paths)}\n')

    lpips_fn = lpips_lib.LPIPS(net='vgg').to(device)
    results  = []
    res_root = Path(_ROOT) / 'eval_results' / 'reconstruction'
    res_root.mkdir(parents=True, exist_ok=True)

    # ── HierInv-Region v2 ────────────────────────────────────────────────────
    print('[HierInv-Region v2]')
    from models.hierinv_region import HierInvRegionModel
    model = HierInvRegionModel(cfg, device).to(device)
    ckpt  = torch.load(latest_ckpt(os.path.join(_ROOT, 'checkpoints', 'hierinv')), map_location=device)
    model.encoder.load_state_dict(ckpt['encoder'])
    model.global_mapper.load_state_dict(ckpt['global_mapper'])
    model.region_mappers.load_state_dict(ckpt['region_mappers'])
    run_inference(model, test_paths, res_root / 'hierinv', device)
    del model; torch.cuda.empty_cache()
    r = compute_metrics(test_paths, res_root / 'hierinv', lpips_fn, device)
    r['model'] = 'HierInv-Region v2'; results.append(r)
    print(f"  LPIPS={r['LPIPS']}  PSNR={r['PSNR']}  SSIM={r['SSIM']}\n")

    # ── PSP ──────────────────────────────────────────────────────────────────
    print('[PSP]')
    from models.psp import PSPModel
    model = PSPModel(cfg, device).to(device)
    ckpt  = torch.load(latest_ckpt(os.path.join(_ROOT, 'checkpoints', 'psp')), map_location=device)
    model.encoder.load_state_dict(ckpt['encoder'])
    model.mapper.load_state_dict(ckpt['mapper'])
    run_inference(model, test_paths, res_root / 'psp', device)
    del model; torch.cuda.empty_cache()
    r = compute_metrics(test_paths, res_root / 'psp', lpips_fn, device)
    r['model'] = 'PSP'; results.append(r)
    print(f"  LPIPS={r['LPIPS']}  PSNR={r['PSNR']}  SSIM={r['SSIM']}\n")

    # ── E4E ──────────────────────────────────────────────────────────────────
    print('[E4E]')
    from models.e4e import E4EModel
    model = E4EModel(cfg, device).to(device)
    ckpt  = torch.load(latest_ckpt(os.path.join(_ROOT, 'checkpoints', 'e4e')), map_location=device)
    model.encoder.load_state_dict(ckpt['encoder'])
    model.mapper.load_state_dict(ckpt['mapper'])
    run_inference(model, test_paths, res_root / 'e4e', device)
    del model; torch.cuda.empty_cache()
    r = compute_metrics(test_paths, res_root / 'e4e', lpips_fn, device)
    r['model'] = 'E4E'; results.append(r)
    print(f"  LPIPS={r['LPIPS']}  PSNR={r['PSNR']}  SSIM={r['SSIM']}\n")

    # ── W-Encoder ────────────────────────────────────────────────────────────
    print('[W-Encoder]')
    from models.w_encoder import WEncoderModel
    model = WEncoderModel(cfg, device).to(device)
    ckpt  = torch.load(latest_ckpt(os.path.join(_ROOT, 'checkpoints', 'wenc')), map_location=device)
    model.encoder.load_state_dict(ckpt['encoder'])
    model.mapper.load_state_dict(ckpt['mapper'])
    run_inference(model, test_paths, res_root / 'wenc', device)
    del model; torch.cuda.empty_cache()
    r = compute_metrics(test_paths, res_root / 'wenc', lpips_fn, device)
    r['model'] = 'W-Encoder'; results.append(r)
    print(f"  LPIPS={r['LPIPS']}  PSNR={r['PSNR']}  SSIM={r['SSIM']}\n")

    # ── 결과 출력 ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"{'Model':<22} {'LPIPS↓':>8} {'PSNR↑':>8} {'SSIM↑':>8}  N")
    print('-'*60)
    for r in sorted(results, key=lambda x: x['LPIPS']):
        marker = ' ◀' if r['model'] == 'HierInv-Region v2' else ''
        print(f"{r['model']:<22} {r['LPIPS']:>8.4f} {r['PSNR']:>8.2f} {r['SSIM']:>8.4f}  {r['n']}{marker}")
    print('='*60)

    out = Path(_ROOT) / 'eval_results' / 'metrics_reconstruction.json'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved → {out}')


if __name__ == '__main__':
    main()
