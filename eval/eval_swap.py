"""
Face Region Swap 편집 품질 평가.

비교:
  HierInv-Region v2  → region swap (눈/코/입 등 특정 부위만 교체)
  PSP / E4E / W-Enc  → W+ interpolation α=0.5 (region swap 기능 없으므로
                        동등한 수준의 편집: base/donor latent 평균)

측정 메트릭:
  RTF  (Region Transfer Fidelity)  : 타깃 부위가 donor와 유사한가          ↓ LPIPS
  BGP  (Background Preservation)   : 비-타깃 부위가 base에서 얼마나 유지됐나  ↓ LPIPS
  IDP  (Identity Preservation)     : 전체 정체성이 base와 얼마나 유지됐나    ↑ ArcFace cos
  DS   (Disentanglement Score)      : 비-타깃 개별 region의 평균 변화량       ↓ LPIPS

실행:
    python eval/eval_swap.py
    python eval/eval_swap.py --n_pairs 50 --test_dir data/test_images
"""

import os, sys, warnings, json, argparse
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
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),
])


def latest_ckpt(d):
    ckpts = sorted(Path(d).glob('ckpt_epoch*.pth'))
    if not ckpts:
        raise FileNotFoundError(f'No checkpoints in {d}')
    return ckpts[-1]


# ── 마스킹 LPIPS ─────────────────────────────────────────────────────────────
def masked_lpips(lpips_fn, img_a, img_b, mask):
    """mask 영역에서만 LPIPS 비교. img: [B,3,H,W] [-1,1], mask: [B,1,H,W] [0,1]"""
    a_m = img_a * mask
    b_m = img_b * mask
    with torch.no_grad():
        return lpips_fn(a_m, b_m).mean().item()


# ── W+ interpolation 편집 래퍼 (PSP / E4E / W-Encoder) ──────────────────────
class WPlusInterpolationEditor:
    """
    PSP / E4E / W-Encoder용 편집 래퍼.
    region swap 기능이 없으므로 W+ interpolation으로 동등한 편집 시뮬레이션.
    W+_result = (1-α) * W+_base + α * W+_donor   (α=0.5 default)
    """
    def __init__(self, model, alpha=0.5):
        self.model = model
        self.alpha = alpha

    @torch.no_grad()
    def swap(self, x_base, x_donor, swap_regions=None):
        out_base  = self.model.encode(x_base)
        out_donor = self.model.encode(x_donor)
        w_base  = out_base[0]  if isinstance(out_base,  tuple) else out_base
        w_donor = out_donor[0] if isinstance(out_donor, tuple) else out_donor
        w_interp = (1 - self.alpha) * w_base + self.alpha * w_donor
        return self.model.decode(w_interp)

    def __getattr__(self, name):
        return getattr(self.model, name)


# ── ArcFace 유사도 추출기 ─────────────────────────────────────────────────────
class ArcFaceExtractor:
    def __init__(self, ir_se50_path, device):
        from id_loss import IDLoss
        self.id_loss = IDLoss(ir_se50_path).to(device)

    @torch.no_grad()
    def similarity(self, img_a, img_b):
        """img_a, img_b: [B,3,256,256] [-1,1] → scalar cos-sim"""
        fa = self.id_loss._extract(img_a)
        fb = self.id_loss._extract(img_b)
        return torch.cosine_similarity(fa, fb, dim=1).mean().item()


# ── 평가 루프 ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_swap(editor, parser_fn, pairs, swap_regions, all_region_names,
                  lpips_fn, arcface, device):
    rtf_list = []
    bgp_list = []
    idp_list = []
    ds_lists = {r: [] for r in all_region_names if r not in swap_regions}

    for base_path, donor_path in pairs:
        x_base  = to_tensor(Image.open(base_path ).convert('RGB')).unsqueeze(0).to(device)
        x_donor = to_tensor(Image.open(donor_path).convert('RGB')).unsqueeze(0).to(device)

        result = editor.swap(x_base, x_donor, swap_regions=swap_regions)

        masks_base  = parser_fn(x_base)
        masks_donor = parser_fn(x_donor)

        swap_mask_base  = sum(masks_base [r] for r in swap_regions).clamp(0, 1)
        swap_mask_donor = sum(masks_donor[r] for r in swap_regions).clamp(0, 1)
        swap_mask       = ((swap_mask_base + swap_mask_donor) / 2).clamp(0, 1)
        non_swap_mask   = (1 - swap_mask_base).clamp(0, 1)

        rtf_list.append(masked_lpips(lpips_fn, result, x_donor, swap_mask))
        bgp_list.append(masked_lpips(lpips_fn, result, x_base,  non_swap_mask))
        idp_list.append(arcface.similarity(result, x_base))

        for rname in ds_lists:
            rm = masks_base[rname]
            if rm.sum() < 50:
                continue
            ds_lists[rname].append(masked_lpips(lpips_fn, result, x_base, rm))

    def safe_mean(lst):
        return round(float(np.mean(lst)), 4) if lst else None

    ds_per_region = {r: safe_mean(v) for r, v in ds_lists.items()}
    ds_mean = safe_mean([v for lst in ds_lists.values() for v in lst])

    return {
        'RTF':           safe_mean(rtf_list),
        'BGP':           safe_mean(bgp_list),
        'IDP':           safe_mean(idp_list),
        'DS':            ds_mean,
        'DS_per_region': ds_per_region,
        'n_pairs':       len(pairs),
    }


# ── 모델 로드 ─────────────────────────────────────────────────────────────────
def load_hierinv(cfg, device):
    from models.hierinv_region import HierInvRegionModel
    model = HierInvRegionModel(cfg, device).to(device)
    ckpt  = torch.load(latest_ckpt(os.path.join(_ROOT, 'checkpoints', 'hierinv')), map_location=device)
    model.encoder.load_state_dict(ckpt['encoder'])
    model.global_mapper.load_state_dict(ckpt['global_mapper'])
    model.region_mappers.load_state_dict(ckpt['region_mappers'])
    model.eval()
    return model


def load_psp(cfg, device):
    from models.psp import PSPModel
    model = PSPModel(cfg, device).to(device)
    ckpt  = torch.load(latest_ckpt(os.path.join(_ROOT, 'checkpoints', 'psp')), map_location=device)
    model.encoder.load_state_dict(ckpt['encoder'])
    model.mapper.load_state_dict(ckpt['mapper'])
    model.eval()
    return WPlusInterpolationEditor(model)


def load_e4e(cfg, device):
    from models.e4e import E4EModel
    model = E4EModel(cfg, device).to(device)
    ckpt  = torch.load(latest_ckpt(os.path.join(_ROOT, 'checkpoints', 'e4e')), map_location=device)
    model.encoder.load_state_dict(ckpt['encoder'])
    model.mapper.load_state_dict(ckpt['mapper'])
    model.eval()
    return WPlusInterpolationEditor(model)


def load_wenc(cfg, device):
    from models.w_encoder import WEncoderModel
    model = WEncoderModel(cfg, device).to(device)
    ckpt  = torch.load(latest_ckpt(os.path.join(_ROOT, 'checkpoints', 'wenc')), map_location=device)
    model.encoder.load_state_dict(ckpt['encoder'])
    model.mapper.load_state_dict(ckpt['mapper'])
    model.eval()
    return WPlusInterpolationEditor(model)


# ── 출력 포매터 ───────────────────────────────────────────────────────────────
def print_table(results_dict, swap_label):
    print(f'\n{"="*72}')
    print(f'  편집 방식: {swap_label}')
    print(f'  (HierInv: region swap | PSP/E4E/W-Enc: W+ interpolation α=0.5)')
    print(f'{"="*72}')
    print(f"{'Model':<22} {'RTF↓':>8} {'BGP↓':>8} {'IDP↑':>8} {'DS↓':>8}  N")
    print('-'*72)
    for name, r in results_dict.items():
        marker = ' ◀' if 'HierInv' in name else ''
        fmt = lambda v, fmt='f': f"{v:{fmt}}" if v is not None else '   N/A'
        print(f"{name:<22} {fmt(r['RTF'],'.4f'):>8} {fmt(r['BGP'],'.4f'):>8} "
              f"{fmt(r['IDP'],'.4f'):>8} {fmt(r['DS'],'.4f'):>8}  {r['n_pairs']}{marker}")
    print('='*72)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str,
                        default=os.path.join(_ROOT, 'data', 'test_images'))
    parser.add_argument('--n_pairs', type=int, default=100, help='평가 페어 수')
    parser.add_argument('--alpha',   type=float, default=0.5, help='W+ interp alpha')
    parser.add_argument('--seed',    type=int, default=42)
    args = parser.parse_args()

    import lpips as lpips_lib, random
    random.seed(args.seed); np.random.seed(args.seed)

    cfg    = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    all_imgs = sorted(Path(args.test_dir).glob('*.png'))
    shuffled = list(all_imgs)
    random.shuffle(shuffled)
    half  = len(shuffled) // 2
    pairs = list(zip(shuffled[:half], shuffled[half:]))[:args.n_pairs]
    print(f'Test images: {len(all_imgs)}  |  Pairs: {len(pairs)}\n')

    print('Loading LPIPS...')
    lpips_fn = lpips_lib.LPIPS(net='vgg').to(device)

    print('Loading ArcFace...')
    arcface = ArcFaceExtractor(cfg.ir_se50_weights, device)

    print('Loading models...')
    hierinv = load_hierinv(cfg, device)
    psp     = load_psp(cfg, device)
    e4e     = load_e4e(cfg, device)
    wenc    = load_wenc(cfg, device)

    parser_fn        = hierinv.parser
    all_region_names = list(cfg.REGIONS.keys())

    editors = {
        'HierInv-Region v2':  hierinv,
        'PSP (W+ interp)':    psp,
        'E4E (W+ interp)':    e4e,
        'W-Enc (W+ interp)':  wenc,
    }

    swap_scenarios = [
        ['eye'],
        ['nose'],
        ['mouth'],
        ['eye', 'nose', 'mouth'],
        ['brow', 'eye'],
    ]

    all_results = {}

    for swap_regions in swap_scenarios:
        label = '+'.join(swap_regions)
        print(f'\n{"─"*50}\n[Swap: {label}]')

        results = {}
        for name, editor in editors.items():
            print(f'  {name}...', end=' ', flush=True)
            results[name] = evaluate_swap(
                editor, parser_fn, pairs, swap_regions,
                all_region_names, lpips_fn, arcface, device
            )
            r = results[name]
            print(f"RTF={r['RTF']}  BGP={r['BGP']}  IDP={r['IDP']}  DS={r['DS']}")

        all_results[label] = results
        print_table(results, label)

    # 전체 평균 요약
    print(f'\n\n{"="*72}')
    print('  전체 시나리오 평균')
    print('='*72)
    print(f"{'Model':<22} {'RTF↓':>8} {'BGP↓':>8} {'IDP↑':>8} {'DS↓':>8}")
    print('-'*72)
    for name in editors:
        rtfs, bgps, idps, dss = [], [], [], []
        for label_data in all_results.values():
            r = label_data.get(name, {})
            if r.get('RTF') is not None: rtfs.append(r['RTF'])
            if r.get('BGP') is not None: bgps.append(r['BGP'])
            if r.get('IDP') is not None: idps.append(r['IDP'])
            if r.get('DS')  is not None: dss.append(r['DS'])
        marker = ' ◀' if 'HierInv' in name else ''
        fmt = lambda lst: f"{np.mean(lst):.4f}" if lst else '  N/A'
        print(f"{name:<22} {fmt(rtfs):>8} {fmt(bgps):>8} {fmt(idps):>8} {fmt(dss):>8}{marker}")
    print('='*72)
    print('\n※ RTF↓: 타깃 부위가 donor에 가까울수록 좋음')
    print('※ BGP↓: 비타깃 부위가 base에 가까울수록 좋음 (핵심 지표)')
    print('※ IDP↑: ArcFace 정체성이 base에 가까울수록 좋음')
    print('※ DS↓:  비타깃 region들의 변화가 적을수록 좋음')

    out = Path(_ROOT) / 'eval_results' / 'metrics_swap.json'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f'\nSaved → {out}')


if __name__ == '__main__':
    main()
