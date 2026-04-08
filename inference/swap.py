"""
HierInv-Region v2 얼굴 부위 교환 추론 스크립트.

사용법:
    # 단일 부위 swap (base의 눈을 donor의 눈으로 교체)
    python inference/swap.py --base img_a.jpg --donor img_b.jpg --swap eye

    # 복수 부위 swap
    python inference/swap.py --base img_a.jpg --donor img_b.jpg --swap eye nose mouth

    # 결과 저장 경로 지정
    python inference/swap.py --base img_a.jpg --donor img_b.jpg --swap eye --out result.png

교환 가능한 부위:
    skin, brow, eye, nose, mouth, hair, ear, neck, background
"""

import os, sys, argparse, warnings
warnings.filterwarnings('ignore')

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'shared'))

import torch
import torchvision.transforms as T
import torchvision.utils as vutils
from pathlib import Path
from PIL import Image

from config import Config

VALID_REGIONS = ['skin', 'brow', 'eye', 'nose', 'mouth', 'hair', 'ear', 'neck', 'background']

to_tensor = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),
])
denorm = T.Normalize([-1]*3, [2]*3)


def load_image(path, device):
    img = Image.open(path).convert('RGB')
    return to_tensor(img).unsqueeze(0).to(device)


def save_image(tensor, path):
    img = denorm(tensor.squeeze(0).cpu()).clamp(0, 1)
    T.ToPILImage()(img).save(path)


def load_model(cfg, device, ckpt_dir=None):
    from models.hierinv_region import HierInvRegionModel
    model = HierInvRegionModel(cfg, device).to(device)
    model.eval()

    if ckpt_dir is None:
        ckpt_dir = os.path.join(_ROOT, 'checkpoints', 'hierinv')

    ckpts = sorted(Path(ckpt_dir).glob('ckpt_epoch*.pth'))
    if not ckpts:
        raise FileNotFoundError(
            f'No checkpoints found in {ckpt_dir}\n'
            f'Please train first: python train/train_hierinv.py\n'
            f'Or download pretrained weights (see README).'
        )
    ckpt_path = ckpts[-1]
    print(f'Loading checkpoint: {ckpt_path.name}')

    ckpt = torch.load(ckpt_path, map_location=device)
    model.encoder.load_state_dict(ckpt['encoder'])
    model.global_mapper.load_state_dict(ckpt['global_mapper'])
    model.region_mappers.load_state_dict(ckpt['region_mappers'])
    return model


def main():
    parser = argparse.ArgumentParser(
        description='HierInv-Region v2: 얼굴 부위 교환 추론',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--base',  type=str, required=True, help='기준 이미지 경로 (얼굴형·정체성 보존)')
    parser.add_argument('--donor', type=str, required=True, help='부위 제공 이미지 경로')
    parser.add_argument('--swap',  type=str, nargs='+', required=True,
                        choices=VALID_REGIONS,
                        help=f'교환할 부위: {VALID_REGIONS}')
    parser.add_argument('--out',   type=str, default=None,
                        help='결과 저장 경로 (기본: results/swap_<regions>.png)')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='체크포인트 디렉토리 (기본: checkpoints/hierinv/)')
    parser.add_argument('--grid',  action='store_true',
                        help='base | donor | result 비교 그리드도 함께 저장')
    args = parser.parse_args()

    cfg    = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Swap regions: {args.swap}')

    # 결과 경로 결정
    if args.out is None:
        region_tag = '_'.join(args.swap)
        out_dir = Path(_ROOT) / 'results'
        out_dir.mkdir(exist_ok=True)
        args.out = str(out_dir / f'swap_{region_tag}.png')

    # 모델 로드
    model = load_model(cfg, device, args.ckpt_dir)

    # 이미지 로드
    x_base  = load_image(args.base,  device)
    x_donor = load_image(args.donor, device)

    # 추론
    print('Running inference...')
    with torch.no_grad():
        result = model.swap(x_base, x_donor, swap_regions=args.swap)

    # 결과 저장
    save_image(result, args.out)
    print(f'Result saved → {args.out}')

    # 비교 그리드 저장
    if args.grid:
        grid_path = str(Path(args.out).with_suffix('')) + '_grid.png'
        grid = vutils.make_grid(
            torch.cat([x_base, x_donor, result], dim=0),
            nrow=3, normalize=True, value_range=(-1, 1)
        )
        vutils.save_image(grid, grid_path)
        print(f'Grid saved → {grid_path}  (left: base | center: donor | right: result)')


if __name__ == '__main__':
    main()
