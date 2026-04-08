# HierInv-Region v2: Hierarchical Dual-Path GAN Inversion for Region-Specific Face Editing

> **KCC 2026** | [Paper](paper/KCC_PAPER.md) | [Experiment Report](paper/EXPERIMENT_REPORT.md)

HierInv-Region v2는 얼굴을 9개 semantic region으로 분리하여 특정 부위(눈, 코, 입 등)만 정밀하게 교환할 수 있는 GAN inversion 프레임워크입니다.

**핵심 아이디어:** W+ = W_avg + **global_delta** (재구성) + Σ **region_delta_k** (편집) × 9개 독립 경로

---

## Results

### Reconstruction Quality (CelebA-HQ 256×256, 200장)

| Model | LPIPS ↓ | PSNR ↑ | SSIM ↑ |
|-------|---------|--------|--------|
| **HierInv-Region v2 (Ours)** | **0.3705** | **19.83** | **0.6086** |
| E4E | 0.3962 | 19.00 | 0.5939 |
| PSP | 0.3978 | 19.13 | 0.6006 |
| W-Encoder | 0.4377 | 17.75 | 0.5569 |

### Editing Quality — Eye Swap (100 쌍, 평균)

| Model | RTF ↓ | BGP ↓ | IDP ↑ | DS ↓ |
|-------|-------|-------|-------|------|
| **HierInv-Region v2 (Ours)** | 0.0159 | **0.3740** | **0.7254** | **0.0557** |
| PSP (W+ interp) | 0.0126 | 0.4953 | 0.2153 | 0.0782 |
| E4E (W+ interp) | 0.0128 | 0.5018 | 0.4108 | 0.0791 |
| W-Enc (W+ interp) | 0.0144 | 0.5223 | 0.1391 | 0.0823 |

> **RTF**: Region Transfer Fidelity (↓) — 타깃 부위가 donor와 유사한가  
> **BGP**: Background Preservation (↓) — 비타깃 부위가 base를 유지하는가 (핵심)  
> **IDP**: Identity Preservation (↑) — ArcFace 정체성이 base와 얼마나 일치하는가  
> **DS**: Disentanglement Score (↓) — 비타깃 region들의 변화량 (낮을수록 분리 잘 됨)

---

## Setup

### 1. 의존 저장소 클론

```bash
cd paper_release/

# StyleGAN2-ADA (NVIDIA, .pkl 형식)
git clone https://github.com/NVlabs/stylegan2-ada-pytorch stylegan2-ada-pytorch

# BiSeNet face parser
git clone https://github.com/zllrunning/face-parsing.PyTorch face-parsing.PyTorch
```

### 2. Python 패키지 설치

```bash
pip install torch torchvision
pip install lpips scikit-image pillow ninja
pip install dlib opencv-python  # 얼굴 정렬에 필요
```

### 3. Pretrained Weights 다운로드

`pretrained/` 디렉토리를 생성하고 아래 파일들을 배치합니다:

```
pretrained/
├── stylegan2-celebahq-256x256.pkl   # NVIDIA StyleGAN2-ADA CelebA-HQ 256
├── bisenet.pth                       # CelebAMask-HQ BiSeNet face parser
└── model_ir_se50.pth                 # ArcFace IR-SE50 (학습/평가에 필요)
```

| 파일 | 출처 |
|------|------|
| `stylegan2-celebahq-256x256.pkl` | [NVIDIA StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch#pretrained-networks) |
| `bisenet.pth` | [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) |
| `model_ir_se50.pth` | [InsightFace / TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) |

### 4. 데이터셋 준비 (CelebA-HQ)

```bash
# CelebA-HQ 256×256 다운로드 (약 3GB)
# https://github.com/tkarras/progressive_growing_of_gans 참고
# 또는 HuggingFace:
# huggingface-cli download tkarras/celeba-hq --local-dir data/celebahq_raw

# 얼굴 정렬 (선택 — 이미 정렬된 경우 생략)
python scripts/align_faces.py --src data/celebahq_raw --dst data/celebahq_aligned

# 학습/테스트 분할 (선택)
python scripts/prepare_test_set.py --data_dir data/celebahq_aligned
```

---

## Training

### HierInv-Region v2 (제안 모델)

```bash
python train/train_hierinv.py

# 재개
python train/train_hierinv.py --resume checkpoints/hierinv/ckpt_epoch0049.pth
```

### Baselines (PSP / E4E / W-Encoder)

```bash
python train/train_baselines.py --model psp
python train/train_baselines.py --model e4e
python train/train_baselines.py --model wenc

# 재개
python train/train_baselines.py --model e4e --resume checkpoints/e4e/ckpt_epoch0049.pth
```

학습 설정 (`config.py`):
- Optimizer: Adam, lr=2e-4, β=(0.0, 0.99)
- Batch size: 8
- Epochs: 100
- Loss: L2(1.0) + LPIPS(0.8) + ArcFace ID(0.1) + W-norm(0.005)

---

## Inference

### 얼굴 부위 교환

```bash
# 눈 교환
python inference/swap.py --base img_a.jpg --donor img_b.jpg --swap eye

# 복수 부위 교환 (눈+코+입)
python inference/swap.py --base img_a.jpg --donor img_b.jpg --swap eye nose mouth

# 비교 그리드 함께 저장 (left: base | center: donor | right: result)
python inference/swap.py --base img_a.jpg --donor img_b.jpg --swap nose --grid

# 결과 경로 지정
python inference/swap.py --base img_a.jpg --donor img_b.jpg --swap mouth --out result.png
```

**교환 가능한 부위:** `skin`, `brow`, `eye`, `nose`, `mouth`, `hair`, `ear`, `neck`, `background`

---

## Evaluation

### 재구성 품질 (LPIPS / PSNR / SSIM)

```bash
python eval/eval_reconstruction.py
python eval/eval_reconstruction.py --test_dir data/test_images --n 200
```

### 편집 품질 (RTF / BGP / IDP / DS)

```bash
python eval/eval_swap.py
python eval/eval_swap.py --n_pairs 100 --test_dir data/test_images
```

결과는 `eval_results/` 에 JSON 형식으로 저장됩니다.

---

## Architecture

```
Input Image (256×256)
        │
   ResNet34 Encoder
        │
   4-scale features [64, 128, 256, 512]
        │
   ┌────┴──────────────────────────────────────────┐
   │                                               │
   GlobalHierMapper                          9× RegionHierMapper
   (hidden=512, std=0.01)                    (hidden=128, std=0.001)
   [재구성 전담]                              [편집 전담, 작은 delta]
        │                                               │
   global_delta [B,14,512]              region_delta_k [B,14,512] × 9
        │                                               │
        └──────────────── + ────────────────────────────┘
                          │
               total_delta = global_delta + Σ region_delta_k
                          │
              W+ = W_avg + total_delta
                          │
          StyleGAN2-ADA (CelebA-HQ 256, 완전 동결)
                          │
               Output Image (256×256)
```

### Region 정의 (BiSeNet 19-class → 9 region)

| Region | BiSeNet 클래스 ID |
|--------|-----------------|
| skin | [1] |
| brow | [2, 3] |
| eye | [4, 5] |
| nose | [10] |
| mouth | [11, 12, 13] |
| hair | [17] |
| ear | [7, 8] |
| neck | [14] |
| background | [0, 6, 9, 15, 16, 18] |

### W+ 계층적 매핑 (256px, 14 레이어)

| W+ 레이어 | Encoder Feature | 역할 |
|-----------|----------------|------|
| W[0-3] | f3 (512ch, 8×8) | 전체 구조·얼굴형 |
| W[4-7] | f2 (256ch, 16×16) | 부위 위치·형태 |
| W[8-10] | f1 (128ch, 32×32) | 중간 텍스처 |
| W[11-13] | f0 (64ch, 64×64) | 세밀한 텍스처 |

---

## File Structure

```
paper_release/
├── config.py                    # 공통 설정 (경로, 하이퍼파라미터)
├── shared/                      # 공유 모듈
│   ├── encoder.py               # ResNet34 multi-scale encoder
│   ├── parser.py                # BiSeNet face parser (frozen)
│   ├── stylegan_wrapper.py      # StyleGAN2-ADA wrapper (.pkl)
│   ├── id_loss.py               # ArcFace identity loss
│   ├── losses.py                # ReconLoss, w_norm_loss
│   └── dataset.py               # CelebA-HQ DataLoader
├── models/
│   ├── hierinv_region.py        # HierInv-Region v2 (제안 모델)
│   ├── psp.py                   # PSP baseline
│   ├── e4e.py                   # E4E baseline
│   └── w_encoder.py             # W-Encoder baseline
├── train/
│   ├── train_hierinv.py         # HierInv 학습
│   └── train_baselines.py       # PSP/E4E/W-Enc 통합 학습
├── eval/
│   ├── eval_reconstruction.py   # LPIPS/PSNR/SSIM 평가
│   └── eval_swap.py             # RTF/BGP/IDP/DS 편집 평가
├── inference/
│   └── swap.py                  # 얼굴 부위 교환 추론
├── checkpoints/
│   ├── hierinv/                 # HierInv-Region v2 가중치
│   ├── psp/                     # PSP 가중치
│   ├── e4e/                     # E4E 가중치
│   └── wenc/                    # W-Encoder 가중치
└── paper/
    ├── KCC_PAPER.md             # KCC 2026 논문
    ├── EXPERIMENT_REPORT.md     # 상세 실험 보고서
    └── figures/                 # 논문 그림
```

---

## Citation

```bibtex
@inproceedings{hierinv_region_2026,
  title     = {HierInv-Region v2: 계층적 이중 경로 GAN 역변환을 통한 부위별 얼굴 편집},
  booktitle = {한국컴퓨터종합학술대회 (KCC)},
  year      = {2026},
}
```
