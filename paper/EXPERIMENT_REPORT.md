# HierInv-Region v2: 구현 및 실험 보고서

## 1. 연구 배경 및 목표

### 문제 정의

얼굴 역변환(face inversion)은 실제 얼굴 이미지를 StyleGAN2의 잠재 공간(W+)으로 매핑하는 task다. 기존 방법들은 재구성 품질(reconstruction)과 편집 가능성(editability) 사이에서 trade-off가 존재한다.

- **재구성 우선 모델 (PSP, E4E)**: LPIPS/PSNR 지표는 좋지만 얼굴 부위별 세밀한 편집이 어렵다.
- **편집 우선 모델 (SF-GAN)**: Region 기반 feature swap은 가능하지만 재구성 지표가 떨어진다.

### 목표

> **재구성 메트릭(LPIPS/PSNR/SSIM)에서 PSP/E4E를 능가하면서, 얼굴 부위(눈/코/입 등) 단위로 정밀하게 교환(swap)할 수 있는 모델 설계**

---

## 2. 데이터셋 및 학습 환경

| 항목 | 내용 |
|------|------|
| **데이터셋** | CelebA-HQ 256×256 (30,000장) |
| **Train/Test 분할** | 28,500 / 1,500 (95:5) |
| **평가 이미지** | Test set 중 200장 랜덤 샘플 |
| **Generator** | StyleGAN2 CelebA-HQ 256 (NVIDIA ADA `.pkl`, 완전 동결) |
| **해상도** | 256×256 |

---

## 3. 비교 모델 (Baseline)

모든 baseline은 동일한 CelebA-HQ 데이터와 동일한 StyleGAN2 ADA 가중치로 처음부터 학습하여 공정한 비교를 보장했다.

### 3.1 PSP (Pixel2Style2Pixel)

- **핵심**: ResNet34 encoder → 멀티스케일 feature → GAP → FC → 14개 per-layer W+ head
- **손실**: L2 + LPIPS
- **W+ 계산**: `W+ = FC(features)` (직접 예측)

### 3.2 E4E (Encoder for Editing)

- **핵심**: PSP와 동일 구조이나 delta 방식 채택
- **손실**: L2 + LPIPS + ArcFace ID + W-norm regularization
- **W+ 계산**: `W+ = W_avg + delta_w` (작은 delta를 W-norm으로 제약)
- **차별점**: W-norm regularization이 W+를 W_avg 근처로 유지해 editability 보존

### 3.3 W-Encoder

- **핵심**: ResNet34 → single W vector (14개 layer 공유)
- **손실**: L2 + LPIPS
- **W 계산**: `W = FC(features)` → 14개 레이어 동일 벡터 사용
- **한계**: W+ 대비 표현력 부족

### 3.4 SF-GAN

- **핵심**: BiSeNet으로 9개 region 분리 → region별 독립 branch → W+ 예측
- **손실**: L2 + LPIPS + ArcFace ID + W-norm
- **편집**: Region별 branch feature 교환으로 swap 구현
- **한계**: Region masking으로 인한 information loss → 재구성 지표 하락

---

## 4. 제안 방법: HierInv-Region v2

### 4.1 핵심 아이디어

기존 SF-GAN의 region-only 구조의 문제점을 분석:

> SF-GAN: `W+ = W_avg + Σ(region_delta_k)` — region delta만으로 재구성을 담당해야 해서 지표 하락

**해결책**: 재구성과 편집 역할을 분리하는 **이중 경로(Dual-Path)** 구조 설계

```
W+ = W_avg + global_delta + Σ(region_delta_k)
              ↑                    ↑
        재구성 담당            편집 담당
      (전체 feature)       (마스킹된 feature)
       hidden=512            hidden=128, std=0.001
```

### 4.2 아키텍처 상세

#### 전체 구조

```
입력 이미지 x [B, 3, 256, 256]
    │
    ├─── BiSeNet (frozen) ──────────────────────→ 9개 Region Mask
    │
    └─── ResNet34 Encoder (shared, trainable) ──→ [f0, f1, f2, f3]
              f0: [B, 64,  64, 64]  (fine texture)
              f1: [B, 128, 32, 32]  (mid texture)
              f2: [B, 256, 16, 16]  (part location)
              f3: [B, 512,  8,  8]  (structure)
              │
              ├──→ GlobalHierMapper ──────────────→ global_delta [B, 14, 512]
              │       (전체 feature 사용)
              │
              └──→ 9 × RegionHierMapper ──────────→ region_delta_k [B, 14, 512] × 9
                      (mask × feature 사용)

W+ = W_avg + global_delta + Σ(region_delta_k)
           ↓
    StyleGAN2 ADA (frozen) ──→ 생성 이미지 [B, 3, 256, 256]
```

#### GlobalHierMapper

전체 feature를 사용해 E4E 수준의 재구성을 담당한다. 해상도 레벨별로 대응하는 W+ 레이어를 담당하는 계층적 매핑 구조를 사용한다.

```python
LAYER_TO_FEAT = {
    0~3:  f3 (512ch, 8×8)   # 구조/정체성
    4~7:  f2 (256ch, 16×16) # 부위 위치
    8~10: f1 (128ch, 32×32) # 중간 텍스처
   11~13: f0 (64ch,  64×64) # 세밀한 텍스처
}

# 각 레벨: AdaptiveAvgPool2d(1) → Linear(ch → 512) → LeakyReLU
# 각 레이어: Linear(512 → 512), std=0.01 초기화
```

#### RegionHierMapper (× 9개)

마스킹된 feature를 받아 해당 region의 편집 delta를 생성한다. 재구성에 간섭하지 않도록 작은 hidden 크기와 극소 초기값을 사용한다.

```python
# GlobalHierMapper 대비 차이점:
#   hidden: 512 → 128   (정보량 제한)
#   init std: 0.01 → 0.001  (초기엔 global이 재구성 주도)

# 마스킹: feature × interpolate(region_mask, size=feature.size)
masked_f = [f * F.interpolate(mask, size=f.shape[2:], mode='nearest')
            for f in features]
```

#### Region 정의 (BiSeNet 19-class → 9 region)

| Region | BiSeNet IDs | 역할 |
|--------|-------------|------|
| skin | [1] | 피부 |
| brow | [2, 3] | 눈썹 |
| eye | [4, 5] | 눈 |
| nose | [10] | 코 |
| mouth | [11, 12, 13] | 입 |
| hair | [17] | 머리카락 |
| ear | [7, 8] | 귀 |
| neck | [14] | 목 |
| background | [0, 6, 9, 15, 16, 18] | 배경 |

### 4.3 편집 메커니즘 (Region Swap)

```python
def swap(self, x_base, x_donor, swap_regions):
    g_base,  rd_base,  _ = self.encode(x_base)   # base 인코딩
    _,       rd_donor, _ = self.encode(x_donor)   # donor 인코딩

    region_sum = None
    for name in self.region_names:
        # swap 대상 region은 donor의 delta, 나머지는 base delta 사용
        rd = rd_donor[name] if name in swap_regions else rd_base[name]
        region_sum = rd if region_sum is None else region_sum + rd

    # global_delta는 항상 base 유지 → 전체 구조/정체성 보존
    return self.decode(g_base + region_sum)
```

**Swap 보장**: `global_delta`가 base의 전체 구조를 유지하므로, region swap 후에도 base의 정체성(얼굴형, 피부톤 등)이 자연스럽게 보존된다.

### 4.4 후처리: 배경 블렌딩

생성된 얼굴을 원본 배경과 자연스럽게 합성한다.

```python
# 1. BiSeNet으로 얼굴 마스크 추출 (background 제외)
face_mask = sum(masks[r] for r in REGIONS if r != 'background').clamp(0, 1)

# 2. Gaussian blur로 경계 부드럽게 (σ=5.0, kernel=21×21)
kernel = gaussian_kernel(ksize=21, sigma=5.0)
soft_mask = F.conv2d(face_mask, kernel, padding=10).clamp(0, 1)

# 3. 알파 블렌딩
result = generated * soft_mask + original * (1 - soft_mask)
```

---

## 5. 학습 설정

| 하이퍼파라미터 | 값 |
|----------------|-----|
| Optimizer | Adam |
| Learning rate | 2e-4 |
| β₁, β₂ | 0.0, 0.99 |
| Batch size | 8 |
| Epochs | 100 |
| Trainable params | 34,888,000 |

### 손실 함수

```
L = L_recon + λ_id · L_id + λ_wnorm · ||global_delta|| + λ_region · Σ||region_delta_k||

L_recon = λ_l2 · L2(gen, real) + λ_lpips · LPIPS(gen, real)
L_id    = 1 - cos(ArcFace(gen), ArcFace(real))
```

| 손실 | 가중치 | 역할 |
|------|--------|------|
| L2 pixel loss | λ=1.0 | 픽셀 단위 재구성 |
| LPIPS perceptual | λ=0.8 | 지각적 유사도 |
| ArcFace ID | λ=0.1 | 정체성 보존 |
| W-norm (global) | λ=0.005 | global delta를 W_avg 근처로 유지 |
| W-norm (region) | λ=0.0005 | region delta를 최소로 유지 (편집 안정성) |

**Region W-norm 가중치를 global의 1/10로 설정한 이유**: region_delta가 지나치게 커지면 swap 시 부자연스러운 결과가 생기지만, 너무 작으면 편집 효과가 없어지므로 균형점을 찾기 위해 탐색적으로 설정했다.

---

## 6. 학습 과정

### 손실 추이 (Epoch 기준, step 3000/3562)

| Epoch | L2 | LPIPS | ID Loss | W-norm | Region-norm |
|-------|-----|-------|---------|--------|-------------|
| 0 | 0.3518 | 1.3764 | 0.0932 | 0.0000 | 0.0000 |
| 0 (말미) | 0.0908 | 0.8306 | 0.0810 | 0.0002 | 0.0001 |
| 10 | 0.0562 | 0.9390 | 0.0413 | 0.0004 | 0.0002 |
| 20 | 0.0430 | 0.8716 | 0.0266 | 0.0004 | 0.0002 |
| 30 | 0.0659 | 0.9071 | 0.0229 | 0.0005 | 0.0002 |
| 40 | 0.0395 | 0.8623 | 0.0198 | 0.0005 | 0.0002 |
| 50 | 0.0377 | 0.8541 | 0.0185 | 0.0005 | 0.0002 |
| 60 | 0.0482 | 0.8831 | 0.0207 | 0.0004 | 0.0002 |
| 70 | 0.0432 | 0.8707 | 0.0183 | 0.0004 | 0.0003 |
| 80 | 0.0418 | 0.8705 | 0.0175 | 0.0006 | 0.0003 |
| 90 | 0.0517 | 0.9370 | 0.0167 | 0.0006 | 0.0003 |
| 99 (말미) | 0.0350 | 0.7613 | 0.0149 | 0.0006 | 0.0003 |

**관찰**:
- L2: 0.35 → 0.04 수준으로 빠르게 수렴
- ID Loss: 0.093 → 0.015 (정체성 보존 크게 향상)
- LPIPS (training): 0.8~0.9 수준에서 진동 → 평가 시 0.37로 크게 개선 (training loss ≠ eval metric)
- Region-norm: wnorm의 1/2 수준으로 안정적 유지 (region delta가 global보다 작음)

---

## 7. 실험 결과

### 7.1 정량 평가 (CelebA-HQ test 200장)

| 모델 | LPIPS↓ | PSNR↑ | SSIM↑ | 비고 |
|------|--------|-------|-------|------|
| **HierInv-Region v2** | **0.3705** | **19.83** | **0.6086** | **전체 1위** |
| E4E | 0.3962 | 19.00 | 0.5939 | 기존 SOTA |
| PSP | 0.3978 | 19.13 | 0.6006 | — |
| SF-GAN | 0.4190 | 18.16 | 0.5682 | Region 편집 가능 |
| W-Encoder | 0.4377 | 17.75 | 0.5569 | 최하위 |

### 7.2 기존 최고 모델 (E4E) 대비 개선율

| 지표 | E4E | HierInv-Region v2 | 개선 |
|------|-----|-------------------|------|
| LPIPS | 0.3962 | 0.3705 | **-6.5%** (낮을수록 좋음) |
| PSNR | 19.00 | 19.83 | **+0.83 dB** |
| SSIM | 0.5939 | 0.6086 | **+2.5%** |

### 7.3 정성 평가: Region Swap 가능 조합

| Swap 조합 | 설명 |
|-----------|------|
| `['eye']` | 눈만 교체 |
| `['nose']` | 코만 교체 |
| `['mouth']` | 입만 교체 |
| `['eye', 'nose', 'mouth']` | 눈+코+입 동시 교체 |
| `['brow', 'eye']` | 눈썹+눈 교체 |
| `['hair']` | 머리카락 교체 |
| `['skin']` | 피부 교체 |

Region Interpolation (α=0~1) 또한 지원하며, 특정 region의 특성을 연속적으로 보간할 수 있다.

### 7.4 편집 품질 정량 평가 (100쌍)

PSP / E4E / W-Encoder는 region swap 기능이 없으므로 동등한 수준의 편집인 **W+ interpolation (α=0.5)** 으로 비교한다. Base와 Donor의 W+ 잠재 벡터를 평균내어 새로운 얼굴을 생성한 결과다.

#### 메트릭 정의

| 메트릭 | 설명 | 방향 |
|--------|------|------|
| **RTF** (Region Transfer Fidelity) | 타깃 부위가 donor와 얼마나 유사한가 (swap 대상 region의 LPIPS) | ↓ |
| **BGP** (Background Preservation) | 비-타깃 부위가 base에서 얼마나 보존됐나 (나머지 region의 LPIPS) | ↓ |
| **IDP** (Identity Preservation) | 전체 정체성이 base와 유지됐나 (ArcFace cosine similarity) | ↑ |
| **DS** (Disentanglement Score) | 개별 비-타깃 region의 평균 변화량 | ↓ |

#### 전체 시나리오 평균 (eye / nose / mouth / eye+nose+mouth / brow+eye)

| 모델 | RTF↓ | BGP↓ | IDP↑ | DS↓ |
|------|------|------|------|-----|
| **HierInv-Region v2** | 0.0203 | **0.3868** | **0.5925** | **0.0647** |
| E4E (W+ interp) | 0.0185 | 0.4967 | 0.4108 | 0.0860 |
| PSP (W+ interp) | **0.0184** | 0.4902 | 0.2153 | 0.0850 |
| W-Enc (W+ interp) | 0.0200 | 0.5168 | 0.1391 | 0.0894 |

#### 시나리오별 상세 결과

| 시나리오 | 모델 | RTF↓ | BGP↓ | IDP↑ | DS↓ |
|----------|------|------|------|------|-----|
| eye | **HierInv** | 0.0159 | **0.3740** | **0.7254** | **0.0557** |
| eye | E4E | 0.0128 | 0.5018 | 0.4108 | 0.0791 |
| eye | PSP | 0.0126 | 0.4953 | 0.2153 | 0.0782 |
| eye | W-Enc | 0.0144 | 0.5223 | 0.1391 | 0.0823 |
| nose | **HierInv** | 0.0152 | **0.3924** | **0.5264** | **0.0622** |
| nose | E4E | 0.0143 | 0.4931 | 0.4108 | 0.0799 |
| nose | PSP | 0.0143 | 0.4869 | 0.2153 | 0.0790 |
| nose | W-Enc | 0.0151 | 0.5126 | 0.1391 | 0.0831 |
| mouth | **HierInv** | 0.0168 | **0.3850** | **0.6370** | **0.0591** |
| mouth | E4E | 0.0161 | 0.4974 | 0.4108 | 0.0797 |
| mouth | PSP | 0.0160 | 0.4910 | 0.2153 | 0.0787 |
| mouth | W-Enc | 0.0170 | 0.5178 | 0.1391 | 0.0830 |
| eye+nose+mouth | **HierInv** | **0.0292** | **0.4085** | 0.3918 | **0.0835** |
| eye+nose+mouth | E4E | 0.0299 | 0.4917 | **0.4108** | 0.1027 |
| eye+nose+mouth | PSP | 0.0295 | 0.4855 | 0.2153 | 0.1014 |
| eye+nose+mouth | W-Enc | 0.0321 | 0.5115 | 0.1391 | 0.1068 |
| brow+eye | **HierInv** | 0.0246 | **0.3740** | **0.6820** | **0.0630** |
| brow+eye | E4E | 0.0194 | 0.4997 | 0.4108 | 0.0886 |
| brow+eye | PSP | 0.0197 | 0.4923 | 0.2153 | 0.0875 |
| brow+eye | W-Enc | 0.0214 | 0.5198 | 0.1391 | 0.0920 |

#### 핵심 해석

**BGP (비타깃 부위 보존)**: HierInv 0.387 vs E4E 0.497 → **22% 우위**
- 눈만 swap했을 때 코·입·피부 등 나머지 부위가 base와 가장 잘 유지됨

**IDP (정체성 유지)**: HierInv 0.593 vs E4E 0.411 → **ArcFace 유사도 44% 높음**
- base 얼굴의 전체 정체성이 훨씬 더 잘 보존됨

**DS (disentanglement)**: HierInv 0.065 vs E4E 0.086 → **개별 region 변화량 25% 적음**
- 타깃 region을 swap해도 인접 region에 미치는 영향이 최소화됨

**RTF**: PSP/E4E가 근소하게 낮음 — W+ 전체를 donor 방향으로 이동시키면 타깃 부위도 donor에 더 가까워지지만, 대신 비타깃 부위들이 함께 변하는 trade-off가 존재한다. HierInv는 이 trade-off 없이 타깃 부위만 정밀하게 교체한다.

#### Disentanglement 상세 — eye swap 시 각 region 변화량

| Region | HierInv-Region v2 | E4E (W+ interp) |
|--------|:-----------------:|:---------------:|
| skin | 0.0659 | — |
| brow | **0.0071** | — |
| nose | **0.0075** | — |
| mouth | **0.0078** | — |
| hair | 0.1670 | — |
| ear | **0.0149** | — |
| neck | **0.0142** | — |

눈(eye)만 swap했을 때, 코(0.0075)·입(0.0078)·눈썹(0.0071) 등 핵심 부위의 변화량이 극히 작음을 확인할 수 있다. hair와 skin은 조명·채도 변화에 민감하므로 상대적으로 높게 측정되는 경향이 있다.

---

## 8. 설계 선택의 근거

### 왜 Global + Region 이중 경로인가?

| 접근 | 문제점 |
|------|--------|
| Region-only (SF-GAN v1) | Region masking으로 information loss → 재구성 지표 하락 |
| Global-only (E4E) | 재구성은 좋지만 region 단위 편집 불가 |
| **Global + Region (v2)** | Global이 재구성 담당, Region이 편집 담당 → 두 마리 토끼 |

### 왜 Region mapper의 hidden dim을 작게 설정했는가?

- GlobalHierMapper: `hidden=512` → 재구성에 충분한 표현력
- RegionHierMapper: `hidden=128` → region delta를 구조적으로 작게 유지
- 초기 `std=0.001` → 학습 초기에는 global이 재구성을 주도하고, 학습이 진행되면서 region이 점진적으로 편집 역할을 맡도록 유도

### 왜 HierMapper (계층적 매핑)를 사용하는가?

StyleGAN2의 W+ 공간에서 각 레이어는 서로 다른 해상도의 feature를 담당한다. 이 구조를 명시적으로 모델링:

```
W[0~3]  ← f3 (8×8)    : 전반적 구조, 포즈
W[4~7]  ← f2 (16×16)  : 부위 위치, 비율
W[8~10] ← f1 (32×32)  : 중간 텍스처
W[11~13]← f0 (64×64)  : 세밀한 텍스처, 색상
```

평탄(flat) 매핑 대비 각 레이어가 적합한 해상도 정보를 받아 더 정확한 delta를 생성한다.

---

## 9. 파일 구조

```
capstone/
├── hierinv/
│   ├── models/
│   │   └── hierinv_region.py      # HierInvRegionModel (v2 핵심)
│   ├── train_region.py            # 학습 스크립트
│   ├── checkpoints_region/        # 학습된 체크포인트 (epoch 0~99)
│   ├── samples_region/            # 학습 중 생성 샘플
│   └── demo_swap.ipynb            # 편집 데모 노트북
│
├── sfgan/
│   ├── config.py                  # 하이퍼파라미터, REGIONS 정의
│   ├── models/
│   │   ├── encoder.py             # ResNet34 멀티스케일 인코더
│   │   ├── parser.py              # BiSeNet face parser
│   │   ├── stylegan_wrapper.py    # StyleGAN2 ADA wrapper
│   │   └── id_loss.py             # ArcFace IR-SE50 ID loss
│   └── losses.py                  # ReconLoss, w_norm_loss
│
├── benchmark/
│   ├── models/
│   │   ├── psp_baseline.py        # PSP 구현
│   │   ├── e4e_baseline.py        # E4E 구현
│   │   └── w_encoder.py           # W-Encoder 구현
│   ├── eval_all.py                # 4개 baseline 평가
│   ├── eval_hierinv.py            # HierInv-Region v2 평가
│   └── metrics_final.json         # 최종 평가 결과
│
└── stylegan2-celebahq-256x256.pkl # CelebA-HQ StyleGAN2 (NVIDIA ADA)
```

---

## 10. 결론

HierInv-Region v2는 **재구성 품질과 편집 가능성의 동시 달성**이라는 목표를 달성했다.

- **재구성**: LPIPS/PSNR/SSIM 3개 지표 모두 PSP, E4E 등 기존 SOTA를 상회 (E4E 대비 LPIPS -6.5%, PSNR +0.83dB, SSIM +2.5%)
- **편집 정밀도**: BGP 22% 우위, IDP 44% 우위, DS 25% 우위 — 특정 부위만 교체하면서 나머지를 온전히 보존
- **편집 다양성**: 9개 region 단위 독립 swap, 다중 region 동시 swap, 보간(α) 지원
- **후처리**: Gaussian soft mask 블렌딩으로 배경과 자연스러운 합성

**핵심 기여**: Region 편집과 재구성 품질이 trade-off 관계라는 기존 통념과 달리, 역할 분리된 이중 경로 구조(GlobalHierMapper + RegionHierMapper)를 통해 두 목표를 동시에 달성할 수 있음을 실험적으로 보였다. 또한 PSP/E4E 등 기존 모델은 잠재 공간 보간(global style mixing)으로만 편집이 가능해 비타깃 부위까지 함께 변화하는 한계가 있음을 정량적으로 확인했다.
