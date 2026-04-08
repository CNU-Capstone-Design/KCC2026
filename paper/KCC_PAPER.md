# KCC 논문 초안
## 계층적 이중 경로 구조를 통한 영역 분리형 얼굴 GAN 역변환

---

## 논문 흐름 (스토리라인)

```
[문제 제기]
얼굴 역변환에는 두 가지 상충 목표가 존재한다.
  - 재구성 품질 (LPIPS/PSNR/SSIM): 원본을 얼마나 잘 복원하는가
  - 편집 가능성: 눈/코/입 등 특정 부위를 독립적으로 바꿀 수 있는가
기존 방법은 둘 중 하나를 포기해야 했다.
        ↓
[기존 방법의 한계]
  PSP/E4E → 재구성은 좋지만, W+ 전체를 하나의 벡터로 예측 → 부위별 편집 불가
  Region 기반(SF-GAN) → 편집은 되지만, 마스킹으로 인한 정보 손실 → 재구성 지표 하락
        ↓
[우리의 해결책]
  역할을 분리한다.
  Global mapper  → 재구성 전담 (전체 feature 사용, 표현력 충분)
  Region mapper  → 편집 전담   (마스킹 feature 사용, 작은 delta 유지)
  W+ = W_avg + global_delta + Σ(region_delta_k)
        ↓
[결과]
  재구성: 3개 지표 모두 PSP/E4E 초과 (E4E 대비 LPIPS -6.5%)
  편집:   BGP +22%, IDP +44%, DS +25% — 타깃 부위만 정밀하게 교체
```

---

## 논문 구성

| 절 | 제목 | 분량 |
|----|------|------|
| 1 | 서론 | 0.5p |
| 2 | 관련 연구 | 0.5p |
| 3 | 제안 방법 | 1.5p |
| 4 | 실험 및 결과 | 1.2p |
| 5 | 결론 | 0.3p |

---
---

# 본문 초안

---

## 1. 서론

얼굴 역변환(face inversion)은 실제 얼굴 이미지를 StyleGAN2[1]의 잠재 공간(W+)으로 매핑하는 task로, 얼굴 편집·합성·변환 등 다양한 응용에서 핵심 기반 기술로 활용된다. 이상적인 역변환 모델은 두 가지 조건을 동시에 만족해야 한다. 첫째, **재구성 품질**: 원본 이미지를 픽셀·지각적 수준에서 충실히 복원해야 한다. 둘째, **편집 가능성**: 눈, 코, 입 등 얼굴의 특정 부위를 독립적으로 제어할 수 있어야 한다.

그러나 기존 방법들은 이 두 목표 사이에서 trade-off를 피하지 못했다. PSP[2], E4E[3] 등 encoder 기반 방법은 전체 얼굴을 단일 W+ 벡터로 압축하므로 재구성 지표는 우수하지만, 얼굴 부위별 독립 편집이 불가능하다. 반면 region별 W+ 예측을 시도한 방법들은 마스킹에 따른 정보 손실로 재구성 품질이 저하된다.

본 논문은 이 trade-off를 **역할 분리 이중 경로(dual-path)** 구조로 해결한 HierInv-Region v2를 제안한다. 재구성을 담당하는 GlobalHierMapper와 편집을 담당하는 RegionHierMapper를 분리함으로써, 재구성 품질과 부위별 편집 가능성을 동시에 달성한다. CelebA-HQ 256×256에서의 실험 결과, 제안 방법은 재구성 지표(LPIPS/PSNR/SSIM) 전 항목에서 기존 SOTA를 상회하며, 편집 품질 지표(BGP/IDP/DS)에서도 비교 모델 대비 유의미한 우위를 보인다.

---

## 2. 관련 연구

**얼굴 GAN 역변환.** PSP[2]는 ResNet34 encoder로 멀티스케일 feature를 추출하고, per-layer FC head로 W+ 벡터를 직접 예측한다. E4E[3]는 PSP와 동일한 구조에 delta 방식과 W-norm regularization을 도입해 W+를 W_avg 근처로 유지함으로써 편집 가능성을 높였다. 두 방법 모두 전체 얼굴을 하나의 잠재 벡터로 표현하므로, 특정 부위만 독립적으로 제어하기 어렵다.

**부위별 얼굴 편집.** Barbershop[4]은 FS 공간(W+ + feature map F)에서 최적화 기반 역변환을 수행하며 헤어스타일 전이에 특화되어 있다. MegaFS[5]는 HieRFE를 통해 계층적 W+ 표현을 구성하고 Face Transfer Module로 identity를 이식하지만, 편집 단위가 얼굴 전체이며 부위별 제어는 지원하지 않는다. 본 논문의 제안 방법은 9개 region을 독립적으로 제어하면서도 재구성 품질을 유지한다는 점에서 차별화된다.

---

## 3. 제안 방법

### 3.1 전체 구조

제안 방법의 W+ 예측 수식은 다음과 같다.

```
W+ = W_avg + global_delta + Σ_{k=1}^{9} region_delta_k
              ↑                         ↑
        GlobalHierMapper           RegionHierMapper × 9
        (재구성 전담)                (편집 전담)
```

입력 이미지는 공유 ResNet34 encoder를 통해 4개 스케일의 feature `[f0, f1, f2, f3]`로 변환된다. 이 feature는 GlobalHierMapper와 9개의 RegionHierMapper에 동시에 공급된다. 최종 W+는 StyleGAN2 ADA generator(완전 동결)에 입력되어 얼굴 이미지를 생성한다.

BiSeNet[6]으로 추출한 9개 region(skin, brow, eye, nose, mouth, hair, ear, neck, background)의 마스크는 RegionHierMapper의 입력 feature를 마스킹하는 데 사용된다.

### 3.2 GlobalHierMapper

GlobalHierMapper는 전체 encoder feature를 이용해 E4E 수준의 재구성을 담당한다. StyleGAN2 W+ 공간의 각 레이어가 서로 다른 해상도의 시각 정보를 담당한다는 점에 착안해, feature 해상도와 W+ 레이어를 계층적으로 대응시킨다.

```
W[0~3]  ← f3 (512ch, 8×8)    구조·포즈
W[4~7]  ← f2 (256ch, 16×16)  부위 위치·비율
W[8~10] ← f1 (128ch, 32×32)  중간 텍스처
W[11~13]← f0 (64ch,  64×64)  세밀한 텍스처·색상
```

각 레벨은 AdaptiveAvgPool2d → Linear(ch→512) → LeakyReLU로 처리되며, per-layer FC head(std=0.01)로 delta를 생성한다.

### 3.3 RegionHierMapper

RegionHierMapper는 GlobalHierMapper와 동일한 계층 구조를 따르지만, **마스킹된 feature**를 입력으로 받아 해당 region의 편집 delta만을 생성한다. 재구성에 간섭하지 않도록 두 가지 설계 선택을 적용했다.

- **hidden dim 축소**: 512 → 128 (표현력 제한 → region delta를 구조적으로 작게 유지)
- **초기값 억제**: FC head std 0.01 → 0.001 (학습 초기에는 global이 재구성을 주도, 이후 점진적으로 편집 역할 분담)

마스킹은 `masked_f = f × interpolate(mask, size=f.shape[2:])` 방식으로 feature 해상도에 맞게 적용된다.

### 3.4 Region Swap 메커니즘

편집 시에는 base 이미지의 global_delta를 유지하면서, 교체할 region의 delta만 donor 이미지의 것으로 대체한다.

```python
W+_result = W_avg + global_delta_base
           + Σ_{k ∉ swap} region_delta_k_base
           + Σ_{k ∈ swap} region_delta_k_donor
```

global_delta가 base의 전체 구조·정체성을 고정하므로, region swap 후에도 base의 얼굴형·피부톤 등이 자연스럽게 유지된다. 생성된 얼굴은 BiSeNet 마스크 기반 Gaussian soft mask(σ=5.0)로 원본 배경과 블렌딩된다.

---

## 4. 실험 및 결과

### 4.1 실험 설정

- **데이터셋**: CelebA-HQ 256×256, 28,500장 학습 / 1,500장 테스트
- **Generator**: StyleGAN2 CelebA-HQ 256 (NVIDIA ADA `.pkl`, 완전 동결)
- **비교 모델**: PSP, E4E, W-Encoder — 동일 데이터·동일 generator로 처음부터 학습하여 공정 비교
- **학습**: Adam(lr=2e-4, β=(0.0, 0.99)), batch=8, 100 epoch
- **손실**: L2(λ=1.0) + LPIPS(λ=0.8) + ArcFace ID(λ=0.1) + W-norm(λ=0.005)

### 4.2 재구성 품질 비교

test set 200장에 대한 정량 평가 결과는 아래와 같다. 제안 방법은 LPIPS, PSNR, SSIM 세 지표 모두에서 기존 방법을 상회한다.

| 모델 | LPIPS↓ | PSNR↑ | SSIM↑ |
|------|--------|-------|-------|
| **HierInv-Region v2 (제안)** | **0.3705** | **19.83** | **0.6086** |
| E4E [3] | 0.3962 | 19.00 | 0.5939 |
| PSP [2] | 0.3978 | 19.13 | 0.6006 |
| W-Encoder | 0.4377 | 17.75 | 0.5569 |

기존 최고 성능인 E4E 대비 LPIPS 6.5% 감소, PSNR +0.83dB, SSIM +2.5% 향상을 달성했다. 이는 GlobalHierMapper가 재구성을 전담함으로써 region masking에 따른 정보 손실 없이 E4E 수준 이상의 재구성이 가능함을 보여준다.

### 4.3 편집 품질 비교

PSP, E4E, W-Encoder는 region swap 기능이 없으므로, 동등한 수준의 편집인 **W+ interpolation(α=0.5)**—base와 donor의 W+ 잠재 벡터를 평균하는 방식—과 비교한다. 평가는 100쌍의 (base, donor) 이미지에 대해 eye/nose/mouth/eye+nose+mouth/brow+eye 5개 시나리오 평균으로 산출했다.

**평가 지표 정의**
- **RTF** (Region Transfer Fidelity): 타깃 부위가 donor와 얼마나 유사한가 ↓
- **BGP** (Background Preservation): 비-타깃 부위가 base에서 얼마나 보존됐나 ↓
- **IDP** (Identity Preservation): ArcFace cosine similarity (base 기준) ↑
- **DS** (Disentanglement Score): 개별 비-타깃 region의 평균 변화량 ↓

| 모델 | RTF↓ | BGP↓ | IDP↑ | DS↓ |
|------|------|------|------|-----|
| **HierInv-Region v2 (제안)** | 0.0203 | **0.3868** | **0.5925** | **0.0647** |
| E4E (W+ interp) | 0.0185 | 0.4967 | 0.4108 | 0.0860 |
| PSP (W+ interp) | **0.0184** | 0.4902 | 0.2153 | 0.0850 |
| W-Enc (W+ interp) | 0.0200 | 0.5168 | 0.1391 | 0.0894 |

비교 모델(W+ interpolation)은 RTF에서 근소하게 낮은 값을 보이는데, 이는 W+ 전체를 donor 방향으로 이동시키면 타깃 부위도 donor에 더 가까워지기 때문이다. 그러나 이 과정에서 비-타깃 부위까지 함께 변하는 trade-off가 발생한다. 제안 방법은 BGP에서 22%, IDP에서 44%, DS에서 25%의 우위를 보이며, **타깃 부위만 정밀하게 교체하면서 나머지를 온전히 보존**한다는 핵심 목표를 정량적으로 달성했음을 확인했다.

### 4.4 정성 결과

**[그림 1] 재구성 품질 비교**

![fig1_recon](benchmark/paper_figures/final/fig3_recon_compare.png)

그림 1은 동일 입력에 대한 각 모델의 재구성 결과다. HierInv-Region v2(2열)는 전 행에 걸쳐 원본(1열)과 가장 유사한 결과를 생성하며, 피부 텍스처·조명·배경 등 세밀한 부분까지 잘 보존한다. W-Encoder(5열)는 표현력 한계로 전반적으로 블러 및 색감 차이가 발생하고, PSP·E4E는 중간 수준의 품질을 보인다.

---

**[그림 2] HierInv Region Swap 쇼케이스**

![fig2_swap](benchmark/paper_figures/final/fig1_swap_showcase.png)

그림 2는 6쌍의 (Base, Donor) 이미지에 대해 eye, nose, mouth, eye+nose+mouth, hair 5가지 시나리오로 swap한 결과다. 각 행에서 swap된 부위 이외의 영역(피부·헤어·얼굴형 등)이 Base와 동일하게 유지됨을 확인할 수 있다. 특히 Row 2(흑인 여성+금발 여성)에서 Hair swap 시 헤어 색상만 명확하게 교체되는 것이 두드러진다.

---

**[그림 3] 모델 간 편집 비교 (Eye swap)**

![fig3_compare](benchmark/paper_figures/final/fig2_model_compare_eye.png)

그림 3은 동일한 (Base, Donor) 쌍에 대해 HierInv의 eye-only swap과 PSP·E4E·W-Enc의 W+ interpolation(α=0.5) 결과를 비교한다. HierInv(3열)는 Base의 전체 얼굴 구조·피부톤·헤어를 유지하면서 눈 영역만 변화시키는 반면, PSP·E4E·W-Enc(4~6열)는 얼굴 전체가 Donor 방향으로 블렌딩되어 Base의 정체성이 크게 훼손된다. 이는 정량 평가(BGP +22%, IDP +44%)와 일치하는 정성적 결과다.

---

## 5. 결론

본 논문은 얼굴 역변환에서 재구성 품질과 부위별 편집 가능성이 상충한다는 기존 문제를 **이중 경로 구조**로 해결한 HierInv-Region v2를 제안했다. GlobalHierMapper가 재구성을 전담하고 RegionHierMapper가 편집을 전담하는 역할 분리를 통해, 두 목표를 동시에 달성할 수 있음을 실험적으로 검증했다. 제안 방법은 재구성 지표 전 항목에서 PSP, E4E 등 기존 SOTA를 상회하며, 편집 품질 지표에서도 W+ interpolation 기반 비교 대비 유의미한 우위를 보였다.

향후 연구로는 고해상도(1024px) 도메인으로의 확장, 영상(video)에서의 시간적 일관성 유지, 그리고 더 세밀한 region 분할을 통한 편집 정밀도 향상을 고려한다.

---

## 참고문헌

```
[1] Karras et al., "Analyzing and Improving the Image Quality of StyleGAN", CVPR 2020.
[2] Richardson et al., "Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation", CVPR 2021.
[3] Tov et al., "Designing an Encoder for StyleGAN Image Manipulation", SIGGRAPH 2021.
[4] Zhu et al., "Barbershop: GAN-based Image Compositing using Segmentation Masks", SIGGRAPH Asia 2021.
[5] Zhu et al., "One Shot Face Swapping on Megapixels", CVPR 2021.
[6] Yu et al., "BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation", ECCV 2018.
```
