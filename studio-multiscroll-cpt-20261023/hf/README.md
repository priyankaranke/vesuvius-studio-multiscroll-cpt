---
library_name: pytorch
pipeline_tag: image-segmentation
base_model: paranke/vesuvius-814-841-9b-resunet-studio-20260926
tags:
- vesuvius
- ink-detection
- 3d-unet
- micro-ct
- image-segmentation
---

# Studio · multi-scroll continued pretraining (2026-10-23)

The [Studio ink model](https://huggingface.co/paranke/vesuvius-814-841-9b-resunet-studio-20260926) (epoch 9, trained on
PHerc0814 / PHerc0841 / PHerc0009B sparse labels) **continued for 30,000 updates on labeled surfaces of twelve other scroll
groups, with Studio's own 814 / 841 / 9B data replayed in every batch**. Same architecture, input and inference recipe as
Studio: `DeepResidualUNet(deeper=False)`, 223,355,921 parameters, 16 native planes in Studio's depth order, 128 px windows.

* `model.safetensors` — the **EMA** weights at the end of the run (decay 0.9995), SHA-256 `ec1f239a…0606f5`. Recommended.
* `model-raw.safetensors` — the raw final weights, SHA-256 `6f8c1c49…9db0e9` (equivalent on every metric below).
* `records/` — run configuration and data mixture, per-update log, every evaluation, check results, review images.

Code (private): [priyankaranke/vesuvius-studio-multiscroll-cpt](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt),
folder `studio-multiscroll-cpt-20261023/`.

## Results

**No PHerc0211 pixel was used for training.** The PHerc0211 evaluation is the nine labeled PHerc0211 Label Studio
segments (13 Oct 2026 labels, 6.6 M labeled pixels), Studio's published tiling (128 px window, central 96 px, stride 96),
AP over labeled pixels. Studio holdout = its fixed `841-expanded-layer-03-approved-r000001` holdout (core 32 px, 69
supervised patches; noisy).

| | PHerc0211 mean AP | PHerc0211 pixel-weighted AP | PHerc0211 pooled AP | Studio 841 holdout AP / AUC |
| --- | ---: | ---: | ---: | ---: |
| Studio epoch 9 (start) | 0.8306 | 0.8081 | 0.7951 | 0.7656 / 0.8304 |
| **this model (EMA, update 30,000)** | **0.8417** | **0.8222** | **0.8066** | **0.7791 / 0.8391** |
| raw weights, update 30,000 | 0.8411 | 0.8221 | 0.8067 | 0.7806 / 0.8407 |

Per segment (EMA vs Studio): 7 of 9 better.

| PHerc0211 segment | labeled px | Studio | this model | Δ |
| --- | ---: | ---: | ---: | ---: |
| 3305-r002-c01-manual-20261008-203736 | 1,651,082 | 0.7632 | 0.7800 | +0.0168 |
| Manual-03-da99cfea-p824-upper-long-villa | 796,434 | 0.8233 | 0.8439 | +0.0206 |
| P737m01st3-grown | 1,647,972 | 0.8301 | 0.8346 | +0.0045 |
| P737x1t-wplus1r-20261012 | 141,511 | 0.8459 | 0.8397 | −0.0062 |
| P824rowsX-20261012 | 820,958 | 0.8574 | 0.8527 | −0.0047 |
| S302c1x-r3t-20261012 | 182,879 | 0.9094 | 0.9144 | +0.0050 |
| pure-will-outer-…-c001-component00-part01 | 116,550 | 0.8477 | 0.8565 | +0.0088 |
| pure-will-outer-…-c001-component01-part00 | 43,619 | 0.8273 | 0.8462 | +0.0189 |
| r005-c01-manual-20261008-195107 | 1,202,216 | 0.7713 | 0.8076 | +0.0363 |

Evaluation curve (pixel-weighted PHerc0211 AP; raw / EMA): 2k 0.795 / – · 4k 0.811 / 0.812 · 6k 0.809 / 0.817 ·
8k 0.819 / 0.820 · 10k 0.812 / 0.823 · 14k 0.811 / 0.820 · 20k 0.821 / 0.822 · 26k 0.822 / 0.822 · 30k 0.822 / 0.822.
The gain arrives in the first ~8k updates and then holds; the final checkpoint was not selected on these labels
(the EMA at 10k scored 0.8228, within noise of the final 0.8222).

Limits: one scroll, nine segments whose labels are partly in the same physical regions as each other; AP measures
pixel ranking against sparse human labels, not readability. The `records/pherc0211-labeled-stock-vs-cpt.png` sheet shows
stock vs this model on three labeled segments; `records/crop1711-stock-vs-cpt.jpg` on the unlabeled stitched crop 1711
DinoVol revision (mean probability 0.129 vs 0.094 — this model responds over a somewhat wider area).

## Training

| | |
| --- | --- |
| Start | Studio epoch-9 weights (SHA-256 `ffffbf4f…dbd91145`), fresh fused AdamW |
| Batch | 64 per update (4 × 16, BF16, `torch.compile`): **20 Studio replay** (814 / 841 / 9B equal shares, Studio's scroll-balanced epoch order) + **44 other-scroll** examples drawn per update by scroll-group weight |
| Loss / augmentation | Studio's unchanged: masked 0.5 BCE + 0.5 Dice on the central 96 px; scroll-shift-v2 augmentation |
| Schedule | 30,000 updates; 600-update warm-up, cosine to 2 % of peak; peak LR 1e-4 for updates 0–2,167, then 6e-5 (the step-2,000 evaluation dipped and Studio-scroll training losses rose at 1e-4); EMA from update 2,167 |
| Data stages | updates 0–2,024: Expanded9 corpus rows only; from 2,024: + Reader v2/v3 label rows |
| Hardware | one NVIDIA GH200; ~1.0 s per update; 8 h 46 min wall clock including restarts and evaluations |

Other-scroll data (dense labels → Studio's state: unknown outside the label mask or 64 < label < 128, background ≤ 64, ink ≥ 128):

* **Expanded9 corpus** ([HF `paranke/vesuvius-expanded9-training-data`](https://huggingface.co/datasets/paranke/vesuvius-expanded9-training-data)),
  69 rows: PHercParis4 / Scroll 1 (29), PHerc0172 / Scroll 5 (13), PHerc0139 (14), PHerc1667 (6 + fragment 5), fragments 1–3, 6,
  PHerc0500P2 (1), PHercMAN5 (1). Expanded9's own 814 / 841 / 9B rows were left out (Studio replay covers them), as was
  `fragment4-exposed` (depth-order check ambiguous).
* **Reader v2 / Reader V3 labels** ([HF `domenicor046/reader-v2-labels`](https://huggingface.co/datasets/domenicor046/reader-v2-labels), MIT;
  Reader V3 kept Reader v2's data), 47 segments on the public native surface volumes: PHerc0139 (20) and PHerc0500P2 (18)
  at 9.362 µm / 113 keV — the PHerc0211 scan's setup — PHerc0343P (5) and PHerc0172 (4).

Group weights: PHerc0139 0.20 · PHercParis4 0.16 · PHerc0172 0.12 · PHerc1667 0.12 · PHerc0500P2 0.10 · PHerc0343P 0.06 ·
fragments 0.06 · PHercMAN5 0.02 (renormalized over present groups; stage 1 had PHerc0500P2 at 0.04 and no PHerc0343P).

Before training, the stock Studio scored higher AP on every included row with its stored plane order than reversed
(depth-order check), and the step-0 evaluation reproduced Studio's published holdout (0.7656 vs 0.7657).

## Inference

Exactly as Studio: `[16, H, W]` uint8 surface planes in Studio's depth order, `clip(raw, 0, 200) / 255`, 128 px windows.

```bash
hf download paranke/vesuvius-studio-multiscroll-cpt-20261023 --local-dir studio-ms
cd studio-ms && python -m pip install -r requirements.txt
python inference.py --input surface_16planes_uint8.npy --output ink_probability.npy --checkpoint model.safetensors --device cuda
```

The GH200 dashboards run it with the reference engine (`Engine('studio', ckpt, checkpoint_sha256=…)`), layer `studioms`.
