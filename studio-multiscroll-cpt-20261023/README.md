# Studio · multi-scroll continued pretraining (2026-10-23)

Continued pretraining of the [Studio ink model](https://huggingface.co/paranke/vesuvius-814-841-9b-resunet-studio-20260926)
(PHerc0814 / PHerc0841 / PHerc0009B sparse Studio labels, epoch 9) on labeled surfaces of **twelve other scroll groups**,
with **Studio's own 814 / 841 / 9B data replayed in every batch** so its original skill is kept. Goal: close the gap
Studio shows on PHerc0211. No PHerc0211 pixel is used for training; the nine labeled PHerc0211 Label Studio segments
are the held-out evaluation.

## Recipe

| | |
| --- | --- |
| Start | Studio epoch-9 `model.safetensors` (SHA-256 `ffffbf4f…dbd91145`), fresh fused AdamW |
| Model | unchanged `DeepResidualUNet(deeper=False)`, 223,355,921 parameters, 16 native planes, 128 px window |
| Batch | 64 examples per update (4 × 16, BF16, `torch.compile`), **20 Studio replay** (814 / 841 / 9B in equal shares, Studio's scroll-balanced epoch order consumed continuously) **+ 44 other-scroll examples** drawn per update from scroll groups (weights below) |
| Loss / augmentation | Studio's, unchanged: masked 0.5 BCE + 0.5 Dice on the central 96 px averaged over contributing examples; `sparse_data.augment` (scroll-shift-v2) |
| Schedule | 30,000 updates; peak LR 1e-4, 600-update linear warm-up, cosine to 2e-6; weight decay 0.01; gradient clip 1.0 |
| Evaluation | every 2,000 updates: PHerc0211 held-out AP (`eval211.py`) and Studio's fixed 841 holdout AP; weights exported at each evaluation |
| Hardware | one NVIDIA GH200 (96 GB): Studio's 8,340 tiles resident in VRAM (34.6 GiB), the other scrolls in a host-RAM atlas (~37 GB stage 1, ~61 GB stage 2); ~1.0 s per update |

### Other-scroll data

Dense labels become Studio's categorical state: unknown outside the label's mask (and for soft labels strictly between
64 and 128), background ≤ 64, ink ≥ 128 (`multiscroll_data.categorical`).

* **Expanded9 corpus** ([HF `paranke/vesuvius-expanded9-training-data`](https://huggingface.co/datasets/paranke/vesuvius-expanded9-training-data)):
  every row except Expanded9's donor rows (814 / 841 / 9B are covered by Studio's own replay), `pherc0814-46527` (same
  reason) and `fragment4-exposed` (depth-order check ambiguous): 69 rows — PHercParis4 / Scroll 1 (29), PHerc0172 /
  Scroll 5 (13), PHerc0139 (14), PHerc1667 (6 + fragment 5), fragments 1–3 and 6, PHerc0500P2 (1), PHercMAN5 (1).
* **Reader v2 / Reader V3 labels** ([HF `domenicor046/reader-v2-labels`](https://huggingface.co/datasets/domenicor046/reader-v2-labels), MIT;
  Reader V3 kept Reader v2's data and labels), from the restart after the step-2000 evaluation: 47 segments — PHerc0139
  (20) and PHerc0500P2 (18) on the 9.362 µm / 113 keV native scans (the setup of the PHerc0211 scan), PHerc0343P (5, 8.64 µm)
  and PHerc0172 (4, capped at 700 patches each). `prepare_reader.py` builds them from the public S3 surface volumes (only the
  chunks around labeled patches; planes `label_z-8 … label_z+7`). Reader's 814 / 9B segments are not used (Studio replay).

Group weights (renormalized over the groups present; stage 1 had no PHerc0343P and a single PHerc0500P2 row at 0.04):
PHerc0139 0.20 · PHercParis4 0.16 · PHerc0172 0.12 · PHerc1667 0.12 · PHerc0500P2 0.10 · PHerc0343P 0.06 · fragments 0.06 · PHercMAN5 0.02.

### Checks before training

* `check_orientation.py`: the stock Studio scores higher AP on every included row with its stored plane order than with
  the order reversed (e.g. PHerc0139 0.43 vs 0.18, Scroll 1 0.59 vs 0.24, Reader PHerc0139 0.53 vs 0.21).
* The step-0 evaluation inside `train_cpt.py` reproduces stock Studio exactly: PHerc0211 pixel-weighted AP 0.8081,
  Studio 841 holdout AP 0.7656 (published 0.7657).

## Files

* `train_cpt.py` — the training loop (resumable; exits with code 75 after a checkpoint on SIGTERM).
* `run_cpt.sh` — keeps `train_cpt.py` running (re-acquires the GH200 shared GPU lease after a manual round preempted it).
* `multiscroll_data.py` — host-RAM tile atlas for Expanded9-format rows and per-group samplers.
* `mixture.json` — rows and group weights (the stage-2 version; the stage history is in `config.json` / `resume.pt`).
* `eval211.py` — PHerc0211 held-out AP; `check_orientation.py` — depth-order check; `gpu_lease.py` — the host's lease.
* `prepare_reader.py` — Reader v2 label rows from the public data.
* `compare211.py` — review sheet; `publish_hf.py` — HF upload; `hf/` — model card and inference package; `reference/` — this run's records.
* `dashboard/` — `infer_studioms.py` (dashboard-recipe inference with these weights) and `add_studioms_layer.py`
  (adds the layer to a manual-revision viewer).

## Results (update 30,000; training finished 2026-10-23 15:48 UTC, 8 h 46 min)

No PHerc0211 pixel was used for training. PHerc0211 = the nine labeled PHerc0211 Label Studio segments (13 Oct labels,
6.6 M labeled pixels), Studio's published tiling, AP over labeled pixels. Studio holdout = its fixed 841 holdout (69
supervised patches; noisy).

| | PHerc0211 mean AP | pixel-weighted AP | pooled AP | Studio 841 holdout AP / AUC |
| --- | ---: | ---: | ---: | ---: |
| Studio epoch 9 (start) | 0.8306 | 0.8081 | 0.7951 | 0.7656 / 0.8304 |
| **EMA weights (published `model.safetensors`)** | **0.8417** | **0.8222** | **0.8066** | **0.7791 / 0.8391** |
| raw weights | 0.8411 | 0.8221 | 0.8067 | 0.7806 / 0.8407 |

7 of 9 segments improve: r005 +0.036, Manual-03 +0.021, pure-will c01 +0.019, 3305 +0.017, pure-will c00 +0.009,
S302 +0.005, P737m01st3 +0.005; P737x1t −0.006, P824rowsX −0.005 (table in the HF model card; every evaluation in
`reference/evaluations.json`). Pixel-weighted PHerc0211 AP, raw / EMA: 2k 0.795 / – · 4k 0.811 / 0.812 · 6k 0.809 / 0.817 ·
8k 0.819 / 0.820 · 10k 0.812 / 0.823 · 20k 0.821 / 0.822 · 30k 0.822 / 0.822 — the gain arrives in the first ~8k
updates and holds; the final checkpoint was not selected on these labels. Review sheets: `reference/pherc0211-labeled-stock-vs-cpt.png`
(three labeled segments) and `reference/crop1711-stock-vs-cpt.jpg` (the unlabeled stitched crop 1711 DinoVol revision;
mean probability 0.129 vs 0.094 — the new model responds over a somewhat wider area).

Run notes: at peak LR 1e-4 the step-2,000 evaluation dipped (PHerc0211 0.795, holdout 0.734) and Studio-scroll training
losses rose (814 0.51 → 0.59), so from update 2,167 the peak LR is 6e-5 with an EMA (decay 0.9995); `config.json`
`lr_history` / `data_history` record both restarts. The run was restarted twice by design (stage-2 data at 2,024; LR/EMA at
2,167), each time from the exact resumable state.

Weights: [HF `paranke/vesuvius-studio-multiscroll-cpt-20261023`](https://huggingface.co/paranke/vesuvius-studio-multiscroll-cpt-20261023)
(model card and records; the weight files are pending because the HF account's private storage quota was reached on
2026-10-23) and this repository's [GitHub release `studio-multiscroll-cpt-20261023`](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/releases/tag/studio-multiscroll-cpt-20261023):
EMA `ec1f239a9764a6f5d5ed8ec5c217db07cc6c524ededa2467adbc4afd0c0606f5`, raw `6f8c1c49bbd11765ecdd178cbd0aa8c4ad34062bf8a6b1bd05b9d9f7129db0e9`.

## Reproduce

1. `../download_data.py` (Studio data) and `hf download paranke/vesuvius-expanded9-training-data --repo-type dataset --local-dir ../../expanded9`.
2. `hf download domenicor046/reader-v2-labels --repo-type dataset --local-dir ../../reader-v2 && tar -xf ../../reader-v2/reader-v2-dense-native-labels.tar -C ../../reader-v2`,
   then `python prepare_reader.py --labels ../../reader-v2/reader-v2-labels --manifest ../../reader-v2/manifest.csv --out ../../reader-rows`
   (~4.5 min, 24 GB; deterministic patch selection).
3. Studio epoch-9 weights to `../../weights/vesuvius-814-841-9b-resunet-studio-20260926/model.safetensors` (or set `CPT_BASE`).
4. `PY=python ./run_cpt.sh` (one 96 GB GPU; `STUDIO_GPU_LEASE=0` off the GH200 hosts). To replay this run exactly, run with
   stage-1 rows (`mixture.json` minus `reader-rows`) and `CPT_PEAK_LR=1e-4` to update 2,024, then the stage-2 `mixture.json`
   to 2,167, then the defaults.
5. `python eval211.py --checkpoint checkpoints/model-ema.safetensors` for the PHerc0211 numbers.
