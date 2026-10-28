# Studio ink model (PHerc0814 / PHerc0841 / PHerc0009B) — training stack

> **2026-10-23: Studio · multi-scroll continued pretraining** — Studio epoch 9 continued on twelve other scroll groups with
> 814 / 841 / 9B replay in every batch; PHerc0211 held-out AP 0.8081 → 0.8222 (pixel-weighted), Studio 841 holdout kept
> (0.7656 → 0.7791). See [`studio-multiscroll-cpt-20261023/`](studio-multiscroll-cpt-20261023/README.md). The rest of this
> README describes the parent Studio stack, unchanged.
>
## Train the complete model from scratch

```bash
bash reproduce.sh
```

Run this in a fresh clone. It trains Studio from random initialization for nine epochs (74,250 updates),
then trains on the other scrolls for 30,000 updates with 814/841/9B replay, and runs the final 3,000-update
soft-loss stage. Every stage uses the weights produced by the preceding stage; this command downloads no
pretrained model weights. It follows the recorded data, learning-rate and EMA transitions. Floating-point
results can differ across hardware; this is not a promise of identical weights.

### All training datasets

| Dataset | Used for |
| --- | --- |
| [814 / 841 / 9B training data](https://huggingface.co/datasets/paranke/vesuvius-814-841-9b-studio-training-data) | Initial scratch training and replay in the later stages: labels, masks and rendered CT tiles. |
| [Expanded9 training data](https://huggingface.co/datasets/paranke/vesuvius-expanded9-training-data) | 69 other-scroll rows, including Scroll 1, Scroll 5, PHerc0139, PHerc1667, PHerc0500P2, PHercMAN5 and exposed fragments. |
| [Reader v2/v3 labels](https://huggingface.co/datasets/domenicor046/reader-v2-labels) | 47 further segments from PHerc0139, PHerc0500P2, PHerc0343P and PHerc0172; the script downloads their CT from public S3. |

The script downloads all three datasets (about 120 GB), prepares the Reader rows and checks every required
row before training. Other-scroll data goes in `../expanded9`, `../reader-v2` and `../reader-rows` beside the
checkout. Missing rows or missing CT chunks stop the run. PHerc0211 is excluded from training.

Use Linux, `uv`, Hugging Face access to the datasets, a compatible NVIDIA GPU with at least 96 GB VRAM,
at least 128 GB host RAM and 200 GB free disk. The full run includes the initial scratch training and takes
longer than the roughly ten hours required by the two later stages alone. Install the pinned dependencies
via the script and authenticate with Hugging Face before downloading.

Re-running the same command resumes its saved stages. Use a separate clean clone for a different run;
existing training outputs are never automatically replaced. To run only the final stage from the published
intermediate weights, use `bash reproduce.sh softloss` (requires GitHub CLI access to the release).
Final output: `studio-softloss-20261023/checkpoints/model-ema.safetensors`.

The PHerc0211 evaluation set is optional and does not contribute training pixels; set `PHERC0211_EVAL_DATA`
to its local directory to reproduce those evaluation scores. Inference uses
`studio-multiscroll-cpt-20261023/hf/inference.py --stride 64 --blend hann` with the released soft-loss weights.

## Base Studio training details


Everything needed to prepare the training data, train the **Studio** ink model from random initialization, and run
inference. The model is the `DeepResidualUNet(deeper=False)` (223,355,921 parameters) trained on 2026-09-26/27 on the
Donor Studio sparse ink/background labels of PHerc0814, PHerc0841 and PHerc0009B (run
`three-donor-studio-sparse-scratch-20260926`). Training was paused for review after **9 of 10 planned epochs**
(74,250 of 82,500 optimizer updates); the epoch-9 weights are the published Studio model. No PHerc0211, Scroll 1 or
pretrained data was used.

## Where everything lives

| Location | Contents |
| --- | --- |
| [HF model `paranke/vesuvius-814-841-9b-resunet-studio-20260926`](https://huggingface.co/paranke/vesuvius-814-841-9b-resunet-studio-20260926) | Published epoch-9 `model.safetensors`, model card, `inference.py`, `model.py`, training metrics |
| [HF dataset `paranke/vesuvius-814-841-9b-studio-training-data`](https://huggingface.co/datasets/paranke/vesuvius-814-841-9b-studio-training-data) | Training data (per-dataset plans, labels, native CT tiles), frozen run records, resumable epoch-9 checkpoint, epoch-9 weights; geometry backup, renderer and released stacks needed only to re-render the CT |
| This repository | Data preparation, CT tile rendering, training, inference and verification code; frozen run records; pinned environment; the original run's per-update log (`reference/`) |

No credentials are stored here or needed by the code. The Hugging Face repositories are private for now: until they are
public, downloading needs a Hugging Face read token with access to them (put it in `.env` as `HF_TOKEN`, or run
`hf auth login`). Once they are public, no token is needed. Weights & Biases logging is optional and off by default.

## Requirements

- One NVIDIA GPU with at least 80 GB of memory: all 8,340 CT tiles and their labels are kept on the GPU (37 GB, plus a
  36 GB working margin the loader checks). The original run used an NVIDIA GH200 (96 GB), ~1 s per update.
- Python 3.11 and the exact packages in `requirements-training.txt` (PyTorch 2.13.0 + CUDA 13.0, cuDNN 9.20).
- On NVIDIA Grace Hopper (GH200, ATS addressing) the GPU memory is also a NUMA node of the operating system, and the
  page cache of large file reads can occupy it (it shows as used GPU memory with no process). The loader then fails
  its memory check. Before training, flush the page cache (`sync; echo 1 | sudo tee /proc/sys/vm/drop_caches`) or
  run under `numactl --membind=<CPU node>`.
- About 70 GB of disk for the training data (+2.7 GB for the resumable checkpoint, +2 GB for the re-render inputs).

## Reproduce the run

```bash
git clone https://github.com/priyankaranke/vesuvius-814-841-9b-studio-training.git studio && cd studio
uv venv --python 3.11.16 .venv
uv pip install --python .venv --index-strategy unsafe-best-match -r requirements-training.txt
#   (or: python3.11 -m venv .venv && .venv/bin/pip install -r requirements-training.txt)
cp .env.example .env               # optional settings; HF_TOKEN only while the data is private
.venv/bin/python download_data.py  # ~64 GB, every file checked against the Hub record of the pinned revision
cd donor-studio-resunet-scratch-20260926
../.venv/bin/python train.py       # the full 10-epoch scratch run; writes training.jsonl, checkpoints/, progress.json
```

- **Check that it is the same run** at any point: `../.venv/bin/python ../compare_runs.py training.jsonl`. The data
  order, sampler, augmentation draws, learning rate and contributing/empty example counts are deterministic and must
  match the original run's log (`reference/training-20260926.jsonl`) exactly; losses match closely (BF16 arithmetic and
  `torch.compile` are not bit-for-bit across machines). At step 0, `train.py` also checks that its data loader produces
  exactly the augmented CT and labels of the earlier packed loader (`SHARED_TILE_AUGMENTATION_EXACT`).
- **Short check:** `STUDIO_MAX_UPDATES=300 ../.venv/bin/python train.py`, then `compare_runs.py`.
- **Continue the paused run** from epoch 9 to epoch 10: `download_data.py --resume-checkpoint` places the original
  `checkpoints/resume.pt` (model, optimizer and RNG states); `train.py` resumes from it and refuses a different
  configuration (the config includes the SHA-256 of `data.json` and of the label snapshot).
- **Weights & Biases:** set `STUDIO_WANDB=1` (and `WANDB_ENTITY` / `WANDB_PROJECT`) in `.env`; credentials come from
  `wandb login`.

## Inference

`donor-studio-resunet-scratch-20260926/hf-epoch-9/` is the published inference package (`inference.py`, `model.py`):
`[16, H, W]` uint8 surface planes in model depth order, `clip(raw, 0, 200) / 255`, 128 px windows, central 96 px
output, stride 96. `verify_holdout.py` scores weights on the fixed holdout exactly as `train.py`'s evaluation.

## Re-render the training data from the public scans (optional)

The HF dataset already holds the rendered CT. To regenerate it: `download_data.py --all`; extract the geometry backup
next to the code (`tar xzf backup/remote-artifacts.tar.gz --wildcards 'remote/home/ubuntu/donor-spirals-20260921/*'
--strip-components=3`, same for `backup/geometry-masks.tar.gz`); then in `donor-studio-resunet-scratch-20260926/`:

1. `rebuild_select.py` re-derives every `prepared/<id>/` plan from the frozen label snapshot (`tar xf labels.tar`
   first) with the unchanged `prepare.select_one` and checks each against `data.json`.
2. `render_rebuild.py plan` / `run` restores the level-0 chunks each tile reads from the public scans (PHerc0814
   `20250804134230` 9.362 µm, PHerc0841 `20250821151531` 9.366 µm, PHerc0009B `20250521125136` 8.640 µm), checking each
   chunk's size and MD5 against the S3 object inventory in the backup, and calls the unchanged `render.render_row`
   (renderer image `ghcr.io/scrollprize/villa/volume-cartographer@sha256:bad516f6…` with the run's `vc_render_tifxyz`
   from `render-runtime/`; set `STUDIO_DOCKER` if docker needs `sudo`). Campaign tiles read 134 GiB of chunks.

## Changes from the original run's code (portability only)

The original run read absolute paths on its host, required that host's exclusive-GPU lease and logged to a fixed W&B
project. For reproduction elsewhere, without changing any training computation:

- `studio_env.py`: optional `.env` loading; `localize()` maps the frozen records' original paths (`/home/ubuntu/<folder>/…`)
  onto this checkout. The records themselves are unchanged, because `data.json` is hashed into the run configuration.
- `train.py`: W&B optional (`STUDIO_WANDB`), lease check only when a lease token is given, `STUDIO_MAX_UPDATES`.
- `sparse_data.py`, `prepare.py`, `render.py`, `render_rebuild.py`, `verify_holdout.py`: paths through `localize()`;
  renderer image by full reference; docker without `sudo` unless `STUDIO_DOCKER` says otherwise.
- Removed: `infer_viewer_rois.py` (an internal review tool, not part of the pipeline). The original files are in the
  first commit of this repository.

## Data rebuild check (2026-10-10)

All 199 datasets with samples were re-rendered (8,340 tiles, 32.6 GiB; `render_rebuild.py verify`: 0 incomplete).
`verify_holdout.py` scores the published epoch-9 weights on the rebuilt holdout exactly as `train.py`'s `evaluate()`
(eager instead of compiled): loss 0.51900, AP 0.76566, ROC-AUC 0.83043 on 1,334 patches (15,791 ink / 29,558 background
pixels) against the logged 0.51891 / 0.76560 / 0.83043. HF dataset revision `0bcee887fe8f` holds 1,457 files (67.1 GB).

## Fresh-machine reproduction (2026-10-10)

Commit `b21d83c` cloned fresh, a new environment from `requirements-training.txt`, and a fresh `download_data.py` of
dataset revision `ad919d28` (1,421 files, 64.4 GB, every file checked), on one GH200:

- `train.py` found the configuration identical to the original run (same `data.json` and label snapshot hashes) and
  printed `SHARED_TILE_AUGMENTATION_EXACT` (64 examples).
- Updates 1-300 against the original log (`compare_runs.py`: `REPRODUCED`): learning rate, contributing and empty
  example counts, offset and epoch identical on every update; loss median relative difference 1.8e-4 (max 7.9e-3),
  correlation 0.9992, mean 0.68184 against 0.68179.
- Resumed from the published `resume.pt`: continued at update 74,251 (epoch 10) on the scheduled learning rate. Its
  first 50 updates average loss 0.442, against 0.428 over the original's last 500 updates of epoch 9.

## Results at epoch 9 (fixed holdout, sparse labels)

Masked validation loss 0.519, AP 0.766, ROC-AUC 0.830 (central 32 px core; 15,791 ink / 29,558 background pixels).
These sparse metrics are not comparable to dense-label baselines and do not establish generalization to other scrolls.
