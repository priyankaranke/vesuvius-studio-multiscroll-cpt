# Studio ink model (PHerc0814 / PHerc0841 / PHerc0009B) — training stack

Everything needed to prepare the training data, train the **Studio** ink model from random initialization, and run
inference. The model is the `DeepResidualUNet(deeper=False)` (223,355,921 parameters) trained on 2026-09-26/27 on the
Donor Studio sparse ink/background labels of PHerc0814, PHerc0841 and PHerc0009B (run
`three-donor-studio-sparse-scratch-20260926`). Training was paused for review after **9 of 10 planned epochs**
(74,250 of 82,500 optimizer updates); the epoch-9 weights are the published Studio model. No PHerc0211, Scroll 1 or
pretrained data was used.

## Where everything lives (all private)

| Location | Contents |
| --- | --- |
| [HF model `paranke/vesuvius-814-841-9b-resunet-studio-20260926`](https://huggingface.co/paranke/vesuvius-814-841-9b-resunet-studio-20260926) | Published epoch-9 `model.safetensors`, model card, `inference.py`, `model.py`, training metrics |
| [HF dataset `paranke/vesuvius-814-841-9b-studio-training-data`](https://huggingface.co/datasets/paranke/vesuvius-814-841-9b-studio-training-data) | Frozen label snapshot, per-dataset sample/tile plans, native CT training tiles, campaign geometry and render recipes, renderer binary, epoch-9 weights and the resumable epoch-9 training checkpoint |
| This repository | Data preparation, CT tile rendering, training and inference code, frozen run records (`config.json`, `data.json`, `label-snapshot.json`) |

Reviewers need read access to the two HF repositories and this repository; authenticate with an HF read token
(`hf auth login`). Only ordinary Hub file downloads are used.

## Layout

The code keeps the folder layout of the training host (`/home/ubuntu/...`), because the scripts find their
dependencies as sibling folders (`ROOT.parent / ...`). Clone into an empty working folder and put the data next to it:

```text
<work>/
  donor-studio-resunet-scratch-20260926/   the run: prepare.py, render.py, train.py, sparse_data.py, finalize.py, ...
    labels/                                 frozen label snapshot            (HF dataset: labels.tar; tar xf)
    prepared/<dataset>/                     sample/tile plans + ct-tiles.npy (HF dataset: prepared/)
    hf-epoch-9/                             inference package of the published weights
  donor-continued-training-20260910/        accepted native sampler adapters, volume catalog
  donor-original-recipe-20260910/           original patch eligibility (coordinates)
  donor-resunet-scratch-20260912/           Sept 12 baseline: packed_data.py (step-0 equivalence check), data.json
  pherc1447-architecture-study-20260906/    small_unets.py (architecture), scroll_augment.py (augmentation)
  donor-labels-20260906/expansion/          volume_sampling.py, fast_volume_sampling.py (original-donor sampling)
  donor-spirals-20260921/                   campaign geometry and scan configs (HF dataset: geometry/; only to re-render)
```

Several records contain absolute `/home/ubuntu/...` paths. Either use `/home/ubuntu` as `<work>`, or create a symlink
`/home/ubuntu/donor-studio-resunet-scratch-20260926 -> <work>/donor-studio-resunet-scratch-20260926`.

## 1. Data

`donor-studio-resunet-scratch-20260926/data.json` is the frozen training-set definition (`data_sha256`
`01eecf6d…a7f`, label snapshot `fe9c1dce…c90`, cutoff 2026-09-26 05:37 UTC): 210 labeled datasets, 202 eligible,
424,425 training entries (154,300 distinct 128 px patches; PHerc0814 175,956 / PHerc0841 160,533 / PHerc0009B 87,936)
and a fixed PHerc0841 holdout `841-expanded-layer-03-approved-r000001` with a conservative 2 mm exclusion.
Training reads, per dataset, `prepared/<id>/` — `labels.npy` (0 unknown, 1 background, 2 ink), `coordinates.npy`,
`entry-to-patch.npy`, `tiles.npy` (512 px tile origins) and `ct-tiles.npy` (`[N, 512, 512, 16]` uint8, already in
model depth order) — about 32.6 GiB of CT tiles in total (8,340 tiles).

Download everything into the layout above (the HF paths mirror this repository):

```bash
hf download paranke/vesuvius-814-841-9b-studio-training-data --repo-type dataset --local-dir <work>
(cd <work>/donor-studio-resunet-scratch-20260926 && tar xf labels.tar)   # only needed to re-derive the plans
```

To regenerate instead (this is how the dataset's CT tiles were rebuilt on 2026-10-10, after the run's original
`prepared/` folders and local scan copies had been deleted):

1. `rebuild_select.py` — re-derives every `prepared/<id>/` plan from the frozen label snapshot exactly as
   `prepare.py select` did, and checks each dataset against `data.json` (samples, patches, tiles, supervised ink and
   background, holdout exclusion) without rewriting `data.json`.
2. `render_rebuild.py plan` / `run` — restores the level-0 chunks each tile reads (the renderer's own chunk plan;
   every chunk checked against the size and MD5 in the backup's S3 object inventory) into the scan's local copy and
   calls the unchanged `render.render_row`. Needs the geometry backup restored as `donor-spirals-20260921/`
   (HF dataset `backup/`; extract `remote/home/ubuntu/donor-spirals-20260921/*` from `remote-artifacts.tar.gz` and
   `geometry-masks.tar.gz` with `--strip-components=3`). Campaign tiles read 134 GiB of chunks (PHerc0814 64.7,
   PHerc0841 54.5, PHerc0009B 14.5).
   `render.py` itself renders each 512 px tile from the public scans with the accepted samplers: campaign surfaces through
   `vc_render_tifxyz` (image `ghcr.io/scrollprize/villa/volume-cartographer@sha256:bad516f6…`, 17 native planes at the
   scan pitch, 16 kept in model depth order); original donors through `volume_sampling` / `fast_volume_sampling`.
   Scans: PHerc0814 `20250804134230` (9.362 µm), PHerc0841 `20250821151531` (9.366 µm), PHerc0009B `20250521125136`
   (8.640 µm), all from `s3://vesuvius-challenge-open-data`. The native CT chunks a tile needs must be staged locally
   first (`render.py` runs the renderer with `--chunks-already-staged`).

## 2. Training

```bash
cd donor-studio-resunet-scratch-20260926
WANDB_MODE=offline python train.py
```

Recipe (also in `config.json` and the model card): scratch initialization, 10 epochs, 528,000 draws/epoch (176,000
per donor), effective batch 64 (4 x 16), BF16, `torch.compile(dynamic=False)`, fused AdamW, peak LR 2e-4, weight decay
0.01, gradient clip 1.0; 16 x 128 x 128 inputs from a 352 px measured parent with 224 px augmentation context;
scroll-shift-v2 / gain-bias / D4 / depth-dropout augmentation; sparse loss 0.5 masked BCE + 0.5 masked Dice on the
central 96 px (unknown pixels contribute nothing). Seeds: training 2026090613, augmentation 2026090719, sampler
2026091201. The original run used one GH200 (peak 61.8 GB GPU memory, ~1 s per update). To continue the paused run to
epoch 10, place the HF dataset's `checkpoints/resume.pt` in `donor-studio-resunet-scratch-20260926/checkpoints/`.
A rerun preserves the recipe and seeds but is not promised to reproduce bit-identical weights.

## 3. Inference

See `donor-studio-resunet-scratch-20260926/hf-epoch-9/README.md`: `[16, H, W]` uint8 surface planes in model depth
order, `clip(raw, 0, 200) / 255`, 128 px windows, central 96 px output, stride 96.

## Data rebuild check (2026-10-10)

All 199 datasets with samples were re-rendered (8,340 tiles, 32.6 GiB; `render_rebuild.py verify`: 0 incomplete).
`verify_holdout.py` scores the published epoch-9 weights on the rebuilt holdout exactly as `train.py`'s `evaluate()`
(eager instead of compiled): loss 0.51900, AP 0.76566, ROC-AUC 0.83043 on 1,334 patches (15,791 ink / 29,558 background
pixels) against the logged 0.51891 / 0.76560 / 0.83043. HF dataset revision `0bcee887fe8f` holds 1,457 files (67.1 GB).

## Results at epoch 9 (fixed holdout, sparse labels)

Masked validation loss 0.519, AP 0.766, ROC-AUC 0.830 (central 32 px core; 15,791 ink / 29,558 background pixels).
These sparse metrics are not comparable to dense-label baselines and do not establish generalization to other scrolls.
