# Studio soft-prediction test (2026-10-23)

Priyank's question: Studio's maps are high quality but lack the diffuse, soft responses of Youssef's model
([`YoussefMoNader/ink-8um-v8in`](https://huggingface.co/YoussefMoNader/ink-8um-v8in)). Those weak responses are what a
human reinforces during iterative fine-tuning, so the question is whether they come from v8in's loss (and smaller
window), and whether we can get them on top of Studio's quality.

| | v8in | Studio (all versions) |
| --- | --- | --- |
| BCE | label smoothing 0.25: targets 0.25 background / 0.75 ink | hard 0/1 targets |
| Dice | unsmoothed, over the whole batch (`loss_mode="batch"`) | per example, smooth 1 |
| Window | 64 px, Gaussian-blended at stride 21 | 128 px in, central 96 px |

Evidence before this test: the 13 Oct PHerc0211 v8in LOO fine-tunes used Studio's loss; on all 10 held-out segments the
mean probability fell from 0.34–0.39 to 0.10–0.22 after 600 steps.

Host: GH200 192.222.59.124, beside the parent run (`studio-multiscroll-cpt-20261023/studio/studio-softloss-20261023/`),
donor venv `/home/ubuntu/donor-continued-training-20260910/.venv`. Log: `studio-multiscroll-cpt-20261023/logs/softloss-train.log`.

## Step 1: is the softness already in Studio's ranking? (`soft_check.py`)

AP does not change under any monotone remapping of a model's output. So if Studio · multi-scroll ranks the held-out
labels at least as well as v8in, the difference could be only the value scale, and a log-odds display would recover it.

Nine held-out PHerc0211 Label Studio segments (13 Oct labels). v8in maps are the LOO base predictions (official
64 px / stride 21 recipe); Studio · multi-scroll EMA (`ec1f239a…`) uses eval211's tiling (128 / 96 / stride 96) over every
support cell. Compared on labeled pixels inside v8in's coverage (`soft-check/result.json`):

| segment | v8in AP | Studio-MS AP | v8in ink median | Studio-MS ink median |
| --- | ---: | ---: | ---: | ---: |
| 3305-r002 | 0.605 | 0.785 | 0.52 | 0.83 |
| Manual-03 | 0.720 | 0.844 | 0.55 | 0.95 |
| P737m01st3 | 0.715 | 0.836 | 0.54 | 0.95 |
| P737x1t | 0.714 | 0.840 | 0.55 | 0.92 |
| P824rowsX | 0.712 | 0.853 | 0.56 | 0.95 |
| S302c1x | 0.720 | 0.914 | 0.53 | 0.89 |
| lower-outer c00p01 | 0.821 | 0.857 | 0.58 | 0.89 |
| lower-outer c01p00 | 0.789 | 0.846 | 0.54 | 0.95 |
| r005 | 0.605 | 0.808 | 0.48 | 0.75 |
| **pixel-weighted** | **0.670** | **0.824** | | |

- Studio · multi-scroll ranks the labeled ink better on every segment. Caveat: these labels may have been drawn while
  looking at Studio-family maps, which would favour Studio.
- v8in's whole range is compressed: background median 0.29–0.44, ink 90th percentile at most 0.62.
- Panels (`soft-check/panel-*.jpg`: CT, labels, v8in, Studio prob, Studio log-odds; display stretch 1–99.5 %): v8in is
  continuous but noisy and mostly not letter-shaped on these crops; Studio shows the letters crisply. Studio's log-odds
  view is not a usable soft map: below about p = 0.05 each 96 px window sits on its own baseline (blocky squares), so the
  faint range is window-level offset rather than stroke evidence. A display change alone does not give soft predictions.

## Step 2: Studio · multi-scroll continued with v8in's loss (`train_softloss.py`)

Same data, replay, mixture, augmentation, model and window as the parent run. Changed only: start from the parent's EMA
weights; loss = 0.5 BCE with label smoothing 0.25 + 0.5 microbatch Dice (hard targets, smooth 0, skipped without labeled
ink); 3,000 updates, warm-up 200, peak 6e-5, cosine to 2 %, EMA 0.998, evaluation every 1,000. Step-0 evaluation
reproduces the parent exactly (PHerc0211 pixel-weighted AP 0.8222).

Finished 21:50 UTC (3,000 updates, ~1 h). Final weights on .124 in `checkpoints/`: EMA `model-ema.safetensors`
(`96b32628…`, used below), raw `model.safetensors` (`12ba5a68…`), plus `step-/ema-001000…003000`.

| | PHerc0211 pixel-weighted AP | Studio 841 holdout AP | mean p, labeled pixels |
| --- | ---: | ---: | ---: |
| start (Studio · multi-scroll EMA) | 0.8222 | 0.7791 | 0.45 |
| update 1,000 (EMA) | 0.8173 | 0.7538 | 0.52 |
| update 2,000 (EMA) | 0.8202 | 0.7448 | 0.52 |
| update 3,000 (EMA) | 0.8226 | 0.7536 | 0.52 |

`soft_check.py` on the same nine segments and pixels as step 1 (`soft-check/result-s96.json`, `result-s48.json`):

| pixel-weighted AP | stride 96 (eval211 tiling) | stride 48, Hann-blended logits |
| --- | ---: | ---: |
| v8in (its own recipe) | 0.670 | 0.670 |
| Studio · multi-scroll | 0.8238 | 0.8468 |
| soft-loss | 0.8238 | 0.8457 |

- Ranking on PHerc0211 is unchanged; Studio's own 841 holdout dropped 0.779 → 0.754 (69 patches, noisy, but a real
  ranking change).
- The output scale changed as intended. Background log-odds q10/median went from about −12/−2.9 to −1.7/−1.3 (background
  median p ≈ 0.21–0.40; the window-level collapse to −12 is gone); ink median p 0.85–0.90, 90th percentile ≤ 0.93.
- Panels (`soft-check/panel-s48-*.jpg`, blended): the soft-loss map has a smooth, even background with graded mid-tone
  blobs and stroke extensions that Studio · multi-scroll shows as zero, and letters as crisp as before. v8in stays noisy.
  Whether those mid-tones are real faint ink is not tested here: that needs review on unlabeled regions or a fine-tuning
  round.
- Overlap blending (stride 48, Hann) adds ~0.023 AP to both Studio models over eval211's non-overlapping stride 96.

Conclusion: the loss alone gives graded, Youssef-style soft maps on Studio's architecture and data without losing
PHerc0211 ranking. A full v8in-architecture replication (step 3) is not needed for this goal; the 64 px window arm was
not tested. Keep label smoothing in any fine-tuning round started from these weights, or the softness disappears again
(as in the 13 Oct v8in LOO fine-tunes).

## Disk on .124 (23 Oct)

Freed for a possible step 3 (Youssef architecture on this corpus), free space 209 GiB → 2.0 TB:

- reproducible caches: DinoVol fine-tune CT cache (101 GiB), umbilicus-studio chunk cache (82 GiB; the annotation itself
  was backed up first to the SSD `umbilicus-backup-20261023/` and the Mac), 180 GiB of cold payloads from the shared CT
  cache (its reclaim API; the most recent ~62 GB kept);
- render stacks of the 79 superseded m20000/m30000 spiral windings (434.6 GiB, predictions kept);
- chosen by Priyank: DinoVol fine-tune checkpoints (167 GiB), spiral milestones 500–30,000 and the spacing trial
  (146 GiB), `ct17.npy` of the 120 final windings (523 GiB, byte-identical rebuild from their TIFFs), DinoVol old-fit work
  (71 GiB), public input copies (41 GiB).

Each deletion is recorded in the owning run's notes with a GH200 receipt.
