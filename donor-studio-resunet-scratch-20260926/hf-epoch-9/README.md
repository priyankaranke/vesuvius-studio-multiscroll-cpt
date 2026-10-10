---
library_name: pytorch
tags:
- vesuvius
- ink-detection
- 3d-unet
- micro-ct
- image-segmentation
---

# Vesuvius 814 / 841 / 9B — Studio sparse labels, epoch 9

Private research checkpoint from the September 26, 2026 Donor Studio run.
**Training is paused for review after 9 of the planned 10 epochs: 74,250 optimizer updates.**
This is the epoch 9 checkpoint, not a completed 10-epoch run or a validation-selected best checkpoint.
It is separate from the [September 12 baseline](https://huggingface.co/paranke/vesuvius-814-841-9b-resunet).

The unchanged 223,355,921-parameter `DeepResidualUNet(deeper=False)` was initialized
from scratch. Training uses newly saved Studio ink/background labels for PHerc0814,
PHerc0841 and PHerc0009B; unknown pixels do not contribute to BCE or Dice.
There are no Scroll 1 training samples and no pretrained initialization.

## Checkpoint and validation

`model.safetensors` preserves the checkpoint model tensors exactly, with their
original dtypes. Optimizer and random states remain in the private resumable
training checkpoint on the training machine; the inference package contains weights only.
See [model configuration](model-config.json) and [epoch 9 results](training-result.json)
for hashes, actual progress and the original planned configuration.

| Completed epoch | Mean training loss | Masked validation loss | Validation AP | Validation ROC-AUC |
|---|---:|---:|---:|---:|
| 7 | 0.451236 | 0.431282 | 0.769654 | 0.837063 |
| 8 | 0.440875 | 0.592532 | 0.744532 | 0.792395 |
| 9 | 0.433093 | 0.518912 | 0.765598 | 0.830425 |

Validation uses the fixed `841-expanded-layer-03-approved-r000001` holdout:
1,334 patches, of which 69 contain supervision in the central 96px loss window.
AP/AUC use the central 32px core, with 15,791 ink and 29,558 background pixels;
1,320,667 unknown pixels are excluded. Training uses augmentation; validation is fixed.
These sparse metrics are not directly comparable to the older baseline's dense labels.
Training loss decreased while validation remained uneven; these results do not
establish generalization to unseen scrolls or calibrated probabilities.

![Training and validation loss](loss-curves-epoch-9.png)

[W&B training run](https://wandb.ai/aranke-priyank99-personal/donor-s1-continued-training/runs/three-donor-studio-sparse-scratch-20260926)

## Download and inference

Authenticate with a Hugging Face account that has access to this private repository.
Install a suitable PyTorch build, then:

```bash
hf auth login
hf download paranke/vesuvius-814-841-9b-resunet-studio-20260926 --revision epoch-9 --local-dir studio-epoch-9
cd studio-epoch-9
python -m pip install -r requirements.txt
python inference.py --input surface_16planes_uint8.npy --output ink_probability.npy --checkpoint model.safetensors --device cuda --batch-size 1
```

Supply a `.npy` array of shape **[16,H,W]**, dtype **uint8**, with consecutive
surface-normal CT planes in the verified model depth order. Input preprocessing
is `clip(raw,0,200)/255`. Use the original rendering and intensity conversion
contract; do not silently cast 16-bit CT to uint8 or feed already normalized inputs.
Preserve source-specific physical pitch (approximately 9 micrometers), mapping,
normal sign and depth order, recorded in `training/data.json`.

The reused baseline helper applies 128x128 model windows, central 96x96 outputs,
stride 96, reflect padding and no test-time augmentation. Output is float32 [H,W]
sigmoid ink probability. Use `--amp fp32` where BF16 is unsupported. For direct
integration, instantiate `model.DeepResidualUNet(deeper=False)`, load the safetensors
state dict with `strict=True`, and pass preprocessed [N,1,16,128,128] tensors.

## Training recipe

There are 424,425 eligible training entries (154,300 distinct patches). Every entry
appears in each full epoch with its original multiplicity; replacement topups yield
176,000 draws per donor and 528,000 total draws, or 8,250 updates per epoch.
Effective batch is 64, with four microbatches of 16. The originally planned 82,500-update
learning-rate schedule remains unchanged at this pause.

The 256x256 parent windows determine eligibility; the actual model input is
16 planes x 128 x 128. A 352px measured parent supplies 224px augmentation context.
The original uncapped scroll-shift-v2 and gain/bias, D4 and depth-dropout ordering
are preserved. Sparse loss is per-example 0.5 masked BCE + 0.5 masked Dice (smooth=1)
on the central 96px; fully unknown examples are excluded from the batch denominator.

The run uses BF16, `torch.compile(dynamic=False)`, fused AdamW, peak learning rate
2e-4, weight decay 0.01 and gradient norm clipping at 1. Shared native CT tiles and
categorical masks reside in VRAM. Physical holdout exclusion is 2,000 micrometers
and covers the full measured context. The frozen label cutoff is September 26,
2026 at 06:52:57 UTC. Training source and snapshot metadata are included in `training/`;
these are archival source files requiring the original workspace and data.
Raw CT, label databases/images, meshes and prepared caches are not included.
