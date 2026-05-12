# Semantic Correspondence with Vision Foundation Models

Training-free and finetuned semantic correspondence using DINOv2, DINOv3, and SAM. Keypoints are matched by comparing dense patch features with argmax or windowed soft-argmax, with optional MNN verification and score-level ensemble fusion. Benchmarked on SPair-71k, PF-Pascal, PF-Willow, and AP-10K.

<img width="508" height="260" alt="semantic_correspondence" src="https://github.com/user-attachments/assets/c0e09637-21ab-49fe-852e-5d71d95323e1" />

## Pipeline

<img width="600" height="276" alt="window_softargmax" src="https://github.com/user-attachments/assets/254d180f-f513-4568-b6b3-8368cf0828be" />

1. Resize source and target images to model-specific resolution
2. Extract dense patch-token features (optionally multi-layer averaged + PCA-whitened)
3. For each source keypoint, compute cosine similarity against all target patches
4. Predict target keypoint via argmax or windowed soft-argmax; optionally verify with MNN
5. Evaluate PCK at thresholds 0.05 / 0.10 / 0.20

## Project Structure

```
semantic_correspondence/
├── src/                          # Core library
│   ├── features/
│   │   └── extractor.py          # Dense feature extraction (DINOv2/v3, SAM, PCA)
│   ├── matching/
│   │   └── strategies.py         # Argmax, windowed soft-argmax, MNN
│   ├── metrics/
│   │   └── pck.py                # PCK for SPair-71k, PF-Pascal, AP-10K
│   ├── lora/
│   │   └── lora.py               # LoRALinear, inject_lora, count_trainable_params
│   ├── datasets/
│   │   ├── spair_dataset.py      # SPair-71k PyTorch Dataset
│   │   ├── pf_pascal_dataset.py  # PF-Pascal Dataset
│   │   ├── pf_willow_dataset.py  # PF-Willow Dataset
│   │   └── ap10k_dataset.py      # AP-10K Dataset
│   └── models/
│       ├── dinov2/               # DINOv2 model code
│       ├── dinov3/               # DINOv3 model code
│       └── segment_anything/     # SAM model code
├── experiments/                  # Runnable experiment scripts
│   ├── evaluate.py               # Core evaluation loop + save_results
│   ├── evaluate_baseline.py      # Step 1: zero-shot baseline (PF-Pascal/Willow)
│   ├── finetune.py               # Step 2: block-unfreeze finetuning
│   ├── grid_search.py            # Step 3: K × temperature grid search
│   ├── evaluate_ensemble.py      # Step 4: ensemble (DINOv2+DINOv3+SAM)
│   └── report/
│       ├── generate_tables.py            # LaTeX comparison tables
│       ├── keypoint_images.py            # Keypoint visualisation figures
│       ├── generate_keypoint_analysis_figures.py
│       └── sensitivity_tables.py
├── tools/                        # SPair-71k devkit utilities
│   ├── visualize_pair_annotation.py
│   ├── visualize_image_annotation.py
│   └── report_statistics.py
├── configs/
│   └── paths.py                  # Centralised data / weight / results paths
├── notebooks/                    # Colab-ready step-by-step notebooks
│   ├── step1_baseline.ipynb      # Zero-shot evaluation (DINOv2, DINOv3, SAM)
│   ├── step2_finetune.ipynb      # Block-unfreeze finetuning + ablations
│   ├── step3_softargmax.ipynb    # Grid search: K × temperature
│   ├── step4a_lora.ipynb         # LoRA rank ablation
│   ├── step4b_mnn.ipynb          # MNN verification
│   ├── step4c_ensemble.ipynb     # Learned ensemble weights
│   └── step4d_ap10k.ipynb        # AP-10K cross-species evaluation
├── data/                         # Datasets (not tracked by git)
│   ├── SPair-71k/
│   ├── PF-Pascal/
│   ├── PF-Willow/
│   └── AP-10K/
├── weights/                      # Model weights (not tracked by git)
│   ├── dinov2_vitb14_pretrain.pth
│   ├── dinov3_vitb16_pretrain.pth
│   ├── sam_vit_b.pth
│   └── finetuned/
│       ├── dinov2_best.pth
│       ├── dinov3_best.pth
│       ├── sam_best.pth
│       └── lora/
├── results/                      # Experiment outputs (not tracked by git)
│   ├── step1/                    # Baseline results
│   ├── step2/                    # Finetuning results + ablations/
│   ├── step3/                    # Grid-search results
│   └── step4/                    # LoRA / MNN / ensemble / AP-10K results
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/YOUR_USER/semantic_correspondence.git
cd semantic_correspondence
pip install -r requirements.txt
```

### Model weights

| File | Source |
|------|--------|
| `weights/dinov2_vitb14_pretrain.pth` | [Meta DINOv2 releases](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth) |
| `weights/dinov3_vitb16_pretrain.pth` | Obtain from project maintainer |
| `weights/sam_vit_b.pth` | [Meta SAM releases](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) |

### Datasets

| Dataset | Default path | Notes |
|---------|-------------|-------|
| SPair-71k | `data/SPair-71k/` | Extract so `PairAnnotation/`, `Layout/`, `JPEGImages/` are direct children |
| PF-Pascal  | `data/PF-Pascal/` | |
| PF-Willow  | `data/PF-Willow/` | |
| AP-10K     | `data/AP-10K/`    | Needs `images/` and `annotations/` subdirs |

All paths are configured in [configs/paths.py](configs/paths.py). Set `USE_DRIVE=true` to resolve paths from Google Drive instead (Colab usage).

## Quick Start

### Notebooks (recommended for Colab)

Open the notebooks in order. Each notebook is self-contained — it mounts Drive, clones the repo, downloads weights, and runs the experiment end-to-end.

| Notebook | What it does |
|----------|-------------|
| [step1_baseline.ipynb](notebooks/step1_baseline.ipynb) | Zero-shot evaluation of DINOv2, DINOv3, SAM on SPair-71k |
| [step2_finetune.ipynb](notebooks/step2_finetune.ipynb) | Temperature / blocks / LR ablations + full finetuning |
| [step3_softargmax.ipynb](notebooks/step3_softargmax.ipynb) | K × temperature grid search for windowed soft-argmax |
| [step4a_lora.ipynb](notebooks/step4a_lora.ipynb) | LoRA rank ablation (r = 2, 4, 8, 16) vs block-unfreezing |
| [step4b_mnn.ipynb](notebooks/step4b_mnn.ipynb) | MNN verification (max_patch_dist ablation) |
| [step4c_ensemble.ipynb](notebooks/step4c_ensemble.ipynb) | Learned score-level ensemble of DINOv2 + DINOv3 + SAM |
| [step4d_ap10k.ipynb](notebooks/step4d_ap10k.ipynb) | Full pipeline on AP-10K with diagonal PCK |

### Scripts

```bash
# Zero-shot baseline on PF-Pascal
python experiments/evaluate_baseline.py

# Finetune DINOv3 on SPair-71k
python experiments/finetune.py

# Grid search over soft-argmax hyperparameters
python experiments/grid_search.py

# Ensemble evaluation (DINOv2 + DINOv3 + SAM)
python experiments/evaluate_ensemble.py

# Generate LaTeX tables from saved results
python experiments/report/generate_tables.py --output_dir tables/
```

## Methods

### Feature Extraction

- **DINOv2** ViT-B/14: images resized to 518×518; patch tokens from the last 1 or 3 blocks
- **DINOv3** ViT-B/16: images resized to 512×512; patch tokens from the last 1 or 3 blocks
- **SAM** ViT-B: images resized to 512×512; image-encoder feature maps (32×32)
- **Multi-layer averaging**: average patch tokens from the last N blocks before matching
- **PCA whitening**: compress D-dimensional features to 64 dimensions on a per-pair basis

### Matching Strategies

| Strategy | Description |
|----------|-------------|
| Argmax | Cosine-similarity argmax over all target patches |
| Windowed soft-argmax | Argmax to find window centre, then soft-argmax within K×K window |
| MNN (Mutual Nearest Neighbour) | Forward soft-argmax + backward argmax; fallback on mismatch |

### Finetuning

- Freeze all layers; unfreeze the last N transformer blocks and norm
- Per-pair cross-entropy loss: each source keypoint patch is treated as a query, the ground-truth target patch index is the class label
- Temperature-scaled cosine similarity logits (T ≈ 10–15)
- LinearLR warmup + CosineAnnealingLR; early stopping on val PCK@0.10

### LoRA

Low-rank adaptation injected into `qkv` and `proj` linear layers: `W' = W + (α/r) · B·A`. Rank ablation over r ∈ {2, 4, 8, 16} vs full block-unfreeze.

### Ensemble

Score-level fusion on a shared 32×32 SAM grid. Per-model weights are either fixed or learned via a softmax over trainable logits, optimised with cross-entropy on the SPair-71k training split.

## Evaluation Metrics

| Metric | Normalisation |
|--------|---------------|
| PCK@α (SPair-71k) | max(bbox width, bbox height) |
| PCK@α (PF-Pascal / PF-Willow) | Image diagonal |
| PCK@α (AP-10K) | Image diagonal |

α ∈ {0.05, 0.10, 0.20}.

## Acknowledgements

- [DINOv2](https://github.com/facebookresearch/dinov2) — Meta AI self-supervised ViT
- [Segment Anything](https://github.com/facebookresearch/segment-anything) — Meta AI SAM
- [SPair-71k](http://cvlab.postech.ac.kr/research/SPair-71k/) — POSTECH semantic correspondence benchmark
- [PF-Pascal / PF-Willow](https://github.com/juhongm999/hpf) — Proposal Flow benchmarks
- [AP-10K](https://github.com/AlexTheBad/AP-10K) — Animal pose dataset
