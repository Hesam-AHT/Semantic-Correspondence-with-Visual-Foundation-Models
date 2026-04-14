# Semantic Correspondence with Visual Foundation Models

## Project Structure

```
semantic_correspondence/
├── data/
│   └── spair.py          ← SPair-71k dataset loader
├── models/
│   ├── extractors.py     ← DINOv2 / DINOv3 / SAM feature extractors
│   └── matcher.py        ← cosine similarity + argmax / soft-argmax
├── evaluation/
│   └── pck.py            ← PCK@T metric accumulator
├── evaluate.py           ← Stage 1 main script
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

### Download SPair-71k
```bash
wget http://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz
tar xzf SPair-71k.tar.gz
```

### Download SAM checkpoint
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## Stage 1: Training-Free Baseline

Run with DINOv2 (default):
```bash
python evaluate.py \
  --backbone dinov2 \
  --dinov2_variant vitb \
  --spair_root /path/to/SPair-71k \
  --image_size 840 \
  --split test
```

Run with SAM:
```bash
python evaluate.py \
  --backbone sam \
  --sam_checkpoint /path/to/sam_vit_b_01ec64.pth \
  --spair_root /path/to/SPair-71k \
  --image_size 1024 \
  --split test
```

Quick test on 50 pairs:
```bash
python evaluate.py \
  --backbone dinov2 \
  --spair_root /path/to/SPair-71k \
  --max_pairs 50 \
  --split test
```