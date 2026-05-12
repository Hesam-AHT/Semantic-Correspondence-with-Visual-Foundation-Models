"""
configs/paths.py — Central configuration for all dataset, model weight, and results paths.

Set the environment variable USE_DRIVE=true to resolve paths relative to Google Drive
(for Colab usage), otherwise paths are resolved relative to this repository root.
"""
import os

BASE_DRIVE = '/content/drive/MyDrive/semantic_correspondence'
BASE_LOCAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USE_DRIVE = os.environ.get('USE_DRIVE', 'false').lower() == 'true'
BASE = BASE_DRIVE if USE_DRIVE else BASE_LOCAL

# ── Datasets ────────────────────────────────────────────────────────────────

SPAIR71K_ROOT   = os.path.join(BASE, 'data', 'SPair-71k')
SPAIR71K_PAIRS  = os.path.join(SPAIR71K_ROOT, 'PairAnnotation')
SPAIR71K_LAYOUT = os.path.join(SPAIR71K_ROOT, 'Layout')
SPAIR71K_IMAGES = os.path.join(SPAIR71K_ROOT, 'JPEGImages')

PF_PASCAL_ROOT = os.path.join(BASE, 'data', 'PF-Pascal')
PF_WILLOW_ROOT = os.path.join(BASE, 'data', 'PF-Willow')
AP10K_ROOT     = os.path.join(BASE, 'data', 'AP-10K')

# ── Model weights ────────────────────────────────────────────────────────────

WEIGHTS_DIR    = os.path.join(BASE, 'weights')
DINOV2_WEIGHTS = os.path.join(WEIGHTS_DIR, 'dinov2_vitb14_pretrain.pth')
DINOV3_WEIGHTS = os.path.join(WEIGHTS_DIR, 'dinov3_vitb16_pretrain.pth')
SAM_WEIGHTS    = os.path.join(WEIGHTS_DIR, 'sam_vit_b.pth')

FINETUNED_DIR    = os.path.join(WEIGHTS_DIR, 'finetuned')
DINOV2_FINETUNED = os.path.join(FINETUNED_DIR, 'dinov2_best.pth')
DINOV3_FINETUNED = os.path.join(FINETUNED_DIR, 'dinov3_best.pth')
SAM_FINETUNED    = os.path.join(FINETUNED_DIR, 'sam_best.pth')

LORA_DIR    = os.path.join(FINETUNED_DIR, 'lora')
DINOV2_LORA = os.path.join(LORA_DIR, 'dinov2_lora_best.pth')
DINOV3_LORA = os.path.join(LORA_DIR, 'dinov3_lora_best.pth')
SAM_LORA    = os.path.join(LORA_DIR, 'sam_lora_best.pth')

# ── Results ──────────────────────────────────────────────────────────────────

RESULTS_DIR    = os.path.join(BASE_LOCAL, 'results')
RESULTS_STEP1  = os.path.join(RESULTS_DIR, 'step1')
RESULTS_STEP2  = os.path.join(RESULTS_DIR, 'step2')
RESULTS_STEP2_ABLATIONS = os.path.join(RESULTS_STEP2, 'ablations')
RESULTS_STEP3  = os.path.join(RESULTS_DIR, 'step3')
RESULTS_STEP3_GRID = os.path.join(RESULTS_STEP3, 'grid_search')

RESULTS_STEP4_LORA     = os.path.join(RESULTS_DIR, 'step4', 'lora')
RESULTS_STEP4_MNN      = os.path.join(RESULTS_DIR, 'step4', 'mnn')
RESULTS_STEP4_ENSEMBLE = os.path.join(RESULTS_DIR, 'step4', 'ensemble')
RESULTS_STEP4_AP10K    = os.path.join(RESULTS_DIR, 'step4', 'ap10k')

LEARNED_WEIGHTS_PATH = os.path.join(RESULTS_STEP4_ENSEMBLE, 'learned_weights.json')
