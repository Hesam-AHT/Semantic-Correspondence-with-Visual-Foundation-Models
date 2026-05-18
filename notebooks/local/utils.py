"""
notebooks/local/utils.py

Shared utilities for local notebook runs.
Handles progress tracking, path resolution, dataset downloads,
float16 helpers, and NaN-safe cosine similarity.
"""

import os
import json
import sys
import requests
import zipfile
import tarfile
import shutil
from pathlib import Path
from tqdm import tqdm


# ── Repo root resolution ──────────────────────────────────────────────────────

def get_repo_root():
    """
    Returns the absolute path to the repo root.
    Works whether the notebook is run from notebooks/local/ or the repo root.
    """
    current = Path(os.path.abspath('.')).resolve()
    for parent in [current] + list(current.parents):
        if (parent / 'setup.py').exists() or (parent / 'src').exists():
            return str(parent)
    raise RuntimeError(
        "Could not find repo root. "
        "Make sure you are running from inside the repository."
    )


# ── Path config ───────────────────────────────────────────────────────────────

def get_paths(repo_root=None):
    """
    Returns a dict of all standard paths used across local notebooks.
    All paths are absolute.
    """
    root = repo_root or get_repo_root()
    return {
        'repo_root':    root,
        'data':         os.path.join(root, 'data'),
        'weights':      os.path.join(root, 'weights'),
        'finetuned':    os.path.join(root, 'weights', 'finetuned'),
        'lora':         os.path.join(root, 'weights', 'finetuned', 'lora'),
        'results':      os.path.join(root, 'results'),
        'step1':        os.path.join(root, 'results', 'step1'),
        'step2':        os.path.join(root, 'results', 'step2'),
        'step2_abl':    os.path.join(root, 'results', 'step2', 'ablations'),
        'step3':        os.path.join(root, 'results', 'step3'),
        'step3_grid':   os.path.join(root, 'results', 'step3', 'grid_search'),
        'step4_lora':   os.path.join(root, 'results', 'step4', 'lora'),
        'step4_mnn':    os.path.join(root, 'results', 'step4', 'mnn'),
        'step4_ens':    os.path.join(root, 'results', 'step4', 'ensemble'),
        'step4_ap10k':  os.path.join(root, 'results', 'step4', 'ap10k'),
        'spair71k':     os.path.join(root, 'data', 'SPair-71k'),
        'pfpascal':     os.path.join(root, 'data', 'PF-Pascal'),
        'pfwillow':     os.path.join(root, 'data', 'PF-Willow'),
        'ap10k':        os.path.join(root, 'data', 'AP-10K'),
        'dinov2_w':     os.path.join(root, 'weights', 'dinov2_vitb14_pretrain.pth'),
        'dinov3_w':     os.path.join(root, 'weights', 'dinov3_vitb16_pretrain.pth'),
        'sam_w':        os.path.join(root, 'weights', 'sam_vit_b.pth'),
        'dinov2_ft':    os.path.join(root, 'weights', 'finetuned', 'dinov2_best.pth'),
        'dinov3_ft':    os.path.join(root, 'weights', 'finetuned', 'dinov3_best.pth'),
        'sam_ft':       os.path.join(root, 'weights', 'finetuned', 'sam_best.pth'),
        'dinov2_lora':  os.path.join(root, 'weights', 'finetuned', 'lora', 'dinov2_lora_best.pth'),
        'dinov3_lora':  os.path.join(root, 'weights', 'finetuned', 'lora', 'dinov3_lora_best.pth'),
        'sam_lora':     os.path.join(root, 'weights', 'finetuned', 'lora', 'sam_lora_best.pth'),
        'learned_w':    os.path.join(root, 'results', 'step4', 'ensemble', 'learned_weights.json'),
    }


def create_folders(paths):
    """Create all result and data folders if they do not exist."""
    folder_keys = [
        'data', 'weights', 'finetuned', 'lora',
        'step1', 'step2', 'step2_abl', 'step3', 'step3_grid',
        'step4_lora', 'step4_mnn', 'step4_ens', 'step4_ap10k',
        'spair71k', 'pfpascal', 'pfwillow', 'ap10k',
    ]
    for key in folder_keys:
        os.makedirs(paths[key], exist_ok=True)
    print("All folders ready.")


# ── Progress tracking ─────────────────────────────────────────────────────────

def load_progress(progress_path):
    """Load progress dict from JSON. Returns empty dict if file does not exist."""
    if os.path.exists(progress_path):
        with open(progress_path, 'r') as f:
            return json.load(f)
    return {}


def save_progress(progress, progress_path):
    """Save progress dict to JSON."""
    with open(progress_path, 'w') as f:
        json.dump(progress, f, indent=2)


def is_done(progress, backbone, stage, key):
    """
    Check if a specific ablation combination is already done.

    Args:
        progress:  dict loaded from load_progress()
        backbone:  'dinov2', 'dinov3', 'sam'
        stage:     'temp_ablation', 'blocks_ablation', 'lr_ablation', 'final_training'
        key:       the specific value (e.g. 15 for temperature, 2 for n_blocks)

    Returns:
        True if already done, False otherwise
    """
    try:
        return str(key) in [str(x) for x in progress[backbone][stage]['done']]
    except KeyError:
        return False


def mark_done(progress, backbone, stage, key, progress_path):
    """Mark a specific ablation combination as done and save immediately."""
    if backbone not in progress:
        progress[backbone] = {}
    if stage not in progress[backbone]:
        progress[backbone][stage] = {'done': []}
    if str(key) not in [str(x) for x in progress[backbone][stage]['done']]:
        progress[backbone][stage]['done'].append(key)
    save_progress(progress, progress_path)


def init_progress(backbones, stages_config):
    """
    Initialize a fresh progress dict.

    Args:
        backbones:     list of backbone names
        stages_config: dict of {stage_name: [list of values]}

    Example:
        init_progress(
            ['dinov2', 'dinov3', 'sam'],
            {
                'temp_ablation':   [1, 5, 10, 15],
                'blocks_ablation': [1, 2, 3, 4],
                'lr_ablation':     ['5e-5', '1e-4', '2e-4'],
                'final_training':  ['run'],
            }
        )
    """
    progress = {}
    for backbone in backbones:
        progress[backbone] = {}
        for stage, values in stages_config.items():
            progress[backbone][stage] = {
                'done':    [],
                'pending': [str(v) for v in values]
            }
    return progress


# ── Download helpers ──────────────────────────────────────────────────────────

def download_file(url, dest_path, desc=None):
    """
    Download a file from url to dest_path with a tqdm progress bar.
    Skips if dest_path already exists.
    """
    if os.path.exists(dest_path):
        print(f"Already exists: {dest_path} — skipping download")
        return
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"Downloading {desc or url} ...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get('content-length', 0))
    with open(dest_path, 'wb') as f, tqdm(
        total=total, unit='B', unit_scale=True, desc=desc or 'Downloading'
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"Saved to {dest_path}")


def extract_zip(zip_path, extract_to, remove_zip=True):
    """Extract a zip file and optionally delete it after."""
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)
    if remove_zip:
        os.remove(zip_path)
    print(f"Extracted to {extract_to}")


def extract_tar(tar_path, extract_to, remove_tar=True):
    """Extract a tar.gz file and optionally delete it after."""
    print(f"Extracting {tar_path} ...")
    with tarfile.open(tar_path, 'r:gz') as t:
        t.extractall(extract_to)
    if remove_tar:
        os.remove(tar_path)
    print(f"Extracted to {extract_to}")


# ── Float16 + NaN-safe helpers ────────────────────────────────────────────────

def to_device(model, device, use_fp16=True):
    """Move model to device and optionally cast to float16."""
    model = model.to(device)
    if use_fp16:
        model = model.half()
    return model


def safe_cosine_similarity(src_feat, tgt_flat):
    """
    Compute cosine similarity in float32 regardless of input dtype.
    Prevents NaN from float16 near-zero norm vectors.

    Args:
        src_feat: [D] source feature vector (any dtype)
        tgt_flat: [H*W, D] target feature map (any dtype)

    Returns:
        similarities: [H*W] float32 tensor
    """
    import torch.nn.functional as F
    return F.cosine_similarity(
        src_feat.float().unsqueeze(0),
        tgt_flat.float(),
        dim=1
    )


def prepare_image(img_tensor, device, resized_size, use_fp16=True):
    """
    Resize image tensor and move to device with correct dtype.

    Args:
        img_tensor:   [1, C, H, W] float32 tensor
        device:       torch device
        resized_size: int, target spatial size
        use_fp16:     bool

    Returns:
        resized tensor on device in correct dtype
    """
    import torch.nn.functional as F
    img = F.interpolate(
        img_tensor,
        size=(resized_size, resized_size),
        mode='bilinear',
        align_corners=False
    ).to(device)
    if use_fp16:
        img = img.half()
    return img
