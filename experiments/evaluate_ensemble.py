import json
from collections import defaultdict
from matplotlib.style import use
import numpy as np
from sklearn import base
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from datetime import datetime
import pandas as pd
from pathlib import Path

from src.datasets.spair_dataset import SPairDataset
from src.datasets.pf_pascal_dataset import PFPascalDataset
from src.datasets.pf_willow_dataset import PFWillowDataset
from src.features.extractor import extract_dense_features, extract_dense_features_SAM, pixel_to_patch_coord, patch_to_pixel_coord
from src.matching.strategies import find_best_match_argmax, find_best_match_window_softargmax
from src.metrics.pck import compute_pck_spair71k, compute_pck_pfpascal
from src.models.dinov3.dinov3.models.vision_transformer import vit_base as dinov3_vit_base
from src.models.dinov2.dinov2.models.vision_transformer import vit_base as dinov2_vit_base
from src.models.segment_anything.segment_anything import sam_model_registry
import configs.paths as paths


class LearnedEnsembleWeights(nn.Module):
    """Learnable fusion weights for an ensemble of n models.

    The weights are derived from unconstrained logit parameters via softmax,
    so they always sum to 1 and stay positive:

        weights = softmax(logits)

    Initialising logits to zero produces uniform weights [1/n, 1/n, ..., 1/n]
    at the start of training.
    """

    def __init__(self, n_models: int = 3):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_models))

    def forward(self) -> torch.Tensor:
        """Return the current normalised weights as a [n_models] tensor."""
        return torch.softmax(self.logits, dim=0)


# ==================== CONFIG ====================
IMG_SIZE_DINOV2 = 518
PATCH_SIZE_DINOV2 = 14
IMG_SIZE_DINOV3 = 512
PATCH_SIZE_DINOV3 = 16
IMG_SIZE_SAM = 512
PATCH_SIZE_SAM = 16

THRESHOLDS = [0.05, 0.1, 0.2]

CHECKPOINT_PATHS = {
    "DINOv2": paths.DINOV2_FINETUNED,
    "DINOv3": paths.DINOV3_FINETUNED,
    "SAM": paths.SAM_FINETUNED,
}


# ==================== HELPER FUNCTIONS ====================

def load_models(device):
    """Load all three finetuned models."""
    print("Loading models...")

    # DINOv2
    dinov2 = dinov2_vit_base(
        img_size=(IMG_SIZE_DINOV2, IMG_SIZE_DINOV2),
        patch_size=PATCH_SIZE_DINOV2,
        num_register_tokens=0,
        block_chunks=0,
        init_values=1.0,
    )
    ckpt_dinov2 = torch.load(CHECKPOINT_PATHS["DINOv2"], map_location=device)
    dinov2.load_state_dict(ckpt_dinov2, strict=True)
    dinov2.to(device)
    dinov2.eval()

    # DINOv3
    dinov3 = dinov3_vit_base(
        img_size=(IMG_SIZE_DINOV3, IMG_SIZE_DINOV3),
        patch_size=PATCH_SIZE_DINOV3,
        n_storage_tokens=4,
        layerscale_init=1.0,
        mask_k_bias=True,
    )
    ckpt_dinov3 = torch.load(CHECKPOINT_PATHS["DINOv3"], map_location=device)
    dinov3.load_state_dict(ckpt_dinov3["model_state_dict"], strict=True)
    dinov3.to(device)
    dinov3.eval()

    # SAM
    sam = sam_model_registry["vit_b"](checkpoint=None)
    sam.to(device)

    print(f"Loading finetuned SAM checkpoint from {CHECKPOINT_PATHS['SAM']}")
    checkpoint = torch.load(CHECKPOINT_PATHS["SAM"], map_location=device)

    if 'model_state_dict' in checkpoint:
        sam.load_state_dict(checkpoint['model_state_dict'])
        print("Successfully loaded 'model_state_dict' from checkpoint.")
    else:
        sam.load_state_dict(checkpoint)
        print("Successfully loaded checkpoint directly as state_dict.")
    sam.eval()

    print("All models loaded successfully")
    return dinov2, dinov3, sam


def resize_map(m, H_t, W_t, H_ref, W_ref):
    if (H_t, W_t) == (H_ref, W_ref):
        return m
    return F.interpolate(m.unsqueeze(0).unsqueeze(0), size=(H_ref, W_ref),
                         mode='bilinear', align_corners=False).squeeze(0).squeeze(0)


def evaluate_ensemble_with_params(
    models_dict,
    dataset,
    device,
    K,
    temperature,
    weights,
    thresholds=None,
    use_windowed_softargmax=True,
):
    """
    Evaluate ensemble with weighted_avg fusion.

    Args:
        models_dict: dict with 'dinov2', 'dinov3', 'sam' models
        dataset: evaluation dataset
        device: torch device
        K: window size for softargmax
        temperature: softmax temperature
        weights: [w_dinov2, w_dinov3, w_sam] for weighted fusion (must sum to 1)
        thresholds: PCK thresholds
        use_windowed_softargmax: whether to use windowed softargmax or argmax
    Returns:
        per_image_metrics: list of dicts with PCK scores
        all_keypoint_metrics: list of dicts with per-keypoint metrics
    """
    if thresholds is None:
        thresholds = THRESHOLDS

    per_image_metrics = []
    all_keypoint_metrics = []
    dinov2, dinov3, sam = models_dict['dinov2'], models_dict['dinov3'], models_dict['sam']

    with torch.no_grad():
        for idx, sample in enumerate(dataset):
            # Load and resize images
            src_tensor = sample['src_img'].unsqueeze(0).to(device)
            tgt_tensor = sample['trg_img'].unsqueeze(0).to(device)

            # Resize for DINOv2
            src_dinov2 = F.interpolate(src_tensor, size=(IMG_SIZE_DINOV2, IMG_SIZE_DINOV2),
                                       mode='bilinear', align_corners=False)
            tgt_dinov2 = F.interpolate(tgt_tensor, size=(IMG_SIZE_DINOV2, IMG_SIZE_DINOV2),
                                       mode='bilinear', align_corners=False)

            # Resize for DINOv3
            src_dinov3 = F.interpolate(src_tensor, size=(IMG_SIZE_DINOV3, IMG_SIZE_DINOV3),
                                       mode='bilinear', align_corners=False)
            tgt_dinov3 = F.interpolate(tgt_tensor, size=(IMG_SIZE_DINOV3, IMG_SIZE_DINOV3),
                                       mode='bilinear', align_corners=False)

            # Resize for SAM
            src_sam = F.interpolate(src_tensor, size=(IMG_SIZE_SAM, IMG_SIZE_SAM),
                                    mode='bilinear', align_corners=False)
            tgt_sam = F.interpolate(tgt_tensor, size=(IMG_SIZE_SAM, IMG_SIZE_SAM),
                                    mode='bilinear', align_corners=False)

            # Extract features from all models
            src_feat_dinov2 = extract_dense_features(dinov2, src_dinov2)
            tgt_feat_dinov2 = extract_dense_features(dinov2, tgt_dinov2)

            src_feat_dinov3 = extract_dense_features(dinov3, src_dinov3)
            tgt_feat_dinov3 = extract_dense_features(dinov3, tgt_dinov3)

            src_feat_sam = extract_dense_features_SAM(sam, src_sam, image_size=IMG_SIZE_SAM)
            tgt_feat_sam = extract_dense_features_SAM(sam, tgt_sam, image_size=IMG_SIZE_SAM)

            # Get original sizes
            src_original_size = (sample['src_imsize'][2], sample['src_imsize'][1])
            tgt_original_size = (sample['trg_imsize'][2], sample['trg_imsize'][1])

            # Get keypoints
            src_kps = sample['src_kps'].numpy()
            trg_kps = sample['trg_kps'].numpy()
            kps_ids = sample['kps_ids']
            category = sample['category']
            trg_bbox = sample['trg_bbox']

            # Prepare target features for score-level fusion
            tgt_feat_dinov2_squeezed = tgt_feat_dinov2.squeeze(0)  # [H2, W2, D2]
            tgt_feat_dinov3_squeezed = tgt_feat_dinov3.squeeze(0)  # [H3, W3, D3]
            tgt_feat_sam_squeezed    = tgt_feat_sam.squeeze(0)     # [Hs, Ws, Ds]

            # Use SAM grid as the reference grid
            ref_shape = tgt_feat_sam_squeezed.shape  # (Hs, Ws, Ds)
            H_ref, W_ref = ref_shape[0], ref_shape[1]

            # Precompute normalized target flats for score-level fusion
            H2, W2, D2 = tgt_feat_dinov2_squeezed.shape
            H3, W3, D3 = tgt_feat_dinov3_squeezed.shape
            Hs, Ws, Ds = tgt_feat_sam_squeezed.shape

            tgt_v2_flat = F.normalize(tgt_feat_dinov2_squeezed.reshape(H2 * W2, D2), dim=1)
            tgt_v3_flat = F.normalize(tgt_feat_dinov3_squeezed.reshape(H3 * W3, D3), dim=1)
            tgt_s_flat  = F.normalize(tgt_feat_sam_squeezed.reshape(Hs * Ws, Ds),    dim=1)

            pred_matches = []

            # Process each keypoint
            for i in range(src_kps.shape[0]):
                src_x, src_y = src_kps[i]

                # Source features per model
                px2, py2 = pixel_to_patch_coord(src_x, src_y, src_original_size,
                                                patch_size=PATCH_SIZE_DINOV2, resized_size=IMG_SIZE_DINOV2)
                src_v2 = F.normalize(src_feat_dinov2[0, py2, px2, :], dim=0)

                px3, py3 = pixel_to_patch_coord(src_x, src_y, src_original_size,
                                                patch_size=PATCH_SIZE_DINOV3, resized_size=IMG_SIZE_DINOV3)
                src_v3 = F.normalize(src_feat_dinov3[0, py3, px3, :], dim=0)

                pxs, pys = pixel_to_patch_coord(src_x, src_y, src_original_size,
                                                patch_size=PATCH_SIZE_SAM, resized_size=IMG_SIZE_SAM)
                src_vs = F.normalize(src_feat_sam[0, pys, pxs, :], dim=0)

                # Score-level fusion: build per-model sim maps, upsample to ref grid, then weight-sum
                sim2 = F.cosine_similarity(src_v2.unsqueeze(0), tgt_v2_flat, dim=1).view(H2, W2)
                sim3 = F.cosine_similarity(src_v3.unsqueeze(0), tgt_v3_flat, dim=1).view(H3, W3)
                sims = F.cosine_similarity(src_vs.unsqueeze(0),  tgt_s_flat,  dim=1).view(Hs, Ws)

                sim2_r = resize_map(sim2, H2, W2, H_ref, W_ref)
                sim3_r = resize_map(sim3, H3, W3, H_ref, W_ref)
                sims_r = resize_map(sims, Hs, Ws, H_ref, W_ref)

                similarities = (weights[0] * sim2_r + weights[1] * sim3_r + weights[2] * sims_r).reshape(-1)

                # Find best match on ensemble similarity map
                if use_windowed_softargmax:
                    match_patch_x, match_patch_y = find_best_match_window_softargmax(similarities, W_ref, H_ref, K, temperature)
                else:
                    match_patch_x, match_patch_y = find_best_match_argmax(similarities, W_ref)
                # Convert to original image coords (ref grid = SAM)
                match_x, match_y = patch_to_pixel_coord(
                    match_patch_x, match_patch_y, tgt_original_size,
                    patch_size=PATCH_SIZE_SAM, resized_size=IMG_SIZE_SAM
                )
                pred_matches.append([match_x, match_y])

            # Compute PCK
            image_pcks = {}
            for threshold in thresholds:
                pck, correct_mask, distances = compute_pck_spair71k(
                    pred_matches,
                    trg_kps.tolist(),
                    trg_bbox,
                    threshold
                )
                # store keypoint-wise metrics
                for kps_id, pred, gt, dist, correct in zip(
                        kps_ids, pred_matches, trg_kps.tolist(), distances, correct_mask
                ):
                    all_keypoint_metrics.append({
                        'image_idx': idx,
                        'category': category,
                        'keypoint_id': kps_id,
                        'pred': pred,
                        'gt': gt,
                        'distance': dist,
                        'correct_at_threshold': correct,
                        'threshold': threshold
                    })
                image_pcks[threshold] = pck

            # store per-image metrics
            per_image_metrics.append({
                'category': category,
                'source_path': str(sample['src_imname']),
                'target_path': str(sample['trg_imname']),
                'num_keypoints': src_kps.shape[0],
                'pck_scores': image_pcks,
                'pred_points': pred_matches,
                'gt_points': trg_kps.tolist(),
                'kps_ids': kps_ids,
            })

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(dataset)} images")

    return per_image_metrics, all_keypoint_metrics


def save_results(per_image_metrics, all_keypoint_metrics, results_dir, total_inference_time_sec, thresholds):
    print(f"Total inference time: {total_inference_time_sec:.2f} seconds")

    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)

    overall_stats = {"inference_time_sec": total_inference_time_sec}

    for threshold in thresholds:
        all_pcks = np.array([img['pck_scores'][threshold] for img in per_image_metrics])

        mean_pck = float(np.mean(all_pcks))
        std_pck = float(np.std(all_pcks))
        median_pck = float(np.median(all_pcks))
        p25 = float(np.percentile(all_pcks, 25))
        p75 = float(np.percentile(all_pcks, 75))

        overall_stats[f"pck@{threshold:.2f}"] = {
            "mean": mean_pck,
            "std": std_pck,
            "median": median_pck,
            "p25": p25,
            "p75": p75,
        }

        print(f"PCK@{threshold:.2f}: "
              f"mean={mean_pck:.2f}%, std={std_pck:.2f}%, "
              f"median={median_pck:.2f}%, "
              f"p25={p25:.2f}%, p75={p75:.2f}%")

    with open(f'{results_dir}/overall_stats.json', 'w') as f:
        json.dump(overall_stats, f, indent=2)

    df_all_kp = pd.DataFrame(all_keypoint_metrics)
    csv_path = f'{results_dir}/all_keypoint_metrics.csv'
    df_all_kp.to_csv(csv_path, index=False)
    print(f"Saved all keypoint metrics to '{csv_path}'")


def train_ensemble_weights(
    models_dict,
    train_dataset,
    val_dataset,
    device,
    K=5,
    temperature=0.2,
    num_epochs=3,
    lr=1e-2,
    results_dir=None,
):
    """Learn optimal ensemble fusion weights by minimising cross-entropy on training keypoints.

    All three backbone models are kept frozen throughout.  Only the logits of a
    LearnedEnsembleWeights module are trained, driving softmax weights that
    linearly combine per-model cosine-similarity maps on a shared SAM reference
    grid (32×32 patches).

    For each training sample the per-model similarity maps are computed once,
    resized to the SAM reference grid, and fused with the current softmax weights.
    A cross-entropy loss is computed for every annotated keypoint (using the same
    formula as finetune.py) and the mean is back-propagated per sample.

    After every epoch the model is validated with evaluate_ensemble_with_params
    and the best weights are serialised to ``results_dir/learned_weights.json``.

    Args:
        models_dict: dict with keys 'dinov2', 'dinov3', 'sam'.
        train_dataset: Iterable of SPair-71k training samples.
        val_dataset: Iterable of SPair-71k validation samples.
        device: torch.device.
        K: Window size for soft-argmax during validation.
        temperature: Softmax temperature applied to the fused similarity map
            during both training loss and validation matching.
        num_epochs: Number of full passes over the training set.
        lr: Learning rate for the Adam optimiser.
        results_dir: Directory where learned_weights.json is saved.

    Returns:
        List of three floats [w_dinov2, w_dinov3, w_sam] corresponding to the
        weights that achieved the best val PCK@0.1.
    """
    if results_dir is None:
        results_dir = paths.RESULTS_STEP4_ENSEMBLE

    dinov2 = models_dict['dinov2']
    dinov3 = models_dict['dinov3']
    sam    = models_dict['sam']

    # Freeze all backbone parameters and set eval mode
    for m in [dinov2, dinov3, sam]:
        m.eval()
        for param in m.parameters():
            param.requires_grad = False

    weight_module = LearnedEnsembleWeights(n_models=3).to(device)
    optimizer = torch.optim.Adam(weight_module.parameters(), lr=lr)

    os.makedirs(results_dir, exist_ok=True)

    best_val_pck = 0.0
    best_weights = weight_module().detach().cpu().tolist()  # uniform [1/3, 1/3, 1/3]

    for epoch in range(1, num_epochs + 1):
        weight_module.train()
        epoch_loss = 0.0
        num_samples = 0

        for idx, sample in enumerate(train_dataset):
            src_tensor = sample['src_img'].unsqueeze(0).to(device)
            tgt_tensor = sample['trg_img'].unsqueeze(0).to(device)

            # Resize per backbone
            src_dinov2 = F.interpolate(src_tensor, size=(IMG_SIZE_DINOV2, IMG_SIZE_DINOV2), mode='bilinear', align_corners=False)
            tgt_dinov2 = F.interpolate(tgt_tensor, size=(IMG_SIZE_DINOV2, IMG_SIZE_DINOV2), mode='bilinear', align_corners=False)
            src_dinov3 = F.interpolate(src_tensor, size=(IMG_SIZE_DINOV3, IMG_SIZE_DINOV3), mode='bilinear', align_corners=False)
            tgt_dinov3 = F.interpolate(tgt_tensor, size=(IMG_SIZE_DINOV3, IMG_SIZE_DINOV3), mode='bilinear', align_corners=False)
            src_sam_t  = F.interpolate(src_tensor, size=(IMG_SIZE_SAM, IMG_SIZE_SAM),       mode='bilinear', align_corners=False)
            tgt_sam_t  = F.interpolate(tgt_tensor, size=(IMG_SIZE_SAM, IMG_SIZE_SAM),       mode='bilinear', align_corners=False)

            # Extract features with no_grad — backbones are frozen
            with torch.no_grad():
                sf_v2  = extract_dense_features(dinov2, src_dinov2)
                tf_v2  = extract_dense_features(dinov2, tgt_dinov2)
                sf_v3  = extract_dense_features(dinov3, src_dinov3)
                tf_v3  = extract_dense_features(dinov3, tgt_dinov3)
                sf_sam = extract_dense_features_SAM(sam, src_sam_t, image_size=IMG_SIZE_SAM)
                tf_sam = extract_dense_features_SAM(sam, tgt_sam_t, image_size=IMG_SIZE_SAM)

            _, H2, W2, D2 = tf_v2.shape
            _, H3, W3, D3 = tf_v3.shape
            _, Hs, Ws, Ds = tf_sam.shape
            H_ref, W_ref = 32, 32  # SAM reference grid (512 / 16)

            # Pre-normalised target flats — no grad required
            tgt_v2_flat = F.normalize(tf_v2.squeeze(0).reshape(H2 * W2, D2), dim=1)
            tgt_v3_flat = F.normalize(tf_v3.squeeze(0).reshape(H3 * W3, D3), dim=1)
            tgt_s_flat  = F.normalize(tf_sam.squeeze(0).reshape(Hs * Ws, Ds), dim=1)

            src_kps = sample['src_kps'].numpy()
            trg_kps = sample['trg_kps'].numpy()
            src_original_size = (sample['src_imsize'][2], sample['src_imsize'][1])
            tgt_original_size = (sample['trg_imsize'][2], sample['trg_imsize'][1])

            keypoint_losses = []

            for i in range(src_kps.shape[0]):
                src_x, src_y = src_kps[i]
                tgt_x, tgt_y = trg_kps[i]

                # Source patch indices for each backbone
                px2, py2 = pixel_to_patch_coord(src_x, src_y, src_original_size,
                                                 patch_size=PATCH_SIZE_DINOV2, resized_size=IMG_SIZE_DINOV2)
                px3, py3 = pixel_to_patch_coord(src_x, src_y, src_original_size,
                                                 patch_size=PATCH_SIZE_DINOV3, resized_size=IMG_SIZE_DINOV3)
                pxs, pys = pixel_to_patch_coord(src_x, src_y, src_original_size,
                                                 patch_size=PATCH_SIZE_SAM, resized_size=IMG_SIZE_SAM)

                # Normalised source descriptors (no grad — features from frozen models)
                src_f2 = F.normalize(sf_v2[0, py2, px2, :], dim=0)
                src_f3 = F.normalize(sf_v3[0, py3, px3, :], dim=0)
                src_fs = F.normalize(sf_sam[0, pys, pxs, :], dim=0)

                # Per-model cosine similarity maps
                sim2 = F.cosine_similarity(src_f2.unsqueeze(0), tgt_v2_flat, dim=1).view(H2, W2)
                sim3 = F.cosine_similarity(src_f3.unsqueeze(0), tgt_v3_flat, dim=1).view(H3, W3)
                sims = F.cosine_similarity(src_fs.unsqueeze(0), tgt_s_flat,  dim=1).view(Hs, Ws)

                # Resize all maps to the SAM reference grid
                sim2_r = resize_map(sim2, H2, W2, H_ref, W_ref)
                sim3_r = resize_map(sim3, H3, W3, H_ref, W_ref)
                sims_r = resize_map(sims, Hs, Ws, H_ref, W_ref)

                # Fuse with learned weights — gradient flows through w into logits
                w = weight_module()  # [3]
                similarities = (w[0] * sim2_r + w[1] * sim3_r + w[2] * sims_r).reshape(-1)

                # GT index on the SAM reference grid
                gt_patch_x, gt_patch_y = pixel_to_patch_coord(
                    tgt_x, tgt_y, tgt_original_size,
                    patch_size=PATCH_SIZE_SAM, resized_size=IMG_SIZE_SAM
                )
                gt_idx = gt_patch_y * W_ref + gt_patch_x

                # Cross-entropy loss (same formula as finetune.py)
                log_probs = F.log_softmax(similarities * temperature, dim=0)
                loss = -log_probs[gt_idx]
                keypoint_losses.append(loss)

            if not keypoint_losses:
                continue

            sample_loss = torch.stack(keypoint_losses).mean()
            optimizer.zero_grad()
            sample_loss.backward()
            optimizer.step()

            epoch_loss  += sample_loss.item()
            num_samples += 1

            if (idx + 1) % 100 == 0:
                print(f"  Epoch {epoch}, sample {idx + 1}, avg loss: {epoch_loss / num_samples:.4f}")

        avg_epoch_loss = epoch_loss / max(num_samples, 1)
        print(f"\nEpoch {epoch}/{num_epochs} — avg train loss: {avg_epoch_loss:.4f}")

        # Evaluate on val_dataset with current weights
        current_weights = weight_module().detach().cpu().tolist()
        val_metrics, _ = evaluate_ensemble_with_params(
            models_dict=models_dict,
            dataset=val_dataset,
            device=device,
            K=K,
            temperature=temperature,
            weights=current_weights,
            thresholds=THRESHOLDS,
            use_windowed_softargmax=True,
        )
        val_pck = float(np.mean([m['pck_scores'][0.1] for m in val_metrics]))
        print(f"Val PCK@0.1: {val_pck:.2f}%  |  weights: "
              f"[{current_weights[0]:.4f}, {current_weights[1]:.4f}, {current_weights[2]:.4f}]")

        if val_pck > best_val_pck:
            best_val_pck  = val_pck
            best_weights  = current_weights
            with open(f'{results_dir}/learned_weights.json', 'w') as f:
                json.dump({'weights': best_weights, 'val_pck': best_val_pck}, f, indent=2)
            print(f"New best weights saved to {results_dir}/learned_weights.json")

    return best_weights


# ==================== MAIN ====================

if __name__ == "__main__":
    USE_LEARNED_WEIGHTS = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load models
    dinov2, dinov3, sam = load_models(device)
    models_dict = {'dinov2': dinov2, 'dinov3': dinov3, 'sam': sam}

    # Fixed window soft-argmax params
    K = 5
    temperature = 0.2

    use_windowed_softargmax = True

    # Weighted-average fusion weights: [DINOv2, DINOv3, SAM]
    if USE_LEARNED_WEIGHTS:
        print("\nLearning ensemble weights on training split...")
        train_dataset_lw = SPairDataset(paths.SPAIR71K_PAIRS, paths.SPAIR71K_LAYOUT, paths.SPAIR71K_IMAGES, 'large', 0.1, datatype='trn')
        val_dataset_lw   = SPairDataset(paths.SPAIR71K_PAIRS, paths.SPAIR71K_LAYOUT, paths.SPAIR71K_IMAGES, 'large', 0.1, datatype='val')
        print(f"  Train pairs: {len(train_dataset_lw)}  |  Val pairs: {len(val_dataset_lw)}")
        weights = train_ensemble_weights(
            models_dict, train_dataset_lw, val_dataset_lw, device,
            K=K, temperature=temperature,
            results_dir=paths.RESULTS_STEP4_ENSEMBLE,
        )
        print(f"Learned weights: {weights}")
    else:
        weights = [0.25, 0.65, 0.10]

    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = SPairDataset(paths.SPAIR71K_PAIRS, paths.SPAIR71K_LAYOUT, paths.SPAIR71K_IMAGES, 'large', 0.1, datatype='test')

    print(f"Test set loaded: {len(test_dataset)} pairs")

    # Results dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wtag = f"{weights[0]:.2f}-{weights[1]:.2f}-{weights[2]:.2f}"
    results_dir = f'{paths.RESULTS_STEP4_ENSEMBLE}/weighted_avg/K{K}_T{temperature}_w{wtag}_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    # Evaluate with weighted_avg fusion
    start = time.time()
    per_image_metrics, all_keypoint_metrics = evaluate_ensemble_with_params(
        models_dict=models_dict,
        dataset=test_dataset,
        device=device,
        K=K,
        temperature=temperature,
        weights=weights,
        thresholds=THRESHOLDS,
        use_windowed_softargmax=use_windowed_softargmax,
    )
    elapsed = time.time() - start

    # Save results using the same format as evaluate.py
    save_results(per_image_metrics, all_keypoint_metrics, results_dir, elapsed, THRESHOLDS)
