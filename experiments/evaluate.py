import json
import time

import numpy as np
import pandas as pd
import torch

from src.features.extractor import (
    extract_dense_features,
    extract_dense_features_multilayer,
    apply_pca_whitening,
    pixel_to_patch_coord,
    patch_to_pixel_coord,
)
from src.matching.strategies import (
    find_best_match_argmax,
    find_best_match_window_softargmax,
    find_best_match_mnn,
    apply_mnn_filter,
)
from src.metrics.pck import compute_pck_spair71k, compute_pck_pfpascal
import torch.nn.functional as F

def evaluate(model, dataset, device, thresholds=[0.05, 0.1, 0.2], use_windowed_softargmax=False, early_stop=False,K=5, temperature=0.1):
    inference_start_time = time.time()
    per_image_metrics = []
    all_keypoint_metrics = []

    with torch.no_grad():
        for idx, sample in enumerate(dataset):  # type: ignore
            # extract tensors and move to device
            src_tensor = sample['src_img'].unsqueeze(0).to(device)  # [1, 3, H, W]
            tgt_tensor = sample['trg_img'].unsqueeze(0).to(device)  # [1, 3, H, W]

            # resize to 518x518
            src_tensor = F.interpolate(src_tensor, size=(518, 518), mode='bilinear', align_corners=False)
            tgt_tensor = F.interpolate(tgt_tensor, size=(518, 518), mode='bilinear', align_corners=False)

            # save original sizes ([C, H, W] -> (W, H))
            src_original_size = (sample['src_imsize'][2], sample['src_imsize'][1])
            tgt_original_size = (sample['trg_imsize'][2], sample['trg_imsize'][1])

            # extract dense features
            src_features = extract_dense_features(model, src_tensor)
            tgt_features = extract_dense_features(model, tgt_tensor)

            # reshape
            _, H, W, D = tgt_features.shape  # B=1
            tgt_flat = tgt_features.reshape(H * W, D)

            # extract keypoints
            src_kps = sample['src_kps'].numpy()  # [N, 2]
            trg_kps = sample['trg_kps'].numpy()  # [N, 2]
            kps_ids = sample['kps_ids']  # [N]

            trg_bbox = sample['trg_bbox']

            pred_matches = []

            # iterate over keypoints and predict matches
            for i in range(src_kps.shape[0]):
                src_x, src_y = src_kps[i]
                tgt_x, tgt_y = trg_kps[i]

                patch_x, patch_y = pixel_to_patch_coord(src_x, src_y, src_original_size)

                # extract source feature at the keypoint patch
                src_feature = src_features[0, patch_y, patch_x, :]  # [D]

                # compute cosine similarities with all target features
                similarities = F.cosine_similarity(
                    src_feature.unsqueeze(0),  # [1, D]
                    tgt_flat,  # [H*W, D]
                    dim=1
                )  # [H*W]

                # find best matching patch in target
                if use_windowed_softargmax:
                    match_patch_x, match_patch_y = find_best_match_window_softargmax(similarities, W, H, K, temperature)
                else:
                    match_patch_x, match_patch_y = find_best_match_argmax(similarities, W)
                match_x, match_y = patch_to_pixel_coord(
                    match_patch_x, match_patch_y, tgt_original_size
                )

                pred_matches.append([match_x, match_y])

            # compute PCK per diverse threshold
            image_pcks = {}
            category = sample['category']

            for threshold in thresholds:
                pck, correct_mask, distances = compute_pck_spair71k(
                    pred_matches,
                    trg_kps.tolist(),
                    trg_bbox,
                    threshold
                )
                image_pcks[threshold] = pck
                # category_metrics[category][threshold].append(pck)

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
                print(f"Processed {idx + 1} pairs...")

            # debug early stopping
            if early_stop and idx == 50:
                break
    return per_image_metrics, all_keypoint_metrics, time.time() - inference_start_time

def evaluate_no_spair71k(model, dataset, device, thresholds=[0.05, 0.1, 0.2], use_windowed_softargmax=False, early_stop=False,K=5, temperature=0.1):
    inference_start_time = time.time()
    per_image_metrics = []
    all_keypoint_metrics = []

    with torch.no_grad():
        for idx, sample in enumerate(dataset):  # type: ignore
            # extract tensors and move to device
            src_tensor = sample['src_img'].unsqueeze(0).to(device)  # [1, 3, H, W]
            tgt_tensor = sample['trg_img'].unsqueeze(0).to(device)  # [1, 3, H, W]

            # resize to 518x518
            src_tensor = F.interpolate(src_tensor, size=(518, 518), mode='bilinear', align_corners=False)
            tgt_tensor = F.interpolate(tgt_tensor, size=(518, 518), mode='bilinear', align_corners=False)

            # save original sizes ([C, H, W] -> (W, H))
            src_original_size = (sample['src_imsize'][2], sample['src_imsize'][1])
            tgt_original_size = (sample['trg_imsize'][2], sample['trg_imsize'][1])

            # extract dense features
            src_features = extract_dense_features(model, src_tensor)
            tgt_features = extract_dense_features(model, tgt_tensor)

            # reshape
            _, H, W, D = tgt_features.shape  # B=1
            tgt_flat = tgt_features.reshape(H * W, D)

            # extract keypoints
            src_kps = sample['src_kps'].numpy()  # [N, 2]
            trg_kps = sample['trg_kps'].numpy()  # [N, 2]
            kps_ids = sample['kps_ids']  # [N]
            pred_matches = []

            # iterate over keypoints and predict matches
            for i in range(src_kps.shape[0]):
                src_x, src_y = src_kps[i]
                tgt_x, tgt_y = trg_kps[i]

                patch_x, patch_y = pixel_to_patch_coord(src_x, src_y, src_original_size)

                # extract source feature at the keypoint patch
                src_feature = src_features[0, patch_y, patch_x, :]  # [D]

                # compute cosine similarities with all target features
                similarities = F.cosine_similarity(
                    src_feature.unsqueeze(0),  # [1, D]
                    tgt_flat,  # [H*W, D]
                    dim=1
                )  # [H*W]

                # find best matching patch in target
                if use_windowed_softargmax:
                    match_patch_x, match_patch_y = find_best_match_window_softargmax(similarities, W, H, K, temperature)
                else:
                    match_patch_x, match_patch_y = find_best_match_argmax(similarities, W)
                match_x, match_y = patch_to_pixel_coord(
                    match_patch_x, match_patch_y, tgt_original_size
                )

                pred_matches.append([match_x, match_y])

            # compute PCK per diverse threshold
            image_pcks = {}
            category = sample['category']

            for threshold in thresholds:
                pck, correct_mask, distances = compute_pck_pfpascal(
                    pred_matches, trg_kps, tgt_original_size, threshold
                )
                image_pcks[threshold] = pck
                # category_metrics[category][threshold].append(pck)

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
                print(f"Processed {idx + 1} pairs...")

            # debug early stopping
            if early_stop and idx == 50:
                break
    return per_image_metrics, all_keypoint_metrics, time.time() - inference_start_time

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


def evaluate_multilayer(
    model,
    dataset,
    device,
    thresholds=[0.05, 0.1, 0.2],
    use_windowed_softargmax=False,
    use_pca=False,
    n_components=64,
    early_stop=False,
    K=5,
    temperature=0.1,
    patch_size=14,
    resized_size=518,
    n_last_layers=3,
):
    """Evaluate semantic correspondence using multi-layer averaged DINOv2 features.

    Identical to evaluate() except features are extracted by averaging the last
    n_last_layers transformer layers via extract_dense_features_multilayer.
    Optionally applies PCA whitening to the joint source/target feature space
    before matching.

    Args:
        model: DINOv2 model with get_intermediate_layers support.
        dataset: Iterable of sample dicts (SPair-71k format).
        device: torch.device for tensor placement.
        thresholds: List of PCK thresholds (e.g. [0.05, 0.1, 0.2]).
        use_windowed_softargmax: Use window soft-argmax instead of hard argmax.
        use_pca: Apply PCA whitening after feature extraction.
        n_components: Number of PCA components (effective value clamped to min(n, H*W, D)).
        early_stop: Stop after 50 samples (debug mode).
        K: Window size for soft-argmax (must be odd).
        temperature: Softmax temperature for soft-argmax.
        patch_size: Patch size in pixels (default 14 for DINOv2).
        resized_size: Image size after interpolation (default 518).
        n_last_layers: Number of last transformer layers to average (default 3).

    Returns:
        Tuple (per_image_metrics, all_keypoint_metrics, inference_time_sec).
    """
    inference_start_time = time.time()
    per_image_metrics = []
    all_keypoint_metrics = []

    with torch.no_grad():
        for idx, sample in enumerate(dataset):  # type: ignore
            # extract tensors and move to device
            src_tensor = sample['src_img'].unsqueeze(0).to(device)  # [1, 3, H, W]
            tgt_tensor = sample['trg_img'].unsqueeze(0).to(device)  # [1, 3, H, W]

            # resize to resized_size x resized_size
            src_tensor = F.interpolate(src_tensor, size=(resized_size, resized_size), mode='bilinear', align_corners=False)
            tgt_tensor = F.interpolate(tgt_tensor, size=(resized_size, resized_size), mode='bilinear', align_corners=False)

            # save original sizes ([C, H, W] -> (W, H))
            src_original_size = (sample['src_imsize'][2], sample['src_imsize'][1])
            tgt_original_size = (sample['trg_imsize'][2], sample['trg_imsize'][1])

            # extract dense features (multi-layer)
            src_features = extract_dense_features_multilayer(model, src_tensor, n_last_layers)
            tgt_features = extract_dense_features_multilayer(model, tgt_tensor, n_last_layers)

            # optional PCA whitening
            if use_pca:
                src_features, tgt_features = apply_pca_whitening(src_features, tgt_features, n_components)

            # reshape
            _, H, W, D = tgt_features.shape  # B=1
            tgt_flat = tgt_features.reshape(H * W, D)

            # extract keypoints
            src_kps = sample['src_kps'].numpy()  # [N, 2]
            trg_kps = sample['trg_kps'].numpy()  # [N, 2]
            kps_ids = sample['kps_ids']  # [N]

            trg_bbox = sample['trg_bbox']

            pred_matches = []

            # iterate over keypoints and predict matches
            for i in range(src_kps.shape[0]):
                src_x, src_y = src_kps[i]
                tgt_x, tgt_y = trg_kps[i]

                patch_x, patch_y = pixel_to_patch_coord(src_x, src_y, src_original_size, patch_size, resized_size)

                # extract source feature at the keypoint patch
                src_feature = src_features[0, patch_y, patch_x, :]  # [D]

                # compute cosine similarities with all target features
                similarities = F.cosine_similarity(
                    src_feature.unsqueeze(0),  # [1, D]
                    tgt_flat,  # [H*W, D]
                    dim=1
                )  # [H*W]

                # find best matching patch in target
                if use_windowed_softargmax:
                    match_patch_x, match_patch_y = find_best_match_window_softargmax(similarities, W, H, K, temperature)
                else:
                    match_patch_x, match_patch_y = find_best_match_argmax(similarities, W)
                match_x, match_y = patch_to_pixel_coord(
                    match_patch_x, match_patch_y, tgt_original_size, patch_size, resized_size
                )

                pred_matches.append([match_x, match_y])

            # compute PCK per diverse threshold
            image_pcks = {}
            category = sample['category']

            for threshold in thresholds:
                pck, correct_mask, distances = compute_pck_spair71k(
                    pred_matches,
                    trg_kps.tolist(),
                    trg_bbox,
                    threshold
                )
                image_pcks[threshold] = pck

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
                print(f"Processed {idx + 1} pairs...")

            # debug early stopping
            if early_stop and idx == 50:
                break

    return per_image_metrics, all_keypoint_metrics, time.time() - inference_start_time


def evaluate_with_mnn(
    model,
    dataset,
    device,
    thresholds=[0.05, 0.1, 0.2],
    K=5,
    temperature=0.1,
    max_patch_dist=1,
    use_multilayer=False,
    use_pca=False,
    n_components=64,
    patch_size=14,
    resized_size=518,
    early_stop=False,
):
    """Evaluate semantic correspondence with Mutual Nearest Neighbour (MNN) filtering.

    Identical to evaluate() except each candidate match is verified by a backward
    pass: the predicted target patch is matched back to the source, and the
    round-trip L-inf error is checked against max_patch_dist.  If the check fails
    the forward match is masked out and the next best target patch is used instead.

    Args:
        model: DINOv2 model (standard or with get_intermediate_layers support).
        dataset: Iterable of sample dicts (SPair-71k format).
        device: torch.device for tensor placement.
        thresholds: List of PCK thresholds (e.g. [0.05, 0.1, 0.2]).
        K: Window size for soft-argmax inside MNN (must be odd).
        temperature: Softmax temperature for soft-argmax inside MNN.
        max_patch_dist: Maximum allowed L-inf round-trip error in patches.
        use_multilayer: Use extract_dense_features_multilayer instead of extract_dense_features.
        use_pca: Apply PCA whitening after feature extraction.
        n_components: Number of PCA components (effective value clamped to min(n, H*W, D)).
        patch_size: Patch size in pixels (default 14 for DINOv2).
        resized_size: Image size after interpolation (default 518).
        early_stop: Stop after 50 samples (debug mode).

    Returns:
        Tuple (per_image_metrics, all_keypoint_metrics, inference_time_sec).
    """
    inference_start_time = time.time()
    per_image_metrics = []
    all_keypoint_metrics = []

    with torch.no_grad():
        for idx, sample in enumerate(dataset):  # type: ignore
            # extract tensors and move to device
            src_tensor = sample['src_img'].unsqueeze(0).to(device)  # [1, 3, H, W]
            tgt_tensor = sample['trg_img'].unsqueeze(0).to(device)  # [1, 3, H, W]

            # resize to resized_size x resized_size
            src_tensor = F.interpolate(src_tensor, size=(resized_size, resized_size), mode='bilinear', align_corners=False)
            tgt_tensor = F.interpolate(tgt_tensor, size=(resized_size, resized_size), mode='bilinear', align_corners=False)

            # save original sizes ([C, H, W] -> (W, H))
            src_original_size = (sample['src_imsize'][2], sample['src_imsize'][1])
            tgt_original_size = (sample['trg_imsize'][2], sample['trg_imsize'][1])

            # extract dense features
            if use_multilayer:
                src_features = extract_dense_features_multilayer(model, src_tensor)
                tgt_features = extract_dense_features_multilayer(model, tgt_tensor)
            else:
                src_features = extract_dense_features(model, src_tensor)
                tgt_features = extract_dense_features(model, tgt_tensor)

            # optional PCA whitening
            if use_pca:
                src_features, tgt_features = apply_pca_whitening(src_features, tgt_features, n_components)

            # reshape
            _, H, W, D = tgt_features.shape  # B=1
            tgt_flat = tgt_features.reshape(H * W, D)

            # extract keypoints
            src_kps = sample['src_kps'].numpy()  # [N, 2]
            trg_kps = sample['trg_kps'].numpy()  # [N, 2]
            kps_ids = sample['kps_ids']  # [N]

            trg_bbox = sample['trg_bbox']

            pred_matches = []

            # iterate over keypoints and predict matches
            for i in range(src_kps.shape[0]):
                src_x, src_y = src_kps[i]
                tgt_x, tgt_y = trg_kps[i]

                patch_x, patch_y = pixel_to_patch_coord(src_x, src_y, src_original_size, patch_size, resized_size)

                # extract source feature at the keypoint patch
                src_feature = src_features[0, patch_y, patch_x, :]  # [D]

                # compute cosine similarities with all target features
                similarities = F.cosine_similarity(
                    src_feature.unsqueeze(0),  # [1, D]
                    tgt_flat,  # [H*W, D]
                    dim=1
                )  # [H*W]

                # MNN: forward match + consistency check
                fwd_x, fwd_y, back_x, back_y = find_best_match_mnn(
                    similarities, src_features, tgt_features, W, H, K, temperature
                )
                match_patch_x, match_patch_y = apply_mnn_filter(
                    fwd_x, fwd_y, back_x, back_y,
                    patch_x, patch_y,
                    similarities, W, H, K, temperature, max_patch_dist
                )

                match_x, match_y = patch_to_pixel_coord(
                    match_patch_x, match_patch_y, tgt_original_size, patch_size, resized_size
                )

                pred_matches.append([match_x, match_y])

            # compute PCK per diverse threshold
            image_pcks = {}
            category = sample['category']

            for threshold in thresholds:
                pck, correct_mask, distances = compute_pck_spair71k(
                    pred_matches,
                    trg_kps.tolist(),
                    trg_bbox,
                    threshold
                )
                image_pcks[threshold] = pck

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
                print(f"Processed {idx + 1} pairs...")

            # debug early stopping
            if early_stop and idx == 50:
                break

    return per_image_metrics, all_keypoint_metrics, time.time() - inference_start_time
