import numpy as np

def compute_pck(pred_points, gt_points, img_size, threshold):
    """
    Compute PCK@threshold

    Args:
        pred_points: list of [x, y] predictions
        gt_points: list of [x, y] ground truth
        img_size: (width, height) of the image
        threshold: normalized threshold (e.g., 0.05, 0.1, 0.2)

    Returns:
        pck: percentage of correct keypoints
        correct_mask: boolean array indicating which keypoints are correct
    """
    pred_points = np.array(pred_points)
    gt_points = np.array(gt_points)

    #compute Euclidean distance
    distances = np.sqrt(np.sum((pred_points - gt_points) ** 2, axis=1))

    #normalize by image diagonal (standard protocol)
    img_diagonal = np.sqrt(img_size[0] ** 2 + img_size[1] ** 2)
    normalized_distances = distances / img_diagonal

    #check which keypoints are within threshold
    correct_mask = normalized_distances <= threshold
    pck = np.mean(correct_mask) * 100  # percentage

    return pck, correct_mask, normalized_distances

def compute_pck_spair71k(pred_points, gt_points, bbox, threshold):
    """
    Compute PCK@threshold

    Args:
        pred_points: list of [x, y] predictions
        gt_points: list of [x, y] ground truth
        bbox: [xmin, ymin, xmax, ymax]
        threshold: normalized threshold (e.g., 0.05, 0.1, 0.2)

    Returns:
        pck: percentage of correct keypoints
        correct_mask: boolean array indicating which keypoints are correct
    """
    pred_points = np.array(pred_points)
    gt_points = np.array(gt_points)

    #compute Euclidean distance
    distances = np.sqrt(np.sum((pred_points - gt_points) ** 2, axis=1))

    # Normalize by max(bbox_width, bbox_height) - STANDARD SPAIR-71K
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    normalization_factor = max(bbox_width, bbox_height)
    normalized_distances = distances / normalization_factor

    #check which keypoints are within threshold
    correct_mask = normalized_distances <= threshold
    pck = np.mean(correct_mask) * 100  # percentage

    return pck, correct_mask, normalized_distances


def compute_pck_pfpascal(pred_points, gt_points, img_size, threshold):
    """
    Compute PCK@threshold for PF-Pascal (Standard Protocol)

    Args:
        pred_points: list of [x, y] predictions
        gt_points: list of [x, y] ground truth
        img_size: (width, height) of the image
        threshold: normalized threshold (e.g., 0.05, 0.1, 0.2)

    Returns:
        pck: percentage of correct keypoints
        correct_mask: boolean array indicating which keypoints are correct
    """
    pred_points = np.array(pred_points)
    gt_points = np.array(gt_points)

    # Compute Euclidean distance
    distances = np.sqrt(np.sum((pred_points - gt_points) ** 2, axis=1))

    # Normalize by max(H, W) - STANDARD PF-PASCAL
    normalization_factor = max(img_size[0], img_size[1])

    normalized_distances = distances / normalization_factor

    # Check which keypoints are within threshold
    correct_mask = normalized_distances <= threshold
    pck = np.mean(correct_mask) * 100  # percentage

    return pck, correct_mask, normalized_distances


def compute_pck_ap10k(pred_points, gt_points, img_size, threshold):
    """
    Compute PCK@threshold for AP-10K (Standard Protocol)

    Args:
        pred_points: list of [x, y] predictions
        gt_points: list of [x, y] or np.ndarray ground truth
        img_size: (width, height) of the image
        threshold: normalized threshold (e.g., 0.05, 0.1, 0.2)

    Returns:
        pck: percentage of correct keypoints
        correct_mask: boolean array indicating which keypoints are correct
        normalized_distances: array of per-keypoint distances normalized by image diagonal
    """
    pred_points = np.array(pred_points)
    gt_points = np.array(gt_points)

    # Compute Euclidean distance
    distances = np.sqrt(np.sum((pred_points - gt_points) ** 2, axis=1))

    # Normalize by image diagonal - STANDARD AP-10K
    normalization_factor = np.sqrt(img_size[0] ** 2 + img_size[1] ** 2)

    normalized_distances = distances / normalization_factor

    # Check which keypoints are within threshold
    correct_mask = normalized_distances <= threshold
    pck = np.mean(correct_mask) * 100  # percentage

    return pck, correct_mask, normalized_distances
