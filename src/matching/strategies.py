import torch
import torch.nn.functional as F


def find_best_match_argmax(s, width):
    best_match_idx = s.argmax().item()
    y = best_match_idx // width
    x = best_match_idx % width
    return x, y


def find_best_match_window_softargmax(
    s: torch.Tensor,
    width: int,
    height: int,
    K: int = 5,
    temperature: float = 1.0,
):
    """
    s: [H*W] similarities
    width: W
    height: H
    K: window size in patches (odd number, e.g. 3,5,7)
    temperature: softmax temperature (softmax(s / temperature))
    returns (x_hat, y_hat) as continuous patch coordinates (floats)
    """
    assert K % 2 == 1, "K must be odd"

    # reshape to 2D similarity map [H, W]
    sim_map = s.view(height, width)  # [H, W]

    # hard argmax to find window center
    cx, cy = find_best_match_argmax(s, width)

    # half window
    r = K // 2

    # window bounds (clamped to image)
    y_min = max(cy - r, 0)
    y_max = min(cy + r + 1, height)  # exclusive
    x_min = max(cx - r, 0)
    x_max = min(cx + r + 1, width)   # exclusive

    # crop window
    window = sim_map[y_min:y_max, x_min:x_max]  # [h_win, w_win]

    # build coordinate grid for patches in window
    ys = torch.arange(y_min, y_max, device=s.device)
    xs = torch.arange(x_min, x_max, device=s.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # [h_win, w_win]

    # softmax over window with temperature
    logits = window / temperature
    weights = F.softmax(logits.view(-1), dim=0)  # [h_win*w_win]

    grid_x_flat = grid_x.reshape(-1).float()
    grid_y_flat = grid_y.reshape(-1).float()

    # soft-argmax expectation in patch space
    x_hat = (weights * grid_x_flat).sum()
    y_hat = (weights * grid_y_flat).sum()

    return x_hat.item(), y_hat.item()


def find_best_match_mnn(
    similarities: torch.Tensor,
    src_features: torch.Tensor,
    tgt_features: torch.Tensor,
    width: int,
    height: int,
    K: int = 5,
    temperature: float = 1.0,
):
    """Find the best match using Mutual Nearest Neighbour (MNN) verification.

    Performs a forward match from source to target via soft-argmax, then a
    backward match from the predicted target patch back to the source to
    verify consistency.

    Args:
        similarities: Flattened similarity vector [H*W] between one source
                      patch and all target patches.
        src_features:  Source dense feature map [1, H_s, W_s, D].
        tgt_features:  Target dense feature map [1, H_t, W_t, D].
        width:         Number of patch columns in the target map (W_t).
        height:        Number of patch rows in the target map (H_t).
        K:             Window size for soft-argmax (must be odd).
        temperature:   Softmax temperature for soft-argmax.

    Returns:
        Tuple (fwd_x, fwd_y, back_x, back_y) where fwd_* are the forward
        soft-argmax coordinates (floats) and back_* are the backward
        hard-argmax coordinates (ints) in source patch space.
    """
    # Forward match: source patch -> target patch (continuous coords)
    fwd_x, fwd_y = find_best_match_window_softargmax(
        similarities, width, height, K=K, temperature=temperature
    )

    # Round and clamp to valid target patch indices
    fwd_patch_x = int(round(fwd_x))
    fwd_patch_y = int(round(fwd_y))
    fwd_patch_x = max(0, min(fwd_patch_x, width - 1))
    fwd_patch_y = max(0, min(fwd_patch_y, height - 1))

    # Backward match: extract predicted target patch descriptor
    tgt_patch = tgt_features[0, fwd_patch_y, fwd_patch_x, :]  # [D]

    # Cosine similarity against all source patches
    _, H_s, W_s, D = src_features.shape
    src_flat = src_features.reshape(H_s * W_s, D)             # [H_s*W_s, D]
    tgt_patch_norm = F.normalize(tgt_patch.unsqueeze(0), dim=1)  # [1, D]
    src_flat_norm = F.normalize(src_flat, dim=1)                  # [H_s*W_s, D]
    back_sims = (src_flat_norm @ tgt_patch_norm.T).squeeze(1)     # [H_s*W_s]

    back_idx = back_sims.argmax().item()
    back_y = back_idx // W_s
    back_x = back_idx % W_s

    return fwd_x, fwd_y, back_x, back_y


def apply_mnn_filter(
    fwd_x: float,
    fwd_y: float,
    back_x: int,
    back_y: int,
    src_patch_x: int,
    src_patch_y: int,
    similarities: torch.Tensor,
    width: int,
    height: int,
    K: int = 5,
    temperature: float = 1.0,
    max_patch_dist: int = 1,
):
    """Accept or reject a forward match based on MNN consistency.

    Computes the L-infinity distance between the backward match location and
    the original source patch.  If the round-trip error is within
    ``max_patch_dist`` patches the forward match is accepted; otherwise the
    best-scoring target patch that does *not* coincide with the failed
    forward match is returned as a fallback.

    Args:
        fwd_x:          Forward soft-argmax x coordinate (float).
        fwd_y:          Forward soft-argmax y coordinate (float).
        back_x:         Backward hard-argmax x coordinate in source space (int).
        back_y:         Backward hard-argmax y coordinate in source space (int).
        src_patch_x:    Original source query patch x coordinate (int).
        src_patch_y:    Original source query patch y coordinate (int).
        similarities:   Flattened similarity vector [H*W].
        width:          Number of patch columns in the target map.
        height:         Number of patch rows in the target map.
        K:              Window size for fallback soft-argmax (must be odd).
        temperature:    Softmax temperature for fallback soft-argmax.
        max_patch_dist: Maximum allowed L-inf round-trip error (default 1).

    Returns:
        Tuple (x, y) of accepted or fallback continuous patch coordinates.
    """
    linf_dist = max(abs(back_x - src_patch_x), abs(back_y - src_patch_y))

    if linf_dist <= max_patch_dist:
        return fwd_x, fwd_y

    # MNN failed: mask the forward match location and find the next best match
    fwd_patch_x = int(round(fwd_x))
    fwd_patch_y = int(round(fwd_y))
    fwd_patch_x = max(0, min(fwd_patch_x, width - 1))
    fwd_patch_y = max(0, min(fwd_patch_y, height - 1))

    masked_sims = similarities.clone()
    masked_sims[fwd_patch_y * width + fwd_patch_x] = -1e9

    return find_best_match_window_softargmax(
        masked_sims, width, height, K=K, temperature=temperature
    )
