import torch


def extract_dense_features(model, img_tensor, training=False):
    """Extract dense features from DINOv2 model given an input image tensor."""
    context = torch.no_grad() if not training else torch.enable_grad()

    with context:
        #get tokens
        features_dict = model.forward_features(img_tensor)
        patch_tokens = features_dict['x_norm_patchtokens']  # [B, N_patches, D]

        #reshaping to dense feature map
        B, N, D = patch_tokens.shape
        H_patches = W_patches = int(N ** 0.5)  # per img 518x518 con patch 14: 37x37
        dense_features = patch_tokens.reshape(B, H_patches, W_patches, D)
    return dense_features

def extract_dense_features_SAM(model, img_tensor, training=False, image_size=1024):
    """Extract dense features from SAM encoder given an input image tensor."""
    import torch.nn.functional as F
    from torch.cuda.amp import autocast
    import torch.nn as nn

    context = torch.no_grad() if not training else torch.enable_grad()

    with context:
        img_resized = F.interpolate(img_tensor, size=(image_size, image_size), mode='bilinear', align_corners=False)

        # Interpolate positional embeddings if using non-standard size
        if image_size != 1024:
            original_pos_embed = model.image_encoder.pos_embed

            # pos_embed shape: [1, 64, 64, 1024] for 1024x1024 input
            batch_size, old_grid_size, _, embed_dim = original_pos_embed.shape
            new_grid_size = image_size // 16

            # reshape and interpolate: [1, 64, 64, 1024] -> [1, 1024, 64, 64] -> [1, 1024, new_grid_size, 32]
            pos_tokens_resized = F.interpolate(
                original_pos_embed.permute(0, 3, 1, 2),
                size=(new_grid_size, new_grid_size),
                mode='bicubic',
                align_corners=False
            ).permute(0, 2, 3, 1)

            model.image_encoder.pos_embed = nn.Parameter(pos_tokens_resized, requires_grad=training)

        # Use autocast only in eval mode for speed
        if training:
            embeddings = model.image_encoder(img_resized)  # [1, 256, H, W]
        else:
            with autocast():
                embeddings = model.image_encoder(img_resized)

        # restore original positional embeddings
        if image_size != 1024:
            model.image_encoder.pos_embed = original_pos_embed

        # reshape to [1, H_patches, W_patches, D]
        dense_features = embeddings.permute(0, 2, 3, 1)

    return dense_features

def extract_dense_features_SAM_dep(model, img_tensor, training=False, image_size=1024):
    """Extract dense features from SAM encoder with flexible input size."""
    import torch.nn.functional as F
    from torch.cuda.amp import autocast
    import torch.nn as nn

    if training:
        raise NotImplementedError("Training mode not implemented for SAM feature extraction.")

    with torch.no_grad():
        img_resized = F.interpolate(img_tensor, size=(image_size, image_size), mode='bilinear', align_corners=False)

        if image_size != 1024:
            original_pos_embed = model.image_encoder.pos_embed

            # Shape: [1, 64, 64, 1024] = [batch, height, width, embed_dim]
            batch_size, old_grid_h, old_grid_w, embed_dim = original_pos_embed.shape

            if old_grid_h != old_grid_w:
                raise ValueError(f"Expected square grid, got {old_grid_h}x{old_grid_w}")

            old_grid_size = old_grid_h  # 64
            new_grid_size = image_size // 16  # 32 per 512x512

            # Permute per interpolazione: [1, 64, 64, 1024] -> [1, 1024, 64, 64]
            pos_tokens = original_pos_embed.permute(0, 3, 1, 2)

            # Interpolazione bicubica
            pos_tokens_resized = F.interpolate(
                pos_tokens,
                size=(new_grid_size, new_grid_size),
                mode='bicubic',
                align_corners=False
            )

            # Ripristina formato originale: [1, 1024, 32, 32] -> [1, 32, 32, 1024]
            pos_tokens_resized = pos_tokens_resized.permute(0, 2, 3, 1)

            # Wrappa in nn.Parameter (necessario per l'assegnazione)
            model.image_encoder.pos_embed = nn.Parameter(pos_tokens_resized, requires_grad=False)

        with autocast():
            embeddings = model.image_encoder(img_resized)

        # Ripristina pos_embed originale
        if image_size != 1024:
            model.image_encoder.pos_embed = original_pos_embed

        # Reshape output
        dense_features = embeddings.permute(0, 2, 3, 1)

    return dense_features


def extract_layer_features(model, img_tensor, layer_idx):
    with torch.no_grad():

        # get_intermediate_layers returns patch tokens only (CLS + storage are already stripped)
        patch_tokens = model.get_intermediate_layers(img_tensor, n=[layer_idx], norm=True)[0]  # [B, N_patches, D]

        #reshaping to dense feature map
        B, N, D = patch_tokens.shape
        H_patches = W_patches = int(N ** 0.5)
        dense_features = patch_tokens.reshape(B, H_patches, W_patches, D)
    return dense_features


def pixel_to_patch_coord(x, y, original_size, patch_size=14, resized_size=518):
    """convert pixel coordinates to patch coordinates"""
    #scale to resized image
    scale_x = resized_size / original_size[0]
    scale_y = resized_size / original_size[1]
    x_resized = x * scale_x
    y_resized = y * scale_y

    #compute patch coordinates
    patch_x = int(x_resized // patch_size)
    patch_y = int(y_resized // patch_size)

    #clamp to valid range
    max_patch = resized_size // patch_size - 1
    patch_x = min(max(patch_x, 0), max_patch)
    patch_y = min(max(patch_y, 0), max_patch)

    return patch_x, patch_y


def patch_to_pixel_coord(patch_x, patch_y, original_size, patch_size=14, resized_size=518):
    """Convert patch coordinates back to pixel coordinates with a centering strategy"""
    #center of the patch in resized image
    x_resized = patch_x * patch_size + patch_size / 2
    y_resized = patch_y * patch_size + patch_size / 2

    #scale back to original image size
    scale_x = original_size[0] / resized_size
    scale_y = original_size[1] / resized_size
    x = x_resized * scale_x
    y = y_resized * scale_y

    return x, y


def extract_dense_features_multilayer(model, img_tensor, n_last_layers=3, training=False):
    """Extract and average dense features from the last n transformer layers of a DINOv2 model.

    Args:
        model: DINOv2 model with get_intermediate_layers support.
        img_tensor: Input image tensor [B, C, H, W].
        n_last_layers: Number of last layers to aggregate (default 3).
        training: If False, wraps forward pass in torch.no_grad().

    Returns:
        Tensor of shape [B, H_patches, W_patches, D] — averaged over the selected layers.
    """
    context = torch.no_grad() if not training else torch.enable_grad()

    with context:
        layers = model.get_intermediate_layers(
            img_tensor, n=n_last_layers, return_class_token=False, norm=True
        )  # list of n_last_layers tensors, each [B, N, D]

        stacked = torch.stack(layers, dim=0)  # [n_layers, B, N, D]
        averaged = stacked.mean(dim=0)        # [B, N, D]

        B, N, D = averaged.shape
        H_patches = W_patches = int(N ** 0.5)
        dense_features = averaged.reshape(B, H_patches, W_patches, D)

    return dense_features


def apply_pca_whitening(src_features, tgt_features, n_components=64):
    """Apply PCA with whitening to source and target feature maps using the target distribution.

    Fits PCA on the target features, then transforms both source and target with the same
    projection so that distances are comparable across the two maps.

    Args:
        src_features: Source feature tensor [1, H, W, D].
        tgt_features: Target feature tensor [1, H, W, D].
        n_components: Number of PCA components to keep (default 64).
                      Clamped to min(n_components, H*W, D) to avoid rank errors.

    Returns:
        Tuple (src_pca, tgt_pca) each of shape [1, H, W, K] as float32 tensors
        on the same device as the inputs, where K = effective n_components.
    """
    from sklearn.decomposition import PCA

    device = tgt_features.device

    _, H, W, D = tgt_features.shape
    n_pixels = H * W

    tgt_np = tgt_features.squeeze(0).reshape(n_pixels, D).cpu().float().numpy()
    src_np = src_features.squeeze(0).reshape(n_pixels, D).cpu().float().numpy()

    k = min(n_components, n_pixels, D)
    pca = PCA(n_components=k, whiten=True)
    pca.fit(tgt_np)

    tgt_pca = pca.transform(tgt_np)  # [H*W, K]
    src_pca = pca.transform(src_np)  # [H*W, K]

    tgt_out = torch.from_numpy(tgt_pca).float().reshape(1, H, W, k).to(device)
    src_out = torch.from_numpy(src_pca).float().reshape(1, H, W, k).to(device)

    return src_out, tgt_out
