import json
from collections import defaultdict
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from datetime import datetime
import torch.nn.functional as F

from src.datasets.spair_dataset import SPairDataset
from src.features.extractor import extract_dense_features, pixel_to_patch_coord, patch_to_pixel_coord
from src.matching.strategies import find_best_match_argmax
from src.metrics.pck import compute_pck_spair71k
from src.models.dinov2.dinov2.models.vision_transformer import vit_base
import configs.paths as paths


def freeze_model(model):
    """Freeze all model parameters"""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_last_n_blocks(model, n_blocks):
    """
    Unfreeze the last n_blocks transformer blocks + final norm layer

    Args:
        model: DINOv2 model
        n_blocks: number of blocks to unfreeze (counting from the end)
    """
    total_blocks = len(model.blocks)

    # Unfreeze last n blocks
    for i in range(total_blocks - n_blocks, total_blocks):
        for param in model.blocks[i].parameters():
            param.requires_grad = True

    # Also unfreeze the final normalization layer
    for param in model.norm.parameters():
        param.requires_grad = True

    print(f"Unfrozen last {n_blocks} blocks + norm layer")


def compute_cross_entropy_loss(src_features, tgt_features, src_kps, trg_kps,
                               src_original_size, tgt_original_size, img_size, patch_size, temperature=10.0):
    """
    Compute cross-entropy loss for semantic correspondence.
    Treats correspondence as a classification problem where each target patch is a class.

    Args:
        src_features: [1, H, W, D] source dense features
        tgt_features: [1, H, W, D] target dense features
        src_kps: [N, 2] source keypoints in pixel coordinates
        trg_kps: [N, 2] target keypoints in pixel coordinates
        src_original_size: (width, height) of original source image
        tgt_original_size: (width, height) of original target image
        img_size: resizing size used during feature extraction
        patch_size: size of each patch
        temperature: softmax temperature (higher = more peaked distribution)

    Returns:
        loss: mean cross-entropy loss across all keypoints
    """
    _, H, W, D = tgt_features.shape
    tgt_flat = tgt_features.reshape(H * W, D)  # [H*W, D]

    losses = []

    for i in range(src_kps.shape[0]):
        src_x, src_y = src_kps[i]
        tgt_x, tgt_y = trg_kps[i]

        # Get source feature at keypoint location
        src_patch_x, src_patch_y = pixel_to_patch_coord(src_x, src_y, src_original_size, patch_size=patch_size, resized_size=img_size)
        src_feature = src_features[0, src_patch_y, src_patch_x, :]  # [D]

        # Get ground truth target patch coordinates
        tgt_patch_x, tgt_patch_y = pixel_to_patch_coord(tgt_x, tgt_y, tgt_original_size, patch_size=patch_size, resized_size=img_size)
        # Compute cosine similarities with all target patches
        similarities = F.cosine_similarity(
            src_feature.unsqueeze(0),  # [1, D]
            tgt_flat,  # [H*W, D]
            dim=1
        )  # [H*W]

        # Convert similarities to log-probabilities
        log_probs = F.log_softmax(similarities * temperature, dim=0)

        # Ground truth index (flatten 2D coordinates to 1D)
        gt_idx = tgt_patch_y * W + tgt_patch_x

        # Negative log-likelihood loss
        loss = -log_probs[gt_idx]
        losses.append(loss)

    return torch.stack(losses).mean()


def train_epoch(model, dataloader, optimizer, device, epoch, img_size=518, patch_size=14, temperature=10.0, scheduler=None):
    """
    Train for one epoch

    Args:
        model: DINOv2 model
        dataloader: training data loader
        optimizer: optimizer
        device: 'cuda' or 'cpu'
        epoch: current epoch number
        img_size: size to which images are resized for feature extraction
        patch_size: size of each patch
        temperature: softmax temperature for loss

    Returns:
        avg_loss: average loss over the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0

    for idx, sample in enumerate(dataloader):
        # Prepare data
        src_tensor = sample['src_img'].to(device)  # [1, 3, H, W]
        tgt_tensor = sample['trg_img'].to(device)  # [1, 3, H, W]

        # Resize to 518x518 (DINOv2 expects this size)
        src_tensor = F.interpolate(src_tensor, size=(img_size, img_size), mode='bilinear', align_corners=False)
        tgt_tensor = F.interpolate(tgt_tensor, size=(img_size, img_size), mode='bilinear', align_corners=False)

        # Store original sizes for coordinate conversion
        src_original_size = (sample['src_imsize'][2], sample['src_imsize'][1])
        tgt_original_size = (sample['trg_imsize'][2], sample['trg_imsize'][1])

        # Get keypoints
        src_kps = sample['src_kps'].numpy()[0]  # [N, 2]
        trg_kps = sample['trg_kps'].numpy()[0]  # [N, 2]

        # Extract dense features
        src_features = extract_dense_features(model, src_tensor, training=True)
        tgt_features = extract_dense_features(model, tgt_tensor, training=True)

        # Compute loss
        loss = compute_cross_entropy_loss(
            src_features, tgt_features,
            src_kps, trg_kps,
            src_original_size, tgt_original_size,
            img_size, patch_size,
            temperature=temperature
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        # Print progress
        if (idx + 1) % 50 == 0:
            print(f"Epoch {epoch}, Batch {idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, val_dataset, device, img_size=518, patch_size=14,
             threshold=0.1, max_samples=500):
    """Evaluate mean PCK@threshold on a validation set using argmax matching.

    Args:
        model: DINOv2 model.
        val_dataset: Iterable of sample dicts (SPair-71k format), no DataLoader needed.
        device: torch.device for tensor placement.
        img_size: Image size after bilinear interpolation.
        patch_size: Patch size in pixels.
        threshold: PCK threshold (e.g. 0.1).
        max_samples: Stop after this many pairs (None = full dataset).

    Returns:
        Mean PCK@threshold as a float (0–100).
    """
    model.eval()
    all_pck = []

    with torch.no_grad():
        for idx, sample in enumerate(val_dataset):
            if max_samples is not None and idx >= max_samples:
                break

            src_tensor = sample['src_img'].unsqueeze(0).to(device)
            tgt_tensor = sample['trg_img'].unsqueeze(0).to(device)

            src_tensor = F.interpolate(src_tensor, size=(img_size, img_size), mode='bilinear', align_corners=False)
            tgt_tensor = F.interpolate(tgt_tensor, size=(img_size, img_size), mode='bilinear', align_corners=False)

            src_original_size = (sample['src_imsize'][2], sample['src_imsize'][1])
            tgt_original_size = (sample['trg_imsize'][2], sample['trg_imsize'][1])

            src_features = extract_dense_features(model, src_tensor)
            tgt_features = extract_dense_features(model, tgt_tensor)

            _, H, W, D = tgt_features.shape
            tgt_flat = tgt_features.reshape(H * W, D)

            src_kps = sample['src_kps'].numpy()
            trg_kps = sample['trg_kps'].numpy()
            trg_bbox = sample['trg_bbox']

            pred_matches = []
            for i in range(src_kps.shape[0]):
                src_x, src_y = src_kps[i]
                kp_patch_x, kp_patch_y = pixel_to_patch_coord(src_x, src_y, src_original_size, patch_size, img_size)
                src_feature = src_features[0, kp_patch_y, kp_patch_x, :]
                similarities = F.cosine_similarity(src_feature.unsqueeze(0), tgt_flat, dim=1)
                match_patch_x, match_patch_y = find_best_match_argmax(similarities, W)
                match_x, match_y = patch_to_pixel_coord(match_patch_x, match_patch_y, tgt_original_size, patch_size, img_size)
                pred_matches.append([match_x, match_y])

            pck, _, _ = compute_pck_spair71k(pred_matches, trg_kps.tolist(), trg_bbox, threshold)
            all_pck.append(pck)

    model.train()
    return float(np.mean(all_pck))


def train_with_scheduler(model, train_loader, val_dataset, optimizer, device,
                          num_epochs, img_size, patch_size, temperature,
                          results_dir, n_blocks, patience=2, warmup_steps=100):
    """Train with linear warmup + cosine annealing LR schedule and early stopping.

    The scheduler operates at the optimizer-step level: LinearLR ramps the LR
    from (1/warmup_steps) to 1.0 over the first warmup_steps gradient steps,
    then CosineAnnealingLR decays it for the remaining steps.

    After each epoch, validate() is called and the best checkpoint is saved.
    Training stops early if val PCK does not improve for `patience` epochs.

    Args:
        model: DINOv2 model (partially unfrozen).
        train_loader: DataLoader for the training split.
        val_dataset: Iterable for the validation split (no DataLoader).
        optimizer: Configured optimizer (e.g. AdamW).
        device: torch.device.
        num_epochs: Maximum number of epochs to train.
        img_size: Image resize target (pixels).
        patch_size: Patch size in pixels.
        temperature: Softmax temperature for the cross-entropy loss.
        results_dir: Directory where best_checkpoint.pth will be saved.
        n_blocks: Number of unfrozen transformer blocks (stored in checkpoint).
        patience: Epochs without improvement before early stopping.
        warmup_steps: Number of gradient steps for the linear warmup phase.

    Returns:
        Tuple (best_pck, training_history) where best_pck is the highest
        val PCK@0.1 seen and training_history is a list of per-epoch dicts
        {epoch, train_loss, val_pck, lr}.
    """
    os.makedirs(results_dir, exist_ok=True)

    total_steps = num_epochs * len(train_loader)
    remaining_steps = max(total_steps - warmup_steps, 1)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0 / warmup_steps, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=remaining_steps
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
    )

    best_pck = 0.0
    best_epoch = 0
    no_improve_count = 0
    training_history = []

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print('=' * 60)

        train_loss = train_epoch(
            model, train_loader, optimizer, device, epoch,
            img_size=img_size, patch_size=patch_size,
            temperature=temperature, scheduler=scheduler
        )

        val_pck = validate(model, val_dataset, device, img_size=img_size, patch_size=patch_size)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch}: train_loss={train_loss:.4f}  val_pck@0.1={val_pck:.2f}%  lr={current_lr:.2e}")

        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_pck': val_pck,
            'lr': current_lr,
        })

        if val_pck > best_pck:
            best_pck = val_pck
            best_epoch = epoch
            no_improve_count = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_pck': val_pck,
                'n_blocks': n_blocks,
            }, f'{results_dir}/best_checkpoint.pth')
            print(f"New best model saved — PCK@0.1: {best_pck:.2f}%")
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"\nEarly stopping after epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"\nTraining finished. Best PCK@0.1: {best_pck:.2f}% at epoch {best_epoch}")
    return best_pck, training_history


def main():
    """Main training and evaluation pipeline"""

    # ========== CONFIGURATION ==========
    num_epochs = 5
    learning_rate = 1e-4
    batch_size = 1  #SPair-71k has variable-sized images
    temperature = 10  #softmax temperature for cross-entropy loss
    patience = 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'{paths.RESULTS_STEP2}/dinov2_finetuned_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    # ========== LOAD DATASETS ==========
    print("\nLoading SPair-71k dataset...")
    base = paths.SPAIR71K_ROOT

    train_dataset = SPairDataset(
        paths.SPAIR71K_PAIRS,
        paths.SPAIR71K_LAYOUT,
        paths.SPAIR71K_IMAGES,
        'large',
        0.1,  # dummy pck_alpha, not used during training
        datatype='trn'  # training split
    )

    val_dataset = SPairDataset(
        paths.SPAIR71K_PAIRS,
        paths.SPAIR71K_LAYOUT,
        paths.SPAIR71K_IMAGES,
        'large',
        0.1,
        datatype='val'
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True if device == 'cuda' else False
    )

    for n_blocks in [1,2,3,4]:
        print("\n" + "=" * 80)
        print(f"FINETUNING WITH LAST {n_blocks} BLOCKS UNFROZEN")
        print("=" * 80)
        # ========== LOAD MODEL ==========
        print("\nLoading DINOv2-base model...")
        model = vit_base(
            img_size=(518, 518),
            patch_size=14,
            num_register_tokens=0,
            block_chunks=0,
            init_values=1.0,
        )

        # load pretrained weights
        ckpt = torch.load(paths.DINOV2_WEIGHTS, map_location=device)
        model.load_state_dict(ckpt, strict=True)
        model.to(device)

        # freeze entire model, then unfreeze last N blocks
        freeze_model(model)
        unfreeze_last_n_blocks(model, n_blocks)

        # count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.2f}%)")


        # ========== OPTIMIZER ==========
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=0.01
        )

        # ========== TRAINING LOOP ==========
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)

        n_blocks_results_dir = f'{results_dir}/{n_blocks}blocks'
        best_pck, training_history = train_with_scheduler(
            model, train_loader, val_dataset, optimizer, device,
            num_epochs=num_epochs, img_size=518, patch_size=14, temperature=temperature,
            results_dir=n_blocks_results_dir, n_blocks=n_blocks, patience=patience
        )

        print(f"\nFinetuning finished for {n_blocks} unfrozen blocks. Best PCK@0.1: {best_pck:.2f}%")

        with open(f'{n_blocks_results_dir}/training_history.json', 'w') as f:
            json.dump(training_history, f, indent=2)


if __name__ == "__main__":
    main()
