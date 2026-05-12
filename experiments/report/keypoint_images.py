import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch
import torch.nn.functional as F
from pathlib import Path
import argparse

from src.features.extractor import extract_dense_features, pixel_to_patch_coord, patch_to_pixel_coord
from src.matching.strategies import find_best_match_argmax, find_best_match_window_softargmax
from src.models.dinov3.dinov3.models.vision_transformer import vit_base


def load_model(checkpoint_path, device, img_size=512, patch_size=16, model_state_dict=True):
    """Load a DINOv3 model from checkpoint"""
    model = vit_base(
        img_size=(img_size, img_size),
        patch_size=patch_size,
        n_storage_tokens=4,
        layerscale_init=1.0,
        mask_k_bias=True,
    )

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device)
        if model_state_dict:
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
        else:
            model.load_state_dict(ckpt, strict=True)

    model.to(device)
    model.eval()
    return model


def predict_keypoint_match(model, src_img_tensor, tgt_img_tensor, src_kp,
                           src_original_size, tgt_original_size,
                           img_size=512, patch_size=16, use_softargmax=False):
    """
    Predict matching keypoint in target image

    Args:
        model: DINOv3 model
        src_img_tensor: source image tensor [1, 3, H, W]
        tgt_img_tensor: target image tensor [1, 3, H, W]
        src_kp: source keypoint [x, y]
        src_original_size: (W, H) of original source image
        tgt_original_size: (W, H) of original target image
        img_size: resized image size
        patch_size: patch size
        use_softargmax: whether to use windowed soft-argmax

    Returns:
        pred_x, pred_y: predicted keypoint coordinates in target image
    """
    # Extract dense features
    src_features = extract_dense_features(model, src_img_tensor)
    tgt_features = extract_dense_features(model, tgt_img_tensor)

    # Reshape target features
    _, H, W, D = tgt_features.shape
    tgt_flat = tgt_features.reshape(H * W, D)

    # Get source keypoint patch coordinates
    src_x, src_y = src_kp
    patch_x, patch_y = pixel_to_patch_coord(
        src_x, src_y, src_original_size,
        patch_size=patch_size, resized_size=img_size
    )

    # Extract source feature at keypoint
    src_feature = src_features[0, patch_y, patch_x, :]

    # Compute cosine similarities
    similarities = F.cosine_similarity(
        src_feature.unsqueeze(0),
        tgt_flat,
        dim=1
    )

    # Find best match
    if use_softargmax:
        match_patch_x, match_patch_y = find_best_match_window_softargmax(
            similarities, W, H, K=5, temperature=0.2
        )
    else:
        match_patch_x, match_patch_y = find_best_match_argmax(similarities, W)

    # Convert to pixel coordinates
    pred_x, pred_y = patch_to_pixel_coord(
        match_patch_x, match_patch_y, tgt_original_size,
        patch_size=patch_size, resized_size=img_size
    )

    return pred_x, pred_y


def load_and_preprocess_image(image_path, img_size=512):
    """Load image and prepare tensor"""
    img = Image.open(image_path).convert('RGB')
    original_size = img.size  # (W, H)

    # Convert to tensor
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)  # [1, 3, H, W]

    # Resize
    img_tensor = F.interpolate(
        img_tensor, size=(img_size, img_size),
        mode='bilinear', align_corners=False
    )

    return img, img_tensor, original_size


def plot_keypoint_comparison(src_img, tgt_img, src_kp, gt_kp, pred_zero, pred_fine,
                             keypoint_name, save_path, show_errors=True):
    """
    Create a 4-panel visualization showing source, target with predictions

    Args:
        src_img: PIL Image (source)
        tgt_img: PIL Image (target)
        src_kp: [x, y] source keypoint
        gt_kp: [x, y] ground truth target keypoint
        pred_zero: [x, y] zero-shot prediction
        pred_fine: [x, y] fine-tuned prediction
        keypoint_name: string describing the keypoint
        save_path: where to save the figure
        show_errors: whether to show error lines
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    marker_size = 12
    line_width = 2.5

    # Panel 1: Source image with keypoint
    axes[0].imshow(src_img)
    axes[0].plot(src_kp[0], src_kp[1], 'go', markersize=marker_size + 3,
                 markeredgecolor='white', markeredgewidth=2.5,
                 label='Source KP', zorder=10)
    axes[0].set_title('Source Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    axes[0].legend(loc='upper right', fontsize=10)

    # Panel 2: Target with Ground Truth
    axes[1].imshow(tgt_img)
    axes[1].plot(gt_kp[0], gt_kp[1], 'go', markersize=marker_size + 3,
                 markeredgecolor='white', markeredgewidth=2.5,
                 label='Ground Truth', zorder=10)
    axes[1].set_title('Target Image\n(Ground Truth)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    axes[1].legend(loc='upper right', fontsize=10)

    # Panel 3: Target with Zero-shot prediction
    axes[2].imshow(tgt_img)
    axes[2].plot(gt_kp[0], gt_kp[1], 'go', markersize=marker_size,
                 markeredgecolor='white', markeredgewidth=2,
                 label='GT', zorder=10, alpha=0.7)
    axes[2].plot(pred_zero[0], pred_zero[1], 'rx', markersize=marker_size + 3,
                 markeredgewidth=3, label='Zero-shot', zorder=11)

    if show_errors:
        axes[2].plot([gt_kp[0], pred_zero[0]], [gt_kp[1], pred_zero[1]],
                     'r--', linewidth=line_width, alpha=0.7, zorder=9)

    # Calculate error
    error_zero = np.sqrt((gt_kp[0] - pred_zero[0]) ** 2 + (gt_kp[1] - pred_zero[1]) ** 2)
    axes[2].set_title(f'Zero-shot Prediction\nError: {error_zero:.1f}px',
                      fontsize=14, fontweight='bold')
    axes[2].axis('off')
    axes[2].legend(loc='upper right', fontsize=10)

    # Panel 4: Target with Fine-tuned prediction
    axes[3].imshow(tgt_img)
    axes[3].plot(gt_kp[0], gt_kp[1], 'go', markersize=marker_size,
                 markeredgecolor='white', markeredgewidth=2,
                 label='GT', zorder=10, alpha=0.7)
    axes[3].plot(pred_fine[0], pred_fine[1], 'bx', markersize=marker_size + 3,
                 markeredgewidth=3, label='Fine-tuned', zorder=11)

    if show_errors:
        axes[3].plot([gt_kp[0], pred_fine[0]], [gt_kp[1], pred_fine[1]],
                     'b--', linewidth=line_width, alpha=0.7, zorder=9)

    # Calculate error
    error_fine = np.sqrt((gt_kp[0] - pred_fine[0]) ** 2 + (gt_kp[1] - pred_fine[1]) ** 2)
    axes[3].set_title(f'Fine-tuned Prediction\nError: {error_fine:.1f}px',
                      fontsize=14, fontweight='bold')
    axes[3].axis('off')
    axes[3].legend(loc='upper right', fontsize=10)

    # Overall title
    plt.suptitle(f'{keypoint_name}', fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def find_example_pairs(results_dir, keypoint_id, category='person', num_examples=2,
                       threshold=0.10, prefer_difficult=True, dataset=None):
    csv_path = Path(results_dir) / 'all_keypoint_metrics.csv'
    df = pd.read_csv(csv_path)

    df['keypoint_id'] = df['keypoint_id'].astype(int)
    if df['correct_at_threshold'].dtype == object:
        df['correct_at_threshold'] = df['correct_at_threshold'].astype(str).str.lower().map({'true': 1, 'false': 0})
    else:
        df['correct_at_threshold'] = df['correct_at_threshold'].astype(int)

    mask = (df['category'] == category) & \
           (df['keypoint_id'] == keypoint_id) & \
           (np.isclose(df['threshold'], threshold))

    df_kp = df[mask].copy()

    if prefer_difficult:
        candidates = df_kp[df_kp['correct_at_threshold'] == 0]
    else:
        candidates = df_kp[df_kp['correct_at_threshold'] == 1]

    if len(candidates) < num_examples:
        print(f"Note: Few candidates for KP {keypoint_id}, using fallback.")
        candidates = df_kp

    sampled_df = candidates.sample(n=min(num_examples * 5, len(candidates)))

    examples = []
    for _, row in sampled_df.iterrows():
        if len(examples) >= num_examples:
            break

        image_idx = int(row['image_idx'])

        if dataset is not None:
            try:
                sample = dataset[image_idx]
                if keypoint_id not in [int(k) for k in sample['kps_ids']]:
                    continue
            except:
                continue

        examples.append({
            'image_idx': image_idx,
            'keypoint_id': keypoint_id,
            'distance': row['distance']
        })

    print(f"Found {len(examples)} random examples for keypoint {keypoint_id} "
          f"({'difficult' if prefer_difficult else 'easy'} cases)")
    return examples

def get_image_info_from_dataset(dataset, image_idx):
    """
    Get image paths and keypoint info from dataset

    Args:
        dataset: SPairDataset instance
        image_idx: index in the dataset

    Returns:
        dict with image paths, keypoints, etc.
    """
    sample = dataset[image_idx]

    return {
        'src_img_path': sample['src_imname'],
        'tgt_img_path': sample['trg_imname'],
        'src_kps': sample['src_kps'].numpy(),
        'tgt_kps': sample['trg_kps'].numpy(),
        'kps_ids': sample['kps_ids'],
        'src_size': (sample['src_imsize'][2], sample['src_imsize'][1]),
        'tgt_size': (sample['trg_imsize'][2], sample['trg_imsize'][1]),
        'category': sample['category']
    }


def create_visualization_grid(examples_easy, examples_hard,
                              dataset, model_zero, model_fine,
                              keypoint_id_easy, keypoint_id_hard,
                              output_path, device, dataset_path, img_size=512, patch_size=16):
    """
    Create a grid visualization with 4 rows:
    - 2 rows for easy keypoint examples
    - 2 rows for hard keypoint examples
    """
    fig = plt.figure(figsize=(20, 20))

    all_examples = [
        ('easy', examples_easy[0], keypoint_id_easy, 0),
        ('easy', examples_easy[1], keypoint_id_easy, 1),
        ('hard', examples_hard[0], keypoint_id_hard, 2),
        ('hard', examples_hard[1], keypoint_id_hard, 3),
    ]

    for row_idx, (difficulty, example, kp_id, plot_row) in enumerate(all_examples):
        # Get image info from dataset
        img_info = get_image_info_from_dataset(dataset, example['image_idx'])
        img_info['kps_ids'] = [int(kp) for kp in img_info['kps_ids']]

        # Verify keypoint is in this sample
        if kp_id not in img_info['kps_ids']:
            print(f"Warning: Keypoint {kp_id} not found in image {example['image_idx']}, skipping")
            continue

        # Find the keypoint index in this sample
        kp_idx = list(img_info['kps_ids']).index(kp_id)
        src_kp = img_info['src_kps'][kp_idx]
        gt_kp = img_info['tgt_kps'][kp_idx]

        base_path = Path(dataset_path)
        full_src_path = base_path / 'JPEGImages' / 'person' /img_info['src_img_path']
        full_tgt_path = base_path / 'JPEGImages' / 'person' /img_info['tgt_img_path']

        # Load images
        src_img, src_tensor, src_size = load_and_preprocess_image(
            str(full_src_path), img_size
        )
        tgt_img, tgt_tensor, tgt_size = load_and_preprocess_image(
            str(full_tgt_path), img_size
        )

        src_tensor = src_tensor.to(device)
        tgt_tensor = tgt_tensor.to(device)

        # Get predictions from zero-shot model
        with torch.no_grad():
            pred_zero_x, pred_zero_y = predict_keypoint_match(
                model_zero, src_tensor, tgt_tensor, src_kp,
                src_size, tgt_size, img_size, patch_size, use_softargmax=False
            )

        # Get predictions from fine-tuned model
        with torch.no_grad():
            pred_fine_x, pred_fine_y = predict_keypoint_match(
                model_fine, src_tensor, tgt_tensor, src_kp,
                src_size, tgt_size, img_size, patch_size, use_softargmax=True
            )

        # Plot panels for this row
        marker_size = 10

        # Panel 1: Source
        ax = plt.subplot(4, 4, plot_row * 4 + 1)
        ax.imshow(src_img)
        ax.plot(src_kp[0], src_kp[1], 'go', markersize=marker_size + 2,
                markeredgecolor='white', markeredgewidth=2, zorder=10)
        if plot_row == 0:
            ax.set_title('Source', fontsize=12, fontweight='bold')
        if plot_row % 2 == 0:
            label = f"KP {kp_id}\n({'Easy' if difficulty == 'easy' else 'Hard'})"
            ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.axis('off')

        # Panel 2: Ground Truth
        ax = plt.subplot(4, 4, plot_row * 4 + 2)
        ax.imshow(tgt_img)
        ax.plot(gt_kp[0], gt_kp[1], 'go', markersize=marker_size + 2,
                markeredgecolor='white', markeredgewidth=2, zorder=10)
        if plot_row == 0:
            ax.set_title('Ground Truth', fontsize=12, fontweight='bold')
        ax.axis('off')

        # Panel 3: Zero-shot
        ax = plt.subplot(4, 4, plot_row * 4 + 3)
        ax.imshow(tgt_img)
        ax.plot(gt_kp[0], gt_kp[1], 'go', markersize=marker_size,
                markeredgecolor='white', markeredgewidth=1.5, alpha=0.6, zorder=9)
        ax.plot(pred_zero_x, pred_zero_y, 'rx', markersize=marker_size + 2,
                markeredgewidth=2.5, zorder=10)
        ax.plot([gt_kp[0], pred_zero_x], [gt_kp[1], pred_zero_y],
                'r--', linewidth=2, alpha=0.6, zorder=8)

        error_zero = np.sqrt((gt_kp[0] - pred_zero_x) ** 2 + (gt_kp[1] - pred_zero_y) ** 2)
        if plot_row == 0:
            ax.set_title('Zero-shot', fontsize=12, fontweight='bold')
        ax.text(0.5, -0.1, f'Err: {error_zero:.1f}px',
                transform=ax.transAxes, ha='center', fontsize=9, color='red')
        ax.axis('off')

        # Panel 4: Fine-tuned
        ax = plt.subplot(4, 4, plot_row * 4 + 4)
        ax.imshow(tgt_img)
        ax.plot(gt_kp[0], gt_kp[1], 'go', markersize=marker_size,
                markeredgecolor='white', markeredgewidth=1.5, alpha=0.6, zorder=9)
        ax.plot(pred_fine_x, pred_fine_y, 'bx', markersize=marker_size + 2,
                markeredgewidth=2.5, zorder=10)
        ax.plot([gt_kp[0], pred_fine_x], [gt_kp[1], pred_fine_y],
                'b--', linewidth=2, alpha=0.6, zorder=8)

        error_fine = np.sqrt((gt_kp[0] - pred_fine_x) ** 2 + (gt_kp[1] - pred_fine_y) ** 2)
        if plot_row == 0:
            ax.set_title('Fine-tuned + Soft-argmax', fontsize=12, fontweight='bold')
        ax.text(0.5, -0.1, f'Err: {error_fine:.1f}px',
                transform=ax.transAxes, ha='center', fontsize=9, color='blue')
        ax.axis('off')

    plt.suptitle('Keypoint Localization Examples: Easy vs Hard Keypoints\n'
                 '(Person Category)',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization grid: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize keypoint prediction examples"
    )
    parser.add_argument('--results_zeroshot', type=str, required=True,
                        help='Path to zero-shot results directory')
    parser.add_argument('--checkpoint_zeroshot', type=str, default=None,
                        help='Path to zero-shot checkpoint (None for random init)')
    parser.add_argument('--checkpoint_finetuned', type=str, required=True,
                        help='Path to fine-tuned checkpoint')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to SPair-71k dataset')
    parser.add_argument('--output_dir', type=str, default='figures2',
                        help='Directory to save visualizations')
    parser.add_argument('--keypoint_easy', type=int, default=0,
                        help='Keypoint ID for easy examples')
    parser.add_argument('--keypoint_hard', type=int, default=19,
                        help='Keypoint ID for hard examples')
    parser.add_argument('--img_size', type=int, default=512,
                        help='Image size for processing')
    parser.add_argument('--patch_size', type=int, default=16,
                        help='Patch size')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("=" * 80)
    print("GENERATING KEYPOINT VISUALIZATION EXAMPLES")
    print("=" * 80)

    # Load models
    print("\n[1/5] Loading models...")
    model_zero = load_model(args.checkpoint_zeroshot, device, args.img_size, args.patch_size, model_state_dict=False)
    model_fine = load_model(args.checkpoint_finetuned, device, args.img_size, args.patch_size, model_state_dict=True)
    print("Models loaded")

    # Load dataset
    print("\n[2/5] Loading dataset...")
    from src.datasets.spair_dataset import SPairDataset

    base = args.dataset_path
    dataset = SPairDataset(
        f'{base}/PairAnnotation',
        f'{base}/Layout',
        f'{base}/JPEGImages',
        'large',
        0.1,
        datatype='test'
    )
    print(f"Loaded dataset with {len(dataset)} pairs")

    # Find example pairs
    print("\n[3/5] Finding example pairs...")
    examples_easy = find_example_pairs(
        args.results_zeroshot, args.keypoint_easy,
        category='person', num_examples=2, prefer_difficult=False,
        dataset=dataset
    )

    examples_hard = find_example_pairs(
        args.results_zeroshot, args.keypoint_hard,
        category='person', num_examples=2, prefer_difficult=True,
        dataset=dataset
    )

    # Create visualization grid
    kpts = f'kpe_{args.keypoint_easy}_kph_{args.keypoint_hard}'
    output_dir_kp = Path(args.output_dir+'\\'+kpts)
    output_dir_kp.mkdir(exist_ok=True, parents=True)
    print("\n[4/5] Creating visualization grid...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_filename = f'keypoint_grid_{timestamp}.png'
    grid_path = output_dir / kpts / grid_filename
    create_visualization_grid(
        examples_easy, examples_hard,
        dataset, model_zero, model_fine,
        args.keypoint_easy, args.keypoint_hard,
        grid_path, device, args.dataset_path, args.img_size, args.patch_size
    )

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE!")
    print("=" * 80)
    print(f"\nOutput saved to: {grid_path}")


if __name__ == "__main__":
    main()
