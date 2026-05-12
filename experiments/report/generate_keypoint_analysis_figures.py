import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path
import argparse
from sklearn.decomposition import PCA
import cv2


def load_spair_image(dataset_path, image_name):
    """Carica un'immagine dal dataset SPair-71k"""
    img_path = Path(dataset_path) / 'JPEGImages' / image_name
    if img_path.exists():
        return Image.open(img_path).convert('RGB')
    return None


def visualize_keypoint_pair(source_img, target_img, source_kp, target_kp,
                            pred_kp, keypoint_name, distance, is_difficult=True):
    """
    Visualizza una coppia source-target con keypoint GT e predetto

    Args:
        source_img: PIL Image or numpy array
        target_img: PIL Image or numpy array
        source_kp: (x, y) tuple
        target_kp: (x, y) tuple - ground truth
        pred_kp: (x, y) tuple - prediction
        keypoint_name: string
        distance: normalized distance error
        is_difficult: bool, se True evidenzia problemi (occlusioni, etc)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Source image
    ax1.imshow(source_img)
    ax1.plot(source_kp[0], source_kp[1], 'go', markersize=15,
             markeredgecolor='white', markeredgewidth=2.5, label='Source KP', zorder=10)
    ax1.set_title(f'Source Image\nKeypoint: {keypoint_name}',
                  fontsize=13, fontweight='bold')
    ax1.axis('off')
    ax1.legend(loc='upper right', fontsize=10)

    # Target image
    ax2.imshow(target_img)
    ax2.plot(target_kp[0], target_kp[1], 'go', markersize=15,
             markeredgecolor='white', markeredgewidth=2.5, label='Ground Truth', zorder=10)
    ax2.plot(pred_kp[0], pred_kp[1], 'rx', markersize=15,
             markeredgewidth=3, label='Prediction', zorder=10)

    # Error line
    ax2.plot([target_kp[0], pred_kp[0]], [target_kp[1], pred_kp[1]],
             'r--', linewidth=2.5, alpha=0.8, zorder=9)

    # Highlight problematic area for difficult cases
    if is_difficult and distance > 0.15:
        circle = patches.Circle(target_kp, 60, fill=False,
                                edgecolor='yellow', linewidth=3,
                                linestyle='--', zorder=8)
        ax2.add_patch(circle)

        # Add annotation explaining the difficulty
        annotations = {
            'wrist': 'Occluded/\nSmall area',
            'ankle': 'Articulation/\nOcclusion',
            'shoulder': 'Side view/\nAmbiguity',
            'knee': 'Unusual pose'
        }

        annotation_text = 'Challenging\nregion'
        for key in annotations:
            if key in keypoint_name.lower():
                annotation_text = annotations[key]
                break

        ax2.text(target_kp[0], target_kp[1] - 70, annotation_text,
                 color='yellow', fontweight='bold', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
                 ha='center', zorder=11)

    error_pct = distance * 100
    ax2.set_title(f'Target Image\nNormalized Error: {distance:.3f} ({error_pct:.1f}%)',
                  fontsize=13, fontweight='bold')
    ax2.axis('off')
    ax2.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    return fig


def create_difficulty_comparison_grid(examples_difficult, examples_easy, save_path):
    """
    Crea una griglia 2x3: sopra esempi difficili, sotto esempi facili

    Args:
        examples_difficult: lista di dict con {source_img, target_img, source_kp,
                           target_kp, pred_kp, kp_name, distance}
        examples_easy: stessa struttura
    """
    fig = plt.figure(figsize=(18, 12))

    # Difficult examples (top row)
    for i, example in enumerate(examples_difficult[:3]):
        # Source
        ax = plt.subplot(4, 3, i + 1)
        ax.imshow(example['source_img'])
        ax.plot(example['source_kp'][0], example['source_kp'][1],
                'go', markersize=12, markeredgecolor='white', markeredgewidth=2)
        if i == 0:
            ax.set_ylabel('Difficult\nKeypoints\n\nSource', fontsize=12, fontweight='bold')
        ax.set_title(f"{example['kp_name']}", fontsize=11, fontweight='bold')
        ax.axis('off')

        # Target
        ax = plt.subplot(4, 3, i + 4)
        ax.imshow(example['target_img'])
        ax.plot(example['target_kp'][0], example['target_kp'][1],
                'go', markersize=12, markeredgecolor='white', markeredgewidth=2, label='GT')
        ax.plot(example['pred_kp'][0], example['pred_kp'][1],
                'rx', markersize=12, markeredgewidth=2.5, label='Pred')
        ax.plot([example['target_kp'][0], example['pred_kp'][0]],
                [example['target_kp'][1], example['pred_kp'][1]],
                'r--', linewidth=2, alpha=0.7)

        # Highlight error
        circle = patches.Circle(example['target_kp'], 50, fill=False,
                                edgecolor='red', linewidth=2.5, linestyle='--', alpha=0.8)
        ax.add_patch(circle)

        if i == 0:
            ax.set_ylabel('Target', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
        ax.set_title(f"Error: {example['distance']:.3f}", fontsize=10, color='red', fontweight='bold')
        ax.axis('off')

    # Easy examples (bottom row)
    for i, example in enumerate(examples_easy[:3]):
        # Source
        ax = plt.subplot(4, 3, i + 7)
        ax.imshow(example['source_img'])
        ax.plot(example['source_kp'][0], example['source_kp'][1],
                'go', markersize=12, markeredgecolor='white', markeredgewidth=2)
        if i == 0:
            ax.set_ylabel('Easy\nKeypoints\n\nSource', fontsize=12, fontweight='bold')
        ax.set_title(f"{example['kp_name']}", fontsize=11, fontweight='bold')
        ax.axis('off')

        # Target
        ax = plt.subplot(4, 3, i + 10)
        ax.imshow(example['target_img'])
        ax.plot(example['target_kp'][0], example['target_kp'][1],
                'go', markersize=12, markeredgecolor='white', markeredgewidth=2, label='GT')
        ax.plot(example['pred_kp'][0], example['pred_kp'][1],
                'gx', markersize=12, markeredgewidth=2.5, label='Pred')
        ax.plot([example['target_kp'][0], example['pred_kp'][0]],
                [example['target_kp'][1], example['pred_kp'][1]],
                'g--', linewidth=2, alpha=0.7)

        # Highlight success
        circle = patches.Circle(example['target_kp'], 30, fill=False,
                                edgecolor='green', linewidth=2.5, linestyle='-', alpha=0.8)
        ax.add_patch(circle)

        if i == 0:
            ax.set_ylabel('Target', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
        ax.set_title(f"Error: {example['distance']:.3f}", fontsize=10, color='green', fontweight='bold')
        ax.axis('off')

    plt.suptitle('Challenging vs. Easy Keypoint Examples\n(Person Category)',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved difficulty comparison grid to: {save_path}")
    plt.close()


def plot_per_keypoint_comparison(results_dir_zeroshot, results_dir_finetuned, save_dir):
    """
    Crea visualizzazioni comparative dettagliate per keypoint
    """
    # Load data
    df_zero = pd.read_csv(f'{results_dir_zeroshot}/keypoint_stats/person_pck0.10.csv')
    df_fine = pd.read_csv(f'{results_dir_finetuned}/keypoint_stats/person_pck0.10.csv')

    # Merge
    df_merged = df_zero.merge(df_fine, on='keypoint_id', suffixes=('_zero', '_fine'))
    df_merged['improvement'] = df_merged['accuracy_pct_fine'] - df_merged['accuracy_pct_zero']
    df_merged = df_merged.sort_values('keypoint_id')

    # Create comprehensive comparison figure
    fig = plt.figure(figsize=(16, 10))

    # 1. Bar chart comparison (top left)
    ax1 = plt.subplot(2, 2, 1)
    x = np.arange(len(df_merged))
    width = 0.35
    bars1 = ax1.bar(x - width / 2, df_merged['accuracy_pct_zero'], width,
                    label='Zero-shot', color='#ff6b6b', alpha=0.85, edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x + width / 2, df_merged['accuracy_pct_fine'], width,
                    label='Fine-tuned', color='#51cf66', alpha=0.85, edgecolor='black', linewidth=0.5)

    ax1.set_xlabel('Keypoint ID', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('(a) Per-Keypoint Performance Comparison', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_merged['keypoint_id'], rotation=45, ha='right', fontsize=9)
    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.set_ylim(0, 100)

    # Highlight top 3 improvements
    top_improvements = df_merged.nlargest(3, 'improvement')
    for _, row in top_improvements.iterrows():
        idx = df_merged[df_merged['keypoint_id'] == row['keypoint_id']].index[0]
        ax1.annotate(f"+{row['improvement']:.1f}%",
                     xy=(idx, row['accuracy_pct_fine']),
                     xytext=(0, 8), textcoords='offset points',
                     ha='center', fontsize=8, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    # 2. Improvement bar chart (top right)
    ax2 = plt.subplot(2, 2, 2)
    colors = ['#51cf66' if x > 0 else '#ff6b6b' for x in df_merged['improvement']]
    bars = ax2.barh(df_merged['keypoint_id'], df_merged['improvement'],
                    color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)

    ax2.set_xlabel('Accuracy Improvement (%)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Keypoint ID', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Fine-tuning Impact per Keypoint', fontsize=12, fontweight='bold', pad=10)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    ax2.grid(True, alpha=0.3, axis='x', linestyle='--')

    # Annotate extremes
    max_imp = df_merged.loc[df_merged['improvement'].idxmax()]
    min_imp = df_merged.loc[df_merged['improvement'].idxmin()]
    ax2.annotate(f"Best: +{max_imp['improvement']:.1f}%",
                 xy=(max_imp['improvement'], max_imp['keypoint_id']),
                 xytext=(10, 0), textcoords='offset points',
                 fontsize=9, fontweight='bold', color='darkgreen',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))

    # 3. Scatter plot correlation (bottom left)
    ax3 = plt.subplot(2, 2, 3)
    scatter = ax3.scatter(df_merged['accuracy_pct_zero'], df_merged['accuracy_pct_fine'],
                          s=120, alpha=0.7, c=df_merged['improvement'],
                          cmap='RdYlGn', edgecolors='black', linewidth=0.5,
                          vmin=-5, vmax=df_merged['improvement'].max())

    ax3.plot([0, 100], [0, 100], 'k--', alpha=0.4, linewidth=1.5, label='No change')
    ax3.set_xlabel('Zero-shot Accuracy (%)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Fine-tuned Accuracy (%)', fontsize=11, fontweight='bold')
    ax3.set_title('(c) Accuracy Correlation', fontsize=12, fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(fontsize=9)
    ax3.set_xlim(-5, 105)
    ax3.set_ylim(-5, 105)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Improvement (%)', fontsize=10, fontweight='bold')

    # Annotate outliers (top 3 improvements)
    for _, row in df_merged.nlargest(3, 'improvement').iterrows():
        ax3.annotate(f"KP {row['keypoint_id']}",
                     (row['accuracy_pct_zero'], row['accuracy_pct_fine']),
                     xytext=(8, 8), textcoords='offset points',
                     fontsize=8, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                               edgecolor='black', alpha=0.8),
                     arrowprops=dict(arrowstyle='->', lw=1))

    # 4. Difficulty stratification (bottom right)
    ax4 = plt.subplot(2, 2, 4)

    # Categorize keypoints by difficulty
    df_merged['difficulty'] = pd.cut(df_merged['accuracy_pct_zero'],
                                     bins=[0, 33, 66, 100],
                                     labels=['Hard', 'Medium', 'Easy'])

    difficulty_stats = df_merged.groupby('difficulty').agg({
        'improvement': ['mean', 'std', 'count']
    })

    categories = ['Hard', 'Medium', 'Easy']
    means = [difficulty_stats.loc[cat, ('improvement', 'mean')] if cat in difficulty_stats.index else 0
             for cat in categories]
    stds = [difficulty_stats.loc[cat, ('improvement', 'std')] if cat in difficulty_stats.index else 0
            for cat in categories]
    counts = [int(difficulty_stats.loc[cat, ('improvement', 'count')]) if cat in difficulty_stats.index else 0
              for cat in categories]

    bars = ax4.bar(categories, means, yerr=stds, capsize=10,
                   color=['#ff6b6b', '#ffd43b', '#51cf66'],
                   alpha=0.85, edgecolor='black', linewidth=1)

    ax4.set_ylabel('Mean Improvement (%)', fontsize=11, fontweight='bold')
    ax4.set_title('(d) Improvement by Initial Difficulty', fontsize=12, fontweight='bold', pad=10)
    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Add count annotations
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'n={count}', ha='center', va='bottom',
                 fontsize=9, fontweight='bold')

    plt.suptitle('Comprehensive Per-Keypoint Analysis: Zero-Shot vs Fine-Tuned\n(Person Category, PCK@0.10)',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = f'{save_dir}/keypoint_comparison_comprehensive.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comprehensive keypoint comparison to: {save_path}")
    plt.close()

    return df_merged


def create_category_difficulty_table(results_dir, save_path):
    """
    Crea una tabella LaTeX con le performance per categoria
    """
    with open(f'{results_dir}/per_category_stats.json') as f:
        category_stats = json.load(f)

    # Create DataFrame
    rows = []
    for category, stats in category_stats.items():
        row = {
            'category': category,
            'pck_005': stats['pck@0.05']['mean'],
            'pck_010': stats['pck@0.10']['mean'],
            'pck_020': stats['pck@0.20']['mean'],
            'std_010': stats['pck@0.10']['std']
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values('pck_010')

    # Generate LaTeX table
    latex = "\\begin{table}[h]\n\\centering\n"
    latex += "\\caption{Performance comparison across object categories (DINOv3 zero-shot)}\n"
    latex += "\\label{tab:category_difficulty}\n"
    latex += "\\begin{tabular}{lcccc}\n"
    latex += "\\toprule\n"
    latex += "\\textbf{Category} & \\textbf{PCK@0.05} & \\textbf{PCK@0.10} & \\textbf{PCK@0.20} & \\textbf{Std Dev} \\\\\n"
    latex += "\\midrule\n"

    for _, row in df.iterrows():
        latex += f"{row['category']} & {row['pck_005']:.2f}\\% & {row['pck_010']:.2f}\\% & "
        latex += f"{row['pck_020']:.2f}\\% & {row['std_010']:.2f}\\% \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    with open(save_path, 'w') as f:
        f.write(latex)

    print(f"✓ Saved category difficulty table to: {save_path}")
    return df


def find_example_cases(df_keypoints, category='person', threshold=0.10, dataset_path=None):
    """
    Trova esempi rappresentativi di keypoint difficili e facili

    Returns:
        difficult_examples: list of dicts
        easy_examples: list of dicts
    """
    df_cat = df_keypoints[
        (df_keypoints['category'] == category) &
        (df_keypoints['threshold'] == threshold)
        ]

    # Identifica keypoint difficili e facili
    kp_difficulty = df_cat.groupby('keypoint_id')['correct_at_threshold'].mean()
    difficult_kps = kp_difficulty.nsmallest(5).index.tolist()
    easy_kps = kp_difficulty.nlargest(5).index.tolist()

    print(f"\nMost difficult keypoints: {difficult_kps}")
    print(f"Easiest keypoints: {easy_kps}")

    difficult_examples = []
    easy_examples = []

    # Trova worst cases per keypoint difficili
    for kp_id in difficult_kps[:3]:
        df_kp = df_cat[df_cat['keypoint_id'] == kp_id]
        # Seleziona casi con errore moderato-alto (non solo i peggiori assoluti)
        candidates = df_kp[df_kp['distance'] > 0.15].nsmallest(5, 'distance')

        if len(candidates) > 0:
            example = candidates.iloc[0]
            difficult_examples.append({
                'image_idx': example['image_idx'],
                'keypoint_id': example['keypoint_id'],
                'distance': example['distance'],
                'kp_name': f"KP {kp_id}"
            })

    # Trova best cases per keypoint facili
    for kp_id in easy_kps[:3]:
        df_kp = df_cat[df_cat['keypoint_id'] == kp_id]
        # Seleziona casi corretti con basso errore
        candidates = df_kp[df_kp['correct_at_threshold'] == 1].nsmallest(5, 'distance')

        if len(candidates) > 0:
            example = candidates.iloc[0]
            easy_examples.append({
                'image_idx': example['image_idx'],
                'keypoint_id': example['keypoint_id'],
                'distance': example['distance'],
                'kp_name': f"KP {kp_id}"
            })

    print(f"\nFound {len(difficult_examples)} difficult examples")
    print(f"Found {len(easy_examples)} easy examples")

    return difficult_examples, easy_examples


def main():
    parser = argparse.ArgumentParser(description="Generate keypoint analysis figures")
    parser.add_argument('--results_zeroshot', type=str, required=True,
                        help='Path to zero-shot results directory')
    parser.add_argument('--results_finetuned', type=str, required=True,
                        help='Path to fine-tuned results directory')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Path to SPair-71k dataset (for loading images)')
    parser.add_argument('--output_dir', type=str, default='figures',
                        help='Directory to save generated figures')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 60)
    print("GENERATING KEYPOINT ANALYSIS FIGURES")
    print("=" * 60)

    # 1. Create category difficulty table (LaTeX)
    print("\n[1/4] Generating category difficulty table...")
    create_category_difficulty_table(
        args.results_zeroshot,
        output_dir / 'category_difficulty_table.tex'
    )

    # 2. Create comprehensive keypoint comparison
    print("\n[2/4] Generating comprehensive keypoint comparison...")
    df_merged = plot_per_keypoint_comparison(
        args.results_zeroshot,
        args.results_finetuned,
        output_dir
    )

    # 3. Find and visualize example cases (if dataset path provided)
    if args.dataset_path:
        print("\n[3/4] Finding representative examples...")
        df_keypoints_zero = pd.read_csv(f'{args.results_zeroshot}/all_keypoint_metrics.csv')

        difficult_examples, easy_examples = find_example_cases(
            df_keypoints_zero,
            category='person',
            threshold=0.10,
            dataset_path=args.dataset_path
        )

        print("\nNOTE: To complete the visualization, you need to:")
        print("1. Load the actual images using the image_idx from the examples")
        print("2. Extract source_kp, target_kp, and pred_kp coordinates")
        print("3. Call create_difficulty_comparison_grid() with complete data")
        print("\nExample data structure needed:")
        print("  - source_img: PIL Image")
        print("  - target_img: PIL Image")
        print("  - source_kp: (x, y)")
        print("  - target_kp: (x, y)")
        print("  - pred_kp: (x, y)")
        print("  - kp_name: string")
        print("  - distance: float")
    else:
        print("\n[3/4] Skipping example visualization (no dataset path provided)")

    # 4. Summary statistics
    print("\n[4/4] Generating summary statistics...")

    # Most improved keypoints
    top_improvements = df_merged.nlargest(5, 'improvement')[
        ['keypoint_id', 'accuracy_pct_zero', 'accuracy_pct_fine', 'improvement']]
    print("\nTop 5 Most Improved Keypoints:")
    print(top_improvements.to_string(index=False))

    # Save to file
    with open(output_dir / 'improvement_summary.txt', 'w') as f:
        f.write("KEYPOINT IMPROVEMENT SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write("Top 5 Most Improved Keypoints:\n")
        f.write(top_improvements.to_string(index=False))
        f.write("\n\n")

        f.write("Overall Statistics:\n")
        f.write(f"Mean improvement: {df_merged['improvement'].mean():.2f}%\n")
        f.write(f"Std improvement: {df_merged['improvement'].std():.2f}%\n")
        f.write(f"Min improvement: {df_merged['improvement'].min():.2f}%\n")
        f.write(f"Max improvement: {df_merged['improvement'].max():.2f}%\n")

    print("\n" + "=" * 60)
    print("✓ ALL FIGURES GENERATED SUCCESSFULLY!")
    print(f"✓ Output directory: {output_dir}")
    print("=" * 60)

    print("\nGenerated files:")
    for file in sorted(output_dir.glob('*')):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()