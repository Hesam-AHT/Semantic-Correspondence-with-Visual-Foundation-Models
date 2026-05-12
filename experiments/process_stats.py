import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from collections import defaultdict


def load_and_analyze(results_dir):
    """
    Load all_keypoint_metrics.csv and compute overall and per-category statistics.
    """

    print(f"Loading data from {results_dir}...")
    df_keypoints = pd.read_csv(f'{results_dir}/all_keypoint_metrics.csv')

    print(f"Loaded {len(df_keypoints)} keypoint measurements")
    print(f"Unique images: {df_keypoints['image_idx'].nunique()}")
    print(f"Categories: {df_keypoints['category'].unique()}")
    print(f"Thresholds: {df_keypoints['threshold'].unique()}")

    thresholds = sorted(df_keypoints['threshold'].unique())

    # crea cartella per keypoint stats
    keypoint_stats_dir = os.path.join(results_dir, 'keypoint_stats')
    os.makedirs(keypoint_stats_dir, exist_ok=True)

    # ===================== OVERALL RESULTS =====================
    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)

    overall_stats = {}

    for threshold in thresholds:
        df_thresh = df_keypoints[df_keypoints['threshold'] == threshold]

        # compute per-image PCK (average over all keypoints per image)
        per_image_pck = df_thresh.groupby('image_idx')['correct_at_threshold'].mean() * 100

        mean_pck = float(per_image_pck.mean())
        std_pck = float(per_image_pck.std())
        median_pck = float(per_image_pck.median())
        p25 = float(per_image_pck.quantile(0.25))
        p75 = float(per_image_pck.quantile(0.75))

        # compute per-keypoint PCK (average per keypoint globally)
        per_keypoint_pck = df_thresh['correct_at_threshold'].mean() * 100

        overall_stats[f"pck@{threshold:.2f}"] = {
            "mean": mean_pck,
            "std": std_pck,
            "median": median_pck,
            "p25": p25,
            "p75": p75,
            "per_keypoint_mean": per_keypoint_pck,
        }

        print(f"PCK@{threshold:.2f}: "
              f"mean={mean_pck:.2f}%, std={std_pck:.2f}%, "
              f"median={median_pck:.2f}%, "
              f"p25={p25:.2f}%, p75={p75:.2f}%"
              f"per-keypoint mean={per_keypoint_pck:.2f}%")

    with open(f'{results_dir}/overall_stats.json', 'w') as f:
        json.dump(overall_stats, f, indent=2)

    # ===================== PER-CATEGORY RESULTS =====================
    print("\n" + "=" * 60)
    print("PER-CATEGORY RESULTS")
    print("=" * 60)

    per_category_stats = {}

    for category in sorted(df_keypoints['category'].unique()):
        print(f"\nCategory: {category}")
        per_category_stats[category] = {}

        df_cat = df_keypoints[df_keypoints['category'] == category]

        for threshold in thresholds:
            df_thresh = df_cat[df_cat['threshold'] == threshold]

            per_image_pck = df_thresh.groupby('image_idx')['correct_at_threshold'].mean() * 100

            mean_pck = float(per_image_pck.mean())
            std_pck = float(per_image_pck.std())
            median_pck = float(per_image_pck.median())

            per_category_stats[category][f"pck@{threshold:.2f}"] = {
                "mean": mean_pck,
                "std": std_pck,
                "median": median_pck,
            }

            print(f"  PCK@{threshold:.2f}: "
                  f"mean={mean_pck:.2f}%, std={std_pck:.2f}%, median={median_pck:.2f}%")

    with open(f'{results_dir}/per_category_stats.json', 'w') as f:
        json.dump(per_category_stats, f, indent=2)

    # category CSV
    category_rows = []
    for category in sorted(df_keypoints['category'].unique()):
        df_cat = df_keypoints[df_keypoints['category'] == category]
        for threshold in thresholds:
            df_thresh = df_cat[df_cat['threshold'] == threshold]
            per_image_pck = df_thresh.groupby('image_idx')['correct_at_threshold'].mean() * 100

            category_rows.append({
                'category': category,
                'threshold': threshold,
                'mean_pck': per_image_pck.mean(),
                'std_pck': per_image_pck.std(),
                'median_pck': per_image_pck.median(),
                'count': len(per_image_pck)
            })

    df_category = pd.DataFrame(category_rows)
    df_category.to_csv(f'{results_dir}/per_category_metrics.csv', index=False)

    # ===================== PER-CATEGORY KEYPOINT ANALYSIS =====================
    print("\n" + "=" * 60)
    print("PER-CATEGORY KEYPOINT ANALYSIS")
    print("=" * 60)

    # analizza per ogni categoria separatamente
    for category in sorted(df_keypoints['category'].unique()):
        print(f"\n=== Category: {category} ===")

        df_cat = df_keypoints[df_keypoints['category'] == category]

        # statistiche per keypoint in questa categoria
        for threshold in thresholds:
            df_thresh = df_cat[df_cat['threshold'] == threshold]

            keypoint_stats = df_thresh.groupby('keypoint_id').agg({
                'correct_at_threshold': ['mean', 'count'],
                'distance': ['mean', 'std']
            }).round(4)

            keypoint_stats.columns = ['accuracy', 'count', 'avg_distance', 'std_distance']
            keypoint_stats['accuracy_pct'] = keypoint_stats['accuracy'] * 100
            keypoint_stats = keypoint_stats.sort_values('accuracy_pct')

            print(f"\nPCK@{threshold:.2f}:")
            print(keypoint_stats[['accuracy_pct', 'avg_distance', 'count']].to_string())

            # salva CSV nella cartella keypoint_stats
            keypoint_stats.to_csv(
                f'{keypoint_stats_dir}/{category}_pck{threshold:.2f}.csv'
            )

    # ===================== HEATMAP ACCURACY PER THRESHOLD =====================
    print("\n--- Generating Accuracy Heatmaps (one per threshold) ---")

    for threshold in thresholds:
        df_thresh = df_keypoints[df_keypoints['threshold'] == threshold]

        pivot_accuracy = df_thresh.pivot_table(
            values='correct_at_threshold',
            index='keypoint_id',
            columns='category',
            aggfunc='mean'
        ) * 100

        fig, ax = plt.subplots(figsize=(12, 8))

        im = ax.imshow(pivot_accuracy.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        ax.set_title(f'Keypoint Accuracy per Category (PCK@{threshold:.2f})',
                     fontsize=16, fontweight='bold')
        ax.set_xlabel('Category', fontsize=13)
        ax.set_ylabel('Keypoint ID', fontsize=13)
        ax.set_xticks(range(len(pivot_accuracy.columns)))
        ax.set_xticklabels(pivot_accuracy.columns, rotation=45, ha='right', fontsize=11)
        ax.set_yticks(range(len(pivot_accuracy.index)))
        ax.set_yticklabels(pivot_accuracy.index, fontsize=10)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Accuracy (%)', fontsize=13)

        # aggiungi valori nelle celle
        for i in range(len(pivot_accuracy.index)):
            for j in range(len(pivot_accuracy.columns)):
                if not np.isnan(pivot_accuracy.values[i, j]):
                    ax.text(j, i, f'{pivot_accuracy.values[i, j]:.1f}',
                            ha="center", va="center", color="black", fontsize=8)

        plt.tight_layout()
        plt.savefig(f'{results_dir}/keypoint_accuracy_heatmap_pck{threshold:.2f}.png',
                    dpi=300, bbox_inches='tight')
        print(f"Accuracy heatmap for PCK@{threshold:.2f} saved")
        plt.close()

    # ===================== HEATMAP SAMPLE COUNT =====================
    print("\n--- Generating Sample Count Heatmap ---")

    # usa threshold qualsiasi (il count è lo stesso per tutti)
    df_any = df_keypoints[df_keypoints['threshold'] == thresholds[0]]

    pivot_count = df_any.pivot_table(
        values='correct_at_threshold',
        index='keypoint_id',
        columns='category',
        aggfunc='count'
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(pivot_count.values, cmap='Blues', aspect='auto')
    ax.set_title('Number of Samples per Keypoint', fontsize=16, fontweight='bold')
    ax.set_xlabel('Category', fontsize=13)
    ax.set_ylabel('Keypoint ID', fontsize=13)
    ax.set_xticks(range(len(pivot_count.columns)))
    ax.set_xticklabels(pivot_count.columns, rotation=45, ha='right', fontsize=11)
    ax.set_yticks(range(len(pivot_count.index)))
    ax.set_yticklabels(pivot_count.index, fontsize=10)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Sample Count', fontsize=13)

    # aggiungi valori nelle celle
    for i in range(len(pivot_count.index)):
        for j in range(len(pivot_count.columns)):
            if not np.isnan(pivot_count.values[i, j]):
                ax.text(j, i, f'{int(pivot_count.values[i, j])}',
                        ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{results_dir}/keypoint_sample_count_heatmap.png',
                dpi=300, bbox_inches='tight')
    print(f"Sample count heatmap saved")
    plt.close()

    # ===================== ERROR DISTRIBUTION PER CATEGORY =====================
    print("\n--- Generating Per-Category Error Distributions ---")

    categories = sorted(df_keypoints['category'].unique())
    n_cats = len(categories)

    # grid layout per le categorie
    n_cols = 3
    n_rows = (n_cats + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten() if n_cats > 1 else [axes]

    threshold_to_analyze = 0.10

    for idx, category in enumerate(categories):
        df_cat = df_keypoints[
            (df_keypoints['category'] == category) &
            (df_keypoints['threshold'] == threshold_to_analyze)
            ]

        keypoint_ids = sorted(df_cat['keypoint_id'].unique())
        distances_by_kp = [df_cat[df_cat['keypoint_id'] == kp]['distance'].values
                           for kp in keypoint_ids]

        bp = axes[idx].boxplot(distances_by_kp, labels=keypoint_ids, patch_artist=True)

        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')

        axes[idx].axhline(y=threshold_to_analyze, color='r', linestyle='--',
                          label=f'Threshold ({threshold_to_analyze})', linewidth=2)
        axes[idx].set_xlabel('Keypoint ID', fontsize=10)
        axes[idx].set_ylabel('Normalized Distance', fontsize=10)
        axes[idx].set_title(f'{category}', fontsize=12, fontweight='bold')
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3)

    # nascondi assi vuoti
    for idx in range(n_cats, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(f'{results_dir}/error_distribution_per_category.png',
                dpi=300, bbox_inches='tight')
    print(f"Per-category error distribution saved")
    plt.close()

    # ===================== DIFFICULTY ANALYSIS =====================
    print("\n" + "=" * 60)
    print("DIFFICULTY ANALYSIS")
    print("=" * 60)

    threshold_main = 0.10
    df_main = df_keypoints[df_keypoints['threshold'] == threshold_main]

    # 1. varianza intra-categoria (quanto sono consistenti i keypoint?)
    print("\n--- Keypoint Consistency within Categories ---")
    consistency_stats = []

    for category in sorted(df_main['category'].unique()):
        df_cat = df_main[df_main['category'] == category]

        kp_accuracies = df_cat.groupby('keypoint_id')['correct_at_threshold'].mean()

        consistency_stats.append({
            'category': category,
            'mean_accuracy': kp_accuracies.mean() * 100,
            'std_accuracy': kp_accuracies.std() * 100,
            'min_accuracy': kp_accuracies.min() * 100,
            'max_accuracy': kp_accuracies.max() * 100,
            'range': (kp_accuracies.max() - kp_accuracies.min()) * 100,
            'num_keypoints': len(kp_accuracies)
        })

    df_consistency = pd.DataFrame(consistency_stats).sort_values('std_accuracy', ascending=False)
    print("\nCategories by keypoint difficulty variance (high = inconsistent keypoints):")
    print(df_consistency.to_string(index=False))
    df_consistency.to_csv(f'{results_dir}/category_keypoint_consistency.csv', index=False)

    # 2. threshold sensitivity (quanto cambia la performance tra threshold?)
    print("\n--- Threshold Sensitivity Analysis ---")
    threshold_sensitivity = []

    for category in sorted(df_keypoints['category'].unique()):
        df_cat = df_keypoints[df_keypoints['category'] == category]

        pck_values = {}
        for threshold in thresholds:
            df_thresh = df_cat[df_cat['threshold'] == threshold]
            per_image_pck = df_thresh.groupby('image_idx')['correct_at_threshold'].mean()
            pck_values[threshold] = per_image_pck.mean() * 100

        # calcola sensitivity come differenza tra threshold massimo e minimo
        sensitivity = pck_values[max(thresholds)] - pck_values[min(thresholds)]

        threshold_sensitivity.append({
            'category': category,
            'pck@0.05': pck_values.get(0.05, np.nan),
            'pck@0.10': pck_values.get(0.10, np.nan),
            'pck@0.20': pck_values.get(0.20, np.nan),
            'sensitivity': sensitivity
        })

    df_sensitivity = pd.DataFrame(threshold_sensitivity).sort_values('sensitivity')
    print("\nCategories by threshold sensitivity (low = more precise predictions):")
    print(df_sensitivity.to_string(index=False))
    df_sensitivity.to_csv(f'{results_dir}/threshold_sensitivity.csv', index=False)

    # 3. most/least difficult keypoints per category
    print("\n--- Most Difficult Keypoints per Category ---")
    difficult_keypoints = []

    for category in sorted(df_main['category'].unique()):
        df_cat = df_main[df_main['category'] == category]

        kp_stats = df_cat.groupby('keypoint_id').agg({
            'correct_at_threshold': 'mean',
            'distance': 'mean'
        })
        kp_stats['accuracy_pct'] = kp_stats['correct_at_threshold'] * 100

        if len(kp_stats) > 0:
            hardest = kp_stats['accuracy_pct'].idxmin()
            easiest = kp_stats['accuracy_pct'].idxmax()

            difficult_keypoints.append({
                'category': category,
                'hardest_keypoint': hardest,
                'hardest_accuracy': kp_stats.loc[hardest, 'accuracy_pct'],
                'easiest_keypoint': easiest,
                'easiest_accuracy': kp_stats.loc[easiest, 'accuracy_pct'],
                'accuracy_gap': kp_stats.loc[easiest, 'accuracy_pct'] - kp_stats.loc[hardest, 'accuracy_pct']
            })

    df_difficult = pd.DataFrame(difficult_keypoints).sort_values('accuracy_gap', ascending=False)
    print("\nHardest and easiest keypoints per category:")
    print(df_difficult.to_string(index=False))
    df_difficult.to_csv(f'{results_dir}/keypoint_difficulty_extremes.csv', index=False)

    # ===================== PLOT HARDEST/EASIEST KEYPOINTS =====================
    print("\n--- Generating Hardest/Easiest Keypoints Visualization ---")


    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    categories_sorted = df_difficult['category'].tolist()
    hardest_acc = df_difficult['hardest_accuracy'].tolist()
    easiest_acc = df_difficult['easiest_accuracy'].tolist()
    hardest_kp = df_difficult['hardest_keypoint'].tolist()
    easiest_kp = df_difficult['easiest_keypoint'].tolist()

    x = np.arange(len(categories_sorted))
    width = 0.35

    # Subplot 1: Accuracy comparison
    bars1 = axes[0].bar(x - width / 2, hardest_acc, width, label='Hardest KP',
                        color='#ff6b6b', alpha=0.8, edgecolor='black')
    bars2 = axes[0].bar(x + width / 2, easiest_acc, width, label='Easiest KP',
                        color='#51cf66', alpha=0.8, edgecolor='black')

    axes[0].set_xlabel('Category', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Hardest vs Easiest Keypoint Accuracy per Category',
                      fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories_sorted, rotation=45, ha='right')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim(0, 100)


    for i, (bar1, bar2, h_kp, e_kp) in enumerate(zip(bars1, bars2, hardest_kp, easiest_kp)):
        #hardest keypoint
        height1 = bar1.get_height()
        axes[0].text(bar1.get_x() + bar1.get_width() / 2., height1 + 2,
                     f'KP {h_kp}', ha='center', va='bottom', fontsize=8,
                     fontweight='bold', color='#c92a2a')

        #easiest keypoint
        height2 = bar2.get_height()
        axes[0].text(bar2.get_x() + bar2.get_width() / 2., height2 + 2,
                     f'KP {e_kp}', ha='center', va='bottom', fontsize=8,
                     fontweight='bold', color='#2f9e44')

    # Subplot 2: Accuracy gap
    gaps = df_difficult['accuracy_gap'].tolist()
    bars3 = axes[1].bar(x, gaps, color='#4dabf7', alpha=0.8, edgecolor='black')

    axes[1].set_xlabel('Category', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy Gap (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Keypoint Difficulty Range per Category\n(Easiest - Hardest)',
                      fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories_sorted, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')

    for bar, gap in zip(bars3, gaps):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2., height + 1,
                     f'{gap:.1f}%', ha='center', va='bottom', fontsize=9,
                     fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{results_dir}/keypoint_difficulty_comparison.png',
                dpi=300, bbox_inches='tight')
    print("Hardest/Easiest keypoints comparison saved")
    plt.close()

    fig, ax = plt.subplots(figsize=(14, 8))

    data_matrix = []
    row_labels = []

    for _, row in df_difficult.iterrows():
        category = row['category']
        df_cat = df_main[df_main['category'] == category]

        kp_stats = df_cat.groupby('keypoint_id')['correct_at_threshold'].mean() * 100
        kp_stats_sorted = kp_stats.sort_values()

        data_matrix.append(kp_stats_sorted.values)
        row_labels.append(f"{category}\n({len(kp_stats_sorted)} KPs)")

    max_len = max(len(row) for row in data_matrix)
    data_matrix_padded = np.array([
        np.pad(row, (0, max_len - len(row)), constant_values=np.nan)
        for row in data_matrix
    ])

    im = ax.imshow(data_matrix_padded, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xlabel('Keypoint Rank (sorted by difficulty)', fontsize=12, fontweight='bold')
    ax.set_title('Keypoint Accuracy Distribution per Category\n(Left = Hardest, Right = Easiest)',
                 fontsize=14, fontweight='bold', pad=20)

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Accuracy (%)', fontsize=12, fontweight='bold')

    for i, (row_data, label) in enumerate(zip(data_matrix, row_labels)):

        ax.text(-0.5, i, '◀', fontsize=16, color='#c92a2a',
                ha='right', va='center', fontweight='bold')

        last_idx = len(row_data) - 1
        ax.text(last_idx + 0.5, i, '▶', fontsize=16, color='#2f9e44',
                ha='left', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{results_dir}/keypoint_accuracy_distribution_heatmap.png',
                dpi=300, bbox_inches='tight')
    print("Keypoint accuracy distribution heatmap saved")
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 10))

    df_plot = df_difficult.sort_values('accuracy_gap', ascending=True)

    y_pos = np.arange(len(df_plot))

    for i, (_, row) in enumerate(df_plot.iterrows()):

        ax.plot([row['hardest_accuracy'], row['easiest_accuracy']], [i, i],
                'o-', linewidth=2.5, markersize=8, color='#4dabf7', alpha=0.7)

        ax.plot(row['hardest_accuracy'], i, 'o', markersize=10,
                color='#ff6b6b', zorder=3, markeredgecolor='black', markeredgewidth=1)

        ax.plot(row['easiest_accuracy'], i, 'o', markersize=10,
                color='#51cf66', zorder=3, markeredgecolor='black', markeredgewidth=1)

        ax.text(row['hardest_accuracy'] - 3, i, f"KP {row['hardest_keypoint']}",
                ha='right', va='center', fontsize=8, fontweight='bold', color='#c92a2a')
        ax.text(row['easiest_accuracy'] + 3, i, f"KP {row['easiest_keypoint']}",
                ha='left', va='center', fontsize=8, fontweight='bold', color='#2f9e44')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot['category'].tolist(), fontsize=11)
    ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Keypoint Difficulty Range per Category\n(Red = Hardest, Green = Easiest)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, 100)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff6b6b',
               markersize=10, label='Hardest Keypoint', markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#51cf66',
               markersize=10, label='Easiest Keypoint', markeredgecolor='black')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

    plt.tight_layout()
    plt.savefig(f'{results_dir}/keypoint_difficulty_range_lollipop.png',
                dpi=300, bbox_inches='tight')
    print("Keypoint difficulty range lollipop chart saved")
    plt.close()

    # ===================== SUMMARY REPORT =====================
    with open(f'{results_dir}/summary_report.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"{results_dir} REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("OVERALL RESULTS\n")
        f.write("-" * 60 + "\n")
        for threshold in thresholds:
            stats = overall_stats[f"pck@{threshold:.2f}"]
            f.write(f"PCK@{threshold:.2f}: mean={stats['mean']:.2f}%, "
                    f"std={stats['std']:.2f}%, median={stats['median']:.2f}%\n")

        f.write("\n\nPER-CATEGORY RESULTS\n")
        f.write("-" * 60 + "\n")
        for category in sorted(per_category_stats.keys()):
            f.write(f"\n{category}:\n")
            for threshold in thresholds:
                stats = per_category_stats[category][f"pck@{threshold:.2f}"]
                f.write(f"  PCK@{threshold:.2f}: mean={stats['mean']:.2f}%, "
                        f"std={stats['std']:.2f}%\n")

        f.write("\n\nKEYPOINT CONSISTENCY ANALYSIS\n")
        f.write("-" * 60 + "\n")
        f.write("Categories with most inconsistent keypoint difficulty:\n")
        f.write(df_consistency.head(5).to_string(index=False))

        f.write("\n\n\nTHRESHOLD SENSITIVITY\n")
        f.write("-" * 60 + "\n")
        f.write("Categories most sensitive to threshold changes:\n")
        f.write(df_sensitivity.tail(5).to_string(index=False))

    print(f"\n✓ Analysis complete! All results_SPair71K saved to: {results_dir}")
    print(f"✓ Keypoint stats saved in: {keypoint_stats_dir}")
    return df_keypoints, overall_stats, per_category_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process keypoint metrics from all_keypoint_metrics.csv")
    parser.add_argument(
        "results_dir",
        nargs="?",
        default=".",
        help="Directory containing all_keypoint_metrics.csv"
    )
    args = parser.parse_args()
    results_dir = args.results_dir

    if not os.path.isdir(results_dir):
        print(f"Error: Directory not found: {results_dir}")
        exit(1)

    csv_path = os.path.join(results_dir, "all_keypoint_metrics.csv")
    if not os.path.isfile(csv_path):
        print(f"Error: all_keypoint_metrics.csv file not found in: {results_dir}")
        exit(1)

    # controlla che la cartella contenga solo il CSV (e opzionalmente una sottocartella keypoint_stats)
    files_in_dir = [f for f in os.listdir(results_dir)
                    if os.path.isfile(os.path.join(results_dir, f))]

    if (len(files_in_dir) != 2
            or ((files_in_dir[0] != "all_keypoint_metrics.csv" and files_in_dir[1]!='overall_stats.json')
            and (files_in_dir[1] != "all_keypoint_metrics.csv" and files_in_dir[0]!='overall_stats.json'))):
        print(f"Warning: Directory should contain only all_keypoint_metrics.csv and overall_stats.json")
        print(f"Found files: {files_in_dir}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            exit(1)

    load_and_analyze(results_dir)