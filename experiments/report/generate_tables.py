import json
import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import configs.paths as paths


def load_category_stats(results_dir):
    """
    Load per-category statistics from results directory

    Returns:
        dict: category -> {pck@threshold: {mean, std, median}}
    """
    stats_file = Path(results_dir) / 'per_category_stats.json'

    if not stats_file.exists():
        raise FileNotFoundError(f"File not found: {stats_file}")

    with open(stats_file) as f:
        return json.load(f)


def load_overall_stats(results_dir):
    """
    Load overall statistics from results directory

    Returns:
        dict: pck@threshold -> {mean, std, median}
    """
    stats_file = Path(results_dir) / 'overall_stats.json'

    if not stats_file.exists():
        raise FileNotFoundError(f"File not found: {stats_file}")

    with open(stats_file) as f:
        return json.load(f)


def create_comparison_dataframe(results_dirs, model_names):
    """
    Create a comprehensive comparison DataFrame

    Args:
        results_dirs: list of paths to results directories
        model_names: list of model names (e.g., ['Zero-shot', 'Fine-tuned', 'Fine-tuned + Soft-argmax'])

    Returns:
        pd.DataFrame with multi-index (model, threshold) and categories as columns
    """
    thresholds = [0.05, 0.10, 0.20]

    # Load all data
    all_category_stats = {}
    all_overall_stats = {}
    all_categories = set()

    for model_name, results_dir in zip(model_names, results_dirs):
        category_stats = load_category_stats(results_dir)
        overall_stats = load_overall_stats(results_dir)

        all_category_stats[model_name] = category_stats
        all_overall_stats[model_name] = overall_stats
        all_categories.update(category_stats.keys())

    # Sort categories alphabetically
    categories = sorted(all_categories)

    # Build DataFrame
    rows = []
    index = []

    for model_name in model_names:
        for threshold in thresholds:
            row_data = {}

            # Add per-category data
            for category in categories:
                if category in all_category_stats[model_name]:
                    pck_key = f"pck@{threshold:.2f}"
                    if pck_key in all_category_stats[model_name][category]:
                        value = all_category_stats[model_name][category][pck_key]['mean']
                        row_data[category] = value
                    else:
                        row_data[category] = np.nan
                else:
                    row_data[category] = np.nan

            # Add overall mean
            pck_key = f"pck@{threshold:.2f}"
            if pck_key in all_overall_stats[model_name]:
                row_data['Mean'] = all_overall_stats[model_name][pck_key]['mean']
            else:
                row_data['Mean'] = np.nan

            rows.append(row_data)
            index.append((model_name, f"PCK@{threshold:.2f}"))

    df = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(index, names=['Model', 'Metric']))
    df.columns = categories + ['Mean']

    return df


def generate_latex_table(df, output_path, highlight_best=True):
    """
    Generate a LaTeX table from the comparison DataFrame

    Args:
        df: DataFrame with comparison data
        output_path: where to save the .tex file
        highlight_best: whether to bold the best values per category
    """

    # Find best value per category (across all model-threshold combinations)
    best_values = {}
    if highlight_best:
        for col in df.columns:
            best_values[col] = df[col].max()

    # Start LaTeX table
    n_cols = len(df.columns)
    col_spec = 'l' + 'l' + 'c' * n_cols  # Model + Metric + categories

    latex = []
    latex.append("\\begin{table*}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Performance comparison across models and object categories}")
    latex.append("\\label{tab:model_category_comparison}")
    latex.append("\\resizebox{\\textwidth}{!}{%")
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append("\\toprule")

    # Header row
    header = "\\textbf{Model} & \\textbf{Metric}"
    for col in df.columns:
        # Capitalize category names and handle special characters
        col_display = col.replace('_', '\\_').title()
        header += f" & \\textbf{{{col_display}}}"
    header += " \\\\"
    latex.append(header)
    latex.append("\\midrule")

    # Data rows
    current_model = None
    for (model, metric), row in df.iterrows():
        # Add model name only on first threshold row
        if model != current_model:
            model_display = model.replace('_', '\\_')
            model_cell = f"\\multirow{{3}}{{*}}{{\\textbf{{{model_display}}}}}"
            current_model = model
        else:
            model_cell = ""

        # Metric cell
        metric_cell = metric

        # Build row
        row_str = f"{model_cell} & {metric_cell}"

        for col in df.columns:
            value = row[col]
            if pd.isna(value):
                cell = "--"
            else:
                # Format value
                cell = f"{value:.2f}\\%"

                # Bold if best value
                if highlight_best and col in best_values:
                    if abs(value - best_values[col]) < 0.01:  # floating point tolerance
                        cell = f"\\textbf{{{cell}}}"

            row_str += f" & {cell}"

        row_str += " \\\\"
        latex.append(row_str)

        # Add midrule after each model
        if metric == "PCK@0.20":
            latex.append("\\midrule")

    # Remove last midrule and add bottomrule
    if latex[-1] == "\\midrule":
        latex[-1] = "\\bottomrule"
    else:
        latex.append("\\bottomrule")

    latex.append("\\end{tabular}%")
    latex.append("}")
    latex.append("\\end{table*}")

    # Write to file
    latex_str = "\n".join(latex)

    with open(output_path, 'w') as f:
        f.write(latex_str)

    return latex_str


def generate_summary_table(df, output_path):
    """
    Generate a simplified summary table with only overall means
    """
    # Extract only the 'Mean' column
    df_summary = df[['Mean']].copy()

    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Overall performance comparison across models}")
    latex.append("\\label{tab:model_overall_comparison}")
    latex.append("\\begin{tabular}{llc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Model} & \\textbf{Metric} & \\textbf{Mean PCK (\\%)} \\\\")
    latex.append("\\midrule")

    # Find best values per metric
    best_per_metric = {}
    for metric in ['PCK@0.05', 'PCK@0.10', 'PCK@0.20']:
        metric_rows = [idx for idx in df_summary.index if idx[1] == metric]
        best_per_metric[metric] = df_summary.loc[metric_rows, 'Mean'].max()

    current_model = None
    for (model, metric), row in df_summary.iterrows():
        # Add model name only on first row
        if model != current_model:
            model_display = model.replace('_', '\\_')
            model_cell = f"\\multirow{{3}}{{*}}{{\\textbf{{{model_display}}}}}"
            current_model = model
        else:
            model_cell = ""

        value = row['Mean']
        value_str = f"{value:.2f}\\%"

        # Bold if best
        if abs(value - best_per_metric[metric]) < 0.01:
            value_str = f"\\textbf{{{value_str}}}"

        latex.append(f"{model_cell} & {metric} & {value_str} \\\\")

        if metric == "PCK@0.20":
            latex.append("\\midrule")

    # Remove last midrule
    if latex[-1] == "\\midrule":
        latex[-1] = "\\bottomrule"
    else:
        latex.append("\\bottomrule")

    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    latex_str = "\n".join(latex)

    with open(output_path, 'w') as f:
        f.write(latex_str)

    return latex_str


def print_comparison_summary(df):
    """
    Print a text summary of the comparison
    """
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)

    # Overall means
    print("\nOVERALL PERFORMANCE:")
    print("-" * 80)
    for (model, metric), row in df.iterrows():
        if 'Mean' in df.columns:
            print(f"{model:30s} {metric:12s}: {row['Mean']:6.2f}%")

    # Best model per metric
    print("\nBEST MODEL PER METRIC:")
    print("-" * 80)
    for metric in ['PCK@0.05', 'PCK@0.10', 'PCK@0.20']:
        metric_rows = [(idx, df.loc[idx, 'Mean']) for idx in df.index if idx[1] == metric]
        if metric_rows:
            best_model, best_value = max(metric_rows, key=lambda x: x[1])
            print(f"{metric:12s}: {best_model[0]:30s} ({best_value:.2f}%)")

    # Improvements from zero-shot
    if len(df.index.get_level_values(0).unique()) >= 2:
        print("\nIMPROVEMENTS FROM ZERO-SHOT:")
        print("-" * 80)

        zero_shot_model = df.index.get_level_values(0)[0]  # First model assumed to be zero-shot

        for metric in ['PCK@0.05', 'PCK@0.10', 'PCK@0.20']:
            zero_shot_value = df.loc[(zero_shot_model, metric), 'Mean']

            for model in df.index.get_level_values(0).unique():
                if model != zero_shot_model:
                    model_value = df.loc[(model, metric), 'Mean']
                    improvement = model_value - zero_shot_value
                    rel_improvement = (improvement / zero_shot_value) * 100
                    print(f"{model:30s} {metric:12s}: +{improvement:5.2f}% (relative: +{rel_improvement:5.2f}%)")

    # Best category per model
    print("\nBEST AND WORST CATEGORIES:")
    print("-" * 80)

    category_cols = [col for col in df.columns if col != 'Mean']

    for model in df.index.get_level_values(0).unique():
        # Use PCK@0.10 for comparison
        model_data = df.loc[(model, 'PCK@0.10'), category_cols]

        best_cat = model_data.idxmax()
        best_val = model_data.max()
        worst_cat = model_data.idxmin()
        worst_val = model_data.min()

        print(f"\n{model}:")
        print(f"  Best:  {best_cat:15s} {best_val:6.2f}%")
        print(f"  Worst: {worst_cat:15s} {worst_val:6.2f}%")
        print(f"  Range: {best_val - worst_val:6.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX comparison table across multiple model versions"
    )
    # Paths relative to results/step2/ — update the subdirectory names to match your run timestamps
    #dinov3
    args_zeroshot = os.path.join(paths.RESULTS_STEP2, 'dinov3/zero-shot/base/dinov3_base_spair71k_20251221_162509_512')
    args_finetuned = os.path.join(paths.RESULTS_STEP2, 'dinov3/finetuned/dinov3_base_spair71k_argmax_20260112_172817_3bl_15t_0p0001lr')
    args_finetuned_softargmax = os.path.join(paths.RESULTS_STEP2, 'dinov3/finetuned/dinov3_base_spair71k_wsoftargmax_20260113_100320_3bl_15t_0p0001')
    #dinov2
    # args_zeroshot = os.path.join(paths.RESULTS_STEP2, 'dinov2/zero_shot/base')
    # args_finetuned = os.path.join(paths.RESULTS_STEP2, 'dinov2/finetuned/1epoch_2blocks_argmax')
    # args_finetuned_softargmax = os.path.join(paths.RESULTS_STEP2, 'dinov2/finetuned/1epoch_2blocks_wsoftargmax_k9_t0p1')
    #SAM
    # args_zeroshot = os.path.join(paths.RESULTS_STEP2, 'SAM/vit_b_argmax_512')
    # args_finetuned = os.path.join(paths.RESULTS_STEP2, 'SAM/finetuned/SAM_finetuned_4bl_15t_0.0001lr_argmax_20260120_095246')
    # args_finetuned_softargmax = os.path.join(paths.RESULTS_STEP2, 'SAM/finetuned/SAM_finetuned_4bl_15t_0.0001lr__wsoftargmax_20260124_164741')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='tables',
        help='Directory to save generated LaTeX tables'
    )
    args_model_names=['Zero-shot', 'Fine-tuned', 'Fine-tuned + Soft-argmax']


    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 80)
    print("GENERATING MODEL COMPARISON TABLES")
    print("=" * 80)

    # Verify all directories exist
    results_dirs = [args_zeroshot, args_finetuned, args_finetuned_softargmax]
    for i, path in enumerate(results_dirs):
        if not Path(path).exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        print(f"✓ Found {args_model_names[i]}: {path}")

    print("\n[1/4] Loading data...")
    df = create_comparison_dataframe(results_dirs, args_model_names)
    print(f"✓ Loaded data for {len(df.columns)} categories")
    print(f"✓ Categories: {', '.join([c for c in df.columns if c != 'Mean'])}")

    print("\n[2/4] Generating full comparison table...")
    full_table_path = output_dir / 'model_category_comparison_full.tex'
    generate_latex_table(df, full_table_path, highlight_best=True)
    print(f"✓ Saved full table to: {full_table_path}")

    print("\n[3/4] Generating summary table...")
    summary_table_path = output_dir / 'model_overall_comparison.tex'
    generate_summary_table(df, summary_table_path)
    print(f"✓ Saved summary table to: {summary_table_path}")

    print("\n[4/4] Generating CSV for reference...")
    csv_path = output_dir / 'model_comparison.csv'
    df.to_csv(csv_path)
    print(f"✓ Saved CSV to: {csv_path}")

    # Print summary
    print_comparison_summary(df)

    print("\n" + "=" * 80)
    print("✓ ALL TABLES GENERATED SUCCESSFULLY!")
    print("=" * 80)

    print("\nGenerated files:")
    for file in sorted(output_dir.glob('*')):
        print(f"  - {file.name}")

    print("\nTo include in your LaTeX document:")
    print(f"  \\input{{{full_table_path}}}")
    print(f"  \\input{{{summary_table_path}}}")


if __name__ == "__main__":
    main()