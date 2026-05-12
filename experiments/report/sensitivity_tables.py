import pandas as pd
import argparse
from pathlib import Path


def generate_keypoint_consistency_latex_table(results_dir, output_path, top_n=10):
    """
    Generate a LaTeX table for keypoint consistency analysis

    Args:
        results_dir: path to results directory containing category_keypoint_consistency.csv
        output_path: where to save the .tex file
        top_n: number of categories to include (sorted by std_accuracy)
    """
    # Load the consistency data
    csv_path = Path(results_dir) / 'category_keypoint_consistency.csv'

    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Sort by std_accuracy (most inconsistent first) and take top N
    df_sorted = df.sort_values('std_accuracy', ascending=False).head(top_n)

    # Start LaTeX table
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append(
        "\\caption{Keypoint consistency analysis across object categories. Categories are sorted by standard deviation of keypoint accuracy, with higher values indicating greater inconsistency in keypoint difficulty.}")
    latex.append("\\label{tab:keypoint_consistency}")
    latex.append("\\resizebox{\\textwidth}{!}{%")
    latex.append("\\begin{tabular}{lcccccc}")
    latex.append("\\toprule")
    latex.append(
        "\\textbf{Category} & \\textbf{Mean Acc.} & \\textbf{Std Dev} & \\textbf{Min Acc.} & \\textbf{Max Acc.} & \\textbf{Range} & \\textbf{\\# KPs} \\\\")
    latex.append("\\midrule")

    # Add data rows
    for idx, row in df_sorted.iterrows():
        category = row['category'].replace('_', '\\_').title()
        mean_acc = row['mean_accuracy']
        std_acc = row['std_accuracy']
        min_acc = row['min_accuracy']
        max_acc = row['max_accuracy']
        range_acc = row['range']
        num_kps = int(row['num_keypoints'])

        # Highlight the most inconsistent category (first row)
        if idx == df_sorted.index[0]:
            latex.append(
                f"\\textbf{{{category}}} & {mean_acc:.2f}\\% & \\textbf{{{std_acc:.2f}\\%}} & {min_acc:.2f}\\% & {max_acc:.2f}\\% & {range_acc:.2f}\\% & {num_kps} \\\\")
        else:
            latex.append(
                f"{category} & {mean_acc:.2f}\\% & {std_acc:.2f}\\% & {min_acc:.2f}\\% & {max_acc:.2f}\\% & {range_acc:.2f}\\% & {num_kps} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}%")
    latex.append("}")
    latex.append("\\end{table}")

    # Write to file
    latex_str = "\n".join(latex)

    with open(output_path, 'w') as f:
        f.write(latex_str)

    print(f"✓ Saved keypoint consistency table to: {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("KEYPOINT CONSISTENCY ANALYSIS")
    print("=" * 80)
    print("\nCategories with most inconsistent keypoint difficulty:")
    print(df_sorted.to_string(index=False))
    print("\n" + "=" * 80)

    return latex_str


def generate_threshold_sensitivity_latex_table(results_dir, output_path, top_n=10):
    """
    Generate a LaTeX table for threshold sensitivity analysis

    Args:
        results_dir: path to results directory containing threshold_sensitivity.csv
        output_path: where to save the .tex file
        top_n: number of categories to include (sorted by sensitivity)
    """
    # Load the sensitivity data
    csv_path = Path(results_dir) / 'threshold_sensitivity.csv'

    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Sort by sensitivity (least sensitive = most precise)
    df_sorted = df.sort_values('sensitivity', ascending=True).head(top_n)

    # Start LaTeX table
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append(
        "\\caption{Threshold sensitivity analysis across object categories. Lower sensitivity indicates more precise predictions. Sensitivity is calculated as the difference between PCK@0.20 and PCK@0.05.}")
    latex.append("\\label{tab:threshold_sensitivity}")
    latex.append("\\begin{tabular}{lcccc}")
    latex.append("\\toprule")
    latex.append(
        "\\textbf{Category} & \\textbf{PCK@0.05} & \\textbf{PCK@0.10} & \\textbf{PCK@0.20} & \\textbf{Sensitivity} \\\\")
    latex.append("\\midrule")

    # Add data rows
    for idx, row in df_sorted.iterrows():
        category = row['category'].replace('_', '\\_').title()
        pck_005 = row['pck@0.05']
        pck_010 = row['pck@0.10']
        pck_020 = row['pck@0.20']
        sensitivity = row['sensitivity']

        # Highlight the least sensitive category (first row)
        if idx == df_sorted.index[0]:
            latex.append(
                f"\\textbf{{{category}}} & {pck_005:.2f}\\% & {pck_010:.2f}\\% & {pck_020:.2f}\\% & \\textbf{{{sensitivity:.2f}\\%}} \\\\")
        else:
            latex.append(
                f"{category} & {pck_005:.2f}\\% & {pck_010:.2f}\\% & {pck_020:.2f}\\% & {sensitivity:.2f}\\% \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    # Write to file
    latex_str = "\n".join(latex)

    with open(output_path, 'w') as f:
        f.write(latex_str)

    print(f"✓ Saved threshold sensitivity table to: {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 80)
    print("\nCategories by threshold sensitivity (low = more precise predictions):")
    print(df_sorted.to_string(index=False))
    print("\n" + "=" * 80)

    return latex_str


def generate_difficulty_extremes_latex_table(results_dir, output_path):
    """
    Generate a LaTeX table for hardest/easiest keypoints per category

    Args:
        results_dir: path to results directory containing keypoint_difficulty_extremes.csv
        output_path: where to save the .tex file
    """
    # Load the difficulty extremes data
    csv_path = Path(results_dir) / 'keypoint_difficulty_extremes.csv'

    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Sort by accuracy gap (largest gap first)
    df_sorted = df.sort_values('accuracy_gap', ascending=False)

    # Start LaTeX table
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append(
        "\\caption{Hardest and easiest keypoints per category. Accuracy gap represents the difference between the easiest and hardest keypoint in each category, indicating the range of keypoint difficulty.}")
    latex.append("\\label{tab:keypoint_difficulty_extremes}")
    latex.append("\\resizebox{\\textwidth}{!}{%")
    latex.append("\\begin{tabular}{lccccc}")
    latex.append("\\toprule")
    latex.append(
        "\\textbf{Category} & \\textbf{Hardest KP} & \\textbf{Hard Acc.} & \\textbf{Easiest KP} & \\textbf{Easy Acc.} & \\textbf{Gap} \\\\")
    latex.append("\\midrule")

    # Add data rows
    for idx, row in df_sorted.iterrows():
        category = row['category'].replace('_', '\\_').title()
        hardest_kp = int(row['hardest_keypoint'])
        hardest_acc = row['hardest_accuracy']
        easiest_kp = int(row['easiest_keypoint'])
        easiest_acc = row['easiest_accuracy']
        gap = row['accuracy_gap']

        # Highlight the category with largest gap (first row)
        if idx == df_sorted.index[0]:
            latex.append(
                f"\\textbf{{{category}}} & {hardest_kp} & {hardest_acc:.2f}\\% & {easiest_kp} & {easiest_acc:.2f}\\% & \\textbf{{{gap:.2f}\\%}} \\\\")
        else:
            latex.append(
                f"{category} & {hardest_kp} & {hardest_acc:.2f}\\% & {easiest_kp} & {easiest_acc:.2f}\\% & {gap:.2f}\\% \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}%")
    latex.append("}")
    latex.append("\\end{table}")

    # Write to file
    latex_str = "\n".join(latex)

    with open(output_path, 'w') as f:
        f.write(latex_str)

    print(f"✓ Saved keypoint difficulty extremes table to: {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("HARDEST AND EASIEST KEYPOINTS PER CATEGORY")
    print("=" * 80)
    print("\nCategories sorted by accuracy gap:")
    print(df_sorted.to_string(index=False))
    print("\n" + "=" * 80)

    return latex_str


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables for keypoint difficulty analysis"
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Path to results directory containing analysis CSVs'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='tables',
        help='Directory to save generated LaTeX tables'
    )
    parser.add_argument(
        '--top_n',
        type=int,
        default=10,
        help='Number of categories to include in consistency and sensitivity tables'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    print("=" * 80)
    print("GENERATING KEYPOINT DIFFICULTY ANALYSIS TABLES")
    print("=" * 80)
    print(f"\nResults directory: {results_dir}")
    print(f"Output directory: {output_dir}")

    # 1. Keypoint Consistency Table
    print("\n[1/3] Generating keypoint consistency table...")
    try:
        consistency_path = output_dir / 'keypoint_consistency_table.tex'
        generate_keypoint_consistency_latex_table(
            results_dir,
            consistency_path,
            top_n=args.top_n
        )
    except FileNotFoundError as e:
        print(f"⚠ Skipping consistency table: {e}")

    # 2. Threshold Sensitivity Table
    print("\n[2/3] Generating threshold sensitivity table...")
    try:
        sensitivity_path = output_dir / 'threshold_sensitivity_table.tex'
        generate_threshold_sensitivity_latex_table(
            results_dir,
            sensitivity_path,
            top_n=args.top_n
        )
    except FileNotFoundError as e:
        print(f"⚠ Skipping sensitivity table: {e}")

    # 3. Difficulty Extremes Table
    print("\n[3/3] Generating difficulty extremes table...")
    try:
        extremes_path = output_dir / 'keypoint_difficulty_extremes_table.tex'
        generate_difficulty_extremes_latex_table(
            results_dir,
            extremes_path
        )
    except FileNotFoundError as e:
        print(f"⚠ Skipping extremes table: {e}")

    print("\n" + "=" * 80)
    print("✓ TABLE GENERATION COMPLETE!")
    print("=" * 80)

    print("\nGenerated files:")
    for file in sorted(output_dir.glob('*.tex')):
        print(f"  - {file.name}")

    print("\nTo include in your LaTeX document:")
    if (output_dir / 'keypoint_consistency_table.tex').exists():
        print(f"  \\input{{{output_dir / 'keypoint_consistency_table.tex'}}}")
    if (output_dir / 'threshold_sensitivity_table.tex').exists():
        print(f"  \\input{{{output_dir / 'threshold_sensitivity_table.tex'}}}")
    if (output_dir / 'keypoint_difficulty_extremes_table.tex').exists():
        print(f"  \\input{{{output_dir / 'keypoint_difficulty_extremes_table.tex'}}}")


if __name__ == "__main__":
    main()