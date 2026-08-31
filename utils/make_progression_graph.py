#!/usr/bin/env python3
"""Generate performance progression graph across all model versions."""
import csv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPORTS_DIR = "reports"

# Key versions to track (in chronological order)
VERSIONS = [
    ("v4", "dalle_v4", "Baseline (mixed data)"),
    ("v5", "dalle_v5_nocifake", "Drop CIFAKE"),
    ("v6", "dalle3_v6_sidadone", "SID_Set only"),
    ("v7", "dalle3_v7_cw5", "Class weighting"),
    ("v8", "dalle3_v8_aug_w15", "Transform aug"),
    ("v9", "dalle3_v9_strat_w15_nonoise", "Stratified aug"),
    ("v10", "dalle3_v10_combo5_w35_nonoise", "Combo5 + w3.5"),
    ("v11", "dalle3_v11_all_w45_nonoise", "All features + w4.5"),
]


def get_metrics(report_dir):
    """Get clean_acc and mean_transformed from a report CSV."""
    csv_path = os.path.join(REPORTS_DIR, report_dir, "robustness_report.csv")
    if not os.path.exists(csv_path):
        return None, None, None

    clean_acc = None
    transform_accs = []
    auroc = None

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            transform = row['transform'].strip()
            acc = float(row['accuracy'])
            if transform == 'clean':
                clean_acc = acc
                auroc = float(row['auroc'])
            else:
                transform_accs.append(acc)

    mean_transformed = np.mean(transform_accs) if transform_accs else None
    return clean_acc, mean_transformed, auroc


def main():
    labels = []
    clean_accs = []
    mean_accs = []
    aurocs = []

    for version, report_dir, desc in VERSIONS:
        clean, mean_t, auroc = get_metrics(report_dir)
        if clean is not None and mean_t is not None:
            labels.append(f"{version}\n{desc}")
            clean_accs.append(clean)
            mean_accs.append(mean_t)
            aurocs.append(auroc if auroc else 0)
            print(f"{version}: clean={clean:.4f}, mean_transformed={mean_t:.4f}, auroc={auroc:.4f}")
        else:
            print(f"{version}: NO DATA ({report_dir})")

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Model Progression — TikTok TechJam 2026 Track 5', fontsize=14, fontweight='bold')

    x = np.arange(len(labels))
    width = 0.6

    # Plot 1: Clean Accuracy
    bars1 = axes[0].bar(x, clean_accs, width, color='#2196F3', alpha=0.8)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Clean Accuracy')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([l.split('\n')[0] for l in labels], rotation=45)
    axes[0].set_ylim(0.7, 1.0)
    axes[0].axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='95% target')
    axes[0].legend()
    for bar, val in zip(bars1, clean_accs):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # Plot 2: Mean Transform Accuracy
    bars2 = axes[1].bar(x, mean_accs, width, color='#4CAF50', alpha=0.8)
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Mean Transform Accuracy')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([l.split('\n')[0] for l in labels], rotation=45)
    axes[1].set_ylim(0.7, 1.0)
    axes[1].axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='95% target')
    axes[1].legend()
    for bar, val in zip(bars2, mean_accs):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # Plot 3: AUROC
    bars3 = axes[2].bar(x, aurocs, width, color='#FF9800', alpha=0.8)
    axes[2].set_ylabel('AUROC')
    axes[2].set_title('Clean AUROC')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([l.split('\n')[0] for l in labels], rotation=45)
    axes[2].set_ylim(0.7, 1.0)
    axes[2].axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='95% target')
    axes[2].legend()
    for bar, val in zip(bars3, aurocs):
        axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('performance_progression.png', dpi=150, bbox_inches='tight')
    print("\nSaved: performance_progression.png")


if __name__ == "__main__":
    main()
