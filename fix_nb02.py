#!/usr/bin/env python
"""Fix the broken f-string in notebook 02 cell 6."""

import json

with open('notebooks/02_policy_evaluation.ipynb') as f:
    nb = json.load(f)

# Find and fix the PR curves cell
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        source_str = ''.join(source) if isinstance(source, list) else source
        if 'Best F1' in source_str and 'label=f' in source_str:
            print(f"Found broken cell at index {i}")
            # Reconstruct with fixed f-string (single line)
            cell['source'] = [
                'fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)\n',
                "colors = {'B: Regex + Scoring': '#e67e22', 'C: + Classifier': '#8e44ad'}\n",
                '\n',
                'for ax, (name, data) in zip(axes, sweeps.items()):\n',
                "    pts = data['sweep']\n",
                '    recalls    = [p.recall    for p in pts]\n',
                '    precisions = [p.precision for p in pts]\n',
                '    f1s        = [p.f1        for p in pts]\n',
                '\n',
                '    ax.plot(recalls, precisions, color=colors[name], linewidth=2)\n',
                '    best = max(pts, key=lambda p: p.f1)\n',
                "    ax.scatter([best.recall], [best.precision], color='red', s=80, zorder=5,\n",
                "               label=f'Best F1={best.f1:.3f} (thr={best.threshold:.2f})')\n",
                "    ax.axvline(0.80, color='grey', linestyle=':', alpha=0.6, label='Recall=0.80')\n",
                '    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")\n',
                '    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)\n',
                "    ax.set_title(name, fontweight='bold')\n",
                '    ax.legend(fontsize=9)\n',
                '\n',
                'plt.suptitle("Precision-Recall Curves", fontweight=\'bold\')\n',
                'plt.tight_layout()\n',
                'plt.savefig("pr_curves.png", dpi=130, bbox_inches=\'tight\')\n',
                'plt.show()'
            ]
            print("Fixed! Saving...")
            break

with open('notebooks/02_policy_evaluation.ipynb', 'w') as f:
    json.dump(nb, f, indent=4)
print("Done!")
