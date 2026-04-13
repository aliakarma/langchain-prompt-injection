#!/usr/bin/env python
"""Fix notebook 05 to use correct BenchmarkRunner.run() signature."""

import json

with open('notebooks/05_evaluation_metrics.ipynb') as f:
    nb = json.load(f)

# Find and fix the BenchmarkRunner.run call
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        source_str = ''.join(source) if isinstance(source, list) else source
        if 'runner.run(' in source_str and 'real_world_dataset=' in source_str:
            print(f"Found old API call at cell {i}")
            # Replace old: runner.run(train_ds, test_ds, real_world_dataset=rw)
            # With new: runner.run(train_ds, rw, test_ds)
            new_source = []
            for line in source:
                if 'runner.run(' in line:
                    # This is a multi-line call, reconstruct it
                    new_source.append('result = runner.run(train_ds, rw, test_ds)\n')
                    # Skip the old multi-line call - just output one new line
                    continue
                elif 'real_world_dataset=rw' in line or ')' in line and i > 0:
                    # Skip closing paren from old call
                    continue
                else:
                    new_source.append(line)
            
            # Properly reconstruct the cell
            cell['source'] = [
                'runner = BenchmarkRunner(\n',
                '    threshold=0.50,\n',
                '    n_latency_runs=30,\n',
                '    latency_budget_ms=10.0,\n',
                '    sweep_thresholds=True,\n',
                ')\n',
                'result = runner.run(train_ds, rw, test_ds)\n',
                'print("Benchmark complete.")'
            ]
            print("Fixed! Saving...")
            break

with open('notebooks/05_evaluation_metrics.ipynb', 'w') as f:
    json.dump(nb, f, indent=4)
print("Done!")
