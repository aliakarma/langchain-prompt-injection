#!/usr/bin/env python
import json

with open('notebooks/02_policy_evaluation.ipynb') as f:
    nb = json.load(f)

# Find PR curves cell by searching for "Best F1"
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        source_str = ''.join(source) if isinstance(source, list) else source
        if 'Best F1' in source_str:
            print(f'Found at cell {i}:')
            for j, line in enumerate(source):
                print(f'{j:2d}: {repr(line)}')
            break
