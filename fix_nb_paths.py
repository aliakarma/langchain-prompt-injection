#!/usr/bin/env python
"""Fix relative paths in notebooks 04 and 05."""

import nbformat

for nb_file in ['notebooks/04_rag_injection_testing.ipynb', 'notebooks/05_evaluation_metrics.ipynb']:
    print(f"Fixing paths in {nb_file}...")
    nb = nbformat.read(nb_file, as_version=4)
    
    modified = False
    for cell in nb.cells:
        if cell.cell_type == 'code':
            # Replace ../data with data
            if isinstance(cell.source, list):
                new_source = [line.replace('../data/', 'data/') for line in cell.source]
                if new_source != cell.source:
                    cell.source = new_source
                    modified = True
            else:
                new_source = cell.source.replace('../data/', 'data/')
                if new_source != cell.source:
                    cell.source = new_source
                    modified = True
    
    if modified:
        nbformat.write(nb, nb_file)
        print(f"  ✓ Fixed and saved {nb_file}")
    else:
        print(f"  (no changes needed)")

print("Done!")
