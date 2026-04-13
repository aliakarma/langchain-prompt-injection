#!/usr/bin/env python
"""Run all 5 notebooks and report status."""

import nbclient
import nbformat
from pathlib import Path
import sys

notebooks = [
    "notebooks/01_detector_experiments.ipynb",
    "notebooks/02_policy_evaluation.ipynb",
    "notebooks/03_agent_integration_demo.ipynb",
    "notebooks/04_rag_injection_testing.ipynb",
    "notebooks/05_evaluation_metrics.ipynb",
]

results = {}
for nb_path_str in notebooks:
    nb_path = Path(nb_path_str)
    try:
        print(f"\n{'='*60}")
        print(f"Running: {nb_path.name}")
        print(f"{'='*60}")
        
        nb = nbformat.read(nb_path, as_version=4)
        client = nbclient.NotebookClient(nb, timeout=180)
        client.execute()
        results[nb_path.name] = "[PASS]"
        print(f"[PASS] {nb_path.name} executed successfully\n")
        
    except Exception as e:
        results[nb_path.name] = f"[FAIL]: {str(e)[:80]}"
        print(f"[FAIL] {nb_path.name} failed: {e}\n")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for nb_name, status in results.items():
    print(f"{nb_name:40s} {status}")

passed = sum(1 for s in results.values() if "[PASS]" in s)
total = len(results)
print(f"\nTotal: {passed}/{total} passed")

sys.exit(0 if passed == total else 1)
