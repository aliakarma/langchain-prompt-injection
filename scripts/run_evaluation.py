#!/usr/bin/env python
"""
Unified Evaluation Pipeline for langchain-prompt-injection.
===========================================================

This script handles both the minimal offline benchmark (using `data/sample`) 
and the full evaluation pipeline (using external downloaded datasets).
"""

import sys
import logging
import argparse
import pathlib
import time

# Ensure src/ exists in path
sys.path.insert(0, 'src')

from prompt_injection.evaluation.dataset import SyntheticDataset
from prompt_injection.evaluation.benchmark import BenchmarkRunner
from prompt_injection.evaluation.report import ReportSerializer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run Evaluation Pipeline")
    parser.add_argument("--mode", type=str, choices=["minimal", "full"], default="minimal",
                        help="Choose 'minimal' for offline sample dataset, 'full' for external datasets.")
    args = parser.parse_args()
    
    logger.info(f"=======================================")
    logger.info(f"Running Evaluation Pipeline: {args.mode.upper()} MODE")
    logger.info(f"=======================================")
    
    if args.mode == "minimal":
        # MINIMAL MODE (Offline) 
        # Rely primarily on the included small dataset: data/sample
        
        sample_inj = pathlib.Path('data/sample/injections.jsonl')
        sample_benign = pathlib.Path('data/sample/benign.jsonl')
        
        if not sample_inj.exists() or not sample_benign.exists():
            logger.error(f"Sample datasets missing at {sample_inj} or {sample_benign}. Please check the repo.")
            sys.exit(1)
        
        logger.info("Loading minimal sample datasets for offline benchmarking...")
        ds_inj = SyntheticDataset()
        ds_inj.load_from_path(sample_inj)
        
        ds_benign = SyntheticDataset()
        ds_benign.load_from_path(sample_benign)
        
        # Merge for training
        logger.info("Setting up Train/Test splits...")
        ds_all = SyntheticDataset(n_injections=0, n_benign=0, seed=42)
        ds_all._records = ds_inj._records + ds_benign._records
        tr, te = ds_all.train_test_split(0.20, seed=42)
        
        rw = ds_all  # use the same as "real world" equivalent for minimal
        ext = None
        benign_val = ds_benign
        
    else:
        # FULL MODE
        # Uses synthetic data, large 'data/real' and 'data/external' datasets if available
        
        logger.info("Generating synthetic datasets (n=250)...")
        ds = SyntheticDataset(n_injections=250, n_benign=250, seed=42).generate()
        tr, te = ds.train_test_split(0.20, seed=42)
        
        logger.info("Loading data/real datasets...")
        rw = SyntheticDataset()
        for p in pathlib.Path('data/real').glob('*.jsonl'):
            rw.load_from_path(p)
            
        logger.info("Loading external dataset (if any)...")
        ext = SyntheticDataset()
        ext_path = pathlib.Path('data/external_raw/hackaprompt_raw.jsonl')
        if ext_path.exists():
            ext.load_external_dataset(ext_path, train_texts=set(tr.texts()))
        else:
            logger.warning("External dataset not found. Make sure to run 'bash scripts/download_datasets.sh' beforehand.")
            ext = None
            
        logger.info("Loading large benign datasets...")
        benign_val = SyntheticDataset()
        benign_path = pathlib.Path('data/benign/benign_real_v2.jsonl')
        if benign_path.exists():
            benign_val.load_from_path(benign_path)
        else:
            benign_val = None

    logger.info("Running benchmark...")
    start = time.time()
    runner = BenchmarkRunner(n_latency_runs=5 if args.mode == "minimal" else 30)
    result = runner.run(
        train_dataset=tr,
        real_world_dataset=rw,
        test_dataset=te,
        external_eval_dataset=ext if ext and len(ext) > 0 else None,
        benign_dataset=benign_val if benign_val and len(benign_val) > 0 else None
    )
    elapsed = time.time() - start
    logger.info(f"Benchmark completed in {elapsed:.2f}s.")

    logger.info("Generating reports...")
    s = ReportSerializer(result)
    s.print_summary()
    
    reports_dir = pathlib.Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    s.to_json(reports_dir / "benchmark.json")
    s.to_csv(reports_dir / "benchmark.csv")
    s.category_csv(reports_dir / "category_breakdown.csv")
    
    logger.info(f"Reports successfully generated in '{reports_dir}'")

if __name__ == "__main__":
    main()
