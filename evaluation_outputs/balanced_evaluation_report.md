# BALANCED EVALUATION REPORT v5
## Publication-Ready Dual Dataset Evaluation

---

## Executive Summary

### Full Dataset
- Total: 5,000
- Injections: 4,585 (91.7%)
- Benign: 415 (8.3%)
- Runtime sampling applied: 5,000/63,243 (max=5,000)

### Balanced Dataset
- Total: 10,343
- Injections: 5,093 (49.2%)
- Benign: 5,250 (50.8%)

## Dataset Difficulty Analysis

### Full Difficulty
- Avg length (chars): inj=1941.2, ben=174.2
- Avg length (tokens): inj=320.9, ben=26.4
- Vocabulary overlap (Jaccard): 0.1091 (shared=1176)
- Token distribution JS divergence: 0.3578
- Trivial separability risk: True
- Note: Classes appear potentially over-separable (low vocab overlap + high token divergence).

### Balanced Difficulty
- Avg length (chars): inj=1950.2, ben=173.2
- Avg length (tokens): inj=321.4, ben=26.5
- Vocabulary overlap (Jaccard): 0.1054 (shared=1678)
- Token distribution JS divergence: 0.3399
- Trivial separability risk: True
- Note: Classes appear potentially over-separable (low vocab overlap + high token divergence).

## Evaluation Table (Test Set)

| Dataset | Model | Primary F1@0.5 | Optimal F1 | F1@FPR<=0.05 | F1@FPR<=0.01 | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| full | Ablation: No Normalization | 0.999 | 0.999 | 0.996 | 0.999 | 1.000 | 1.000 |
| full | Ablation: No Semantic Model | 0.368 | 0.994 | 0.993 | 0.994 | 0.936 | 0.988 |
| full | Config A: Rules | 0.362 | 0.362 | 0.362 | 0.362 | 0.572 | 0.928 |
| full | Config B: Hybrid | 0.107 | 0.881 | 0.439 | 0.433 | 0.692 | 0.947 |
| full | Config C: Full | 0.368 | 0.994 | 0.993 | 0.994 | 0.936 | 0.988 |
| full | Semantic Baseline | 0.999 | 1.000 | 0.994 | 1.000 | 1.000 | 1.000 |
| balanced | Ablation: No Normalization | 0.999 | 0.999 | 0.981 | 0.999 | 1.000 | 1.000 |
| balanced | Ablation: No Semantic Model | 0.374 | 0.988 | 0.983 | 0.841 | 0.989 | 0.981 |
| balanced | Config A: Rules | 0.362 | 0.362 | 0.362 | 0.362 | 0.599 | 0.580 |
| balanced | Config B: Hybrid | 0.099 | 0.721 | 0.412 | 0.063 | 0.753 | 0.708 |
| balanced | Config C: Full | 0.374 | 0.988 | 0.983 | 0.841 | 0.989 | 0.981 |
| balanced | Semantic Baseline | 0.999 | 0.999 | 0.980 | 0.999 | 1.000 | 1.000 |

## Performance Comparison (Primary Threshold)

| Dataset | Model | Threshold | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| full | Ablation: No Normalization | 0.50 | 0.999 | 0.999 | 0.999 |
| full | Ablation: No Semantic Model | 0.50 | 0.975 | 0.227 | 0.368 |
| full | Config A: Rules | 0.50 | 0.968 | 0.223 | 0.362 |
| full | Config B: Hybrid | 0.50 | 0.886 | 0.057 | 0.107 |
| full | Config C: Full | 0.50 | 0.975 | 0.227 | 0.368 |
| full | Semantic Baseline | 0.50 | 1.000 | 0.999 | 0.999 |
| balanced | Ablation: No Normalization | 0.50 | 1.000 | 0.999 | 0.999 |
| balanced | Ablation: No Semantic Model | 0.50 | 1.000 | 0.230 | 0.374 |
| balanced | Config A: Rules | 0.50 | 0.879 | 0.228 | 0.362 |
| balanced | Config B: Hybrid | 0.50 | 0.631 | 0.054 | 0.099 |
| balanced | Config C: Full | 0.50 | 1.000 | 0.230 | 0.374 |
| balanced | Semantic Baseline | 0.50 | 1.000 | 0.999 | 0.999 |

## Robustness Check

| Dataset | Model | Scenario | F1 | F1 Drop vs Primary |
|---|---|---|---:|---:|
| full | Ablation: No Semantic Model | shuffled_text | 0.059 | 0.309 |
| full | Ablation: No Semantic Model | slight_perturbation | 0.369 | -0.000 |
| full | Config C: Full | shuffled_text | 0.059 | 0.309 |
| full | Config C: Full | slight_perturbation | 0.369 | -0.000 |
| full | Config A: Rules | shuffled_text | 0.054 | 0.308 |
| full | Config A: Rules | slight_perturbation | 0.363 | -0.001 |
| full | Ablation: No Normalization | shuffled_text | 0.999 | 0.000 |
| full | Ablation: No Normalization | slight_perturbation | 0.999 | 0.000 |
| full | Config B: Hybrid | shuffled_text | 0.032 | 0.075 |
| full | Config B: Hybrid | slight_perturbation | 0.107 | -0.000 |
| full | Semantic Baseline | shuffled_text | 0.999 | 0.000 |
| full | Semantic Baseline | slight_perturbation | 0.999 | 0.000 |
| balanced | Ablation: No Semantic Model | shuffled_text | 0.056 | 0.319 |
| balanced | Ablation: No Semantic Model | slight_perturbation | 0.374 | 0.000 |
| balanced | Config C: Full | shuffled_text | 0.056 | 0.319 |
| balanced | Config C: Full | slight_perturbation | 0.374 | 0.000 |
| balanced | Config A: Rules | shuffled_text | 0.053 | 0.308 |
| balanced | Config A: Rules | slight_perturbation | 0.363 | -0.002 |
| balanced | Ablation: No Normalization | shuffled_text | 0.999 | 0.000 |
| balanced | Ablation: No Normalization | slight_perturbation | 0.999 | 0.000 |
| balanced | Config B: Hybrid | shuffled_text | 0.036 | 0.063 |
| balanced | Config B: Hybrid | slight_perturbation | 0.099 | -0.001 |
| balanced | Semantic Baseline | shuffled_text | 0.999 | 0.000 |
| balanced | Semantic Baseline | slight_perturbation | 0.999 | 0.000 |

## Length Normalization Experiment

All samples were truncated to the first 200 whitespace-delimited tokens and re-evaluated.

| Dataset | Model | Original F1@0.5 | Length-Normalized F1@0.5 | Delta (Norm - Orig) |
|---|---|---:|---:|---:|
| full | Ablation: No Semantic Model | 0.368 | 0.363 | -0.006 |
| full | Config C: Full | 0.368 | 0.363 | -0.006 |
| full | Config A: Rules | 0.362 | 0.356 | -0.006 |
| full | Ablation: No Normalization | 0.999 | 0.999 | 0.000 |
| full | Config B: Hybrid | 0.107 | 0.102 | -0.005 |
| full | Semantic Baseline | 0.999 | 0.999 | 0.000 |
| balanced | Ablation: No Semantic Model | 0.374 | 0.374 | 0.000 |
| balanced | Config C: Full | 0.374 | 0.374 | 0.000 |
| balanced | Config A: Rules | 0.362 | 0.362 | 0.000 |
| balanced | Ablation: No Normalization | 0.999 | 0.999 | -0.001 |
| balanced | Config B: Hybrid | 0.099 | 0.097 | -0.002 |
| balanced | Semantic Baseline | 0.999 | 0.999 | 0.000 |

### Impact Interpretation
Large negative deltas after truncation suggest reliance on long-context lexical cues or text-length artifacts; small deltas indicate stronger dependence on local, shorter-span signals.

## Realism Warnings
- Dataset limitations: current corpora are mostly English and 2023-2024 attack styles.
- Potential over-separability: lexical overlap and token divergence can make some splits easier than production data.
- Balanced vs full differences are expected: balanced improves class parity diagnostics while full preserves real-world imbalance.
- Near-perfect semantic baseline scores should not be over-interpreted; they can indicate lexical memorization and benchmark easiness.

## Notes
- Thresholds are selected using validation split only.
- Test split is used only for final reporting.
- Report generated directly from results.json to prevent mismatch.