# BALANCED EVALUATION REPORT v5
## Publication-Ready Dual Dataset Evaluation

---

## Executive Summary

### Full Dataset
- Total: 20,000
- Injections: 18,340 (91.7%)
- Benign: 1,660 (8.3%)
- Runtime sampling applied: 20,000/63,243 (max=20,000)

### Balanced Dataset
- Total: 10,343
- Injections: 5,093 (49.2%)
- Benign: 5,250 (50.8%)

## Dataset Difficulty Analysis

### Full Difficulty
- Avg length (chars): inj=1953.8, ben=173.6
- Avg length (tokens): inj=322.9, ben=26.6
- Vocabulary overlap (Jaccard): 0.1046 (shared=1914)
- Token distribution JS divergence: 0.3545
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
| full | Ablation: No Normalization | 0.999 | 1.000 | 0.997 | 1.000 | 1.000 | 1.000 |
| full | Ablation: No Semantic Model | 0.371 | 0.998 | 0.999 | 0.395 | 0.995 | 0.999 |
| full | Config A: Rules | 0.366 | 0.366 | 0.366 | 0.366 | 0.600 | 0.933 |
| full | Config B: Hybrid | 0.124 | 0.898 | 0.444 | 0.001 | 0.759 | 0.966 |
| full | Config C: Full | 0.371 | 0.998 | 0.999 | 0.395 | 0.995 | 0.999 |
| full | Semantic Baseline | 0.999 | 1.000 | 0.997 | 1.000 | 1.000 | 1.000 |
| balanced | Ablation: No Normalization | 0.999 | 0.999 | 0.981 | 0.999 | 1.000 | 1.000 |
| balanced | Ablation: No Semantic Model | 0.374 | 0.988 | 0.983 | 0.838 | 0.989 | 0.981 |
| balanced | Config A: Rules | 0.362 | 0.362 | 0.362 | 0.362 | 0.599 | 0.580 |
| balanced | Config B: Hybrid | 0.099 | 0.721 | 0.412 | 0.063 | 0.753 | 0.708 |
| balanced | Config C: Full | 0.374 | 0.988 | 0.983 | 0.838 | 0.989 | 0.981 |
| balanced | Semantic Baseline | 0.999 | 0.999 | 0.981 | 0.999 | 1.000 | 1.000 |

## Performance Comparison (Primary Threshold)

| Dataset | Model | Threshold | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| full | Ablation: No Normalization | 0.50 | 1.000 | 0.997 | 0.999 |
| full | Ablation: No Semantic Model | 0.50 | 1.000 | 0.228 | 0.371 |
| full | Config A: Rules | 0.50 | 0.990 | 0.225 | 0.366 |
| full | Config B: Hybrid | 0.50 | 0.968 | 0.067 | 0.124 |
| full | Config C: Full | 0.50 | 1.000 | 0.228 | 0.371 |
| full | Semantic Baseline | 0.50 | 1.000 | 0.998 | 0.999 |
| balanced | Ablation: No Normalization | 0.50 | 1.000 | 0.999 | 0.999 |
| balanced | Ablation: No Semantic Model | 0.50 | 1.000 | 0.230 | 0.374 |
| balanced | Config A: Rules | 0.50 | 0.879 | 0.228 | 0.362 |
| balanced | Config B: Hybrid | 0.50 | 0.631 | 0.054 | 0.099 |
| balanced | Config C: Full | 0.50 | 1.000 | 0.230 | 0.374 |
| balanced | Semantic Baseline | 0.50 | 1.000 | 0.999 | 0.999 |

## Robustness Check

| Dataset | Model | Scenario | F1 | F1 Drop vs Primary |
|---|---|---|---:|---:|
| full | Config B: Hybrid | shuffled_text | 0.049 | 0.076 |
| full | Config B: Hybrid | slight_perturbation | 0.125 | -0.000 |
| full | Config A: Rules | shuffled_text | 0.066 | 0.300 |
| full | Config A: Rules | slight_perturbation | 0.366 | -0.000 |
| full | Ablation: No Semantic Model | shuffled_text | 0.072 | 0.299 |
| full | Ablation: No Semantic Model | slight_perturbation | 0.371 | 0.000 |
| full | Semantic Baseline | shuffled_text | 0.998 | 0.001 |
| full | Semantic Baseline | slight_perturbation | 0.999 | 0.000 |
| full | Config C: Full | shuffled_text | 0.072 | 0.299 |
| full | Config C: Full | slight_perturbation | 0.371 | 0.000 |
| full | Ablation: No Normalization | shuffled_text | 0.999 | -0.000 |
| full | Ablation: No Normalization | slight_perturbation | 0.999 | 0.000 |
| balanced | Config B: Hybrid | shuffled_text | 0.036 | 0.063 |
| balanced | Config B: Hybrid | slight_perturbation | 0.099 | -0.001 |
| balanced | Config A: Rules | shuffled_text | 0.053 | 0.308 |
| balanced | Config A: Rules | slight_perturbation | 0.363 | -0.002 |
| balanced | Ablation: No Semantic Model | shuffled_text | 0.056 | 0.319 |
| balanced | Ablation: No Semantic Model | slight_perturbation | 0.374 | 0.000 |
| balanced | Semantic Baseline | shuffled_text | 0.999 | 0.000 |
| balanced | Semantic Baseline | slight_perturbation | 0.999 | 0.000 |
| balanced | Config C: Full | shuffled_text | 0.056 | 0.319 |
| balanced | Config C: Full | slight_perturbation | 0.374 | 0.000 |
| balanced | Ablation: No Normalization | shuffled_text | 0.999 | 0.000 |
| balanced | Ablation: No Normalization | slight_perturbation | 0.999 | 0.000 |

## Realism Warnings
- Dataset limitations: current corpora are mostly English and 2023-2024 attack styles.
- Potential over-separability: lexical overlap and token divergence can make some splits easier than production data.
- Balanced vs full differences are expected: balanced improves class parity diagnostics while full preserves real-world imbalance.
- Near-perfect semantic baseline scores should not be over-interpreted; they can indicate lexical memorization and benchmark easiness.

## Notes
- Thresholds are selected using validation split only.
- Test split is used only for final reporting.
- Report generated directly from results.json to prevent mismatch.