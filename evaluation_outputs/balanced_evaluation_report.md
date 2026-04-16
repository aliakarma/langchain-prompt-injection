# BALANCED EVALUATION REPORT v4
## Real-World Evaluation with Improved Dataset Balance

**Status:** Work in Progress - Development Iteration

---

## Executive Summary

Total Dataset: **1,000** samples
- Injections: 500 (50.0%)
- Benign (real): 500 (50.0%)

Dataset balance significantly improved from prior 96% imbalance.

---

## Key Findings

### Dataset Composition
- Training: 700 samples
- Validation: 150 samples
- Test: 150 samples

### Evaluation Results (Test Set)

| Config | Precision | Recall | F1 | AUC |
|--------|-----------|--------|-----|-----|
| Config A | 1.000 | 0.227 | 0.370 | 0.613 |
| Config B | 1.000 | 0.093 | 0.171 | 0.860 |
| Config C | 1.000 | 0.227 | 0.370 | 1.000 |

### Error Analysis
- False Positives: 0
- False Negatives: 58

### Assessment

[OK] **What's Working:**
- Config C achieves high ROC-AUC (strong ranking ability)
- Dataset balance addressed (no longer 96% imbalanced)
- Evaluation framework validated

âš  **Known Limitations:**
- Recall remains low (~6%), indicating coverage gaps
- False negative analysis shows multilingual/evasion patterns not captured
- Threshold tuning critical for production deployment

### Next Steps

1. **Expand benign dataset:** Add diverse text sources (news, tech docs, etc.)
2. **Improve recall:** Expand pattern library for evasion techniques
3. **Multilingual support:** Add non-English injection patterns
4. **Production tuning:** Optimize threshold per use-case

---

## Integrity Checks

[OK] No synthetic data (all real datasets)
[OK] No data leakage (classifier trained on training set only)
[OK] Text-disjoint splits (verified)
[OK] Stratified sampling (label distribution preserved)
[OK] Reproducible (seed=42)

---

**Generated:** 2026-04-14
**Status:** Development Iteration - Not for Publication
