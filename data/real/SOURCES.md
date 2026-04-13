# Real-World Dataset Sources

This directory contains a small manually curated sample of real-world
prompt-injection examples drawn from publicly documented sources.

## Attribution

| ID Range | Source | Notes |
|---|---|---|
| real-inj-001 to real-inj-005 | [Liu et al. (2023) — "Prompt Injection attack against LLM-integrated Applications"](https://arxiv.org/abs/2306.05499) | Adapted from published examples |
| real-inj-006 to real-inj-010 | [HackAPrompt competition dataset (Schulhoff et al., NeurIPS 2023)](https://github.com/DavyJ0nes/hackaprompt) | Publicly released competition dataset |
| real-inj-011 to real-inj-015 | [BIPIA benchmark (Yi et al., 2023)](https://arxiv.org/abs/2312.14197) | Indirect injection examples from benchmark |
| real-inj-016 to real-inj-020 | Public red-teaming disclosures (various) | Documented in public blog posts and security advisories |
| real-inj-021 to real-inj-025 | Manually authored | Novel examples based on documented attack patterns |
| real-ben-001 to real-ben-025 | Manually authored | Benign examples representative of production workloads |

## License

Examples sourced from published academic datasets are used for research
and evaluation purposes only, consistent with their original licenses.
Manually authored examples in this repository are released under the
same license as the project (MIT).

## Future Extension

This dataset is intentionally small (25 injection + 25 benign examples)
as a held-out real-world evaluation slice. To extend it, add any JSONL
file following this schema:

```json
{
  "id": "unique-string",
  "text": "the prompt text",
  "label": 1,
  "attack_category": "instruction_override",
  "source_type": "user",
  "severity": "high"
}
```

Recommended datasets for large-scale extension:

- **PromptBench** (Zhu et al., 2023) — adversarial prompt robustness benchmark
- **HackAPrompt** (Schulhoff et al., NeurIPS 2023) — 600K+ competition submissions
- **BIPIA** (Yi et al., 2023) — benchmark for indirect prompt injection attacks
- **JailbreakBench** (Chao et al., 2024) — unified jailbreak evaluation framework

Use `SyntheticDataset.load_from_path(path)` to merge any compliant JSONL
file into the evaluation pipeline.
