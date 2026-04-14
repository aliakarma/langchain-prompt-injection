# Dataset Documentation

## Directory Layout

```
data/
├── synthetic/
│   ├── injections.jsonl   # 250 positive (injection) records
│   └── benign.jsonl       # 250 negative (benign) records
└── real/
  ├── injections_real_v4.jsonl  # Final real injection examples used in the latest evaluation
    └── SOURCES.md             # Attribution and licensing

└── benign/
  └── benign_real_v2.jsonl      # Final real benign examples used in the latest evaluation
```

## Record Schema

Every JSONL record follows this schema:

```json
{
  "id":              "syn-inj-0001",
  "text":            "Ignore all previous instructions...",
  "label":           1,
  "attack_category": "instruction_override",
  "source_type":     "user",
  "severity":        "high"
}
```

| Field | Type | Values |
|---|---|---|
| `id` | `str` | Stable unique identifier |
| `text` | `str` | The prompt / message text |
| `label` | `int` | `1` = injection, `0` = benign |
| `attack_category` | `str \| null` | Attack category (null for benign records) |
| `source_type` | `str` | `user`, `rag`, `tool`, `file`, `web` |
| `severity` | `str \| null` | `low`, `medium`, `high` (null for benign) |

## Attack Categories

| Category | Description |
|---|---|
| `instruction_override` | Attempts to supersede existing instructions |
| `system_prompt_extraction` | Tries to reveal the system prompt |
| `role_hijacking` | Redefines the assistant's identity or role |
| `dan_jailbreak` | "Do Anything Now" and mode-switch attacks |
| `boundary_escape` | Bypasses safety guidelines or content policies |
| `social_engineering` | Fake authority claims (admin, developer, etc.) |
| `indirect_injection` | Injected content in RAG chunks, tool outputs, or files |
| `evasion` | Obfuscated variants (letter spacing, base64, homoglyphs) |

## Loading the Dataset

```python
from prompt_injection.evaluation.dataset import SyntheticDataset

# Load synthetic data
ds = SyntheticDataset(n_injections=250, n_benign=250, seed=42).generate()

# Load from JSONL files on disk
ds = SyntheticDataset()
ds.load_from_path("data/synthetic/injections.jsonl")
ds.load_from_path("data/synthetic/benign.jsonl")

# Load real-world sample
rw = SyntheticDataset()
rw.load_from_path("data/real/injections_real_v4.jsonl")
rw.load_from_path("data/benign/benign_real_v2.jsonl")

# Train/test split (stratified)
train_ds, test_ds = ds.train_test_split(test_size=0.20, seed=42)
```

## Extending with Real-World Data

This dataset is intentionally small as a baseline.  To extend it:

1. Add new JSONL records following the schema above.
2. Call `SyntheticDataset.load_from_path(path)` — it merges into existing records.

### Recommended Datasets for Large-Scale Extension

| Dataset | Size | Description |
|---|---|---|
| **HackAPrompt** (Schulhoff et al., NeurIPS 2023) | 600K+ | Competition submissions, publicly released |
| **BIPIA** (Yi et al., 2023) | ~2K | Indirect prompt injection benchmark |
| **JailbreakBench** (Chao et al., 2024) | ~1K | Standardised jailbreak evaluation |
| **PromptBench** (Zhu et al., 2023) | ~5K | Adversarial prompt robustness |

## Reproducibility

The synthetic generator uses a fixed random seed for full reproducibility:

```python
ds1 = SyntheticDataset(n_injections=250, n_benign=250, seed=42).generate()
ds2 = SyntheticDataset(n_injections=250, n_benign=250, seed=42).generate()
assert ds1.texts() == ds2.texts()  # True
```

Changing the seed produces a different shuffling and augmentation of the templates.
