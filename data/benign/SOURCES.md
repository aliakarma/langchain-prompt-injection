# benign_corpus_v2 sources

This file is intentionally a mixed benign corpus used for false-positive testing.

## Composition
- curated: manually authored safe examples that quote attack phrases in explanatory context
- synthetic: generated operational text patterns (logs, support replies, Q&A phrasing)
- public text style: prompts written to mimic common documentation, StackOverflow answers, and incident updates

## Notes
- All records are schema-aligned JSONL with fields: id, text, label, attack_category
- Labels are benign (label=0) and attack_category is null
