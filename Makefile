# Makefile — langchain-prompt-injection
# ──────────────────────────────────────
# Targets:
#   install       Install runtime + dev dependencies
#   test          Run the full pytest suite with coverage
#   test-unit     Run only unit tests
#   test-int      Run only integration tests
#   lint          Run ruff linter
#   format        Run ruff formatter (auto-fix)
#   type          Run mypy type checker
#   check         lint + type (no auto-fix)
#   demo-block    Run the block-strategy demo
#   demo-annotate Run the annotate-strategy demo
#   demo-rag      Run the RAG pipeline demo
#   benchmark     Run the full three-config ablation benchmark
#   notebooks     Launch Jupyter with notebooks/
#   build         Build a distributable wheel
#   clean         Remove build artefacts and __pycache__

.PHONY: install test test-unit test-int lint format type check \
        demo-block demo-annotate demo-rag benchmark notebooks build clean

PYTHON   ?= python3
SRC_DIR   = src
TEST_DIR  = tests
COV_FLAGS = --cov=$(SRC_DIR) --cov-report=term-missing --cov-fail-under=85

# ── Environment ──────────────────────────────────────────────────────────

install:
	$(PYTHON) -m pip install -r requirements-dev.txt

# ── Tests ────────────────────────────────────────────────────────────────

test:
	$(PYTHON) -m pytest $(TEST_DIR) $(COV_FLAGS) -v

test-unit:
	$(PYTHON) -m pytest $(TEST_DIR)/unit $(COV_FLAGS) -v

test-int:
	$(PYTHON) -m pytest $(TEST_DIR)/integration -v

test-fast:
	$(PYTHON) -m pytest $(TEST_DIR) -q --tb=short

# ── Lint and format ──────────────────────────────────────────────────────

lint:
	$(PYTHON) -m ruff check $(SRC_DIR) $(TEST_DIR)

format:
	$(PYTHON) -m ruff format $(SRC_DIR) $(TEST_DIR)
	$(PYTHON) -m ruff check --fix $(SRC_DIR) $(TEST_DIR)

type:
	$(PYTHON) -m mypy $(SRC_DIR)/prompt_injection

check: lint type

# ── Demos ────────────────────────────────────────────────────────────────

demo-block:
	PYTHONPATH=$(SRC_DIR) $(PYTHON) demo/demo_block.py

demo-annotate:
	PYTHONPATH=$(SRC_DIR) $(PYTHON) demo/demo_annotate.py

demo-rag:
	PYTHONPATH=$(SRC_DIR) $(PYTHON) demo/demo_rag_pipeline.py

# ── Benchmark ────────────────────────────────────────────────────────────

benchmark:
	PYTHONPATH=$(SRC_DIR) $(PYTHON) -c "\
import sys; sys.path.insert(0,'$(SRC_DIR)'); \
from prompt_injection.evaluation.dataset import SyntheticDataset; \
from prompt_injection.evaluation.benchmark import BenchmarkRunner; \
from prompt_injection.evaluation.report import ReportSerializer; \
import pathlib; \
ds = SyntheticDataset(n_injections=250, n_benign=250, seed=42).generate(); \
tr, te = ds.train_test_split(0.20, seed=42); \
rw = SyntheticDataset(); \
[rw.load_from_path(p) for p in pathlib.Path('data/real').glob('*.jsonl')]; \
ext = SyntheticDataset(); \
ext_path = pathlib.Path('data/external/synthetic_stress_test.jsonl'); \
ext.load_external_dataset(ext_path, train_texts=set(tr.texts())) if ext_path.exists() else None; \
benign = SyntheticDataset(); \
benign_path = pathlib.Path('data/benign/benign_corpus_v2.jsonl'); \
benign.load_from_path(benign_path) if benign_path.exists() else None; \
result = BenchmarkRunner(n_latency_runs=30).run(tr, rw, te, external_eval_dataset=(ext if len(ext) else None), benign_dataset=(benign if len(benign) else None)); \
s = ReportSerializer(result); \
s.print_summary(); \
pathlib.Path('reports').mkdir(exist_ok=True); \
s.to_json('reports/benchmark.json'); \
s.to_csv('reports/benchmark.csv'); \
s.category_csv('reports/category_breakdown.csv'); \
"

# ── Notebooks ────────────────────────────────────────────────────────────

notebooks:
	cd notebooks && PYTHONPATH=../$(SRC_DIR) jupyter notebook

# ── Build ────────────────────────────────────────────────────────────────

build:
	$(PYTHON) -m pip install build --quiet
	$(PYTHON) -m build

# ── Clean ────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache  -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache  -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ reports/ .coverage htmlcov/
	@echo "Clean complete."
