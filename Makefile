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
	$(PYTHON) scripts/run_evaluation.py --mode minimal

benchmark-full:
	$(PYTHON) scripts/run_evaluation.py --mode full

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
