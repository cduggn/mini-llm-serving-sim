VENV := .venv
PYTHON ?= $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
export PYTHONPATH := $(CURDIR)

.PHONY: setup test trace part2 part2-overload router-traces part3 all clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -U pip
	$(VENV)/bin/pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/

# --- Part 2: scheduler comparison -----------------------------------------

# Regenerate the workload. 7 req/s is ~1.14x measured capacity: the narrow band
# where all six metrics are non-zero. See docs/part2-findings.md for why.
trace:
	$(PYTHON) traces/gen_mixed.py --rate 7 --out traces/mixed.jsonl --seed 7

# The headline experiment: one trace, three policies, 60 simulated seconds.
part2: trace
	$(PYTHON) serve.py --trace traces/mixed.jsonl --policy all

# Sensitivity case at 8 req/s, where the server tips into congestion collapse.
part2-overload:
	$(PYTHON) traces/gen_mixed.py --rate 8 --out traces/mixed_overload.jsonl --seed 7
	$(PYTHON) serve.py --trace traces/mixed_overload.jsonl --policy all

# --- Part 3: routing ------------------------------------------------------

# The three router workloads, plus the scaled copies the fleet sweeps use.
# Same seed every time: the traces are inputs to a comparison, not a sample.
router-traces:
	$(PYTHON) traces/gen_router.py all --seed 7
	$(PYTHON) traces/gen_router.py t1 --rate 28 --out traces/t1_unique_prefix_4w.jsonl --seed 7
	$(PYTHON) traces/gen_router.py t1 --rate 56 --out traces/t1_unique_prefix_8w.jsonl --seed 7
	$(PYTHON) traces/gen_router.py t3 --rate 18 --out traces/t3_stale_8w.jsonl --seed 7

# Part 2 re-run for reproduction, then T1/T2/T3, into results/ and plots/.
part3:
	$(PYTHON) experiments.py

# Everything, from a clean checkout.
all: test part2 router-traces part3

clean:
	rm -rf results/*.json results/*.csv plots/*.png __pycache__ .pytest_cache
