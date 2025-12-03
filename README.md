# Molecular GNN (ESOL + QM9 subset)

Lightweight, reproducible project for **molecular property prediction** using **GIN/MPNN** on **ESOL** (and an optional QM9 subset). 
This scaffold is aligned with our implementation checklist — **Phase 0 completed (no pre-commit)**.

## Quickstart

```bash
# Create & activate env (Conda preferred)
conda env create -f environment.yml
conda activate mol-gnn

# OR with pip (CPU)
pip install -r requirements.txt
```

## Run a smoke test

```bash
pytest -q
```

## Project structure (initial)

```
mol-gnn/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ pyproject.toml
├─ requirements.txt
├─ environment.yml
├─ Makefile
├─ src/
│  └─ __init__.py
├─ tests/
│  └─ test_smoke.py
└─ .github/workflows/ci.yml  (optional CI)
```

Next phases: add data modules, baselines, models, training loop, and experiments.
