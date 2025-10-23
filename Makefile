.PHONY: setup cpu pip test clean

setup:
	conda env create -f environment.yml || true
	@echo "Activate with: conda activate mol-gnn"

pip:
	pip install -r requirements.txt

test:
	pytest -q

clean:
	rm -rf __pycache__ .pytest_cache dist build
