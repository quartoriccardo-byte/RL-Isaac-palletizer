.PHONY: install test train_smoke clean

install:
	pip install -e .

test:
	pytest tests/ -v

train_smoke:
	@echo "Smoke test requires Isaac Lab environment"
	@echo "Run: python scripts/train.py --help"

clean:
	rm -rf __pycache__ *.egg-info .pytest_cache
	find . -name "*.pyc" -delete
