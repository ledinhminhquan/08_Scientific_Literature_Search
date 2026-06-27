.PHONY: help install install-all data pairs train evaluate demo serve ui test grade report slides autopilot clean

help:
	@echo "Exploratory Scientific Literature Search — common tasks"
	@echo "  make install       core install (pip install -e .)"
	@echo "  make install-all   full install (.[all])"
	@echo "  make data          load corpus + build (query,paper) pairs"
	@echo "  make pairs         show generated training pairs"
	@echo "  make train         fine-tune the dense retriever (MNRL)"
	@echo "  make evaluate      retriever vs BM25/zero-shot (Recall/MRR/nDCG)"
	@echo "  make demo          run the agent on sample queries"
	@echo "  make serve / ui    FastAPI server / + Gradio UI at /ui"
	@echo "  make test report slides autopilot grade"

install:
	pip install -e .

install-all:
	pip install -e ".[all]"

data:
	scisearch data

pairs:
	scisearch pairs

train:
	scisearch --config configs/train.yaml train

evaluate:
	scisearch evaluate

demo:
	scisearch demo-agent --tfidf

serve:
	scisearch --config configs/infer.yaml serve --host 0.0.0.0 --port 8000

ui:
	scisearch serve --ui --host 0.0.0.0 --port 7860

test:
	pytest -q

grade:
	scisearch grade

report:
	scisearch generate-report

slides:
	scisearch generate-slides

autopilot:
	scisearch autopilot --no-train

clean:
	rm -rf artifacts __pycache__ .pytest_cache src/*.egg-info build dist
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
