# ViFoodVQA Evaluation

Evaluation scaffold for the ViFoodVQA benchmark. It runs the agreed protocol:

- No-KG prompting with 0-shot, 1-shot, and 2-shot.
- KG-augmented 0-shot with Hybrid, Graph-only, Vector-only, BM25, and Oracle.
- Retrieval defaults to top-10 because ViFoodVQA images are multi-dish meal trays.
- Each model predicts question type and visible food items before KG retrieval.

## Setup

```bash
cd ViFoodVQA/evaluation
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev,hf,api]
```

Set secrets through environment variables or `.env`:

```bash
OPENAI_COMPAT_API_KEY=...
OPENAI_COMPAT_BASE_URL=https://api.openai.com/v1
NEO4J_URI=...
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...
```

## Commands

Prepare the dataset under `evaluation/data/vifoodvqa` if it is missing. The
script accepts the current Hugging Face Parquet export and materializes compact
JSONL split files used by the runner:

```bash
python -m vifood_eval.prepare_data --config configs/eval.yaml
```

Run a deterministic local smoke test for API-backed models:

```bash
python -m vifood_eval.run --config configs/eval.yaml --models gpt_5_2 --conditions no_kg_0shot no_kg_1shot no_kg_2shot oracle --sample-ids 191 192 --run-id smoke_local_api
```

Run KG retrieval locally only when Neo4j credentials are configured and the E5
embedding model is available locally:

```bash
python -m vifood_eval.run --config configs/eval.yaml --models gpt_5_2 --conditions hybrid graph_only vector_only bm25 --sample-ids 191 192 --run-id smoke_kg
```

Run Hugging Face vision model smoke tests on Kaggle or another GPU environment
when model weights need to be pulled:

```bash
python -m vifood_eval.run --config configs/eval.yaml --models qwen3_vl_2b phi3_vision --conditions no_kg_0shot oracle --sample-ids 191 192 --run-id smoke_kaggle_hf
python -m vifood_eval.run --config configs/eval.yaml --models qwen3_vl_2b phi3_vision --conditions hybrid graph_only vector_only bm25 --sample-ids 191 192 --run-id smoke_kaggle_hf_kg
```

Run the full matrix:

```bash
python -m vifood_eval.run --config configs/eval.yaml --resume
python -m vifood_eval.report --run-dir outputs/<run_id>
```

## Output Contract

Predictions are written as JSONL files under `outputs/<run_id>/predictions/`.
Each row includes model, condition, gold/predicted answer, parse status,
classifier output, retrieved triples, retrieval metrics, raw response, and
latency.

The report command writes aggregate CSV and Markdown summaries from raw outputs.
