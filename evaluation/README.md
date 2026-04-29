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

The default `hf` extra follows the Qwen/latest stack and requires a recent
Transformers build with Qwen3-VL support.

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

Run Hugging Face vision smoke tests in model-specific environments when model
weights need to be pulled. The old combined command is only suitable when the
runtime package stack supports both models.

```bash
python -m vifood_eval.run --config configs/eval.yaml --models qwen3_vl_2b phi3_vision --conditions no_kg_0shot oracle --sample-ids 191 192 --run-id smoke_kaggle_hf
python -m vifood_eval.run --config configs/eval.yaml --models qwen3_vl_2b phi3_vision --conditions hybrid graph_only vector_only bm25 --sample-ids 191 192 --run-id smoke_kaggle_hf_kg
```

## HF Model-Specific Environments

Run Qwen3-VL and Phi-3.5 Vision in separate Colab runtimes. Qwen needs a newer
Transformers stack, while Phi is faster and more stable with the older stack
recommended by its model card. Restart the runtime after switching stacks or
after any CUDA device-side assert.

Qwen runtime:

```bash
cd /content/drive/Shareddrives/Intro2SLA/code/inference_code/hoang/ViFoodVQA/evaluation
pip install -e ".[dev,api,hf-qwen]"
python -m vifood_eval.run --config configs/eval_qwen.yaml --models qwen3_vl_2b --conditions no_kg_0shot oracle --sample-ids 191 192 --run-id smoke_qwen_env
python -m vifood_eval.report --run-dir outputs/smoke_qwen_env
```

Phi runtime:

```bash
cd /content/drive/Shareddrives/Intro2SLA/code/inference_code/hoang/ViFoodVQA/evaluation
pip uninstall -y transformers torch torchvision torchaudio accelerate
pip install -e ".[dev,api,hf-phi]"
python -c "from vifood_eval.config import load_config; print(load_config('configs/eval_phi.yaml')['models'].keys())"
python -m vifood_eval.run --config configs/eval_phi.yaml --models phi3_vision --conditions no_kg_0shot oracle --sample-ids 191 192 --run-id smoke_phi_env
python -m vifood_eval.report --run-dir outputs/smoke_phi_env
```

Use `configs/eval_qwen.yaml` for Qwen-only runs and `configs/eval_phi.yaml` for
Phi-only runs. Do not run both models in the same runtime after package changes.
The Phi extra does not pin NumPy so it can coexist with Colab packages that
require NumPy 2.x. If the sanity check does not print `dict_keys(['phi3_vision'])`,
pull the latest repo files and make sure the run command uses
`--config configs/eval_phi.yaml`.

Run the full matrix:

```bash
python -m vifood_eval.run --config configs/eval.yaml --resume
python -m vifood_eval.report --run-dir outputs/<run_id>
```

## HF Smoke Diagnostics

The early Colab smoke outputs under `outputs/smoke_qwen_basic` and
`outputs/smoke_phi_cache_on` are useful failure examples, not final model
quality claims.

- `smoke_qwen_basic`: no-KG 0-shot hallucinated the top-left dish as bread-like
  food and failed to produce a parseable final answer, while oracle KG answered
  both smoke samples correctly. Treat this as a no-KG visual grounding failure
  that KG can mitigate.
- `smoke_phi_cache_on`: both no-KG and oracle responses degenerated into repeated
  `Inc` tokens. Treat this as an HF generation/adapter failure. The default
  shared config keeps Phi conservative with `use_cache: false`; the Phi-specific
  environment uses the older model-card stack and `use_cache: true` for speed.
- If PowerShell shows text such as `XÃ©t`, reread files with
  `Get-Content -Encoding UTF8`; the JSONL files are UTF-8.

## Output Contract

Predictions are written as JSONL files under `outputs/<run_id>/predictions/`.
Each row includes model, condition, gold/predicted answer, parse status,
classifier output, retrieved triples, retrieval metrics, raw response, and
latency.

The report command writes aggregate CSV and Markdown summaries from raw outputs.
