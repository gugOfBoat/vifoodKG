# Full Evaluation Summary: All Models And Conditions

This document summarizes all available ViFoodVQA evaluation results for the
4-model by 8-condition matrix.

Snapshot date: 2026-05-08.

## Sources

| Source label | Meaning |
| --- | --- |
| `current eval outputs` | Metrics read from `ViFoodVQA/evaluation/outputs/*/metrics_overall.csv` and `metrics_retrieval.csv`. |
| `legacy midterm run` | Results provided from the midterm evaluation stage. These conditions were not rerun with the current evaluation codebase and do not have raw prediction files in `ViFoodVQA/evaluation/outputs`. |

Legacy midterm results used here:

| Model | Condition | Accuracy |
| --- | --- | ---: |
| Qwen3-VL-2B | `no_kg_0shot` | 46.84% |
| Qwen3-VL-2B | `oracle` | 84.92% |
| Phi-3.5-Vision-Instruct | `no_kg_0shot` | 27.00% |
| Phi-3.5-Vision-Instruct | `oracle` | 41.58% |

## Coverage Matrix

Counting the legacy midterm runs, the evaluation matrix is complete: 32/32
model-condition results are available.

| Model | Current output conditions | Legacy conditions | Total available |
| --- | ---: | ---: | ---: |
| GPT-5.2 | 8 | 0 | 8/8 |
| GPT-5.5 | 8 | 0 | 8/8 |
| Qwen3-VL-2B | 6 | 2 | 8/8 |
| Phi-3.5-Vision-Instruct | 6 | 2 | 8/8 |
| **All models** | **28** | **4** | **32/32** |

## Overall Accuracy

`Parse failure` and `QType classifier` are reported only when available from the
current evaluation outputs. Legacy midterm rows include only accuracy.

| Model | Condition | Test N | Accuracy | Parse failure | QType classifier | Source |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| GPT-5.2 | `no_kg_0shot` | 1,410 | 88.16% | 0.50% | n/a | current eval outputs |
| GPT-5.2 | `no_kg_1shot` | 1,410 | 88.58% | 0.35% | n/a | current eval outputs |
| GPT-5.2 | `no_kg_2shot` | 1,410 | 86.24% | 1.35% | n/a | current eval outputs |
| GPT-5.2 | `bm25` | 1,410 | 87.09% | 1.42% | 96.38% | current eval outputs |
| GPT-5.2 | `vector_only` | 1,410 | 87.09% | 1.42% | 96.38% | current eval outputs |
| GPT-5.2 | `graph_only` | 1,410 | 88.23% | 1.21% | 96.38% | current eval outputs |
| GPT-5.2 | `hybrid` | 1,410 | 88.65% | 1.06% | 96.38% | current eval outputs |
| GPT-5.2 | `oracle` | 1,410 | 95.18% | 0.64% | n/a | current eval outputs |
| GPT-5.5 | `no_kg_0shot` | 1,410 | 90.57% | 0.00% | n/a | current eval outputs |
| GPT-5.5 | `no_kg_1shot` | 1,410 | 90.78% | 0.00% | n/a | current eval outputs |
| GPT-5.5 | `no_kg_2shot` | 1,410 | 90.85% | 0.00% | n/a | current eval outputs |
| GPT-5.5 | `bm25` | 1,410 | 89.79% | 0.71% | 97.52% | current eval outputs |
| GPT-5.5 | `vector_only` | 1,410 | 89.72% | 0.00% | 97.52% | current eval outputs |
| GPT-5.5 | `graph_only` | 1,410 | 90.64% | 0.00% | 97.52% | current eval outputs |
| GPT-5.5 | `hybrid` | 1,410 | 90.71% | 0.00% | 97.52% | current eval outputs |
| GPT-5.5 | `oracle` | 1,410 | 95.46% | 0.92% | n/a | current eval outputs |
| Qwen3-VL-2B | `no_kg_0shot` | not recorded | 46.84% | n/a | n/a | legacy midterm run |
| Qwen3-VL-2B | `no_kg_1shot` | 1,410 | 54.26% | 0.92% | n/a | current eval outputs |
| Qwen3-VL-2B | `no_kg_2shot` | 1,410 | 52.98% | 0.64% | n/a | current eval outputs |
| Qwen3-VL-2B | `bm25` | 1,410 | 54.75% | 4.40% | 68.15% | current eval outputs |
| Qwen3-VL-2B | `vector_only` | 1,410 | 55.67% | 4.68% | 68.15% | current eval outputs |
| Qwen3-VL-2B | `graph_only` | 1,410 | 56.88% | 2.55% | 68.15% | current eval outputs |
| Qwen3-VL-2B | `hybrid` | 1,410 | 54.18% | 1.99% | 68.96% | current eval outputs |
| Qwen3-VL-2B | `oracle` | not recorded | 84.92% | n/a | n/a | legacy midterm run |
| Phi-3.5-Vision-Instruct | `no_kg_0shot` | not recorded | 27.00% | n/a | n/a | legacy midterm run |
| Phi-3.5-Vision-Instruct | `no_kg_1shot` | 1,410 | 24.82% | 16.81% | n/a | current eval outputs |
| Phi-3.5-Vision-Instruct | `no_kg_2shot` | 1,410 | 27.38% | 8.72% | n/a | current eval outputs |
| Phi-3.5-Vision-Instruct | `bm25` | 1,410 | 33.33% | 0.28% | 28.97% | current eval outputs |
| Phi-3.5-Vision-Instruct | `vector_only` | 1,410 | 38.44% | 0.85% | 28.97% | current eval outputs |
| Phi-3.5-Vision-Instruct | `graph_only` | 1,410 | 35.67% | 1.13% | 28.97% | current eval outputs |
| Phi-3.5-Vision-Instruct | `hybrid` | 1,410 | 36.10% | 1.13% | 28.97% | current eval outputs |
| Phi-3.5-Vision-Instruct | `oracle` | not recorded | 41.58% | n/a | n/a | legacy midterm run |

## Retrieval Metrics

Retrieval metrics are available only for current output rows with
`metrics_retrieval.csv`. No retrieval metrics are reported for legacy-only rows.

| Model | Condition | Test N | P@10 | R@10 | F1@10 | Source |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| GPT-5.2 | `bm25` | 1,410 | 0.0147 | 0.1331 | 0.0262 | current eval outputs |
| GPT-5.2 | `vector_only` | 1,410 | 0.0289 | 0.2357 | 0.0509 | current eval outputs |
| GPT-5.2 | `graph_only` | 1,410 | 0.3310 | 0.4064 | 0.3547 | current eval outputs |
| GPT-5.2 | `hybrid` | 1,410 | 0.3311 | 0.4067 | 0.3548 | current eval outputs |
| GPT-5.2 | `oracle` | 1,410 | 1.0000 | 1.0000 | 1.0000 | current eval outputs |
| GPT-5.5 | `bm25` | 1,410 | 0.0160 | 0.1483 | 0.0288 | current eval outputs |
| GPT-5.5 | `vector_only` | 1,410 | 0.0297 | 0.2453 | 0.0525 | current eval outputs |
| GPT-5.5 | `graph_only` | 1,410 | 0.3602 | 0.4872 | 0.3973 | current eval outputs |
| GPT-5.5 | `hybrid` | 1,410 | 0.3608 | 0.4904 | 0.3984 | current eval outputs |
| GPT-5.5 | `oracle` | 1,410 | 1.0000 | 1.0000 | 1.0000 | current eval outputs |
| Qwen3-VL-2B | `bm25` | 1,410 | 0.0056 | 0.0491 | 0.0100 | current eval outputs |
| Qwen3-VL-2B | `vector_only` | 1,410 | 0.0085 | 0.0681 | 0.0150 | current eval outputs |
| Qwen3-VL-2B | `graph_only` | 1,410 | 0.0322 | 0.0394 | 0.0343 | current eval outputs |
| Qwen3-VL-2B | `hybrid` | 1,410 | 0.0268 | 0.0323 | 0.0285 | current eval outputs |
| Phi-3.5-Vision-Instruct | `bm25` | 1,410 | 0.0031 | 0.0272 | 0.0054 | current eval outputs |
| Phi-3.5-Vision-Instruct | `vector_only` | 1,410 | 0.0060 | 0.0493 | 0.0106 | current eval outputs |
| Phi-3.5-Vision-Instruct | `graph_only` | 1,410 | 0.0152 | 0.0181 | 0.0159 | current eval outputs |
| Phi-3.5-Vision-Instruct | `hybrid` | 1,410 | 0.0152 | 0.0181 | 0.0159 | current eval outputs |

## Key Findings

- GPT-5.5 has the strongest overall current-run performance. Its best
  non-oracle condition is `no_kg_2shot` at 90.85%, while `hybrid` is very close
  at 90.71%.
- Oracle is the strongest condition for every model with available oracle
  results. The best overall result is GPT-5.5 `oracle` at 95.46%.
- For GPT models, graph-based retrieval is much stronger than vector-only and
  BM25 by retrieval metrics. GPT-5.5 `hybrid` reaches the best non-oracle
  retrieval score with F1@10 = 0.3984.
- Open-weight baselines are substantially weaker than GPT-5.2/GPT-5.5 on this
  benchmark. Qwen3-VL-2B current non-oracle runs are in the low-to-mid 50%
  range, while Phi-3.5-Vision-Instruct current runs are mostly in the 20-30%
  range.
- Legacy oracle runs still show large gains over legacy no-KG 0-shot for the
  open-weight models: Qwen3-VL-2B improves from 46.84% to 84.92%, and
  Phi-3.5-Vision-Instruct improves from 27.00% to 41.58%.

## Notes And Limitations

- Qwen outputs use both `qwen3_vl_2b` and `qwen3_vl_2b_4bit` model identifiers
  across current output folders. This summary reports them under the single
  display label `Qwen3-VL-2B`.
- Legacy midterm rows are included to complete the 32-condition matrix, but
  their raw prediction files, parse failure rates, qtype classifier accuracies,
  and retrieval metrics are not present in the current output directory.
- Current no-KG conditions do not have qtype classifier metrics because query
  planning is only used by KG retrieval conditions.
