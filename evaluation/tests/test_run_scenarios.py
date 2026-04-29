from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vifood_eval.data import VQASample, write_jsonl_row
from vifood_eval.report import main as report_main
from vifood_eval.run import _run_condition, _select_samples_by_id


def temp_dir() -> TemporaryDirectory[str]:
    root = Path(os.environ.get("VIFOOD_EVAL_TEST_TMP", "C:/tmp"))
    root.mkdir(parents=True, exist_ok=True)
    return TemporaryDirectory(dir=root)


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        messages: list[dict[str, object]],
        *,
        max_new_tokens: int,
        temperature: float,
        response_format: dict[str, object] | None = None,
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "response_format": response_format,
            }
        )
        if "Classify the ViFoodVQA sample" in _message_text(messages):
            return '{"qtype":"ingredients","food_items":["Pho Bo"]}'
        return "Short rationale.\nAnswer: B"


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str, dict[str, object]]] = []

    def retrieve(
        self,
        sample: VQASample,
        strategy: str,
        classifier: dict[str, object],
    ) -> list[dict[str, str]]:
        self.calls.append((sample.vqa_id, strategy, classifier))
        return [{"subject": "Pho Bo", "relation": "hasIngredient", "target": "Beef"}]


class FakeTqdm:
    instances: list["FakeTqdm"] = []

    def __init__(
        self,
        *,
        total: int,
        initial: int,
        desc: str,
        unit: str,
        disable: bool,
    ) -> None:
        self.total = total
        self.n = initial
        self.desc = desc
        self.unit = unit
        self.disable = disable
        self.postfixes: list[tuple[dict[str, object], bool]] = []
        self.updates: list[int] = []
        FakeTqdm.instances.append(self)

    def __enter__(self) -> "FakeTqdm":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def update(self, n: int = 1) -> None:
        self.n += n
        self.updates.append(n)

    def set_postfix(self, values: dict[str, object], *, refresh: bool) -> None:
        self.postfixes.append((values, refresh))


class RunScenarioTests(unittest.TestCase):
    def test_select_samples_by_id_preserves_requested_order(self) -> None:
        with temp_dir() as tmp:
            root = Path(tmp)
            samples = [_sample(root, 1), _sample(root, 2), _sample(root, 3)]

            selected = _select_samples_by_id(samples, [3, 1])

            self.assertEqual([sample.vqa_id for sample in selected], [3, 1])
            with self.assertRaises(KeyError):
                _select_samples_by_id(samples, [99])

    def test_no_kg_conditions_write_parseable_predictions_with_expected_shots(self) -> None:
        with temp_dir() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            sample = _sample(root, 1)
            shots = [_sample(root, 10), _sample(root, 11)]
            model = FakeModel()

            for shots_count in [0, 1, 2]:
                _run_condition(
                    cfg=_cfg(),
                    run_dir=run_dir,
                    model_name="fake",
                    model=model,
                    condition={
                        "name": f"no_kg_{shots_count}shot",
                        "knowledge": "none",
                        "shots": shots_count,
                    },
                    samples=[sample],
                    shots=shots,
                    classifier_cache={},
                    retriever=None,
                    resume=False,
                )

            for shots_count in [0, 1, 2]:
                row = _read_single_row(run_dir / "predictions" / f"fake__no_kg_{shots_count}shot.jsonl")
                self.assertEqual(row["shots"], shots_count)
                self.assertEqual(row["answer_pred"], "B")
                self.assertTrue(row["parse_status"].startswith("ok"))
                self.assertEqual(row["retrieved_triples"], [])

            answer_calls = [call for call in model.calls if call["response_format"] is None]
            self.assertEqual([_assistant_count(call["messages"]) for call in answer_calls], [0, 1, 2])
            self.assertTrue(all("(No external triples.)" in _message_text(call["messages"]) for call in answer_calls))

    def test_oracle_uses_gold_triples_without_retriever(self) -> None:
        with temp_dir() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            sample = _sample(root, 1)

            _run_condition(
                cfg=_cfg(),
                run_dir=run_dir,
                model_name="fake",
                model=FakeModel(),
                condition={"name": "oracle", "knowledge": "oracle", "shots": 0},
                samples=[sample],
                shots=[],
                classifier_cache={},
                retriever=None,
                resume=False,
            )

            row = _read_single_row(run_dir / "predictions" / "fake__oracle.jsonl")
            self.assertEqual(row["retrieved_triples"], sample.gold_triples)
            self.assertEqual(row["precision_at_10"], 1.0)
            self.assertEqual(row["recall_at_10"], 1.0)
            self.assertEqual(row["f1_at_10"], 1.0)

    def test_retrieved_conditions_reuse_classifier_cache_and_resume_cleanly(self) -> None:
        with temp_dir() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            samples = [_sample(root, 1), _sample(root, 2)]
            model = FakeModel()
            retriever = FakeRetriever()
            cache: dict[int, dict[str, object]] = {}
            conditions = [
                ("hybrid", "hybrid"),
                ("graph_only", "graph_only"),
                ("vector_only", "vector_only"),
                ("bm25", "bm25"),
            ]

            for condition_name, strategy in conditions:
                _run_condition(
                    cfg=_cfg(),
                    run_dir=run_dir,
                    model_name="fake",
                    model=model,
                    condition={
                        "name": condition_name,
                        "knowledge": "retrieved",
                        "retrieval_strategy": strategy,
                        "shots": 0,
                    },
                    samples=samples,
                    shots=[],
                    classifier_cache=cache,
                    retriever=retriever,
                    resume=False,
                )

            classifier_rows = _read_rows(run_dir / "classifier" / "fake.jsonl")
            self.assertEqual(len(classifier_rows), 2)
            self.assertTrue(all(row["status"] == "ok" for row in classifier_rows))
            classifier_calls = [call for call in model.calls if call["response_format"] is not None]
            self.assertEqual(len(classifier_calls), 2)
            self.assertTrue(
                all(call["response_format"] == {"type": "json_object"} for call in classifier_calls)
            )
            self.assertTrue(
                all("Do not answer the multiple-choice question" in _message_text(call["messages"]) for call in classifier_calls)
            )
            self.assertTrue(
                all('"qtype":"<one canonical qtype>"' in _message_text(call["messages"]) for call in classifier_calls)
            )

            self.assertEqual([call[1] for call in retriever.calls], [strategy for _, strategy in conditions for _ in samples])
            for condition_name, strategy in conditions:
                rows = _read_rows(run_dir / "predictions" / f"fake__{condition_name}.jsonl")
                self.assertEqual(len(rows), 2)
                self.assertTrue(all(row["retrieval_strategy"] == strategy for row in rows))
                self.assertTrue(all(row["qtype_pred"] == "ingredients" for row in rows))
                self.assertTrue(all(row["food_items_pred"] == ["Pho Bo"] for row in rows))
                self.assertTrue(all("precision_at_10" in row for row in rows))

            before_calls = len(model.calls)
            before_rows = len(_read_rows(run_dir / "predictions" / "fake__graph_only.jsonl"))
            _run_condition(
                cfg=_cfg(),
                run_dir=run_dir,
                model_name="fake",
                model=model,
                condition={
                    "name": "graph_only",
                    "knowledge": "retrieved",
                    "retrieval_strategy": "graph_only",
                    "shots": 0,
                },
                samples=samples,
                shots=[],
                classifier_cache=cache,
                retriever=retriever,
                resume=True,
            )
            self.assertEqual(len(model.calls), before_calls)
            self.assertEqual(
                len(_read_rows(run_dir / "predictions" / "fake__graph_only.jsonl")),
                before_rows,
            )

    def test_progress_tracks_condition_without_changing_predictions(self) -> None:
        with temp_dir() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            samples = [_sample(root, 1), _sample(root, 2)]
            model = FakeModel()
            condition = {"name": "no_kg_0shot", "knowledge": "none", "shots": 0}

            FakeTqdm.instances.clear()
            with patch("vifood_eval.run.tqdm", FakeTqdm):
                _run_condition(
                    cfg=_cfg(),
                    run_dir=run_dir,
                    model_name="fake",
                    model=model,
                    condition=condition,
                    samples=samples,
                    shots=[],
                    classifier_cache={},
                    retriever=None,
                    resume=False,
                    progress=True,
                    scenario_index=2,
                    scenario_total=8,
                )

                before_calls = len(model.calls)
                _run_condition(
                    cfg=_cfg(),
                    run_dir=run_dir,
                    model_name="fake",
                    model=model,
                    condition=condition,
                    samples=samples,
                    shots=[],
                    classifier_cache={},
                    retriever=None,
                    resume=True,
                    progress=True,
                    scenario_index=3,
                    scenario_total=8,
                )

            rows = _read_rows(run_dir / "predictions" / "fake__no_kg_0shot.jsonl")
            self.assertEqual(len(rows), 2)
            self.assertEqual([row["vqa_id"] for row in rows], [1, 2])
            self.assertTrue(all(row["answer_pred"] == "B" for row in rows))
            self.assertEqual(len(model.calls), before_calls)

            first_bar, resumed_bar = FakeTqdm.instances
            self.assertEqual(first_bar.total, 2)
            self.assertEqual(first_bar.n, 2)
            self.assertEqual(first_bar.desc, "[2/8] fake/no_kg_0shot")
            self.assertEqual(first_bar.unit, "q")
            self.assertFalse(first_bar.disable)
            self.assertEqual(first_bar.updates, [1, 1])
            self.assertEqual(first_bar.postfixes[-1][0]["done"], "2/2")
            self.assertEqual(first_bar.postfixes[-1][0]["out"], "fake__no_kg_0shot.jsonl")
            self.assertEqual(first_bar.postfixes[-1][0]["vqa_id"], 2)

            self.assertEqual(resumed_bar.total, 2)
            self.assertEqual(resumed_bar.n, 2)
            self.assertEqual(resumed_bar.desc, "[3/8] fake/no_kg_0shot")
            self.assertEqual(resumed_bar.updates, [])

    def test_progress_can_be_disabled_for_direct_runner_calls(self) -> None:
        with temp_dir() as tmp:
            root = Path(tmp)
            run_dir = root / "run"

            FakeTqdm.instances.clear()
            with patch("vifood_eval.run.tqdm", FakeTqdm):
                _run_condition(
                    cfg=_cfg(),
                    run_dir=run_dir,
                    model_name="fake",
                    model=FakeModel(),
                    condition={"name": "no_kg_0shot", "knowledge": "none", "shots": 0},
                    samples=[_sample(root, 1)],
                    shots=[],
                    classifier_cache={},
                    retriever=None,
                    resume=False,
                    progress=False,
                )

            self.assertTrue(FakeTqdm.instances[0].disable)

    def test_report_writes_summary_retrieval_and_error_review_files(self) -> None:
        with temp_dir() as tmp:
            run_dir = Path(tmp) / "run"
            prediction_dir = run_dir / "predictions"
            base_row = {
                "model": "fake",
                "condition": "no_kg_0shot",
                "vqa_id": 1,
                "image_id": "image1",
                "question": "Q?",
                "qtype_gold": "ingredients",
                "answer_gold": "B",
                "answer_pred": "B",
                "correct": True,
                "parse_status": "ok",
            }
            write_jsonl_row(prediction_dir / "fake__no_kg_0shot.jsonl", base_row)
            write_jsonl_row(
                prediction_dir / "fake__oracle.jsonl",
                {
                    **base_row,
                    "condition": "oracle",
                    "precision_at_10": 1.0,
                    "recall_at_10": 1.0,
                    "f1_at_10": 1.0,
                },
            )

            with patch.object(sys, "argv", ["report", "--run-dir", str(run_dir)]):
                report_main()

            self.assertTrue((run_dir / "metrics_overall.csv").exists())
            self.assertTrue((run_dir / "metrics_per_qtype.csv").exists())
            self.assertTrue((run_dir / "metrics_retrieval.csv").exists())
            self.assertTrue((run_dir / "metrics_summary.md").exists())
            self.assertTrue((run_dir / "human_error_subset.csv").exists())


def _cfg() -> dict[str, object]:
    return {
        "evaluation": {
            "top_k": 10,
            "temperature": 0,
            "max_new_tokens": 32,
        }
    }


def _sample(root: Path, vqa_id: int) -> VQASample:
    image_path = root / "images" / f"image{vqa_id}.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"fake image")
    return VQASample(
        row={
            "vqa_id": vqa_id,
            "image_id": f"image{vqa_id}",
            "image": f"images/image{vqa_id}.jpg",
            "qtype": "ingredients",
            "question": "Which ingredient is shown?",
            "choices": {"A": "Rice", "B": "Beef", "C": "Fish", "D": "Egg"},
            "answer": "B",
            "rationale": "The image shows beef.",
            "triples_used": [{"subject": "Pho Bo", "relation": "hasIngredient", "target": "Beef"}],
        },
        split="test",
        data_dir=root,
    )


def _read_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_single_row(path: Path) -> dict[str, object]:
    rows = _read_rows(path)
    assert len(rows) == 1
    return rows[0]


def _message_text(messages: object) -> str:
    result = []
    for message in messages:
        for part in message["content"]:
            if part["type"] == "text":
                result.append(str(part["text"]))
    return "\n".join(result)


def _assistant_count(messages: object) -> int:
    return sum(1 for message in messages if message["role"] == "assistant")


if __name__ == "__main__":
    unittest.main()
