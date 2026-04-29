from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vifood_eval.parsing import parse_answer_letter, parse_classifier_response


class ParsingTests(unittest.TestCase):
    def test_parse_answer_uses_final_answer_line(self) -> None:
        answer, status = parse_answer_letter("Maybe A.\nRationale...\nAnswer: C")
        self.assertEqual(answer, "C")
        self.assertEqual(status, "ok")

    def test_parse_answer_rejects_ambiguous_text(self) -> None:
        answer, status = parse_answer_letter("A or B could be right")
        self.assertIsNone(answer)
        self.assertEqual(status, "unparsed")

    def test_parse_answer_accepts_vietnamese_answer_labels(self) -> None:
        answer, status = parse_answer_letter("Lý do ngắn.\nĐáp án: **B. Măng**")
        self.assertEqual(answer, "B")
        self.assertEqual(status, "ok_localized")

    def test_parse_answer_accepts_final_markdown_letter_line(self) -> None:
        answer, status = parse_answer_letter("Lý do ngắn.\n**C**")
        self.assertEqual(answer, "C")
        self.assertEqual(status, "ok_final_line")

    def test_parse_classifier_json(self) -> None:
        parsed, status = parse_classifier_response(
            '{"qtype":"ingredients","food_items":["Pho Bo", "Com Tam"]}',
            {"ingredients"},
        )
        self.assertEqual(status, "ok")
        self.assertEqual(parsed["qtype"], "ingredients")
        self.assertEqual(parsed["food_items"], ["Pho Bo", "Com Tam"])

    def test_parse_classifier_rejects_unknown_qtype(self) -> None:
        parsed, status = parse_classifier_response(
            '{"qtype":"unknown","food_items":["Pho Bo"]}',
            {"ingredients"},
        )
        self.assertEqual(status, "invalid_qtype")
        self.assertIsNone(parsed["qtype"])


if __name__ == "__main__":
    unittest.main()
