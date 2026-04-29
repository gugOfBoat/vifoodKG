from __future__ import annotations

from pathlib import Path
from typing import Any

from .data import VQASample
from .triples import verbalize_triples

CANONICAL_QTYPES = [
    "ingredients",
    "ingredient_category",
    "cooking_technique",
    "food_pairings",
    "dietary_restrictions",
    "allergen_restrictions",
    "dish_classification",
    "flavor_profile",
    "origin_locality",
    "substitution_rules",
]


def build_answer_messages(
    sample: VQASample,
    shots: list[VQASample],
    knowledge_triples: list[dict[str, Any]] | None,
    top_k: int,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are evaluating Vietnamese multiple-choice visual question "
                        "answering. Use the image, question, choices, and any provided "
                        "knowledge triples. Give a brief Vietnamese rationale, then end "
                        "with exactly one line: Answer: A, Answer: B, Answer: C, or Answer: D."
                    ),
                }
            ],
        }
    ]

    for shot in shots:
        messages.append(_answer_user_message(shot, [], top_k))
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"{shot.row.get('rationale', '')}\nAnswer: {shot.row['answer']}",
                    }
                ],
            }
        )

    messages.append(_answer_user_message(sample, knowledge_triples or [], top_k))
    return messages


def build_classifier_messages(sample: VQASample) -> list[dict[str, Any]]:
    qtypes = ", ".join(CANONICAL_QTYPES)
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Classify the ViFoodVQA sample before KG retrieval. Return only one "
                        "valid JSON object, with no Markdown and no explanation. The JSON keys "
                        f"must be qtype and food_items. qtype must be one of: {qtypes}. "
                        "food_items must list visible Vietnamese dish names that can anchor a food KG."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "path": sample.image_path},
                {"type": "text", "text": f"Question:\n{sample.row['question']}"},
            ],
        },
    ]


def _answer_user_message(
    sample: VQASample,
    knowledge_triples: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    text = (
        f"Question:\n{sample.row['question']}\n\n"
        f"Choices:\n{_format_choices(sample.row['choices'])}\n\n"
        f"Knowledge triples:\n{verbalize_triples(knowledge_triples, limit=top_k)}"
    )
    return {
        "role": "user",
        "content": [
            {"type": "image", "path": sample.image_path},
            {"type": "text", "text": text},
        ],
    }


def _format_choices(choices: dict[str, str]) -> str:
    return "\n".join(f"{letter}. {choices[letter]}" for letter in ["A", "B", "C", "D"])
