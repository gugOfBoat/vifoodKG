"""
Step 1B - Normalize & Classify food entities using Gemini LLM.

Reads:  data/raw_unique_labels.json
Writes: data/master_entities.json

Usage:  python src/02_kg_entity_classifier.py
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

INPUT_FILE = PROJECT_ROOT / "data" / "raw_unique_labels.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "master_entities.json"
PROGRESS_FILE = PROJECT_ROOT / "data" / "_classifier_progress.json"

GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
BATCH_SIZE = 50

SYSTEM_PROMPT = """Bạn là chuyên gia ẩm thực Việt Nam kiêm kỹ sư tri thức cho dự án ViFoodKG.

NHIỆM VỤ:
Nhận danh sách nhãn thức ăn thô (raw labels), chuẩn hóa và phân loại từng nhãn.

== A. CHUẨN HÓA (canonical_label) ==
1. Unicode NFC, Title Case (trừ giới từ: và, với, của, cho).
2. Gộp đồng nghĩa về 1 tên chuẩn:
   Thịt heo = Thịt lợn -> "Thịt Lợn"
   Nước lèo = Nước dùng -> "Nước Dùng"
   Bún tươi = Bún -> "Bún"
   Giá = Giá đỗ = Giá sống -> "Giá Đỗ"
   Đậu phụ = Đậu hũ = Tàu hũ -> "Đậu Hũ"
   Đồ chua = Dưa góp -> "Đồ Chua"
   Không dấu -> có dấu: "Banh mi" -> "Bánh Mì", "Pho" -> "Phở"
3. Sửa lỗi chính tả/OCR:
   "Nuoc mam" -> "Nước Mắm"
4. Tự suy luận thêm các cặp synonym phổ biến khác.

== B. PHÂN LOẠI (category) ==
Gán ĐÚNG MỘT trong 5 giá trị:

MainDish - Món hoàn chỉnh, gọi riêng tại quán.
VD: Phở, Bún Chả, Cơm Tấm, Cá Kho Tộ, Canh Chua Cá Lóc

Component/Ingredient - Nguyên liệu riêng lẻ nhìn thấy trong ảnh.
VD: Bún (sợi), Thịt Bò, Tôm, Giá Đỗ, Cơm Trắng, Hành Lá

Condiment - Nước chấm, gia vị, sốt.
VD: Nước Mắm, Mắm Tôm, Tương Ớt, Tiêu, Nước Chấm

SideDish - Món ăn kèm, rau sống, đồ phụ.
VD: Rau Sống, Quẩy, Đồ Chua, Chả Lụa, Pate

Discard - LOẠI BỎ:
vật dụng (đũa, bát, đĩa, khăn giấy),
nhãn chung chung (thức ăn, món ăn, đồ ăn, thực phẩm, nguyên liệu, gia vị chung),
tính từ (ngon, nóng, tươi),
nhãn vô nghĩa,
đồ uống không phải thức ăn (7 Up, bia, nước ngọt, trà đá),
nhãn tiếng Anh/ngoại quốc KHÔNG phải món Việt.

LƯU Ý:
- Nhãn tiếng Anh của món Việt -> chuẩn hóa về tiếng Việt:
  "Pho" -> "Phở"
  "Banh Mi" -> "Bánh Mì"
- Món nước ngoài (Sushi, Pizza, Tacos...) -> Discard (chỉ giữ món Việt Nam)
- "Cơm" đơn thuần -> Component/Ingredient
- "Bún" đơn thuần -> Component/Ingredient

== OUTPUT ==
Trả về JSON array.
Mỗi phần tử:

{
  "raw_label": "...",
  "canonical_label": "...",
  "category": "..."
}

CHỈ JSON array, KHÔNG giải thích, KHÔNG markdown.
"""


def call_gemini(labels, retries=3):
    from google import genai
    
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY missing")
        sys.exit(1)
    client = genai.Client(api_key=GEMINI_API_KEY)

    numbered = "\n".join(f"{i+1}. {lbl}" for i, lbl in enumerate(labels))
    user_msg = f"Xu ly {len(labels)} nhan sau:\n\n{numbered}"

    for attempt in range(1, retries + 1):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=SYSTEM_PROMPT + "\n\n" + user_msg,
                config=genai.types.GenerateContentConfig(
                    temperature=0.05,
                    max_output_tokens=8192,
                    response_mime_type="application/json",
                ),
            )
            raw = resp.text.strip()
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "entities" in parsed:
                return parsed["entities"]
            if isinstance(parsed, list):
                return parsed
            print(f"  attempt {attempt}: unexpected JSON shape")
        except json.JSONDecodeError as e:
            print(f"  attempt {attempt}: JSON error: {e}")
        except Exception as e:
            print(f"  attempt {attempt}: {type(e).__name__}: {e}")
        if attempt < retries:
            time.sleep(2 ** attempt)

    print("  FAILED after all retries")
    return []


def load_progress():
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    return {"completed_batches": 0, "results": []}


def save_progress(progress):
    PROGRESS_FILE.write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    if not INPUT_FILE.exists():
        print(f"Input not found: {INPUT_FILE}")
        print("Run 01_kg_entity_extractor.py first.")
        sys.exit(1)

    raw_labels = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    print(f"Loaded {len(raw_labels)} raw labels")

    batches = [raw_labels[i:i + BATCH_SIZE] for i in range(0, len(raw_labels), BATCH_SIZE)]
    total_batches = len(batches)
    print(f"Batches: {total_batches} x {BATCH_SIZE} | Model: {GEMINI_MODEL}\n")

    progress = load_progress()
    start_batch = progress["completed_batches"]
    all_results = progress["results"]

    if start_batch > 0:
        print(f"Resuming from batch {start_batch + 1}/{total_batches}")

    for idx in range(start_batch, total_batches):
        batch = batches[idx]
        print(f"[{idx + 1}/{total_batches}] {len(batch)} labels...", end=" ", flush=True)

        results = call_gemini(batch)
        all_results.extend(results)

        kept = sum(1 for r in results if r.get("category") != "Discard")
        discarded = len(results) - kept
        print(f"-> {kept} kept, {discarded} discarded")

        progress["completed_batches"] = idx + 1
        progress["results"] = all_results
        save_progress(progress)

        if idx + 1 < total_batches:
            time.sleep(1)

    # Post-process
    print(f"\n-- Post-processing --")
    total_discarded = sum(1 for e in all_results if e.get("category") == "Discard")
    print(f"Discard removed: {total_discarded}")

    kept = [e for e in all_results if e.get("category") != "Discard"]

    seen = {}
    for e in kept:
        key = e.get("canonical_label", "").strip().lower()
        if not key:
            continue
        if key not in seen:
            seen[key] = {
                "canonical_label": e["canonical_label"].strip(),
                "category": e["category"],
                "raw_labels": [e.get("raw_label", "")],
            }
        else:
            raw = e.get("raw_label", "")
            if raw and raw not in seen[key]["raw_labels"]:
                seen[key]["raw_labels"].append(raw)

    order = {"MainDish": 0, "Component/Ingredient": 1, "SideDish": 2, "Condiment": 3}
    master = sorted(
        seen.values(),
        key=lambda x: (order.get(x["category"], 9), x["canonical_label"]),
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(master, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()

    from collections import Counter
    cats = Counter(e["category"] for e in master)
    print(f"\n-- Results --")
    print(f"  MainDish:             {cats.get('MainDish', 0)}")
    print(f"  Component/Ingredient: {cats.get('Component/Ingredient', 0)}")
    print(f"  SideDish:             {cats.get('SideDish', 0)}")
    print(f"  Condiment:            {cats.get('Condiment', 0)}")
    print(f"  TOTAL:                {len(master)}")
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
