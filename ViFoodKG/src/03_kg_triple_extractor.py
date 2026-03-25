"""
Step 2 - Web-Grounded Triple Extraction
========================================

For each MainDish in master_entities.json:
  1. Crawl Vietnamese Wikipedia + food blogs for factual content
  2. Feed crawled text + ontology schema to Gemini
  3. Gemini extracts all triples with source_url + evidence

Batches 5 MainDishes per LLM call. Checkpoint-resumable.

Usage:
  python src/03_kg_triple_extractor.py --limit 3    # test with 3 dishes
  python src/03_kg_triple_extractor.py               # full run all MainDishes
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

MASTER_FILE = PROJECT_ROOT / "data" / "master_entities.json"
ONTOLOGY_FILE = PROJECT_ROOT / "config" / "ontology_config.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "triples"
TRIPLES_FILE = OUTPUT_DIR / "extracted_triples.json"
PROGRESS_FILE = OUTPUT_DIR / "_progress.json"

GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
BATCH_SIZE = 5

# ══════════════════════════════════════════════════════════════════════════
# WEB CRAWL — fetch factual content about a dish
# ══════════════════════════════════════════════════════════════════════════

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


def crawl_wikipedia_vi(dish_name):
    """Search Vietnamese Wikipedia for a dish. Returns (text, url) or (None, None)."""
    search_url = "https://vi.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": dish_name,
        "format": "json",
        "srlimit": 1,
    }
    try:
        resp = requests.get(search_url, params=params, headers=HEADERS, timeout=10)
        data = resp.json()
        results = data.get("query", {}).get("search", [])
        if not results:
            return None, None

        title = results[0]["title"]
        page_url = f"https://vi.wikipedia.org/wiki/{urllib.parse.quote(title)}"

        # Fetch page content
        content_params = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
            "exsectionformat": "plain",
            "format": "json",
        }
        resp2 = requests.get(search_url, params=content_params, headers=HEADERS, timeout=10)
        pages = resp2.json().get("query", {}).get("pages", {})
        for page in pages.values():
            text = page.get("extract", "")
            if text and len(text) > 100:
                return text[:10000], page_url
        return None, None
    except Exception:
        return None, None


def crawl_for_dish(dish_name):
    """Crawl web content for a dish. Returns list of {text, source_url, source}."""
    sources = []

    # Layer 1: Wikipedia VN exact match
    text, url = crawl_wikipedia_vi(dish_name)
    if text:
        sources.append({"text": text, "source_url": url, "source": "Wikipedia VI"})
        return sources

    # Layer 2: LLM tự sinh triples từ kiến thức nội tại
    sources.append({
        "text": (
            f"Không tìm được bài viết Wikipedia nào cho món '{dish_name}'. "
            f"Bạn PHẢI sử dụng 100% Kiến thức Chuyên gia (Cognitive Reasoning) để điền vào TẤT CẢ 10 quan hệ trong Ontology cho món này."
        ),
        "source_url": "None",
        "source": "None"
    })

    return sources


# ══════════════════════════════════════════════════════════════════════════
# LLM EXTRACTION — Gemini extracts triples from crawled content
# ══════════════════════════════════════════════════════════════════════════

EXTRACTION_PROMPT = """
Bạn là Kỹ sư Tri thức (Knowledge Engineer) & Chuyên gia Ẩm thực cấp cao cho dự án ViFoodKG.

NHIỆM VỤ: 
Trích xuất một Đồ thị Tri thức (Knowledge Graph) TOÀN DIỆN cho mỗi món ăn, tuân thủ nghiêm ngặt Lược đồ đồ thị (Ontology) gồm 10 loại quan hệ dưới đây.
Để hoàn thành nhiệm vụ, bạn phải sử dụng chiến lược "Hybrid Extraction":
1. Bám sát Văn bản Nguồn (Web-Grounded): Trích xuất tối đa thông tin từ văn bản (được crawl từ Wikipedia) cung cấp bên dưới.
2. Suy luận Lấp đầy (Reasoning & Enrichment): Đối với các quan hệ nằm trong Ontology nhưng BỊ THIẾU trong văn bản web (ví dụ: hasAllergen, flavorProfile, originRegion, servedWith), HÃY SỬ DỤNG KIẾN THỨC CHUYÊN GIA ẨM THỰC của bạn để tự suy luận và điền vào nhằm lấp đầy đồ thị. Tuyệt đối không để trống các quan hệ quan trọng nếu bạn biết câu trả lời.

== 1. LƯỢC ĐỒ QUAN HỆ (ONTOLOGY BẮT BUỘC) ==
- hasIngredient: Dish -> Ingredient (Thành phần chính)
- servedWith: Dish -> SideDish/Condiment (Rau thơm, chanh ớt, nước mắm...)
- originRegion: Dish -> Region (Miền Bắc, Miền Nam, Huế...)
- dishType: Dish -> DishType (Món nước, nướng, lẩu, xào, gỏi...)
- cookingTechnique: Dish -> CookingTechnique (Ninh, chiên, nướng...)
- flavorProfile: Dish -> FlavorProfile (Ngọt thanh, chua cay, đậm đà...)
- ingredientCategory: Ingredient -> IngredientCategory (Thịt đỏ, hải sản...)
- hasAllergen: Ingredient -> Allergen (GIÁP XÁC, ĐẬU PHỘNG, GLUTEN, ĐẬU NÀNH... Rất quan trọng! Nếu món có tôm/mắm tôm => Giáp xác. Nếu có nước tương => Đậu nành)
- hasDietaryTag: Ingredient -> DietaryTag ("Chay" hoặc "Mặn")
- hasSubRule: Dish -> SubstitutionRule (Có quy tắc thay thế nguyên liệu không?)
- fromIngredient / toIngredient: SubstitutionRule -> Ingredient

== 2. QUY TẮC "NGUỒN & BẰNG CHỨNG" (SOURCE & EVIDENCE) ==
Mỗi Triple bắt buộc phải có `source_url` và `evidence`:
- NẾU thông tin lấy từ văn bản Wikipedia cung cấp: 
   + `source_url` = URL của bài viết đó (đã cung cấp bên dưới dấu [Nguon: ...])
   + `evidence` = Trích dẫn y nguyên câu văn gốc trong bài làm căn cứ.
- NẾU thông tin do bạn tự suy luận bằng kiến thức chuyên gia (Common Sense):
   + `source_url` = Nếu bạn BIẾT CHẮC 1 URL uy tín có thông tin này, hãy ghi URL đó. Nếu không, ghi "Cognitive_Reasoning" hoặc "LLM_Knowledge".
   + `evidence` = Viết 1 câu NGẮN GỌN giải thích logic (VD: "Mắm tôm làm từ tôm tép lên men, là động vật giáp xác").

== 3. CHUẨN HÓA NHÃN CỦA NODE ==
Tên các thực thể (canonical_label) phải viết bằng tiếng Việt có dấu, Title Case (Ví dụ: "Hành Lá", "Thịt Bò").

== 4. ĐỊNH DẠNG ĐẦU RA (JSON ONLY) ==
Trích xuất tối đa có thể cho 10 quan hệ của MỖI món ăn. Trả về một JSON Array duy nhất:
[
  {
    "dish": "Tên Món Ăn",
    "triples": [
      {
        "subject": "Tên Thực Thể Đầu",
        "relation": "hasAllergen",
        "target": "Giáp Xác",
        "target_type": "Allergen",
        "source_url": "Cognitive_Reasoning",
        "evidence": "Nước mắm làm từ cá (một loại hải sản) thường gây dị ứng."
      }
    ]
  }
]
Chỉ trả về JSON, không giải thích thêm.
"""

def call_gemini_extract(dishes_with_content, retries=3):
    """Send batch of dishes with crawled content to Gemini for triple extraction."""
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY missing")
        sys.exit(1)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    # Build user message with crawled content per dish
    parts = []
    for dish_name, sources in dishes_with_content:
        parts.append(f"\n=== MON AN: {dish_name} ===")
        for src in sources:
            parts.append(f"[Nguon: {src['source_url']}]")
            parts.append(src["text"][:10000])
        if not sources:
            parts.append("(Khong co noi dung crawl duoc)")

    user_msg = f"Trich xuat triples cho {len(dishes_with_content)} mon an:\n" + "\n".join(parts)

    for attempt in range(1, retries + 1):
        try:
            resp = model.generate_content(
                EXTRACTION_PROMPT + "\n\n" + user_msg,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3, # Allow cognitive reasoning natively
                    max_output_tokens=16384,
                    response_mime_type="application/json",
                ),
            )
            raw = resp.text.strip()
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict) and "dishes" in parsed:
                return parsed["dishes"]
            print(f"  attempt {attempt}: unexpected shape")
        except json.JSONDecodeError as e:
            print(f"  attempt {attempt}: JSON error: {e}")
        except Exception as e:
            print(f"  attempt {attempt}: {type(e).__name__}: {e}")
        if attempt < retries:
            time.sleep(2 ** attempt)

    print("  FAILED after retries")
    return []


# ══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def load_progress():
    if PROGRESS_FILE.exists():
        data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            # If it has "last_index", use it. Otherwise guess based on completed_batches
            last_idx = data.get("last_index", data.get("completed_batches", 0) * BATCH_SIZE)
            return last_idx, data.get("all_triples", [])
        elif isinstance(data, list):
            return 0, data
    return 0, []


def save_progress(last_index, all_triples):
    data = {"last_index": last_index, "all_triples": all_triples}
    PROGRESS_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description="ViFoodKG Step 2 - Triple Extraction")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of MainDishes to process (0 = all)")
    parser.add_argument("--start", type=int, default=0, help="Force start from specific index in the MainDish list")
    args = parser.parse_args()

    # Load entities
    entities = json.loads(MASTER_FILE.read_text(encoding="utf-8"))
    main_dishes = [e["canonical_label"] for e in entities if e["category"] == "MainDish"]
    print(f"Loaded {len(main_dishes)} MainDish entities")

    # Resume by index
    saved_index, all_triples = load_progress()
    
    start_idx = args.start if args.start > 0 else saved_index
    if start_idx >= len(main_dishes):
        print("All dishes processed!")
        return
        
    remaining_dishes = main_dishes[start_idx:]
    print(f"Resuming from index {start_idx} ({len(remaining_dishes)} dishes remaining)\n")

    if args.limit > 0:
        remaining_dishes = remaining_dishes[:args.limit]
        print(f"  --limit {args.limit}: processing only {len(remaining_dishes)} dishes")

    # Create batches
    batches = [remaining_dishes[i:i + BATCH_SIZE] for i in range(0, len(remaining_dishes), BATCH_SIZE)]
    total_batches = len(batches)
    print(f"Batches: {total_batches} x {BATCH_SIZE} | Model: {GEMINI_MODEL}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    current_idx = start_idx
    for idx, batch in enumerate(batches):
        print(f"[{idx + 1}/{total_batches}] (Index {current_idx} -> {current_idx + len(batch) - 1}): {', '.join(batch)}")

        # Step 1: Crawl web for each dish
        dishes_with_content = []
        for dish in batch:
            print(f"  Crawling: {dish}...", end=" ", flush=True)
            sources = crawl_for_dish(dish)
            dishes_with_content.append((dish, sources))
            src_count = len(sources)
            print(f"{src_count} source(s)")
            time.sleep(0.5)

        # Step 2: Extract triples via LLM
        print(f"  Extracting triples...", end=" ", flush=True)
        results = call_gemini_extract(dishes_with_content)
        triple_count = sum(len(r.get("triples", [])) for r in results)
        print(f"{triple_count} triples")

        all_triples.extend(results)
        
        # Advance index and save checkpoint
        current_idx += len(batch)
        save_progress(current_idx, all_triples)

        if idx + 1 < total_batches:
            time.sleep(2)

    # Save final output
    TRIPLES_FILE.write_text(
        json.dumps(all_triples, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Summary
    total_t = sum(len(r.get("triples", [])) for r in all_triples)
    dishes_with_t = sum(1 for r in all_triples if r.get("triples"))
    print(f"\n-- Results --")
    print(f"  Dishes processed:  {len(all_triples)}")
    print(f"  Dishes with data:  {dishes_with_t}")
    print(f"  Total triples:     {total_t}")
    print(f"\nSaved to {TRIPLES_FILE}")


if __name__ == "__main__":
    main()
