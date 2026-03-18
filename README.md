# ViFoodKG - Knowledge Graph cho Ẩm thực Việt Nam

## Mô tả

**ViFoodKG** là hệ thống xây dựng Knowledge Graph (đồ thị tri thức) về ẩm thực Việt Nam, phục vụ bài toán **Visual Question Answering (VQA)**. Pipeline bao gồm 3 bước chính:

1. **Entity Extraction** — Trích xuất nhãn thực thể thức ăn từ Supabase.
2. **Entity Classification** — Chuẩn hóa & phân loại thực thể bằng Gemini LLM.
3. **Triple Extraction** — Thu thập dữ liệu web + trích xuất bộ ba tri thức (triples) theo ontology.

> Dự án sử dụng **Google Gemini API** cho các bước LLM, **Supabase** làm nguồn dữ liệu ảnh, và lược đồ ontology tùy chỉnh với **12 quan hệ** & **11 loại câu hỏi** cho VQA.

---

## Cấu trúc dự án

```
ViFoodKG/
├── src/
│   ├── 01_kg_entity_extractor.py   # Bước 1: Trích xuất nhãn food entities từ Supabase
│   ├── 02_kg_entity_classifier.py  # Bước 2: Chuẩn hóa & phân loại bằng Gemini
│   └── 03_kg_triple_extractor.py   # Bước 3: Crawl web + trích xuất triples bằng Gemini
├── config/
│   ├── dishes.txt                  # Danh sách 78 món ăn Việt Nam
│   ├── ontology_config.json        # Lược đồ ontology (entity classes, relations, question types)
│   └── test_dishes.json            # Dữ liệu test nhỏ
├── data/
│   ├── raw_unique_labels.json      # Output Bước 1: nhãn thực thể unique
│   ├── master_entities.json        # Output Bước 2: thực thể đã chuẩn hóa & phân loại
│   ├── raw_data/                   # Dữ liệu .md crawled cho 78 món ăn
│   ├── triples/
│   │   └── extracted_triples.json  # Output Bước 3: bộ ba tri thức
│   └── extracted/
│       └── Question Type.md        # Tài liệu phân loại câu hỏi
├── tests/
├── monan.txt                       # Danh sách 78 món ăn (plain text)
├── .env.example                    # Template biến môi trường
├── pyproject.toml                  # Cấu hình project & dependencies
├── .gitignore
└── README.md
```

---

## Yêu cầu

- **Python** ≥ 3.11
- API keys: **Supabase**, **Google Gemini**
- *(Tùy chọn)* Firecrawl API, Exa API (fallback scraper)

---

## Cài đặt

```bash
# Tạo virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/macOS

# Cài đặt dependencies
pip install -e ".[dev]"

# Cấu hình environment
cp .env.example .env
# Chỉnh sửa .env với API keys thực
```

### Biến môi trường (`.env`)

| Biến | Mô tả | Bắt buộc |
|------|--------|----------|
| `SUPABASE_URL` | URL project Supabase | ✅ |
| `SUPABASE_KEY` | Anon/Service key Supabase | ✅ |
| `GEMINI_API_KEY` | Google Gemini API key | ✅ |
| `NEO4J_URI` | Neo4j bolt URI | ⬜ *(chưa dùng)* |
| `NEO4J_USERNAME` | Neo4j username | ⬜ *(chưa dùng)* |
| `NEO4J_PASSWORD` | Neo4j password | ⬜ *(chưa dùng)* |
| `FIRECRAWL_API_KEY` | Firecrawl API key (fallback) | ⬜ |
| `EXA_API_KEY` | Exa API key (fallback) | ⬜ |

---

## Pipeline chạy

### Bước 1 — Entity Extraction (Supabase → JSON)

```bash
python src/01_kg_entity_extractor.py
```

- Kết nối Supabase, đọc cột `food_items` từ bảng `image`.
- Deduplicate & chuẩn hóa Unicode NFC.
- **Output:** `data/raw_unique_labels.json`

### Bước 2 — Entity Classification (Gemini LLM)

```bash
python src/02_kg_entity_classifier.py
```

- Gửi nhãn thô theo batch (50 nhãn/batch) đến Gemini để chuẩn hóa & phân loại.
- Phân loại thành 5 categories: **MainDish**, **Component/Ingredient**, **SideDish**, **Condiment**, **Discard**.
- Hỗ trợ **checkpoint/resume** tự động (lưu progress vào `_classifier_progress.json`).
- **Output:** `data/master_entities.json`

### Bước 3 — Triple Extraction (Web-Grounded)

```bash
python src/03_kg_triple_extractor.py               # Chạy toàn bộ MainDishes
python src/03_kg_triple_extractor.py --limit 3      # Test với 3 món
```

- Với mỗi MainDish, crawl nội dung từ **Wikipedia tiếng Việt** + **Google Search** (fallback).
- Gửi nội dung crawled + ontology schema đến Gemini để trích xuất triples.
- Mỗi triple bắt buộc có `source_url` + `evidence` (web-grounded, không hallucination).
- Batch 5 món/lần, checkpoint-resumable.
- **Output:** `data/triples/extracted_triples.json`

---

## Ontology Schema

### Entity Classes

| Class | Mô tả | Ví dụ |
|-------|--------|-------|
| `Dish` | Món ăn hoàn chỉnh | Phở Bò, Bún Chả, Cơm Tấm |
| `Ingredient` | Nguyên liệu riêng lẻ | Thịt Bò, Tôm, Giá Đỗ |
| `IngredientCategory` | Phân loại nguyên liệu | Thịt, Hải sản, Rau lá |
| `Region` | Vùng miền xuất xứ | Hà Nội, Huế, Sài Gòn |
| `DishType` | Loại món | Món nước, Món khô, Bánh |
| `CookingTechnique` | Kỹ thuật chế biến | Ninh, Chiên, Xào, Hấp |
| `FlavorProfile` | Hương vị đặc trưng | Cay, Chua, Ngọt, Mặn |
| `DietaryTag` | Chế độ ăn | animal_product, plant_based |
| `Allergen` | Chất gây dị ứng | Giáp xác, Gluten, Đậu nành |
| `SideDish` | Món ăn kèm | Rau Sống, Quẩy, Đồ Chua |

### Quan hệ (Relations)

| Relation | Domain → Range | Hops |
|----------|---------------|------|
| `hasIngredient` | Dish → Ingredient | 1 |
| `servedWith` | Dish → SideDish | 1 |
| `originRegion` | Dish → Region | 1 |
| `dishType` | Dish → DishType | 1 |
| `cookingTechnique` | Dish → CookingTechnique | 1 |
| `flavorProfile` | Dish → FlavorProfile | 1 |
| `ingredientCategory` | Ingredient → IngredientCategory | 2 |
| `hasAllergen` | Ingredient → Allergen | 2 |
| `hasDietaryTag` | Ingredient → DietaryTag | 2 |
| `hasSubRule` | Dish → SubstitutionRule | 1 |
| `fromIngredient` | SubstitutionRule → Ingredient | — |
| `toIngredient` | SubstitutionRule → Ingredient | — |

---

## Trạng thái phát triển

| Bước | Mô tả | Trạng thái |
|------|-------|------------|
| 1 | Entity Extraction (Supabase) | ✅ Done |
| 2 | Entity Classification (Gemini LLM) | ✅ Done |
| 3 | Web-Grounded Triple Extraction | 🔄 In Progress |
| 4 | Neo4j Graph Loading | ⏳ Pending |

---

## Tech Stack

- **Python 3.11+** — Ngôn ngữ chính
- **Google Gemini API** (`google-generativeai`) — LLM cho classification & triple extraction
- **Supabase** (`supabase-py`) — Nguồn dữ liệu ảnh thực phẩm
- **BeautifulSoup4** + **lxml** — Web scraping
- **Pydantic** — Data validation
- **Rich** — Console output formatting
- **Ruff** / **MyPy** / **Pytest** — Linting, type checking, testing