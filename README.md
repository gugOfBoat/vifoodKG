# ViFoodVQA - Knowledge Graph for Vietnamese Cuisine

## Mô tả
**ViFoodVQA** là hệ thống Knowledge Graph về ẩm thực Việt Nam, được xây dựng trên Neo4j. Hệ thống sử dụng Gemini API để trích xuất bộ triple ngữ nghĩa từ dữ liệu thu thập được, phục vụ bài toán Visual Question Answering (VQA) cho ẩm thực Việt.

## Cấu trúc dự án

```
ViFoodVQA/
├── src/
│   ├── 01_scraper.py       # Thu thập dữ liệu từ Wikipedia + Fallback
│   ├── 02_extractor.py     # (Phase 3) Trích xuất triple bằng Gemini
│   └── 03_neo4j_loader.py  # (Phase 4) Nạp vào Neo4j
├── config/
│   ├── dishes.txt          # Danh sách 78 món ăn Việt
│   └── ontology_config.json
├── data/
│   └── raw_data/           # File .md cho từng món ăn
├── tests/
├── .env.example
├── pyproject.toml
└── README.md
```

## Cài đặt

```bash
# Tạo virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows

# Cài đặt dependencies
pip install -e ".[dev]"

# Cấu hình environment
cp .env.example .env
# Chỉnh sửa .env với API keys thực
```

## Phases

| Phase | Mô tả | Trạng thái |
|-------|-------|------------|
| 1 | Project Scaffolding | ✅ Done |
| 2 | Data Acquisition (Scraping) | 🔄 In Progress |
| 3 | LLM Triple Extraction (Gemini) | ⏳ Pending |
| 4 | Neo4j Graph Loading | ⏳ Pending |

## Chạy scraper

```bash
python src/01_scraper.py
```
Kết quả sẽ được lưu vào `data/raw_data/*.md`.

## Quan hệ (Relations) trong Knowledge Graph

Dựa trên `Question Type.md`:
- `hasIngredient`, `servedWith`, `originRegion`, `dishType`
- `ingredientCategory`, `hasAllergen`, `flavorProfile`, `hasDietary`
- `hasSubRule`, `fromIngredient`, `toIngredient`
