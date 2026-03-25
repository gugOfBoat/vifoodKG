<div align="center">
  <img src="https://img.shields.io/badge/Neo4j-008CC1?style=for-the-badge&logo=neo4j&logoColor=white" />
  <img src="https://img.shields.io/badge/Gemini-8E75B2?style=for-the-badge&logo=google-gemini&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Sentence_Transformers-FF9D00?style=for-the-badge&logo=huggingface&logoColor=white" />

  # 🍜 ViFoodKG: Vietnamese Food Knowledge Graph
  ### *Hệ thống Tri thức Ẩm thực Việt Nam phục vụ Visual Question Answering (VQA)*
</div>

---

**ViFoodKG** là một Knowledge Graph (Đồ thị Tri thức) chuyên sâu về ẩm thực Việt Nam, được thiết kế đặc biệt để làm nền tảng lưu trữ và truy xuất cho hệ thống RAG (Retrieval-Augmented Generation) phục vụ bài toán Visual Question Answering (VQA).

Dự án này tự động khai phá tri thức từ web tĩnh (Wikipedia), trích xuất các bộ ba tri thức (triples) thông qua LLM (Google Gemini), lưu trữ cấu trúc Topology trên hệ quản trị cơ sở dữ liệu đồ thị **Neo4j**, và tích hợp **Vector Search** trên đồ thị (Hybrid Retrieval).

---

## 🌟 Tính năng nổi bật

1. **Web-Grounded Extraction:** 100% tri thức trích xuất đều được neo với Nguồn (URL) và Bằng chứng (Evidence textual snippet), chống hội chứng "ảo giác" (hallucination) của LLM.
2. **Comprehensive Ontology:** Mô hình hóa 11 loại thực thể (Dish, Ingredient, Region, Allergen, v.v.) và 10 mối quan hệ phức tạp (1-hop, 2-hop, và reification qua quy tắc thay thế nguyên liệu).
3. **Graph-Vector Hybrid Retrieval:** Chiến lược truy xuất kết hợp Lược đồ Đồ thị (Neo's Graph Traversal) và Không gian Vector (Cosine Similarity) để đem lại độ chính xác cục bộ cao nhất.
---

## 🏗️ Kiến trúc & Quy trình (The Pipeline)

Dự án được chia thành 5 giai đoạn liên tiếp. Mỗi giai đoạn đại diện cho một script trong thư mục `src/`:

### 1. Trích xuất Thực thể (Entity Extraction)
**Script:** `01_kg_entity_extractor.py`
- Lọc danh sách món ăn từ các nhãn thô (raw labels) có trong cơ sở dữ liệu ảnh (Supabase). Output: `raw_unique_labels.json`.

### 2. Phân loại & Chuẩn hóa Thực thể (Entity Classification)
**Script:** `02_kg_entity_classifier.py`
- Sử dụng Gemini API để làm sạch dữ liệu nhiễu, chuyển các nhãn thô thành tên món ăn chuẩn (ví dụ "bun_bo" → "Bún Bò Huế") và phân loại thành các danh mục (MainDish, Ingredient, Region...). Output: `master_entities.json`.

### 3. Khai phá Tri thức (Web-Grounded Triple Extraction)
**Script:** `03_kg_triple_extractor.py`
- Thu thập dữ liệu từ Wikipedia Tiếng Việt cho từng món ăn.
- Lấy tri thức lai (Hybrid Extraction): Prompt LLM Gemini đọc hiểu văn bản web và trích xuất thành các Knowledge Triples `(Subject) -[Relation]-> (Target)`. Sau đó, yêu cầu LLM sử dụng "Kiến thức Nội bộ Chuyên gia" (Common Sense/Cognitive Reasoning) để điền vào các vùng thông tin còn thiếu (như Dị ứng, Hương vị, Vùng miền).
- Fallback: Nếu không tìm thấy thông tin trên Web, ép LLM bật chế độ Suy luận Đặc biệt để sinh bộ ba (đánh dấu source là `LLM_Knowledge`).

### 3b. Khám phá và Thống kê Đồ thị (Neo4j EDA)
**Notebook:** `notebooks/Neo4j_KG_EDA.ipynb`
- File Jupyter Notebook cung cấp giao diện tương tác trực tiếp với cơ sở dữ liệu Neo4j.
- Chứa các hàm cơ bản để thống kê số lượng Nodes, Relationships, truy vấn các món ăn mồ côi, đếm phân bố loại quan hệ, và trực quan hóa subgraph mẫu.

### 4. Đổ dữ liệu lên Neo4j (Neo4j Ingestion)
**Script:** `04_kg_neo4j_ingestor.py`
- Parse file JSON để tạo các Nodes và Relationships tren Neo4j Cloud (AuraDB).
- Xử lý phức tạp: **Reification** (Vật hóa) đối với quy tắc mô tả linh hoạt sự thay thế nguyên liệu (ví dụ: *Bún Chả có thể thay thịt lợn bằng thịt bò* tạo thành chuỗi 3 nodes `Dish → SubstitutionRule → Ingredient`).
- Văn bản hóa (Verbalization) các cung (edges) phục vụ cho Vectorization sau này.

### 5. Nhúng Vector & Indexing (NeoEdge Vectorization)
**Script / Notebook:** `05_kg_vectorizer.py` & `notebooks/05_vectorizer_colab.py`
- Sử dụng mô hình `intfloat/multilingual-e5-small` để biến các thuộc tính `verbalized_text` nằm trên Edge (Mối quan hệ) thành Vector 384 chiều.
- Lưu lại vào thuộc tính `embedding` của cạnh và tạo **Vector Index** trên Neo4j bằng thuật toán độ đo Cosine.

---

## 🔍 Chiến lược trích xuất triples

Dữ liệu đầu vào: `danh_sách_thực_thể_trong_ảnh` (từ mô hình CV) + `câu_hỏi_của_user` (Text).

1. **Neo (Anchor):** 
   Thay vì chạy "Vector Search" toàn bộ cơ sở dữ liệu làm nhiễu thông tin, chúng ta giới hạn vùng không gian đồ thị bằng cách neo (MATCH) vào những Node Thực thể đang xuất hiện cụ thể trong ảnh.
2. **Traverse (Mở rộng):**
   Tìm tất cả các mối quan hệ xuất phát từ những Node được neo. Hỗ trợ quét các mối quan hệ trực tiếp (1-hop) lẫn các quan hệ bắc cầu nâng cao (2-hop) như *Dish → Ingredient → Allergen*.
3. **Rank (Xếp hạng):**
   Cộng gộp danh sách thực thể với câu hỏi của người dùng để tạo thành một Vector Ý định (Query Vector). Dùng tích Vô hướng Cosine so sánh Vector Ý định này với các Vector của cung (Edge Embeddings) đã thu được ở bước 2. Đặc biệt, nếu một cung nối 2 đầu là 2 thực thể cùng có mặt trong ảnh (Internal Link), cung đó sẽ được cộng điểm thưởng ưu tiên.

Kết quả cuối cùng là **Top K Knowledge Triples** phù hợp nhất theo cả ngữ cảnh thị giác và ngữ nghĩa để đẩy vào LLM sinh câu trả lời VQA. 

---

## ⚙️ Cài đặt & Cấu hình (Local)

1. **Clone dự án & Cài đặt môi trường:**
   ```bash
   git clone https://github.com/gugOfBoat/vifoodKG.git
   cd vifoodKG
   # Cài đặt qua uv hoặc pip
   pip install -r requirements.txt # (nếu có)
   # Hoặc cài module tay
   pip install neo4j sentence-transformers python-dotenv google-generativeai requests beautifulsoup4
   ```

2. **Tạo biến môi trường `.env`:**
   Dựa vào file `.env.example`, tạo file `.env` tại thư mục gốc và điền các khóa bảo mật (API keys, Neo4j credentials).

3. **Thực thi Query Testing:**
   ```bash
   # Truy vấn xem nguyên liệu món ăn
   python src/query.py -i "Phở Bò" "Thịt Bò" -q "nguyên liệu chính" -k 5
   
   # Cần xuất JSON để pipe cho các service khác
   python src/query.py -i "Bánh Xèo" -q "chất gây dị ứng" -k 3 --json
   ```
