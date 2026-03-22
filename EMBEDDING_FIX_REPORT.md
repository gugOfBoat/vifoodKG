# 📈 Báo Cáo Sửa Lỗi: Mở rộng Chỉ mục Vector (Vector Index Expansion)

## 1. Tóm tắt Vấn đề
- **Nguyên nhân:** Sau khi hệ thống chạy Stage 2b (Enrichment), số lượng quan hệ (triples) trong đồ thị tăng vọt lên **9,765** (so với 5,752 ban đầu). Các quan hệ suy luận mới (`hasSubRule`, `fromIngredient`, `toIngredient`) chưa được cấu hình nhúng vector trong hàm script `05_kg_vectorizer.py`.
- **Hậu quả:** Gây ra "vùng mù" khiến cho hàm `retrieve` trên các câu hỏi liên quan đến kiến thức bổ sung này trả về rỗng vì bị filter theo điều kiện `embedding IS NOT NULL`.

## 2. Các Bước Đã Xử Lý
1. **Sửa đổi Script `05_kg_vectorizer.py`**:
   - Thêm 3 quan hệ bị thiếu: `hasSubRule`, `fromIngredient`, `toIngredient` vào mảng `VECTORIZE_REL_TYPES`.
   - Nâng cấp phần logic tạo Index: Thêm câu lệnh `DROP INDEX triple_vector_index IF EXISTS` và cập nhật `CREATE VECTOR INDEX` để bao hàm trọn vẹn 12 loại quan hệ.
   - Bổ sung cấu hình để script chạy mặc định trên `CPU`.
2. **Đo đạc "Vùng Mù"**: 
   - Kiểm tra bằng script `debug_blind_spots.py`, phát hiện có **13,820** vector edges (tính cả có hướng và phân mảnh) còn thiếu embedding.
3. **Thực thi Nhúng (Embedding)**: 
   - Chạy pipeline nhúng qua mô hình `intfloat/multilingual-e5-small` trực tiếp trên CPU. Quá trình thành công cập nhật 100% các cạnh.
4. **Cập nhật Notebook EDA**:
   - Chỉnh sửa file `notebooks/Neo4j_KG_EDA.ipynb`: Thêm cell `Thống kê Vector Embeddings (Vectorization Coverage)` nhằm đếm chính xác số lượng relationship đã có vector vs. số vùng mù còn tồn đọng.

## 3. Kết Quả Cuối Cùng
- **Total Newly Embedded:** `13,820` edges.
- **Neo4j Null Embedding Count:** Bằng `0` edges (Không còn Vùng Mù nào).
- **Query Test ("Phở Bò"):** Module `query.py` giờ đây có thể Vector-Search chính xác và lấy lại được cả kiến thức bổ sung (như Dị ứng, Quy tắc thay thế nguyên liệu) mà Graph Rag không thể tìm thấy trước kia.

Mọi lỗ hổng trong đồ thị đã được trám kín. 100% Pipeline Trích xuất - Lưu trữ - Nhúng đã tự động hóa và vững chắc!
