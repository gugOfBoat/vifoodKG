# Báo cáo lỗi coverage KG khi sinh VQA cho ảnh bị thiếu

## 1. Bối cảnh

Hiện tại bảng `image` có **1430 ảnh hợp lệ** (`is_checked = True`, `is_drop = False`), nhưng bảng `vqa` mới chỉ có câu hỏi cho **1201 ảnh**.  
Nhóm ảnh còn thiếu là **229 ảnh** (danh sách ảnh còn thiếu nằm ở file [`missing_image_ids.txt`](../data/vqa_rerun_missing/missing_image_ids.txt)).

Để chẩn đoán nguyên nhân, tui đã rerun riêng đúng 229 `image_id` bị thiếu và bật chế độ debug để ghi lại fail stage theo từng ảnh/qtype. File debug ([`debug_failures.csv`](../data/vqa_rerun_missing/debug_run1/debug_failures.csv)) cho thấy vấn đề chính **không nằm ở Gemini/parser**, mà nằm ở **coverage của KG và bước map dish / retrieval**.

---

## 2. Kết quả tổng hợp

### 2.1. Phân bố nguyên nhân theo ảnh unique

| fail_stage | so_anh_unique | y_nghia |
| --- | --- | --- |
| no_anchor_dish | 117 | Không map được food_items của ảnh sang Dish node trong KG |
| retrieve_empty | 102 | Map được Dish rồi nhưng retriever không lấy ra edge/tri thức nào đã vectorize |
| no_candidates | 10 | Có retrieve một ít dữ liệu nhưng không đủ relation path để tạo candidate |

### 2.2. Kết luận ngắn gọn

- **117/229 ảnh (51.1%)** fail ở `no_anchor_dish`  
  → chưa map được `food_items` của ảnh sang một `Dish` trong KG.

- **102/229 ảnh (44.5%)** fail ở `retrieve_empty`  
  → đã map được `anchor_dish`, nhưng retriever không lấy ra được tri thức hữu ích.

- **10/229 ảnh (4.4%)** fail ở `no_candidates`  
  → có retrieve được một ít dữ liệu, nhưng sau khi lọc relation path thì không đủ để tạo câu hỏi.

### 2.3. Ý nghĩa kỹ thuật

Pipeline hiện tại fail chủ yếu ở **3 bước trước LLM**:

1. **Dish matching / alias matching**
2. **Neo4j retrieval trên các edge đã có embedding**
3. **Candidate construction từ relation path**

Không thấy lỗi nổi bật ở các bước như:
- `parse_failed`
- `validate_failed`
- `gemini_empty`

=> Ưu tiên xử lý nên là **bổ sung tri thức và alias trong KG**, không phải tinh chỉnh prompt Gemini.

---

## 3. Phân tích theo từng nhóm lỗi

## 3.1. Nhóm `no_anchor_dish` (117 ảnh)

### Mô tả
Ảnh có `food_items` đầy đủ, nhưng hàm chọn `anchor_dish` không tìm được món nào trong danh sách đó khớp với một `Dish` node hiện có trong KG.

### Ví dụ thực tế
| image_id | detail |
| --- | --- |
| image000006 | food_items=['Bún chả', 'Bún', 'Chả nướng', 'Rau sống', 'Lạc rang', 'Nước chấm'] |
| image000011 | food_items=['Bún chả', 'Bún tươi', 'Chả nướng', 'Rau sống', 'Nước chấm'] |
| image000014 | food_items=['Bún bò Huế', 'Rau sống', 'Chanh', 'Ớt', 'Hành tím', 'Tỏi', 'Chả', 'Xương bò hầm', 'Huyết', 'Thịt bò', 'Bún'] |
| image000026 | food_items=['Canh cua mồng tơi mướp', 'Đậu phụ sốt cà chua', 'Thịt rang cháy cạnh', 'Nem chua', 'Cà pháo muối', 'Nước mắm tỏi ớt', 'Tương ớt'] |
| image000055 | food_items=['Vịt quay', 'Bún tươi', 'Bánh đa nem', 'Rau sống', 'Dưa chuột', 'Cà rốt', 'Dứa', 'Nước chấm'] |
| image000102 | food_items=['Vịt quay', 'Bí đỏ xào tôm', 'Bắp cải luộc', 'Bò kho', 'Bưởi', 'Cà pháo'] |
| image000121 | food_items=['Bún chả', 'Bún', 'Rau sống', 'Tỏi', 'Ớt', 'Tương ớt', 'Nước mắm'] |
| image000139 | food_items=['Bún chả', 'Thịt nướng', 'Chả viên nướng', 'Nước chấm', 'Rau sống', 'Bún'] |
| image000155 | food_items=['Canh cà chua trứng', 'Đậu phụ kho nấm', 'Mực xào dưa chuột', 'Cam'] |
| image000174 | food_items=['Cơm trắng', 'Thịt heo nướng', 'Xà lách', 'Cà chua', 'Dưa leo', 'Đồ chua', 'Nước chấm'] |

### Nhận định
Nhiều món trong nhóm này là món rất phổ biến, ví dụ:
- `Bún chả`
- `Bún bò Huế`
- `Vịt quay`
- `Canh cua mồng tơi mướp`
- `Cơm trắng`
- `Canh cà chua trứng`

Điều này cho thấy khả năng cao là:
- KG **chưa có node Dish** cho món đó, hoặc
- KG có rồi nhưng **thiếu alias/biến thể tên món**, hoặc
- tên trong `food_items` và tên trong KG đang lệch chuẩn hóa.

### Yêu cầu bổ sung cho KG
Ưu tiên kiểm tra và bổ sung:
- node `Dish` còn thiếu
- alias / tên thay thế / biến thể viết hoa-thường / biến thể có dấu
- canonical name hợp lý cho các món phổ biến

### Gợi ý normalize/alias cần chú ý
- `Bún chả`
- `Bún bò Huế`
- `Vịt quay`
- `Canh cua mồng tơi mướp`
- `Canh cà chua trứng`
- `Cơm trắng`
- `Cá kho tộ` / `Cá kho`
- `Cá rán` / `Cá chiên`
- `Bún` hoặc `Bún tươi` **không nên** là anchor dish nếu chỉ là thành phần/phần ăn kèm

---

## 3.2. Nhóm `retrieve_empty` (102 ảnh)

### Mô tả
Pipeline đã chọn được `anchor_dish`, nhưng retriever không lấy ra được relation nào đủ dùng (`retrieved = 0` về mặt hữu ích cho downstream).

### Ví dụ thực tế
| image_id | detail |
| --- | --- |
| image000001 | anchor_dish=Cá Kho; top_k=8 |
| image000029 | anchor_dish=Tôm Rim; top_k=8 |
| image000034 | anchor_dish=Canh Măng Vịt; top_k=8 |
| image000046 | anchor_dish=Cá Kho; top_k=8 |
| image000058 | anchor_dish=Canh Ngao Nấu Cà Chua; top_k=8 |
| image000079 | anchor_dish=Canh Khoai Sọ Nấu Xương; top_k=8 |
| image000080 | anchor_dish=Xôi; top_k=8 |
| image000106 | anchor_dish=Cá Kho; top_k=8 |
| image000128 | anchor_dish=Tôm Rim; top_k=8 |
| image000150 | anchor_dish=Tôm Rim; top_k=8 |

### Nhận định
Nhóm này cho thấy:
- `Dish` node **đã tồn tại trong KG**
- nhưng tri thức liên kết với dish đó còn quá ít, hoặc
- có tri thức nhưng **chưa được vectorize / chưa có embedding**, hoặc
- relation hiện có không nằm trong các relation path mà VQA generator đang dùng

### Các qtype hiện tại cần hỗ trợ
Pipeline đang dùng các question type sau:
- `ingredients`
- `cooking_technique`
- `flavor_profile`
- `origin_locality`
- `allergen_restrictions`
- `dietary_restrictions`
- `ingredient_category`
- `food_pairings`
- `dish_classification`

### Yêu cầu bổ sung cho KG
Với mỗi `Dish` bị `retrieve_empty`, cần ưu tiên bổ sung triples cho các nhóm tri thức sau:

- **ingredients**  
  Dish → containsIngredient / hasIngredient → Ingredient

- **cooking_technique**  
  Dish → hasCookingTechnique → CookingTechnique

- **flavor_profile**  
  Dish → hasFlavor / flavorProfile → Flavor

- **origin_locality**  
  Dish → originatesFrom / hasOrigin → Locality/Region

- **allergen_restrictions**  
  Dish/Ingredient → containsAllergen → Allergen

- **dietary_restrictions**  
  Dish → suitableFor / unsuitableFor → DietaryPattern

- **ingredient_category**  
  Ingredient → belongsToCategory → IngredientCategory

- **food_pairings**  
  Dish → pairsWith → Dish/Drink/Side

- **dish_classification**  
  Dish → belongsToClass → DishClass

### Lưu ý rất quan trọng
Sau khi bổ sung triples, cần **chạy lại bước vectorize embedding cho relationship**.  
Nếu edge tồn tại nhưng chưa có embedding, retriever vẫn có thể trả rỗng.

---

## 3.3. Nhóm `no_candidates` (10 ảnh)

### Mô tả
Retriever có trả về một ít dữ liệu, nhưng sau khi lọc relation path thì không còn row nào phù hợp để tạo candidate answer.

### Ví dụ thực tế
| image_id | detail |
| --- | --- |
| image000247 | anchor_dish=Cà Tím Kho; retrieved=1; filtered_rows=0; relations=[] |
| image000709 | anchor_dish=Cơm Niêu; retrieved=1; filtered_rows=0; relations=[] |
| image000932 | anchor_dish=Cua Hấp; retrieved=1; filtered_rows=0; relations=[] |
| image001391 | anchor_dish=Cá Rán; retrieved=1; filtered_rows=0; relations=[] |
| image001403 | anchor_dish=Cơm Niêu; retrieved=1; filtered_rows=0; relations=[] |
| image001487 | anchor_dish=Cơm Niêu; retrieved=1; filtered_rows=0; relations=[] |
| image001761 | anchor_dish=Cơm Niêu; retrieved=1; filtered_rows=0; relations=[] |
| image001926 | anchor_dish=Gan Xào Tỏi; retrieved=1; filtered_rows=0; relations=[] |
| image001939 | anchor_dish=Cơm Niêu; retrieved=1; filtered_rows=0; relations=[] |
| image002301 | anchor_dish=Cá Trê Chiên; retrieved=2; filtered_rows=0; relations=[] |

### Nhận định
Đây là nhóm coverage “lưng chừng”:
- KG có dữ liệu
- nhưng relation path chưa đúng hoặc chưa đủ đậm đặc
- candidate constructor không tìm được câu trả lời/ distractor phù hợp

### Yêu cầu bổ sung cho KG
Với các dish thuộc nhóm này, cần:
- tăng số triple liên quan trực tiếp đến dish
- bổ sung entity đích rõ ràng hơn cho qtype tương ứng
- tránh tạo relation quá generic nhưng không có object đủ chuẩn

---

## 4. Việc cần thành viên phụ trách KG thực hiện

## Ưu tiên 1 — mở rộng `Dish` coverage và alias
Cần rà soát các món fail ở `no_anchor_dish` và:
- thêm node `Dish` còn thiếu
- thêm alias cho món phổ biến
- thống nhất canonical name
- tách món chính với món ăn kèm/nguyên liệu phụ

## Ưu tiên 2 — enrich tri thức cho các dish bị `retrieve_empty`
Với mỗi dish đã map được nhưng retrieve rỗng:
- bổ sung triples theo 9 qtype đang dùng
- đặc biệt ưu tiên `ingredients`, `cooking_technique`, `dish_classification`, `origin_locality`
- bảo đảm entity đích là node có nghĩa, không phải text rời

## Ưu tiên 3 — chạy lại vectorizer
Sau khi thêm triples, cần chạy lại bước vector hóa edge để retriever nhìn thấy tri thức mới.

Nếu không rerun vectorizer, tình trạng `retrieve_empty` vẫn sẽ giữ nguyên.

---

## 5. Checklist bàn giao cho người làm KG

### A. Với `no_anchor_dish`
- [ ] Món đã có node `Dish` chưa?
- [ ] Có alias đúng như trong `food_items` chưa?
- [ ] Có biến thể tên món phổ biến chưa?
- [ ] Có đang dùng một tên quá hẹp/quá khác thực tế không?
- [ ] Có đang thiếu món vùng miền phổ biến không?

### B. Với `retrieve_empty`
- [ ] Dish đã có edge nào hữu ích cho 9 qtype chưa?
- [ ] Edge đó đã được vectorize chưa?
- [ ] Object node của edge có tồn tại và chuẩn hóa chưa?
- [ ] Có đủ tri thức để tạo distractor không?

### C. Với `no_candidates`
- [ ] Relation path có khớp logic qtype không?
- [ ] Có ít nhất 1 đáp án đúng rõ ràng không?
- [ ] Có đủ lựa chọn sai hợp lý để sinh MCQ không?

---

## 6. Kết luận

229 ảnh bị thiếu VQA **không phải do script bỏ sót khi chạy**, mà chủ yếu do **KG chưa đủ coverage** ở 2 lớp:

1. **Coverage của Dish / alias matching**
2. **Coverage của relation + embedding cho retrieval**

Do đó, hướng xử lý đúng là:
- **bổ sung node Dish và alias**
- **bổ sung triples theo qtype**
- **vectorize lại edge**
- sau đó mới rerun sinh VQA

Nếu cần ưu tiên công việc, nên làm theo thứ tự:
1. Sửa `no_anchor_dish`
2. Sửa `retrieve_empty`
3. Sửa `no_candidates`

---

## 7. Gợi ý đầu ra mong muốn sau khi phía KG cập nhật

Sau khi thành viên KG bổ sung xong, nên có:
- danh sách món mới được thêm vào `Dish`
- danh sách alias mới
- số triple mới theo từng qtype
- xác nhận đã rerun vectorizer
- một lần rerun VQA cho đúng 229 ảnh để đo mức cải thiện coverage
