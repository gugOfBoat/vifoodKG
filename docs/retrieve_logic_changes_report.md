# Báo cáo thay đổi logic retrieve triple từ KG

Tài liệu này tóm tắt các thay đổi đã được áp dụng vào `query.py` và `06_generate_vqa.py` để cải thiện bước retrieve triple/path từ Knowledge Graph (KG) trước khi sinh VQA.

## 1. Bối cảnh và vấn đề của logic cũ

Logic cũ trong `query.py` có thể tóm tắt là:

1. Anchor vào món ăn (`Dish`) xuất hiện trong ảnh.
2. Traverse neighborhood 1-hop và 2-hop quanh món đó trong Neo4j.
3. Embed **một câu retrieval query dài**.
4. Tính cosine giữa query vector và **embedding của từng edge** trong KG.
5. Lấy `top_k` toàn cục.

### Vấn đề thực tế

Cách làm này gây ra 3 vấn đề chính:

#### 1. Query vector bị “dish-heavy”
Trong `06_generate_vqa.py`, retrieval query cũ chứa nhiều thông tin như:
- tên món neo (`anchor dish`)
- danh sách `food_items`
- `image_description`
- `question_type`
- `detail_description`
- `relationship_path`

Hệ quả là vector truy vấn bị kéo mạnh về các tín hiệu liên quan tới **tên món** và ngữ cảnh ảnh.

#### 2. 1-hop dễ lấn át 2-hop
Trong logic cũ, mỗi row được rank bằng embedding của **một cạnh đơn lẻ** (`r.embedding`).

- 1-hop thường có dạng `Dish -> relation -> target`, nên edge text thường đã chứa trực tiếp tên món.
- 2-hop thường có dạng `Dish -> Ingredient -> relation -> target`, nhưng embedding dùng để rank lại chỉ nằm ở **cạnh thứ hai**, nên thiếu ngữ cảnh từ `Dish`.

Vì vậy, khi `top_k` nhỏ (ví dụ `top_k = 8`), các cạnh 1-hop như `Dish -> hasIngredient -> X` rất dễ chiếm hết top-k, còn các quan hệ 2-hop cần cho qtype như `hasAllergen`, `hasDietaryTag`, `ingredientCategory` thường bị đẩy xuống sau.

#### 3. Lọc theo qtype diễn ra quá muộn
Luồng cũ là:

```text
retrieve local rows -> rank global -> lấy top_k -> select_candidates(qtype, rows)
```

Tức là hệ thống cắt top-k **trước**, rồi mới chọn candidate theo question type. Nếu top-k ban đầu bị các relation phổ biến chiếm hết, qtype hiện tại có thể không còn row phù hợp để dựng candidate, dù subgraph thật ra có chứa tri thức đúng.

---

## 2. Mục tiêu của bản revise

Bản sửa nhắm tới 3 mục tiêu:

1. **Giảm bias do tên món / image context trong retrieval query**.
2. **Lọc relation theo qtype trước khi top-k**.
3. **Cho 2-hop được rank trên full path text**, thay vì chỉ rank trên edge cuối.

---

## 3. Thay đổi trong `query.py`

## 3.1. Chiến lược mới: `Neo -> Traverse -> Prefilter -> Rank`

Header của `query.py` đã được cập nhật từ:

```text
Neo -> Traverse -> Rank
```

thành:

```text
Neo -> Traverse -> Prefilter -> Rank
```

Ý nghĩa của từng bước:

1. **Neo**: anchor vào các `Dish` dựa trên `items`.
2. **Traverse**: lấy neighborhood cục bộ 1-hop / 2-hop quanh các dish đó.
3. **Prefilter**: nếu có `allowed_relations`, chỉ giữ những relation phục vụ qtype hiện tại.
4. **Rank**: embed query intent và so khớp với **full path text** của từng row.

### Điều quan trọng
Bản mới **vẫn không search toàn bộ KG**. Nó vẫn chỉ rank trên tập local rows quanh anchor dish. Điều thay đổi là cách lọc và cách rank tập local đó.

---

## 3.2. Bỏ phụ thuộc vào `r.embedding` trong Neo4j

### Trước đây
Cypher query chỉ lấy các cạnh có:

```cypher
r.embedding IS NOT NULL
```

Điều này khiến retriever phụ thuộc trực tiếp vào bước vector hóa edge trong Neo4j.

### Bây giờ
Điều kiện `r.embedding IS NOT NULL` đã được bỏ khỏi `_TRAVERSE_QUERY`.

Hệ quả:
- Retriever không còn yêu cầu edge embedding phải tồn tại trong Neo4j.
- Miễn là edge tồn tại trong graph, row đó vẫn có thể được retrieve và rank.
- `substitution_rules` không còn bị chặn chỉ vì chưa vectorize một số relation.

### Lợi ích
- Tăng coverage của retrieval.
- Giảm coupling giữa `query.py` và `05_kg_vectorizer.py`.
- Hữu ích cho các relation hiếm hoặc chưa được vectorize đầy đủ.

### Trade-off
- Bước rank phải embed `rank_text` tại runtime, nên nặng hơn một chút so với dùng sẵn `r.embedding`.
- Tuy nhiên chi phí vẫn chấp nhận được vì chỉ rank trên local subgraph quanh anchor dish.

---

## 3.3. Traverse query trả về thêm metadata cho path

`_TRAVERSE_QUERY` hiện trả về thêm các field như:
- `via`
- `via_type`
- `hop`

Điều này giúp downstream code biết row đó là:
- 1-hop direct relation,
- 2-hop qua `Ingredient`,
- hay 2-hop qua `SubstitutionRule`.

Metadata này là nền tảng để dựng **full path text** và verbalize triple đúng ngữ nghĩa hơn.

---

## 3.4. Thêm mapping relation -> tiếng Việt để dựng path text

`query.py` có thêm `_RELATION_TO_VI`, ví dụ:

- `hasIngredient` -> `có thành phần`
- `hasAllergen` -> `có chất gây dị ứng`
- `hasDietaryTag` -> `mang nhãn chế độ ăn`
- `fromIngredient` -> `thay nguyên liệu gốc`
- `toIngredient` -> `bằng nguyên liệu`

Mapping này được dùng để tạo `rank_text` dễ hiểu và ổn định ngữ nghĩa hơn.

---

## 3.5. Thay đổi quan trọng nhất: rank theo **full path text**

### Trước đây
Mỗi row được rank bằng:

```text
cosine(query_vec, edge_embedding)
```

Trong đó `edge_embedding` chỉ đại diện cho **một cạnh**.

### Bây giờ
Mỗi row được chuyển thành một chuỗi `rank_text`, rồi embed chuỗi đó tại runtime.

Ví dụ:

#### 1-hop
```text
Bánh bèo thuộc loại món Món khai vị
```

#### 2-hop qua Ingredient
```text
Phở bò có thành phần thịt bò; thịt bò có chất gây dị ứng Gluten
```

#### 2-hop qua SubstitutionRule
```text
Bún riêu có quy tắc thay thế Quy tắc A; Quy tắc A thay nguyên liệu gốc mắm tôm
```

### Vì sao thay đổi này quan trọng?
Nó giải quyết đúng gốc vấn đề 2-hop bị lép vế:
- `rank_text` của 2-hop giờ chứa cả `Dish`, `via`, `relation`, `target`.
- Query intent vì thế có thể match với **cả path hoàn chỉnh**, không chỉ với cạnh cuối.
- 2-hop không còn bị bất lợi chỉ vì edge cuối thiếu tên món.

---

## 3.6. Thêm cache embedding cho `rank_text`

`query.py` có thêm `self._text_embedding_cache` và `_embed_many()`.

Mục đích:
- tránh encode lặp lại cùng một `rank_text` nhiều lần,
- giảm chi phí runtime khi một path text xuất hiện lặp lại qua nhiều lần retrieve.

Đây là tối ưu phụ trợ cho cách rank mới.

---

## 3.7. `retrieve()` nhận thêm `allowed_relations`

Signature mới của `retrieve()` là:

```python
retrieve(items, question, top_k=5, allowed_relations=None)
```

Nếu truyền `allowed_relations`, retriever sẽ:

1. Traverse local rows quanh `items`.
2. **Prefilter** chỉ giữ các row có `relation` nằm trong whitelist.
3. Mới rank và cắt `top_k`.

### Đây là thay đổi rất quan trọng
Vì nó biến logic từ:

```text
rank global -> top_k -> mới lọc qtype
```

thành:

```text
lọc theo qtype trước -> rank trong tập liên quan -> top_k
```

Nhờ vậy, các relation 2-hop của qtype hiện tại không còn bị những relation phổ biến nhưng không liên quan lấn át.

---

## 3.8. `print_results()` hiển thị thêm `rank_text`

CLI output của `query.py` hiện có thêm trường:
- `rank_text`

Điều này giúp debug retrieval dễ hơn vì teammate có thể nhìn thấy **chuỗi nào đã được dùng để rank**, thay vì chỉ thấy raw relation hoặc `verbalized_text` từ Neo4j.

---

## 4. Thay đổi trong `06_generate_vqa.py`

Các thay đổi ở `06_generate_vqa.py` giúp **tận dụng** logic retrieve mới trong `query.py`.

## 4.1. Rút gọn retrieval query

### Trước đây
Retrieval query được build từ rất nhiều thành phần, gồm cả:
- anchor dish,
- food items trong ảnh,
- image description,
- qtype,
- detail,
- path.

### Bây giờ
`build_retrieval_query(qmeta)` chỉ còn tập trung vào **retrieval intent**:
- `canonical_qtype`
- `keywords`
- `detail_description`
- `relationship_sequence`

Ví dụ dạng mới:

```text
Loại câu hỏi: dietary_restrictions.
Từ khóa: vegan, plant-based.
Mục tiêu truy xuất: Xác định nhãn chế độ ăn liên quan đến món.
Chuỗi quan hệ ưu tiên: hasDietaryTag.
```

### Lý do
Tên món đã được dùng ở bước anchor/traverse trong `query.py`, nên không cần nhúng thêm vào query vector nữa. Việc rút gọn retrieval query giúp giảm bias về dish name và tăng trọng số cho **ý định quan hệ** của qtype.

---

## 4.2. Tạo relation whitelist theo qtype

`06_generate_vqa.py` thêm hàm `get_retrieval_relations(qmeta)`.

Logic hiện tại:
- phần lớn qtype sẽ lấy `primary_relation`,
- riêng `substitution_rules` dùng whitelist `['fromIngredient', 'toIngredient']`.

Whitelist này được truyền vào:

```python
kg.retrieve(..., allowed_relations=retrieval_relations)
```

### Hệ quả
Retriever chỉ rank những row thực sự liên quan đến qtype hiện tại.

Ví dụ:
- `ingredients` chỉ rank trên `hasIngredient`
- `allergen_restrictions` chỉ rank trên `hasAllergen`
- `dietary_restrictions` chỉ rank trên `hasDietaryTag`
- `ingredient_category` chỉ rank trên `ingredientCategory`

Đây chính là thay đổi “prefilter trước top-k” ở tầng caller.

---

## 4.3. Tăng `top_k` riêng cho `substitution_rules`

Trong `generate_one_sample()`, nếu qtype là `substitution_rules` thì script tự nâng `retrieve_top_k` lên tối thiểu 12.

Mục đích:
- vì relation này thường hiếm hơn,
- cần nới không gian candidate một chút để tăng khả năng dựng đáp án đúng.

Đây là heuristic bổ sung, không phải thay đổi nền tảng, nhưng khá hữu ích trong thực tế.

---

## 4.4. Candidate 2-hop được biểu diễn đúng ngữ nghĩa hơn

Một vấn đề phụ của logic cũ là triple 2-hop có thể bị truyền cho Gemini ở dạng chưa thể hiện rõ node trung gian `via`.

Trong bản mới:
- `two_hop_candidates()` đã dựng candidate dưới dạng **hai triple nối tiếp**:
  1. `Dish -> hasIngredient -> via`
  2. `via -> relation -> target`

Ví dụ:

```text
Phở bò --hasIngredient--> thịt bò
thịt bò --hasAllergen--> Gluten
```

Điều này quan trọng vì LLM cần nhìn thấy path reasoning đầy đủ, thay vì chỉ một row rút gọn.

---

## 4.5. Prompt builder dùng `candidate["triples"]` làm KG facts chính

`build_indifoodvqa_prompt()` hiện format phần `kg_triples` từ `candidate["triples"]`.

Điều này giúp đảm bảo:
- facts đưa vào prompt bám đúng candidate đã chọn,
- đặc biệt với 2-hop thì path reasoning được giữ nguyên,
- tránh việc LLM thấy raw row không đủ ngữ nghĩa.

Ngoài ra, script vẫn giữ `retrieved_facts` như metadata debug, bao gồm cả `rank_text`, `score`, `hop`, `evidence`, `source_url`.

---

## 4.6. Bỏ hard dependency vào edge embedding trong Neo4j

Module docstring của `06_generate_vqa.py` đã được cập nhật:
- không còn nói script phụ thuộc vào edge embeddings trong Neo4j,
- thay vào đó nhấn mạnh rằng retrieval giờ dùng **relation-aware local retrieval** và **runtime ranking trên full path text**.

Điều này đồng bộ với `query.py` bản mới.

---

## 5. Tóm tắt thay đổi theo “3 mức” đã đề xuất

## Mức 1 — Rút gọn retrieval query

### Đã làm
- Bỏ dish name, `food_items`, `image_description` khỏi retrieval query embedding.
- Chỉ giữ `qtype + keywords + detail + relationship_sequence`.

### Mục đích
Giảm bias do tên món và bối cảnh ảnh; để query vector tập trung vào **ý định quan hệ**.

---

## Mức 2 — Prefilter theo qtype trước khi top-k

### Đã làm
- `query.py.retrieve()` nhận `allowed_relations`.
- `06_generate_vqa.py` sinh relation whitelist từ qtype và truyền vào retriever.

### Mục đích
Không để các relation phổ biến nhưng không liên quan chiếm hết top-k của qtype hiện tại.

---

## Mức 3 — Rank theo full path text

### Đã làm
- Bỏ rank bằng `r.embedding` của cạnh đơn lẻ.
- Dựng `rank_text` từ full path rồi embed/rank tại runtime.

### Mục đích
Đảm bảo 2-hop được chấm điểm công bằng hơn vì path text giờ chứa đủ `Dish + via + relation + target`.

---

## 6. Những gì **không** thay đổi

Một số điểm cốt lõi vẫn giữ nguyên:

- Vẫn anchor retrieval vào `Dish` từ ảnh.
- Vẫn chỉ traverse subgraph cục bộ quanh anchor dish.
- Vẫn dùng `multilingual-e5-small` làm embedding model.
- Vẫn để `select_candidates()` làm bước dựng answer candidate sau retrieve.
- Vẫn dùng Gemini để sinh câu hỏi cuối cùng từ candidate đã grounded.

Nói cách khác, pipeline tổng thể không bị thay kiến trúc; phần thay đổi chủ yếu nằm ở **cách rank và lọc các local rows**.

---

## 7. Lợi ích kỳ vọng

Bản revise này kỳ vọng cải thiện các điểm sau:

1. **Tăng recall cho qtype 2-hop**
   - `allergen_restrictions`
   - `dietary_restrictions`
   - `ingredient_category`
   - một phần của `substitution_rules`

2. **Giảm việc top-k bị đầy bởi `hasIngredient`**
   nhất là khi `top_k` nhỏ.

3. **Tăng tính đúng ngữ nghĩa của candidate**
   vì path 2-hop giờ được biểu diễn đầy đủ hơn.

4. **Giảm phụ thuộc vào vectorization state của Neo4j**
   nên retrieval ổn định hơn khi KG chưa được vectorize hoàn chỉnh.

---

## 8. Trade-offs và điểm cần lưu ý

Dù logic mới tốt hơn, vẫn có vài trade-off cần lưu ý:

### 8.1. Runtime nặng hơn một chút
Do phải embed `rank_text` tại runtime thay vì đọc sẵn `r.embedding`.

### 8.2. Chất lượng vẫn phụ thuộc vào ontology/qtype mapping
Nếu `primary_relation` hoặc `relationship_sequence` trong `question_types.csv` sai hoặc chưa nhất quán, prefilter có thể lọc quá hẹp.

### 8.3. Relation whitelist hiện còn tương đối chặt
Trong một số qtype phức tạp, có thể cần mở whitelist nhiều hơn 1 relation nếu muốn tăng coverage.

Ví dụ tương lai có thể cân nhắc:
- `food_pairings` dùng thêm relation phụ nào đó,
- `dish_classification` kết hợp nhiều relation nếu ontology mở rộng.

### 8.4. `rank_text` hiện đang được verbalize chủ yếu bằng tiếng Việt
Đây là chủ đích để match tốt hơn với retrieval intent tiếng Việt. Tuy nhiên nếu KG có nhiều label tiếng Anh lẫn tiếng Việt, nên tiếp tục chuẩn hóa verbalization để ổn định hơn nữa.

---

## 9. Cách hiểu nhanh cho teammate

Nếu cần giải thích ngắn gọn trong một câu:

> Trước đây hệ thống rank các edge local bằng một query hơi loãng và cắt top-k quá sớm; bây giờ hệ thống vẫn lấy local subgraph quanh món, nhưng sẽ lọc theo relation của qtype trước, rồi rank trên full path text để 2-hop không bị lép vế.

Hoặc dạng cực ngắn:

```text
Cũ: anchor dish -> lấy local edges -> rank edge -> top-k -> mới chọn qtype
Mới: anchor dish -> lấy local paths -> lọc relation theo qtype -> rank full path -> top-k
```

---

## 10. Kết luận

Bản revise không thay đổi toàn bộ pipeline sinh VQA, nhưng thay đổi đúng phần gây nghẽn chất lượng retrieval:

- query embedding bớt bị dish-heavy,
- top-k không còn bị relation phổ biến nhưng sai mục tiêu chiếm hết,
- 2-hop được biểu diễn và chấm điểm đúng bản chất hơn.

Đây là một thay đổi **có tính cơ chế**, không chỉ là tuning thông số. Vì vậy nó đặc biệt quan trọng với các question type phụ thuộc reasoning qua ingredient hoặc qua substitution path.
