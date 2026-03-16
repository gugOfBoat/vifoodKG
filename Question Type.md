# Question Type

| Question Type  | Relationship | Keywords | Detail description |
| --- | --- | --- | --- |
| ingredients | hasIngredient | các nguyên liệu, hương vị và mùi thơm tổng thể của món ăn | nguyên liệu chính của món ăn là gì và vai trò của nó trong món ăn, và chúng đóng góp như thế nào vào hương vị và mùi thơm tổng thể của món ăn |
| side dish | servedWith | món ăn kèm, nước chấm dùng kèm món ăn | món ăn kèm / nước chấm đi kèm
món ăn thường được ăn kèm với món gì (nước chấm, rau thơ,…)? |
| origin | originRegion | địa phương, vùng miền, nguồn gốc | món có nguồn gốc ở đâu (tỉnh, thành, miền,…)? |
| type of dish | dishType | loại món ăn | đây là loại món ăn gì (món cơm, món nước, bánh, bún khô,…) |
| category of ingredient  | hasIngredient → ingredientCategory | loại nguyên liệu | nguyên liệu làm ra món ăn này thuộc loại nào (hải sản, thịt, rau,…) |
| allergens restrictions | hasIngredient → hasAllergen | dị ứng, nguyên liệu hoặc phương pháp chế biến thay thế để món ăn không chứa chất gây dị ứng | trong món ăn này có chất gây dị ứng nào không, và có những nguyên liệu hoặc phương pháp chế biến thay thế nào để làm cho thực phẩm không chứa chất gây dị ứng. |
| cooking technique |  | kỹ thuật nấu nướng, ảnh hưởng đến thời gian chuẩn bị, màu sắc, kết cấu và hương vị. | kỹ thuật nấu nướng này khác biệt như thế nào so với các món ăn tương tự khác, và nó ảnh hưởng ra sao đến thời gian chuẩn bị, màu sắc, kết cấu và hương vị của món ăn. Kỹ thuật nấu món này (chiên, xào, luộc,…) là gì? |
| taste and flavor profile | flavorProfile | hương vị và đặc điểm mùi vị, sự cân bằng giữa vị ngọt, vị mặn và vị cay. | làm thế nào những món ăn này tạo nên sự cân bằng giữa vị ngọt, vị mặn và vị cay, và sự đa dạng này làm tăng thêm trải nghiệm ẩm thực như thế nào. |
| health and nutritional aspects | hasIngredient | lợi ích sức khỏe và dinh dưỡng, hàm lượng protein, chất xơ, chất dinh dưỡng và khoáng chất. | so sánh lợi ích dinh dưỡng của các món ăn này với các món ăn tương tự khác, hãy nhấn mạnh hàm lượng protein, chất xơ và các chất dinh dưỡng, khoáng chất khác trong từng món ăn. |
| dietary restrictions | hasIngredient → hasDietary | chế độ ăn, nguyên liệu hoặc phương pháp chế biến thay thế để món ăn phù hợp với chế độ ăn | hỏi món có phù hợp với chế độ ăn cụ thể (chay, Địa Trung Hải,…) không dựa trên dietaryTag của ingredient (animal_product/plant_based). |
| ingredient substitutions | `Dish → hasSubRule → SubRule_X
SubRule_X → fromIngredient → Ingredient_1
SubRule_X → toIngredient → Ingredient_2` | nguyên liệu thay thế, tương tự | khả năng thay thế một số nguyên liệu trong món ăn bằng nguyên liệu khác, và ảnh hưởng của việc này đến kết cấu, hương vị và giá trị dinh dưỡng. |