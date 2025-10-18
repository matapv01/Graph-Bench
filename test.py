import json
import pandas as pd

# # Đường dẫn file gốc và file mới
# input_file = "C:\\Users\\pvmkt\\OneDrive\\Desktop\\GraphRAG-Benchmark\\results\\rag\\medical_preds.json"
# output_file = "C:\\Users\\pvmkt\\OneDrive\\Desktop\\GraphRAG-Benchmark\\results\\rag\\filtered.json"

# # Đọc dữ liệu JSON gốc
# with open(input_file, "r", encoding="utf-8") as f:
#     data = json.load(f)

# # Lọc dữ liệu
# filtered_data = [
#     item for item in data
#     if item.get("question_type") in ["Creative Generation"]
# ]

# # Ghi dữ liệu đã lọc ra file mới
# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(filtered_data, f, ensure_ascii=False, indent=2)

# print(f"Đã lưu {len(filtered_data)} mục vào {output_file}")


df = pd.read_json('C:\\Users\\pvmkt\\OneDrive\\Desktop\\GraphRAG-Benchmark\\results\\rag\\debug_medical_preds_filtered.json')
# so luong ban ghi
print(f"Số lượng bản ghi: {len(df)}")