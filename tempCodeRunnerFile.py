import pandas as pd

df = pd.read_csv('C:\\Users\\pvmkt\\OneDrive\\Desktop\\GraphRAG-Benchmark\\analysis_rag_vs_lightrag.csv')

# df = pd.read_json('C:\\Users\\pvmkt\\OneDrive\\Desktop\\GraphRAG-Benchmark\\results\\rag\\medical_preds.json')
# # keep column: context
# df = df[['context']]
# print(df.head(5))
# print(df.columns)
# print(df.info())

import numpy as np

# 🟢 1. Xóa khoảng trắng CUỐI chuỗi, giữ nguyên khoảng trắng GIỮA từ
df = df.apply(lambda col: col.str.rstrip() if col.dtype == "object" else col)

# 🟢 2. Chuyển các giá trị dạng chuỗi “NaN”/“NULL”/“nan”/chuỗi rỗng => NaN thật
df = df.replace(
    [r'^\s*$',  # chuỗi toàn khoảng trắng
     r'(?i)^nan$',  # NaN hoặc nan (không phân biệt hoa thường)
     r'(?i)^null$'  # NULL hoặc null
    ],
    np.nan,
    regex=True
)

print("\n✅ Sau khi chuẩn hóa NaN:")
print(df)

# 🟢 3. (Tuỳ chọn) Đếm số lượng NaN thật sự
print("\n🔹 Số NaN theo từng cột:")
print(df.isna().sum())

print("\n🔹 Tổng số NaN trong DataFrame:")
print(df.isna().sum().sum())