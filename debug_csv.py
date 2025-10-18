import pandas as pd

# df = pd.read_csv('C:\\Users\\pvmkt\\OneDrive\\Desktop\\GraphRAG-Benchmark\\analysis_rag_vs_lightrag.csv')

# # df = pd.read_json('C:\\Users\\pvmkt\\OneDrive\\Desktop\\GraphRAG-Benchmark\\results\\rag\\medical_preds_backup.json')
# # df = pd.read_json('C:\\Users\\pvmkt\\OneDrive\\Desktop\\GraphRAG-Benchmark\\results\\lightrag\\Medical\\predictions_Medical_backup.json')
# # # keep column: context
# # df = df[['context']]
# # print(df.head(5))
# # print(df.columns)
# # print(df.info())

import numpy as np

# # 🟢 1. Xóa khoảng trắng CUỐI chuỗi, giữ nguyên khoảng trắng GIỮA từ
# df = df.apply(lambda col: col.str.rstrip() if col.dtype == "object" else col)

# # 🟢 2. Chuyển các giá trị dạng chuỗi “NaN”/“NULL”/“nan”/chuỗi rỗng => NaN thật
# df = df.replace(
#     [r'^\s*$',  # chuỗi toàn khoảng trắng
#      r'(?i)^nan$',  # NaN hoặc nan (không phân biệt hoa thường)
#      r'(?i)^null$'  # NULL hoặc null
#     ],
#     np.nan,
#     regex=True
# )

# print("\n✅ Sau khi chuẩn hóa NaN:")
# print(df)

# print(df.columns)

# # 🟢 3. (Tuỳ chọn) Đếm số lượng NaN thật sự
# print("\n🔹 Số NaN theo từng cột:")
# print(df.isna().sum())

# print("\n🔹 Tổng số NaN trong DataFrame:")
# print(df.isna().sum().sum())

# print("\n🔹 So luong ban ghi nan ca cot envidence ca cot context_lightrag khong nan:")
# print(df[df['evidence'].isna() & ~df['context_lightrag'].isna()].shape[0])

# print("\n🔹 ban ghi nan ca cot envidence ca cot context_lightrag thuoc kieu question nao:")
# print(df[df['evidence'].isna() & ~df['context_lightrag'].isna()]['question_type'].value_counts())

# df1 = pd.read_json('C:\\Users\\pvmkt\\OneDrive\\Desktop\\GraphRAG-Benchmark\\results\\rag\\medical_preds_backup.json')
# df2 = pd.read_json('C:\\Users\\pvmkt\\OneDrive\\Desktop\\GraphRAG-Benchmark\\results\\lightrag\\Medical\\predictions_Medical_backup.json')


# # 🟢 2. Lấy danh sách id cần giữ
# ids_keep = df[df['evidence'].isna() & ~df['context_lightrag'].isna()]['id'].unique()
# print(f"🔎 Số lượng ID cần giữ: {len(ids_keep)}")

# # 🟢 3. Lọc df1 và df2 chỉ giữ các id này
# df1_filtered = df1[df1['id'].isin(ids_keep)]
# df2_filtered = df2[df2['id'].isin(ids_keep)]

# print(f"✅ df1_filtered: {len(df1_filtered)} bản ghi")
# print(f"✅ df2_filtered: {len(df2_filtered)} bản ghi")

# # 🟢 4. Lưu lại dưới dạng JSON (mảng các dict như file gốc)
# out1 = r"C:\\Users\\pvmkt\\OneDrive\\Desktop\\GraphRAG-Benchmark\\results\\rag\\debug_medical_preds_filtered.json"
# out2 = r"C:\\Users\\pvmkt\\OneDrive\\Desktop\\GraphRAG-Benchmark\\results\\lightrag\\Medical\\debug_predictions_Medical_filtered.json"

# # orient='records' => mảng các dict
# df1_filtered.to_json(out1, orient='records', force_ascii=False, indent=2)
# df2_filtered.to_json(out2, orient='records', force_ascii=False, indent=2)

# print(f"\n📂 Đã lưu JSON mới:\n- {out1}\n- {out2}")




# #------------------------------------------------------------------------------------------------------------

# import pandas as pd
# import numpy as np

# df = pd.read_csv('C:\\Users\\pvmkt\\OneDrive\\Desktop\\GraphRAG-Benchmark\\analysis_rag_vs_lightrag.csv')

# # 🟢 1. Xóa khoảng trắng CUỐI chuỗi
# df = df.apply(lambda col: col.str.rstrip() if col.dtype == "object" else col)

# # 🟢 2. Chuẩn hóa chuỗi rỗng, NaN, NULL thành NaN thật
# df = df.replace(
#     [r'^\s*$', r'(?i)^nan$', r'(?i)^null$'],
#     np.nan,
#     regex=True
# )

# print("\n✅ Sau khi chuẩn hóa NaN:")
# print(df.head())
# print(df.columns)

# # 🟢 3. Đếm NaN
# print("\n🔹 Số NaN theo từng cột:")
# print(df.isna().sum())

# print("\n🔹 Tổng số NaN trong DataFrame:")
# print(df.isna().sum().sum())

# # 🟢 4. Điều kiện mới: evidence NaN, context_lightrag NaN, context_rag không NaN
# mask = df['evidence'].isna() & df['context_lightrag'].isna() & ~df['context_rag'].isna()

# print("\n🔹 Số bản ghi thỏa điều kiện mới:")
# print(df[mask].shape[0])

# print("\n🔹 question_type của các bản ghi này:")
# print(df[mask]['question_type'].value_counts())

# # Đọc 2 file JSON gốc
# df1 = pd.read_json('C:\\Users\\pvmkt\\OneDrive\\Desktop\\GraphRAG-Benchmark\\results\\rag\\medical_preds_backup.json')
# df2 = pd.read_json('C:\\Users\\pvmkt\\OneDrive\\Desktop\\GraphRAG-Benchmark\\results\\lightrag\\Medical\\predictions_Medical_backup.json')

# # 🟢 5. Lấy danh sách id cần giữ
# ids_keep = df[mask]['id'].unique()
# print(f"🔎 Số lượng ID cần giữ: {len(ids_keep)}")

# # 🟢 6. Lọc df1 và df2 theo ids
# df1_filtered = df1[df1['id'].isin(ids_keep)]
# df2_filtered = df2[df2['id'].isin(ids_keep)]

# print(f"✅ df1_filtered: {len(df1_filtered)} bản ghi")
# print(f"✅ df2_filtered: {len(df2_filtered)} bản ghi")

# # 🟢 7. Lưu kết quả JSON
# out1 = r"C:\\Users\\pvmkt\\OneDrive\\Desktop\\GraphRAG-Benchmark\\results\\rag\\debug_medical_preds_filtered_2.json"
# out2 = r"C:\\Users\\pvmkt\\OneDrive\\Desktop\\GraphRAG-Benchmark\\results\\lightrag\\Medical\\debug_predictions_Medical_filtered_2.json"

# df1_filtered.to_json(out1, orient='records', force_ascii=False, indent=2)
# df2_filtered.to_json(out2, orient='records', force_ascii=False, indent=2)

# print(f"\n📂 Đã lưu JSON mới:\n- {out1}\n- {out2}")



import pandas as pd
import numpy as np

df = pd.read_csv('C:\\Users\\pvmkt\\OneDrive\\Desktop\\GraphRAG-Benchmark\\analysis_rag_vs_lightrag.csv')

# 🟢 1. Xóa khoảng trắng CUỐI chuỗi
df = df.apply(lambda col: col.str.rstrip() if col.dtype == "object" else col)

# 🟢 2. Chuẩn hóa chuỗi rỗng, NaN, NULL thành NaN thật
df = df.replace(
    [r'^\s*$', r'(?i)^nan$', r'(?i)^null$'],
    np.nan,
    regex=True
)

print("\n✅ Sau khi chuẩn hóa NaN:")
print(df.head())
print(df.columns)

# 🟢 3. Đếm NaN
print("\n🔹 Số NaN theo từng cột:")
print(df.isna().sum())

print("\n🔹 Tổng số NaN trong DataFrame:")
print(df.isna().sum().sum())

# 🟢 4. Điều kiện mới: evidence NaN, context_lightrag NaN, context_rag không NaN
mask = df['evidence'].isna() & df['context_lightrag'].isna() & ~df['context_rag'].isna()

print("\n🔹 Số bản ghi thỏa điều kiện mới:")
print(df[mask].shape[0])

print("\n🔹 question_type của các bản ghi này:")
print(df[mask]['question_type'].value_counts())

# Đọc 2 file JSON gốc
df1 = pd.read_json('C:\\Users\\pvmkt\\OneDrive\\Desktop\\GraphRAG-Benchmark\\Datasets\\Questions\\medical_questions.json')

# 🟢 5. Lấy danh sách id cần giữ
ids_keep = df[mask]['id'].unique()
print(f"🔎 Số lượng ID cần giữ: {len(ids_keep)}")

# 🟢 6. Lọc df1 và df2 theo ids
df1_filtered = df1[df1['id'].isin(ids_keep)]

print(f"✅ df1_filtered: {len(df1_filtered)} bản ghi")

# 🟢 7. Lưu kết quả JSON
out1 = r"C:\\Users\\pvmkt\\OneDrive\\Desktop\\GraphRAG-Benchmark\\Datasets\\Questions\\medical_questions_filtered.json"

df1_filtered.to_json(out1, orient='records', force_ascii=False, indent=2)

print(f"\n📂 Đã lưu JSON mới:\n- {out1}")