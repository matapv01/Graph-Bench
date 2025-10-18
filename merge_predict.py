import pandas as pd
import numpy as np

# --- 1. Đường dẫn file ---
file_paths = {
    'rag': r"C:\Users\pvmkt\OneDrive\Desktop\GraphRAG-Benchmark\results\rag\medical_preds_backup.json",
    'lightrag': r"C:\Users\pvmkt\OneDrive\Desktop\GraphRAG-Benchmark\results\lightrag\Medical\predictions_Medical_backup.json"
}

data_frames = {}

# --- 2. Đọc & chuẩn hóa từng file ---
for model, path in file_paths.items():
    df = pd.read_json(path)

    # 🟢 1. Xóa khoảng trắng CUỐI chuỗi (nhưng giữ nguyên khoảng trắng GIỮA từ)
    df = df.apply(lambda col: col.str.rstrip() if col.dtype == "object" else col)

    # 🟢 2. Chuyển các giá trị dạng chuỗi “NaN”/“NULL”/“nan”/chuỗi rỗng => NaN thật
    df = df.replace(
        [
            r'^\s*$',      # chuỗi toàn khoảng trắng
            r'(?i)^nan$',  # NaN hoặc nan (không phân biệt hoa thường)
            r'(?i)^null$'  # NULL hoặc null
        ],
        np.nan,
        regex=True
    )

    # 🟢 3. Giữ lại cột cần thiết
    df = df[['id', 'question', 'question_type', 'evidence', 'ground_truth',
             'generated_answer', 'context']]

    # 🟢 4. Đổi tên answer/context để phân biệt 2 model
    df = df.rename(columns={
        'generated_answer': f'answer_{model}',
        'context': f'context_{model}'
    })

    # 👉 Với lightrag: chỉ giữ id, answer và context để tránh trùng cột
    if model == 'lightrag':
        df = df[['id', f'answer_{model}', f'context_{model}']]

    data_frames[model] = df

# --- 3. Đếm NaN trong cột evidence của RAG trước khi merge ---
nan_before = data_frames['rag']['evidence'].isna().sum()
print(f"🔎 Số lượng evidence bị NaN TRƯỚC khi merge: {nan_before}")

# --- 4. Merge theo id (rag làm gốc) ---
merged_df = pd.merge(
    data_frames['rag'],
    data_frames['lightrag'],
    on='id',
    how='left'
)

# --- 5. Đếm NaN trong cột evidence sau khi merge ---
nan_after = merged_df['evidence'].isna().sum()
print(f"🔎 Số lượng evidence bị NaN SAU khi merge: {nan_after}")

# --- 6. Sắp xếp lại thứ tự cột ---
merged_df = merged_df[[
    'id',
    'question',
    'question_type',
    'evidence',
    'ground_truth',
    'answer_rag',
    'context_rag',
    'answer_lightrag',
    'context_lightrag'
]]

# --- 7. Xuất kết quả ra CSV ---
output_csv = "analysis_rag_vs_lightrag.csv"
merged_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

print("\n✅ DataFrame đã merge & chuẩn hóa:")
print(merged_df.head())
print(f"\n📂 Đã lưu tại: {output_csv}")
