import json

def migrate_answer_to_ground_truth(json_path, output_path=None):
    """
    Đổi key 'answer' thành 'ground_truth' trong file JSON chứa mảng dict.
    Nếu output_path=None thì sẽ ghi đè lên file gốc.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("File JSON phải chứa một mảng (list) các dict")

    migrated = []
    for item in data:
        if "answer" in item:
            item["ground_truth"] = item.pop("answer")
        migrated.append(item)

    # nếu không truyền output_path thì ghi đè file gốc
    output_path = output_path or json_path
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(migrated, f, ensure_ascii=False, indent=2)

    print(f"✅ Đã migrate xong. Kết quả lưu tại: {output_path}")


migrate_answer_to_ground_truth("C:\\Users\\pvmkt\\OneDrive\\Desktop\\GraphRAG-Benchmark\\results\\llama-index\\medical_preds.json")              # ghi đè file gốc
# hoặc
# migrate_answer_to_ground_truth("questions.json", "new.json")  # lưu ra file mới
