# run_inference.py
import os
import json
import argparse
import asyncio
import logging
from tqdm import tqdm

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "LightRAG"))

from lightrag import LightRAG, QueryParam

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

SYSTEM_PROMPT = """
You are a helpful assistant answering strictly based on the Knowledge Base.
If unknown, reply with "I don't know".
"""

async def run_inference(subset, base_dir, model_name, retrieve_topk, output_dir):
    SUBSET_PATHS = {
        "medical": "./Datasets/Questions/medical_questions.json",
        "novel": "./Datasets/Questions/novel_questions.json"
    }

    with open(SUBSET_PATHS[subset], "r", encoding="utf-8") as f:
        questions = json.load(f)

    rag = LightRAG(
        working_dir=os.path.join(base_dir, subset),
        llm_model_name=model_name,
    )

    results = []
    for q in tqdm(questions, desc=f"Answering {subset}"):
        query_param = QueryParam(
            mode="hybrid",
            top_k=retrieve_topk,
            chunk_top_k=5,
            max_total_tokens=2048,
            response_type="Multiple Paragraphs",
            stream=False,
        )
        response, context = rag.query(q["question"], param=query_param, system_prompt=SYSTEM_PROMPT)
        if asyncio.iscoroutine(response):
            response = await response
        results.append({
            "id": q["id"],
            "question": q["question"],
            "generated_answer": str(response),
            "ground_truth": q.get("answer"),
            "source": subset
        })

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"predictions_{subset}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info(f"💾 Saved predictions to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True, choices=["medical", "novel"])
    parser.add_argument("--base_dir", default="./graphdb")
    parser.add_argument("--model_name", default="HuggingFaceH4/zephyr-7b-beta")
    parser.add_argument("--retrieve_topk", type=int, default=5)
    parser.add_argument("--output_dir", default="./results/lightrag")
    args = parser.parse_args()

    asyncio.run(run_inference(args.subset, args.base_dir, args.model_name, args.retrieve_topk, args.output_dir))
