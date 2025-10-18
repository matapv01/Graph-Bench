import os

import json
import argparse
import logging
from tqdm import tqdm
from HippoRAG.src.hipporag import HippoRAG




logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.info

def load_corpus(corpus_path):
    log(f"Loading corpus from {corpus_path} ...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    docs = []
    for i, item in enumerate(data):
        text = item.get("text", item.get("context", ""))
        if text.strip():
            docs.append(text.strip())
        else:
            logging.warning(f"Skipped empty doc at index {i}")
    log(f"Loaded {len(docs)} documents.")
    return docs

def load_questions(questions_path):
    log(f"Loading questions from {questions_path} ...")
    with open(questions_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    log(f"Loaded {len(data)} questions.")
    return data

def main(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # === Load corpus + questions ===
    corpus = load_corpus(args.corpus)
    questions = load_questions(args.questions)

    # === Setup environment for custom API endpoint ===
    if args.llm_api_key:
        os.environ["OPENAI_API_KEY"] = args.llm_api_key
    if args.llm_base_url:
        os.environ["OPENAI_BASE_URL"] = args.llm_base_url

    # === Initialize HippoRAG ===
    log(f"Initializing HippoRAG with LLM={args.llm_model_name}, embedding={args.embed_model_name}")
    hipporag = HippoRAG(
        save_dir=args.save_dir,
        llm_model_name=args.llm_model_name,
        embedding_model_name=args.embed_model_name
    )

    # === Index corpus ===
    log("Indexing documents ...")
    hipporag.index(docs=corpus)
    log("Indexing complete.")

    preds = []
    for i, q in enumerate(tqdm(questions, desc="Running HippoRAG QA")):
        qid = q.get("id", f"Q{i+1}")
        question = q.get("question", "").strip()
        ground_truth = q.get("ground_truth", q.get("answer", ""))
        source = q.get("source", "")
        q_type = q.get("question_type", "")
        evidence = q.get("evidence", [])

        if not question:
            logging.warning(f"Skipped empty question id={qid}")
            continue

        try:
            # Retrieval + QA
            response = hipporag.rag_qa(
                queries=[question],
                top_k=args.top_k  # ✅ thêm top_k để giới hạn số đoạn context
            )

            if isinstance(response, list) and len(response) > 0:
                ans_obj = response[0]
                generated_answer = ans_obj.get("generated_answer", ans_obj.get("answer", ""))
                retrieved_context = " ".join(ans_obj.get("retrieved_docs", []))
            else:
                generated_answer, retrieved_context = "", ""

            pred = {
                "id": qid,
                "source": source,
                "question": question,
                "context": retrieved_context,
                "evidence": evidence,
                "question_type": q_type,
                "generated_answer": generated_answer,
                "ground_truth": ground_truth
            }
            preds.append(pred)
            logging.info(f"[SUCCESS] {qid}")

        except Exception as e:
            logging.error(f"[ERROR] {qid}: {e}")
            preds.append({
                "id": qid,
                "source": source,
                "question": question,
                "context": "",
                "evidence": evidence,
                "question_type": q_type,
                "generated_answer": None,
                "ground_truth": ground_truth,
                "error": str(e)
            })

    # === Save results ===
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)
    log(f"✅ Saved predictions to {args.output}")

if __name__ == "__main__":

    import multiprocessing
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--questions", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./outputs/hipporag")
    parser.add_argument("--llm_model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--embed_model_name", type=str, default="text-embedding-3-small")
    parser.add_argument("--llm_base_url", type=str, default=None)
    parser.add_argument("--llm_api_key", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=5, help="Number of documents to retrieve per query")  # ✅
    args = parser.parse_args()

    main(args)
