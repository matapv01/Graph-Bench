import argparse
import json
import os
import sys
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_openai import ChatOpenAI
from transformers import AutoTokenizer


def log(msg):
    print(f"[INFO] {msg}", flush=True)


def load_corpus(corpus_path, tokenizer): # , soft_limit=480
    log(f"Loading corpus from {corpus_path} ...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for i, item in enumerate(data):
        text = item.get("text", item.get("context", ""))
        if text.strip():
            tokens = tokenizer.encode(text, truncation=True) # , max_length=soft_limit
            truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)
            docs.append(Document(text=truncated_text))
        else:
            log(f"⚠️  Skipped empty doc at index {i}")
    log(f"Loaded {len(docs)} documents")
    return docs


def load_questions(questions_path, tokenizer): # , soft_limit=480
    log(f"Loading questions from {questions_path} ...")
    with open(questions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = []
    for i, q in enumerate(data):
        query = q.get("question", "").strip()
        if query:
            tokens = tokenizer.encode(query, truncation=True) # , max_length=soft_limit
            truncated_query = tokenizer.decode(tokens, skip_special_tokens=True)

            # copy toàn bộ keys, chỉ override question + đổi "answer" -> "ground_truth"
            q_obj = {**q, "question": truncated_query}
            if "answer" in q_obj:
                q_obj["ground_truth"] = q_obj.pop("answer")

            questions.append(q_obj)
        else:
            log(f"⚠️  Skipped empty question at index {i}")

    log(f"Loaded {len(questions)} questions")
    return questions




def build_llm(args):
    log("Initializing LLM client ...")
    return ChatOpenAI(
        model=args.model_name,
        api_key=args.llm_api_key,
        base_url=args.llm_base_url,
        temperature=0
        # max_tokens=512   # Hard limit
    )


def build_embed_model(embed_model_name):
    log(f"Loading embedding model: {embed_model_name}")
    return HuggingFaceEmbedding(model_name=embed_model_name)


def main(args):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Load data
    docs = load_corpus(args.corpus, tokenizer)
    questions = load_questions(args.questions, tokenizer)

    # Build LLM + Embedding
    embed_model = build_embed_model(args.embed_model)
    llm = build_llm(args)

    Settings.llm = llm
    Settings.embed_model = embed_model

    # Build Index
    log("Building index ...")
    index = VectorStoreIndex.from_documents(docs)
    query_engine = index.as_query_engine()
    log("Index built successfully")

    # Query loop
    stopIndex = 100000
    preds = []
    for i, q_obj in enumerate(questions, 1):
        stopIndex -= 1
        qid = q_obj.get("id", f"Q{i}")
        question = q_obj["question"]

        log(f"[{i}/{len(questions)}] Querying: {question}")
        # print(q_obj)

        try:
            # Query LlamaIndex
            response = query_engine.query(question, similarity_top_k=5)

            # Sinh câu trả lời
            generated_answer = str(response).strip() if response else ""

            # Lấy context từ source_nodes (LightRAG cũng join các chunk text lại)
            retrieved_contexts = []
            if hasattr(response, "source_nodes"):
                for node in response.source_nodes:
                    retrieved_contexts.append(node.node.get_content())
            context = " ".join(retrieved_contexts)

            # Giữ nguyên toàn bộ keys từ câu hỏi gốc
            pred = {
                **q_obj,
                "context": context,                 # override context
                "generated_answer": generated_answer  # override/generated
            }

            preds.append(pred)
            log(f"[SUCCESS] {qid}")

        except Exception as e:
            log(f"[ERROR] {qid}: {e}")
            pred = {
                **q_obj,
                "context": "",
                "generated_answer": None,
                "error": str(e)
            }
            preds.append(pred)


        if stopIndex == 0:
            log("Stopping early for testing purposes.")
            break


    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)

    log(f"✅ Finished! Saved results to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--questions", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--embed_model", type=str, required=True)
    parser.add_argument("--llm_base_url", type=str, required=True)
    parser.add_argument("--llm_api_key", type=str, required=True)
    args = parser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        log("❌ Interrupted by user")
        sys.exit(1)
