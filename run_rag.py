# run_rag.py
import os
import json
import argparse
import asyncio
import logging
import nest_asyncio
from tqdm import tqdm

from lightrag import LightRAG, QueryParam
from lightrag.llm.hf import hf_embed
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from transformers import AutoModel, AutoTokenizer

nest_asyncio.apply()
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

SYSTEM_PROMPT = """
You are a helpful assistant.
Answer based strictly on the provided context.
If the answer is unknown, reply: "I don't know".
"""


async def my_llm_model_func(prompt: str, system_prompt: str = None, history_messages=None, **kwargs):
    """LLM interface for OpenAI-compatible APIs"""
    model_name = kwargs.get("model_name", "qwen2.5-14b-instruct")
    base_url = kwargs.get("base_url", "")
    api_key = kwargs.get("api_key", "")

    safe_kwargs = {k: v for k, v in kwargs.items() if k not in ["model_name", "base_url", "api_key"]}

    return await openai_complete_if_cache(
        model=model_name,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        base_url=base_url,
        api_key=api_key,
        **safe_kwargs
    )


async def build_rag(work_dir, mode, model_name, embed_model, llm_base_url, llm_api_key):
    """Khởi tạo LightRAG"""
    os.makedirs(work_dir, exist_ok=True)

    if mode == "API":
        tokenizer = AutoTokenizer.from_pretrained(embed_model)
        embed_model_obj = AutoModel.from_pretrained(embed_model)
        embedding_func = EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=lambda texts: hf_embed(texts, tokenizer, embed_model_obj),
        )
        llm_kwargs = {"model_name": model_name, "base_url": llm_base_url, "api_key": llm_api_key}
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    rag = LightRAG(
        working_dir=work_dir,
        llm_model_func=my_llm_model_func,
        llm_model_name=model_name,
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        embedding_func=embedding_func,
        llm_model_kwargs=llm_kwargs,
    )

    await rag.initialize_storages()
    # ✅ BẮT BUỘC: khởi tạo pipeline_status
    await initialize_pipeline_status()
    return rag

async def safe_aquery(rag, question, param, system_prompt):
    try:
        result = await rag.aquery(question, param=param, system_prompt=system_prompt)
        if isinstance(result, tuple):
            if len(result) == 2:
                response, ctx = result
            else:
                response, ctx = result[0], ""
        else:
            response, ctx = result, ""
    except UnboundLocalError:
        logging.warning("LightRAG returned no context, using empty string.")
        response, ctx = "I don't know", ""
    except Exception as e:
        logging.error(f"safe_aquery failed: {e}")
        response, ctx = "I don't know", ""

    # 🔎 Debug thêm
    logging.debug(f"safe_aquery: response={str(response)[:100]}..., ctx_len={len(ctx)}")
    return response, ctx



async def process(corpus_file, question_file, output_file, sample=None, **kwargs):
    # Load corpus
    with open(corpus_file, "r", encoding="utf-8") as f:
        corpus_data = json.load(f)
    logging.info(f"Loaded corpus with {len(corpus_data)} docs")

    # Load questions
    with open(question_file, "r", encoding="utf-8") as f:
        questions = json.load(f)
    logging.info(f"Loaded {len(questions)} questions")

    # Nếu có sample thì chỉ lấy bấy nhiêu câu hỏi
    if sample is not None and sample < len(questions):
        questions = questions[:sample]
        logging.info(f"Using {len(questions)} sampled questions")

    results = []

    for doc in corpus_data:
        corpus_name = doc.get("corpus_name", "default")
        context = doc.get("context", "")

        # 🔥 Ép context thành string an toàn
        if isinstance(context, dict):
            context = json.dumps(context, ensure_ascii=False)
        elif isinstance(context, list):
            context = "\n".join([
                json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else str(x)
                for x in context
            ])
        else:
            context = str(context)

        rag = await build_rag(
            work_dir=os.path.join(kwargs["base_dir"], corpus_name),
            mode=kwargs["mode"],
            model_name=kwargs["model_name"],
            embed_model=kwargs["embed_model"],
            llm_base_url=kwargs["llm_base_url"],
            llm_api_key=kwargs["llm_api_key"],
        )

        # Index corpus (luôn truyền text là string)
        await rag.ainsert([context], ids=[corpus_name])
        logging.info(f"✅ Indexed corpus: {corpus_name}")

        for q in tqdm(questions, desc=f"Answering for {corpus_name}"):
            query_param = QueryParam(mode="global", top_k=kwargs["topk"])

            response, ctx = await safe_aquery(
                rag, q["question"], param=query_param, system_prompt=SYSTEM_PROMPT
            )

            # Handle cả async lẫn sync responses
            if asyncio.iscoroutine(response):
                response = await response
            predicted_answer = str(response)

            results.append({
                "id": q["id"],
                "question": q["question"],
                "source": corpus_name,
                "context": context,
                "evidence": q["evidence"],
                "question_type": q["question_type"],
                "generated_answer": predicted_answer,
                "ground_truth": q.get("answer"),
            })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)  # ✅ tạo thư mục cha nếu chưa có
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logging.info(f"💾 Saved {len(results)} answers to {output_file}")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, help="Path to corpus JSON")
    parser.add_argument("--questions", required=True, help="Path to questions JSON")
    parser.add_argument("--output", default="./results/predictions.json", help="Where to save results")
    parser.add_argument("--base_dir", default="./workspace", help="Workspace directory")
    parser.add_argument("--mode", default="API", choices=["API"], help="LLM mode (default API)")
    parser.add_argument("--model_name", default="qwen2.5-14b-instruct")
    parser.add_argument("--embed_model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--llm_base_url", default="https://integrate.api.nvidia.com/v1")
    parser.add_argument("--llm_api_key", default=os.getenv("LLM_API_KEY", ""))
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--sample", type=int, default=None, help="Number of questions to sample")

    args = parser.parse_args()

    asyncio.run(
        process(
            corpus_file=args.corpus,
            question_file=args.questions,
            output_file=args.output,
            base_dir=args.base_dir,
            mode=args.mode,
            model_name=args.model_name,
            embed_model=args.embed_model,
            llm_base_url=args.llm_base_url,
            llm_api_key=args.llm_api_key,
            topk=args.topk,
            sample=args.sample,
        )
    )


if __name__ == "__main__":
    main()
