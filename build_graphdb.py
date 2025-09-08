# build_graphdb.py
import os
import json
import asyncio
import logging
import argparse

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "LightRAG"))

from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc
from lightrag.llm.hf import hf_embed
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

async def build_graphdb(subset, base_dir, embed_model_name):
    SUBSET_PATHS = {
        "medical": "./Datasets/Corpus/medical.json",
        "novel": "./Datasets/Corpus/novel.json"
    }

    corpus_path = SUBSET_PATHS[subset]
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
    embed_model = AutoModel.from_pretrained(embed_model_name)
    embedding_func = EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=lambda texts: hf_embed(texts, tokenizer, embed_model),
    )

    rag = LightRAG(
        working_dir=os.path.join(base_dir, subset),
        embedding_func=embedding_func,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    for item in corpus_data:
        corpus_name = item["corpus_name"]
        context = item["context"]
        rag.insert(context)
        logging.info(f"✅ Indexed {corpus_name}")

    logging.info(f"💾 Graph DB for {subset} saved at {os.path.join(base_dir, subset)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True, choices=["medical", "novel"])
    parser.add_argument("--base_dir", default="./graphdb")
    parser.add_argument("--embed_model", default="BAAI/bge-large-en-v1.5")
    args = parser.parse_args()

    os.makedirs(args.base_dir, exist_ok=True)
    asyncio.run(build_graphdb(args.subset, args.base_dir, args.embed_model))
