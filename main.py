import argparse
import asyncio
import os
import sys
from functools import lru_cache


def ensure_project_venv() -> None:
    project_root = os.path.dirname(os.path.abspath(__file__))
    venv_root = os.path.join(project_root, ".venv")
    venv_python = os.path.join(project_root, ".venv", "bin", "python")

    if not os.path.exists(venv_python):
        return

    current_prefix = os.path.realpath(sys.prefix)
    expected_prefix = os.path.realpath(venv_root)

    if current_prefix != expected_prefix:
        os.execv(venv_python, [venv_python, *sys.argv])


ensure_project_venv()


import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc


load_dotenv()


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = os.getenv("ASKCHOMSKY_LLM_MODEL", "openai/gpt-4o-mini")
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_WORKING_DIR = "./lightrag_store"
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT", "600"))
MAX_ASYNC_LLM_CALLS = int(os.getenv("MAX_ASYNC", "2"))
MAX_PARALLEL_INSERT = int(os.getenv("MAX_PARALLEL_INSERT", "2"))


def get_langfuse_client():
    """Return a configured Langfuse client or None if unavailable/invalid."""
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip().strip('"').strip("'")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip().strip('"').strip("'")
    base_url = (
        os.getenv("LANGFUSE_BASE_URL", "").strip().strip('"').strip("'")
        or os.getenv("LANGFUSE_HOST", "").strip().strip('"').strip("'")
        or "https://cloud.langfuse.com"
    )

    if (
        not public_key
        or public_key.startswith("pk-lf-...")
        or not secret_key
        or secret_key.startswith("sk-lf-...")
    ):
        return None

    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            base_url=base_url,
            debug=os.getenv("LANGFUSE_DEBUG", "false").lower() == "true",
        )

        if not client.auth_check():
            print("Langfuse auth check failed. Verify keys and LANGFUSE_BASE_URL.")
            return None

        return client
    except Exception as exc:
        print(f"Langfuse disabled: {exc}")
        return None


def configure_langfuse() -> bool:
    """Backward-compatible bool helper used by older call sites."""
    return get_langfuse_client() is not None


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


def embed_texts(texts: list[str]) -> np.ndarray:
    embeddings = get_embedder().encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await asyncio.to_thread(embed_texts, texts)


async def llm_model_func(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    **kwargs,
) -> str:
    api_key = os.getenv("openrouter_key")
    if not api_key:
        raise ValueError("Missing openrouter_key in .env")

    if history_messages is None:
        history_messages = []

    return await openai_complete_if_cache(
        model=LLM_MODEL,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        timeout=LLM_TIMEOUT_SECONDS,
        keyword_extraction=keyword_extraction,
        **kwargs,
    )


async def initialize_rag(working_dir: str = DEFAULT_WORKING_DIR) -> LightRAG:
    os.makedirs(working_dir, exist_ok=True)

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        llm_model_name=LLM_MODEL,
        default_llm_timeout=LLM_TIMEOUT_SECONDS,
        llm_model_max_async=MAX_ASYNC_LLM_CALLS,
        max_parallel_insert=MAX_PARALLEL_INSERT,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            model_name=EMBED_MODEL,
            func=embedding_func,
        ),
    )

    await rag.initialize_storages()
    return rag


def load_corpus_texts(limit: int) -> list[str]:
    ds = load_dataset("mmoise00/chomsky-corpus", split="train")
    count = min(limit, len(ds))

    texts = []
    for row in ds.select(range(count)):
        title = row.get("article_title") or "Untitled"
        date = row.get("article_date") or ""
        content = row.get("content") or ""
        texts.append(f"Title: {title}\nDate: {date}\n\n{content}")

    return texts


async def ingest_corpus(doc_limit: int = 200, working_dir: str = DEFAULT_WORKING_DIR) -> int:
    rag = None
    try:
        rag = await initialize_rag(working_dir)
        docs = load_corpus_texts(doc_limit)
        await rag.ainsert(docs)
        return len(docs)
    finally:
        if rag is not None:
            await rag.finalize_storages()


async def query_rag(
    question: str,
    mode: str = "hybrid",
    working_dir: str = DEFAULT_WORKING_DIR,
) -> str:
    def _looks_like_no_answer(answer: str) -> bool:
        text = answer.lower()
        return (
            "[no-context]" in text
            or "i do not have enough information to answer" in text
            or "sorry, i'm not able to provide an answer" in text
        )

    def _response_to_text(response: object) -> str:
        if isinstance(response, str):
            return response
        content = getattr(response, "content", None)
        if content is not None:
            return str(content)
        return str(response)

    rag = None
    try:
        rag = await initialize_rag(working_dir)

        base_top_k = int(os.getenv("TOP_K", "40"))
        base_chunk_top_k = int(os.getenv("CHUNK_TOP_K", "20"))

        base_param = QueryParam(
            mode=mode,
            top_k=base_top_k,
            chunk_top_k=base_chunk_top_k,
            enable_rerank=os.getenv("RERANK_BY_DEFAULT", "false").lower() == "true",
        )
        first_response = await rag.aquery(question, param=base_param)
        first_text = _response_to_text(first_response)
        if not _looks_like_no_answer(first_text):
            return first_text

        # Retry with broader retrieval and rerank disabled to recover from sparse contexts.
        retry_top_k = max(base_top_k, 80)
        retry_chunk_top_k = max(base_chunk_top_k, 80)
        retry_param = QueryParam(
            mode=mode,
            top_k=retry_top_k,
            chunk_top_k=retry_chunk_top_k,
            enable_rerank=False,
        )
        retry_response = await rag.aquery(question, param=retry_param)
        retry_text = _response_to_text(retry_response)
        if not _looks_like_no_answer(retry_text) or mode == "mix":
            return retry_text

        # Final fallback: switch to mixed retrieval mode with broader context.
        mix_response = await rag.aquery(
            question,
            param=QueryParam(
                mode="mix",
                top_k=retry_top_k,
                chunk_top_k=retry_chunk_top_k,
                enable_rerank=False,
            ),
        )
        return _response_to_text(mix_response)
    finally:
        if rag is not None:
            await rag.finalize_storages()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LightRAG over the Chomsky corpus")
    parser.add_argument("--ingest", action="store_true", help="Index dataset into LightRAG")
    parser.add_argument("--query", type=str, help="Question to ask")
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["naive", "local", "global", "hybrid", "mix"],
        help="LightRAG query mode",
    )
    parser.add_argument("--doc-limit", type=int, default=200, help="How many docs to index")
    parser.add_argument(
        "--working-dir",
        type=str,
        default=DEFAULT_WORKING_DIR,
        help="Directory where LightRAG stores graph/vector state",
    )
    return parser.parse_args()


async def run_cli(args: argparse.Namespace) -> None:
    if args.ingest:
        count = await ingest_corpus(doc_limit=args.doc_limit, working_dir=args.working_dir)
        print(f"Indexed {count} documents into LightRAG store: {args.working_dir}")

    if args.query:
        answer = await query_rag(args.query, mode=args.mode, working_dir=args.working_dir)
        print(f"\nQ: {args.query}")
        print(f"\nA: {answer}")

    if not args.ingest and not args.query:
        print("Nothing to do. Use --ingest and/or --query.")


if __name__ == "__main__":
    asyncio.run(run_cli(parse_args()))

