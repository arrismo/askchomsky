import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
from typing import Any, TYPE_CHECKING


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


load_dotenv()


def configure_logging() -> None:
    """Configure app and dependency logging level.

    Defaults to WARNING to keep CLI output concise. Override with
    ASKCHOMSKY_LOG_LEVEL (e.g., INFO, DEBUG) when troubleshooting.
    """
    level_name = os.getenv("ASKCHOMSKY_LOG_LEVEL", "WARNING").upper()
    level = getattr(logging, level_name, logging.WARNING)

    # Keep root and noisy dependencies aligned with the selected verbosity.
    logging.getLogger().setLevel(level)
    for logger_name in (
        "lightrag",
        "nano-vectordb",
        "sentence_transformers",
        "sentence_transformers.SentenceTransformer",
        "httpx",
        "openai",
    ):
        logging.getLogger(logger_name).setLevel(level)


configure_logging()


if TYPE_CHECKING:
    # These imports are heavy (transitively pull in torch, CUDA, etc.).
    # Import them only for type checking; at runtime we import lazily.
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.openai import openai_complete_if_cache
    from lightrag.utils import EmbeddingFunc

# LightRAG configures its own logger during import, so apply our level again
# once we actually import it lazily at runtime (see initialize_rag).
configure_logging()


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = os.getenv("ASKCHOMSKY_LLM_MODEL", "openai/gpt-4o-mini")
EMBED_MODEL = os.getenv("ASKCHOMSKY_EMBED_MODEL", "openai/text-embedding-3-small")
EMBED_DIM = 1536
DEFAULT_WORKING_DIR = "./lightrag_store"
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT", "600"))
MAX_ASYNC_LLM_CALLS = int(os.getenv("MAX_ASYNC", "2"))
MAX_PARALLEL_INSERT = int(os.getenv("MAX_PARALLEL_INSERT", "2"))
REWRITE_QUERY_ENABLED = os.getenv("REWRITE_QUERY", "true").lower() == "true"
VERIFY_CLAIMS_ENABLED = os.getenv("VERIFY_CLAIMS", "true").lower() == "true"
QUERY_CACHE_TTL_SECONDS = int(os.getenv("QUERY_CACHE_TTL", "86400"))
QUERY_CACHE_PATH = os.path.join(DEFAULT_WORKING_DIR, "query_cache.json")


CITATION_SYSTEM_PROMPT = """You are a retrieval-grounded assistant.

Use only the provided context data.
If context is insufficient, say: I do not have enough information to answer from the retrieved corpus.

Citation rules:
1) Every factual claim must include at least one citation marker like [1].
2) Do not invent citation IDs.
3) Keep citation IDs consistent with the provided references.

Response style: {response_type}
User preference: {user_prompt}

Context:
{context_data}
"""


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


# ---------------------------------------------------------------------------
# API-based embeddings (OpenRouter / OpenAI-compatible)
# ---------------------------------------------------------------------------


def _get_api_key() -> str:
    api_key = os.getenv("openrouter_key") or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("Missing openrouter_key or OPENAI_API_KEY in .env")
    return api_key


def _api_embed_single(text: str) -> list[float]:
    import httpx

    api_key = _get_api_key()
    payload = {"input": text, "model": EMBED_MODEL}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            OPENROUTER_BASE_URL + "/embeddings", json=payload, headers=headers
        )
        resp.raise_for_status()
        data = resp.json()
    return data["data"][0]["embedding"]


def embed_texts(texts: list[str]) -> np.ndarray:
    embeddings = [_api_embed_single(t) for t in texts]
    arr = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await asyncio.to_thread(embed_texts, texts)


# ---------------------------------------------------------------------------
# Query result cache (disk-based, TTL-evicted)
# ---------------------------------------------------------------------------


def _load_query_cache() -> dict[str, dict[str, Any]]:
    if not os.path.exists(QUERY_CACHE_PATH):
        return {}
    try:
        with open(QUERY_CACHE_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_query_cache(cache: dict[str, dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(QUERY_CACHE_PATH), exist_ok=True)
    with open(QUERY_CACHE_PATH, "w") as f:
        json.dump(cache, f)


def _cache_key(question: str, mode: str) -> str:
    raw = f"{question.strip().lower()}|{mode}"
    return hashlib.sha256(raw.encode()).hexdigest()


def get_cached_answer(question: str, mode: str) -> str | None:
    if QUERY_CACHE_TTL_SECONDS <= 0:
        return None
    key = _cache_key(question, mode)
    cache = _load_query_cache()
    entry = cache.get(key)
    if not entry:
        return None
    if time.time() - entry.get("ts", 0) > QUERY_CACHE_TTL_SECONDS:
        del cache[key]
        _save_query_cache(cache)
        return None
    return entry.get("answer")


def cache_answer(question: str, mode: str, answer: str) -> None:
    if QUERY_CACHE_TTL_SECONDS <= 0:
        return
    key = _cache_key(question, mode)
    cache = _load_query_cache()
    cache[key] = {"answer": answer, "ts": time.time()}
    _save_query_cache(cache)


async def llm_model_func(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    **kwargs,
) -> str:
    # Import here to avoid pulling in heavy dependencies during module import.
    from lightrag.llm.openai import openai_complete_if_cache

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


async def initialize_rag(working_dir: str = DEFAULT_WORKING_DIR) -> "LightRAG":
    # Lazy imports keep startup fast and avoid loading the full
    # LightRAG/torch stack until we actually need RAG functionality.
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc

    os.makedirs(working_dir, exist_ok=True)

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        llm_model_name=LLM_MODEL,
        default_llm_timeout=LLM_TIMEOUT_SECONDS,
        llm_model_max_async=MAX_ASYNC_LLM_CALLS,
        max_parallel_insert=MAX_PARALLEL_INSERT,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBED_DIM,
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


async def ingest_corpus(
    doc_limit: int = 200, working_dir: str = DEFAULT_WORKING_DIR
) -> int:
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

    def _has_citation_marker(text: str) -> bool:
        return bool(re.search(r"\[\d+\]", text))

    def _extract_json_object(text: str) -> dict[str, Any] | None:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    def _extract_references(raw_result: dict[str, Any]) -> list[dict[str, str]]:
        data = raw_result.get("data", {})
        references = data.get("references", [])
        if isinstance(references, list):
            return [r for r in references if isinstance(r, dict)]
        return []

    def _extract_chunks(raw_result: dict[str, Any]) -> list[dict[str, Any]]:
        data = raw_result.get("data", {})
        chunks = data.get("chunks", [])
        if isinstance(chunks, list):
            return [c for c in chunks if isinstance(c, dict)]
        return []

    def _extract_llm_text(raw_result: dict[str, Any]) -> str:
        llm_response = raw_result.get("llm_response", {})
        content = llm_response.get("content")
        if content is None:
            return ""
        return str(content)

    def _render_references(references: list[dict[str, str]]) -> str:
        if not references:
            return ""
        lines: list[str] = ["Sources:"]
        for ref in references:
            ref_id = str(ref.get("reference_id", "")).strip()
            file_path = str(ref.get("file_path", "")).strip() or "unknown"
            if ref_id:
                lines.append(f"[{ref_id}] {file_path}")
        return "\n".join(lines)

    def _enforce_citation_answer(answer: str, references: list[dict[str, str]]) -> str:
        if not references:
            return answer

        rendered_references = _render_references(references)
        safe_answer = answer.strip()
        if not _has_citation_marker(safe_answer):
            first_ref = str(references[0].get("reference_id", "1")).strip() or "1"
            safe_answer = f"{safe_answer}\n\nPrimary support [{first_ref}]."

        if rendered_references and "Sources:" not in safe_answer:
            safe_answer = f"{safe_answer}\n\n{rendered_references}"
        return safe_answer

    async def _rewrite_query_for_retrieval(original_question: str) -> str:
        if not REWRITE_QUERY_ENABLED:
            return original_question

        rewrite_prompt = (
            "Rewrite this question for retrieval over a Noam Chomsky corpus. "
            "Preserve intent and named entities. Return one line only, no extra text.\n\n"
            f"Question: {original_question}"
        )

        try:
            rewritten = await llm_model_func(
                rewrite_prompt,
                system_prompt="You are a retrieval query rewriter.",
                history_messages=[],
            )
            candidate = _response_to_text(rewritten).strip().splitlines()[0].strip()
            if not candidate:
                return original_question
            return candidate[:600]
        except Exception:
            return original_question

    def _dynamic_query_param(
        selected_mode: str,
        original_question: str,
        rewritten_question: str,
        *,
        retry_level: int = 0,
    ) -> "QueryParam":
        base_top_k = int(os.getenv("TOP_K", "40"))
        base_chunk_top_k = int(os.getenv("CHUNK_TOP_K", "20"))

        text = f"{original_question} {rewritten_question}".lower()
        token_count = len(re.findall(r"\w+", rewritten_question))

        top_k = base_top_k
        chunk_top_k = base_chunk_top_k

        if token_count > 18:
            top_k += 15
            chunk_top_k += 15
        if any(k in text for k in ("compare", "versus", "difference", "contrast")):
            top_k += 20
            chunk_top_k += 20
        if any(k in text for k in ("timeline", "history", "evolution", "over time")):
            top_k += 20
            chunk_top_k += 20
        if any(k in text for k in ("why", "how", "explain", "analyze")):
            top_k += 10
            chunk_top_k += 10

        if retry_level == 1:
            top_k = max(top_k, 80)
            chunk_top_k = max(chunk_top_k, 80)
        elif retry_level >= 2:
            top_k = max(top_k, 100)
            chunk_top_k = max(chunk_top_k, 100)

        rerank_default = os.getenv("RERANK_BY_DEFAULT", "false").lower() == "true"
        enable_rerank = rerank_default and retry_level == 0

        return QueryParam(
            mode=selected_mode,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            enable_rerank=enable_rerank,
            include_references=True,
            response_type="Multiple Paragraphs",
        )

    async def _verify_claims(
        answer_text: str,
        chunks: list[dict[str, Any]],
    ) -> str:
        if not VERIFY_CLAIMS_ENABLED or not answer_text.strip() or not chunks:
            return ""

        evidence_lines: list[str] = []
        for chunk in chunks[:8]:
            ref_id = str(chunk.get("reference_id", "?")).strip() or "?"
            content = str(chunk.get("content", "")).strip().replace("\n", " ")
            if content:
                evidence_lines.append(f"[{ref_id}] {content[:700]}")

        if not evidence_lines:
            return ""

        verifier_prompt = (
            "Verify claims in the answer using ONLY the provided evidence snippets. "
            "Return strict JSON with keys: verdict, unsupported_claims, notes. "
            "verdict must be one of supported|partially_supported|unsupported.\n\n"
            f"Answer:\n{answer_text}\n\n"
            f"Evidence:\n{os.linesep.join(evidence_lines)}"
        )

        try:
            verifier_response = await llm_model_func(
                verifier_prompt,
                system_prompt="You are a strict factual verifier.",
                history_messages=[],
            )
            verifier_text = _response_to_text(verifier_response)
            verifier_json = _extract_json_object(verifier_text)
            if not verifier_json:
                return ""

            verdict = str(verifier_json.get("verdict", "")).strip().lower()
            unsupported_claims = verifier_json.get("unsupported_claims", [])
            if verdict in {"supported", ""} or not isinstance(unsupported_claims, list):
                return ""

            cleaned_claims = [
                str(c).strip() for c in unsupported_claims if str(c).strip()
            ][:5]
            if not cleaned_claims:
                return ""

            joined = "\n".join(f"- {claim}" for claim in cleaned_claims)
            return (
                "\n\nClaim verification: some claims could not be fully supported by retrieved evidence."
                f"\n{joined}"
            )
        except Exception:
            return ""

    cached = get_cached_answer(question, mode)
    if cached is not None:
        return cached

    rag = None
    try:
        rag = await initialize_rag(working_dir)

        rewritten_question = await _rewrite_query_for_retrieval(question)
        selected_result: dict[str, Any] | None = None

        attempt_modes = [mode, mode, "mix"] if mode != "mix" else ["mix", "mix"]

        for retry_level, attempt_mode in enumerate(attempt_modes):
            param = _dynamic_query_param(
                attempt_mode,
                question,
                rewritten_question,
                retry_level=retry_level,
            )
            result = await rag.aquery_llm(
                rewritten_question,
                param=param,
                system_prompt=CITATION_SYSTEM_PROMPT,
            )
            answer_text = _extract_llm_text(result)
            selected_result = result
            if answer_text and not _looks_like_no_answer(answer_text):
                break

        if selected_result is None:
            return (
                "I do not have enough information to answer from the retrieved corpus."
            )

        answer_text = _extract_llm_text(selected_result)
        references = _extract_references(selected_result)
        chunks = _extract_chunks(selected_result)

        answer_with_citations = _enforce_citation_answer(answer_text, references)
        verification_summary = await _verify_claims(answer_with_citations, chunks)

        final_answer = f"{answer_with_citations}{verification_summary}".strip()
        cache_answer(question, mode, final_answer)
        return final_answer
    finally:
        if rag is not None:
            await rag.finalize_storages()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LightRAG over the Chomsky corpus")
    parser.add_argument(
        "--ingest", action="store_true", help="Index dataset into LightRAG"
    )
    parser.add_argument("--query", type=str, help="Question to ask")
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["naive", "local", "global", "hybrid", "mix"],
        help="LightRAG query mode",
    )
    parser.add_argument(
        "--doc-limit", type=int, default=200, help="How many docs to index"
    )
    parser.add_argument(
        "--working-dir",
        type=str,
        default=DEFAULT_WORKING_DIR,
        help="Directory where LightRAG stores graph/vector state",
    )
    return parser.parse_args()


async def run_cli(args: argparse.Namespace) -> None:
    if args.ingest:
        count = await ingest_corpus(
            doc_limit=args.doc_limit, working_dir=args.working_dir
        )
        print(f"Indexed {count} documents into LightRAG store: {args.working_dir}")

    if args.query:
        answer = await query_rag(
            args.query, mode=args.mode, working_dir=args.working_dir
        )
        print(f"\nQ: {args.query}")
        print(f"\nA: {answer}")

    if not args.ingest and not args.query:
        print("Nothing to do. Use --ingest and/or --query.")


if __name__ == "__main__":
    asyncio.run(run_cli(parse_args()))
