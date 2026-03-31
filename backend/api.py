"""
FastAPI backend for AskChomsky.

Streams pipeline stage events via Server-Sent Events (SSE) so the
Next.js frontend can animate nodes in real-time.

Endpoint: POST /query
Body:      { "question": "..." }
Stream:    text/event-stream  — one JSON object per pipeline stage
"""

import asyncio
import json
import os
import re
import sys
from typing import Any, AsyncGenerator

# ---------------------------------------------------------------------------
# Make sure we can import main.py from the project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from main import (
    CITATION_SYSTEM_PROMPT,
    DEFAULT_WORKING_DIR,
    initialize_rag,
    llm_model_func,
    query_rag,
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="AskChomsky API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str
    # Optional override for retrieval mode (naive, local, global, hybrid, mix)
    mode: str | None = None


class CompareRequest(BaseModel):
    question: str
    mode_a: str | None = None
    mode_b: str | None = None


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------
def _sse(event_type: str, data: dict) -> str:
    """Format a single SSE frame."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def _stage_event(
    stage_id: str,
    label: str,
    status: str,  # "running" | "done" | "error"
    detail: str = "",
    extra: dict | None = None,
) -> str:
    payload: dict[str, Any] = {
        "id": stage_id,
        "label": label,
        "status": status,
        "detail": detail,
    }
    if extra:
        payload.update(extra)
    return _sse("stage", payload)


# ---------------------------------------------------------------------------
# Utility functions (mirrored from app.py)
# ---------------------------------------------------------------------------
_INTENT_CACHE: dict[str, dict[str, Any]] = {}


def _normalize_intent_text(text: str) -> tuple[str, list[str]]:
    lowered = text.lower().strip()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", lowered)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    tokens = cleaned.split() if cleaned else []
    return cleaned, tokens


def _rule_based_intent_router(text: str) -> dict[str, Any] | None:
    cleaned, tokens = _normalize_intent_text(text)
    if not cleaned:
        return {
            "intent": "small_talk",
            "confidence": 1.0,
            "source": "rules-empty",
            "reason": "Empty input.",
        }

    phrase = " ".join(tokens)
    greeting_matches = {
        "hi",
        "hello",
        "hey",
        "hiya",
        "greetings",
        "good day",
        "yo",
        "sup",
        "what s up",
        "howdy",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
        "how r u",
        "who are you",
        "thanks",
        "thank you",
        "thx",
    }
    if phrase in greeting_matches:
        return {
            "intent": "small_talk",
            "confidence": 0.99,
            "source": "rules-direct",
            "reason": "Matched greeting.",
        }

    query_markers = {
        "what",
        "why",
        "how",
        "when",
        "where",
        "which",
        "explain",
        "describe",
        "analyze",
        "compare",
        "difference",
        "tell",
        "about",
    }
    corpus_markers = {
        "chomsky",
        "linguistics",
        "language",
        "grammar",
        "syntax",
        "phonology",
        "semantics",
        "universal",
        "theory",
        "corpus",
    }

    has_query_intent = any(m in tokens for m in query_markers) or "?" in text
    has_corpus_intent = any(m in tokens for m in corpus_markers)

    if has_query_intent and len(tokens) >= 4:
        return {
            "intent": "corpus_question",
            "confidence": 0.95 if has_corpus_intent else 0.8,
            "source": "rules-query",
            "reason": "Detected query structure.",
        }

    greeting_heads = {"hi", "hello", "hey", "hiya", "yo", "greetings", "howdy"}
    if tokens[0] in greeting_heads and len(tokens) <= 4:
        return {
            "intent": "small_talk",
            "confidence": 0.95,
            "source": "rules-head",
            "reason": "Greeting head.",
        }

    return None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


async def _classify_intent_with_llm(text: str) -> dict[str, Any] | None:
    router_prompt = (
        "Classify the user message intent for a Chomsky Q&A app. "
        "Return strict JSON only with keys: intent, confidence, reason. "
        "intent must be one of: small_talk, corpus_question, other. "
        "confidence must be a number from 0 to 1.\n\n"
        f"Message: {text}"
    )
    try:
        response = await llm_model_func(
            router_prompt,
            system_prompt="You are an intent router. Output JSON only, no markdown. Prefer corpus_question when the user asks for factual content.",
            history_messages=[],
        )
        parsed = _extract_json_object(str(response))
        if not parsed:
            return None
        raw_intent = str(parsed.get("intent", "other")).strip().lower()
        if raw_intent not in {"small_talk", "corpus_question", "other"}:
            raw_intent = "other"
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
        return {
            "intent": raw_intent,
            "confidence": confidence,
            "source": "llm",
            "reason": str(parsed.get("reason", "")).strip(),
        }
    except Exception:
        return None


async def _detect_intent(text: str) -> dict[str, Any]:
    cache_key, _ = _normalize_intent_text(text)
    if cache_key in _INTENT_CACHE:
        cached = dict(_INTENT_CACHE[cache_key])
        cached["source"] = f"{cached.get('source', 'cache')}-cache"
        return cached
    rule_decision = _rule_based_intent_router(text)
    if rule_decision and float(rule_decision.get("confidence", 0.0)) >= 0.9:
        _INTENT_CACHE[cache_key] = rule_decision
        return rule_decision
    llm_decision = await _classify_intent_with_llm(text)
    if llm_decision is not None:
        _INTENT_CACHE[cache_key] = llm_decision
        return llm_decision
    if rule_decision is not None:
        _INTENT_CACHE[cache_key] = rule_decision
        return rule_decision
    fallback = {
        "intent": "corpus_question",
        "confidence": 0.5,
        "source": "default-fallback",
        "reason": "Defaulting to corpus query.",
    }
    _INTENT_CACHE[cache_key] = fallback
    return fallback


def _extract_references(raw_result: dict[str, Any]) -> list[dict[str, str]]:
    data = raw_result.get("data", {})
    references = data.get("references", [])
    return (
        [r for r in references if isinstance(r, dict)]
        if isinstance(references, list)
        else []
    )


def _extract_chunks(raw_result: dict[str, Any]) -> list[dict[str, Any]]:
    data = raw_result.get("data", {})
    chunks = data.get("chunks", [])
    return (
        [c for c in chunks if isinstance(c, dict)] if isinstance(chunks, list) else []
    )


def _extract_llm_text(raw_result: dict[str, Any]) -> str:
    content = raw_result.get("llm_response", {}).get("content")
    return str(content) if content is not None else ""


def _has_citation_marker(text: str) -> bool:
    return bool(re.search(r"\[\d+\]", text))


def _strip_citation_markers(text: str) -> str:
    """Remove standalone [n] markers when we have no structured references.

    This prevents the UI from showing fake citation numbers like [23]
    when the retrieval layer did not return any references.
    """
    # Remove optional leading whitespace and the marker itself.
    return re.sub(r"\s*\[\d+\]", "", text).strip()


def _render_references(references: list[dict[str, str]]) -> str:
    if not references:
        return ""
    lines = ["**Sources:**"]
    for ref in references:
        ref_id = str(ref.get("reference_id", "")).strip()
        file_path = str(ref.get("file_path", "")).strip() or "unknown"
        if ref_id:
            lines.append(f"[{ref_id}] {file_path}")
    return "\n".join(lines)


def _enforce_citation_answer(answer: str, references: list[dict[str, str]]) -> str:
    if not references:
        # No structured references: strip any citation-like markers the
        # model may have invented so the UI does not imply phantom sources.
        return _strip_citation_markers(answer)
    rendered = _render_references(references)
    safe = answer.strip()
    if not _has_citation_marker(safe):
        first_ref = str(references[0].get("reference_id", "1")).strip() or "1"
        safe = f"{safe}\n\nPrimary support [{first_ref}]."
    if rendered and "Sources:" not in safe:
        safe = f"{safe}\n\n{rendered}"
    return safe


def _looks_like_no_answer(answer: str) -> bool:
    text = answer.lower()
    return (
        "[no-context]" in text
        or "i do not have enough information to answer" in text
        or "sorry, i'm not able to provide an answer" in text
    )


def _dynamic_query_param(
    mode: str, original: str, rewritten: str, retry_level: int
) -> "QueryParam":
    base_top_k = int(os.getenv("TOP_K", "40"))
    base_chunk_top_k = int(os.getenv("CHUNK_TOP_K", "20"))
    text = f"{original} {rewritten}".lower()
    token_count = len(re.findall(r"\w+", rewritten))
    top_k, chunk_top_k = base_top_k, base_chunk_top_k
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
    return QueryParam(
        mode=mode,
        top_k=top_k,
        chunk_top_k=chunk_top_k,
        enable_rerank=rerank_default and retry_level == 0,
        include_references=True,
        response_type="Multiple Paragraphs",
    )


async def _rewrite_query(question: str) -> str:
    if os.getenv("REWRITE_QUERY", "true").lower() != "true":
        return question
    try:
        rewritten = await llm_model_func(
            f"Rewrite this question for retrieval over a Noam Chomsky corpus. Preserve intent and named entities. Return one line only, no extra text.\n\nQuestion: {question}",
            system_prompt="You are a retrieval query rewriter.",
            history_messages=[],
        )
        candidate = str(rewritten).strip().splitlines()[0].strip()
        return candidate[:600] if candidate else question
    except Exception:
        return question


async def _verify_claims(answer_text: str, chunks: list[dict[str, Any]]) -> str:
    if (
        os.getenv("VERIFY_CLAIMS", "true").lower() != "true"
        or not answer_text.strip()
        or not chunks
    ):
        return ""
    evidence_lines = []
    for chunk in chunks[:8]:
        ref_id = str(chunk.get("reference_id", "?")).strip() or "?"
        content = str(chunk.get("content", "")).strip().replace("\n", " ")
        if content:
            evidence_lines.append(f"[{ref_id}] {content[:700]}")
    if not evidence_lines:
        return ""
    try:
        resp = await llm_model_func(
            f"Verify claims in the answer using ONLY the provided evidence snippets.\nReturn strict JSON with keys: verdict, unsupported_claims, notes.\nverdict must be one of supported|partially_supported|unsupported.\n\nAnswer:\n{answer_text}\n\nEvidence:\n{chr(10).join(evidence_lines)}",
            system_prompt="You are a strict factual verifier.",
            history_messages=[],
        )
        parsed = _extract_json_object(str(resp))
        if not parsed:
            return "No structured verifier response."
        verdict = str(parsed.get("verdict", "unknown")).strip().lower()
        unsupported = parsed.get("unsupported_claims", [])
        if verdict == "supported":
            return "All major claims supported by retrieved evidence."
        if not isinstance(unsupported, list) or not unsupported:
            return f"Verdict: {verdict}."
        claims = [str(c).strip() for c in unsupported if str(c).strip()][:5]
        return f"Verdict: {verdict}\n" + "\n".join(f"- {c}" for c in claims)
    except Exception as exc:
        return f"Verifier failed: {exc}"


async def _generate_followup_questions(
    question: str,
    answer: str,
    references: list[dict[str, Any]],
) -> list[str]:
    """Propose a small set of follow-up questions for the UI.

    This uses the same LLM backend and returns 0–3 concise questions
    as plain strings. Any parsing errors simply result in an empty list
    so the rest of the pipeline is unaffected.
    """
    if not answer.strip():
        return []

    # Compact textual summary of sources (no need to be exhaustive)
    ref_lines: list[str] = []
    for ref in references[:5]:
        ref_id = str(ref.get("reference_id", "")).strip()
        file_path = str(ref.get("file_path", "")).strip()
        if not (ref_id or file_path):
            continue
        if ref_id and file_path:
            ref_lines.append(f"[{ref_id}] {file_path}")
        else:
            ref_lines.append(ref_id or file_path)

    refs_block = "\n".join(ref_lines) if ref_lines else "N/A"

    prompt = (
        "You are helping a student explore Noam Chomsky's work in depth. "
        "Given the original question, the answer, and a short list of sources, "
        "propose 3 concise, specific follow-up questions that naturally extend the discussion. "
        "Each question must be grounded in the same Chomsky corpus.\n\n"
        "Return strict JSON with a single key `follow_up_questions` containing an array of 1–3 strings. "
        "No additional keys, no markdown, no explanation.\n\n"
        f"Original question:\n{question}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Sources:\n{refs_block}"
    )

    try:
        resp = await llm_model_func(
            prompt,
            system_prompt=(
                "You propose thoughtful follow-up questions about Noam Chomsky's work. "
                "Output strict JSON only with the key `follow_up_questions`."
            ),
            history_messages=[],
        )
        parsed = _extract_json_object(str(resp))
        if not parsed:
            return []

        raw = parsed.get("follow_up_questions", [])
        if not isinstance(raw, list):
            return []

        followups: list[str] = []
        for item in raw:
            text = str(item).strip()
            if not text:
                continue
            followups.append(text[:400])
            if len(followups) >= 3:
                break
        return followups
    except Exception:
        # Follow-up generation is best-effort; failures should be silent.
        return []


# ---------------------------------------------------------------------------
# Core streaming generator
# ---------------------------------------------------------------------------
async def _stream_pipeline(
    question: str,
    mode_override: str | None = None,
) -> AsyncGenerator[str, None]:
    rag = None
    try:
        # ── Stage: Intent Router ──────────────────────────────────────────
        yield _stage_event("intent", "Intent Router", "running")
        intent = await _detect_intent(question)
        yield _stage_event(
            "intent",
            "Intent Router",
            "done",
            detail=(
                f"intent: {intent.get('intent')}\n"
                f"confidence: {float(intent.get('confidence', 0)):.0%}\n"
                f"source: {intent.get('source')}\n"
                f"reason: {intent.get('reason')}"
            ),
        )

        # Small-talk short circuit
        if intent.get("intent") == "small_talk":
            yield _stage_event(
                "answer",
                "Answer",
                "done",
                detail="This looks like small talk. Ask me a question about Noam Chomsky's work — e.g. 'What is Universal Grammar?'",
            )
            yield _sse("done", {"answer": "Ask me anything about Chomsky's work."})
            return

        # ── Stage: Query Rewrite ─────────────────────────────────────────
        yield _stage_event("rewrite", "Query Rewrite", "running")
        rewritten = await _rewrite_query(question)
        yield _stage_event(
            "rewrite",
            "Query Rewrite",
            "done",
            detail=f"Original: {question}\n\nRewritten: {rewritten}",
        )

        # ── Stage: RAG Init ──────────────────────────────────────────────
        yield _stage_event("rag_init", "Loading RAG Store", "running")
        # RAG_WORKING_DIR controls where the LightRAG index is stored.
        working_dir = os.getenv("RAG_WORKING_DIR", DEFAULT_WORKING_DIR)
        rag = await initialize_rag(working_dir)
        yield _stage_event(
            "rag_init", "Loading RAG Store", "done", detail=f"Store: {working_dir}"
        )

        # ── Stage: Retrieval (with retries, data only — no LLM) ──────────
        selected_mode = (
            mode_override or os.getenv("CHAINLIT_MODE") or "hybrid"
        ).lower()
        if selected_mode not in {"naive", "local", "global", "hybrid", "mix"}:
            selected_mode = "hybrid"

        attempt_modes = (
            [selected_mode, selected_mode, "mix"]
            if selected_mode != "mix"
            else ["mix", "mix"]
        )
        selected_data: dict[str, Any] | None = None
        references: list[dict[str, str]] = []
        chunks: list[dict[str, Any]] = []
        final_param: QueryParam | None = None
        last_attempt_index = -1

        for retry_level, attempt_mode in enumerate(attempt_modes):
            stage_id = f"retrieval_{retry_level + 1}"
            param = _dynamic_query_param(attempt_mode, question, rewritten, retry_level)
            label = (
                "Retrieval" if retry_level == 0 else f"Retrieval (retry {retry_level})"
            )
            yield _stage_event(
                stage_id,
                label,
                "running",
                detail=(
                    f"mode: {param.mode}\ntop_k: {param.top_k}\nchunk_top_k: {param.chunk_top_k}\nrerank: {param.enable_rerank}"
                ),
            )
            data_result = await rag.aquery_data(rewritten, param=param)
            refs = _extract_references(data_result)
            cks = _extract_chunks(data_result)
            yield _stage_event(
                stage_id,
                label,
                "done",
                detail=f"references: {len(refs)}\nchunks: {len(cks)}",
                extra={"attempt": retry_level + 1},
            )

            # Track which attempt actually ran
            last_attempt_index = retry_level

            # Accept this attempt if we got chunks or references
            if refs or cks:
                selected_data = data_result
                references = refs
                chunks = cks
                final_param = param
                break

            # Keep going on next retry
            selected_data = data_result
            references = refs
            chunks = cks
            final_param = param

        # Mark any later retrieval stages as "skipped" so the UI shows
        # them as completed even if they were never actually executed.
        for skipped_level in range(last_attempt_index + 1, len(attempt_modes)):
            stage_id = f"retrieval_{skipped_level + 1}"
            label = (
                "Retrieval"
                if skipped_level == 0
                else f"Retrieval (retry {skipped_level})"
            )
            yield _stage_event(
                stage_id,
                label,
                "done",
                detail="Skipped: earlier retrieval attempt returned sufficient context.",
                extra={"skipped": True},
            )

        if not final_param:
            yield _stage_event(
                "answer", "Answer", "error", detail="No data retrieved from the corpus."
            )
            yield _sse(
                "done",
                {
                    "answer": "I do not have enough information to answer from the retrieved corpus."
                },
            )
            return

        # ── Stage: Citation Enforcement (pre-answer, sources only) ───────
        yield _stage_event("citations", "Citation Enforcement", "running")
        sources_text = _render_references(references) or "No references returned."
        yield _stage_event(
            "citations", "Citation Enforcement", "done", detail=sources_text
        )

        # ── Stage: Answer — stream tokens live ───────────────────────────
        yield _stage_event("answer", "Answer", "running", detail="")

        stream_param = _dynamic_query_param(
            final_param.mode,
            question,
            rewritten,
            retry_level=0,
        )
        stream_param.stream = True

        stream_result = await rag.aquery_llm(
            rewritten,
            param=stream_param,
            system_prompt=CITATION_SYSTEM_PROMPT,
        )

        llm_meta = stream_result.get("llm_response", {})
        full_answer = ""

        if llm_meta.get("is_streaming") and llm_meta.get("response_iterator"):
            async for token in llm_meta["response_iterator"]:
                if token:
                    full_answer += token
                    yield _sse("token", {"token": token})
        else:
            # Fallback: non-streaming content
            full_answer = str(llm_meta.get("content") or "")
            if full_answer:
                yield _sse("token", {"token": full_answer})

        if not full_answer or _looks_like_no_answer(full_answer):
            yield _stage_event(
                "answer",
                "Answer",
                "error",
                detail="Could not generate an answer from the retrieved context.",
            )
            yield _sse(
                "done",
                {
                    "answer": "I do not have enough information to answer from the retrieved corpus."
                },
            )
            return

        # Apply citation enforcement to the completed streamed answer
        answer_with_citations = _enforce_citation_answer(full_answer, references)
        suffix = ""
        if sources_text and sources_text != "No references returned.":
            suffix = f"\n\n{sources_text}"

        # ── Stage: Claim Verification ────────────────────────────────────
        yield _stage_event("verification", "Claim Verification", "running")
        verification = await _verify_claims(answer_with_citations, chunks)
        yield _stage_event(
            "verification",
            "Claim Verification",
            "done",
            detail=verification or "All claims supported.",
        )

        # Build final answer string (sources + verification appended)
        final = answer_with_citations + suffix
        if verification:
            final += f"\n\n---\n**Claim Verification:** {verification}"

        # Optionally generate follow-up questions for the frontend
        followups: list[str] = []
        if os.getenv("FOLLOWUP_QUESTIONS", "true").lower() == "true":
            followups = await _generate_followup_questions(question, final, references)

        yield _stage_event("answer", "Answer", "done", detail=final)

        done_payload: dict[str, Any] = {"answer": final}
        if followups:
            done_payload["follow_up_questions"] = followups

        yield _sse("done", done_payload)

    except Exception as exc:
        yield _stage_event("answer", "Answer", "error", detail=str(exc))
        yield _sse("error", {"message": str(exc)})
    finally:
        if rag is not None:
            await rag.finalize_storages()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/query")
async def query(req: QueryRequest) -> StreamingResponse:
    question = req.question.strip()
    if not question:

        async def _empty():
            yield _sse("error", {"message": "Empty question."})

        return StreamingResponse(_empty(), media_type="text/event-stream")

    return StreamingResponse(
        _stream_pipeline(question, mode_override=req.mode),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


_ALLOWED_MODES = {"naive", "local", "global", "hybrid", "mix"}


def _normalize_mode(value: str | None, default: str) -> str:
    if not value:
        return default
    v = value.lower()
    return v if v in _ALLOWED_MODES else default


@app.post("/compare")
async def compare(req: CompareRequest) -> dict:
    question = req.question.strip()
    if not question:
        return {"error": "Empty question."}

    mode_a = _normalize_mode(req.mode_a, "hybrid")
    mode_b = _normalize_mode(req.mode_b, "hybrid")

    # Use the same working directory as the streaming pipeline so
    # comparisons run against the actual ingested LightRAG store.
    # Use the same working directory as the streaming pipeline so
    # comparisons run against the actual ingested LightRAG store.
    working_dir = os.getenv("RAG_WORKING_DIR", DEFAULT_WORKING_DIR)

    answer_a, answer_b = await asyncio.gather(
        query_rag(question, mode=mode_a, working_dir=working_dir),
        query_rag(question, mode=mode_b, working_dir=working_dir),
    )

    return {
        "question": question,
        "mode_a": mode_a,
        "answer_a": answer_a,
        "mode_b": mode_b,
        "answer_b": answer_b,
    }
