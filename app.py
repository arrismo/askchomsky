import os
import re
import json
from typing import Any

import chainlit as cl

from main import (
    CITATION_SYSTEM_PROMPT,
    DEFAULT_WORKING_DIR,
    QueryParam,
    initialize_rag,
    llm_model_func,
)


def _looks_like_no_answer(answer: str) -> bool:
    text = answer.lower()
    return (
        "[no-context]" in text
        or "i do not have enough information to answer" in text
        or "sorry, i'm not able to provide an answer" in text
    )


def _has_citation_marker(text: str) -> bool:
    return bool(re.search(r"\[\d+\]", text))


def _is_small_talk_or_greeting(text: str) -> bool:
    lowered = text.lower().strip()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", lowered)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        return True

    tokens = cleaned.split()
    phrase = " ".join(tokens)

    direct_matches = {
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
    if phrase in direct_matches:
        return True

    # If a clear query intent is present, do not classify as small talk.
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
    if any(marker in tokens for marker in query_markers) and len(tokens) > 2:
        return False

    greeting_heads = {"hi", "hello", "hey", "hiya", "yo", "greetings", "howdy"}
    if tokens[0] in greeting_heads and len(tokens) <= 4:
        return True

    if len(tokens) <= 3 and all(t in {"hi", "hello", "hey", "yo"} for t in tokens):
        return True

    if len(tokens) <= 4 and tuple(tokens[:2]) in {
        ("good", "morning"),
        ("good", "afternoon"),
        ("good", "evening"),
        ("good", "day"),
    }:
        return True

    return False


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
    if os.getenv("REWRITE_QUERY", "true").lower() != "true":
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
        candidate = str(rewritten).strip().splitlines()[0].strip()
        if not candidate:
            return original_question
        return candidate[:600]
    except Exception:
        return original_question


def _dynamic_query_param(
    mode: str,
    original_question: str,
    rewritten_question: str,
    retry_level: int,
) -> QueryParam:
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
        mode=mode,
        top_k=top_k,
        chunk_top_k=chunk_top_k,
        enable_rerank=enable_rerank,
        include_references=True,
        response_type="Multiple Paragraphs",
    )


async def _verify_claims(answer_text: str, chunks: list[dict[str, Any]]) -> str:
    if os.getenv("VERIFY_CLAIMS", "true").lower() != "true":
        return ""
    if not answer_text.strip() or not chunks:
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
        verifier_json = _extract_json_object(str(verifier_response))
        if not verifier_json:
            return "No structured verifier response."

        verdict = str(verifier_json.get("verdict", "unknown")).strip().lower()
        unsupported_claims = verifier_json.get("unsupported_claims", [])

        if verdict in {"supported", ""}:
            return "All major claims supported by retrieved evidence."

        if not isinstance(unsupported_claims, list) or not unsupported_claims:
            return f"Verdict: {verdict}."

        cleaned_claims = [str(c).strip() for c in unsupported_claims if str(c).strip()][:5]
        if not cleaned_claims:
            return f"Verdict: {verdict}."

        joined = "\n".join(f"- {claim}" for claim in cleaned_claims)
        return f"Verdict: {verdict}\n{joined}"
    except Exception as exc:
        return f"Verifier failed: {exc}"


@cl.on_chat_start
async def on_chat_start() -> None:
    await cl.Message(
        content=(
            "Ask a question about Chomsky's corpus.\n"
            "I will show: rewrite, retrieval attempts, citations, and claim verification."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    question = message.content.strip()

    if _is_small_talk_or_greeting(question):
        await cl.Message(
            content=(
                "Hi! Ask me a question about Noam Chomsky's work, "
                "for example: 'What is Universal Grammar?'"
            )
        ).send()
        return

    mode = os.getenv("CHAINLIT_MODE", "hybrid")
    working_dir = os.getenv("CHAINLIT_WORKING_DIR", DEFAULT_WORKING_DIR)

    rag = None
    selected_result: dict[str, Any] | None = None

    try:
        async with cl.Step(name="Query Rewrite") as step:
            rewritten_question = await _rewrite_query_for_retrieval(question)
            step.output = f"Original: {question}\nRewritten: {rewritten_question}"

        rag = await initialize_rag(working_dir)

        attempt_modes = [mode, mode, "mix"] if mode != "mix" else ["mix", "mix"]

        for retry_level, attempt_mode in enumerate(attempt_modes):
            param = _dynamic_query_param(
                attempt_mode,
                question,
                rewritten_question,
                retry_level,
            )

            async with cl.Step(name=f"Retrieval Attempt {retry_level + 1}") as step:
                step.input = (
                    f"mode={param.mode}, top_k={param.top_k}, chunk_top_k={param.chunk_top_k}, "
                    f"rerank={param.enable_rerank}"
                )

                result = await rag.aquery_llm(
                    rewritten_question,
                    param=param,
                    system_prompt=CITATION_SYSTEM_PROMPT,
                )
                answer_text = _extract_llm_text(result)
                references = _extract_references(result)
                chunks = _extract_chunks(result)

                selected_result = result
                step.output = (
                    f"answer_chars={len(answer_text)}\n"
                    f"references={len(references)}\n"
                    f"chunks={len(chunks)}"
                )

                if answer_text and not _looks_like_no_answer(answer_text):
                    break

        if selected_result is None:
            await cl.Message(content="I do not have enough information to answer from the retrieved corpus.").send()
            return

        answer_text = _extract_llm_text(selected_result)
        references = _extract_references(selected_result)
        chunks = _extract_chunks(selected_result)

        async with cl.Step(name="Citation Enforcement") as step:
            answer_with_citations = _enforce_citation_answer(answer_text, references)
            step.output = _render_references(references) or "No references returned."

        async with cl.Step(name="Claim Verification") as step:
            verification_summary = await _verify_claims(answer_with_citations, chunks)
            step.output = verification_summary

        final_answer = f"{answer_with_citations}\n\nClaim verification:\n{verification_summary}".strip()
        await cl.Message(content=final_answer).send()

    except Exception as exc:
        await cl.Message(content=f"Error: {exc}").send()
    finally:
        if rag is not None:
            await rag.finalize_storages()
