import os
import re
import json
import random
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


_INTENT_CACHE: dict[str, dict[str, Any]] = {}
_SMALL_TALK_RESPONSES_CACHE: dict[str, list[str]] | None = None

_DEFAULT_SMALL_TALK_RESPONSES: dict[str, list[str]] = {
    "greeting": [
        "Hi! I can answer questions about Noam Chomsky's work. Try: '{example}'",
        "Hello! Ask me anything from the Chomsky corpus. For example: '{example}'",
    ],
    "thanks": [
        "You're welcome. If you want, try: '{example}'",
        "Happy to help. A good next question is: '{example}'",
    ],
    "identity": [
        "I'm a Chomsky-focused research assistant backed by your indexed corpus. Try: '{example}'",
    ],
    "capabilities": [
        "I can explain, compare, and summarize topics from Noam Chomsky's work. For example: '{example}'",
        "I can help with theory explanations, timelines, and concept comparisons from the corpus. Try: '{example}'",
    ],
    "fallback": [
        "I can help with questions about Noam Chomsky's work. Try: '{example}'",
    ],
}

_EXAMPLE_QUESTIONS: list[str] = [
    "What is Universal Grammar?",
    "How did Chomsky critique behaviorism?",
    "Compare Principles and Parameters with Minimalism.",
    "How did Chomsky's views on language acquisition evolve over time?",
]


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
            "reason": "Input is empty after normalization.",
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
            "reason": "Matched direct greeting/pleasantry phrase.",
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

    has_query_intent = any(marker in tokens for marker in query_markers) or "?" in text
    has_corpus_intent = any(marker in tokens for marker in corpus_markers)

    if has_query_intent and len(tokens) >= 4:
        return {
            "intent": "corpus_question",
            "confidence": 0.95 if has_corpus_intent else 0.8,
            "source": "rules-query",
            "reason": "Detected clear query structure.",
        }

    greeting_heads = {"hi", "hello", "hey", "hiya", "yo", "greetings", "howdy"}
    if tokens[0] in greeting_heads and len(tokens) <= 4:
        return {
            "intent": "small_talk",
            "confidence": 0.95,
            "source": "rules-head",
            "reason": "Greeting head token in a short message.",
        }

    if len(tokens) <= 3 and all(t in {"hi", "hello", "hey", "yo"} for t in tokens):
        return {
            "intent": "small_talk",
            "confidence": 0.95,
            "source": "rules-short",
            "reason": "Short multi-token greeting.",
        }

    if len(tokens) <= 4 and tuple(tokens[:2]) in {
        ("good", "morning"),
        ("good", "afternoon"),
        ("good", "evening"),
        ("good", "day"),
    }:
        return {
            "intent": "small_talk",
            "confidence": 0.95,
            "source": "rules-timeofday",
            "reason": "Time-of-day greeting in short message.",
        }

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
            system_prompt=(
                "You are an intent router. Output JSON only, no markdown. "
                "Prefer corpus_question when the user asks for factual content."
            ),
            history_messages=[],
        )
        parsed = _extract_json_object(str(response))
        if not parsed:
            return None

        raw_intent = str(parsed.get("intent", "other")).strip().lower()
        if raw_intent not in {"small_talk", "corpus_question", "other"}:
            raw_intent = "other"

        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        reason = str(parsed.get("reason", "")).strip() or "LLM classification"
        return {
            "intent": raw_intent,
            "confidence": confidence,
            "source": "llm",
            "reason": reason,
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
        fallback_rule = dict(rule_decision)
        fallback_rule["source"] = f"{fallback_rule.get('source', 'rules')}-fallback"
        _INTENT_CACHE[cache_key] = fallback_rule
        return fallback_rule

    fallback = {
        "intent": "corpus_question",
        "confidence": 0.5,
        "source": "default-fallback",
        "reason": "No classifier result; defaulting to corpus query route.",
    }
    _INTENT_CACHE[cache_key] = fallback
    return fallback


def _small_talk_bucket(text: str) -> str:
    cleaned, tokens = _normalize_intent_text(text)
    phrase = " ".join(tokens)

    if any(t in {"thanks", "thank", "thx"} for t in tokens):
        return "thanks"
    if phrase in {"who are you", "what are you"}:
        return "identity"

    capability_hints = {"help", "can", "do", "able", "support", "features"}
    if any(t in capability_hints for t in tokens):
        return "capabilities"

    if cleaned:
        return "greeting"
    return "fallback"


def _load_small_talk_responses() -> dict[str, list[str]]:
    responses = {k: list(v) for k, v in _DEFAULT_SMALL_TALK_RESPONSES.items()}
    raw = os.getenv("SMALL_TALK_RESPONSES_JSON", "").strip()
    if not raw:
        return responses

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return responses

    if not isinstance(parsed, dict):
        return responses

    for key, value in parsed.items():
        bucket = str(key).strip().lower()
        if bucket not in responses:
            continue
        if isinstance(value, list):
            cleaned = [str(item).strip() for item in value if str(item).strip()]
            if cleaned:
                responses[bucket] = cleaned

    return responses


def _get_small_talk_responses() -> dict[str, list[str]]:
    global _SMALL_TALK_RESPONSES_CACHE
    if _SMALL_TALK_RESPONSES_CACHE is None:
        _SMALL_TALK_RESPONSES_CACHE = _load_small_talk_responses()
    return _SMALL_TALK_RESPONSES_CACHE


async def _build_small_talk_response(text: str) -> str:
    bucket = _small_talk_bucket(text)
    example = random.choice(_EXAMPLE_QUESTIONS)

    use_llm = os.getenv("SMALL_TALK_USE_LLM", "false").lower() == "true"
    if use_llm:
        prompt = (
            "Write a short, friendly reply for a Chomsky Q&A assistant. "
            "User sent small talk, not a factual query. "
            "Reply in 1-2 sentences and include one concrete example question the user can ask next. "
            "No markdown.\n\n"
            f"User message: {text}\n"
            f"Suggested example: {example}"
        )
        try:
            generated = await llm_model_func(
                prompt,
                system_prompt="You are a concise assistant that redirects small talk into useful questions.",
                history_messages=[],
            )
            candidate = str(generated).strip()
            if candidate:
                return candidate
        except Exception:
            pass

    responses = _get_small_talk_responses()
    options = (
        responses.get(bucket)
        or responses.get("fallback")
        or ["Ask a Chomsky-related question."]
    )
    template = random.choice(options)
    return template.format(example=example)


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


def _format_trace_sections(sections: list[tuple[str, str]]) -> str:
    parts: list[str] = []
    for name, details in sections:
        cleaned_details = str(details).strip()
        if not cleaned_details:
            continue
        parts.append(f"#### {name}\n{cleaned_details}")
    return "\n\n".join(parts)


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

        cleaned_claims = [str(c).strip() for c in unsupported_claims if str(c).strip()][
            :5
        ]
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
    trace_sections: list[tuple[str, str]] = []

    async with cl.Step(name="Intent Router") as step:
        intent_decision = await _detect_intent(question)
        intent_summary = (
            f"intent={intent_decision.get('intent', 'other')}\n"
            f"confidence={float(intent_decision.get('confidence', 0.0)):.2f}\n"
            f"source={intent_decision.get('source', 'unknown')}\n"
            f"reason={intent_decision.get('reason', '')}"
        )
        step.output = intent_summary
    trace_sections.append(("Intent Router", intent_summary))

    if intent_decision.get("intent") == "small_talk":
        small_talk_reply = await _build_small_talk_response(question)
        trace_text = _format_trace_sections(trace_sections)
        content = small_talk_reply
        if trace_text:
            content = f"{content}\n\n---\n### Trace\n\n{trace_text}"
        await cl.Message(content=content).send()
        return

    mode = os.getenv("CHAINLIT_MODE", "hybrid")
    working_dir = os.getenv("CHAINLIT_WORKING_DIR", DEFAULT_WORKING_DIR)

    rag = None
    selected_result: dict[str, Any] | None = None

    try:
        async with cl.Step(name="Query Rewrite") as step:
            rewritten_question = await _rewrite_query_for_retrieval(question)
            rewrite_summary = f"Original: {question}\nRewritten: {rewritten_question}"
            step.output = rewrite_summary
        trace_sections.append(("Query Rewrite", rewrite_summary))

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
                retrieval_input = (
                    f"mode={param.mode}, top_k={param.top_k}, chunk_top_k={param.chunk_top_k}, "
                    f"rerank={param.enable_rerank}"
                )
                step.input = retrieval_input

                result = await rag.aquery_llm(
                    rewritten_question,
                    param=param,
                    system_prompt=CITATION_SYSTEM_PROMPT,
                )
                answer_text = _extract_llm_text(result)
                references = _extract_references(result)
                chunks = _extract_chunks(result)

                selected_result = result
                retrieval_output = (
                    f"answer_chars={len(answer_text)}\n"
                    f"references={len(references)}\n"
                    f"chunks={len(chunks)}"
                )
                step.output = retrieval_output
            trace_sections.append(
                (
                    f"Retrieval Attempt {retry_level + 1}",
                    f"{retrieval_input}\n{retrieval_output}",
                )
            )

            if answer_text and not _looks_like_no_answer(answer_text):
                break

        if selected_result is None:
            trace_text = _format_trace_sections(trace_sections)
            content = (
                "I do not have enough information to answer from the retrieved corpus."
            )
            if trace_text:
                content = f"{content}\n\n---\n### Trace\n\n{trace_text}"
            await cl.Message(content=content).send()
            return

        answer_text = _extract_llm_text(selected_result)
        references = _extract_references(selected_result)
        chunks = _extract_chunks(selected_result)

        async with cl.Step(name="Citation Enforcement") as step:
            answer_with_citations = _enforce_citation_answer(answer_text, references)
            citation_summary = (
                _render_references(references) or "No references returned."
            )
            step.output = citation_summary
        trace_sections.append(("Citation Enforcement", citation_summary))

        async with cl.Step(name="Claim Verification") as step:
            verification_summary = await _verify_claims(answer_with_citations, chunks)
            step.output = verification_summary
        trace_sections.append(
            ("Claim Verification", verification_summary or "No verification summary.")
        )

        final_answer = f"{answer_with_citations}\n\nClaim verification:\n{verification_summary}".strip()
        trace_text = _format_trace_sections(trace_sections)
        if trace_text:
            final_answer = f"{final_answer}\n\n---\n### Trace\n\n{trace_text}"
        await cl.Message(content=final_answer).send()

    except Exception as exc:
        trace_text = _format_trace_sections(trace_sections)
        content = f"Error: {exc}"
        if trace_text:
            content = f"{content}\n\n---\n### Trace\n\n{trace_text}"
        await cl.Message(content=content).send()
    finally:
        if rag is not None:
            await rag.finalize_storages()
