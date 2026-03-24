import time
from main import get_query_engine, get_langfuse_client


def ask(question: str, query_engine) -> None:
    lf = get_langfuse_client()
    root_obs = None
    rag_obs = None

    if lf is not None:
        root_obs = lf.start_observation(
            name="askchomsky",
            as_type="chain",
            input=question,
        )
        rag_obs = root_obs.start_observation(
            name="rag-query",
            as_type="span",
            input=question,
        )

    t0 = time.perf_counter()
    response = query_engine.query(question)
    latency = time.perf_counter() - t0

    # Collect source citations
    sources = []
    if hasattr(response, "source_nodes"):
        for node in response.source_nodes:
            meta = node.node.metadata
            sources.append(
                f"  - [{meta.get('article_title', 'unknown')}] "
                f"{meta.get('article_date', '')} | "
                f"score: {node.score:.3f}"
            )

    print(f"\nQ: {question}")
    print(f"\nA: {response}")
    if sources:
        print("\nSources:")
        print("\n".join(sources))
    print(f"\n[latency: {latency:.2f}s]")

    if lf is not None and root_obs is not None:
        if rag_obs is not None:
            rag_obs.update(
                output=str(response),
                metadata={"latency_s": round(latency, 3), "sources": sources},
            )
            rag_obs.end()
        root_obs.update(
            output=str(response),
            metadata={"latency_s": round(latency, 3), "sources": sources},
        )
        root_obs.end()
        lf.flush()
        trace_url = lf.get_trace_url(trace_id=root_obs.trace_id)
        if trace_url:
            print(f"\nLangfuse trace: {trace_url}")


if __name__ == "__main__":
    query_engine = get_query_engine(similarity_top_k=5)
    question = "How does Chomsky connect corporate power to constraints on public discourse?"
    ask(question, query_engine)
