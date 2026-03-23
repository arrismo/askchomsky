import time
from main import get_query_engine, configure_langfuse


def ask(question: str, query_engine) -> None:
    langfuse_enabled = configure_langfuse()

    if langfuse_enabled:
        from langfuse import Langfuse
        lf = Langfuse()
        trace = lf.trace(name="askchomsky", input=question)
        span = trace.span(name="rag-query", input=question)

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

    if langfuse_enabled:
        span.end(
            output=str(response),
            metadata={"latency_s": round(latency, 3), "sources": sources},
        )
        trace.update(output=str(response))
        lf.flush()


if __name__ == "__main__":
    query_engine = get_query_engine(similarity_top_k=5)
    question = "How does Chomsky's manufacturing consent model explain modern social media platforms?"
    ask(question, query_engine)
