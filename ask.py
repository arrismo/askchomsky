import argparse
import asyncio
import os
import sys


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


from main import get_langfuse_client, ingest_corpus, query_rag


def ask(question: str, mode: str = "hybrid", working_dir: str = "./lightrag_store") -> None:
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

    response = asyncio.run(query_rag(question, mode=mode, working_dir=working_dir))

    print(f"\nQ: {question}")
    print(f"\nA: {response}")

    if lf is not None and root_obs is not None:
        if rag_obs is not None:
            rag_obs.update(
                output=str(response),
                metadata={"mode": mode, "working_dir": working_dir},
            )
            rag_obs.end()
        root_obs.update(
            output=str(response),
            metadata={"mode": mode, "working_dir": working_dir},
        )
        root_obs.end()
        lf.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask Chomsky with LightRAG")
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
        default="./lightrag_store",
        help="Directory where LightRAG stores graph/vector state",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    if args.ingest:
        count = asyncio.run(ingest_corpus(doc_limit=args.doc_limit, working_dir=args.working_dir))
        print(f"Indexed {count} documents into LightRAG store: {args.working_dir}")

    if args.query:
        ask(args.query, mode=args.mode, working_dir=args.working_dir)

    if not args.ingest and not args.query:
        question = "How does Chomsky distinguish propaganda in democratic societies versus authoritarian ones?"
        ask(question, mode=args.mode, working_dir=args.working_dir)


if __name__ == "__main__":
    run(parse_args())
