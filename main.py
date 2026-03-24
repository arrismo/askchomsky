import os
from typing import Optional
import nest_asyncio

nest_asyncio.apply()

from dotenv import load_dotenv
load_dotenv()  

import chromadb
from datasets import load_dataset
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Settings, VectorStoreIndex, Document, StorageContext
from llama_index.core.base.llms.types import LLMMetadata, MessageRole
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI


class OpenRouterOpenAI(OpenAI):
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=128000,
            num_output=self.max_tokens or 4096,
            is_chat_model=True,
            is_function_calling_model=False,
            model_name=self.model,
            system_role=MessageRole.SYSTEM,
        )

    @property
    def _tokenizer(self) -> Optional[object]:
        return None


_index: Optional[VectorStoreIndex] = None


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


def configure_models() -> None:
    openrouter_key = os.getenv("openrouter_key")
    if not openrouter_key:
        raise ValueError("Missing OpenRouter key. Set openrouter_key in your .env file.")

    Settings.llm = OpenRouterOpenAI(
        model="openai/gpt-oss-120b",
        api_base="https://openrouter.ai/api/v1",
        api_key=openrouter_key,
        temperature=0.1,
        max_tokens=2512,
    )

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-base-en-v1.5"
    )


def get_or_create_index() -> VectorStoreIndex:
    global _index
    if _index is not None:
        return _index

    print("Loading embedding model...")
    configure_models()
    print("Embedding model ready.")

    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = chroma_client.get_or_create_collection(name="chomsky_corpus")
    vector_store = ChromaVectorStore(chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    existing_vectors = chroma_collection.count()

    if existing_vectors > 0:
        print(f"Using existing Chroma index with {existing_vectors} vectors.")
        _index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        return _index

    print("Chroma collection empty. Building index from dataset (first run only)...")
    ds = load_dataset("mmoise00/chomsky-corpus", split="train")

    documents = [
        Document(
            text=row["content"],
            metadata={
                "record_id": row["record_id"],
                "section": row["section"],
                "article_title": row["article_title"],
                "article_date": row["article_date"],
                "index_url": row["index_url"],
            },
        )
        for row in ds
    ]

    _index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )
    print("Index build complete.")
    return _index


def get_query_engine(similarity_top_k: int = 5):
    index = get_or_create_index()
    return index.as_query_engine(similarity_top_k=similarity_top_k)


if __name__ == "__main__":
    print("Library module ready. Use ask.py to run queries.")

