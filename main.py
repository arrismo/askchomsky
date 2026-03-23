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
        # OpenRouter model ids are not in tiktoken's OpenAI model map.
        return None


_index: Optional[VectorStoreIndex] = None


def configure_langfuse() -> bool:
    """Initialize Langfuse OTEL tracing. Returns True if configured."""
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    if not public_key or public_key.startswith("pk-lf-...") or not secret_key or secret_key.startswith("sk-lf-..."):
        return False
    try:
        import langfuse.otel  # Langfuse v3 OTEL setup
        langfuse.otel.configure(
            public_key=public_key,
            secret_key=secret_key,
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        return True
    except Exception:
        return False


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

