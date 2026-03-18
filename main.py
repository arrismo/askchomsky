from dotenv import load_dotenv
load_dotenv()  
import os
import nest_asyncio

nest_asyncio.apply()

import chromadb
from datasets import load_dataset
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Settings, VectorStoreIndex, Document, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.lmstudio import LMStudio


# Setup LM Studio Config

llm = LMStudio(
    model_name=os.getenv("model_name"),
    base_url=os.getenv("base_url"),
    temperature=0.1,
    max_tokens=512,
)


# set the embed model
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-base-en-v1.5"
)

# Load the Chomsky corpus dataset
ds = load_dataset("mmoise00/chomsky-corpus", split="train")

# Convert the dataset into a list of Document objects
documents = [
    Document(
        text=row["content"],
        metadata={
            "record_id": row["record_id"],
            "section": row["section"],
            "article_title": row["article_title"],
            "page_title": row["page_title"],
            "article_date": row["article_date"],
            "index_url": row["index_url"],
        },
    )
    for row in ds
]

# Create a Chroma vector store and index
chroma_client = chromadb.PersistentClient(path="./chroma_db")
# Create or get the Chroma collection for the Chomsky corpus
chroma_collection = chroma_client.get_or_create_collection(name="chomsky_corpus")
vector_store = ChromaVectorStore(chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create the VectorStoreIndex using the documents and storage context
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)


