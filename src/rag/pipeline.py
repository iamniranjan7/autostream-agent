"""
RAG pipeline for the AutoStream agent.

Loads the product knowledge base from a Markdown file, chunks it,
embeds it using Google's embedding model, and stores it in a FAISS
vector store for similarity-based retrieval.
"""

import os
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


# Path to the knowledge base file
KB_PATH = Path(__file__).parent.parent.parent / "knowledge_base" / "autostream_kb.md"

# Module-level cache for the vector store (singleton)
_vector_store = None


def _build_vector_store() -> FAISS:
    """Load the knowledge base, chunk it, and build a FAISS vector store.

    Returns:
        A FAISS vector store populated with embedded knowledge base chunks.
    """
    # Load the markdown knowledge base
    loader = TextLoader(str(KB_PATH), encoding="utf-8")
    documents = loader.load()

    # Split into chunks — small chunks for precise retrieval
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["\n## ", "\n### ", "\n- ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(documents)

    # Embed using Google's embedding model
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
    )

    # Work around langchain-google-genai batching bug with gemini-embedding-2:
    # embed each document individually then build FAISS from texts+embeddings
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    all_embeddings = []
    for text in texts:
        emb = embeddings.embed_documents([text])
        all_embeddings.append(emb[0])

    # Build FAISS from pre-computed embeddings
    text_embedding_pairs = list(zip(texts, all_embeddings))
    vector_store = FAISS.from_embeddings(
        text_embedding_pairs,
        embeddings,
        metadatas=metadatas,
    )
    return vector_store


def get_retriever(k: int = 4):
    """Get a retriever backed by the FAISS vector store.

    Uses a module-level singleton so the vector store is only built once
    per process lifetime.

    Args:
        k: Number of documents to retrieve per query.

    Returns:
        A LangChain retriever instance.
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = _build_vector_store()
    return _vector_store.as_retriever(search_kwargs={"k": k})

