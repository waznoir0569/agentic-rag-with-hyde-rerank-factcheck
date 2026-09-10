import asyncio
import os
from pathlib import Path
from uuid import UUID, uuid4

from langchain.embeddings import init_embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from loguru import logger

from app.config import settings

os.environ.setdefault("PINECONE_API_KEY", settings.pinecone_api_key.get_secret_value())

embeddings = init_embeddings(
    model=settings.embeddings_model_name,
    base_url=settings.embeddings_base_url,
    provider=settings.model_provider,
    api_key=settings.api_key,
)


vector_store = PineconeVectorStore(
    index_name=settings.pinecone_index_name,
    embedding=embeddings,  # type: ignore
    namespace=settings.pinecone_namespace or None,
)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)


DOCUMENT_LOADER_MAPPING: dict[str, type[BaseLoader]] = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
}

allowd_extensions = list(DOCUMENT_LOADER_MAPPING.keys())


async def _load_and_split_documents(file_path: Path) -> list[Document]:
    """
    Load and split documents based on file extension.
    Raises:
        UnsupportedFileTypeError: If the file extension is not supported.
        Exception: For other loading or splitting errors.
    """

    file_extension = file_path.suffix.lower()
    if file_extension not in DOCUMENT_LOADER_MAPPING:
        raise ValueError(f"Unsupported file type: {file_extension}, Allowed types: {', '.join(allowd_extensions)}")
    loader = DOCUMENT_LOADER_MAPPING[file_extension](file_path)  # type: ignore
    documents = await loader.aload()
    splits = text_splitter.split_documents(documents)
    logger.info(f"Successfully loaded and split {file_path} into {len(splits)} chunks.")

    return splits


async def index_document_to_pgvector(file_path: Path, document_id: UUID, thread_id: UUID, user_id: UUID) -> list[str]:
    """Index a document to Pinecone."""

    logger.info(f"Starting indexing for document: {file_path} with document_id: {document_id}")
    splits = await _load_and_split_documents(file_path)
    for split in splits:
        split.metadata["id"] = str(uuid4())
        split.metadata["file_name"] = file_path.name
        split.metadata["document_id"] = str(document_id)
        split.metadata["thread_id"] = str(thread_id)
        split.metadata["user_id"] = str(user_id)

    try:
        doc_ids = await asyncio.to_thread(
            vector_store.add_documents,
            splits,
            ids=[split.metadata["id"] for split in splits],
        )
        logger.info(
            f"Successfully indexed {len(splits)} chunks for document {file_path} (document_id: {document_id}) to Pinecone."
        )
        return doc_ids
    except Exception as e:
        logger.error(f"Error adding documents to Pinecone: {e}")
        raise


async def search_documents_in_pgvector(query: str = "", k: int = 1, filter: dict | None = None) -> list[Document]:
    """Search documents in Pinecone based on query, k and filter."""

    logger.info(f"Search documents with query: {query}, k: {k}, filter: {filter} in Pinecone.")
    documents = await asyncio.to_thread(vector_store.similarity_search, query, k=k, filter=filter)
    if not documents:
        logger.info(f"No documents found for query: {query} and filter: {filter} in Pinecone.")
    else:
        logger.info(f"Found {len(documents)} document chunks for query: {query} and filter: {filter} in Pinecone.")
    return documents


async def delete_document_from_pgvector(document_ids: list[str]) -> None:
    """Delete documents from Pinecone based on Document ID"""

    logger.info(f"Attempting to delete {len(document_ids)} document chunks from Pinecone.")
    await asyncio.to_thread(vector_store.delete, ids=document_ids)
    logger.info(f"Successfully deleted {len(document_ids)} document chunks from Pinecone.")


async def delete_documents_by_filter(filter: dict) -> None:
    """Delete all vectors matching metadata without requiring an embedding query."""

    logger.info(f"Attempting to delete Pinecone vectors matching filter: {filter}")
    await asyncio.to_thread(vector_store.delete, filter=filter)
    logger.info(f"Successfully deleted Pinecone vectors matching filter: {filter}")
