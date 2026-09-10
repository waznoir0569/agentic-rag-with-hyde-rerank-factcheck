from collections.abc import Sequence
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, Thread

from .schemas import DocumetCreate


async def _ensure_thread_access(thread_id: UUID, user_id: UUID, session: AsyncSession) -> Thread:
    statement = select(Thread).where(Thread.id == thread_id)
    result = await session.execute(statement)
    db_thread = result.scalar_one_or_none()

    if db_thread is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread with ID {thread_id} not found.",
        )
    if db_thread.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this thread.",
        )

    return db_thread


async def get_documents(thread_id: UUID, user_id: UUID, session: AsyncSession) -> Sequence[Document]:
    await _ensure_thread_access(thread_id, user_id, session)
    statement = select(Document).where(Document.thread_id == thread_id)
    result = await session.execute(statement)
    return result.scalars().all()


async def insert_document(document_data: DocumetCreate, user_id: UUID, session: AsyncSession) -> Document:
    await _ensure_thread_access(document_data.thread_id, user_id, session)
    new_document = Document(**document_data.model_dump())
    session.add(new_document)
    await session.commit()
    await session.refresh(new_document)
    return new_document


async def delete_document(document_id: UUID, user_id: UUID, session: AsyncSession) -> None:
    db_document = await session.get(Document, document_id)
    if db_document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found.",
        )

    await _ensure_thread_access(db_document.thread_id, user_id, session)

    await session.delete(db_document)
    await session.commit()
