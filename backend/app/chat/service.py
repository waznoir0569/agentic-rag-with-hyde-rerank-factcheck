from collections.abc import AsyncIterator
from uuid import UUID, uuid4

from app.db.checkpointer import get_checkpointer
from fastapi import HTTPException
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from .langgraph_agent import build_retrival_graph
from .schemas import Message, PromptInput


async def simple_chat_stream(prompt_input: PromptInput) -> AsyncIterator:
    """Run the same advisory graph for users without an authenticated thread.

    The generated identities provide request-scoped context to tools without
    creating an authenticated user or durable application thread.
    """
    return await chat_stream(uuid4(), prompt_input, uuid4())


async def chat_stream(thread_id: UUID, prompt_input: PromptInput, user_id: UUID) -> AsyncIterator:
    """
    Streams the agent's execution steps and final response.
    """
    config = RunnableConfig(configurable={"thread_id": str(thread_id), "user_id": str(user_id)})
    checkpointer = await get_checkpointer()
    graph = build_retrival_graph(checkpointer, prompt_input.model_name)

    return graph.astream(
        input={
            "user_query": prompt_input.prompt,
            "persona_prompt": prompt_input.persona_prompt,
            "messages": [HumanMessage(content=prompt_input.prompt)],
            "tool_messages": [],
            "tool_plan": {},
            "missing_info": {},
            "law_chunks": [],
            "system_prompt": "",
            "final_answer": "",
            "fact_check_attempts": 0,
            "fact_check_feedback": [],
        },
        config=config,
        stream_mode=["updates", "messages"],
    )


async def get_chat_history(thread_id: UUID, user_id: UUID) -> list[Message]:
    config = RunnableConfig(configurable={"thread_id": str(thread_id), "user_id": str(user_id)})
    checkpointer = await get_checkpointer()
    checkpoint = await checkpointer.aget(config)

    if checkpoint is None:
        raise HTTPException(status_code=404, detail="Chat history not found")

    all_messages = checkpoint.get("channel_values", {}).get("messages", [])
    messages = [
        Message(role=message.type, content=message.content)
        for message in all_messages
        if message.content and message.type in ["human", "ai"]
    ]

    return messages
