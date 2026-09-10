import json
from typing import Any, AsyncGenerator, AsyncIterable

from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from loguru import logger
from pydantic import BaseModel, Field


class PromptInput(BaseModel):
    prompt: str
    model_name: str
    persona_prompt: str = ""


class PersonaQuestionnaire(BaseModel):
    """Likert-scale answers used for the pre-chat persona assignment."""

    debtrules3: int = Field(ge=1, le=5)
    debtnorms8: int = Field(ge=1, le=5)
    debtrules16: int = Field(ge=1, le=5)
    debtpers5: int = Field(ge=1, le=5)
    debtnorms9: int = Field(ge=1, le=5)
    debtpers9: int = Field(ge=1, le=5)
    debtpers11: int = Field(ge=1, le=5)
    debtrules9: int = Field(ge=1, le=5)


class Message(BaseModel):
    role: str
    content: str


class ChatStreamResponse(StreamingResponse):
    """
    It processes the LangGraph stream in different stream modes ("updates", "messages") and formats the data
    into JSON objects before sending them to the client.
    """

    def __init__(self, astream: AsyncIterable[dict[str, Any | Any]], **kwargs):
        super().__init__(content=self.process_stream(astream), **kwargs)

    async def process_stream(self, astream: AsyncIterable[dict[str, Any | Any]]) -> AsyncGenerator[str, Any]:
        async for stream_mode, chunk in astream:
            if stream_mode == "messages":
                yield self._handle_messages_stream(chunk)  # type: ignore

            elif stream_mode == "updates":
                async for formatted_chunk in self._handle_updates_stream(chunk):  # type: ignore
                    yield formatted_chunk

    def _handle_messages_stream(self, chunk: tuple[AIMessageChunk | AIMessage, Any]) -> str:
        message = chunk[0]
        metadata = chunk[1] if len(chunk) > 1 and isinstance(chunk[1], dict) else {}
        node_name = metadata.get("langgraph_node")

        # Structured planner, extraction, law-gate, HyDE, and fact-check calls
        # also appear in LangGraph's messages stream. Only expose answer text.
        if node_name not in {"generate_answer", "attach_disclaimer"}:
            return ""

        # Token streaming emits AIMessageChunk values. The corresponding
        # complete AIMessage is also present in the messages stream, but must
        # not be emitted or the frontend will display the answer twice.
        if isinstance(message, AIMessageChunk) and message.content:
            response = {"type": "llm_chunk", "content": str(message.content)}
            return json.dumps(response) + "\n"
        return ""

    async def _handle_updates_stream(self, chunk: dict[str, Any]) -> AsyncGenerator[str, Any]:
        for node_name, node_output in chunk.items():
            for message_key in ("messages", "tool_messages"):
                if message_key not in node_output:
                    continue
                if not node_output[message_key]:
                    continue

                message = node_output[message_key][-1]

                if isinstance(message, AIMessage):
                    if message.tool_calls:
                        for tool_call in message.tool_calls:
                            response = {"type": "tool_call", "name": tool_call["name"], "args": tool_call["args"]}
                            yield json.dumps(response) + "\n"
                    elif node_name == "attach_disclaimer" and message.content:
                        response = {"type": "llm_chunk", "content": str(message.content)}
                        yield json.dumps(response) + "\n"

                elif isinstance(message, ToolMessage):
                    response = {"type": "tool_result", "name": message.name, "content": message.content}
                    yield json.dumps(response) + "\n"

                else:
                    response = {"type": message.type, "content": message.content}
                    logger.debug(f"Unknown message type: {message.type} - {json.dumps(response)}")
