"""
response_nodes.py

Two new nodes to append to the existing agent_graph pipeline:
  aggregate_results -> response_generation -> final_response -> END

Nothing in agent_graph.py, tools.py, or persona.py is changed.
"""

from langchain_core.messages import AIMessage, SystemMessage
from pinecone import Pinecone
import httpx

from app.config import settings
from app.chat.langgraph_agent import create_model

llm = create_model("qwen/qwen3.7-flash")

# ---------------------------------------------------------------------------
# Pinecone legal/policy retriever — initialized once at startup
# ---------------------------------------------------------------------------
_pc = Pinecone(api_key=settings.pinecone_api_key.get_secret_value())
_law_index = _pc.Index(settings.pinecone_index_name)


def _embed_query(text: str) -> list[float]:
    """
    Embed a single string using the Nvidia model via OpenRouter.
    Sends input as a plain string (not a list) — required by Nvidia's API.
    """
    response = httpx.post(
        f"{settings.embeddings_base_url}/embeddings",
        headers={
            "Authorization": f"Bearer {settings.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        },
        json={
            "model": settings.embeddings_model_name,
            "input": text,   # single string — Nvidia does not accept a list
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


def _retrieve_laws(query: str, top_k: int = 4) -> list[dict]:
    """Embed the query and fetch the most relevant law chunks from Pinecone."""
    query_vector = _embed_query(query)
    results = _law_index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )
    return [match["metadata"] for match in results.get("matches", [])]


def _format_law_context(law_chunks: list[dict]) -> str:
    """Turn retrieved law chunks into a readable block for the prompt."""
    if not law_chunks:
        return "No specific laws retrieved."

    lines = []
    for chunk in law_chunks:
        priority_flag = " ⚠ HIGH PRIORITY" if chunk.get("priority") == "HIGH" else ""
        lines.append(
            f"[{chunk.get('law_name', '')} | Art.{chunk.get('article', '')} "
            f"§{chunk.get('paragraph', '')}]{priority_flag}\n"
            f"{chunk.get('summary', chunk.get('text_preview', ''))}"
        )
    return "\n\n".join(lines)


def _has_high_priority_law(law_chunks: list[dict]) -> bool:
    return any(c.get("priority") == "HIGH" for c in law_chunks)


# ---------------------------------------------------------------------------
# Node 1: response_generation
# ---------------------------------------------------------------------------
def response_generation_node(state):
    query              = state["query"]
    aggregated_context = state.get("aggregated_context", "")
    persona_prompt     = state.get("persona_prompt", "")

    law_chunks    = _retrieve_laws(query)
    law_context   = _format_law_context(law_chunks)
    high_priority = _has_high_priority_law(law_chunks)

    # System message — persona + high-priority legal flag if needed
    system_text = "You are a financial debt advisor.\n"
    if persona_prompt:
        system_text += f"\n{persona_prompt}\n"
    if high_priority:
        system_text += (
            "\n⚠ One or more HIGH PRIORITY legal restrictions apply to this query. "
            "You MUST address them explicitly before any recommendation."
        )

    tool_section = (
        f"TOOL RESULTS (financial data computed for this user):\n{aggregated_context}"
        if aggregated_context
        else "TOOL RESULTS: None — this is a general knowledge query."
    )

    user_prompt = f"""{tool_section}

APPLICABLE LAWS & POLICIES:
{law_context}

USER QUESTION:
{query}

Instructions:
- Answer using the tool results and legal context above.
- If a HIGH PRIORITY law applies (e.g. age restriction, eligibility rule), state it first.
- Cite the law name and article when referencing a legal point.
- Tailor tone to the user persona in the system prompt.
- Do not fabricate numbers not present in the tool results.
- Keep the response concise and actionable.
"""

    messages = [
        SystemMessage(content=system_text),
        *state.get("messages", []),
        {"role": "user", "content": user_prompt},
    ]

    response = llm.invoke(messages)

    return {
        "generated_response": response.content,
        "law_chunks": law_chunks,
    }


# ---------------------------------------------------------------------------
# Node 2: final_response
# ---------------------------------------------------------------------------
def final_response_node(state):
    """Packages the generated text as an AIMessage into conversation history."""
    generated = state.get("generated_response", "")
    return {
        "messages": [AIMessage(content=generated)],
        "final_response": generated,
    }