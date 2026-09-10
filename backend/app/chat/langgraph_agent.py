from __future__ import annotations

from typing import Annotated, Optional, TypedDict
from uuid import uuid4

from app.config import settings
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from loguru import logger
from pydantic import BaseModel, Field

from .tools import calculate_amortization, calculate_financial_wellbeing, get_macro_context

FINANCIAL_TOOLS = [
    calculate_financial_wellbeing,
    calculate_amortization,
    get_macro_context,
]

PLANNER_MODEL = "qwen/qwen3.7-flash"
FACT_CHECK_MODEL = "qwen/qwen3-max"


class PipelineState(TypedDict):
    user_query: str
    persona_prompt: str
    messages: Annotated[list[BaseMessage], add_messages]
    tool_messages: list[BaseMessage]
    tool_plan: dict
    missing_info: dict
    law_chunks: list
    system_prompt: str
    final_answer: str
    fact_check_attempts: int
    fact_check_feedback: list[str]


class ToolPlan(BaseModel):
    calculate_financial_wellbeing: bool = Field(
        description=(
            "True if the user's question is asking to assess their financial "
            "wellbeing, savings health, or debt burden - REGARDLESS of whether "
            "they've already given income/expenses/savings/debt figures. "
            "Downstream steps will extract whatever numbers are present and ask "
            "for anything missing, so flag this based on intent, not on whether "
            "every number is already in the message."
        )
    )
    calculate_amortization: bool = Field(
        description=(
            "True if the user's question is asking about a loan's repayment "
            "schedule, monthly payment, or total interest - REGARDLESS of "
            "whether the principal, rate, and term are all already given. "
            "Flag this based on intent, not on data completeness."
        )
    )
    get_macro_context: bool = Field(
        description=(
            "True only if the question needs current macroeconomic indicators "
            "(inflation, unemployment, interest rates, GDP growth) for a "
            "specific country to answer well."
        )
    )
    reasoning: str = Field(description="One sentence explaining the decision.")


class FinancialWellbeingArgs(BaseModel):
    monthly_income: Optional[float] = Field(default=None)
    monthly_expenses: Optional[float] = Field(default=None)
    savings_balance: Optional[float] = Field(default=None)
    total_debt_amount: Optional[float] = Field(default=None)


class AmortizationArgs(BaseModel):
    principal: Optional[float] = Field(default=None)
    annual_interest_rate: Optional[float] = Field(default=None)
    term_months: Optional[int] = Field(default=None)


class MacroContextArgs(BaseModel):
    country_code: Optional[str] = Field(default=None)


class FactCheckResult(BaseModel):
    grounded: bool = Field(
        description=(
            "True only if every specific legal citation and every specific number in the answer "
            "is directly supported by the source material."
        )
    )
    unsupported_claims: list[str] = Field(default_factory=list)


TOOL_ARG_SCHEMAS = {
    "calculate_financial_wellbeing": FinancialWellbeingArgs,
    "calculate_amortization": AmortizationArgs,
    "get_macro_context": MacroContextArgs,
}

REQUIRED_FIELDS = {
    "calculate_financial_wellbeing": ["monthly_income", "monthly_expenses", "savings_balance"],
    "calculate_amortization": ["principal", "annual_interest_rate", "term_months"],
    "get_macro_context": [],
}

MAX_FACT_CHECK_RETRIES = 2

PLANNER_SYSTEM_PROMPT = """
You are the planning layer for a financial advisory assistant with 3 tools:

- calculate_financial_wellbeing: scores income/expenses/savings/debt.
- calculate_amortization: builds a loan repayment schedule.
- get_macro_context: fetches current macroeconomic indicators for a country.

You will see the full conversation so far, not just the latest message.
Decide which of these (if any) the conversation as a whole needs, based on
what the user is asking for - NOT on whether every number has been supplied
yet, and NOT only on the very last message if earlier messages already
established what's being discussed (e.g. if the user asked about a loan
earlier and is now only supplying additional figures, the loan tools are
still relevant). Flag a tool True even if some or all of its required
figures are still missing; a later step extracts whatever values are
present anywhere in the conversation and asks the user directly for
anything still missing. Do not flag a tool "just in case" when nothing in
the conversation calls for it, and do not guess or fill in numeric values
yourself - you are only deciding WHICH tools are relevant, not extracting
their arguments.
"""

FEATURE_EXTRACTION_PROMPT = (
    "Extract the arguments for {tool_name} from the conversation below. The "
    "user may have given different values across different messages - use "
    "the most recent value for a field if it was updated later, but don't "
    "ignore a value just because it was stated in an earlier message.\n\n"
    "Rules:\n"
    "1. Only fill in a value the user explicitly stated as a number for that "
    "exact field, at any point in the conversation. Do not infer, estimate, "
    "round, or derive a value from other numbers (e.g. do not compute "
    "expenses from income minus savings).\n"
    "2. If a value was never explicitly stated anywhere in the conversation, "
    "leave it null - this is expected and correct when the user hasn't given "
    "every figure yet.\n"
    "3. Never substitute a placeholder, average, or default value for a "
    "field the user didn't mention.\n"
    "4. Convert units yourself when unambiguous (e.g. \"35 years\" for a "
    "loan term means 420 months) - don't ask the user to do the conversion."
)

FACT_CHECK_PROMPT = """
You are a strict fact-checker. You are shown SOURCE MATERIAL and an ANSWER
that was supposed to be based on it. Check every specific legal citation
and every specific number in the answer against the source material only.

If the answer states a citation or figure that does not appear in the source
material, mark it unsupported. Do not use your own general knowledge to
"confirm" a claim.
"""


def create_model(model_name: str, streaming: bool = False) -> BaseChatModel:
    """Create a chat model using the configured provider."""

    return init_chat_model(
        model=model_name,
        model_provider=settings.model_provider,
        api_key=settings.api_key,
        base_url=settings.model_base_url or None,
        streaming=streaming,
    )


def _format_law_sources(law_chunks: list) -> str:
    lines = []
    for chunk in law_chunks:
        meta = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
        lines.append(f"- [{meta.get('law_name', '')} {meta.get('article', '')}] {meta.get('condition', '')}")
    return "\n".join(lines)


def _build_system_prompt(state: PipelineState) -> str:
    sections: list[str] = []

    persona_prompt = (state.get("persona_prompt") or "").strip()
    if persona_prompt:
        sections.extend([persona_prompt, ""])

    tool_messages = state.get("tool_messages", [])
    tool_outputs = {
        msg.name: msg.content
        for msg in tool_messages
        if isinstance(msg, ToolMessage)
    }

    if tool_outputs:
        sections.append(
            "FINANCIAL CONTEXT (computed for this specific query - use these "
            "exact numbers, do not recompute or re-derive them). If any field "
            "in here is null or missing, that specific data point could not "
            "be fetched - say so explicitly (e.g. 'unemployment data wasn't "
            "available') rather than treating it as zero or omitting the gap:"
        )
        for name, result in tool_outputs.items():
            sections.append(f"- {name} -> {result}")
        sections.append("")

    missing_info = state.get("missing_info", {})
    if missing_info:
        sections.append(
            "MISSING REQUIRED INPUTS - HIGHEST PRIORITY INSTRUCTION:\n"
            "The user asked for a calculation but did not give every value it "
            "needs. You must NOT guess, estimate, or assume any of the values "
            "listed below, and you must NOT attempt or imply a result for that "
            "specific calculation. Instead, ask the user directly and "
            "explicitly for exactly these values, as clear questions, before "
            "that calculation can be completed:"
        )
        for tool_name, fields in missing_info.items():
            friendly_fields = ", ".join(field.replace("_", " ") for field in fields)
            sections.append(f"- For {tool_name.replace('_', ' ')}, ask for: {friendly_fields}")
        sections.append("")

    law_chunks = state.get("law_chunks", [])
    if law_chunks:
        sections.append(
            "CANDIDATE LAW EXCERPTS (retrieved for this query - these are "
            "candidates, not a guarantee of relevance; many may be tangential "
            "to what the user is actually asking, e.g. corporate licensing or "
            "capital requirements for lending institutions are usually NOT "
            "relevant to whether an individual should take a loan):"
        )
        for chunk in law_chunks:
            meta = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
            sections.append(
                f"- [{meta.get('law_name', 'Unknown law')} {meta.get('article', '')}] "
                f"{meta.get('condition', '')}"
            )
        sections.append("")

    sections.append(
        "HOW TO STRUCTURE YOUR ANSWER (applies whether or not laws were "
        "retrieved):\n"
        "1. Start with ONE direct sentence that actually answers the "
        "question - a clear recommendation or verdict, not a recap of the "
        "data you were given. If the available information genuinely isn't "
        "enough for a firm verdict, say what your best-supported answer is "
        "and name the one or two specific things that would change it - "
        "don't open with a list of caveats before giving any answer at all.\n"
        "2. Follow with a few short paragraphs of reasoning. Each point "
        "should tie to something concrete: a computed number, or a specific "
        "law excerpt that is DIRECTLY on point for the user's actual "
        "question. Silently ignore any candidate law excerpt above that "
        "isn't directly relevant - being retrieved does not obligate you to "
        "mention, cite, or comment on it.\n"
        "3. Cite a law only at the exact point in your reasoning where you "
        "rely on it, in the form [law_name article], copied exactly from the "
        "excerpts above - never invent a citation not listed above, and "
        "never cite the same law twice for the same point.\n"
        "4. Do NOT add a separate 'Sources', 'Citations used', or 'Laws "
        "referenced' section at the end - every citation belongs inline, "
        "at first use, and nowhere else.\n"
        "5. Match the length and depth of your answer to the question. A "
        "yes/no or comparison question gets a focused verdict with its "
        "strongest supporting reasoning - not an inventory of everything "
        "that was retrieved. If none of the retrieved excerpts are actually "
        "relevant to this specific question, say briefly that you don't have "
        "a verified legal source on point, rather than presenting unrelated "
        "provisions as if they mattered.\n"
        "6. If there are missing inputs (above), weave the request for them "
        "into the answer concisely - don't let it turn into its own lengthy "
        "section."
    )

    return "\n".join(section for section in sections if section is not None).strip()

# def _build_system_prompt(state: PipelineState) -> str:
#     sections: list[str] = []

#     persona_prompt = (state.get("persona_prompt") or "").strip()
#     if persona_prompt:
#         sections.extend([persona_prompt, ""])

#     tool_messages = state.get("tool_messages", [])
#     tool_outputs = {
#         msg.name: msg.content
#         for msg in tool_messages
#         if isinstance(msg, ToolMessage)
#     }

#     if tool_outputs:
#         sections.append(
#             "FINANCIAL CONTEXT (computed for this specific query - use these "
#             "exact numbers, do not recompute or re-derive them). If any field "
#             "in here is null or missing, that specific data point could not "
#             "be fetched - say so explicitly (e.g. 'unemployment data wasn't "
#             "available') rather than treating it as zero or omitting the gap:"
#         )
#         for name, result in tool_outputs.items():
#             sections.append(f"- {name} -> {result}")
#         sections.append("")

#     missing_info = state.get("missing_info", {})
#     if missing_info:
#         sections.append(
#             "MISSING REQUIRED INPUTS - HIGHEST PRIORITY INSTRUCTION:\n"
#             "The user asked for a calculation but did not give every value it "
#             "needs. You must NOT guess, estimate, or assume any of the values "
#             "listed below, and you must NOT attempt or imply a result for that "
#             "specific calculation. Instead, ask the user directly and "
#             "explicitly for exactly these values, as clear questions, before "
#             "that calculation can be completed:"
#         )
#         for tool_name, fields in missing_info.items():
#             friendly_fields = ", ".join(field.replace("_", " ") for field in fields)
#             sections.append(f"- For {tool_name.replace('_', ' ')}, ask for: {friendly_fields}")
#         sections.append("")

#     law_chunks = state.get("law_chunks", [])
#     if law_chunks:
#         sections.append("RELEVANT LAWS (retrieved specifically for this query):")
#         for chunk in law_chunks:
#             meta = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
#             sections.append(
#                 f"- [{meta.get('law_name', 'Unknown law')} {meta.get('article', '')}] "
#                 f"{meta.get('condition', '')}"
#             )
#         sections.append("")
#         sections.append(
#             "LEGAL/RECOMMENDATION REQUIREMENTS:\n"
#             "1. Give a clear, direct recommendation or answer to the user's "
#             "question - don't just summarize the laws, take a position.\n"
#             "2. Justify that recommendation with explicit reasoning that ties "
#             "back to the specific laws above (and to the computed financial "
#             "numbers above, if any).\n"
#             "3. Every legal or eligibility claim must be immediately followed "
#             "by a citation in the exact form [law_name article], copied from "
#             "the law entries above - never invent a law name, article number, "
#             "or citation that isn't listed above.\n"
#             "4. If the laws above don't fully answer the question, say so "
#             "explicitly rather than filling the gap with unlabeled general "
#             "knowledge presented as law."
#         )
#         sections.append("")
#     else:
#         sections.append(
#             "No laws were retrieved for this query. Do not state or imply any "
#             "specific legal citation, article number, or regulatory claim - if "
#             "legal/eligibility context would normally matter here, say that "
#             "you don't have a verified legal source for it rather than "
#             "guessing."
#         )
#         sections.append("")

#     sections.append(
#         "Answer using the persona's tone/style above. Address the "
#         "missing-input questions (if any) and the legal recommendation (if "
#         "laws were retrieved) together in one coherent response - don't ask "
#         "about missing figures in a way that ignores or delays the legal "
#         "answer you're already able to give, and vice versa."
#     )

#     return "\n".join(section for section in sections if section is not None).strip()


def build_retrival_graph(checkpointer: BaseCheckpointSaver, model_name: str) -> CompiledStateGraph:
    """Build the finance/law advisory graph."""

    from .law_retrieval import retrieve_laws, condense_query

    planner_llm = create_model(PLANNER_MODEL)
    extractor_llm = create_model(PLANNER_MODEL)
    answer_llm = create_model(model_name)
    fact_check_llm = create_model(FACT_CHECK_MODEL)

    planner_structured = planner_llm.with_structured_output(ToolPlan)
    fact_check_structured = fact_check_llm.with_structured_output(FactCheckResult)
    tool_executor = ToolNode(FINANCIAL_TOOLS, messages_key="tool_messages")

    def planner_node(state: PipelineState) -> dict:
        conversation = state.get("messages", [])
        try:
            plan: ToolPlan = planner_structured.invoke(
                [SystemMessage(content=PLANNER_SYSTEM_PROMPT), *conversation]
            )
        except Exception as exc:
            logger.warning(f"planner_node: structured output failed ({exc}); defaulting to law retrieval")
            plan = ToolPlan(
                calculate_financial_wellbeing=False,
                calculate_amortization=False,
                get_macro_context=False,
                reasoning="Planner output was unavailable; use the law retrieval path safely.",
            )
        return {"tool_plan": plan.model_dump()}

    def route_after_planner(state: PipelineState) -> str:
        plan = state["tool_plan"]
        needs_any_tool = any(
            plan[name]
            for name in ("calculate_financial_wellbeing", "calculate_amortization", "get_macro_context")
        )
        return "feature_extraction" if needs_any_tool else "law_retrieval"

    def feature_extraction_node(state: PipelineState) -> dict:
        plan = state["tool_plan"]
        if isinstance(plan, BaseModel):
            plan = plan.model_dump()

        conversation = state.get("messages", [])
        tool_calls = []
        missing_info = {}

        for tool_name, schema in TOOL_ARG_SCHEMAS.items():
            if not plan.get(tool_name):
                continue

            extractor = extractor_llm.with_structured_output(schema)
            try:
                args: BaseModel = extractor.invoke(
                    [
                        SystemMessage(content=FEATURE_EXTRACTION_PROMPT.format(tool_name=tool_name)),
                        *conversation,
                    ]
                )
            except Exception as exc:
                logger.warning(f"feature_extraction_node: failed to extract {tool_name}: {exc}")
                missing_info[tool_name] = REQUIRED_FIELDS[tool_name]
                continue

            args_dict = args.model_dump()
            missing = [field for field in REQUIRED_FIELDS[tool_name] if args_dict.get(field) is None]
            if missing:
                missing_info[tool_name] = missing
                continue

            clean_args = {key: value for key, value in args_dict.items() if value is not None}
            tool_calls.append(
                {
                    "name": tool_name,
                    "args": clean_args,
                    "id": str(uuid4()),
                }
            )

        return {
            "tool_messages": [AIMessage(content="", tool_calls=tool_calls)] if tool_calls else [],
            "missing_info": missing_info,
        }

    def law_retrieval_node(state: PipelineState) -> dict:
        conversation = state.get("messages", [])
        try:
            if len(conversation) > 1:
                query = condense_query(conversation)
                logger.info(f"law_retrieval_node: condensed query = {query!r}")
            else:
                query = state["user_query"]
            law_chunks = retrieve_laws(query)
            logger.info(f"law_retrieval_node: retrieved {len(law_chunks)} law chunks for query {query!r}")
        except Exception as exc:
            logger.warning(f"law_retrieval_node: retrieval failed, continuing without laws: {exc}")
            law_chunks = []
        return {"law_chunks": law_chunks}   

    def assemble_context_node(state: PipelineState) -> dict:
        system_prompt = _build_system_prompt(state)
        print("\n========== FINAL SYSTEM PROMPT ==========\n")
        print(system_prompt)
        print("\n========== USER QUERY ==========\n")
        print(state["user_query"])
        print("\n========== PERSONA PROMPT ==========\n")
        print(state['persona_prompt'])
        print("\n=========================================\n")
        return {"system_prompt": _build_system_prompt(state)}

    def generate_answer_node(state: PipelineState) -> dict:
        messages = [SystemMessage(content=state["system_prompt"])]

        feedback = state.get("fact_check_feedback") or []
        if feedback:
            messages.append(
                SystemMessage(
                    content=(
                        "Your previous answer included claims not supported by the retrieved sources: "
                        + "; ".join(feedback)
                        + ". Rewrite the answer and remove or correct those claims."
                    )
                )
            )

        messages.extend(state.get("messages", []))
        response = answer_llm.invoke(messages)
        content = str(response.content or "")

        print("FINAL RESPONSE: ", content)
        return {
            "final_answer": content,
            "messages": [AIMessage(content=content, id=str(uuid4()))],
        }

    def fact_check_node(state: PipelineState) -> dict:
        source_material = _format_law_sources(state.get("law_chunks", []))
        try:
            result: FactCheckResult = fact_check_structured.invoke(
                [
                    SystemMessage(content=FACT_CHECK_PROMPT),
                    HumanMessage(
                        content=f"SOURCE MATERIAL:\n{source_material}\n\nANSWER:\n{state['final_answer']}"
                    ),
                ]
            )
        except Exception as exc:
            logger.warning(f"fact_check_node: structured output failed ({exc}); attaching disclaimer")
            return {
                "fact_check_attempts": MAX_FACT_CHECK_RETRIES,
                "fact_check_feedback": ["Automated fact-checking was unavailable."],
            }

        return {
            "fact_check_attempts": state.get("fact_check_attempts", 0) + 1,
            "fact_check_feedback": [] if result.grounded else result.unsupported_claims,
        }

    def attach_disclaimer_node(state: PipelineState) -> dict:
        disclaimer = (
            "\n\n---\n"
            "*Some legal details above could not be fully verified against the retrieved sources. "
            "Please confirm specifics with a licensed legal or financial advisor before relying on them.*"
        )
        return {
            "final_answer": state["final_answer"] + disclaimer,
            "messages": [AIMessage(content=disclaimer, id=str(uuid4()))],
        }
    
    def route_after_feature_extraction(state: PipelineState) -> str:
        """Nothing to execute if every flagged tool was missing required fields -
        skip tool_executor entirely rather than handing it an empty list."""
        tool_messages = state.get("tool_messages") or []
        has_pending_tool_call = any(
            isinstance(msg, AIMessage) and msg.tool_calls for msg in tool_messages
        )
        return "tool_executor" if has_pending_tool_call else "law_retrieval"

    def route_after_generate_answer(state: PipelineState) -> str:
        return "fact_check" if state.get("law_chunks") else "end"

    def route_after_fact_check(state: PipelineState) -> str:
        if not state.get("fact_check_feedback"):
            return "end"
        if state.get("fact_check_attempts", 0) >= MAX_FACT_CHECK_RETRIES:
            return "disclaim"
        return "retry"

    graph = StateGraph(PipelineState)
    graph.add_node("planner", planner_node)
    graph.add_node("feature_extraction", feature_extraction_node)
    graph.add_node("tool_executor", tool_executor)
    graph.add_node("law_retrieval", law_retrieval_node)
    graph.add_node("assemble_context", assemble_context_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("fact_check", fact_check_node)
    graph.add_node("attach_disclaimer", attach_disclaimer_node)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {"feature_extraction": "feature_extraction", "law_retrieval": "law_retrieval"},
    )
    # graph.add_edge("feature_extraction", "tool_executor")
    graph.add_conditional_edges("feature_extraction", route_after_feature_extraction, {"tool_executor": "tool_executor", "law_retrieval": "law_retrieval"})
    graph.add_edge("tool_executor", "law_retrieval")
    graph.add_edge("law_retrieval", "assemble_context")
    graph.add_edge("assemble_context", "generate_answer")
    graph.add_conditional_edges(
        "generate_answer",
        route_after_generate_answer,
        {"fact_check": "fact_check", "end": END},
    )
    graph.add_conditional_edges(
        "fact_check",
        route_after_fact_check,
        {"end": END, "retry": "generate_answer", "disclaim": "attach_disclaimer"},
    )
    graph.add_edge("attach_disclaimer", END)

    return graph.compile(checkpointer=checkpointer)


build_retrieval_graph = build_retrival_graph
