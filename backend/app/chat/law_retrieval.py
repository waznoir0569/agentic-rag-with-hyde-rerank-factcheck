"""
law_retrieval.py

Retrieval pipeline for the legal/policy law database:
  decompose query -> HyDE per sub-query -> dense (Pinecone) + BM25 fusion
  -> dedup across sub-queries -> rerank (Cohere via OpenRouter) -> top-N

Exposes a single entry point for callers: retrieve_laws(query, top_n=8)
response_generation_node (in response_nodes.py) is the intended caller:

    from app.chat.law_retrieval import retrieve_laws
    law_chunks = retrieve_laws(query)

Everything else in this module is an internal implementation detail.
"""

import json
import pickle
from typing import Any, Dict, List

import requests
from pinecone import Pinecone

from app.config import settings
from app.chat.langgraph_agent import create_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Must match the model used to build the Pinecone index in the ingestion
# script (ingest_rules_to_pinecone.py). If these ever drift apart, dense
# search will silently return meaningless nearest-neighbors instead of
# erroring — there's no dimension check that would catch a same-size,
# different-model mismatch.
LAW_EMBEDDING_MODEL = "openai/text-embedding-3-small"
RERANK_MODEL = "cohere/rerank-v3.5"

OPENROUTER_API_KEY = settings.api_key.get_secret_value()

MAX_SUB_QUERIES = 3
DENSE_TOP_K = 10
BM25_TOP_K = 10
RRF_K = 60
PRE_RERANK_CUTOFF = 20
DEFAULT_TOP_N = 4

# ---------------------------------------------------------------------------
# Module-level singletons — loaded once at import time, not per-call
# ---------------------------------------------------------------------------

_pc = Pinecone(api_key=settings.pinecone_api_key.get_secret_value())
_law_index = _pc.Index(settings.pinecone_index_name)

# settings.bm25_index_path must be an absolute path (add it to your config
# if it isn't there yet) — a relative path like "../bm25_rules.pkl" only
# resolves correctly from a notebook's cwd, not from an imported module in
# a running app.
with open(settings.bm25_index_path, "rb") as f:
    _bm25_store = pickle.load(f)

_bm25 = _bm25_store["bm25"]
_bm25_records = _bm25_store["records"]

_llm = create_model("qwen/qwen3.7-flash")


# --- add near the top of law_retrieval.py, alongside the other config ---

GATE_MODEL = "qwen/qwen3.7-flash"  # cheap/fast model — this is a binary check, not reasoning-heavy


def should_run_law_retrieval(query: str) -> bool:
    """
    Cheap pre-check: does this query plausibly touch anything with legal/
    regulatory relevance (eligibility, age, entity status, jurisdiction,
    disclosure, consumer protection, licensing, etc.)?

    Biased toward True on any ambiguity — a missed legal check is a worse
    failure than an unnecessary retrieval call. Never skip retrieval
    based on a parse failure or low model confidence; only skip when the
    query is clearly and purely computational/informational with no
    plausible legal angle.
    """

    GATE_SYSTEM_PROMPT = """
    You are a fast pre-filter deciding whether a user's message to a debt
    advisory assistant needs to be checked against a database of laws and
    financial regulations.

    Answer YES if the message involves, mentions, or could reasonably
    implicate ANY of:
    - age, minors, students, dependents
    - loan/credit eligibility, applying for credit or financing
    - business entities, startups, self-employment, LLCs, sole proprietors
    - a specific country, region, or jurisdiction
    - disclosure, consumer rights, fees, contracts, terms
    - immigration/residency status, employment status
    - any topic where a law, regulation, or eligibility rule might apply

    Answer NO only if the message is PURELY computational or general
    knowledge with no plausible legal or eligibility angle at all — e.g.
    "what's my monthly payment on a $20k loan at 6% for 5 years", "what
    does APR mean", "what's the inflation rate in Germany".

    When genuinely unsure, answer YES. A missed legal check is worse than
    an unnecessary one.

    Return ONLY valid JSON: {"needs_law_retrieval": true} or
    {"needs_law_retrieval": false}
    """

    try:
        response = _llm.invoke([
            ("system", GATE_SYSTEM_PROMPT),
            ("user", query)
        ])

        raw = response.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result = json.loads(raw)
        print('SHOULD RUN RETRIEVAL', result)
        return bool(result.get("needs_law_retrieval", True))

    except Exception as e:
        # Fail open — if the gate itself breaks, don't let that silently
        # skip a compliance-relevant check. Run the full pipeline instead.
        print(f"should_run_law_retrieval: gate failed ({e}), defaulting to running retrieval")
        return True

# ---------------------------------------------------------------------------
# Stage 1: Query decomposition
# ---------------------------------------------------------------------------

def decompose_query(query: str) -> List[str]:
    """Split a query into independent legal sub-questions (capped at
    MAX_SUB_QUERIES). Falls back to [query] on any parse failure so this
    never breaks the pipeline outright."""

    DECOMPOSER_SYSTEM_PROMPT = """
    You are a legal retrieval planner.

    Your job is to decompose a user question into independent legal issues
    that can be searched separately in a law database.

    Rules:
    1. Focus on legal eligibility, obligations, restrictions, requirements.
    2. Remove irrelevant personal details.
    3. Produce independent retrieval questions.
    4. Questions should be answerable by legal rules.
    5. Do not create questions for arithmetic calculations.
    6. Return ONLY valid JSON.

    Output schema:

    {
        "sub_queries": [
            "...",
            "...",
            "..."
        ]
    }
    """

    response = _llm.invoke([
        ("system", DECOMPOSER_SYSTEM_PROMPT),
        ("user", query)
    ])

    raw = response.content.strip()

    # Strip markdown code fences if the model wraps its output
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        result = json.loads(raw)
        sub_queries = result.get("sub_queries", [])
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"decompose_query: failed to parse sub_queries ({e}), falling back to original query")
        sub_queries = [query]

    sub_queries = sub_queries[:MAX_SUB_QUERIES]

    if not sub_queries:
        sub_queries = [query]

    return sub_queries


# ---------------------------------------------------------------------------
# Stage 2: HyDE passage generation
# ---------------------------------------------------------------------------

def generate_hyde_passage(sub_query: str) -> str:
    """Generate a hypothetical statute-style passage for a sub-query, used
    only as the embedding target for dense search (never shown to the
    user, never treated as an actual legal citation)."""

    HYDE_SYSTEM_PROMPT = """
    You are a legal information retrieval assistant.

    Given a legal search question, generate a hypothetical
    statute-style passage that could plausibly contain the
    legal rule needed to answer the question.

    This passage is ONLY for retrieval.

    Rules:
    1. Use formal legal terminology.
    2. Focus on legal conditions, requirements, restrictions,
       eligibility rules, thresholds, or obligations.
    3. Preserve facts explicitly stated in the query.
    4. Do not invent law numbers, article numbers, citations,
       authorities, or dates.
    5. Do not provide legal advice.
    6. Do not explain the answer.
    7. Return only the hypothetical legal passage.
    8. Keep it concise: 1-3 sentences.
    """

    response = _llm.invoke([
        ("system", HYDE_SYSTEM_PROMPT),
        ("user", sub_query)
    ])

    return response.content.strip()


# ---------------------------------------------------------------------------
# Stage 3: Dense + BM25 search
# ---------------------------------------------------------------------------

def get_embeddings(
    texts: List[str],
    api_key: str,
    model: str = LAW_EMBEDDING_MODEL
) -> List[List[float]]:

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "input": texts
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/embeddings",
        json=payload,
        headers=headers,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    return [item["embedding"] for item in data["data"]]


def dense_search(
    hyde_passage: str,
    top_k: int = DENSE_TOP_K
) -> List[Dict[str, Any]]:

    embedding = get_embeddings([hyde_passage], OPENROUTER_API_KEY)[0]

    result = _law_index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True
    )

    candidates = []

    for match in result["matches"]:
        candidates.append({
            "id": match["id"],
            "score": match["score"],
            "metadata": match.get("metadata", {}),
            "retrieval_method": "dense"
        })

    return candidates


def bm25_search(
    sub_query: str,
    top_k: int = BM25_TOP_K
) -> List[Dict[str, Any]]:

    query_tokens = sub_query.lower().split()
    scores = _bm25.get_scores(query_tokens)
    ranked_indices = scores.argsort()[::-1][:top_k]

    candidates = []

    for idx in ranked_indices:
        score = float(scores[idx])

        if score <= 0:
            continue

        record = _bm25_records[idx]

        candidates.append({
            "id": record["id"],
            "score": score,
            "metadata": record.get("metadata", {}),
            "retrieval_method": "bm25"
        })

    return candidates


# ---------------------------------------------------------------------------
# Stage 4: Reciprocal Rank Fusion (per sub-query)
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    k: int = RRF_K
) -> List[Dict[str, Any]]:
    """Combine dense and BM25 ranked results for ONE sub-query using
    Reciprocal Rank Fusion. Also dedupes within this sub-query when the
    same rule id appears in both lists."""

    scores: Dict[str, float] = {}
    result_data: Dict[str, Dict[str, Any]] = {}

    for rank, result in enumerate(dense_results, start=1):
        doc_id = result["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        result_data[doc_id] = result.copy()

    for rank, result in enumerate(bm25_results, start=1):
        doc_id = result["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        if doc_id not in result_data:
            result_data[doc_id] = result.copy()

    fused_results = []

    for doc_id, rrf_score in scores.items():
        result = result_data[doc_id]
        result["rrf_score"] = rrf_score
        fused_results.append(result)

    fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)

    return fused_results


# ---------------------------------------------------------------------------
# Stage 5: Dedup across sub-queries
# ---------------------------------------------------------------------------

def deduplicate_candidates(
    retrieved_candidates: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Pool fused results from all sub-queries into one list. On collision,
    keep the higher rrf_score and track every sub-query that surfaced the
    rule — a rule found by multiple sub-queries is a real relevance
    signal worth carrying into rerank."""

    pooled: Dict[str, Dict[str, Any]] = {}

    for result in retrieved_candidates:

        doc_id = result["id"]
        sub_query = result.get("sub_query")

        if doc_id not in pooled:
            pooled[doc_id] = {
                "id": doc_id,
                "metadata": result["metadata"],
                "rrf_score": result["rrf_score"],
                "sub_queries": [sub_query] if sub_query else [],
            }
        else:
            existing = pooled[doc_id]

            if result["rrf_score"] > existing["rrf_score"]:
                existing["rrf_score"] = result["rrf_score"]

            if sub_query and sub_query not in existing["sub_queries"]:
                existing["sub_queries"].append(sub_query)

    deduped = list(pooled.values())

    for record in deduped:
        record["num_subqueries_matched"] = len(record["sub_queries"])

    # Rules found by multiple sub-queries first, then by rrf_score
    deduped.sort(
        key=lambda r: (r["num_subqueries_matched"], r["rrf_score"]),
        reverse=True
    )

    return deduped


# ---------------------------------------------------------------------------
# Stage 6: Rerank (Cohere rerank-v3.5 via OpenRouter)
# ---------------------------------------------------------------------------

def rerank_via_openrouter(
    query: str,
    documents: List[str],
    api_key: str,
    model: str = RERANK_MODEL,
    top_n: int = DEFAULT_TOP_N
) -> List[Dict[str, Any]]:

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": top_n
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/rerank",
        json=payload,
        headers=headers,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    # Response shape: {"results": [{"index": int, "relevance_score": float}, ...]}
    return data.get("results", [])


def rerank_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
    top_n: int = DEFAULT_TOP_N
) -> List[Dict[str, Any]]:

    if not candidates:
        return []

    documents = []

    for c in candidates:
        meta = c["metadata"]
        documents.append(f"{meta.get('law_name', '')}: {meta.get('condition', '')}")

    try:
        results = rerank_via_openrouter(
            query=query,
            documents=documents,
            api_key=OPENROUTER_API_KEY,
            model=RERANK_MODEL,
            top_n=top_n
        )
    except Exception as e:
        print(f"rerank_candidates: OpenRouter rerank failed ({e}), falling back to pre-rerank order")
        return candidates[:top_n]

    reranked = []

    for r in results:
        idx = r.get("index")
        if idx is None or idx < 0 or idx >= len(candidates):
            continue
        record = candidates[idx].copy()
        record["relevance_score"] = r.get("relevance_score")
        reranked.append(record)

    if not reranked:
        return candidates[:top_n]

    return reranked


# ---------------------------------------------------------------------------
# Public entry point — the only function response_nodes.py should call
# ---------------------------------------------------------------------------

def retrieve_laws(query: str, top_n: int = DEFAULT_TOP_N) -> List[Dict[str, Any]]:
    """
    Full retrieval pipeline: decompose -> HyDE -> dense+BM25+RRF per
    sub-query -> dedup across sub-queries -> rerank -> top-N.

    Returns a list of records shaped:
        {
            "id": str,
            "metadata": {...law fields: law_name, article, condition,
                         priority, ...},
            "rrf_score": float,
            "sub_queries": [str, ...],
            "num_subqueries_matched": int,
            "relevance_score": float,   # from the reranker
        }
    """

    if not should_run_law_retrieval(query):
        return []

    sub_queries = decompose_query(query)

    all_fused: List[Dict[str, Any]] = []

    for sub_query in sub_queries:

        hyde_passage = generate_hyde_passage(sub_query)

        dense_results = dense_search(hyde_passage, top_k=DENSE_TOP_K)
        sparse_results = bm25_search(sub_query, top_k=BM25_TOP_K)

        fused = reciprocal_rank_fusion(dense_results, sparse_results)

        for result in fused:
            result["sub_query"] = sub_query

        all_fused.extend(fused)

    deduped = deduplicate_candidates(all_fused)

    reranked = rerank_candidates(query, deduped[:PRE_RERANK_CUTOFF], top_n=top_n)

    return reranked


CONDENSE_SYSTEM_PROMPT = """
You turn a multi-turn conversation into ONE standalone legal/eligibility
question suitable for searching a law database, representing what still
needs a legal answer right now.

First, check the conversation for a legal/eligibility question that was
raised in an EARLIER turn and has not yet been resolved with a specific
law citation (e.g. the earlier answer said something like "the laws
provided don't address this" or "I can't verify this specifically", or no
law-grounded answer was given for that part of the question at all). If
one exists, that question is still open - reconstruct it as the
standalone question, filling in any concrete details (loan amount, age,
country, etc.) mentioned anywhere in the conversation, even if the LATEST
message doesn't restate them.

Only if there is no such open legal question should you judge the latest
message on its own.

Rules:
1. Focus only on the legal/eligibility angle (age, jurisdiction, entity
   status, disclosure obligations, consumer protection, licensing, etc.) -
   leave out financial figures that only matter for a calculation, unless
   the figure itself is legally relevant (e.g. a loan amount that changes
   which regulation applies).
2. Resolve references to earlier turns (e.g. "that loan", "the amount I
   mentioned") into their actual specifics, using the conversation history.
3. If the latest message is already a complete, standalone legal question
   with no dependency on earlier turns, return it unchanged.
4. If nothing in the conversation - past or present - raises a legal or
   eligibility question, return the latest user message as-is; the gate
   downstream will decide whether retrieval is even needed.
5. Return ONLY the standalone question text - no preamble, no explanation.
"""


def condense_query(messages: List[Any]) -> str:
    """Collapse a multi-turn conversation into one standalone legal question
    for retrieve_laws(). Every downstream stage (decomposition, HyDE, BM25,
    rerank) is tuned for a single focused question, not a raw transcript -
    this must run BEFORE retrieval, not replace passing a query to it."""

    if not messages:
        return ""

    if len(messages) == 1:
        # Nothing to resolve against yet - skip the extra LLM call.
        return getattr(messages[-1], "content", "") or ""

    try:
        response = _llm.invoke([
            ("system", CONDENSE_SYSTEM_PROMPT),
            *messages,
        ])
        condensed = (response.content or "").strip()
        return condensed or (getattr(messages[-1], "content", "") or "")
    except Exception as e:
        print(f"condense_query: failed ({e}), falling back to latest message only")
        return getattr(messages[-1], "content", "") or ""