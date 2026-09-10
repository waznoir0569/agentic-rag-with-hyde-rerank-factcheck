# 🤖 Agentic Retrieval-Augmented Pipeline for Personalized Financial Advisory with Behavioral Personalization and Grounded Legal Retrieval

> A stateful, persona-adaptive financial advisory system featuring **HyDE query translation**, **GMM psychometric clustering**, **hybrid legal retrieval (Dense + BM25)**, **Cohere reranking**, and automated **fact-checking loops** built with FastAPI, LangGraph, and Streamlit.

---

## 💡 Key Architectural Highlights

- **🧠 GMM Behavioral Profiling**: Classifies users into 4 latent financial personas (e.g., *Anxious Saver*, *Risk-Tolerant*) via psychometric intake to dynamically adapt response tone and risk framing.
- **🔄 HyDE Query Translation**: Decomposes legal queries into sub-queries and generates synthetic legal passages (Hypothetical Document Embeddings) to improve semantic matching.
- **🔀 Hybrid Retrieval & RRF**: Parallel dense vector search (`text-embedding-3-small`) and sparse keyword search (`BM25`), fused using **Reciprocal Rank Fusion (RRF)** and rescored via **Cohere Reranker (`rerank-v3.5`)**.
- **⚡ Financial Tool Layer**: Deterministic execution nodes for dynamic parameters (Amortization Calculator, Debt-to-Income / Credit Scoring, and Macroeconomic APIs).
- **🛡️ Fact-Check Loop**: Iterative LangGraph verification node that checks claim grounding against statutory chunks, automatically retrying generation or attaching disclaimers.
- **🔐 Stateful & Secure**: Multi-tenant thread isolation with async PostgreSQL + `pgvector` storage, JWT auth, and LangGraph Postgres checkpointers.

---

## 💻 Tech Stack

- **Orchestration**: LangGraph, LangChain, Pydantic v2
- **Models**: `qwen/qwen3.7-flash` (Planner/HyDE/Fact-Check), `qwen/qwen3-max` (Answer Generation)
- **Retrieval Engine**: Pinecone (Dense), BM25 (Sparse), Cohere Rerank v3.5
- **Backend & Database**: FastAPI, Async SQLAlchemy 2.0, PostgreSQL + `pgvector`
- **Frontend**: Streamlit (Streaming UI)

--
## 📦 Quick Start (Docker Compose)

1. Copy environment template and edit values:
   ```bash
   cp env.example .env
   ```
2. Start the full stack:
   ```bash
   docker compose up --build
   ```

Services:
- Backend API: `http://localhost:8000/api/v1`
- API Docs: `http://localhost:8000/api/v1/docs`
- Frontend UI: `http://localhost:8501`

🛠️ Local Development
1. Backend (FastAPI)
Bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

2. Frontend (Streamlit)
Bash
cd frontend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run gui/main.py

### 🧩 Pipeline Architecture
### 1. Abstract System Lifecycle

```mermaid
graph TD
    UserQuery["User Query & Persona Context"] --> LangGraph["LangGraph Orchestration State Machine"]
    LangGraph <--> Memory[("Checkpointer / Session Memory")]
    
    subgraph PipelineServices ["Pipeline Services"]
        LangGraph --> Tools["Financial Tool Execution Layer"]
        LangGraph --> LegalRAG["Legal Law Retrieval Engine"]
    end

    Tools --> Assembly["Context Assembly Node"]
    LegalRAG --> Assembly
    Assembly --> LLMGen["LLM Answer Generation"]
    LLMGen --> FactCheck{"Fact-Check & Grounding Verification"}
    FactCheck -- Approved / Handled --> FinalResp["Final User Response"]
```

### 2. LangGraph Execution Workflow
```mermaid
graph TD
    START([START]) --> Planner["Planner Node: qwen/qwen3.7-flash"]
    
    Planner --> NeedTool{"Needs Any Financial Tool?"}
    
    NeedTool -- Yes --> Extract["Feature Extraction Node<br/>Extracts tool args and checks missing fields"]
    Extract --> ValidCalls{"Has Valid Tool Calls?"}
    
    ValidCalls -- Yes --> Executor["Tool Executor Node: FINANCIAL_TOOLS"]
    Executor --> LawRet["Law Retrieval Node<br/>Invokes legal RAG pipeline"]
    
    NeedTool -- No --> LawRet
    ValidCalls -- No --> LawRet
    
    LawRet --> ContextNode["Assemble Context Node<br/>Builds System Prompt with Tool Data & Laws"]
    ContextNode --> GenNode["Generate Answer Node: qwen/qwen3-max"]
    
    GenNode --> LawRetrieved{"Were Law Chunks Retrieved?"}
    
    LawRetrieved -- Yes --> FactCheck["Fact Check Node<br/>Strict Grounding Inspector"]
    FactCheck --> FCResult{"Fact Check Result"}
    
    FCResult -- Grounded --> END([END])
    FCResult -- "Attempts < 2" --> GenNode
    FCResult -- "Attempts >= 2" --> Disclaimer["Attach Disclaimer Node<br/>Appends legal fallback message"]
    
    Disclaimer --> END
    LawRetrieved -- No --> END
```

### 3. Query Translation & Retrieval Pipeline
```mermaid
graph TD
    MultiTurn["Multi-turn Conversation Messages"] --> Condense["Condense Query Node<br/>Collapses transcript to single legal query"]
    Condense --> Gate{"Gate Check: should_run_law_retrieval"}
    
    Gate -- No Legal Relevance --> Empty["Return Empty Chunk List"]
    Gate -- Requires Law Verification --> SubQueries
    
    subgraph ParallelQuery ["Parallel Sub-Query Execution"]
        SubQueries["Query Decomposition<br/>Splits into max 3 independent sub-queries"]
        
        SubQueries --> SQ1["Sub-Query 1"]
        SubQueries --> SQ2["Sub-Query 2"]
        
        SQ1 --> HyDE1["HyDE Generation: Statute Passage"]
        HyDE1 --> Dense1["Dense Search<br/>OpenAI text-embedding-3-small + Pinecone"]
        SQ1 --> Sparse1["Sparse Search<br/>BM25 Index"]
        
        Dense1 --> RRF1["Reciprocal Rank Fusion<br/>Combines Dense & BM25 Scores"]
        Sparse1 --> RRF1
        
        SQ2 --> HyDE2["HyDE Generation: Statute Passage"]
        HyDE2 --> Dense2["Dense Search<br/>OpenAI text-embedding-3-small + Pinecone"]
        SQ2 --> Sparse2["Sparse Search<br/>BM25 Index"]
        
        Dense2 --> RRF2["Reciprocal Rank Fusion<br/>Combines Dense & BM25 Scores"]
        Sparse2 --> RRF2
    end
    
    RRF1 --> CrossQuery["Cross-Subquery Deduplication<br/>Pools candidates & ranks by match frequency + RRF"]
    RRF2 --> CrossQuery
    
    CrossQuery --> Cutoff["Pre-Rerank Cutoff: Top 20 candidates"]
    Cutoff --> Cohere["Cohere Reranker: cohere/rerank-v3.5 via OpenRouter"]
    Cohere --> FinalChunks["Top-N Relevant Law Chunks (Default: 4 chunks)"]
```

📝 License
Licensed under the MIT License.
