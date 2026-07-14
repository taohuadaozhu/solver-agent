# Solver Agent

AI-powered optimization agent that converts natural language problem descriptions into structured optimization workflows — problem understanding → dataset selection → variable extraction → objective definition → constraint identification → classification → algorithm recommendation → solver execution → visualization.

## Architecture

```
User Query (NL)
    │
    ▼
┌──────────────────────────────────────────────┐
│  Frontend (React + Vite + Plotly)            │
│  ChatPage: Guided 7-Step Workflow + SSE      │
└──────────────────────────────────────────────┘
    │ POST /workflow/stream (SSE)
    ▼
┌──────────────────────────────────────────────┐
│  FastAPI Backend (:8000 / :8001)             │
│                                              │
│  ┌─────────────┐  ┌──────────────────┐       │
│  │ Agent Layer │  │ RAG Pipeline     │       │
│  │             │  │                  │       │
│  │ Workflow    │  │ Embedding (OAI)  │       │
│  │ State Mach. │──│ pgvector search  │       │
│  │ Smart Skip  │  │ Top-K retrieval  │       │
│  │ Tool Call   │  └──────────────────┘       │
│  │             │                             │
│  │ ┌─────────┐ │  ┌──────────────────┐       │
│  │ │Registry │ │  │ Memory Layer     │       │
│  │ │Dispatcher│ │  │ Conversation     │       │
│  │ │Validator│ │  │ Workflow persist  │       │
│  │ └─────────┘ │  └──────────────────┘       │
│  └─────────────┘                             │
└──────────────────────────────────────────────┘
    │                        │
    ▼                        ▼
┌──────────┐    ┌──────────────────────┐
│ OpenAI   │    │ PostgreSQL + pgvector │
│ GPT-4.1  │    │                      │
│ Embedding│    │ rag_documents (675+)  │
│          │    │ conversation_messages │
│          │    │ workflow_sessions     │
└──────────┘    └──────────────────────┘
```

## Project Structure

```
solver-agent/
├── backend/
│   ├── agent/               # Agent execution engine
│   │   ├── workflow.py      # 7-step state machine & SSE streaming
│   │   ├── solver.py        # Unified Solver base class (GA, NSGA-II, PSO, ...)
│   │   ├── registry.py      # Tool Registry: algorithm name → Solver class
│   │   ├── dispatcher.py    # Tool Dispatcher: function calling → solver execution
│   │   ├── validator.py     # Result Validator: convergence, feasibility, timeout
│   │   ├── visualizer.py    # Plotly chart generator
│   │   ├── tools.py         # OpenAI function calling schema
│   │   └── executor.py      # Dataset loader + execution entry point
│   ├── api/
│   │   ├── recommend.py     # Quick RAG recommendation API (:8000)
│   │   └── workflow.py      # Workflow + Memory + SSE API (:8001)
│   ├── database/
│   │   └── postgres.py      # PostgreSQL connection pool (psycopg2)
│   ├── llm/
│   │   ├── openai_client.py # OpenAI async client (retry, timeout, streaming)
│   │   └── prompt.py        # 7-step prompt templates + smart skip coverage
│   ├── memory/
│   │   ├── db.py             # PostgreSQL table initialization
│   │   └── store.py          # Conversation & workflow CRUD
│   └── rag/
│       ├── build_index.py    # Embedding + pgvector indexing
│       ├── retriever.py      # Semantic search via cosine similarity
│       └── chunker.py        # Document chunker
├── frontend/
│   └── src/
│       ├── ChatPage.jsx      # Guided workflow UI + SSE consumer
│       ├── App.jsx           # Algorithm/project browser + chat router
│       ├── main.jsx          # React entry
│       └── styles.css        # Full app styles
├── knowledge-base/
│   ├── algorithms/           # 330+ optimization algorithms (metadata + docs)
│   ├── projects/             # 8 optimization case studies
│   └── datasets/             # 6 benchmark datasets (TSP, VRP, JSSP, ...)
└── data/                     # SQLite memory DB (deprecated, now PG)
```

## Features

### Agent Workflow
- **7-Step State Machine**: Problem → Dataset → Variables → Objectives → Constraints → Classify → Algorithm
- **Smart Skip Detection**: LLM analyzes user input for pre-covered information, auto-skips steps
- **SSE Streaming**: Real-time token-by-token LLM output via Server-Sent Events
- **Tool Calling**: `load_dataset`, `execute_solver`, `generate_chart` via OpenAI function calling

### RAG (Retrieval-Augmented Generation)
- PostgreSQL + pgvector for vector storage (VECTOR 1536, text-embedding-3-small)
- Cosine similarity semantic search across 675+ algorithm & case documents
- Metadata-rich document indexing with hash-based change detection

### Solver Execution
- **3-Layer Architecture**: Registry → Dispatcher → Validator
- 15+ registered solver names with fuzzy matching + DummySolver fallback
- Result validation: convergence check, non-empty check, timeout check, feasibility check

### Memory & Persistence
- Conversation memory (20-round sliding window)
- Workflow session save/load/resume
- Shared PostgreSQL connection pool

### Frontend
- React 18 + Vite + Plotly charts
- Guided step-by-step interaction with progress bar
- Dataset radio picker with similarity scores
- Execution metrics cards + validation badges
- Dual mode: Guided Workflow / Quick RAG Recommend

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- PostgreSQL 17+ with pgvector extension
- OpenAI API key

### Setup

```bash
# 1. Clone
git clone https://github.com/taohuadaozhu/solver-agent
cd solver-agent

# 2. Backend dependencies
pip install fastapi uvicorn psycopg2-binary openai tenacity httpx python-dotenv

# 3. Set environment variable
export OPENAI_API_KEY="sk-..."

# 4. Install & start PostgreSQL + pgvector (macOS)
brew install postgresql@17 pgvector
# (symlink pgvector extension files if needed)
brew services start postgresql@17
createdb solver_agent

# 5. Build knowledge base index
python backend/rag/build_index.py --source knowledge-base --types algorithms,projects,datasets

# 6. Start backend APIs
python backend/api/recommend.py &   # :8000 - quick RAG
python backend/api/workflow.py &    # :8001 - workflow + SSE

# 7. Start frontend
cd frontend && npm install && npm run dev
```

Then open http://localhost:5173.

## API Endpoints

### Recommend API (:8000)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/recommend` | Quick RAG-based algorithm recommendation |
| GET | `/health` | Health check |

### Workflow API (:8001)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/workflow/start` | STEP_1 analysis + dataset options |
| POST | `/workflow/step` | Run a single workflow step |
| POST | `/workflow/stream` | SSE streaming: run remaining steps |
| POST | `/workflow/execute` | Execute solver + generate chart |
| POST | `/memory/conversations` | Create conversation |
| GET | `/memory/conversations/{id}/messages` | Load chat history |
| POST | `/memory/workflows` | Save workflow session |
| GET | `/workflow/health` | Health check |

## License

MIT
