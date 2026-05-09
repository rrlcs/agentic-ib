# Agentic IB

Agentic IB is a multi-agent investment assistant.  
It accepts a chat question, routes work across specialized agents, streams live trace events, and returns a final answer (plus optional paper-trading action).

## System Overview

### Core Components

- `ui` (React + Vite): Chat UI, pipeline view, metrics panel.
- `api` (FastAPI): Accepts requests, enqueues tasks, exposes results + SSE stream.
- `worker` (Python): Consumes queued tasks and executes agent pipeline.
- `redpanda` (Kafka-compatible broker): Message bus between API and worker.
- `redis`: Trace/event storage used for streaming and metrics.
- `pinecone`: Vector database for long-term memo memory (RAG retrieval + persistence).

### High-Level Flow

```mermaid
flowchart LR
    U[User in Browser] --> UI[React UI :5173]
    UI -->|POST /chat| API[FastAPI API :8000]
    API -->|Publish task| K[(Redpanda API)]
    K -->|Consume task| W[Worker]
    W --> R[Router Agent]
    R --> P[Planned Pipeline]
    P --> A1[Research Agent]
    P --> A2[Financial Agent]
    P --> A3[Risk Agent]
    A1 --> S[Synthesis Agent]
    A2 --> S
    A3 --> S
    A1 --> VC[(Pinecone)]
    S --> VC
    VC --> ACT[Action Agent]
    S --> V[Validator Agent]
    V -->|accept| AN[Answer Agent]
    V -->|re_synthesis / re_research| P
    AN --> T[Write traces/metrics]
    T --> X[(Redis)]
    X -->|SSE /stream/{task_id}| UI
    W -->|Publish final result| K
    API -->|Read result| K
    UI -->|GET /job/{task_id}| API
```

## Run Locally

### 1) Prerequisites

- Docker + Docker Compose
- (Optional for manual UI run) Node.js 20+
- API keys for any external tools you want to use

### 2) Create `.env` in repo root

Create `/Users/raviraja/self-learning/agentic-ib/.env`:

```env
OPENAI_API_KEY=your_openai_key

# Optional but useful
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
LLM_DEFAULT_MODEL=gpt-4o-mini
```

Notes:
- `OPENAI_API_KEY` is required for normal chat responses.
- Alpaca and Alpha Vantage are optional; related features gracefully degrade if not set.

### 3) Start everything with Docker Compose

From repo root:

```bash
docker compose up --build
```

This brings up:
- UI: `http://localhost:5173`
- API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### 4) Stop services

```bash
docker compose down
```

## Optional: Run UI manually (without `ui` container)

If you want frontend hot reload directly on host:

```bash
cd ui
npm install
npm run dev
```

Then open `http://localhost:5173`.  
By default, the UI calls API at `http://<current-host>:8000`.

## Access Existing AWS Deployment

Your app is already live at:

- UI: `http://16.176.12.76:5173/`
- API docs: `http://16.176.12.76:8000/docs`
- API health: `http://16.176.12.76:8000/health`

### Access Steps

1. Open the UI URL in your browser: `http://16.176.12.76:5173/`
2. Ask a question in chat (example: "Should I invest in NVIDIA?")
3. Use the side panel to view pipeline/activity/metrics updates in real time.

### If UI opens but responses do not return

- Check API health: `http://16.176.12.76:8000/health`
- Open API docs: `http://16.176.12.76:8000/docs`
- If API is healthy but no completion appears, the worker or broker may be down.

## Quick Troubleshooting

- UI loads but chat fails: check API reachability at `/health`.
- API works but no result: verify worker container is running.
- No streaming updates: confirm Redis is healthy and `/stream/{task_id}` stays open.
- EC2 not reachable: verify security group rules, NACLs, and host firewall (`ufw`/iptables).