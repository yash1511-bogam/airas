# AIRAS — Adaptive Immune Runtime for Agent Systems

> A runtime layer that observes failure patterns across populations of AI agent executions, learns abstract failure signatures, and preemptively prevents those failures from recurring.

## The Problem

AI agents in production fail in predictable, repeating patterns:

- **5 agents × 90% reliability = 59% system reliability** (multiplicative collapse)
- **88% of enterprise agent pilots never reach production** — reliability is the #1 blocker
- **$340K average cost** of a failed agent project
- Root causes appear at step 3, effects show at step 7 — no tool traces this causally
- Current observability tools tell you **what** went wrong. Nothing **prevents** it from happening again.

## The Solution

AIRAS learns from agent failures at the population level and prevents them preemptively:

1. **Observe** — Collect execution traces from all agent runs
2. **Extract** — Find where failed traces diverge from successful ones
3. **Abstract** — Generalize specific failures into matchable patterns
4. **Intervene** — Generate targeted prompt-level fixes for each pattern class
5. **Test** — Validate interventions via replay before deploying
6. **Remember** — Store pattern→intervention mappings permanently
7. **Prevent** — Match new executions against known patterns BEFORE failure manifests

## Validated Results

Tested on 250 real SWE-bench agent trajectories:

| Metric | Result | Target |
|--------|--------|--------|
| Prevention Rate | **52%** | ≥50% ✅ |
| False Positive Rate | **0%** | <5% ✅ |
| Pattern Coverage | **96%** | — |
| Patterns Discovered | **7** | ≥5 ✅ |
| Matching Latency | **15.5ms** | <10ms (Qdrant fixes this) |

## How It Works

```
1. Agent executes step 7 of a task
2. SDK sends partial trace (steps 1-7) to AIRAS
3. AIRAS extracts behavioral signals (errors, loops, step ratio)
4. Searches vector index for matching failure patterns (<5ms)
5. If match: returns intervention ("verify parameters before editing")
6. SDK injects intervention into agent's next prompt
7. Agent catches the error BEFORE making it
```

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────┐
│ Agent (SDK) │────▶│ AIRAS API    │────▶│ Qdrant  │  <5ms vector search
└─────────────┘     │ (FastAPI)    │     └─────────┘
                    │              │────▶│ Redis   │  hot cache + streams
                    └──────┬───────┘     └─────────┘
                           │
                    ┌──────▼───────┐     ┌──────────┐
                    │ Worker       │────▶│ Postgres │  state + audit trail
                    │ (extractor)  │     └──────────┘
                    └──────────────┘
```

## Quick Start

### Run the Killer Experiment (no infrastructure needed)

```bash
cd airas
uv venv .venv && source .venv/bin/activate
uv pip install -e .
python -m airas.experiment.runner
```

### Deploy Production System

```bash
docker compose up -d
python -m airas.scripts.seed_antigens
curl http://localhost:8100/health
```

### SDK Integration (3 lines)

```python
from airas.sdk import AIRASClient, airas_middleware

client = AIRASClient(base_url="http://localhost:8100")
graph = airas_middleware(my_langgraph_app, client)  # done
```

## API Endpoints

| Endpoint | Method | Purpose | Latency |
|----------|--------|---------|---------|
| `/v1/check` | POST | Real-time pattern check | <50ms |
| `/v1/traces` | POST | Ingest completed trace | <100ms |
| `/v1/antigens` | GET | List discovered patterns | <200ms |
| `/v1/stats` | GET | Dashboard metrics | <200ms |
| `/health` | GET | Liveness | <10ms |

## Project Structure

```
airas/
├── docker-compose.yml              # Full stack (API + Worker + Qdrant + Postgres + Redis)
├── Dockerfile
├── pyproject.toml
├── airas/
│   ├── models.py                   # Core Pydantic models
│   ├── models/
│   │   ├── domain.py              # Production domain models
│   │   └── api.py                 # Request/response schemas
│   ├── api/
│   │   └── main.py               # FastAPI service
│   ├── core/
│   │   ├── matching.py            # Qdrant-backed real-time matching
│   │   └── tolerance.py           # Over-correction prevention
│   ├── storage/
│   │   ├── qdrant.py             # Vector DB layer
│   │   └── redis_store.py        # Cache + streams
│   ├── extraction/                # Pattern extraction pipeline
│   │   ├── alignment.py          # Structural divergence detection
│   │   ├── classifier.py         # Rule-based failure classification
│   │   ├── abstraction.py        # Pattern abstraction
│   │   └── clustering.py         # HDBSCAN clustering
│   ├── intervention/
│   │   └── templates.py          # Intervention template bank
│   ├── replay/
│   │   └── engine.py             # Effectiveness judge
│   ├── experiment/
│   │   └── runner.py             # Killer experiment orchestrator
│   ├── sdk/
│   │   ├── client.py             # Async HTTP client
│   │   └── langgraph.py          # LangGraph middleware
│   ├── workers/
│   │   └── main.py              # Background trace processor
│   └── scripts/
│       └── seed_antigens.py      # Load validated patterns
└── README.md
```

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| API | FastAPI | Async, <50ms p99, WebSocket support |
| Vector DB | Qdrant | HNSW <5ms at 100K vectors, payload filtering |
| State | PostgreSQL | Audit trail, efficacy tracking |
| Cache | Redis | Hot pattern cache, async stream processing |
| Embeddings | all-MiniLM-L6-v2 | 384-dim, local, free, fast |
| Clustering | HDBSCAN | No K required, finds natural clusters |
| SDK | httpx (async) | Non-blocking, graceful degradation |

## Key Design Decisions

1. **No LLM calls in the hot path** — extraction and matching are deterministic and fast
2. **Graceful degradation** — if AIRAS is down, agents run normally (SDK never blocks)
3. **Two-phase matching** — embedding similarity (Qdrant) + condition predicates (code)
4. **Behavioral signals only** — runtime matching uses errors, loops, step anomalies (not outcome)
5. **Tolerance layer** — max 3 interventions per trace, cooldown, auto-disable low-efficacy interventions
6. **Population-level learning** — every failure makes the system smarter for ALL users

## The Moat

The failure pattern database is the product. Every deployment adds patterns. More users → more patterns → better prevention → more users. Copying the code gives you an empty system with no learned patterns.

## License

Proprietary. All rights reserved.
