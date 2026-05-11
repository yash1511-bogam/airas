# AIRAS — Adaptive Immune Runtime for Agent Systems

> An immune system for AI agent swarms. Observes failure patterns across populations of agent executions, learns abstract failure signatures, and preemptively prevents those failures from recurring.

## The Problem

AI agents in production fail in predictable, repeating patterns:

- **5 agents × 90% reliability = 59% system reliability** (multiplicative collapse)
- **88% of enterprise agent pilots never reach production** — reliability is the #1 blocker
- **$340K average cost** of a failed agent project
- Root causes appear at step 3, effects show at step 7 — no tool traces this causally
- Current observability tools tell you **what** went wrong. Nothing **prevents** it from happening again.

## The Solution

AIRAS mirrors the human adaptive immune system:

| Immune System | AIRAS |
|---------------|-------|
| Pathogen enters body | Agent starts failing |
| Antigen presentation | Extract abstract failure pattern from trace |
| T-cell classification | Categorize into failure archetype |
| B-cell response | Generate intervention (prompt gate/check) |
| Clonal selection | A/B test interventions via replay |
| Immunological memory | Store pattern→intervention permanently |
| Preemptive immunity | Match new executions BEFORE failure manifests |

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

### The 7 Layers

1. **Innate** — Circuit breakers, loop detection, schema validation (immediate, no learning)
2. **Trace Ingestion** — Normalize execution steps into standard format
3. **Antigen Extraction** — Find divergence point between failed and successful traces
4. **Classification** — Categorize: wrong_tool, wrong_params, infinite_loop, premature_termination, retrieval_failure, planning_failure
5. **Intervention Generation** — Select and parameterize template for the failure class
6. **Clonal Selection** — Test interventions via replay, track efficacy with Thompson Sampling
7. **Memory + Tolerance** — Permanent pattern store + over-correction prevention

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
| `/v1/check` | POST | Real-time immunity check | <50ms |
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
│   ├── models.py                   # Core Pydantic models (experiment)
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
│   ├── extraction/                # Antigen extraction pipeline
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
| Cache | Redis | Hot antigen cache, async stream processing |
| Embeddings | all-MiniLM-L6-v2 | 384-dim, local, free, fast |
| Clustering | HDBSCAN | No K required, finds natural clusters |
| SDK | httpx (async) | Non-blocking, graceful degradation |

## Key Design Decisions

1. **No LLM calls in the hot path** — extraction and matching are deterministic and fast
2. **Graceful degradation** — if AIRAS is down, agents run normally (SDK never blocks)
3. **Two-phase matching** — embedding similarity (Qdrant) + condition predicates (code)
4. **Behavioral signals only** — runtime matching doesn't know if trace will succeed; uses errors, loops, step anomalies
5. **Tolerance layer** — max 3 interventions per trace, cooldown between them, auto-disable low-efficacy interventions
6. **Population-level learning** — every failure makes the system smarter for ALL users

## The Moat

The failure pattern database is the product. Every deployment adds patterns. More users → more patterns → better immunity → more users. Copying the code gives you an empty immune system with no memory.

## Research Foundation

- Validated on SWE-bench (250 real agent trajectories)
- Prior art: VIGIL (Dec 2025), ReCiSt (Jan 2026), AAAI 2026 immune-inspired detection
- Differentiator: full adaptive loop (detect → abstract → generate fix → test → deploy → remember → preempt)
- No existing product implements population-level preemptive immunity

## License

Proprietary. All rights reserved.
