# GhostPilot — engineering notes

Six posts drawn from building [GhostPilot](https://github.com/faketut/GhostPilot), a personal interview-assistant desktop app. Each one targets a single sharp idea with code from the repo.

| # | Post | One-line take |
|---|------|---------------|
| 1 | [Prompt engineering needs a debugger, not a playground](01-prompt-replay.md) | Record real sessions, swap prompts, replay against the same questions. |
| 2 | [Your LLM retry loop is probably wrong](02-failover-classify.md) | 401 should never retry. 429 should. Most code conflates them. |
| 3 | [One event loop to rule them all: PyQt6 + asyncio in production](03-qasync-prod.md) | qasync, task ownership, cross-boundary cancellation. |
| 4 | [Hybrid RAG when your corpus has 50 chunks, not 5 million](04-tiny-rag.md) | BM25 still wins on small, sparse, named-entity-heavy corpora. |
| 5 | [I made ruff CI-blocking. The whole repo changed 5 lines.](05-ruff-blocking.md) | Gradual strictness: collect → fix → enforce. |
| 6 | [Windows-only desktop app, macOS-friendly contributors](06-cross-os-dev.md) | One audio backend abstraction, one path discipline, one CI matrix. |
