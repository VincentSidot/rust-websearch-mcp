# TODO.md

## Project Summary

This project (`rust-websearch-mcp`) is a **self-hosted pipeline** for:
1. **Scraping** web pages (fetch → clean → segment into coherent chunks).
2. **Analyzing** content locally with embeddings and an optional cross-encoder:
   - Generate embeddings for each chunk (local inference).
   - Rank chunks by representativeness + diversity (centroid + MMR).
   - Optionally re-rank with a cross-encoder for precision.
3. **Summarizing** selected chunks with an LLM (via an OpenAI-compatible API such as LM Studio, Ollama, or remote).
4. **Orchestration** via a CLI: run each step (`scrape`, `analyze`, `summarize`) or the full pipeline end-to-end.

**Design goals**
- Lean and modular: three crates (scraper, analyzer, summarizer) with a shared `core` crate.
- No sprawling Docker setup: embeddings and rerank run **in-process**; only the LLM is an external API call.
- Clear contracts: JSON/MessagePack schemas shared across crates.
- Strong testing: unit, snapshot, integration, and performance checks.

> **Backend decision (Phase 1):** Use **ONNX Runtime** (`ort` crate) as the **first target** for local inference (embeddings + cross-encoder).  
> **Phase 2 (optional):** Add a **Candle** backend behind a feature flag to keep flexibility without code churn.

---

## 0) Workspace Restructure

- [x] Convert repo into a **Cargo workspace**:
  - `crates/websearch` → existing scraper (move current crate here).
  - `crates/analyzer` → embeddings + MMR + optional cross-encoder (local).
  - `crates/summarizer` → LLM summarization (OpenAI-compatible).
  - `crates/core` → shared schemas, types, errors.
  - `crates/cli` → orchestrator CLI (`run`, `scrape`, `analyze`, `summarize`).

**Acceptance**
- Workspace builds with one `cargo build`.
- `cargo run --bin cli run <url>` executes scraper → analyzer → summarizer.

### Progress Log

- **2025-08-23**: M1: Workspace scaffolding (move scraper → crates/websearch)
  - Restructured repository into a Cargo workspace with four crates: `core`, `websearch`, `analyzer`, and `summarizer`
  - Moved the existing scraper code to `crates/websearch` and updated all references
  - Created empty stub crates for `core`, `analyzer`, and `summarizer`
  - Created a new `cli` crate as the orchestrator entry point
  - Added top-level configuration files (`config.example.toml`, `.env.example`, `README.md`)
  - Added fixture directories for future testing
  - Verified that `cargo build` succeeds at the workspace root

---

## 1) Core Types & Contracts

- [ ] Move schemas into `crates/core`:
  - `Document` (id, url, title, lang, segments).
  - `Segment` (id, text, path, position).
  - `AnalyzeResponse` (selected segments, scores, coverage).
  - `SummarizeResponse` (summary_text, bullets, citations, metrics).
- [ ] Provide JSON + MessagePack serialization.
- [ ] Export `SCHEMA_VERSION` and maintain a small changelog.

**Crates**: `serde`, `serde_json`, `rmp-serde`

**Test policy**
- Snapshot round-trips for all core types.
- `serde` tests: serialize → deserialize is identical.

---

## 2) Analyzer Crate (ONNX first target)

- [ ] **Inference backend (Phase 1 = ONNX):**
  - Integrate **ONNX Runtime** via `ort` crate as the default backend.
  - Tokenization via `tokenizers` aligned with chosen models.
  - Backend abstraction (`EmbeddingBackend`, `RerankBackend`) to allow future Candle plug-in.

- [ ] **Embeddings (ONNX):**
  - Default model: **`bge-small-en`** (fast, good quality).  
  - Alternatives (configurable): `e5-small-v2`; multilingual later if needed.
  - Output normalization (L2) and cosine similarity.

- [ ] **Ranking:**
  - Centroid similarity (representativeness).
  - **MMR** (diversity) with configurable `lambda` and `top_n`.

- [ ] **Optional reranker (ONNX):**
  - Cross-encoder (MiniLM/BGE reranker) applied to a shortlist (e.g., top 30).
  - Config flag to enable/disable rerank; include `reranker_model_id` in fingerprints.

- [ ] **Caching:**
  - Cache embeddings per `{segment_id, embedding_model_id}`.
  - Cache rerank scores per `{segment_id, reranker_model_id, seed?}`.
  - Simple local KV (`sled`) or file-based cache.

- [ ] **Outputs:**
  - `AnalyzeResponse` with:
    - Selected segment ids (ordered), scores, and reasons (e.g., cluster center/intro/conclusion).
    - Redundancy/coverage metrics.
    - Model fingerprints (embedding + reranker).

**Crates**: `ort`, `tokenizers`, `ndarray` (or `nalgebra`), `blake3`, `sled`

**Test policy**
- Unit: cosine, L2 normalization, MMR properties (no duplicates; increasing `N` expands set; redundancy decreases).
- Snapshot: deterministic `Document` fixture → stable `AnalyzeResponse` (tolerances if models change).
- Perf: medium doc analysis ≤ ~2s CPU-only on dev machine (document this budget).

> **Phase 2 (optional):** Add a **Candle** backend behind a feature flag:
> - `features = ["backend-candle"]` vs default `["backend-onnx"]`.
> - Same `EmbeddingBackend`/`RerankBackend` trait to avoid touching callers.

---

## 3) Summarizer Crate

- [ ] Summarization via an **OpenAI-compatible API** (LM Studio/Ollama/remote).
- [ ] Configurable: base URL, API key (env), model id, style (abstract/bullets/TL;DR), timeouts.
- [ ] **Map-reduce** mode for large docs (chunk summaries → merge).
- [ ] Extractive fallback when API errors/timeout (stitch top sentences).

**Crates**: `reqwest`, `tokio`, `serde_json`, optional `retry`/`backoff`

**Test policy**
- Mock HTTP endpoint for deterministic CI.
- Snapshot: fixed `AnalyzeResponse` + canned LLM reply → expected `SummarizeResponse`.
- Timeout tests trigger extractive fallback.

---

## 4) Orchestration (CLI)

- [ ] Extend `bin/cli.rs` with subcommands:
  - `scrape <url>` → `Document` JSON to stdout/file.
  - `analyze <doc.json>` → `AnalyzeResponse` JSON.
  - `summarize <analyze.json>` → `SummarizeResponse` JSON.
  - `run <url>` → full pipeline end-to-end.
- [ ] Flags: `--top-n`, `--mmr-lambda`, `--rerank`, `--map-reduce`, `--config <file>`.

**Crates**: `clap`, `color-eyre` (or `anyhow`), `tracing`

**Test policy**
- `assert_cmd` integration for each subcommand.
- End-to-end on fixtures: final summary JSON snapshot.

---

## 5) Caching & IDs

- [ ] `doc_id = blake3(normalized_text + chunking_params + model_fingerprint)`.
- [ ] `segment_id = blake3(normalized_segment_text)`.
- [ ] Cache layers for embeddings, rerank, and summaries (keys described above).
- [ ] CLI subcommands to inspect/clear caches.

**Crates**: `blake3`, `sled`

**Test policy**
- Unit: key computation correctness and invalidation rules.
- Integration: repeated runs show cache hits and lower latency in logs.

---

## 6) Observability

- [ ] Structured logs via `tracing`/`tracing-subscriber` (doc_id, request_id).
- [ ] Metrics: per-stage latency, cache hit rate, token usage, redundancy.
- [ ] `--healthcheck` for analyzer (models load) and summarizer (API reachable).

**Crates**: `tracing`, `tracing-subscriber`, optional `metrics` + `metrics-exporter-prometheus`

**Test policy**
- Healthcheck exits non-zero when model/API unavailable.
- Logs include ids and timings; basic metrics counters validated in tests.

---

## 7) Milestones

- **M1**: Workspace + core types + CLI skeleton (migrated scraper).
- **M2**: Analyzer MVP (**ONNX embeddings** + centroid + MMR, cache, metrics).
- **M3**: Summarizer MVP (OpenAI-compatible call, extractive fallback).
- **M4**: End-to-end pipeline (`run`), golden snapshots for fixtures.
- **M5**: Optional **ONNX reranker** integration and coverage metrics.
- **M6**: Caching tools, healthchecks, and observability polish.
- **M7**: (Optional) Candle backend feature flag.

---

## 8) Test Strategy Recap

- **Unit**: math (cosine/MMR), serde round-trips, cache keys, config parsing.
- **Snapshot**:
  - Scraper outputs → `Document` JSON.
  - Analyzer outputs → `AnalyzeResponse` JSON.
  - End-to-end summaries → `SummarizeResponse` JSON.
- **Integration**: CLI subcommands with temp files and mock LLM server.
- **Performance**: analyzer p95 ≤ ~2s; summarizer p95 ≤ ~5s (medium pages).
- **CI**: `clippy -D warnings`, `rustfmt --check`, test matrix for features (`backend-onnx`, `rerank`).

---

## 9) Configuration & Secrets

- [ ] `config/config.example.toml` with sections:
  - `[scraper]` timeouts, user-agent, robots policy, max_size_kb
  - `[analyzer]` backend = `"onnx"`, embedding_model_id = `"bge-small-en"`, mmr_lambda, top_n, rerank = false, reranker_model_id = ""
  - `[summarizer]` base_url, model, timeout_ms, map_reduce = false
  - `[cache]` path, ttl_days
- [ ] `.env.example` for API keys; env overrides take precedence.

**Acceptance**
- `cli config print` shows effective config (secrets redacted).

---

## Reference: Workspace File Tree (example)

```text
.
├─ Cargo.toml
├─ README.md
├─ TODO.md
├─ config/
│  ├─ config.example.toml
│  └─ config.local.toml            # gitignored
├─ .env.example
├─ fixtures/
│  ├─ html/
│  ├─ documents/
│  ├─ analyze/
│  └─ summarize/
├─ bin/
│  └─ cli.rs                       # orchestrator CLI (scrape/analyze/summarize/run)
└─ crates/
   ├─ core/
   │  ├─ Cargo.toml
   │  └─ src/
   │     ├─ lib.rs
   │     ├─ schema.rs              # Document, Segment, Analyze/Summarize Requests & Responses
   │     ├─ ids.rs                 # blake3 ids
   │     ├─ config.rs
   │     └─ errors.rs
   │  └─ tests/
   │     └─ serde_roundtrip.rs
   │
   ├─ websearch/                   # existing scraper moved here
   │  ├─ Cargo.toml
   │  ├─ src/
   │  │  ├─ lib.rs
   │  │  ├─ fetch.rs
   │  │  ├─ parse.rs
   │  │  ├─ segment.rs
   │  │  └─ lang.rs
   │  ├─ bin/
   │  │  └─ websearch.rs
   │  └─ tests/
   │     ├─ scrape_golden.rs
   │     └─ segmentation_props.rs
   │
   ├─ analyzer/
   │  ├─ Cargo.toml
   │  ├─ src/
   │  │  ├─ lib.rs
   │  │  ├─ embed/
   │  │  │  ├─ mod.rs
   │  │  │  ├─ onnx_backend.rs     # Phase 1: ONNX Runtime via `ort` (default)
   │  │  │  └─ candle_backend.rs   # Phase 2: optional feature-gated backend
   │  │  ├─ rank/
   │  │  │  ├─ centroid.rs
   │  │  │  ├─ mmr.rs
   │  │  │  └─ metrics.rs          # redundancy/coverage metrics
   │  │  ├─ rerank/
   │  │  │  ├─ mod.rs
   │  │  │  └─ cross_encoder.rs    # ONNX reranker (optional)
   │  │  ├─ cache.rs               # sled/file cache
   │  │  └─ model_id.rs            # fingerprints (embedding + reranker)
   │  ├─ bin/
   │  │  └─ analyzer.rs            # reads Document JSON → writes AnalyzeResponse JSON
   │  ├─ benches/
   │  │  └─ analyzer_perf.rs
   │  └─ tests/
   │     ├─ math_unit.rs
   │     └─ analyze_snapshot.rs
   │
   └─ summarizer/
      ├─ Cargo.toml
      ├─ src/
      │  ├─ lib.rs
      │  ├─ client.rs              # OpenAI-compatible HTTP client
      │  ├─ prompt.rs              # abstract/bullets/TL;DR, guardrails
      │  ├─ reduce.rs              # map-reduce logic (optional)
      │  └─ fallback.rs            # extractive fallback on timeout
      ├─ bin/
      │  └─ summarizer.rs          # reads AnalyzeResponse JSON → writes SummarizeResponse JSON
      └─ tests/
         ├─ mock_api.rs
         └─ summarize_snapshot.rs
```

## Notes

- Start with ONNX Runtime for embeddings (and reranker if/when needed). Keep Candle as a feature-gated future option.
- Keep one embedding model at first (bge-small-en) and defer multilingual until needed.
- Use fixtures and snapshots to keep tests deterministic and avoid network in CI.
- Maintain strict contracts between crates; include schema_version and model_fingerprint in outputs for cache correctness.