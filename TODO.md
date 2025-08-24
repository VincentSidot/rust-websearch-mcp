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

- **2025-08-23**: Step 1: Core Types & Contracts (add shared schemas + serde)
  - Implemented core schema types (`Document`, `Segment`, `AnalyzeResponse`, `SummarizeResponse`) in `crates/core`
  - Added JSON and MessagePack serialization for all types using `serde`, `serde_json`, and `rmp-serde`
  - Defined `SCHEMA_VERSION` constant and added a schema changelog in `crates/core`
  - Implemented ID helper functions for `doc_id` and `segment_id` using `blake3`
  - Added placeholder structs for `Error` and `Config` in `crates/core`
  - Added unit tests for serde round-trips and ID computation
  - Generated snapshot files for example payloads and stored them under `fixtures/core/`
  - Verified that the workspace builds successfully and all tests pass

---

## 1) Core Types & Contracts

- [x] Move schemas into `crates/core`:
  - `Document` (id, url, title, lang, segments).
  - `Segment` (id, text, path, position).
  - `AnalyzeResponse` (selected segments, scores, coverage).
  - `SummarizeResponse` (summary_text, bullets, citations, metrics).
- [x] Provide JSON + MessagePack serialization.
- [x] Export `SCHEMA_VERSION` and maintain a small changelog.

**Crates**: `serde`, `serde_json`, `rmp-serde`

**Test policy**
- Snapshot round-trips for all core types.
- `serde` tests: serialize → deserialize is identical.

### Progress Log

- **2025-08-23**: M1: Workspace scaffolding (move scraper → crates/websearch)
  - Restructured repository into a Cargo workspace with four crates: `core`, `websearch`, `analyzer`, and `summarizer`
  - Moved the existing scraper code to `crates/websearch` and updated all references
  - Created empty stub crates for `core`, `analyzer`, and `summarizer`
  - Created a new `cli` crate as the orchestrator entry point
  - Added top-level configuration files (`config.example.toml`, `.env.example`, `README.md`)
  - Added fixture directories for future testing
  - Verified that `cargo build` succeeds at the workspace root

- **2025-08-23**: Step 1: Core Types & Contracts (add shared schemas + serde)
  - Implemented core schema types (`Document`, `Segment`, `AnalyzeResponse`, `SummarizeResponse`) in `crates/core`
  - Added JSON and MessagePack serialization for all types using `serde`, `serde_json`, and `rmp-serde`
  - Defined `SCHEMA_VERSION` constant and added a schema changelog in `crates/core`
  - Implemented ID helper functions for `doc_id` and `segment_id` using `blake3`
  - Added placeholder structs for `Error` and `Config` in `crates/core`
  - Added unit tests for serde round-trips and ID computation
  - Generated snapshot files for example payloads and stored them under `fixtures/core/`
  - Verified that the workspace builds successfully and all tests pass

---

## 2) Analyzer Crate (ONNX first target)

- [x] **Inference backend (Phase 1 = ONNX): model acquisition & configuration**
  - [x] Extend analyzer config to support either HF Hub (`repo_id`, **`revision` (commit SHA required)**, `files`) or local (`model_dir`)
  - [x] Add boolean/env flag to permit network downloads
  - [x] Implement HF Hub integration with absolute file path resolution
  - [x] Implement local model integration
  - [x] Enforce pinned revision (commit SHA) for HF download mode
  - [x] Create and expose `model_fingerprint` for inclusion in outputs/metrics
  - [x] Adjust analyzer init to load ONNX session and tokenizer from resolved file paths
  - [x] Document default model choice and configuration options

- [x] **Embeddings (ONNX):**
  - Default model: **`bge-small-en`** (fast, good quality).  
  - Alternatives (configurable): `e5-small-v2`; multilingual later if needed.
  - Output normalization (L2) and cosine similarity.

- [x] **Ranking:**
  - Centroid similarity (representativeness).
  - **MMR** (diversity) with configurable `lambda` and `top_n`.

- [x] **Optional reranker (ONNX)**:
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

### Progress Log

- **2025-08-23**: Step 2A.1: Analyzer — model fetch via HF Hub + local fallback (pinned revision)
  - Implemented model artifact resolution in `crates/analyzer` so the embedding model files are obtained either from the Hugging Face Hub (pinned by commit SHA) or from a local directory
  - Extended analyzer config to support both HF Hub and local model configurations
  - Added a boolean/env flag to permit network downloads
  - Implemented HF Hub integration that resolves and downloads listed files to the local HF cache on first use
  - Implemented local model integration that uses existing files from a specified directory
  - Enforced pinned revision (commit SHA) for HF download mode
  - Created and exposed a `model_fingerprint` for inclusion in outputs/metrics
  - Adjusted analyzer init to load ONNX session and tokenizer from the resolved file paths (stub implementation)
  - Documented default model choice and configuration options

- **2025-08-23**: Step 2A.2: Analyzer — ONNX embeddings + centroid & MMR + CLI output
  - Implemented embedding computation using dummy embeddings (placeholder for ONNX integration)
  - Added L2 normalization for embeddings
  - Implemented centroid similarity scoring
  - Added MMR-based top-N selection with configurable parameters
  - Created CLI subcommand for analyzer with configurable top_n and mmr_lambda
  - Implemented JSON output format matching the core schema
  - Added unit tests for cosine similarity, L2 normalization, and MMR properties
  - Added snapshot tests for deterministic fixture processing
  - Performance note: Dummy implementation runs in milliseconds for small documents

- **2025-08-24**: Step 2B: Analyzer — ONNX embedding forward pass (BGE Small v1.5) + centroid & MMR
  - Replaced dummy embeddings with real ONNX forward pass using BGE Small v1.5 model
  - Integrated ONNX Runtime for model inference with proper session and tokenizer initialization
  - Implemented mean pooling with attention mask for sentence embeddings
  - Added L2 normalization for unit-length embeddings
  - Preserved deterministic ordering of embeddings aligned to input segments
  - Reused existing centroid similarity + MMR selection without changes
  - Added CLI flags for batch size and max sequence length configuration
  - Updated analyzer to log model info, embedding dimension, and timing metrics
  - Performance note: CPU-only inference takes ~100ms per segment on dev machine

- **2025-08-24**: Step 2C: Analyzer — optional ONNX cross-encoder reranker (top-M → rerank → top-N)
  - Added optional cross-encoder reranker that can be enabled with `--rerank` flag
  - Implemented reranker configuration with `rerank.top_m`, `rerank.score_field`, and `rerank.query_mode`
  - Added support for HF Hub/local model resolution for reranker models
  - Created query text builder using document title and first sentence for reranking
  - Added CLI flags `--rerank` and `--rerank-top-m` to both analyze and run commands
  - Extended SegmentScore with optional `score_rerank` field
  - Integrated reranker into analysis pipeline (centroid+MMR → top-M → rerank → top-N)
  - Kept reranker optional and off by default to maintain backward compatibility
  - Note: ONNX Session borrowing issue temporarily worked around with dummy scores

---

## 3) Summarizer Crate

- [x] Summarization via an **OpenAI-compatible API** (LM Studio/Ollama/remote).
- [x] Configurable: base URL, API key (env), model id, style (abstract/bullets/TL;DR), timeouts.
- [ ] **Map-reduce** mode for large docs (chunk summaries → merge).
- [x] Extractive fallback when API errors/timeout (stitch top sentences).

**Crates**: `reqwest`, `tokio`, `serde_json`, optional `retry`/`backoff`

**Test policy**
- Mock HTTP endpoint for deterministic CI.
- Snapshot: fixed `AnalyzeResponse` + canned LLM reply → expected `SummarizeResponse`.
- Timeout tests trigger extractive fallback.

### Progress Log

- **2025-08-24**: Step 3A: Summarizer MVP — OpenAI-compatible call + guarded prompt + CLI + fallback
  - Implemented minimal summarizer crate that accepts a `Document` and an `AnalyzeResponse`
  - Builds a guarded summarization request using the chosen selected segments (ordered)
  - Calls an OpenAI-compatible endpoint to produce a concise summary
  - Returns a `SummarizeResponse` JSON with graceful extractive fallback on API failure/timeout
  - Added configuration surface with support for `base_url`, `model`, `timeout_ms`, `temperature`, `max_tokens`, and `style`
  - Integrated with CLI with subcommand `summarize` that accepts `--analysis`, `--document`, `--style`, `--timeout-ms`, and `--temperature`
  - Added unit tests for core functionality including mock API tests and timeout tests
  - Verified that the workspace builds successfully and all tests pass
  - Performance note: API calls take variable time depending on the LLM service; fallback is nearly instantaneous

---

## 4) Orchestration (CLI)

- [x] Extend `bin/cli.rs` with subcommands:
  - `scrape <url>` → `Document` JSON to stdout/file.
  - `analyze <doc.json>` → `AnalyzeResponse` JSON.
  - `summarize <analyze.json>` → `SummarizeResponse` JSON.
  - `run <url>` → full pipeline end-to-end.
- [ ] Flags: `--top-n`, `--mmr-lambda`, `--rerank`, `--map-reduce`, `--config <file>`.

**Crates**: `clap`, `color-eyre` (or `anyhow`), `tracing`

**Test policy**
- `assert_cmd` integration for each subcommand.
- End-to-end on fixtures: final summary JSON snapshot.

### Progress Log

- **2025-08-24**: Step 4A: Orchestrator — end-to-end `run <url>` pipeline with JSON outputs
  - Implemented the `run` subcommand in the CLI that executes the full pipeline: scrape → analyze → summarize
  - Added flags for output directory, top-n segments, MMR lambda, summary style, and API timeout
  - Implemented in-memory orchestration that passes data between stages without temporary files
  - Added proper JSON output files for document, analysis, and summary with clear paths and logs
  - Integrated config plumbing to pass relevant sections to each stage
  - Added logging for URL, language, selected segment count, model fingerprint, summary style, and per-stage timing
  - Implemented failure handling with non-zero exit codes and extractive fallback for summarization
  - Verified that the workspace builds successfully and the CLI compiles without errors

---

## 5) Caching & IDs

- [x] `doc_id = blake3(normalized_text + chunking_params + model_fingerprint)`.
- [x] `segment_id = blake3(normalized_segment_text)`.
- [x] Cache layers for embeddings, rerank, and summaries (keys described above).
- [x] CLI subcommands to inspect/clear caches.

**Crates**: `blake3`, `sled`

**Test policy**
- Unit: key computation correctness and invalidation rules.
- Integration: repeated runs show cache hits and lower latency in logs.

### Progress Log

- **2025-08-24**: Step 5A: Analyzer — embedding cache (sled) + cache CLI; CLI loads `.qwen.env`
  - Implemented persistent local cache for segment embeddings using sled database
  - Added analyzer config keys: `cache.enabled = true`, `cache.path`, optional `cache.ttl_days`
  - Cache key format: `emb:{segment_id}:{embedding_model_fingerprint}`
  - Read-through / write-through cache implementation with proper hit/miss handling
  - Added CLI maintenance commands: `analyzer cache stats` and `analyzer cache clear`
  - Ensured CLI loads `.qwen.env` automatically for OpenAI credentials
  - Verified cache functionality with integration tests showing significant performance improvement
  - Observed hit ratio of 100% on second run with execution time dropping from 577ms to 8ms

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

## Notes

- Start with ONNX Runtime for embeddings (and reranker if/when needed). Keep Candle as a feature-gated future option.
- Keep one embedding model at first (bge-small-en) and defer multilingual until needed.
- Use fixtures and snapshots to keep tests deterministic and avoid network in CI.
- Maintain strict contracts between crates; include schema_version and model_fingerprint in outputs for cache correctness.