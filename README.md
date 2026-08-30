# FinDocIQ

Open-weights financial-document intelligence for Indian annual reports and
credit-rating rationales. The project is deliberately built in measured
phases; retrieval and answer quality will be reported separately when the eval
harness lands.

## Current status

| Phase | Status | Gate |
|---|---|---|
| 1. Ingestion | Complete | 15 filings, 1,236 chunks, 5 visual bbox checks |
| 2. Retrieval | Implemented | Dense, hybrid RRF, and hybrid RRF + BGE rerank |
| 3. Evaluation | Implemented | Separate retrieval and answer score tables |
| 4. Two-pass reasoning | Live-smoke validated | Single-pass versus two-pass results |
| 5. Observability | Live-smoke validated | Content-safe latency, cache and failure spans |
| 6. Cross-lingual | Live-smoke validated | Paired English-Hindi results by language |
| 7. Expanded retrieval evaluation | GPU validated | 31-question, six-category corpus-backed sweep |

Every reported `1.000` below comes from an early `n=5` question/fact cohort;
the evaluation is expanding to `n=80`. These runs validate the pipeline and
measurement path, not production quality.

### Published validation environment

All Phase 4-6 live-smoke results in this README were produced on a Windows
workstation using CPU execution. They are correctness and instrumentation
checks, not GPU throughput or latency benchmarks.

| Component | Environment used for published results |
|---|---|
| Generation | Ollama `gemma3:4b` on CPU |
| Embeddings | BGE-M3 on CPU |
| Reranking | bge-reranker-v2-m3 on CPU, including cold model startup |
| Vector store | Qdrant in local Docker |
| GPU/vLLM | 4B AWQ serving smoke-tested separately; not used for the published runs |

### Separate GPU serving smoke test

On 2026-08-13, the pinned laptop tier was live-tested on an NVIDIA GeForce RTX
4080 Laptop GPU (12 GB) through WSL2 Ubuntu, vLLM 0.27.1, and PyTorch
2.13.0+cu130. The exact `gaunernst/gemma-3-4b-it-int4-awq` revision
`8f28faf05c382a2dd81a471090acdb23156eb354` loaded with the AWQ Marlin kernel:
4.39 GiB for the model and 4.64 GiB for the KV cache at an 8,192-token context.

The `/v1/models` route exposed `google/gemma-3-4b-it`, and two
temperature-0/seed-17 requests returned the same valid JSON response. The first
request took 6.5 seconds while Triton kernels were compiled; the warm request
took 1.8 seconds. The repository's concrete `VllmGemmaClient` was then exercised
against the same endpoint and its response parsed as JSON successfully.

These timings are a single-request serving smoke test, not a throughput
benchmark. The published Phase 4-6 evaluations remain CPU runs, and neither the
full evaluation cohort nor the default 12B tier has yet been run on this 12 GB
laptop GPU. WSL2 required `VLLM_USE_V2_MODEL_RUNNER=0`,
`VLLM_USE_FLASHINFER_SAMPLER=0`, and eager mode because the v2 runner expects
UVA and the FlashInfer sampler expects a local CUDA toolkit.

## Production stack

| Component | Default | Fallback / tiers |
|---|---|---|
| Generation | Gemma 3 12B AWQ int4 | Pinned 4B, 12B, and 27B YAML tiers |
| Serving | vLLM OpenAI-compatible endpoint | Ollama `gemma3:4b` |
| Vision | Gemma 3 multimodal through vLLM | Fail closed when unconfigured |
| Embeddings | BGE-M3 dense + sparse | One pinned model revision |
| Reranking | bge-reranker-v2-m3, top-50 to top-8 | None |
| Vector store | Local Qdrant | None required |
| Parsing | PyMuPDF digital fast path; Docling for complex pages | Gemma vision |
| Tracing | Self-hosted Langfuse OTLP + local JSONL | JSONL only |
| API | Thin FastAPI boundary | CLI entry points |
| Packages | uv + `pyproject.toml` + `uv.lock` | No `requirements.txt` |

## Phase 1 quick start

```powershell
uv sync --extra dev
uv run pytest
uv run findociq-ingest tests/fixtures/digital.pdf --output chunks.jsonl
uv run python scripts/spot_check_chunk.py tests/fixtures/digital.pdf chunks.jsonl `
  --chunk-id <id> --output spot-check.png
```

The default parser uses the PyMuPDF coordinate-preserving fast path for digital
pages and tries pinned Docling for scanned or hybrid documents. If Docling
cannot produce grounded blocks, or PyMuPDF table extraction fails, the page is
rendered for the configured local Gemma 3 multimodal endpoint. It is never
silently treated as a reliable text-only dump. The pinned Docling 2.62 adapter
has been exercised against the scanned fixture, including conversion of
Docling's bottom-left boxes into PDF point coordinates.

## Phase 2 retrieval

Start local Qdrant with `docker compose up -d qdrant`, then install the pinned
retrieval extras and index the JSONL chunks produced by ingestion:

```powershell
uv sync --extra retrieval
uv run findociq-retrieval index chunks.jsonl
uv run findociq-retrieval query "What changed in revenue?" --strategy naive
uv run findociq-retrieval query "What changed in revenue?" --strategy hybrid
uv run findociq-retrieval query "What changed in revenue?" --strategy hybrid_rerank
```

The strategies are assembled from `configs/retrieval/`. `naive` uses dense
BGE-M3 search, `hybrid` uses Qdrant's dense+sparse reciprocal-rank fusion, and
`hybrid_rerank` retrieves 50 candidates before scoring the final 8 with
`BAAI/bge-reranker-v2-m3`. Retrieval hits retain the original chunk and its
page-level provenance; answer generation is deliberately not part of this
phase.

## Phase 3 evaluation

The evaluation contract lives in [evals/EVAL_SCHEMA.md](evals/EVAL_SCHEMA.md).
The six initially table-derived benchmark values have also been checked against
rendered source PDF pages; see the
[source value audit](evals/datasets/SOURCE_VALUE_AUDIT.md).
Run the deterministic smoke sweep with:

```powershell
uv run python -m evals.run --sweep `
  --answer-predictions evals/datasets/smoke_answers.json
```

It writes separate retrieval and answer tables to `evals/results/`. Add
`--live` to sweep local Qdrant/BGE results instead of the explicitly labeled
fixture records. Retrieval metrics are never blended with answer or citation
metrics.

## Phase 4 reasoning

Reasoning is explicitly single-pass or two-pass. Pass 1 extracts figures with
page/bbox provenance; pass 2 receives only that structured extraction, never
the raw retrieval chunks. Both paths require at least one grounded citation.
The local OpenAI-compatible Gemma client defaults to
`google/gemma-3-12b-it`. vLLM serves that canonical name from the pinned
`gaunernst/gemma-3-12b-it-int4-awq` artifact. The 4B laptop and 27B ceiling
tiers are in `configs/model_tiers/`; all three use pinned AWQ int4 artifacts,
temperature 0, fixed seeds, and are consumed by `findociq-serve`.

```powershell
docker compose --profile gpu up -d --build qdrant vllm api
uv sync --extra retrieval --extra dev --extra api --extra observability
uv run findociq-reason "What was revenue in FY25?" `
  --retrieval-hits retrieval_hits.json `
  --pipeline-config configs/pipeline/two_pass.yaml
```

To launch a different tier in a native Linux vLLM environment:

```powershell
uv run findociq-serve --tier configs/model_tiers/laptop.yaml
uv run findociq-serve --tier configs/model_tiers/ceiling.yaml
```

`VllmGemmaClient` is a concrete backend-selected client used by the CLI, API,
and live eval composition roots. Its OpenAI-compatible request contract, the
pinned vLLM launch path, and live 4B AWQ GPU generation are tested. Full 12B GPU
inference has not yet been live-run. The published live-smoke results used the
CPU validation environment documented above and the explicitly configured
Ollama 4B fallback.

Ollama remains an explicit laptop fallback:

```powershell
ollama pull gemma3:4b
uv run findociq-reason "What was revenue in FY25?" `
  --retrieval-hits retrieval_hits.json `
  --generation-config configs/reasoning/gemma_ollama.yaml
```

The fixture evaluation compares both modes with `--reasoning-predictions`.
For a live corpus-backed run, start Qdrant, index the audited corpus once, and
run retrieval plus both reasoning modes:

```powershell
docker compose up -d qdrant
uv run findociq-retrieval index corpus/phase1_chunks
uv run python -m evals.run --sweep --live --live-reasoning `
  --dataset evals/datasets/phase4_corpus.jsonl `
  --results-dir evals/results/phase4_live
```

The reasoning table remains separate from retrieval quality. Live prediction
records retain `(document, page, bbox)` citations and are written beside the
summary results.

### Current live smoke result

The pinned five-question English run in `evals/results/phase4_live/` produced:

| Pipeline | Numeric exact (n=5) | Citation F1 (n=5) |
|---|---:|---:|
| Single pass | 1.000 (n=5; expanding to 80) | 0.800 |
| Two pass | 0.400 | 0.200 |

Hybrid reranking improved retrieval Recall@1 from `0.600` to `0.800` and
reached Recall@5 of `1.000` (`n=5`; expanding to `n=80`). Three two-pass cases
in this historical run are scored as failures because Gemma omitted required
pass-1 fields. Debugging showed these were schema-extraction failures, not an
established model-quality ceiling: one response omitted the echoed question
and two omitted citations. The current parser restores the caller's question
and repairs a missing citation only when the figure value matches exactly one
retrieved provenance; ambiguous or ungrounded output still fails closed. The
table remains the original run and has not been rescored after that fix.

## Phase 5 observability

Observability is typed and fans every immutable structural span to local JSONL
and, when enabled, the self-hosted Langfuse OTLP endpoint. Embedding, search,
reranking, text generation, vision generation, reasoning passes, API requests,
and citation validation are spanned. Raw questions, prompts, answers, images,
and document text are never written. Generation spans carry the prompt template
ID, its content-derived version and SHA-256, model revision, configuration hash,
and evaluation dataset hash so prompt changes remain attributable without
capturing rendered borrower content. The evaluation configuration hash covers
single-pass, both two-pass templates, and the two-pass retry template. Traces also carry
counts, durations, and exception types.

FinDocIQ reuses an existing self-hosted Langfuse instance instead of starting
a second stack. Supply that project's keys to the process and keep its base URL
in `configs/observability/langfuse.yaml`:

```powershell
$env:LANGFUSE_PUBLIC_KEY = "<project-public-key>"
$env:LANGFUSE_SECRET_KEY = "<project-secret-key>"
uv sync --extra observability
```

The exporter was integration-tested against the existing local Langfuse
v3.174.1 deployment on port 3000: a structural span was ingested and read back
with one observation and no trace input or output content.

Enable tracing for the corpus-backed evaluation with:

```powershell
uv run python -m evals.run --sweep --live --live-reasoning `
  --dataset evals/datasets/phase4_corpus.jsonl `
  --results-dir evals/results/phase5_observability `
  --observability-config configs/observability/local.yaml
```

The run writes `traces.jsonl`, `observability_summary.json`, and
`observability_summary.md`. Operational cache/non-empty rates and p50/p95
latencies remain separate from Recall@k, answer accuracy, and citation scores.
The no-op recorder is the default, and tests verify that enabling tracing does
not change retrieval rankings or returned provenance.

The current `n=5` question smoke run (expanding to `n=80`) produced 98 spans
across 30 operational runs, with a `0.250` query-cache hit rate and `1.000`
non-empty retrieval rate (`n=5` questions, expanding to `n=80`; 20 retrieval
calls). Three failed runs correspond
to the historical two-pass schema validation failures described above.
`retrieval.rerank` p50 was 105.5 seconds because the pinned reranker ran cold on
CPU; this is a local smoke-run diagnostic, not GPU or production latency.
Retrieval and answer scores were unchanged from Phase 4.

## Phase 6 cross-lingual evaluation

The Phase 6 dataset contains five matched financial facts, each asked once in
English and once in Hindi. Every pair shares the same relevant chunk IDs,
numeric answer, and exact `(document, page, bbox)` citation targets. Pair
validation runs before model loading, preventing corpus or label differences
from being mistaken for language effects. Official company names remain in
English inside Hindi questions to avoid entity-translation ambiguity.

```powershell
uv run python -m evals.run --sweep --live --live-reasoning `
  --dataset evals/datasets/phase6_cross_lingual.jsonl `
  --results-dir evals/results/phase6_cross_lingual `
  --observability-config configs/observability/phase6.yaml `
  --validate-cross-lingual-pairs
```

The pinned paired smoke run used `n=5` facts per language and is expanding to
`n=80` total questions. It produced:

| Pipeline | Metric | English (n=5) | Hindi (n=5) | Hindi - English |
|---|---|---:|---:|---:|
| Hybrid rerank | Recall@1 | 0.800 | 0.800 | +0.000 |
| Hybrid rerank | Recall@5 | 1.000 (n=5; expanding to 80) | 1.000 (n=5; expanding to 80) | +0.000 |
| Single pass | Numeric exact | 1.000 (n=5; expanding to 80) | 1.000 (n=5; expanding to 80) | +0.000 |
| Single pass | Citation F1 | 0.800 | 0.800 | +0.000 |
| Two pass | Numeric exact | 0.400 | 0.000 | -0.400 |
| Two pass | Citation F1 | 0.200 | 0.000 | -0.200 |

Naive and hybrid retrieval showed language gaps, while the pinned multilingual
reranker recovered parity at Recall@1 and Recall@5. All five Hindi two-pass
queries failed pass-1 structured-output validation; the errors are retained in
`reasoning_predictions.json` and are not repaired by tuning on this dataset.
The paired set validates the cross-lingual measurement path but is too small to
support a production benchmark claim.

## Phase 7 expanded retrieval evaluation

Phase 7 expands retrieval evaluation to `n=31` English questions across six
categories: single lookup, multi-year numeric, derived metric, qualitative
flag, cross-document, and negative/abstention. Every question resolves to the
reviewed 1,236-chunk ingestion, and the checked-in dataset contains no
`VERIFY` placeholders.

```powershell
uv run python scripts/validate_dataset.py `
  --dataset evals/datasets/phase7_corpus.jsonl `
  --corpus corpus/phase1_chunks
uv run python -m evals.run --sweep --live `
  --dataset evals/datasets/phase7_corpus.jsonl `
  --results-dir evals/results/phase7_live
```

The live sweep ran on 2026-08-13 with BGE-M3 and
bge-reranker-v2-m3 on an NVIDIA GeForce RTX 4080 Laptop GPU using
`torch 2.13.0+cu130`; Qdrant contained all 1,236 chunks. These are retrieval
measurements only, not answer-generation scores. With `n=31`, they are more
representative than the earlier five-question smoke runs but remain an early,
single-corpus benchmark.

| Strategy | Backend | Questions | Recall@1 | Recall@5 | Recall@8 | MRR | nDCG@8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Naive dense | live GPU | 31 | 0.290 | 0.548 | 0.661 | 0.455 | 0.489 |
| Hybrid RRF | live GPU | 31 | 0.242 | 0.613 | 0.726 | 0.413 | 0.478 |
| Hybrid RRF + BGE rerank | live GPU | 31 | 0.403 | 0.806 | 0.871 | 0.609 | 0.671 |

The complete machine-readable and rendered outputs are checked in under
`evals/results/phase7_live/`; the dataset SHA-256 is
`f1356732204197249c9ded434e93eb74b05ba23a9723e5f24ea7feb573d36882`.

### Phase 7 annual-report extension

Four `phase7_annual_reports_*.jsonl` files add 14 visually verified questions
from the official SBI Card, JM Financial, and Likhitha Infrastructure FY2025
annual reports, including one cross-document PAT comparison. Each row resolves
to a reviewed answer-bearing chunk with exact `(document, page, bbox)`
provenance. The source PDFs are hash-locked in
`configs/corpus/phase7_annual_reports.lock.json`; the full local ingestion is
gitignored, while CI validates the reviewed six-anchor manifest.
The source-page review and exclusions are recorded in
`evals/datasets/PHASE7_ANNUAL_REPORT_AUDIT.md`.

The annual-report extension uses the searchable digital-page fast path because
the configured Gemma vision port was unavailable during this run. It therefore
does not claim complete remediation of every visual page in the three reports.
Two provisional negative questions were excluded: SBI Card actually discloses
an interim dividend of Rs. 2.50 per share, and absence of a Likhitha order-book
value cannot be established from a partial visual ingestion.

```powershell
uv run python scripts/validate_dataset.py `
  --dataset evals/datasets/phase7_annual_reports_sbicard.jsonl `
  --dataset evals/datasets/phase7_annual_reports_jm_financial.jsonl `
  --dataset evals/datasets/phase7_annual_reports_likhitha.jsonl `
  --dataset evals/datasets/phase7_annual_reports_cross_document.jsonl `
  --corpus corpus/phase7_annual_report_chunks
```

## FastAPI service

The HTTP layer only validates requests and delegates to `FinDocIQService`;
retrieval and reasoning logic stays in `src/findociq/`. The query response
schema requires at least one `(document, page, bbox)` citation.

```powershell
uv sync --extra api --extra docling --extra gpu --extra retrieval --extra observability
uv run findociq-api --config configs/api/default.yaml
Invoke-RestMethod http://127.0.0.1:8989/healthz
```

For the conflict-free Docker stack, host ports are FinDocIQ API `8989`, vLLM
`8900`, Qdrant REST `6999`, and Qdrant gRPC `7000`. Container-to-container
traffic retains the images' standard internal ports. The 12 GB laptop Docker
profile uses single-request vLLM concurrency, eager execution, the V1 model
runner for Docker Desktop/WSL compatibility, and an 8K context window. It serves
the pinned 4B AWQ laptop tier. The 12B tier remains the project default for
larger GPUs, but it cannot use vLLM's sleep allocator on this RTX 4080: the
model plus sleep-mode private pools exceed 12 GB of VRAM.

The GPU profile enables vLLM sleep mode on a localhost-only port. During an
authenticated document batch, vLLM level-1 sleep offloads its weights to system
RAM and releases VRAM; Docling/OCR and BGE-M3 then process the complete batch.
FinDocIQ releases ingestion models and wakes vLLM in a failure-safe `finally`
path before generation resumes. The development control endpoints are not
exposed beyond localhost. Local smoke runs measured 5.0-15.7 seconds to enter
level-1 sleep and 1.7-4.9 seconds to wake. Those are single-machine switching
observations, not throughput benchmarks; batching amortizes them across all PDFs.

`POST /v1/query` accepts `question`, optional `question_id`, and optional
`mode` (`single_pass` or `two_pass`). The configured two-pass mode is used when
the request omits it.

`POST /extract` is the versioned black-box integration boundary for downstream
applications such as FunderMatch. It accepts `question` and optional
`question_id`, always runs the two-pass path, and returns contract version `1.0`
with structured figures and a `(document_id, page_number, bbox)` citation on
every figure. Downstream repositories must copy this public response schema and
call it over HTTP; they must not import `findociq` internals.

## Production document-security profile

Set `FINDOCIQ_PRODUCTION_GUARDRAILS_ENABLED=true` to replace the development endpoints
with application-scoped contract `2.0`. In this profile, `/v1/query` returns 404 and
ingestion, activity, extraction, and retention deletion require a five-minute service
JWT with the correct audience, role, application scope, one-time `jti`, and correlation
ID. Extraction accepts allow-listed `metric_ids`; arbitrary production questions are
not accepted.

The producer-owned v1/v2 Pydantic schemas are exported to
`contracts/findociq-public-http-contract.json`. CI verifies that snapshot against the
current FinDocIQ models and checks FunderMatch's independent HTTP consumer models against
the same bundle. This catches cross-repository drift without allowing an internal package
import across the service boundary.

Every PDF is staged in container `tmpfs`, checked by pinned ClamAV, and rejected for
malware, malformed/encrypted structure, JavaScript, launch/open actions, embedded
files, remote links, forms, or other active content before GPU acquisition. Parsed
text is treated as untrusted evidence and locally redacted for Aadhaar, PAN, accounts,
IFSC, phone, email, tax identifiers, credentials, and secrets before BGE-M3/Qdrant.
Qdrant payloads carry `application_id`, and production retrieval requires both the
application and its owned document IDs.

Raw PDFs and quarantine objects use AES-256-GCM envelope encryption with a per-object
data key. The master key is read from `FINDOCIQ_DOCUMENT_MASTER_KEY_FILE`; plaintext is
materialized only in the container's temporary filesystem and removed after parsing.
`findociq-rotate-document-keys` rewraps data keys under a new versioned master key
without rewriting PDF ciphertext. Scan, DLP, storage, ownership, and deletion receipts
contain hashes, counts, versions, and safe disposition codes—never matched PII.

Run `findociq-migrate-security` before starting production. The shared policy in
`configs/guardrails/production.yaml` is typed and hashed, and `/healthz` reports its
identity so FunderMatch can fail readiness on policy drift. Production Docker exposes
no FinDocIQ host port; only the FunderMatch TLS proxy is public. The development token
and localhost HTTP workflow remain available only when production guardrails are off.

For an independent deployment rather than the integrated FunderMatch Compose stack,
provision the FinDocIQ service-JWT key as one half of a coordinated service identity.
Use FunderMatch's `fundermatch-service-identity provision` command to create distinct
deployment copies containing the same high-entropy value, install the FinDocIQ copy
through its secret manager as `FINDOCIQ_SERVICE_JWT_SECRET_FILE`, and run
`fundermatch-service-identity verify` against both installed copies and both production
policy bundles before startup. Rotate both deployments together from new versioned
secret paths during an offline maintenance window; never overwrite or rotate only the
FinDocIQ copy.

## Provenance contract

Every chunk contains one or more provenance objects with a document ID,
one-based page number, and a PDF-coordinate bounding box `(x0, y0, x1, y1)`.
Coordinates use the source page's point-space and include the page dimensions,
so a reviewer can render and visually verify the evidence.

Tables are emitted as one `TableChunk`; the chunker does not apply text size
limits to table content. Adjacent caption text and the preceding paragraph are
attached to the table chunk.

## Known Phase 1 limitations

- The PyMuPDF fast path identifies tables through its rule-based table finder;
  borderless and visually complex tables may require Docling or vision.
- The reproducible Phase 1 corpus uses 18 official-source PDFs. Its historical
  checked-in audit predates the now-configured Gemma multimodal endpoint; rerun
  the corpus audit before claiming those three annual reports as remediated.
  See `docs/phase1_corpus_audit.md`.

## Known Phase 2 limitations

- Retrieval unit tests inject deterministic model/store doubles; running the
  real BGE-M3 and bge-reranker weights requires the pinned `retrieval` extra and
  a local Qdrant service.
- The Phase 7 retrieval benchmark has 31 questions from one ICRA-rationale
  corpus. It is useful for strategy comparison, not a production-quality or
  cross-domain claim.
## Authenticated document ingestion

Downstream local applications can index a PDF through `POST /v1/documents`, or send
multiple PDFs through `POST /v1/document-batches` so one GPU lease covers the entire
batch. Configure an external storage root and a separate write token before starting
the API:

```powershell
$env:FINDOCIQ_DOCUMENT_ROOT = `
  "$env:USERPROFILE\Documents\FunderMatch_Data\findociq_documents"
$env:FINDOCIQ_INGEST_TOKEN = "replace-with-a-separate-random-ingestion-token"
uv run --extra api --extra docling --extra gpu --extra retrieval findociq-api --port 8989
```

The versioned request carries a PDF as base64 plus its SHA-256. FinDocIQ validates the
type, size, and digest, then performs Docling/PyMuPDF parsing, provenance-preserving
chunking, BGE-M3 embedding, and Qdrant indexing. `/extract` and `/v1/query` accept
optional `document_ids`, preventing evidence from unrelated borrowers entering a case.
The batch endpoint uses the same versioned document contracts and returns one receipt
per PDF. Prefer it for borrower uploads: repeatedly sleeping and waking the model
for individual documents adds avoidable switching overhead.

Callers may include a sanitized `batch_id` and poll
`GET /v1/document-batches/{batch_id}/activity?after=<sequence>` with the ingestion
token while the batch request is running. The activity contract reports GPU waiting,
the current filename, parsing/OCR, chunking, embedding, Qdrant indexing, model release,
and vLLM readiness. It is deliberately bounded and in-memory: durable user-facing job
history belongs to the downstream application, and document text or model payloads are
never included in these events.
