# Patrimonio360 Backend

FastAPI for the embed backend.

For the end-to-end local development setup (`frontend` + `backend` + static `tours` + `multiview_worker`), see [../README.md](../README.md).

## Current scope

- Open API surface (no auth/security yet)
- Chat endpoint with LLM service in dev mode
- Hybrid chat pipeline: router (`rag|llm_only`) + in-memory session state (TTL) + final answer step
- Environment-based settings aligned with `../Indexer` for:
  - OpenSearch connection/index names
  - Embedding model ids and dimensions
- LLM provider support using OpenAI-compatible chat completions
- Hybrid retrieval implemented (sentence-transformers for text + Qwen multimodal + OpenSearch vector/text query)
- Optional second-stage reranking with `Qwen/Qwen3-Reranker-4B` (pairwise query-document scoring)
- Image search pipeline: upload image -> similarity search on `cultural_heritage_images` -> artifact lookup by `artifact_id` -> final LLM response
- 3D model search pipeline: upload `.glb|.gltf|.obj` -> persistent multiview worker -> 3 views first, then +2 only when confidence is low -> hybrid image retrieval -> artifact lookup by `artifact_id` -> final LLM response

## Run (dev)

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

If you use image search, ensure `torchvision` is installed in the same environment as `torch` (it is included in `requirements.txt`).

For 3D model search, install the worker dependencies once:

```bash
cd backend/multiview_worker
npm install
```

## Offline benchmark

Run from `backend`:

```bash
python -m benchmarks.run --cases benchmarks/cases/benchmark_cases.json --variant full
```

Useful flags:
- `--variant full --variant no_rewriting`
- `--output-dir benchmark_runs`
- `--no-warmup`
- `--list-variants`

Outputs are written into one timestamped run folder with:
- `results.json`
- `results.csv`
- `summary.json`
- `summary.csv`
- `report.md`

Notes:
- The benchmark reuses the current production retrieval code paths directly from `app.services.*`.
- It preserves the exact current OpenSearch query behavior; it does not introduce benchmark-only query variants.
- Benchmark runs prewarm OpenSearch and model stacks by default so the first measured case does not pay the cold start cost. Use `--no-warmup` if you explicitly want to measure cold start.
- `full` and `no_rewriting` are currently supported without changing retrieval behavior.
- `bm25_only`, `dense_only`, `hybrid`, and `no_filtering` are scaffolded as placeholders and are reported as skipped until the codebase supports them without altering the current query behavior.

Optional API startup prewarm:

```env
CHAT_PREWARM_ON_STARTUP=true
CHAT_PREWARM_INCLUDE_MULTIMODAL=true
CHAT_PREWARM_INCLUDE_RERANKER=false
CHAT_PREWARM_INCLUDE_MULTIVIEW_WORKER=false
```

With these flags, the backend pays the loading cost during startup instead of on the first user interaction.

Tests:

```bash
cd backend
python -m unittest discover -s tests
```

## OpenSearch backup/restore (WSL + Docker)

Current local setup uses:
- container: `opensearch`
- volume: `opensearch-data`

Stop the OpenSearch container before backing up or restoring the data volume.

Backup:

```bash
docker stop opensearch

docker run --rm \
  -v opensearch-data:/from \
  -v "$PWD:/backup" \
  alpine \
  sh -c 'cd /from && tar czf /backup/opensearch-data-2026-04-13.tar.gz .'

docker start opensearch
```

Restore on another machine:

```bash
docker volume create opensearch-data

docker run --rm \
  -v opensearch-data:/to \
  -v "$PWD:/backup" \
  alpine \
  sh -c 'cd /to && tar xzf /backup/opensearch-data-2026-04-13.tar.gz'
```

Use the same OpenSearch image/version on the destination machine when possible.

## Endpoints

- `GET /health`
- `GET /api/v1/chat/health`
- `POST /api/v1/chat/messages`
- `POST /api/v1/chat/messages/regenerate`
- `POST /api/v1/chat/messages/image` (multipart upload)
- `POST /api/v1/chat/messages/model` (multipart upload for `.glb`, `.gltf`, `.obj`)
- `POST /api/v1/chat/messages/stream` (SSE status + final payload)
- `POST /api/v1/chat/messages/regenerate/stream` (SSE status + final payload)
- `POST /api/v1/chat/messages/image/stream` (SSE status + final payload)
- `POST /api/v1/chat/messages/model/stream` (SSE status + final payload)
- `GET /api/v1/chat/images/{original_image_name}` (serve image by original filename when `IMAGE_ASSET_ROOT` is configured)

## Chat payload

`POST /api/v1/chat/messages`

```json
{
  "model_override": "carminho/AMALIA-9B-50-DPO",
  "museum_slug": "mnaz",
  "museum_id": "mnaz",
  "message": "Extrai entidades e intencao.",
  "response_format": {
    "type": "json_object"
  }
}
```

```json
{
  "model_override": "carminho/AMALIA-9B-50-DPO",
  "museum_slug": "mnaz",
  "museum_id": "mnaz",
  "message": "Responde ao visitante sobre o percurso inicial.",
  "response_format": { "type": "text" }
}
```

Notes:
- Use `snake_case` fields in requests (`museum_slug`, `conversation_id`, `response_format`, `model_override`).
- `response_format` must be an object: `{ "type": "text" }` or `{ "type": "json_object" }`.
- Default configured URL is `https://amalia.novasearch.org/vlm/chat/completions`.
- The service uses OpenAI-compatible `chat.completions` and prefers `parse(...)` when supported by the client SDK.
- Model is unified by default (`LLM_MODEL=carminho/AMALIA-9B-50-DPO`) for all formats.
- Embedding defaults are aligned with `../Indexer` (`Qwen3-Embedding-4B`, `2560` dims for text).
- `LLM_MAX_TOKENS=0` means no max token cap from this API layer.
- Session and routing knobs:
  - `CHAT_HISTORY_WINDOW`
  - `CHAT_SESSION_TTL_SECONDS`
  - `CHAT_ROLLING_SUMMARY_MAX_CHARS`
  - `CHAT_ENABLE_RAG`
  - `CHAT_ENABLE_LLM_LEXICAL_QUERY` (uses the configured LLM to clean PT/EN lexical OpenSearch queries; falls back to local normalization)
  - `CHAT_USE_QUERY_EMBEDDINGS` (if `false`, retrieval is skipped)
  - `CHAT_RETRIEVAL_CANDIDATES` (first-stage retrieval candidate count before rerank)
  - `CHAT_RETRIEVAL_TOP_K`
  - `CHAT_IN_TOUR_BOOST` (boost score when `in_tour=true`; set `0` to disable)
  - `CHAT_ENABLE_RERANKING` (enable second-stage rerank)
  - `RERANKER_MODEL_ID` (default `Qwen/Qwen3-Reranker-4B`)
  - `RERANKER_INSTRUCTION`
  - `RERANKER_PREFER_BF16`
  - `RERANKER_MAX_LENGTH`
  - `RERANKER_BATCH_SIZE`
  - `CHAT_IMAGE_RETRIEVAL_TOP_K`
  - `CHAT_IMAGE_ARTIFACT_TOP_K`
  - `CHAT_IMAGE_DEFAULT_MESSAGE`
  - `CHAT_MODEL_DEFAULT_MESSAGE`
  - `CHAT_MODEL_FIRST_PASS_VIEWS`
  - `CHAT_MODEL_TOTAL_VIEWS`
  - `CHAT_MODEL_LOW_CONFIDENCE_SCORE_THRESHOLD`
  - `CHAT_MODEL_CACHE_SIZE`
- `IMAGE_ASSET_ROOT` (filesystem root used to resolve `original_image_name` for chat image display)
- `POI_TOURS_DIR` (optional directory with `panorama-overlays-inventory-<museum>.json` for tour navigation targets)
  - `MULTIVIEW_WORKER_HOST`
  - `MULTIVIEW_WORKER_PORT`
  - `MULTIVIEW_WORKER_START_TIMEOUT_SECONDS`
  - `MULTIVIEW_RENDER_SIZE`
  - `MULTIVIEW_RENDER_BACKGROUND`
  - `MULTIVIEW_RENDER_FOV`
  - `MULTIVIEW_RENDER_DPR`
  - `MULTIVIEW_RENDER_STRATEGY`
  - `MULTIVIEW_RENDER_OVERSAMPLE`
  - `MULTIVIEW_RENDER_ORBIT_MARGIN`
  - `MULTIVIEW_RENDER_ENSURE_TOP`
  - `MULTIVIEW_RENDER_DELAY_MS`
- Logging knobs:
  - `LOG_LEVEL` (`DEBUG`, `INFO`, ...)
  - `LOG_JSON` (`true` for structured JSON logs)
  - `LOG_JSON_PRETTY` (`true` for indented multiline JSON logs)
  - `LOG_JSON_INDENT` (spaces for indentation, default `2`)
  - `LOG_CHAT_MESSAGES` (`true` to include raw user text in logs; default `false`)
  - `LOG_CHAT_STATE_HISTORY` (`true` to include full state history; default `false`)

## Reranking flow example

With:

```env
CHAT_ENABLE_RAG=true
CHAT_USE_QUERY_EMBEDDINGS=true
CHAT_RETRIEVAL_CANDIDATES=15
CHAT_RETRIEVAL_TOP_K=5
CHAT_ENABLE_RERANKING=true
RERANKER_MODEL_ID=Qwen/Qwen3-Reranker-4B
```

Runtime flow:
1. OpenSearch retrieval returns top `N=15` candidates.
2. Reranker scores each `(query, document)` pair.
3. Candidates are reordered by reranker score.
4. Top `K=5` reranked documents are sent to the final answer prompt.

## Image Search Flow (chat upload)

1. Frontend sends multipart request with `image` and optional `message`.
2. Backend generates multimodal embedding using `Qwen/Qwen3-VL-Embedding-2B`.
3. OpenSearch searches `OPENSEARCH_INDEX_IMAGE` by vector similarity.
4. Backend extracts `artifact_id` from image hits and fetches artifact docs from `OPENSEARCH_INDEX_ARTIFACT`.
5. LLM final answer is generated using those artifact docs (with `description` + `inventory`), and `artifact_id` is hidden from final user text.

## 3D Model Search Flow (chat upload)

1. Frontend sends multipart request with `model_file` and optional `message`.
2. Backend calls a persistent Node/Puppeteer worker in `backend/multiview_worker`.
3. The worker renders 3 canonical views in memory and returns PNG bytes without writing to disk.
4. Backend generates multimodal embeddings for the 3 views in batch.
5. OpenSearch runs a hybrid image retrieval query across those views.
6. If the top confidence is low, the worker renders 2 additional views from the same 5-view camera set and the backend repeats retrieval with all 5 embeddings.
7. Backend extracts `artifact_id` from image hits, fetches artifact docs, and sends them to the final answer prompt.
