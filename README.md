# Patrimonio360 Backend

FastAPI for the embed backend.

For the end-to-end local development setup (`frontend` + `backend` + static `tours` + `multiview_worker`), see [../README.md](../README.md).

## Run (dev)

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128 torchvision==0.23.0+cu128
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Install `torch`/`torchvision` separately before `requirements.txt`. Alternative GPU/CPU install commands are in [../INSTALL.md](../INSTALL.md).

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
CHAT_PREWARM_INCLUDE_MULTIVIEW_WORKER=false
```

With these flags, the backend pays the loading cost during startup instead of on the first user interaction.

Operational terminal logging:

```env
BACKEND_LOG_ENABLED=true
BACKEND_LOG_LEVEL=INFO
BACKEND_ACCESS_LOG_ENABLED=true
BACKEND_LOG_HEALTHCHECKS=false
BACKEND_RAG_DEBUG_ENABLED=true
BACKEND_RAG_DEBUG_MAX_CHARS=40000
```

`BACKEND_RAG_DEBUG_ENABLED=true` prints readable, indented JSON blocks for the RAG pipeline in the terminal: router decision, query plan, OpenSearch request body, retrieved docs, visible context, and final prompt metadata. Embedding vectors are summarized instead of dumping all float values. These logs are separate from the JSONL evaluation logs.

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
- `GET /api/v1/chat/images/{image_ref}` (serve image by filename or safe relative `local_path` when `IMAGE_ASSET_ROOT` is configured)

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
