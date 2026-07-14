# Multimodal retrieval — evaluation freeze record (E11A)

Frozen implementation under evaluation for the Phase 3 A/B/C benchmark.

| Field | Value |
|---|---|
| Date/time (UTC) | 2026-07-14T14:25:04Z |
| Branch | `feature/phase3-multimodal-retrieval` |
| HEAD commit | `ecde1d8` — Add image+text fusion with _msearch execution and post-fusion in_tour policy |
| Working tree | clean of Phase-3 files (only pre-existing untracked benchmark assets / settings.local / tmp remain, unrelated) |
| Isolated backend | uvicorn on 127.0.0.1:8001 loading `backend/.env.v4.local` (live backend NOT touched) |

## Models / revisions (pinned)
- Text/entities: `Qwen/Qwen3-Embedding-8B` @ `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`, dim 4096
- Visual (i2i + t2i): `Qwen/Qwen3-VL-Embedding-8B` @ `2c4565515e0f265c6511776e7193b22c0968ddc7`, dim 4096
- Precision bf16, L2-normalized, max_length 2048

## Dependency versions
torch 2.8.0+cu128 · transformers 5.7.0 · sentence-transformers 5.4.1 ·
opensearch-py 3.2.0 · fastapi 0.116.1 · pydantic 2.13.4 · Python 3.11.15

## Physical indices (read-only during benchmark)
- `cultural_heritage_artifacts_v4` = 19112
- `cultural_heritage_images_v4` = 35741
- `cultural_heritage_authors_v4` = 746 · `_sets_v4` = 1274 · `_exhibitions_v4` = 1049
- Search pipeline: `nlp-search-pipeline` (min_max + arithmetic_mean [0.7 dense, 0.3 lexical])

## Frozen multimodal configuration
`MULTIMODAL_RRF_K=60` · `MULTIMODAL_ARTIFACT_WEIGHT=1.0` · `MULTIMODAL_IMAGE_WEIGHT=0.7`
· `MULTIMODAL_MIN_IMAGE_SCORE=0.64` · `MULTIMODAL_IMAGE_TOP_K=30`
· `MULTIMODAL_I2I_WEIGHT=1.0` · `MULTIMODAL_IMAGE_TEXT_ARTIFACT_WEIGHT=0.5`
· `MULTIMODAL_USE_MSEARCH=true` · `MULTIMODAL_IN_TOUR_MARGIN=0.0`

These stay CONSTANT across A/B/C. Only `MULTIMODAL_RETRIEVAL_MODE` varies.

## Modes (formal definitions)
- **OFF** (`MULTIMODAL_RETRIEVAL_MODE=off`): text_to_image is never executed; no
  router; no VL load on the text path; ranking byte-equal to the pre-Phase-3
  baseline.
- **INTENT** (`=intent`): text_to_image runs only when the deterministic visual
  router returns TEXT_AND_VISUAL; artifact_search always runs on RAG-eligible
  textual messages.
- **ALWAYS_ELIGIBLE** (`=always`): text_to_image runs on every message that
  reaches retrieval/RAG. It does NOT turn greetings, small talk, administrative
  questions, or llm_only messages without search intent into retrieval — the
  router still gates whether a message is a search at all. Logged as
  `always_eligible` to avoid the reading "every message no matter what".

No index/mapping/embedding/alias/live-backend change is permitted during the
benchmark. Any objective bug found → document, add regression test, isolated
commit, restart the affected benchmark.
