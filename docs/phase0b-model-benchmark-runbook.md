# Phase 0B — Model Benchmark Runbook (RTX PRO 6000)

Phase 0A (done) prepared the infrastructure: decomposed `bm25_only`/`dense_only`
variants, new case modes `text_to_image`/`image_text` with leave-self-out
(`exclude_image_ids`), metrics `recall@10`/`ndcg@10`/image-level hits, remapped
case ground truth (`benchmark_cases_live_ids.json`), a starter multimodal case
file, the canary parity script, the offline model-bench runner, and config
cleanup (all `.env` keys now declared on `Settings`).

Phase 0B runs the actual comparisons on the GPU workstation. **Everything below
is read-only with respect to OpenSearch** — no index creation, no alias
changes, no re-embedding of the production collection.

Decisions this phase must produce (before any v4 index creation — dims are
immutable):

1. VL model: `Qwen/Qwen3-VL-Embedding-2B` (2048-d) vs `-8B` (4096-d, verify dim).
2. Text model: `Qwen/Qwen3-Embedding-4B` (2560-d) vs `-8B` (4096-d).
3. Query-instruction on/off for the text model (`--query-prompt builtin` ablation).
4. AMALIA `/vlm` viability as caption generator (single probe).

---

## 0. Preconditions

```powershell
# from patrimonio360_assistant/backend — venv with requirements.txt active
python -m unittest tests.test_settings_env_parity tests.test_metrics tests.test_benchmark_loader tests.test_multimodal_case_modes
```

Known pre-existing failures (NOT Phase 0A regressions, safe to ignore for now):
- `tests.test_benchmark_loader.test_smoke_fixture_parses_all_modes_and_resolves_paths`
  — fixture points at machine-local `assets/images/1000.jpg` etc. that are absent.
- `tests.test_opensearch_mapping_fields.test_text_retrieval_page_uses_fixed_retrieval_window`
  — documents the known `knn_k=1000` hardcode (plan §4.11); fix scheduled with
  the Phase 3 retrieval work, not before (it changes production ranking).

## 1. Canary embedding parity (indexer env vs backend env)

```powershell
# in the indexer environment (from patrimonio360_indexer/):
python scripts/canary_embedding_parity.py --write-reference canary_reference.json

# in the assistant backend environment (venv that runs uvicorn):
python ..\patrimonio360_indexer\scripts\canary_embedding_parity.py --check ..\patrimonio360_indexer\canary_reference.json
```

Gate: every cosine ≥ 0.999. If it fails, the two environments produce different
vectors for the same model — align library versions before trusting any
benchmark numbers. Re-run before every future reindex.

## 2. Reproduce the text/image baselines on the remapped cases

```powershell
# from backend/ — full production pipeline + isolated paths, 25 runnable cases
python -m benchmarks.run --cases benchmarks/cases/benchmark_cases_live_ids.json --variant full --variant bm25_only --variant dense_only --output-dir benchmark_runs
```

Notes:
- The historical 0.652/0.895 numbers came from the pre-reindex index+IDs; the
  remapped run establishes the NEW baseline. Record it in the run folder.
- 29 cases are disabled pending manual ground truth (see
  `benchmarks/cases/id_remap_report.json` → `problems` and
  `disabled_unmapped_cases`; mostly pre-existing scaffolds T21–T28, the
  ambiguous `Vestido` text cases, and azulejo image cases I05–I13 whose assets
  are different renditions, not copies). Curate when convenient; the benchmark
  is meaningful with the 25 enabled cases meanwhile.
- `--no-assistant-selection` skips the AMALIA selector if the remote LLM is
  unavailable; retrieval metrics still run.

## 3. First multimodal numbers with the current VL-2B (query-side only)

```powershell
python -m benchmarks.run --cases benchmarks/cases/benchmark_cases_multimodal.json --variant full --output-dir benchmark_runs
```

This exercises `text_to_image` (VL text-mode query → live `visual_embedding`
kNN) and `image_text` (joint `{image,text}` query) against the LIVE images
index with leave-self-out. The `image_text` queries are auto-generated
placeholders — curate them into real compositional constraints (garment,
colour, pose, "homem/mulher", "veste/segura"...) before treating the numbers
as more than plumbing validation.

## 4. Offline VL model comparison (2B vs 8B)

The 8B download (~16 GB) happens here, not earlier.

```powershell
# from patrimonio360_indexer/

# 4a. Baseline VL-2B from cached checkpoint vectors (no doc re-embedding):
python scripts/bench_embedding_models.py --task visual --cases ..\patrimonio360_assistant\backend\benchmarks\cases\benchmark_cases_live_ids.json --extra-cases ..\patrimonio360_assistant\backend\benchmarks\cases\benchmark_cases_multimodal.json --pool-checkpoint .cache\museum_embeddings_mnaz\image_embeddings_qwen3_vl_2b.jsonl --model Qwen/Qwen3-VL-Embedding-2B --images-root ..\patrimonio360_assistant\backend --output bench_vl2b_mnaz.json

python scripts/bench_embedding_models.py --task visual --cases ..\patrimonio360_assistant\backend\benchmarks\cases\benchmark_cases_live_ids.json --extra-cases ..\patrimonio360_assistant\backend\benchmarks\cases\benchmark_cases_multimodal.json --pool-checkpoint .cache\museum_embeddings_traje_raiz_v2\image_embeddings_qwen3_vl_2b.jsonl --model Qwen/Qwen3-VL-Embedding-2B --images-root ..\patrimonio360_assistant\backend --output bench_vl2b_mnt.json

# 4b. Candidate VL-8B on sampled pools (embeds pool+queries with 8B; pool cached
#     in .cache/bench_pools so re-runs are cheap):
python scripts/bench_embedding_models.py --task visual --cases ..\patrimonio360_assistant\backend\benchmarks\cases\benchmark_cases_live_ids.json --extra-cases ..\patrimonio360_assistant\backend\benchmarks\cases\benchmark_cases_multimodal.json --pool-ndjson ..\raiz_scraper\output\raiz_index\museus\museu_nacional_do_azulejo\objetos_com_imagens.ndjson --limit 4000 --model Qwen/Qwen3-VL-Embedding-8B --images-root ..\raiz_scraper\output --batch-size 4 --output bench_vl8b_mnaz.json

python scripts/bench_embedding_models.py --task visual --cases ..\patrimonio360_assistant\backend\benchmarks\cases\benchmark_cases_live_ids.json --extra-cases ..\patrimonio360_assistant\backend\benchmarks\cases\benchmark_cases_multimodal.json --pool-ndjson ..\raiz_scraper\output\raiz_index\museus\museu_nacional_do_traje\objetos_com_imagens.ndjson --limit 4000 --model Qwen/Qwen3-VL-Embedding-8B --images-root ..\raiz_scraper\output --batch-size 4 --output bench_vl8b_mnt.json

# 4c. Apples-to-apples 2B control on the SAME sampled pools (2B pool embedding):
python scripts/bench_embedding_models.py --task visual --cases ..\patrimonio360_assistant\backend\benchmarks\cases\benchmark_cases_live_ids.json --extra-cases ..\patrimonio360_assistant\backend\benchmarks\cases\benchmark_cases_multimodal.json --pool-ndjson ..\raiz_scraper\output\raiz_index\museus\museu_nacional_do_azulejo\objetos_com_imagens.ndjson --limit 4000 --model Qwen/Qwen3-VL-Embedding-2B --images-root ..\raiz_scraper\output --batch-size 8 --output bench_vl2b_sampled_mnaz.json
```

Compare 4b vs 4c (same pool, same seed) — NOT 4b vs 4a (different pool sizes
bias the metrics). Decision inputs: `mode:text_to_image` and `mode:image_text`
recall/nDCG (the new capabilities), `mode:image` (must not regress),
`avg_query_embed_ms`, `peak_cuda_memory_gb`, indexing throughput implied by
pool embed time. The verdict fixes `images_v4` `visual_embedding` dimension.

## 5. Offline text model comparison (4B vs 8B) + instruction ablation

```powershell
# 4B baseline from cached artifact vectors:
python scripts/bench_embedding_models.py --task text --cases ..\patrimonio360_assistant\backend\benchmarks\cases\benchmark_cases_live_ids.json --pool-checkpoint .cache\museum_embeddings_mnaz\artifact_embeddings_qwen3_embedding_4b_2560.jsonl --model Qwen/Qwen3-Embedding-4B --output bench_text4b_mnaz.json

# 8B candidate on a sampled artifact pool (downloads 8B):
python scripts/bench_embedding_models.py --task text --cases ..\patrimonio360_assistant\backend\benchmarks\cases\benchmark_cases_live_ids.json --pool-ndjson ..\raiz_scraper\output\raiz_index\museus\museu_nacional_do_azulejo\objetos_com_imagens.ndjson --limit 4000 --model Qwen/Qwen3-Embedding-8B --output bench_text8b_mnaz.json

# 4B control on the same sampled pool + query-instruction ablation:
python scripts/bench_embedding_models.py --task text --cases ..\patrimonio360_assistant\backend\benchmarks\cases\benchmark_cases_live_ids.json --pool-ndjson ..\raiz_scraper\output\raiz_index\museus\museu_nacional_do_azulejo\objetos_com_imagens.ndjson --limit 4000 --model Qwen/Qwen3-Embedding-4B --output bench_text4b_sampled.json
python scripts/bench_embedding_models.py --task text --cases ..\patrimonio360_assistant\backend\benchmarks\cases\benchmark_cases_live_ids.json --pool-ndjson ..\raiz_scraper\output\raiz_index\museus\museu_nacional_do_azulejo\objetos_com_imagens.ndjson --limit 4000 --model Qwen/Qwen3-Embedding-4B --query-prompt builtin --output bench_text4b_sampled_instruct.json
```

Note: this measures DENSE-only text retrieval; the production text path is
hybrid, so a text-model switch also needs step 2's `dense_only` variant re-run
against a rebuilt index later. Treat step 5 as the go/no-go screen: if 8B does
not clearly beat 4B here, keep 4B (plan §22.2). The `--query-prompt builtin`
run decides `instruction_version` for v4 metadata (query-side only, no doc
re-embed required).

## 6. AMALIA `/vlm` caption probe (one call)

```powershell
# from backend/ (reads LLM_BASE_URL/LLM_API_KEY from .env; sends ONE image):
python - <<'PY'
import asyncio, base64
from pathlib import Path
from openai import AsyncOpenAI
from app.core.config import get_settings

async def main():
    settings = get_settings()
    client = AsyncOpenAI(base_url=settings.llm_openai_base_url_resolved, api_key=settings.LLM_API_KEY or "none")
    image_b64 = base64.b64encode(Path("benchmarks/assets/mnt/images").glob("*.jpg").__next__().read_bytes()).decode()
    response = await client.chat.completions.create(
        model=settings.llm_model_resolved,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text", "text": "Descreve apenas o que é visível nesta peça de museu, em português."},
        ]}],
        max_tokens=200,
    )
    print(response.choices[0].message.content)

asyncio.run(main())
PY
```

If it errors on image content → AMALIA is text-only via this API and Phase 4
captions use local Qwen3-VL-Instruct (plan §22.7).

## 7. Record the verdict

Write the chosen models, dims, instruction decision, and the benchmark numbers
into `benchmark_runs/<run>/` and update plan §22 items 1–3 before Phase 1
(schema) begins — `text_embedding`/`visual_embedding` dimensions are immutable
once `*_v4` indexes are created.
