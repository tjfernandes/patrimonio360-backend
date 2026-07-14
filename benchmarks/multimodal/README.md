# Multimodal retrieval benchmark (Phase 3)

Versioned, reproducible A/B/C evaluation of the multimodal retrieval flow.
Heavy outputs (per-run JSON, logs) stay OUT of git; only the small reproducible
artifacts live here.

## Files
- `freeze_record.md` — frozen implementation under evaluation (commit, models,
  revisions, indices, config, timestamp).
- `go_no_go_thresholds.md` — go/no-go thresholds, committed BEFORE running.
- `manifest.json` — frozen evaluation dataset (query_id, museum, category,
  type, text/image_ref, relevant targets, filters, provenance). No heavy files.
- `router_matrix.json` — labeled PT/EN visual-intent matrix (E12.7 ground truth).
- `split.json` — calibration/holdout split (stratified by type+museum, seeded).
- `report.md` — human-readable summary of the last frozen A/B/C run.

## Reproducing (isolated backend only; never the live backend)
The runner is `p3_eval/bench_runner.py` (kept in the eval scratch, not shipped):
it replicates the chat-service retrieval sequence per mode in-process against
the physical v4 indices, holding `lexical_query=None` constant across modes so
the measurement isolates the fusion effect (not the non-deterministic LLM
rewrite). Retrieval latency is embedding + OpenSearch + fusion only; the remote
LLM answer latency is measured separately via the API integration tests and is
never allowed to mask a retrieval regression.

## Modes
- OFF: baseline, no text_to_image.
- INTENT: text_to_image only when the deterministic router returns
  TEXT_AND_VISUAL.
- ALWAYS_ELIGIBLE: text_to_image on every retrieval-eligible message
  (diagnostic; greetings/admin/llm_only are still not turned into searches).

## Limitations
Visual-category ground truth is a proxy (remapped single targets / category
membership), not human relevance judgments. N is small per category; absolute
counts are reported alongside rates. The holdout split is reserved for a future
tuning round — the first A/B/C comparison uses frozen weights/floor and
`MULTIMODAL_IN_TOUR_MARGIN=0`, with no tuning, so it is reported on all cases.
