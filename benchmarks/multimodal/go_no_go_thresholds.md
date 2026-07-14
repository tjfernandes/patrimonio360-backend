# Go/No-Go thresholds — defined BEFORE running the A/B/C benchmark (E13)

Committed a priori (freeze `ecde1d8`, 2026-07-14) so results cannot move the
goalposts. "Materially" is quantified here, not after seeing numbers.

## Hard gates (any failure ⇒ NO-GO, no trade-off)
| Gate | Threshold |
|---|---|
| Museum contamination rate | **exactly 0** in every mode |
| Hydration survival rate | **100%** (every fused artifact_id resolves to a doc) |
| HTTP 5xx | **0** avoidable across all E12/E13 API calls |
| Full embedding vectors in logs | **0** occurrences |
| OFF ≡ baseline | OFF results **byte-identical** to pre-Phase-3 on the documental set |
| image-only self-retrieval sweep | **24/24** rank-1 preserved |

## Quality thresholds (INTENT vs OFF)
| Dimension | Threshold |
|---|---|
| Documental Recall@5 | INTENT loses **≤ 5 pp** vs OFF (per category and global) |
| Documental MRR | INTENT loses **≤ 5 pp** vs OFF |
| Documental where router = TEXT_ONLY | **identical** results OFF vs INTENT (exact) |
| Visual categories | INTENT gains **≥ 5 pp** Recall@5 on known-target visual cases **OR** reduces visual zero-result rate by **≥ 10 pp** |
| Router visual-intent precision | **≥ 0.85** on the labeled PT/EN matrix |
| Router visual-intent recall | **≥ 0.80** on the labeled PT/EN matrix |

## Image+text thresholds
- Both inputs must **observably** change branches/results (removing image or
  text changes the output) in **≥ 80%** of image+text cases.
- Structured filters (temporal) must be honored: **100%** of returned docs
  satisfy the filter (unknown-date docs excluded per `include_unknown=false`).

## System thresholds
- Added **retrieval** p95 (visual branch, excluding VL cold-load and remote LLM)
  ≤ **+300 ms** vs OFF. The ~8–10 s remote-LLM latency is reported separately
  and must NOT mask a retrieval regression.
- `_msearch` vs sequential: **identical** fused ids/ranks/thumbnails; msearch
  must not be slower at p50.

## ALWAYS_ELIGIBLE
Diagnostic only — measures router missed-opportunities (cases INTENT skipped
that ALWAYS improves) and router over-reach (factual cases ALWAYS degrades). It
is explicitly **not** a production recommendation regardless of its numbers.

## Small-dataset honesty
Visual-category ground truth is a proxy (category membership / remapped
targets), not human relevance judgments. Where N is small, absolute counts are
reported, not just rates, and the limitation is stated in the final report.
