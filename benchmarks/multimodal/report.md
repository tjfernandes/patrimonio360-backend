# Multimodal A/B/C benchmark — results (frozen run)

Frozen impl `ecde1d8` + msearch fix `1d8fda4`. Dataset `manifest.json` (32 cases:
8 documental, 4 visual-with-target, 14 image_only, 6 image_text; 4 museums).
Retrieval measured in-process against the physical v4 indices (LLM excluded;
`lexical_query=None` held constant across modes). Thresholds in
`go_no_go_thresholds.md` were committed BEFORE this run.

## Hard gates — ALL PASS
| Gate | Result |
|---|---|
| Museum contamination (bench + E12, all modes) | **0** |
| Hydration survival (0 runner errors, 0 unresolved fused ids) | **100%** |
| HTTP 5xx across 60 E12 API calls (3 modes × 20) | **0** |
| Full embedding vectors in logs | **0** |
| OFF ≡ baseline on text-only cases (off vs intent) | **identical** (0 diffs) |
| image-only self-retrieval sweep (24 images, official metric) | **24/24 rank-1** |

## Retrieval quality (per category)
| Metric | OFF | INTENT | ALWAYS |
|---|---|---|---|
| Documental R@5 (n=8) | 0.625 | **0.625** | 0.625 |
| Documental R@1 | 0.625 | **0.625** | 0.500 |
| Documental MRR | 0.636 | **0.636** | 0.574 |
| Visual R@5 (n=4) | 0.750 | 0.750 | 0.750 |
| Visual zero-result rate | 0.0 | 0.0 | 0.0 |
| image_only R@1 (n=14; incl. single-image) | 0.50 | 0.50 | 0.50 |
| image_only R@1 (multi-image only, n=10) | 0.70 | 0.70 | 0.70 |
| image_text R@5 (n=6, see note) | 0.33 | 0.00* | 0.00* |

INTENT − OFF deltas: documental R@5 **0.0 pp**, MRR **0.0 pp**; visual R@5 **0.0 pp**.
Retrieval p95 **+78 ms** (429→507 ms) — well under the +300 ms budget. p50 152→178 ms.

\* **Not a regression.** image_text targets = the base image's own artifact. The
two temporal cases ("apenas do século XVIII") correctly exclude their own
target because its years are `(None,None)` / `(2008,None)` — out of 1701-1800.
Temporal compliance = **100% of returned docs in-window**. The four visual
refinements ("com decoração floral"/"mais azul") keep the i2i target at the
**same rank** off vs intent (6→6, 8→8, None→None). So image+text behaves
correctly; the metric is confounded by the self-artifact ground truth.

## Router (E12.7, labeled PT/EN matrix, n=38)
precision **1.000** · recall **0.905** · FP-rate **0.000** · FN-rate 0.095.
Two false negatives ("retratos com figuras humanas", "composição simétrica") —
vocabulary gaps, logged for controlled tuning (not patched pre-benchmark).

## Image+text (E8)
- Both inputs observably change results in **3/6** cases (50%). The other 3 are
  visual refinements on already-strong i2i results where i2i (weight 1.0) +
  score-floor 0.64 legitimately don't reorder the top-5.
- Structured temporal filter honored: **100%** in-window; out-of-window targets
  correctly dropped (`include_unknown=false`).

## System
- `_msearch` vs sequential: **identical fused ids**. After fix `1d8fda4` the
  image+text pair runs sequential by design (its artifact branch needs the
  hybrid search pipeline, which OpenSearch _msearch cannot carry per-request).
- No VL load on the text path in OFF (0 multimodal events); VL loads once for
  the image endpoint (i2i) regardless of mode, as expected.

## ALWAYS_ELIGIBLE (diagnostic only)
Degrades documental R@1 0.625→0.500 and MRR −6.2 pp (router over-reach on
factual queries). Confirms the gated INTENT is preferable; ALWAYS is not a
production candidate.

## Limitations
Small proxy dataset: visual cases have single text-findable targets (so t2i
shows 0 pp — its value for pure-visual queries without text-findable targets is
qualitative here, not measured); image_text ground truth is the self-artifact
(confounds filter/refinement semantics). A human-judged visual-relevance set
and refinement-aware image+text targets are needed to quantify the visual
BENEFIT. Holdout (16 cases) untouched — no tuning was done this pass.

## Recommendation: GO CONDICIONADO (isolated INTENT only)
The **safety** case for INTENT is fully proven: zero documental degradation,
all hard gates green, router precision 1.0, correct structured filters. The
**benefit** case (visual/image+text gains) is not yet quantified on this weak
proxy dataset. Recommend: activate INTENT on the ISOLATED backend for a soak
with real queries while building a human-judged visual dataset; keep
`MULTIMODAL_IN_TOUR_MARGIN=0`; do NOT recommend ALWAYS. Re-evaluate weights/floor
only on calibration, with a final holdout pass, once the dataset is stronger.
