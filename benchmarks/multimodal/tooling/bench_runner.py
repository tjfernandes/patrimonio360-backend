"""E13/E14 — runner do benchmark multimodal, in-process (retrieval puro; sem LLM).

Replica a sequência de retrieval do chat_service por configuração, holding
constant o lexical_query (None) para reprodutibilidade — mede o EFEITO DA
FUSÃO/RERANK, não a reescrita LLM. A latência medida é só de retrieval
(embedding + OpenSearch + fusão + rerank), separada do LLM remoto.

Configurações (selecionáveis SEM editar código):
  off|intent|always -> legado E13: ramos independentes + RRF, sem rerank VL.
  A -> ramos existentes + RRF, sem rerank VL (= intent explícito).
  B -> A + rerank VL local no pool i2i (pré-fusão).
  C -> A + ramo conjunto imagem+texto (joint_image_text) na fusão.
  D -> C + rerank VL GLOBAL no pool fundido (recomendado).
  E -> C + rerank VL local E global (ablação).

Uso: python bench_runner.py <off|intent|always|A|B|C|D|E> <out.json>
"""
import json, ssl, base64, time, sys, math, asyncio, statistics, urllib.request
from pathlib import Path

BK = Path("/home/hedge6000pro/Projects/AMALIA/patrimonio360_assistant/backend")
sys.path.insert(0, str(BK))
CONFIG_NAME = sys.argv[1]
OUT = sys.argv[2]
OUTPUT_ROOT = Path("/home/hedge6000pro/Projects/AMALIA/raiz_scraper/output")
SLUG = {"mnt": "museu_nacional_do_traje", "mnaz": "museu_nacional_do_azulejo",
        "mnsr": "museu_nacional_soares_dos_reis", "mj": "mosteiro_dos_jeronimos"}
MANIFEST = json.loads((BK / "benchmarks/multimodal/manifest.json").read_text(encoding="utf-8"))

CAND, WINDOW = 15, 150

# Cada configuração é um conjunto de overrides de Settings (init kwargs têm
# prioridade máxima no pydantic-settings) — nada de editar código-fonte.
_NO_RERANK = {"CHAT_ENABLE_VL_RERANKING": False,
              "MULTIMODAL_JOINT_IMAGE_TEXT_ENABLED": False}
CONFIGS = {
    "off":    {"MULTIMODAL_RETRIEVAL_MODE": "off", **_NO_RERANK},
    "intent": {"MULTIMODAL_RETRIEVAL_MODE": "intent", **_NO_RERANK},
    "always": {"MULTIMODAL_RETRIEVAL_MODE": "always", **_NO_RERANK},
    "A": {"MULTIMODAL_RETRIEVAL_MODE": "intent", **_NO_RERANK},
    "B": {"MULTIMODAL_RETRIEVAL_MODE": "intent", "CHAT_ENABLE_VL_RERANKING": True,
          "MULTIMODAL_VL_RERANK_STAGE": "local",
          "MULTIMODAL_JOINT_IMAGE_TEXT_ENABLED": False},
    "C": {"MULTIMODAL_RETRIEVAL_MODE": "intent", "CHAT_ENABLE_VL_RERANKING": False,
          "MULTIMODAL_JOINT_IMAGE_TEXT_ENABLED": True},
    "D": {"MULTIMODAL_RETRIEVAL_MODE": "intent", "CHAT_ENABLE_VL_RERANKING": True,
          "MULTIMODAL_VL_RERANK_STAGE": "global",
          "MULTIMODAL_JOINT_IMAGE_TEXT_ENABLED": True},
    "E": {"MULTIMODAL_RETRIEVAL_MODE": "intent", "CHAT_ENABLE_VL_RERANKING": True,
          "MULTIMODAL_VL_RERANK_STAGE": "both",
          "MULTIMODAL_JOINT_IMAGE_TEXT_ENABLED": True},
}

async def main():
    from app.core.config import Settings
    from app.services.opensearch_client import OpenSearchGateway
    from app.services.embeddings import EmbeddingProvider
    from app.services.retrieval.multimodal_retrieval import (
        MultimodalTextRetrieval, apply_global_vl_rerank,
    )
    from app.services.retrieval.visual_intent import decide_visual_intent

    overrides = CONFIGS[CONFIG_NAME]
    settings = Settings(_env_file=str(BK / ".env.v4.local"), **overrides)
    MODE = settings.MULTIMODAL_RETRIEVAL_MODE
    gateway = OpenSearchGateway(settings)
    provider = EmbeddingProvider(settings)
    mm = MultimodalTextRetrieval(settings=settings, opensearch_gateway=gateway, embedding_provider=provider)

    vl_enabled = bool(settings.CHAT_ENABLE_VL_RERANKING)
    vl_stage = settings.MULTIMODAL_VL_RERANK_STAGE if vl_enabled else "off"
    joint_enabled = bool(settings.MULTIMODAL_JOINT_IMAGE_TEXT_ENABLED)
    vl_reranker = None
    if vl_enabled:
        from app.services.reranker import VLRerankerService
        vl_reranker = VLRerankerService(settings)

    flags = {"mode": MODE, "vl_rerank_stage": vl_stage, "joint_enabled": joint_enabled,
             "joint_top_k": int(settings.MULTIMODAL_JOINT_IMAGE_TEXT_TOP_K),
             "joint_weight": float(settings.MULTIMODAL_JOINT_IMAGE_TEXT_WEIGHT),
             "global_rerank_top_n": int(settings.MULTIMODAL_GLOBAL_VL_RERANK_TOP_N)}
    print(f"[{CONFIG_NAME}] flags: {flags}")

    def img_path(ref):
        return OUTPUT_ROOT / ref

    async def local_rerank(hits, image_bytes, text):
        t0 = time.perf_counter()
        out = await vl_reranker.rerank_image_hits(
            query_text=text or "", query_image_bytes=image_bytes,
            image_hits=hits, top_k=CAND)
        return out, (time.perf_counter() - t0) * 1000

    async def run_text(case):
        text = case["text"]; m = case["museum_id"]
        t0 = time.perf_counter()
        emb = await provider.embed_text(text)
        page = await gateway.search_relevant_context_page(
            museum_slug=SLUG[m], museum_id=m, query_text=text, lexical_query=None,
            query_embedding=emb, from_offset=0, page_size=CAND, retrieval_window_size=WINDOW)
        baseline = list(page.results or [])
        used_t2i = False
        docs = baseline
        if MODE != "off":
            decision = decide_visual_intent(text, mode=MODE)
            if decision.use_visual:
                used_t2i = True
                outcome = await mm.fuse_text_search(
                    query_text=text, museum_slug=SLUG[m], museum_id=m,
                    artifact_docs=baseline, router_decision=decision, trace_id=case["query_id"])
                docs = outcome.docs if outcome is not None else baseline
        ms = (time.perf_counter() - t0) * 1000
        ranked = [str(d.get("artifact_id") or "") for d in docs]
        museums = [str(d.get("museum_id") or "") for d in docs]
        return ranked, ms, {"used_t2i": used_t2i, "n": len(ranked), "museums": museums}

    async def run_image_only(case):
        m = case["museum_id"]
        t0 = time.perf_counter()
        with open(img_path(case["image_ref"]), "rb") as fh:
            image_bytes = fh.read()
        emb = await provider.embed_multimodal_image_bytes(image_bytes=image_bytes, text=None)
        page = await gateway.search_similar_images_page(
            museum_slug=SLUG[m], museum_id=m, image_embedding=emb,
            from_offset=0, page_size=CAND, retrieval_window_size=WINDOW)
        excl = set(case.get("exclude_image_ids") or [])
        hits = [h for h in (page.results or []) if str(h.get("image_id") or "") not in excl]
        rerank_ms = None
        # imagem-only: o pool i2i é o pool inteiro -> passe único em qualquer
        # stage != off (espelha o chat_service).
        if vl_stage != "off":
            hits, rerank_ms = await local_rerank(hits, image_bytes, "")
        seen, ranked, museums = set(), [], []
        for h in hits:
            aid = str(h.get("artifact_id") or "")
            if aid and aid not in seen:
                seen.add(aid); ranked.append(aid); museums.append(str(h.get("museum_id") or ""))
        ms = (time.perf_counter() - t0) * 1000
        meta = {"n": len(ranked), "museums": museums}
        if rerank_ms is not None:
            meta["local_rerank_ms"] = round(rerank_ms, 1)
        return ranked, ms, meta

    async def run_image_text(case):
        m = case["museum_id"]; text = case["text"]
        t0 = time.perf_counter()
        with open(img_path(case["image_ref"]), "rb") as fh:
            image_bytes = fh.read()
        emb = await provider.embed_multimodal_image_bytes(image_bytes=image_bytes, text=None)
        page = await gateway.search_similar_images_page(
            museum_slug=SLUG[m], museum_id=m, image_embedding=emb,
            from_offset=0, page_size=CAND, retrieval_window_size=WINDOW)
        excl = set(case.get("exclude_image_ids") or [])
        i2i_hits = [h for h in (page.results or []) if str(h.get("image_id") or "") not in excl]
        meta = {"used_text": MODE != "off"}
        def group(hits):
            seen, ranked, museums = set(), [], []
            for h in hits:
                aid = str(h.get("artifact_id") or "")
                if aid and aid not in seen:
                    seen.add(aid); ranked.append(aid); museums.append(str(h.get("museum_id") or ""))
            return ranked, museums
        if MODE == "off":
            ranked, museums = group(i2i_hits)
            ms = (time.perf_counter() - t0) * 1000
            return ranked, ms, {**meta, "n": len(ranked), "museums": museums}
        # passe local pré-fusão (stage local/both), como no chat_service
        if vl_stage in ("local", "both"):
            i2i_hits, rerank_ms = await local_rerank(i2i_hits, image_bytes, text)
            meta["local_rerank_ms"] = round(rerank_ms, 1)
        decision = decide_visual_intent(text, mode=MODE)
        outcome = await mm.fuse_image_text_search(
            message_text=text, museum_slug=SLUG[m], museum_id=m, i2i_hits=i2i_hits,
            run_t2i=decision.use_visual, temporal_filter=(case.get("filters") or None),
            router_decision=decision, trace_id=case["query_id"],
            query_image_bytes=image_bytes)
        if outcome is not None and vl_stage in ("global", "both"):
            outcome = await apply_global_vl_rerank(
                outcome, vl_reranker=vl_reranker, query_text=text,
                query_image_bytes=image_bytes,
                top_n=int(settings.MULTIMODAL_GLOBAL_VL_RERANK_TOP_N),
                trace_id=case["query_id"])
        elif outcome is None and vl_stage == "global" and i2i_hits:
            i2i_hits, rerank_ms = await local_rerank(i2i_hits, image_bytes, text)
            meta["local_rerank_ms"] = round(rerank_ms, 1)
        if outcome is None:
            ranked, museums = group(i2i_hits)
        else:
            ranked = [str(d.get("artifact_id") or "") for d in outcome.docs]
            museums = [str(d.get("museum_id") or "") for d in outcome.docs]
            fusion = outcome.diagnostics.get("fusion", {})
            meta["branch_sizes"] = fusion.get("branch_sizes")
            joint_diag = outcome.diagnostics.get("joint_image_text") or {}
            if "latency_ms" in joint_diag:
                meta["joint_ms"] = joint_diag["latency_ms"]
            if "hits" in joint_diag:
                meta["joint_hits"] = joint_diag["hits"]
            grr = outcome.diagnostics.get("global_vl_rerank") or {}
            if grr:
                meta["global_rerank_ms"] = grr.get("latency_ms")
                meta["global_rerank_changed"] = grr.get("positions_changed")
                if "fallback" in grr:
                    meta["global_rerank_fallback"] = grr["fallback"]
        ms = (time.perf_counter() - t0) * 1000
        return ranked, ms, {**meta, "run_t2i": decision.use_visual, "n": len(ranked), "museums": museums}

    results = []
    for case in MANIFEST["cases"]:
        try:
            if case["type"] in ("text_documental", "text_visual"):
                ranked, ms, meta = await run_text(case)
            elif case["type"] == "image_only":
                ranked, ms, meta = await run_image_only(case)
            else:
                ranked, ms, meta = await run_image_text(case)
        except Exception as exc:
            results.append({"query_id": case["query_id"], "type": case["type"],
                            "museum": case["museum_id"], "error": str(exc)[:200]})
            continue
        relevant = set(case["relevant"])
        rank = next((i + 1 for i, a in enumerate(ranked) if a in relevant), None)
        museums = meta.pop("museums", None)
        contamination = (
            any(mu and mu != case["museum_id"] for mu in museums) if museums else None
        )
        results.append({
            "query_id": case["query_id"], "type": case["type"], "museum": case["museum_id"],
            "category": case["category"], "ranked_top10": ranked[:10], "target_rank": rank,
            "n": len(ranked), "dup": len(ranked) != len(set(ranked)),
            "contamination": contamination, "latency_ms": round(ms, 1), **meta,
        })
    Path(OUT).write_text(json.dumps({"mode": MODE, "config": CONFIG_NAME, "flags": flags,
                                     "results": results}, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    lat = [r["latency_ms"] for r in results if "latency_ms" in r]
    errs = sum(1 for r in results if "error" in r)
    print(f"[{CONFIG_NAME}] {len(results)} casos, {errs} erros, retrieval p50={statistics.median(lat):.0f}ms "
          f"p95={sorted(lat)[max(0,int(len(lat)*0.95)-1)]:.0f}ms -> {OUT}")

asyncio.run(main())
