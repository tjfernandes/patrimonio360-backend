"""E13 — runner do benchmark A/B/C, in-process (retrieval puro; sem LLM).

Replica a sequência de retrieval do chat_service por modo, holding constant o
lexical_query (None) para reprodutibilidade entre modos — mede o EFEITO DA
FUSÃO, não a reescrita LLM. A latência medida é só de retrieval (embedding +
OpenSearch + fusão), separada do LLM remoto.

Uso: python bench_runner.py <off|intent|always> <out.json>
"""
import json, ssl, base64, time, sys, math, asyncio, statistics, urllib.request
from pathlib import Path

BK = Path("/home/hedge6000pro/Projects/AMALIA/patrimonio360_assistant/backend")
sys.path.insert(0, str(BK))
MODE = sys.argv[1]
OUT = sys.argv[2]
OUTPUT_ROOT = Path("/home/hedge6000pro/Projects/AMALIA/raiz_scraper/output")
SLUG = {"mnt": "museu_nacional_do_traje", "mnaz": "museu_nacional_do_azulejo",
        "mnsr": "museu_nacional_soares_dos_reis", "mj": "mosteiro_dos_jeronimos"}
MANIFEST = json.loads((BK / "benchmarks/multimodal/manifest.json").read_text(encoding="utf-8"))

CAND, WINDOW = 15, 150

async def main():
    from app.core.config import Settings
    from app.services.opensearch_client import OpenSearchGateway
    from app.services.embeddings import EmbeddingProvider
    from app.services.retrieval.multimodal_retrieval import MultimodalTextRetrieval
    from app.services.retrieval.visual_intent import decide_visual_intent

    # init kwarg tem prioridade máxima no pydantic-settings -> força o modo.
    settings = Settings(_env_file=str(BK / ".env.v4.local"), MULTIMODAL_RETRIEVAL_MODE=MODE)
    assert settings.MULTIMODAL_RETRIEVAL_MODE == MODE
    gateway = OpenSearchGateway(settings)
    provider = EmbeddingProvider(settings)
    mm = MultimodalTextRetrieval(settings=settings, opensearch_gateway=gateway, embedding_provider=provider)

    def img_path(ref):
        return OUTPUT_ROOT / ref

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
            emb = await provider.embed_multimodal_image_bytes(image_bytes=fh.read(), text=None)
        page = await gateway.search_similar_images_page(
            museum_slug=SLUG[m], museum_id=m, image_embedding=emb,
            from_offset=0, page_size=CAND, retrieval_window_size=WINDOW)
        excl = set(case.get("exclude_image_ids") or [])
        seen, ranked, museums = set(), [], []
        for h in page.results or []:
            if str(h.get("image_id") or "") in excl:
                continue
            aid = str(h.get("artifact_id") or "")
            if aid and aid not in seen:
                seen.add(aid); ranked.append(aid); museums.append(str(h.get("museum_id") or ""))
        ms = (time.perf_counter() - t0) * 1000
        return ranked, ms, {"n": len(ranked), "museums": museums}

    async def run_image_text(case):
        m = case["museum_id"]; text = case["text"]
        t0 = time.perf_counter()
        with open(img_path(case["image_ref"]), "rb") as fh:
            emb = await provider.embed_multimodal_image_bytes(image_bytes=fh.read(), text=None)
        page = await gateway.search_similar_images_page(
            museum_slug=SLUG[m], museum_id=m, image_embedding=emb,
            from_offset=0, page_size=CAND, retrieval_window_size=WINDOW)
        excl = set(case.get("exclude_image_ids") or [])
        i2i_hits = [h for h in (page.results or []) if str(h.get("image_id") or "") not in excl]
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
            return ranked, ms, {"used_text": False, "n": len(ranked), "museums": museums}
        decision = decide_visual_intent(text, mode=MODE)
        outcome = await mm.fuse_image_text_search(
            message_text=text, museum_slug=SLUG[m], museum_id=m, i2i_hits=i2i_hits,
            run_t2i=decision.use_visual, temporal_filter=(case.get("filters") or None),
            router_decision=decision, trace_id=case["query_id"])
        if outcome is None:
            ranked, museums = group(i2i_hits)
        else:
            ranked = [str(d.get("artifact_id") or "") for d in outcome.docs]
            museums = [str(d.get("museum_id") or "") for d in outcome.docs]
        ms = (time.perf_counter() - t0) * 1000
        return ranked, ms, {"used_text": True, "run_t2i": decision.use_visual, "n": len(ranked), "museums": museums}

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
    Path(OUT).write_text(json.dumps({"mode": MODE, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    lat = [r["latency_ms"] for r in results if "latency_ms" in r]
    errs = sum(1 for r in results if "error" in r)
    print(f"[{MODE}] {len(results)} casos, {errs} erros, retrieval p50={statistics.median(lat):.0f}ms "
          f"p95={sorted(lat)[max(0,int(len(lat)*0.95)-1)]:.0f}ms -> {OUT}")

asyncio.run(main())
