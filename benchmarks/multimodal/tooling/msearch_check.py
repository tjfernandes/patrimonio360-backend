"""E12.6/E13 — _msearch=true vs false: mesmos ids fundidos (in-process, 1 caso)."""
import json, asyncio, sys
from pathlib import Path
BK = Path("/home/hedge6000pro/Projects/AMALIA/patrimonio360_assistant/backend")
sys.path.insert(0, str(BK))
OUTPUT_ROOT = Path("/home/hedge6000pro/Projects/AMALIA/raiz_scraper/output")

async def run(use_msearch):
    from app.core.config import Settings
    from app.services.opensearch_client import OpenSearchGateway
    from app.services.embeddings import EmbeddingProvider
    from app.services.retrieval.multimodal_retrieval import MultimodalTextRetrieval
    from app.services.retrieval.visual_intent import decide_visual_intent
    s = Settings(_env_file=str(BK / ".env.v4.local"),
                 MULTIMODAL_RETRIEVAL_MODE="intent", MULTIMODAL_USE_MSEARCH=use_msearch)
    gw = OpenSearchGateway(s); pv = EmbeddingProvider(s)
    mm = MultimodalTextRetrieval(settings=s, opensearch_gateway=gw, embedding_provider=pv)
    man = json.loads((BK / "benchmarks/multimodal/manifest.json").read_text(encoding="utf-8"))
    case = next(c for c in man["cases"] if c["type"] == "image_text")
    m = case["museum_id"]
    slug = {"mnt": "museu_nacional_do_traje", "mnaz": "museu_nacional_do_azulejo",
            "mnsr": "museu_nacional_soares_dos_reis", "mj": "mosteiro_dos_jeronimos"}[m]
    with open(OUTPUT_ROOT / case["image_ref"], "rb") as fh:
        emb = await pv.embed_multimodal_image_bytes(image_bytes=fh.read(), text=None)
    page = await gw.search_similar_images_page(museum_slug=slug, museum_id=m, image_embedding=emb,
                                               from_offset=0, page_size=15, retrieval_window_size=150)
    excl = set(case.get("exclude_image_ids") or [])
    i2i = [h for h in (page.results or []) if str(h.get("image_id") or "") not in excl]
    dec = decide_visual_intent(case["text"], mode="intent")
    out = await mm.fuse_image_text_search(message_text=case["text"], museum_slug=slug, museum_id=m,
                                          i2i_hits=i2i, run_t2i=dec.use_visual,
                                          temporal_filter=(case.get("filters") or None),
                                          router_decision=dec, trace_id="ms")
    ids = [str(d.get("artifact_id") or "") for d in out.docs] if out else []
    via = out.diagnostics.get("execution", {}).get("via") if out else None
    return ids, via

async def main():
    a, va = await run(True)
    b, vb = await run(False)
    print(f"msearch=true via={va} ids[:5]={a[:5]}")
    print(f"msearch=false via={vb} ids[:5]={b[:5]}")
    print(f"IDS IGUAIS: {a == b}")

asyncio.run(main())
