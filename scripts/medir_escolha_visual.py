"""Mede build_visual_pick_prompt com o modelo de pré-processamento real (CHAT_VISUAL_CARD_CANDIDATES).

  python scripts/medir_escolha_visual.py [--guardar]
Corre da raiz do repositório, com o .env do modelo de pré-processamento. tests/visual_pick_cases.json tem 20
pedidos visuais do MNAA, cada um com as peças que só o ramo das legendas encontrou a 2026-11-19 (título e
aspeto, como vão no prompt) e, em «sim», as que o modelo aceitou nessa data. Não há resposta certa anotada:
o script mostra o que o modelo aceita agora e o que mudou. --guardar grava a escolha atual como referência.
"""
import asyncio, json, os, re, sys, time

sys.path.insert(0, os.getcwd())
import logging  # noqa: E402

logging.disable(logging.CRITICAL)
from app.prompts.chat_prompts import build_visual_pick_prompt  # noqa: E402
from app.schemas.chat import ResponseFormatObject  # noqa: E402
from app.services.llm_service import get_preprocess_llm_service  # noqa: E402

FICHEIRO = os.path.join("tests", "visual_pick_cases.json")


async def principal():
    llm, casos, tempos = get_preprocess_llm_service(), json.load(open(FICHEIRO, encoding="utf-8")), []
    for caso in casos:
        pecas = caso["pecas"]
        prompt = build_visual_pick_prompt(user_message=caso["pedido"], pieces=[(p["titulo"], p["aspeto"]) for p in pecas])
        inicio = time.perf_counter()
        r = await llm.generate(message=prompt, response_format=ResponseFormatObject(type="json_object"),
                               max_tokens=40 + 12 * len(pecas))
        tempos.append((time.perf_counter() - inicio) * 1000)
        sim = []
        for rotulo, veredicto in (r.parsed_json or {}).items():
            n = re.search(r"(\d{1,3})", str(rotulo))
            if n and 1 <= int(n.group(1)) <= len(pecas) and str(veredicto).strip().casefold().startswith("s"):
                sim.append(pecas[int(n.group(1)) - 1]["inventario"])
        antes = caso.get("sim")
        mudou = "" if antes is None or sorted(antes) == sorted(sim) else f"  (antes {len(antes)}: entram {sorted(set(sim) - set(antes))}, saem {sorted(set(antes) - set(sim))})"
        print(f"\n## {caso['pedido']} — {len(sim)} de {len(pecas)}{mudou}")
        for p in pecas:
            print(f"   {'sim' if p['inventario'] in sim else ' · '} {p['inventario']:12} {p['titulo'][:28]:28} | {p['aspeto'][:120]}")
        caso["sim"] = sim
    tempos.sort()
    print(f"\n{len(casos)} pedidos · mediana {tempos[len(tempos) // 2]:.0f} ms · máximo {tempos[-1]:.0f} ms")
    if "--guardar" in sys.argv:
        json.dump(casos, open(FICHEIRO, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("referência gravada em", FICHEIRO)


asyncio.run(principal())
