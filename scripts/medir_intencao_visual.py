"""Mede build_visual_request_prompt com o modelo de pré-processamento real (CHAT_VISUAL_INTENT_MODEL).

  python scripts/medir_intencao_visual.py [x3]
Corre da raiz do repositório, com o .env do modelo de pré-processamento. Diz, para os casos de
tests/visual_intent_cases.py, quantos pedidos visuais recebem «A» e quantos não visuais ficam de fora,
e o que as regras de hoje (visual_intent.py) já decidiam em cada um.
"""
import asyncio, os, sys, time

sys.path.insert(0, os.getcwd())
import logging  # noqa: E402

logging.disable(logging.CRITICAL)
from app.prompts.query_planner_prompts import build_visual_request_prompt  # noqa: E402
from app.schemas.chat import ResponseFormatObject  # noqa: E402
from app.services.llm_service import get_preprocess_llm_service  # noqa: E402
from app.services.retrieval.visual_intent import decide_visual_intent  # noqa: E402
from tests import visual_intent_cases as casos  # noqa: E402

VEZES = next((int(a[1:]) for a in sys.argv[1:] if a.startswith("x") and a[1:].isdigit()), 1)


async def letra(llm, mensagem, tempos):
    inicio = time.perf_counter()
    r = await llm.generate(message=build_visual_request_prompt(message=mensagem),
                           response_format=ResponseFormatObject(type="json_object"), max_tokens=30)
    tempos.append((time.perf_counter() - inicio) * 1000)
    dados = r.parsed_json
    return str(dados.get("resposta") or "").strip().upper()[:1] if isinstance(dados, dict) else "?"


async def principal():
    llm, tempos = get_preprocess_llm_service(), []
    for nome, lista, quer_a in (("VISUAIS", casos.VISUAIS, True), ("NAO_VISUAIS", casos.NAO_VISUAIS, False)):
        certos = estaveis = pelas_regras = finais = 0
        for mensagem in lista:
            respostas = [await letra(llm, mensagem, tempos) for _ in range(VEZES)]
            regra = decide_visual_intent(mensagem, mode="intent")
            ok = all((r == "A") == quer_a for r in respostas)
            certos += ok
            estaveis += len(set(respostas)) == 1
            pelas_regras += regra.use_visual
            # o que a pesquisa faz: as regras mandam; o modelo só é ouvido quando elas dizem que não
            visual_final = regra.use_visual or all(r == "A" for r in respostas)
            finais += visual_final == quer_a
            if not ok or (regra.use_visual != quer_a):
                print(f"  [{nome}] {'ok ' if ok else 'ERR'} modelo={''.join(respostas)} regras={regra.rule if regra.use_visual else '-'} · {mensagem}")
        print(f"{nome}: modelo certo em {certos}/{len(lista)} · iguais nas {VEZES} execuções {estaveis}/{len(lista)} · "
              f"regras dizem visual em {pelas_regras}/{len(lista)} · regras+modelo certos em {finais}/{len(lista)}")
    tempos.sort()
    print(f"chamadas {len(tempos)} · mediana {tempos[len(tempos)//2]:.0f} ms · máximo {tempos[-1]:.0f} ms")


asyncio.run(principal())
