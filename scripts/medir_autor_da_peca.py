"""Mede build_focused_author_prompt com o modelo de pré-processamento real (CHAT_FOCUSED_ARTIFACT_AUTHOR).

  python scripts/medir_autor_da_peca.py [x3]
Corre da raiz do repositório, com o .env do modelo de pré-processamento. Cada mensagem de
tests/focused_author_cases.py é perguntada com as três peças de PECAS; conta como certa quando todas as
respostas (peças × execuções) dão «A» nas que pedem as obras do autor, e nenhuma dá «A» nas outras.
"""
import asyncio, os, sys, time

sys.path.insert(0, os.getcwd())
import logging  # noqa: E402

logging.disable(logging.CRITICAL)
from app.core.config import get_settings  # noqa: E402
from app.prompts.query_planner_prompts import build_focused_author_prompt  # noqa: E402
from app.services.author_cards import build_author_directory, other_author_words  # noqa: E402
from app.services.opensearch_client import OpenSearchGateway  # noqa: E402
from app.schemas.chat import ResponseFormatObject  # noqa: E402
from app.services.llm_service import get_preprocess_llm_service  # noqa: E402
from tests import focused_author_cases as casos  # noqa: E402

VEZES = next((int(a[1:]) for a in sys.argv[1:] if a.startswith("x") and a[1:].isdigit()), 1)


async def principal():
    llm, tempos = get_preprocess_llm_service(), []
    # a lista de autores do MNAA, para a salvaguarda «a mensagem diz o nome de outro autor»
    autores = build_author_directory(await OpenSearchGateway(get_settings()).aggregate_creators(museum_slug="mnaa", museum_id="mnaa"))
    for nome, lista, quer_a in (("OBRAS_DO_AUTOR", casos.OBRAS_DO_AUTOR, True), ("SOBRE_A_PECA", casos.SOBRE_A_PECA, False),
                                ("OUTRA_COISA", casos.OUTRA_COISA, False)):
        certas = finais = 0
        for mensagem in lista:
            letras = []
            for titulo, autor in casos.PECAS:
                for _ in range(VEZES):
                    inicio = time.perf_counter()
                    r = await llm.generate(message=build_focused_author_prompt(message=mensagem, title=titulo, author=autor),
                                           response_format=ResponseFormatObject(type="json_object"), max_tokens=30)
                    tempos.append((time.perf_counter() - inicio) * 1000)
                    dados = r.parsed_json
                    letras.append(str(dados.get("resposta") or "?").strip().upper()[:1] if isinstance(dados, dict) else "?")
            ok = all((letra == "A") == quer_a for letra in letras)
            certas += ok
            # o que o serviço faz: com o nome de outro autor na mensagem, o modelo nem é ouvido
            nomeado = [bool(other_author_words(mensagem, autores, except_name=autor)) for _titulo, autor in casos.PECAS for _ in range(VEZES)]
            final_ok = all(((letra == "A") and not n) == quer_a for letra, n in zip(letras, nomeado))
            finais += final_ok
            if not ok or not final_ok:
                print(f"  [{nome}] modelo {'ok ' if ok else 'ERR'} {''.join(letras)} · com a salvaguarda {'ok ' if final_ok else 'ERR'} · {mensagem}")
        print(f"{nome}: modelo {certas}/{len(lista)} · com a salvaguarda {finais}/{len(lista)} "
              f"(cada uma com {len(casos.PECAS)} peças × {VEZES} execuções)")
    tempos.sort()
    print(f"chamadas {len(tempos)} · mediana {tempos[len(tempos) // 2]:.0f} ms · máximo {tempos[-1]:.0f} ms")


asyncio.run(principal())
