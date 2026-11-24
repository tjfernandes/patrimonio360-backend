"""O que se vê na fotografia de cada peça: se tem pessoas, animais, letras ou inscrições.

    python -m tools.enriquecimento.fotografia --museu mnaa            # pergunta ao modelo de visão o que falta e guarda
    python -m tools.enriquecimento.fotografia --museu mnaa --limite 20

O modelo de visão (POINT_VERIFY_BASE_URL, o mesmo servidor do «apontar») vê até duas fotografias da peça, com o
título e a categoria, e responde preso a um esquema JSON. As respostas ficam em `.cache/<museu>-fotografia.jsonl`
(retoma sozinho). Só três respostas vão para o índice, as que a validação de 2026-11-24 confirmou: nas 200 peças
conferidas pelo Tiago, «tem pessoas», «tem animais» e «tem letras ou inscrição» estavam certas em 599 de 600.

A instrução pede mais do que isso (cores, motivos, temas) porque é a que foi conferida: tirar-lhe campos mudaria
as respostas dos que ficam. As cores principais (468 certas em 486) falham nas fotografias a preto e branco; os
motivos e os temas vistos na imagem saíram maus e não se usam.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

CACHE = Path(__file__).resolve().parent / ".cache"

CORES = ["branco", "preto", "azul", "verde", "amarelo", "vermelho", "castanho", "cinzento", "rosa", "roxo", "laranja", "dourado", "prateado",
         "bege", "creme", "ocre", "manganês", "azul de cobalto", "vinoso", "turquesa", "sépia", "marfim", "âmbar", "carmim", "escarlate",
         "esverdeado", "avermelhado", "azulado", "amarelado", "acastanhado", "acinzentado", "polícromo", "transparente"]
MOTIVOS = ["vegetalista", "floral", "frutos", "geométrico", "heráldico", "zoomórfico", "antropomórfico", "figuras religiosas",
           "figuras mitológicas", "paisagem", "arquitetura", "epigráfico (letras e inscrições)", "grotescos e mascarões", "volutas e concheados",
           "acantos", "putti e anjos", "aves", "cenas narrativas", "chinoiserie", "estrelas e astros", "listas e riscas", "liso (sem decoração)"]
TEMAS = ["Religioso", "Mitologia e alegoria", "Retrato", "Vida quotidiana", "Paisagem e natureza", "Animais",
         "Flora e ornamento vegetal", "Heráldica e poder", "Guerra e vida militar", "Mar e navegação", "Arquitetura e cidade",
         "Ciência e ofícios", "Jogo e infância", "Escrita e texto", "Ornamento geométrico", "Oriente e contactos entre culturas"]
_TRES = ["sim", "não", "não se sabe"]
ESQUEMA = {"type": "object", "additionalProperties": False,
           "required": ["cores_principais", "cores", "motivos", "temas", "tem_figura_humana", "tem_animais", "tem_inscricao"],
           "properties": {
               "cores_principais": {"type": "array", "minItems": 1, "maxItems": 3, "items": {"type": "string", "enum": CORES}},
               "cores": {"type": "array", "maxItems": 8, "items": {"type": "string", "enum": CORES}},
               "motivos": {"type": "array", "maxItems": 6, "items": {"type": "string", "enum": MOTIVOS}},
               "temas": {"type": "array", "maxItems": 4, "items": {"type": "string", "enum": TEMAS}},
               "tem_figura_humana": {"type": "string", "enum": _TRES}, "tem_animais": {"type": "string", "enum": _TRES},
               "tem_inscricao": {"type": "string", "enum": _TRES}}}
SISTEMA = ("És conservador de um museu português e classificas peças para um catálogo pesquisável. Usas só o que te é dado: "
           "não acrescentas autores, datas nem interpretações que não estejam lá. Respondes apenas com o JSON pedido.")
PEDIDO = """As fotografias mostram uma peça de museu (título: «{titulo}»; categoria: {categoria}). Olha só para a peça, não para o fundo nem para o suporte em que está.
Escolhe sempre das listas dadas.
- "cores_principais": uma a três cores que dominam a peça.
- "cores": todas as cores que se veem na peça.
- "motivos": os motivos decorativos ou representados que se veem claramente. «liso (sem decoração)» só se não houver nenhum.
- "temas": os temas que se reconhecem na peça só de a ver. Lista vazia se não for claro.
- "tem_figura_humana", "tem_animais", "tem_inscricao": se na peça se veem pessoas, animais, ou letras e inscrições."""
LADO_MAXIMO, FOTOGRAFIAS = 1024, 2
IMPRESSAO = hashlib.sha256((SISTEMA + PEDIDO + json.dumps(ESQUEMA, sort_keys=True) + f"{LADO_MAXIMO}/{FOTOGRAFIAS}").encode("utf-8")).hexdigest()[:16]

# resposta do modelo -> valor no índice, e as palavras com que um visitante o pede
VISTO = {"tem_figura_humana": "pessoas", "tem_animais": "animais", "tem_inscricao": "letras"}
PALAVRAS = {"pessoas": ["pessoa", "figura humana", "gente"], "animais": ["animal", "bicho"], "letras": ["inscrição", "letra", "legenda"]}


def aceitar(saida: dict[str, Any] | None) -> list[str]:
    """O que se vê na peça, só pelos três campos confirmados."""
    return [valor for chave, valor in VISTO.items() if (saida or {}).get(chave) == "sim"]


def palavras(vistos: list[str]) -> list[str]:
    return [p for v in vistos for p in PALAVRAS.get(v, [])]


def ler_cache(museu: str) -> dict[str, dict[str, Any]]:
    ficheiro = CACHE / f"{museu}-fotografia.jsonl"
    respostas: dict[str, dict[str, Any]] = {}
    if ficheiro.is_file():
        for linha in open(ficheiro, encoding="utf-8"):
            registo = json.loads(linha)
            if registo.get("instrucao") == IMPRESSAO and registo.get("saida"):
                respostas[registo["artifact_id"]] = registo["saida"]
    return respostas


def main() -> None:
    import httpx
    from opensearchpy.helpers import scan
    from PIL import Image

    from app.core.config import get_settings
    from tools.enriquecimento.construir import cliente

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--museu", required=True)
    p.add_argument("--limite", type=int, default=0)
    p.add_argument("--concorrencia", type=int, default=3, help="pedidos em paralelo; o mesmo modelo serve produção")
    args = p.parse_args()
    s = get_settings()
    url = s.POINT_VERIFY_BASE_URL.rstrip("/")
    modelo = httpx.get(url + "/models", timeout=10).json()["data"][0]["id"]
    raiz = Path(s.IMAGE_ASSET_ROOT or "Images")
    raiz = raiz if raiz.is_absolute() else Path(__file__).resolve().parents[2] / raiz
    # IMAGE_ASSET_ROOT aponta para a pasta «Images», e os caminhos das fichas já começam por «Images/»
    base = raiz.parent if raiz.name == "Images" else raiz
    fichas = [h["_source"] for h in scan(cliente(), index=s.OPENSEARCH_INDEX_ARTIFACT, size=500, query={
        "query": {"term": {"museum_id": args.museu}}, "_source": ["artifact_id", "title", "category", "image_paths"]})]
    feitas = ler_cache(args.museu)

    def fotos(ficha: dict[str, Any]) -> list[Path]:
        return [base / c for c in (ficha.get("image_paths") or []) if (base / c).is_file()][:FOTOGRAFIAS]

    por_fazer = sorted((f for f in fichas if f["artifact_id"] not in feitas and fotos(f)), key=lambda f: str(f["artifact_id"]))
    sem_foto = sum(1 for f in fichas if not fotos(f))
    if args.limite:
        por_fazer = por_fazer[: args.limite]
    print(f"{len(fichas)} peças de «{args.museu}» · {sem_foto} sem fotografia em {base} · {len(feitas)} já respondidas · "
          f"{len(por_fazer)} a perguntar a {modelo} · instrução {IMPRESSAO}")
    if not por_fazer:
        return

    def imagem(caminho: Path) -> str:
        im = Image.open(caminho).convert("RGB")
        im.thumbnail((LADO_MAXIMO, LADO_MAXIMO))
        saida = io.BytesIO()
        im.save(saida, "JPEG", quality=88)
        return "data:image/jpeg;base64," + base64.b64encode(saida.getvalue()).decode()

    def perguntar(ficha: dict[str, Any]) -> dict[str, Any]:
        erro = ""
        try:
            conteudo: list[dict[str, Any]] = [{"type": "image_url", "image_url": {"url": imagem(c)}} for c in fotos(ficha)]
            conteudo.append({"type": "text", "text": PEDIDO.format(titulo=ficha.get("title") or "sem título", categoria=ficha.get("category") or "—")})
        except Exception as exc:  # noqa: BLE001 — imagem ilegível
            return {"artifact_id": ficha["artifact_id"], "erro": f"{type(exc).__name__}: {exc}"}
        for _ in range(2):
            try:
                resposta = httpx.post(url + "/chat/completions", timeout=180, json={
                    "model": modelo, "temperature": 0.0, "max_tokens": 600,
                    "messages": [{"role": "system", "content": SISTEMA}, {"role": "user", "content": conteudo}],
                    "response_format": {"type": "json_schema", "json_schema": {"name": "classificacao", "schema": ESQUEMA, "strict": True}}})
                return {"artifact_id": ficha["artifact_id"], "instrucao": IMPRESSAO, "modelo": modelo,
                        "saida": json.loads(resposta.json()["choices"][0]["message"]["content"])}
            except Exception as exc:  # noqa: BLE001 — fica por fazer; a próxima corrida volta a tentar
                erro = f"{type(exc).__name__}: {exc}"
        return {"artifact_id": ficha["artifact_id"], "erro": erro}

    CACHE.mkdir(exist_ok=True)
    inicio, certas, falhadas = time.time(), 0, 0
    with open(CACHE / f"{args.museu}-fotografia.jsonl", "a", encoding="utf-8") as saida, ThreadPoolExecutor(max(args.concorrencia, 1)) as executor:
        for numero, registo in enumerate(executor.map(perguntar, por_fazer), 1):
            if "saida" in registo:
                saida.write(json.dumps(registo, ensure_ascii=False) + "\n")
                saida.flush()
                certas += 1
            else:
                falhadas += 1
            if numero % 200 == 0:
                print(f"  {numero} de {len(por_fazer)} · {time.time() - inicio:.0f} s", flush=True)
    print(f"respondidas {certas} · falhadas {falhadas} · {time.time() - inicio:.0f} s")


if __name__ == "__main__":
    main()
