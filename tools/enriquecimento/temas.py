"""Temas das peças, lidos da ficha por um modelo e aceites segundo a validação de 2026-11-24.

    python -m tools.enriquecimento.temas --museu mnaa            # pergunta ao modelo o que falta e guarda
    python -m tools.enriquecimento.temas --museu mnaa --limite 20

O modelo de pré-processamento (CHAT_PREPROCESS_LLM_BASE_URL) responde, para cada tema, «sim» ou «não», e em
cada «sim» copia da ficha o excerto que o mostra. A resposta vai presa a um esquema JSON, e um «sim» só conta
se o excerto existir mesmo na ficha. O que o modelo disse fica em `.cache/<museu>-temas.jsonl` (retoma sozinho);
o que entra no índice decide-se em `aceitar`, sem modelo, por isso mudar as regras de aceitação não obriga a
voltar a perguntar.

As regras de aceitação vêm de 200 peças do MNAA conferidas uma a uma pelo Tiago (docs/enriquecimento-por-campos.md):
os temas com pelo menos 95% de valores certos em pelo menos 15 conferidos, mais sete que ficaram abaixo disso e que o
Tiago decidiu aceitar como estão (ver ACEITES_DO_MODELO).

A instrução e as definições abaixo são as que foram conferidas. Mudá-las muda as respostas: a validação deixa
de valer e a cache é refeita (a impressão digital da instrução faz parte de cada linha).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from tools.enriquecimento import vocab as V

CACHE = Path(__file__).resolve().parent / ".cache"

TEMAS: list[tuple[str, str, str]] = [
    ("religioso", "Religioso", "representa figuras, cenas ou símbolos de uma religião (Cristo, a Virgem, santos, anjos, cenas da Bíblia) ou serve o culto (cálice, custódia, paramento, livro de orações). Um animal, uma criança, uma coroa ou uma figura alegórica não bastam"),
    ("mitologia", "Mitologia e alegoria", "representa deuses, heróis ou cenas da mitologia antiga, figuras alegóricas (a Justiça, as Virtudes, as Estações) ou putti"),
    ("retrato", "Retrato", "o assunto é uma pessoa concreta retratada, com nome ou sem ele (retrato, busto, efígie)"),
    ("quotidiano", "Vida quotidiana", "mostra cenas de trabalho, festa, mercado, casa ou costumes de gente comum"),
    ("paisagem", "Paisagem e natureza", "a paisagem é o assunto ou tem grande destaque (campos, montanhas, rios, jardins, céu)"),
    ("animais", "Animais", "tem animais representados, reais ou fantásticos (leão, cão, aves, dragão, cavalo)"),
    ("flora", "Flora e ornamento vegetal", "tem flores, folhagens, frutos, grinaldas, acantos ou outros motivos vegetais, como assunto ou como decoração"),
    ("heraldica", "Heráldica e poder", "tem brasões, armas reais, coroas, insígnias ou outros sinais de um rei, de uma família ou de uma ordem"),
    ("guerra", "Guerra e vida militar", "mostra batalhas, soldados, cavaleiros armados, armas ou cercos"),
    ("mar", "Mar e navegação", "mostra navios, barcos, portos ou cenas no mar"),
    ("arquitetura", "Arquitetura e cidade", "mostra edifícios, vistas de cidades ou interiores de edifícios, ou usa elementos de arquitetura (colunas, arcos, nichos) na decoração"),
    ("oficios", "Ciência e ofícios", "mostra ou é um instrumento científico ou de um ofício, ou representa alguém a exercer um ofício ou uma ciência"),
    ("jogo", "Jogo e infância", "é um brinquedo ou um jogo, ou mostra crianças a brincar"),
    ("escrita", "Escrita e texto", "é um manuscrito, um livro ou um documento, ou tem texto ou uma inscrição como parte essencial"),
    ("geometrico", "Ornamento geométrico", "tem decoração de linhas e formas (caneluras, gomos, losangos, entrelaçados, filetes, estrias)"),
    ("oriente", "Oriente e contactos entre culturas", "foi feita na Ásia ou em África, ou mostra o encontro entre culturas (namban, indo-português, chinoiserie, figuras de outros continentes)"),
]
NOME = {chave: nome for chave, nome, _ in TEMAS}
SISTEMA = ("És conservador de um museu português e classificas peças para um catálogo pesquisável. Usas só o que a ficha diz. "
           "Respondes apenas com o JSON pedido.")
PEDIDO = """Ficha de catálogo da peça:
Título: {titulo}
Categoria: {categoria}
Técnica: {tecnica}
Descrição: {descricao}

Para cada tema, diz se a peça o tem. Responde "sim" só se a ficha o disser; nesse caso copia da ficha, letra por letra, o excerto curto (3 a 12 palavras seguidas) que o mostra. Se não, responde "não" e deixa o excerto vazio. A maior parte das peças tem um a três temas; muitas respostas são "não".
""" + "\n".join(f'- "{chave}" ({nome}): a peça {definicao}.' for chave, nome, definicao in TEMAS)
_ITEM = {"type": "object", "additionalProperties": False, "required": ["presente", "excerto"],
         "properties": {"presente": {"type": "string", "enum": ["sim", "não"]}, "excerto": {"type": "string", "maxLength": 140}}}
ESQUEMA = {"type": "object", "additionalProperties": False, "required": [c for c, _, _ in TEMAS], "properties": {c: _ITEM for c, _, _ in TEMAS}}
IMPRESSAO = hashlib.sha256((SISTEMA + PEDIDO + json.dumps(ESQUEMA, sort_keys=True)).encode("utf-8")).hexdigest()[:16]

# ---- o que a validação de 2026-11-24 deixa entrar (certos / conferidos, nas 200 peças) ----
# pelo modelo, em qualquer peça: Religioso 86/86, Vida quotidiana 37/38, Animais 29/30, Paisagem 19/19, Ciência e ofícios 18/18
ACEITES_DO_MODELO = {"Religioso", "Vida quotidiana", "Animais", "Paisagem e natureza", "Ciência e ofícios"}
# abaixo da fasquia ou com poucos casos, aceites tal como estão por decisão do Tiago (2026-11-24): Arquitetura e cidade
# 30/34, Heráldica e poder 22/25, Guerra e vida militar 9/11, Mitologia e alegoria 8/9, Oriente e contactos entre
# culturas 6/7, Mar e navegação 6/6, Jogo e infância 3/4. Nos dois primeiros conta-se com cerca de 1 errado em cada 8.
ACEITES_DO_MODELO |= {"Arquitetura e cidade", "Heráldica e poder", "Guerra e vida militar", "Mitologia e alegoria",
                      "Oriente e contactos entre culturas", "Mar e navegação", "Jogo e infância"}
# pelo modelo, só fora da categoria Pintura: aí são a decoração da própria peça (Flora 71/72, Geométrico 59/59); numa
# pintura ou num fólio iluminado eram um pormenor da cena ou as linhas da página (Flora 7 certos em 21, Geométrico 3 em 18)
SO_EM_OBJETOS = {"Flora e ornamento vegetal", "Ornamento geométrico"}
# por regra e nunca pelo modelo: «Escrita e texto» só quando a própria peça é um escrito (decisão de 2026-11-24: um santo
# com um livro na mão não é uma peça de escrita; pelo modelo eram 33 certos em 41), e «Retrato» só quando o título o diz
# (os 4 erros em 21 vinham todos do modelo: um rosto entalhado numa cadeira, um busto num aparador)
RETRATO_NO_TITULO = re.compile(r"(?<![a-z])(?:auto-?)?retratos?(?![a-z])")

# as palavras com que um visitante pede cada tema (o backend compara-as com o assunto de uma contagem)
PALAVRAS = {
    "Religioso": ["religioso", "religiosa", "religião", "arte sacra", "sacro", "sacra"],
    "Vida quotidiana": ["vida quotidiana", "quotidiano", "costumes"],
    "Animais": ["animal", "bicho"],
    "Paisagem e natureza": ["paisagem", "natureza"],
    "Ciência e ofícios": ["ciência", "ofício", "científico"],
    "Flora e ornamento vegetal": ["flor", "flora", "floral", "vegetal", "vegetalista", "planta"],
    "Ornamento geométrico": ["geométrico", "geométrica"],
    "Escrita e texto": ["manuscrito", "escrita", "texto"],
    "Retrato": ["retrato"],
    "Arquitetura e cidade": ["arquitetura", "cidade", "edifício"],
    "Heráldica e poder": ["heráldica", "brasão", "armas reais"],
    "Guerra e vida militar": ["guerra", "batalha", "militar", "soldado"],
    "Mitologia e alegoria": ["mitologia", "mitológico", "mitológica", "alegoria", "alegórico", "alegórica"],
    "Oriente e contactos entre culturas": ["oriente", "oriental"],
    "Mar e navegação": ["mar", "navegação", "navio", "barco", "marinha"],
    "Jogo e infância": ["jogo", "brinquedo", "infância"],
}


def texto_da_ficha(ficha: dict[str, Any]) -> str:
    return V.n(" . ".join(str(ficha.get(k) or "") for k in ("title", "category", "technique", "support_or_material", "description")))


def do_modelo(ficha: dict[str, Any], saida: dict[str, Any] | None) -> list[dict[str, str]]:
    """Os «sim» do modelo cujo excerto existe na ficha (sem acentos nem maiúsculas)."""
    if not saida:
        return []
    texto = texto_da_ficha(ficha)
    temas = []
    for chave, resposta in saida.items():
        if chave not in NOME or not isinstance(resposta, dict) or resposta.get("presente") != "sim":
            continue
        excerto = V.n(resposta.get("excerto")).strip(" .,;:«»\"'")
        if len(excerto) >= 6 and excerto in texto:
            temas.append({"valor": NOME[chave], "fonte": "texto", "onde": "descrição, pelo modelo", "excerto": str(resposta.get("excerto") or "")})
    return temas


def aceitar(ficha: dict[str, Any], saida: dict[str, Any] | None, tipo: str | None) -> list[dict[str, str]]:
    """Os temas que vão para o índice."""
    categoria = V.n(ficha.get("category"))
    temas = [t for t in do_modelo(ficha, saida)
             if t["valor"] in ACEITES_DO_MODELO or (t["valor"] in SO_EM_OBJETOS and categoria != "pintura")]
    if (tipo or "").startswith("manuscrito iluminado"):
        temas.append({"valor": "Escrita e texto", "fonte": "regra", "onde": "o tipo de objeto é um manuscrito", "excerto": ""})
    if RETRATO_NO_TITULO.search(V.n(ficha.get("title"))):
        temas.append({"valor": "Retrato", "fonte": "campo", "onde": "título", "excerto": ""})
    return temas


def palavras(temas: list[dict[str, str]]) -> list[str]:
    return [p for t in temas for p in PALAVRAS.get(t["valor"], [])]


def ler_cache(museu: str) -> dict[str, dict[str, Any]]:
    """As respostas guardadas para a instrução atual, por artifact_id."""
    ficheiro = CACHE / f"{museu}-temas.jsonl"
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

    from app.core.config import get_settings
    from tools.enriquecimento.construir import CAMPOS, cliente

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--museu", required=True)
    p.add_argument("--limite", type=int, default=0, help="só as primeiras N peças por perguntar")
    p.add_argument("--concorrencia", type=int, default=3, help="pedidos em paralelo; o mesmo modelo serve produção")
    args = p.parse_args()
    s = get_settings()
    url, modelo = s.CHAT_PREPROCESS_LLM_BASE_URL.rstrip("/"), s.CHAT_PREPROCESS_LLM_MODEL
    fichas = [h["_source"] for h in scan(cliente(), index=s.OPENSEARCH_INDEX_ARTIFACT, size=500,
                                         query={"query": {"term": {"museum_id": args.museu}}, "_source": CAMPOS})]
    feitas = ler_cache(args.museu)
    por_fazer = sorted((f for f in fichas if f["artifact_id"] not in feitas), key=lambda f: str(f["artifact_id"]))
    if args.limite:
        por_fazer = por_fazer[: args.limite]
    print(f"{len(fichas)} peças de «{args.museu}» · {len(feitas)} já respondidas · {len(por_fazer)} a perguntar a {modelo} · instrução {IMPRESSAO}")
    if not por_fazer:
        return

    def perguntar(ficha: dict[str, Any]) -> dict[str, Any]:
        texto = PEDIDO.format(titulo=ficha.get("title") or "(sem título)", categoria=ficha.get("category") or "—",
                              tecnica=ficha.get("technique") or "—", descricao=str(ficha.get("description") or "—")[:1800])
        erro = ""
        for _ in range(2):
            try:
                resposta = httpx.post(url + "/chat/completions", timeout=180, json={
                    "model": modelo, "temperature": 0.0, "max_tokens": 900,
                    "messages": [{"role": "system", "content": SISTEMA}, {"role": "user", "content": texto}],
                    "response_format": {"type": "json_schema", "json_schema": {"name": "temas", "schema": ESQUEMA, "strict": True}}})
                return {"artifact_id": ficha["artifact_id"], "instrucao": IMPRESSAO, "modelo": modelo,
                        "saida": json.loads(resposta.json()["choices"][0]["message"]["content"])}
            except Exception as exc:  # noqa: BLE001 — fica por fazer; a próxima corrida volta a tentar
                erro = f"{type(exc).__name__}: {exc}"
        return {"artifact_id": ficha["artifact_id"], "instrucao": IMPRESSAO, "modelo": modelo, "erro": erro}

    CACHE.mkdir(exist_ok=True)
    inicio, certas, falhadas = time.time(), 0, 0
    with open(CACHE / f"{args.museu}-temas.jsonl", "a", encoding="utf-8") as saida, ThreadPoolExecutor(max(args.concorrencia, 1)) as executor:
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
