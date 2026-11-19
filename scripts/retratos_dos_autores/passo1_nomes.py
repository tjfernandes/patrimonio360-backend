"""Passo 1: o Qwen diz o nome da pessoa pela ordem normal e o que o autor é
(pessoa, atribuição, oficina, empresa, desconhecido, várias pessoas). Retomável."""
import concurrent.futures as cf
import sys

from comum import gravar, ler, qwen


def prompt(nome):
    return (
        "Nome de um autor tal como está escrito no catálogo de um museu português:\n"
        f"«{nome}»\n\n"
        "1. Escreve o nome da pessoa pela ordem normal (nome próprio primeiro, apelido no fim), sem datas, sem locais e sem "
        "palavras como «atribuído a» ou «oficina de». Se não houver o nome de uma pessoa, escreve \"\".\n"
        "2. O que é este autor?\n"
        "A) uma pessoa identificada pelo nome\n"
        "B) uma pessoa a quem a obra é atribuída («atribuído a», «atribuição»)\n"
        "C) uma oficina, um círculo, um seguidor, uma escola de uma pessoa, ou uma cópia segundo essa pessoa\n"
        "D) uma fábrica, uma empresa, uma casa comercial ou uma instituição\n"
        "E) desconhecido, anónimo ou sem nome próprio\n"
        "F) várias pessoas\n"
        'Responde só com JSON: {"nome": "...", "tipo": "A" a "F"}'
    )


autores = ler("autores.json")
feito = ler("nomes.json", {})
falta = [a for a in autores if a["key"] not in feito]
if len(sys.argv) > 1:
    falta = falta[: int(sys.argv[1])]


def um(a):
    r = qwen(prompt(a["name"]), max_tokens=80) or {}
    return a["key"], {"catalogo": a["name"], "nome": str(r.get("nome") or "").strip(), "tipo": str(r.get("tipo") or "").strip().upper()[:1]}


with cf.ThreadPoolExecutor(8) as pool:
    for n, (chave, valor) in enumerate(pool.map(um, falta), 1):
        feito[chave] = valor
        if n % 100 == 0:
            gravar("nomes.json", feito)
gravar("nomes.json", feito)
tipos = {}
for v in feito.values():
    tipos[v["tipo"]] = tipos.get(v["tipo"], 0) + 1
print("nomes:", len(feito), "| tipos:", dict(sorted(tipos.items())))
