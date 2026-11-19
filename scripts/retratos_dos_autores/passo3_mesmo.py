"""Passo 3: o Qwen diz, par a par, se o autor do catálogo é a pessoa da Wikidata.

  python passo3_mesmo.py medir     mede a pergunta em pares de que se sabe a resposta pelas datas,
                                   com as datas do catálogo à vista e escondidas
  python passo3_mesmo.py decidir   decide todos os pares com retrato (retomável) -> decisoes.json
"""
import concurrent.futures as cf
import re
import sys

from comum import gravar, ler, qwen

ROMANOS = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII", 9: "IX", 10: "X", 11: "XI", 12: "XII", 13: "XIII",
           14: "XIV", 15: "XV", 16: "XVI", 17: "XVII", 18: "XVIII", 19: "XIX", 20: "XX", 21: "XXI"}
autores = {a["key"]: a for a in ler("autores.json")}
candidatos = ler("candidatos.json")


def seculos(a):
    lo, hi = a.get("start_century"), a.get("end_century")
    if not lo and not hi:
        return ""
    lo, hi = lo or hi, hi or lo
    if lo <= 0 or hi > 21:
        return ""
    return f"do século {ROMANOS[lo]}" if lo == hi else f"dos séculos {ROMANOS[lo]} a {ROMANOS[hi]}"


def lado_catalogo(a, *, com_datas=True):
    nome = a["name"] if com_datas else re.sub(r"\s+", " ", re.sub(r"[(\[][^)\]]*[)\]]?", " ", a["name"])).strip(" ,")
    linhas = [f"- Nome no catálogo: «{nome}»"]
    if a.get("activity"):
        linhas.append(f"- Atividade: {a['activity']}")
    if com_datas:
        nasc = ", ".join(x for x in (a.get("birth_date"), a.get("birth_place")) if x)
        morte = ", ".join(x for x in (a.get("death_date"), a.get("death_place")) if x)
        if nasc:
            linhas.append(f"- Nascimento: {nasc}")
        if morte:
            linhas.append(f"- Morte: {morte}")
    pecas = f"- Peças dele nos museus: {a['count']}"
    if a.get("categories"):
        pecas += f" ({', '.join(a['categories'])})"
    if seculos(a):
        pecas += f", {seculos(a)}"
    linhas.append(pecas)
    if a.get("titles"):
        linhas.append("- Exemplos de peças: " + "; ".join(t[:70] for t in a["titles"][:4]))
    linhas.append("- Museus: " + ", ".join(a["museums"]))
    if com_datas and a.get("biography"):
        linhas.append(f"- Biografia no catálogo: {a['biography'][:350]}")
    return "\n".join(linhas)


def lado_wikidata(c):
    linhas = [f"- Nome: {c.get('label') or '(sem nome em português)'}"]
    if c.get("description"):
        linhas.append(f"- Descrição: {c['description']}")

    def vida(ano, precisao, lugar):
        partes = []
        if ano is not None:
            partes.append(f"século {ROMANOS.get(abs(ano) // 100 + 1, '?')}" if (precisao or 9) < 9 else str(ano))
        if lugar:
            partes.append(lugar)
        return ", ".join(partes)

    nasc, morte = vida(c.get("birth_year"), c.get("birth_precision"), c.get("birth_place")), vida(c.get("death_year"), c.get("death_precision"), c.get("death_place"))
    if nasc:
        linhas.append(f"- Nascimento: {nasc}")
    if morte:
        linhas.append(f"- Morte: {morte}")
    if c.get("occupations"):
        linhas.append(f"- Profissões: {c['occupations'][:160]}")
    if c.get("country"):
        linhas.append(f"- País: {c['country']}")
    return "\n".join(linhas)


def prompt(a, c, *, com_datas=True):
    return (
        "Um catálogo de museus portugueses tem este autor:\n"
        f"{lado_catalogo(a, com_datas=com_datas)}\n\n"
        "A Wikidata tem esta pessoa:\n"
        f"{lado_wikidata(c)}\n\n"
        "O autor do catálogo é esta pessoa?\n"
        "A) sim, é a mesma pessoa\n"
        "B) não, é outra pessoa com um nome parecido (outra época, outra profissão ou outro nome)\n"
        "C) não dá para saber com estes dados\n"
        "D) não: o autor do catálogo é uma oficina, um círculo, um seguidor ou uma cópia, e não a própria pessoa\n"
        'Responde só com JSON: {"resposta": "A", "B", "C" ou "D"}'
    )


def perguntar(a, c, *, com_datas=True):
    r = qwen(prompt(a, c, com_datas=com_datas), max_tokens=30) or {}
    return str(r.get("resposta") or "").strip().upper()[:1]


def anos_do_catalogo(a):
    """Anos de nascimento e morte: dos campos, ou do que o catálogo escreve entre parênteses no nome."""
    n, m = a.get("birth_year"), a.get("death_year")
    if n is None or m is None:
        achados = [int(x) for x in re.findall(r"(?<!\d)(1[0-9]{3}|20[0-2][0-9])(?!\d)", " ".join(re.findall(r"[(\[]([^)\]]*)", a["name"])))]
        if len(achados) == 2 and achados[0] < achados[1]:
            n, m = n or achados[0], m or achados[1]
    return n, m


if __name__ == "__main__":
    modo = sys.argv[1]
    if modo == "medir":
        positivos, negativos = [], []
        for chave, cands in candidatos.items():
            a = autores[chave]
            n, m = anos_do_catalogo(a)
            if n is None or m is None:
                continue
            for c in cands:
                if c.get("birth_year") is None or c.get("death_year") is None:
                    continue
                dn, dm = abs(c["birth_year"] - n), abs(c["death_year"] - m)
                if dn <= 1 and dm <= 1:
                    positivos.append((a, c))
                elif dn > 10 and dm > 10:
                    negativos.append((a, c))
        print(f"pares com resposta conhecida pelas datas: {len(positivos)} iguais, {len(negativos)} diferentes")
        for rotulo, com_datas in (("datas do catálogo à vista", True), ("datas do catálogo escondidas", False)):
            with cf.ThreadPoolExecutor(8) as pool:
                rp = list(pool.map(lambda par: perguntar(*par, com_datas=com_datas), positivos))
                rn = list(pool.map(lambda par: perguntar(*par, com_datas=com_datas), negativos))
            print(f"== {rotulo}")
            print(f"   iguais: A em {rp.count('A')} de {len(rp)} | respostas: { {x: rp.count(x) for x in sorted(set(rp))} }")
            print(f"   diferentes: A em {rn.count('A')} de {len(rn)} | respostas: { {x: rn.count(x) for x in sorted(set(rn))} }")
            for (a, c), r in zip(negativos, rn):
                if r == "A":
                    print(f"      aceitou mal: {a['name'][:50]} ≠ {c['label']} ({c.get('birth_year')}–{c.get('death_year')}; {c.get('description')})")
            falhou = [(a, c, r) for (a, c), r in zip(positivos, rp) if r != "A"]
            for a, c, r in falhou[:12]:
                print(f"      não aceitou ({r}): {a['name'][:50]} = {c['label']} ({c.get('description')})")
    else:
        decisoes = ler("decisoes.json", {})
        # todas as pessoas candidatas dos autores que têm alguma com retrato: as sem retrato também contam,
        # para se saber quando o Qwen aceita mais de uma pessoa para o mesmo autor
        pares = [(k, c) for k, cands in candidatos.items() if any(x.get("image") for x in cands)
                 for c in cands if f"{k}|{c['qid']}" not in decisoes]
        print("pares por decidir:", len(pares))
        with cf.ThreadPoolExecutor(8) as pool:
            for (k, c), r in zip(pares, pool.map(lambda par: perguntar(autores[par[0]], par[1]), pares)):
                decisoes[f"{k}|{c['qid']}"] = r
        gravar("decisoes.json", decisoes)
        valores = list(decisoes.values())
        print({x: valores.count(x) for x in sorted(set(valores))})
