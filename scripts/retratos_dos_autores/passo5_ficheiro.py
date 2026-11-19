"""Passo 5: junta tudo. Escreve a tabela de revisão, as folhas de contacto e, com «escrever <destino>»,
o ficheiro dos retratos para o backend.

  python passo5_ficheiro.py rever
  python passo5_ficheiro.py escrever /caminho/author_portraits.json
"""
import datetime
import json
import os
import re
import sys

from comum import PASTA, ler
from passo3_mesmo import anos_do_catalogo

autores = {a["key"]: a for a in ler("autores.json")}
nomes = ler("nomes.json")
candidatos = ler("candidatos.json")
decisoes = ler("decisoes.json")
imagens = ler("imagens.json", {})
visto = ler("visto.json", {})
AQUI = os.path.dirname(os.path.abspath(__file__))
rejeitados = json.load(open(os.path.join(AQUI, "rejeitados.json"), encoding="utf-8"))   # revisão à mão: chave do autor -> motivo
escolhidos = json.load(open(os.path.join(AQUI, "escolhidos.json"), encoding="utf-8"))   # revisão à mão: chave do autor -> {"qid", "motivo"}


def miniatura(ficheiro):
    return os.path.join(PASTA, "img", re.sub(r"[^A-Za-z0-9._-]+", "_", ficheiro)[:150] + ".thumb")


linhas = []
for chave, cands in candidatos.items():
    a = autores[chave]
    aceites = [c for c in cands if c.get("image") and decisoes.get(f"{chave}|{c['qid']}") == "A"]
    if not aceites:
        continue
    # revisão à mão: quando o Qwen aceitou mais de uma pessoa, a que ficou (as outras saem)
    escolhido = (escolhidos.get(chave) or {}).get("qid")
    if escolhido:
        cands = [c for c in cands if c["qid"] == escolhido]
        aceites = [c for c in aceites if c["qid"] == escolhido]
    n, m = anos_do_catalogo(a)

    def iguais(c):
        return bool(n and m and c.get("birth_year") and c.get("death_year") and abs(n - c["birth_year"]) <= 1 and abs(m - c["death_year"]) <= 1)

    todos_aceites = [c for c in cands if decisoes.get(f"{chave}|{c['qid']}") == "A"]
    for c in aceites:
        img = imagens.get(c["image"]) or {}
        notas = []
        outros = [o for o in todos_aceites if o["qid"] != c["qid"]]
        if any(iguais(o) for o in outros) and not iguais(c):
            notas.append("VÁRIAS: outra pessoa tem as datas iguais")
        elif outros and not iguais(c):
            notas.append(f"VÁRIAS PESSOAS ACEITES ({len(todos_aceites)})")
        if n and c.get("birth_year") and abs(n - c["birth_year"]) > 1:
            notas.append(f"nascimento {n}≠{c['birth_year']}")
        if m and c.get("death_year") and abs(m - c["death_year"]) > 1:
            notas.append(f"morte {m}≠{c['death_year']}")
        datas = "iguais" if (n and m and c.get("birth_year") and c.get("death_year") and abs(n - c["birth_year"]) <= 1 and abs(m - c["death_year"]) <= 1) else \
                ("sem datas no catálogo" if not (n or m) else "parciais")
        # as peças têm de caber na vida da pessoa (com folga para obras póstumas e datas largas)
        lo, hi = a.get("start_century"), a.get("end_century")
        if lo and c.get("birth_year") and hi and hi < (c["birth_year"] // 100 + 1):
            notas.append(f"peças do séc. {hi} antes de nascer ({c['birth_year']})")
        if lo and c.get("death_year") and lo > (c["death_year"] // 100 + 1) + 1:
            notas.append(f"peças do séc. {lo} muito depois de morrer ({c['death_year']})")
        if img.get("missing") or not img.get("url"):
            notas.append("sem ficheiro no Commons")
        if visto.get(c["image"]) != "A":
            notas.append(f"imagem: {visto.get(c['image'], '?')}")
        if nomes[chave]["tipo"] not in ("A", "B"):
            notas.append(f"tipo {nomes[chave]['tipo']}")
        if chave in rejeitados:
            notas.append("REJEITADO À MÃO: " + rejeitados[chave])
        if escolhido:
            datas += "; escolhido na revisão manual entre as pessoas aceites"
        linhas.append({"chave": chave, "a": a, "c": c, "img": img, "datas": datas, "notas": notas})

linhas.sort(key=lambda x: -x["a"]["count"])
bons = [x for x in linhas if not any(n.startswith(("VÁRIAS", "sem ficheiro", "imagem:", "REJEITADO", "tipo")) for n in x["notas"])]
print(f"pares aceites pelo Qwen: {len(linhas)} | utilizáveis: {len(bons)} | datas iguais: {sum(1 for x in bons if x['datas'].startswith('iguais'))} | "
      f"sem datas no catálogo: {sum(1 for x in bons if x['datas'].startswith('sem datas'))} | parciais: {sum(1 for x in bons if x['datas'].startswith('parciais'))}")

if sys.argv[1] == "rever":
    with open(os.path.join(PASTA, "revisao.tsv"), "w", encoding="utf-8") as f:
        for i, x in enumerate(linhas, 1):
            a, c = x["a"], x["c"]
            f.write("\t".join(str(v) for v in [
                i, a["count"], a["name"][:60], "/".join(k[:22] for k in a["museums"]), (a.get("categories") or [""])[0],
                f"séc {a.get('start_century')}-{a.get('end_century')}", "|", c["qid"], (c.get("label") or "")[:40],
                f"{c.get('birth_year')}–{c.get('death_year')}", (c.get("description") or "")[:50], x["datas"], "; ".join(x["notas"]), x["chave"]]) + "\n")
    print("revisao.tsv escrito")
    from PIL import Image, ImageDraw, ImageFont
    fonte, pequena = ImageFont.load_default(15), ImageFont.load_default(13)
    COL, LIN, W, H = 6, 4, 250, 330
    os.makedirs(os.path.join(PASTA, "folhas"), exist_ok=True)
    for pagina in range(0, len(linhas), COL * LIN):
        folha = Image.new("RGB", (COL * W, LIN * H), "white")
        desenho = ImageDraw.Draw(folha)
        for j, x in enumerate(linhas[pagina:pagina + COL * LIN]):
            x0, y0 = (j % COL) * W, (j // COL) * H
            try:
                im = Image.open(miniatura(x["c"]["image"])).convert("RGB")
                im.thumbnail((W - 10, 230))
                folha.paste(im, (x0 + (W - im.width) // 2, y0 + 4))
            except Exception:  # noqa: BLE001
                desenho.text((x0 + 10, y0 + 100), "(sem miniatura)", fill="red", font=fonte)
            cor = "red" if x["notas"] else "black"
            desenho.text((x0 + 5, y0 + 238), f"{pagina + j + 1}. {x['a']['name'][:30]}", fill=cor, font=fonte)
            desenho.text((x0 + 5, y0 + 256), f"WD: {(x['c'].get('label') or '')[:28]}", fill="black", font=pequena)
            desenho.text((x0 + 5, y0 + 272), f"{x['c'].get('birth_year')}–{x['c'].get('death_year')} | {x['datas'][:14]}", fill="black", font=pequena)
            desenho.text((x0 + 5, y0 + 288), "; ".join(x["notas"])[:38], fill="red", font=pequena)
            desenho.text((x0 + 5, y0 + 304), f"{x['a']['count']} peças, séc {x['a'].get('start_century')}-{x['a'].get('end_century')}", fill="gray", font=pequena)
        destino = os.path.join(PASTA, "folhas", f"folha_{pagina // (COL * LIN) + 1:02d}.jpg")
        folha.save(destino, quality=80)
    print("folhas:", (len(linhas) + COL * LIN - 1) // (COL * LIN))
else:
    retratos = {}
    for x in bons:
        a, c, img = x["a"], x["c"], x["img"]
        credito = re.sub(r"\s+", " ", img.get("artist") or "").strip()
        credito = re.sub(r"^(.+?) \1$", r"\1", credito)   # o Commons repete o texto («Unknown author Unknown author»)
        entrada = {
            "image_url": img["url"], "page_url": img.get("page"), "license": img.get("license") or "", "license_url": img.get("license_url") or None,
            "credit": (credito[:117] + "…") if len(credito) > 118 else (credito or None),
            "wikidata": c["qid"], "name": c.get("label"), "catalogue_name": a["name"], "evidence": x["datas"],
        }
        for autor_id in a["ids"]:
            retratos[autor_id] = entrada
    saida = {
        "generated_at": datetime.date.today().isoformat(),
        "source": "Wikidata (P18) e Wikimedia Commons; ver docs/retratos-dos-autores.md",
        "portraits": dict(sorted(retratos.items())),
    }
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump(saida, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"autores com retrato: {len(bons)} | ids: {len(retratos)} → {sys.argv[2]}")
