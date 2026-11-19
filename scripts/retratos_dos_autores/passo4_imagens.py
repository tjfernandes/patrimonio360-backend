"""Passo 4: para os pares aceites, os dados da imagem no Commons (endereço, licença, autor),
a miniatura para verificação e a pergunta ao Qwen (com a imagem): é o retrato de uma pessoa? Retomável."""
import html
import os
import re
import time
import urllib.request

from comum import AGENTE, COMMONS, PASTA, api, gravar, ler, qwen

autores = {a["key"]: a for a in ler("autores.json")}
candidatos = ler("candidatos.json")
decisoes = ler("decisoes.json")
imagens = ler("imagens.json", {})      # ficheiro -> dados do Commons
visto = ler("visto.json", {})          # ficheiro -> resposta do Qwen sobre a imagem
os.makedirs(os.path.join(PASTA, "img"), exist_ok=True)

aceites = [(k, c) for k, cands in candidatos.items() for c in cands if c.get("image") and decisoes.get(f"{k}|{c['qid']}") == "A"]
ficheiros = sorted({c["image"] for _, c in aceites})
print("pares aceites:", len(aceites), "| ficheiros:", len(ficheiros), flush=True)


def limpo(texto):
    texto = re.sub(r"<[^>]+>", " ", texto or "")
    return re.sub(r"\s+", " ", html.unescape(texto)).strip()


falta = [f for f in ficheiros if f not in imagens]
for i in range(0, len(falta), 40):
    lote = falta[i:i + 40]
    r = api(COMMONS, action="query", titles="|".join("File:" + f for f in lote), prop="imageinfo", iiprop="url|size|mime|extmetadata",
            iiurlwidth="330", iiextmetadatafilter="LicenseShortName|Artist|Credit|UsageTerms|AttributionRequired|LicenseUrl|Copyrighted")
    consulta = r.get("query") or {}
    normal = {n["to"]: n["from"] for n in consulta.get("normalized", [])}
    for pagina in (consulta.get("pages") or {}).values():
        titulo = pagina.get("title") or ""
        pedido = normal.get(titulo, titulo)[5:]
        info = (pagina.get("imageinfo") or [{}])[0]
        if not info.get("url"):
            imagens[pedido] = {"missing": True}
            continue
        meta = {k: limpo(v.get("value")) for k, v in (info.get("extmetadata") or {}).items()}
        imagens[pedido] = {
            "url": info.get("thumburl") or info["url"], "original": info["url"], "page": info.get("descriptionurl"),
            "width": info.get("thumbwidth") or info.get("width"), "height": info.get("thumbheight") or info.get("height"),
            "mime": info.get("mime"), "license": meta.get("LicenseShortName"), "license_url": meta.get("LicenseUrl"),
            "artist": meta.get("Artist"), "credit": meta.get("Credit"), "attribution_required": meta.get("AttributionRequired"),
        }
    gravar("imagens.json", imagens)
    print(f"  commons {i + len(lote)}/{len(falta)}", flush=True)
    time.sleep(7)

for n, f in enumerate(ficheiros, 1):
    dados = imagens.get(f) or {}
    destino = os.path.join(PASTA, "img", re.sub(r"[^A-Za-z0-9._-]+", "_", f)[:150] + ".thumb")
    if dados.get("missing") or not dados.get("url"):
        continue
    if not os.path.exists(destino):
        for tentativa in range(4):
            try:
                pedido = urllib.request.Request(dados["url"].split("?")[0], headers={"User-Agent": AGENTE})
                conteudo = urllib.request.urlopen(pedido, timeout=60).read()
                with open(destino, "wb") as saida:
                    saida.write(conteudo)
                break
            except Exception as erro:  # noqa: BLE001
                print("  miniatura falhou", f[:50], str(erro)[:60], flush=True)
                time.sleep(130)
        time.sleep(3)
    if f in visto or not os.path.exists(destino):
        continue
    with open(destino, "rb") as entrada:
        conteudo = entrada.read()
    mime = "image/png" if conteudo[:4] == b"\x89PNG" else "image/jpeg"
    r = qwen(
        "Esta imagem vai aparecer como retrato de um artista na ficha de autor de um museu. O que mostra a imagem?\n"
        "A) o retrato de uma só pessoa (fotografia, pintura, desenho, gravura, busto ou estátua de uma pessoa)\n"
        "B) várias pessoas, sem se perceber qual é a retratada\n"
        "C) uma obra de arte que não é um retrato, um objeto, um edifício, um túmulo, uma lápide, uma assinatura, um brasão ou um documento\n"
        'Responde só com JSON: {"resposta": "A", "B" ou "C"}',
        imagem=conteudo, mime=mime, max_tokens=30) or {}
    visto[f] = str(r.get("resposta") or "").strip().upper()[:1]
    if n % 20 == 0:
        gravar("visto.json", visto)
        print(f"  imagens {n}/{len(ficheiros)}", flush=True)
gravar("visto.json", visto)
valores = [visto.get(f, "?") for f in ficheiros]
print("imagens:", {x: valores.count(x) for x in sorted(set(valores))}, "| sem ficheiro no Commons:", sum(1 for f in ficheiros if (imagens.get(f) or {}).get("missing")))
