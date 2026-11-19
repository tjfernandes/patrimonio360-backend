"""As imagens com menos de 330 px de largura vêm do Commons com o endereço do original, e o servidor dos
originais limita muito os pedidos. Para a verificação basta uma miniatura mais pequena (250 ou 120 px)."""
import os, re, time, urllib.request
from comum import AGENTE, COMMONS, PASTA, api, ler

imagens = ler("imagens.json")
def destino(f):
    return os.path.join(PASTA, "img", re.sub(r"[^A-Za-z0-9._-]+", "_", f)[:150] + ".thumb")
falta = [f for f, d in imagens.items() if d.get("url") and not os.path.exists(destino(f))]
print(len(falta), "em falta")
for largura in ("250", "120"):
    falta = [f for f in falta if not os.path.exists(destino(f))]
    if not falta:
        break
    r = api(COMMONS, action="query", titles="|".join("File:" + f for f in falta), prop="imageinfo", iiprop="url|size", iiurlwidth=largura)
    consulta = r.get("query") or {}
    normal = {n["to"]: n["from"] for n in consulta.get("normalized", [])}
    for pagina in (consulta.get("pages") or {}).values():
        titulo = pagina.get("title") or ""
        f = normal.get(titulo, titulo)[5:]
        info = (pagina.get("imageinfo") or [{}])[0]
        url = info.get("thumburl") or ""
        if "/thumb/" not in url:
            print("  sem miniatura a", largura, ":", f[:60], info.get("width")); continue
        try:
            dados = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": AGENTE}), timeout=60).read()
            open(destino(f), "wb").write(dados)
            print("  ok", largura, f[:60])
        except Exception as e:  # noqa: BLE001
            print("  falhou", f[:60], str(e)[:60])
        time.sleep(2)
    time.sleep(7)
print("ainda em falta:", [f for f in falta if not os.path.exists(destino(f))])
