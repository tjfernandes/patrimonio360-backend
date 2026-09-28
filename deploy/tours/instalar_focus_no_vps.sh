#!/bin/bash
# Repõe nas visitas 3DVista do VPS os três scripts que ligam a visita ao
# assistente. Uma republicação da visita pelo 3DVista reescreve o index.html e
# apaga-os todos (aconteceu ao MNT e ao MNAz a 2026-09-18: o «Ver na tour»
# deixou de fazer nada e a posição do visitante deixou de chegar).
#
# As visitas vivem em /root/tours/<museu>/ (servidas por `npx serve` na :5501,
# atrás do nginx :5500). Instala, por esta ordem:
#   1. <head>  ../tour-focus-artifact.js   modo «focus» (câmara no hotspot,
#              ícone a pulsar, sem abrir a ficha). Tem de vir ANTES dos outros:
#              o listener é de captura e pára a propagação em modo «focus».
#   2. <head>  ../tour-event-bridge.js     eventos para o assistente (mudança de
#              panorama, ficha aberta/fechada) — «onde estou?» e a peça em foco.
#   3. </body> bloco inline (tour-embed-handler.html): navigateToArtifact
#              «clássico» (muda de panorama e abre a ficha) e o aviso de fecho.
#
# Repara ainda o script_general.js: no MNAz o `location` dentro do eval de cada
# hotspot foi escrito com aspas curvas («location: “20_Expo_Temp_P3”»), o eval
# rebenta com «Invalid or unexpected token», o try/catch da própria visita
# engole o erro e o assistente nunca fica a saber que peça foi aberta. Só se
# trocam as aspas desse campo (45 ocorrências no MNAz, 0 nas outras visitas).
#
# Idempotente: se já lá estiverem, só atualiza o ?v=. Faz cópia de segurança do
# index.html na primeira alteração de cada dia. Um bloco inline antigo (sem a
# marca p360-embed-handler) fica como está, para não mexer no que já funciona.
#
#   deploy/tours/instalar_focus_no_vps.sh [mnt mnsr mnaz mj cmag]
#
# Uma visita nova não traz nada disto: o CMAG foi publicado a 2026-09-18 e até
# 2026-09-20 o «Ver na tour» não fazia nada e os hotspots não avisavam a app,
# por não ter os três scripts. Museu novo no VPS => correr isto para ele.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
K="$HOME/.ssh/id_ed25519_vps"; H="root@patrimonio360.webredirect.org"; ROOT=/root/tours
MUSEUS=("$@"); [ ${#MUSEUS[@]} -eq 0 ] && MUSEUS=(mnt mnsr mnaz mj cmag)
V=$(date +%s)
scp -q -i "$K" "$HERE/tour-focus-artifact.js" "$H:$ROOT/tour-focus-artifact.js"
scp -q -i "$K" "$HERE/tour-embed-handler.html" "$H:$ROOT/tour-embed-handler.html"

# Mapa dos números de inventário corrigidos (validar_inventarios_poi.py guarda o
# original em `inventoryIdsOriginal`): a visita tem de anunciar o número que o
# catálogo usa, senão a peça aberta não é reconhecida.
python3 - "$HERE" "${MUSEUS[@]}" > /tmp/p360_inv_map.json <<'PY'
import json, pathlib, sys
here = pathlib.Path(sys.argv[1]); mapa = {}
for museu in sys.argv[2:]:
    ficheiro = here.parents[1] / "poi_tours" / f"panorama-overlays-inventory-{museu}.json"
    if not ficheiro.exists(): continue
    por_museu = {}
    for poi in json.loads(ficheiro.read_text(encoding="utf-8")):
        antigos, novos = poi.get("inventoryIdsOriginal"), poi.get("inventoryIds")
        if antigos and novos and antigos != novos and len(antigos) == 1:
            por_museu[antigos[0]] = novos
    if por_museu: mapa[museu] = por_museu
json.dump(mapa, sys.stdout, ensure_ascii=False)
PY
scp -q -i "$K" /tmp/p360_inv_map.json "$H:$ROOT/inventarios-corrigidos.json"
ssh -i "$K" -o BatchMode=yes "$H" "export V=$V; cd $ROOT && python3 - ${MUSEUS[*]}" <<'PY'
import datetime, json, os, re, shutil, sys

v = os.environ.get("V") or str(int(datetime.datetime.now().timestamp()))
handler = open("tour-embed-handler.html", encoding="utf-8").read().rstrip("\n")
FOCUS = f'<script src="../tour-focus-artifact.js?v={v}"></script>'
try:
    INVENTARIOS = json.load(open("inventarios-corrigidos.json", encoding="utf-8"))
except Exception:
    INVENTARIOS = {}
BRIDGE = '<script src="../tour-event-bridge.js?v=20260624-1"></script>'

for museum in sys.argv[1:]:
    path = f"{museum}/index.html"
    try:
        html = original = open(path, encoding="utf-8").read()
    except FileNotFoundError:
        print(museum, "ERRO: index.html não encontrado"); continue
    feito = []

    # 1. script de foco, logo a seguir ao script.js do 3DVista
    if "tour-focus-artifact.js" in html:
        novo = re.sub(r'<script src="\.\./tour-focus-artifact\.js\?v=\d+"></script>', FOCUS, html)
        if novo != html: feito.append("foco (v atualizado)")
        html = novo
    else:
        m = re.search(r'[ \t]*<script src="script\.js\?v=\d+"></script>\n', html)
        if not m:
            print(museum, "ERRO: não encontrei a tag do script.js"); continue
        indent = re.match(r"[ \t]*", m.group(0)).group(0)
        html = html[:m.end()] + indent + FOCUS + "\n" + html[m.end():]
        feito.append("foco (inserido)")

    # 2. ponte de eventos, a seguir ao script de foco
    if "tour-event-bridge.js" not in html:
        m = re.search(r'[ \t]*<script src="\.\./tour-focus-artifact\.js\?v=\d+"></script>\n', html)
        indent = re.match(r"[ \t]*", m.group(0)).group(0)
        html = html[:m.end()] + indent + BRIDGE + "\n" + html[m.end():]
        feito.append("ponte (inserida)")

    # 3. bloco inline antes de </body>
    if "p360-embed-handler" in html:
        html = re.sub(r"[ \t]*<!-- p360-embed-handler.*?</script>\n", handler + "\n", html, flags=re.S)
        if html != original: feito.append("handler (atualizado)")
    elif "navigateToArtifact" in html:
        feito.append("handler (já lá estava, não mexi)")
    else:
        m = re.search(r"[ \t]*</body>", html)
        if not m:
            print(museum, "ERRO: não encontrei </body>"); continue
        html = html[:m.start()] + handler + "\n" + html[m.start():]
        feito.append("handler (inserido)")

    if html != original:
        bak = f"{path}.bak-{datetime.date.today().isoformat()}"
        if not os.path.exists(bak): shutil.copy2(path, bak)
        open(path, "w", encoding="utf-8").write(html)

    print(f"{museum:5} {'; '.join(feito) if feito else 'sem alterações'}")
PY
echo "instalado (v=$V)"
for m in "${MUSEUS[@]}"; do
  h=$(curl -sk "https://patrimonio360.webredirect.org:5500/$m/")
  printf "  %-5s foco=%s ponte=%s handler=%s\n" "$m" \
    "$(echo "$h" | grep -c 'tour-focus-artifact.js')" \
    "$(echo "$h" | grep -c 'tour-event-bridge.js')" \
    "$(echo "$h" | grep -c 'navigateToArtifact')"
done
