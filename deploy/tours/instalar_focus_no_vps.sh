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
# Só mexe no index.html. O script_general.js e o script_mobile.js NUNCA se
# editam: o player do 3DVista verifica a integridade deles e qualquer alteração
# faz aparecer a marca de água «3DVista Academic» em toda a visita (visto a
# 2026-09-18 e a 2026-09-29). As aspas curvas que partem o aviso de alguns
# hotspots são contornadas pelo tour-focus-artifact.js, e os números de
# inventário errados corrigem-se no mapa dos POIs (ver tools/poi_mapping).
#
# Idempotente: se já lá estiverem, só atualiza o ?v=. Faz cópia de segurança do
# index.html na primeira alteração de cada dia. Um bloco inline antigo (sem a
# marca p360-embed-handler) fica como está, para não mexer no que já funciona.
#
#   deploy/tours/instalar_focus_no_vps.sh [mnt mnsr mnaz mj cmag mnaa]
#   FOCUS_LOCAL=1 deploy/tours/instalar_focus_no_vps.sh   # cópia local do tour-focus-artifact.js
#
# Uma visita nova não traz nada disto: o CMAG foi publicado a 2026-09-18 e até
# 2026-09-20 o «Ver na tour» não fazia nada e os hotspots não avisavam a app,
# por não ter os três scripts. Museu novo no VPS => correr isto para ele.
set -euo pipefail
# Corre com o utilizador normal: a chave do VPS e a sessão do gh são dele. Com
# sudo o ssh não encontra a chave e pede a password do root do VPS.
if [ "$(id -u)" = "0" ]; then echo "não corras isto com sudo: a chave do VPS e a sessão do gh são do teu utilizador"; exit 1; fi
HERE="$(cd "$(dirname "$0")" && pwd)"
K="$HOME/.ssh/id_ed25519_vps"; H="root@patrimonio360.webredirect.org"; ROOT=/root/tours
MUSEUS=("$@"); [ ${#MUSEUS[@]} -eq 0 ] && MUSEUS=(mnt mnsr mnaz mj cmag mnaa)
V=$(date +%s)

# A cópia oficial do tour-focus-artifact.js está no repositório do frontend
# (tour-scripts/). O ficheiro no VPS é partilhado por todas as visitas: instalar
# uma cópia desatualizada daqui muda o aspeto de todas (aconteceu a 2026-09-28,
# em que a cópia de 18/09, com o ícone pintado de verde, substituiu a de 28/09).
# Por isso vai-se buscar a versão do frontend; a cópia local só serve se o
# GitHub não responder, e nesse caso avisa-se.
# FOCUS_LOCAL=1 instala a cópia deste repositório mesmo assim: para quando a
# alteração nasce aqui (2026-10-31: yaw/pitch para peças sem hotspot) e o
# frontend ainda não a copiou para tour-scripts/. Enquanto não copiar, uma
# corrida sem FOCUS_LOCAL volta a pôr a versão do frontend.
FOCUS_JS="$HERE/tour-focus-artifact.js"
TMP_FOCUS=$(mktemp)
if [ "${FOCUS_LOCAL:-0}" = "1" ]; then
  echo "FOCUS_LOCAL=1: instalo a cópia local ($(grep -o 'VERSION = "[^"]*"' "$FOCUS_JS")); copia-a para tour-scripts/ no frontend"
elif gh api repos/tjfernandes/patrimonio360-frontend/contents/tour-scripts/tour-focus-artifact.js \
     -H "Accept: application/vnd.github.raw" > "$TMP_FOCUS" 2>/dev/null && grep -q 'var VERSION' "$TMP_FOCUS"; then
  FOCUS_JS="$TMP_FOCUS"
  if ! cmp -s "$TMP_FOCUS" "$HERE/tour-focus-artifact.js"; then
    echo "aviso: deploy/tours/tour-focus-artifact.js está diferente do frontend; instalo a do frontend ($(grep -o 'VERSION = "[^"]*"' "$TMP_FOCUS"))"
    echo "       para atualizar a cópia daqui: cp $TMP_FOCUS $HERE/tour-focus-artifact.js"
  fi
else
  echo "aviso: não consegui ler o script do repositório do frontend (gh api); instalo a cópia local ($(grep -o 'VERSION = "[^"]*"' "$HERE/tour-focus-artifact.js"))"
fi
scp -q -i "$K" "$FOCUS_JS" "$H:$ROOT/tour-focus-artifact.js"
scp -q -i "$K" "$HERE/tour-embed-handler.html" "$H:$ROOT/tour-embed-handler.html"

# Mapa dos números de inventário corrigidos (validar_inventarios_poi.py guarda o
# original em `inventoryIdsOriginal`): a visita tem de anunciar o número que o
# catálogo usa, senão a peça aberta não é reconhecida.
# O ficheiro no VPS é um só para todas as visitas: constrói-se sempre com todos
# os museus que têm POIs, e não só com os desta corrida (instalar só o mnaa a
# 2026-09-28 deixou-o vazio e apagou as correções do MNAz).
# Ficheiro temporário próprio: um nome fixo em /tmp fica preso ao utilizador que
# o criou (uma corrida com sudo deixou-o do root e as seguintes falhavam).
MAPA=$(mktemp)
trap 'rm -f "$MAPA" "$TMP_FOCUS"' EXIT
python3 - "$HERE" > "$MAPA" <<'PY'
import json, pathlib, sys
here = pathlib.Path(sys.argv[1]); mapa = {}
for ficheiro in sorted((here.parents[1] / "poi_tours").glob("panorama-overlays-inventory-*.json")):
    museu = ficheiro.stem.replace("panorama-overlays-inventory-", "")
    por_museu = {}
    for poi in json.loads(ficheiro.read_text(encoding="utf-8")):
        antigos, novos = poi.get("inventoryIdsOriginal"), poi.get("inventoryIds")
        if antigos and novos and antigos != novos and len(antigos) == 1:
            por_museu[antigos[0]] = novos
    if por_museu: mapa[museu] = por_museu
json.dump(mapa, sys.stdout, ensure_ascii=False)
PY
scp -q -i "$K" "$MAPA" "$H:$ROOT/inventarios-corrigidos.json"
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
        novo = re.sub(r'<script src="\.\./tour-focus-artifact\.js\?v=[\w-]+"></script>', FOCUS, html)
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
        m = re.search(r'[ \t]*<script src="\.\./tour-focus-artifact\.js\?v=[\w-]+"></script>\n', html)
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
