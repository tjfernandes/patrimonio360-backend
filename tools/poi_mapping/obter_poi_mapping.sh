#!/bin/bash
# Gera poi_tours/panorama-overlays-inventory-<museu>.json a partir da visita publicada no VPS.
#
# As visitas só existem no VPS (/root/tours, servidas na porta 5500). Isto descarrega o
# script_general.js (e o script_mobile.js, se houver) e os locale/*.txt para
# tmp/visitas/<museu>/ e corre o extract_poi_mapping.py, que lê os hotspots sem browser e
# dá a cada POI o nome da zona pelo subtitle do panorama. Se já houver um mapa para o
# museu, mostra o que mudou e mantém as listas de inventário que o export novo colapsou.
#
#   tools/poi_mapping/obter_poi_mapping.sh mnaa
#   tools/poi_mapping/obter_poi_mapping.sh mnaa --so-comparar   # não escreve em poi_tours/
#
# Depois: validar os números contra o índice com validate_tour.py (ver README desta pasta).
set -euo pipefail
MUSEU="${1:?uso: obter_poi_mapping.sh <museu> [--so-comparar]}"
AQUI="$(cd "$(dirname "$0")" && pwd)"
RAIZ="$(cd "$AQUI/../.." && pwd)"
BASE="${TOURS_BASE_URL:-https://patrimonio360.webredirect.org:5500}"
PY="${PYTHON:-$HOME/miniconda3/envs/visitas_virtuais/bin/python}"
DESTINO="$RAIZ/tmp/visitas/$MUSEU"
ATUAL="$RAIZ/poi_tours/panorama-overlays-inventory-$MUSEU.json"

mkdir -p "$DESTINO"
for f in script_general.js script_mobile.js; do
  if curl -sfL -m 120 -o "$DESTINO/$f.novo" "$BASE/$MUSEU/$f"; then
    mv "$DESTINO/$f.novo" "$DESTINO/$f"
    echo "descarregado: $BASE/$MUSEU/$f ($(du -h "$DESTINO/$f" | cut -f1))"
  else
    rm -f "$DESTINO/$f.novo"
  fi
done
[ -f "$DESTINO/script_general.js" ] || [ -f "$DESTINO/script_mobile.js" ] || { echo "não encontrei a visita em $BASE/$MUSEU/"; exit 1; }

# os nomes dos panoramas (label e subtitle) estão nos locale/<língua>.txt que o index.html lista
mkdir -p "$DESTINO/locale"
for l in $(curl -sfL -m 60 "$BASE/$MUSEU/" | grep -oE 'locale/[A-Za-z_-]+\.txt' | sort -u); do
  curl -sfL -m 60 -o "$DESTINO/$l" "$BASE/$MUSEU/$l" && echo "descarregado: $BASE/$MUSEU/$l" || rm -f "$DESTINO/$l"
done

# as páginas dos POIs (iFrames/generated-html/<sala>/<peça>.html), para o nationalTreasure:
# conta o selo <b style="color: goldenrod"> Tesouro Nacional </b> da página que cada hotspot abre
paginas=0; falhas=0
while IFS= read -r url; do
  [ -n "$url" ] || continue
  local_path="$DESTINO/$(printf '%b' "${url//%/\\x}")"
  mkdir -p "$(dirname "$local_path")"
  if curl -sfL -m 60 -o "$local_path" "$BASE/$MUSEU/${url// /%20}"; then
    paginas=$((paginas + 1))
  else
    falhas=$((falhas + 1)); rm -f "$local_path"; echo "não descarreguei: $BASE/$MUSEU/$url"
  fi
done < <(cat "$DESTINO"/locale/*.txt "$DESTINO"/script_*.js 2>/dev/null \
           | grep -oE 'PopupWebFrameBehaviour_[0-9A-Fa-f_]+\.url = [^\r]*\.html|i[Ff]rames/generated-html/[^"<>\\]*\.html' \
           | sed -E 's/^PopupWebFrameBehaviour_[0-9A-Fa-f_]+\.url = //' | grep 'generated-html/' | sort -u)
echo "páginas dos POIs: $paginas descarregadas$([ "$falhas" -gt 0 ] && echo ", $falhas em falta")"

ARGS=("$DESTINO")
if [ "${2:-}" = "--so-comparar" ]; then
  ARGS+=(-o "$DESTINO/panorama-overlays-inventory-$MUSEU.json")
else
  ARGS+=(-o "$ATUAL")
fi
# o extract escreve antes de comparar: compara-se com uma cópia do mapa anterior
if [ -f "$ATUAL" ]; then
  cp "$ATUAL" "$DESTINO/anterior.json"
  ARGS+=(--compare "$DESTINO/anterior.json" --keep-better "$DESTINO/anterior.json")
fi
"$PY" "$AQUI/extract_poi_mapping.py" "${ARGS[@]}"
