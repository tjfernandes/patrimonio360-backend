#!/bin/bash
# Serve este ramo numa instância local, ao lado da produção (:8001), para o
# frontend testar o «que peças há aqui?» (GET /api/v1/chat/tour/objects).
#
#   tools/apontar/servir_experiencia.sh            arranca na :8002 (ou diz que já está)
#   tools/apontar/servir_experiencia.sh --status   está a responder?
#   tools/apontar/servir_experiencia.sh --stop     pára e apaga o .env da instância
#
# Usa o .env da produção com estas diferenças: sem pré-aquecimento, sem ditado,
# sem «Apontar» (AMALIA_EXP_POINT=true liga-o; carrega o SAM e os embedders,
# ~20 GB de GPU), imagens e cache das faces da produção, registos em
# logs/experiencia/ deste worktree. Porto por AMALIA_EXP_PORT (8002).
# No frontend: backendBaseUrl = http://127.0.0.1:8002 (CORS aberto).
set -u
RAIZ="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
PROD="${AMALIA_PROD_DIR:-$HOME/Projects/AMALIA/patrimonio360-backend}"
PORTO="${AMALIA_EXP_PORT:-8002}"
PY="${AMALIA_PY:-$HOME/miniconda3/envs/visitas_virtuais/bin/python}"
LOG="$HOME/.amalia_experiencia_$PORTO.log"
PADRAO="uvicorn app.main:app --host 127.0.0.1 --port $PORTO"
BASE="http://127.0.0.1:$PORTO"

verde()    { printf '\033[92m%s\033[0m\n' "$1"; }
vermelho() { printf '\033[91m%s\033[0m\n' "$1"; }
amarelo()  { printf '\033[93m%s\033[0m\n' "$1"; }

vivo() { curl -sf -m 5 "$BASE/api/v1/health" > /dev/null; }

case "${1:---start}" in
  --status)
    if vivo; then verde "experiência ok em $BASE ($(git -C "$RAIZ" rev-parse --abbrev-ref HEAD))"; else vermelho "experiência EM BAIXO ($BASE)"; exit 1; fi ;;
  --stop)
    if pgrep -f "$PADRAO" > /dev/null; then pkill -f "$PADRAO"; sleep 3; verde "instância da :$PORTO parada"; else amarelo "não havia instância na :$PORTO"; fi
    rm -f "$RAIZ/.env" ;;
  --start|*)
    if vivo; then verde "já está a responder em $BASE"; exit 0; fi
    if [ "$RAIZ" = "$PROD" ]; then vermelho "isto é a pasta da produção; corre a partir do worktree do ramo"; exit 2; fi
    [ -f "$PROD/.env" ] || { vermelho "sem $PROD/.env"; exit 2; }
    cp "$PROD/.env" "$RAIZ/.env"
    cat >> "$RAIZ/.env" <<EOF

# --- instância de experiência (servir_experiencia.sh) ---
CHAT_PREWARM_ON_STARTUP=false
SPEECH_STT_ENABLED=false
POINT_AND_ASK_ENABLED=${AMALIA_EXP_POINT:-false}
IMAGE_ASSET_ROOT=$PROD/Images
POINT_TILES_CACHE_DIR=$PROD/tmp/visitas
QUERY_LOG_PATH=logs/experiencia/backend_queries.jsonl
FRONTEND_EVENT_LOG_PATH=logs/experiencia/frontend_events.jsonl
EOF
    mkdir -p "$RAIZ/logs/experiencia"
    amarelo "a arrancar $(git -C "$RAIZ" rev-parse --abbrev-ref HEAD) em $BASE…"
    cd "$RAIZ" || exit 2
    HF_HUB_OFFLINE=1 setsid nohup "$PY" -m $PADRAO > "$LOG" 2>&1 < /dev/null &
    for _ in $(seq 1 90); do sleep 2; vivo && break; done
    if vivo; then
      verde "pronto: $BASE"
      echo "  objetos:  $BASE/api/v1/chat/tour/objects?museum_slug=mnaa&panorama_key=panorama_83D8666B_91ED_0BA2_41D9_9A5E8973B146"
      echo "  log:      $LOG"
      echo "  parar:    $0 --stop"
    else
      vermelho "não subiu em 3 min — ver $LOG"; exit 1
    fi ;;
esac
