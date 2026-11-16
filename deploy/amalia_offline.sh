#!/bin/bash
# Desliga a Amália: o contrário do amalia_online.sh.
#
# Pára, por esta ordem:
#   1. o vigilante (amalia_online.sh, em qualquer modo), senão voltava a pôr
#      tudo de pé no minuto seguinte;
#   2. o túnel SSH para o VPS (autossh): a partir daqui os participantes veem
#      um 502 do nginx (que o browser mostra como erro de CORS);
#   3. o uvicorn da porta 8001, e com ele o SAM 3, os embedders e o ditado,
#      que vivem dentro do processo;
#   4. o LLM local (vLLM na 8004), pelo deploy/llm/amalia_llm.sh --stop;
#   5. o modelo de visão do apontar (vLLM na 8005), pelo deploy/vlm/amalia_vlm.sh --stop.
#
# Não mexe no nginx do VPS nem em instâncias de teste noutros portos: lista-as
# no fim.
#
#   deploy/amalia_offline.sh                desliga tudo
#   deploy/amalia_offline.sh --manter-llm   desliga tudo menos os dois vLLM (LLM
#                                           e modelo de visão), para um reinício
#                                           rápido do backend
#   deploy/amalia_offline.sh --manter-vlm   desliga tudo menos o modelo de visão
#                                           (p. ex. para as ferramentas offline
#                                           de tools/apontar e tools/percursos)
#   deploy/amalia_offline.sh --simular      só mostra o que pararia
#
# Para voltar a ligar: deploy/amalia_online.sh
# A tarefa agendada do Windows («Ao iniciar sessão») volta a lançar o vigilante
# no próximo início de sessão.

set -u
BACKEND="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
LLM_SCRIPT="$BACKEND/deploy/llm/amalia_llm.sh"
VLM_SCRIPT="$BACKEND/deploy/vlm/amalia_vlm.sh"
PORTO_BACKEND=8001
PORTO_VPS=8001
PADRAO_VIGILANTE="amalia_online\.sh"
PADRAO_TUNEL="autossh.*patrimonio360"
# o ssh filho do autossh (o autossh, ao sair, normalmente leva-o consigo)
PADRAO_SSH_TUNEL="^(/usr/bin/)?ssh .*-R 127\.0\.0\.1:$PORTO_VPS:127\.0\.0\.1:"
PADRAO_BACKEND="uvicorn app\.main:app --host 127\.0\.0\.1 --port $PORTO_BACKEND"
PADRAO_LLM="vllm serve "

verde()    { printf '\033[92m%s\033[0m\n' "$1"; }
vermelho() { printf '\033[91m%s\033[0m\n' "$1"; }
amarelo()  { printf '\033[93m%s\033[0m\n' "$1"; }

MANTER_LLM=0
MANTER_VLM=0
SIMULAR=0
for arg in "$@"; do
  case "$arg" in
    --manter-llm) MANTER_LLM=1; MANTER_VLM=1 ;;
    --manter-vlm) MANTER_VLM=1 ;;
    --simular)    SIMULAR=1 ;;
    -h|--help)    sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) vermelho "opção desconhecida: $arg (ver --help)"; exit 2 ;;
  esac
done

if [ "$(id -u)" = "0" ]; then
  vermelho "não corras isto com sudo: os processos são do teu utilizador"; exit 2
fi

# pids que casam com o padrão, sem este script nem o shell que o chamou
pids() {
  pgrep -f "$1" 2>/dev/null | grep -vx -e "$$" -e "$PPID" || true
}

mostra() {
  local p
  for p in $1; do ps -o pid=,args= -p "$p" 2>/dev/null | cut -c1-150 | sed 's/^/      /'; done
}

# SIGTERM, espera até $3 segundos e, se ainda houver alguém, SIGKILL
para() {
  local nome=$1 padrao=$2 espera=$3 lista
  lista=$(pids "$padrao")
  if [ -z "$lista" ]; then
    echo "  $nome: nada a correr"
    return 0
  fi
  if [ "$SIMULAR" = 1 ]; then
    amarelo "  $nome: pararia"; mostra "$lista"
    return 0
  fi
  amarelo "  $nome: a parar…"
  # shellcheck disable=SC2086
  kill $lista 2>/dev/null
  for _ in $(seq 1 "$espera"); do
    [ -z "$(pids "$padrao")" ] && { verde "  $nome: parado"; return 0; }
    sleep 1
  done
  lista=$(pids "$padrao")
  amarelo "  $nome: não saiu em ${espera} s — SIGKILL"
  # shellcheck disable=SC2086
  kill -9 $lista 2>/dev/null
  sleep 1
  if [ -z "$(pids "$padrao")" ]; then verde "  $nome: parado"; return 0; fi
  vermelho "  $nome: CONTINUA a correr"; mostra "$(pids "$padrao")"; return 1
}

falhas=0
[ "$SIMULAR" = 1 ] && echo "simulação em $(date '+%H:%M:%S') (nada é parado):" \
                   || echo "a desligar a Amália em $(date '+%H:%M:%S'):"

para "vigilante (amalia_online.sh)" "$PADRAO_VIGILANTE" 5 || falhas=$((falhas + 1))
para "túnel VPS:$PORTO_VPS (autossh)" "$PADRAO_TUNEL" 10 || falhas=$((falhas + 1))
para "túnel VPS:$PORTO_VPS (ssh)" "$PADRAO_SSH_TUNEL" 5 || falhas=$((falhas + 1))
# o uvicorn descarrega os modelos da GPU ao sair: dar-lhe tempo
para "backend (uvicorn :$PORTO_BACKEND)" "$PADRAO_BACKEND" 30 || falhas=$((falhas + 1))

if [ "$MANTER_LLM" = 1 ]; then
  echo "  LLM local: mantido (--manter-llm)"
elif [ "$SIMULAR" = 1 ]; then
  para "LLM local (vLLM :8004)" "$PADRAO_LLM.*--port 8004" 0
elif [ -x "$LLM_SCRIPT" ]; then
  "$LLM_SCRIPT" --stop | sed 's/^/  LLM local: /'
  # o --stop só procura o modelo e o porto por omissão; confirma que não ficou nenhum vLLM
  if [ -n "$(pids "$PADRAO_LLM.*--port 8004")" ]; then
    para "LLM local (vLLM :8004)" "$PADRAO_LLM.*--port 8004" 20 || falhas=$((falhas + 1))
  fi
else
  para "LLM local (vLLM :8004)" "$PADRAO_LLM.*--port 8004" 20 || falhas=$((falhas + 1))
fi

if [ "$MANTER_VLM" = 1 ]; then
  echo "  modelo de visão: mantido"
elif [ "$SIMULAR" = 1 ]; then
  para "modelo de visão (vLLM :8005)" "$PADRAO_LLM.*--port 8005" 0
elif [ -x "$VLM_SCRIPT" ]; then
  "$VLM_SCRIPT" --stop | sed 's/^/  modelo de visão: /'
  # o --stop só procura o modelo e o porto por omissão; confirma que não ficou nenhum vLLM na 8005
  if [ -n "$(pids "$PADRAO_LLM.*--port 8005")" ]; then
    para "modelo de visão (vLLM :8005)" "$PADRAO_LLM.*--port 8005" 20 || falhas=$((falhas + 1))
  fi
else
  para "modelo de visão (vLLM :8005)" "$PADRAO_LLM.*--port 8005" 20 || falhas=$((falhas + 1))
fi

# o que fica de pé (instâncias de teste, os vLLM mantidos com --manter-llm/--manter-vlm), só para informação
outros=$(pids "uvicorn app\.main:app"; pids "$PADRAO_LLM")
if [ -n "$outros" ] && [ "$SIMULAR" = 0 ]; then
  echo "  continuam a correr:"; mostra "$outros"
fi

if [ "$SIMULAR" = 0 ] && command -v nvidia-smi > /dev/null; then
  echo "  GPU: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1)"
fi

if [ "$SIMULAR" = 1 ]; then
  sufixo=""
  if [ "$MANTER_LLM" = 1 ]; then sufixo=" --manter-llm"; elif [ "$MANTER_VLM" = 1 ]; then sufixo=" --manter-vlm"; fi
  echo; echo "para desligar mesmo: deploy/amalia_offline.sh$sufixo"
elif [ "$falhas" = 0 ]; then
  echo; verde "A Amália está desligada. Para voltar: deploy/amalia_online.sh"
else
  echo; vermelho "Ficaram $falhas processo(s) por parar (ver acima)."; exit 1
fi
