#!/bin/bash
# Põe (e mantém) a Amália acessível de fora, para o estudo remoto.
#
# A cadeia é: browser do participante → nginx no VPS (:8081) → túnel SSH
# → uvicorn nesta máquina → LLM local (vLLM, 127.0.0.1:8004). Se qualquer elo
# cair, o participante vê um erro de CORS — que é enganador: o que se passa é
# um 502 do nginx, e a página de erro dele não traz cabeçalhos CORS. A causa é
# sempre uma destas:
#   • o uvicorn não está a correr;
#   • o uvicorn mudou de porto e o túnel ficou a apontar ao antigo;
#   • o autossh morreu (reinício do Windows, WSL desligado, rede).
# O LLM local não é um elo crítico: se estiver em baixo, o backend usa a API
# remota (LLM_FALLBACK_BASE_URL) até o supervisor o voltar a pôr de pé.
# O modelo de visão do «apontar» (vLLM, 127.0.0.1:8005, deploy/vlm/amalia_vlm.sh)
# também não: só é arrancado e vigiado se o .env tiver POINT_VERIFY_ENABLED=true,
# e sem ele o apontar decide só pelo cosseno.
#
#   deploy/amalia_online.sh            verifica e repara o que estiver em baixo
#   deploy/amalia_online.sh --status   só diagnostica, não mexe
#   deploy/amalia_online.sh --vigiar   fica a vigiar e repara sozinho (Ctrl+C sai)
#   … --sem-vlm                        (com qualquer dos modos) não arranca nem
#                                      vigia o modelo de visão; se já estiver a
#                                      correr, fica como está
#
# Para desligar tudo (vigilante, túnel, backend, LLM e VLM): deploy/amalia_offline.sh
#
# Para arrancar com o Windows: Agendador de Tarefas → nova tarefa → «Ao iniciar
# sessão» → Programa: wsl.exe
# Argumentos: -d ubuntu-24.04 -- bash -lc "~/Projects/AMALIA/patrimonio360-backend/deploy/amalia_online.sh --vigiar"

set -u
# raiz do repositório do backend (este script está em deploy/): é daqui que o
# uvicorn arranca, e é aqui que ele lê o .env
BACKEND="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
K="$HOME/.ssh/id_ed25519_vps"
H="root@patrimonio360.webredirect.org"
PORTO_VPS=8001                     # a ponta do túnel no VPS (o que o nginx procura)
BASE_EXTERNA="https://patrimonio360.webredirect.org:8081"
LOG_TUNEL="$HOME/.amalia_tunnel.log"
LOG_BACKEND="$HOME/.amalia_backend.log"
# uvicorn do ambiente conda quando o PATH não o traz (shell não interativo,
# tarefa agendada sem o conda ativado): sem isto o arranque falhava em silêncio.
UVICORN_BIN="$(command -v uvicorn 2>/dev/null || echo "$HOME/miniconda3/envs/visitas_virtuais/bin/uvicorn")"
LLM_SCRIPT="$BACKEND/deploy/llm/amalia_llm.sh"
VLM_SCRIPT="$BACKEND/deploy/vlm/amalia_vlm.sh"

verde() { printf '\033[92m%s\033[0m\n' "$1"; }
vermelho() { printf '\033[91m%s\033[0m\n' "$1"; }
amarelo() { printf '\033[93m%s\033[0m\n' "$1"; }

porto_backend() {
  pgrep -a -f "uvicorn app.main" 2>/dev/null \
    | grep -oP -- "--port \K[0-9]+" | head -1
}

# Insiste antes de dar algo por morto: com --reload o uvicorn reinicia-se a si
# próprio e uma verificação isolada apanha o intervalo. Sem isto, o modo
# --vigiar reiniciaria serviços que estão bem (e chegaria a lançar um segundo
# uvicorn no mesmo porto).
tenta() {
  local n=$1; shift
  for i in $(seq 1 "$n"); do
    "$@" && return 0
    [ "$i" -lt "$n" ] && sleep 3
  done
  return 1
}

_backend_ok() {
  local p; p=$(porto_backend)
  [ -n "$p" ] && curl -sf -m 5 "http://127.0.0.1:$p/health" > /dev/null
}
_tunel_ok() {
  ssh -i "$K" -o BatchMode=yes -o ConnectTimeout=8 "$H" \
    "curl -sf -m 5 http://127.0.0.1:$PORTO_VPS/health > /dev/null" 2>/dev/null
}
_fora_ok() {
  [ "$(curl -s -o /dev/null -w '%{http_code}' -m 15 -X POST \
       -H 'Content-Type: application/json' -d '{}' \
       "$BASE_EXTERNA/api/v1/chat/messages")" = "422" ]
}

_llm_ok() { [ -x "$LLM_SCRIPT" ] && "$LLM_SCRIPT" --status > /dev/null 2>&1; }

_vlm_ok() { [ -x "$VLM_SCRIPT" ] && "$VLM_SCRIPT" --status > /dev/null 2>&1; }
# o VLM só faz falta se o backend o for usar (POINT_VERIFY_ENABLED no .env)
vlm_pedido() {
  [ "$SEM_VLM" = 0 ] || return 1
  grep -qiE '^[[:space:]]*POINT_VERIFY_ENABLED[[:space:]]*=[[:space:]]*"?(true|1|yes|on)' "$BACKEND/.env" 2>/dev/null
}

backend_vivo() { tenta 3 _backend_ok; }
tunel_vivo()   { tenta 3 _tunel_ok; }
fora_vivo()    { tenta 3 _fora_ok; }
llm_vivo()     { tenta 2 _llm_ok; }
vlm_vivo()     { tenta 2 _vlm_ok; }
vlm_em_falta() { vlm_pedido && ! vlm_vivo; }

arranca_llm() {
  [ -x "$LLM_SCRIPT" ] || return 1
  amarelo "  a arrancar o LLM local (vLLM)…"
  "$LLM_SCRIPT" > /dev/null 2>&1 && return 0
  vermelho "  o LLM local não subiu — o backend usa a API remota (ver ~/.amalia_llm.log)"; return 1
}

arranca_vlm() {
  [ -x "$VLM_SCRIPT" ] || return 1
  amarelo "  a arrancar o modelo de visão do apontar (vLLM :8005)…"
  "$VLM_SCRIPT" > /dev/null 2>&1 && return 0
  vermelho "  o modelo de visão não subiu — o apontar decide só pelo cosseno (ver ~/.amalia_vlm.log)"; return 1
}

arranca_backend() {
  # Um uvicorn a arrancar não responde ao /health durante os imports (~2 min)
  # e o pré-aquecimento dos embedders (CHAT_PREWARM_ON_STARTUP). Lançar outro
  # nessa janela poria dois processos a carregar modelos na GPU: espera por
  # ele e só o substitui se continuar sem responder.
  local padrao="uvicorn app.main:app --host 127.0.0.1 --port 8001"
  if pgrep -f "$padrao" > /dev/null; then
    amarelo "  o uvicorn já está a arrancar — a aguardar o /health…"
    for _ in $(seq 1 45); do sleep 2; backend_vivo && return 0; done
    amarelo "  continua sem responder — a substituí-lo"
    pkill -f "$padrao"; sleep 5
  fi
  amarelo "  a arrancar o uvicorn (porto 8001)…"
  cd "$BACKEND" || return 1
  nohup "$UVICORN_BIN" app.main:app --host 127.0.0.1 --port 8001 \
    > "$LOG_BACKEND" 2>&1 &
  for _ in $(seq 1 60); do sleep 2; backend_vivo && return 0; done
  vermelho "  o uvicorn não subiu — ver $LOG_BACKEND"; return 1
}

arranca_tunel() {
  local p; p=$(porto_backend)
  amarelo "  a levantar o túnel  VPS:$PORTO_VPS → local:$p …"
  pkill -f "autossh.*patrimonio360" 2>/dev/null && sleep 2
  AUTOSSH_GATETIME=0 nohup autossh -M 0 -N \
    -o "ServerAliveInterval 30" -o "ServerAliveCountMax 3" \
    -o "ExitOnForwardFailure yes" -o "StrictHostKeyChecking=accept-new" \
    -o "BatchMode=yes" -i "$K" \
    -R "127.0.0.1:$PORTO_VPS:127.0.0.1:$p" "$H" \
    > "$LOG_TUNEL" 2>&1 &
  for _ in $(seq 1 10); do sleep 2; tunel_vivo && return 0; done
  vermelho "  o túnel não subiu — ver $LOG_TUNEL"; return 1
}

estado() {
  local p; p=$(porto_backend)
  if llm_vivo;     then verde "  LLM local ok (vLLM 127.0.0.1:8004)"; else amarelo "  LLM local EM BAIXO — respostas pela API remota"; fi
  if [ "$SEM_VLM" = 1 ]; then echo "  VLM       ignorado (--sem-vlm)"
  elif ! vlm_pedido; then echo "  VLM       não pedido (POINT_VERIFY_ENABLED desligado no .env)"
  elif vlm_vivo;    then verde "  VLM       ok (vLLM 127.0.0.1:8005, apontar)"
  else amarelo "  VLM       EM BAIXO — o apontar decide só pelo cosseno"; fi
  if backend_vivo; then verde "  backend   ok (porto $p)"; else vermelho "  backend   EM BAIXO"; fi
  if tunel_vivo;  then verde "  túnel     ok (VPS:$PORTO_VPS → local:$p)"; else vermelho "  túnel     EM BAIXO"; fi
  if fora_vivo;   then verde "  de fora   ok — os participantes conseguem usar"; else vermelho "  de fora   EM BAIXO"; fi
}

repara() {
  # não bloqueia: sem LLM local o backend responde pela API remota
  llm_vivo || arranca_llm || true
  repara_cadeia; local rc=$?
  # depois do LLM (que reserva uma fatia da GPU e conta com ela livre) e sem
  # atrasar a cadeia crítica; também não bloqueia
  if vlm_em_falta; then arranca_vlm || true; fi
  return $rc
}

repara_cadeia() {
  backend_vivo || arranca_backend || return 1
  tunel_vivo   || arranca_tunel   || return 1
  fora_vivo && return 0
  # o túnel pode estar preso num porto antigo
  amarelo "  o exterior falha com os elos locais de pé — a refazer o túnel…"
  arranca_tunel && fora_vivo
}

SEM_VLM=0
MODO="--reparar"
for arg in "$@"; do
  case "$arg" in
    --sem-vlm) SEM_VLM=1 ;;
    --status|--vigiar|--reparar) MODO="$arg" ;;
    *) vermelho "opção desconhecida: $arg (--status, --vigiar, --sem-vlm)"; exit 2 ;;
  esac
done

case "$MODO" in
  --status)
    echo "estado em $(date '+%H:%M:%S'):"; estado ;;
  --vigiar)
    echo "a vigiar de 60 em 60 s (Ctrl+C para sair)"
    while true; do
      if ! fora_vivo; then
        amarelo "$(date '+%H:%M:%S') — em baixo, a reparar"
        repara && verde "$(date '+%H:%M:%S') — reposto" \
               || vermelho "$(date '+%H:%M:%S') — NÃO consegui repor"
      elif ! llm_vivo; then
        amarelo "$(date '+%H:%M:%S') — LLM local em baixo (a usar a API remota), a arrancar"
        arranca_llm && verde "$(date '+%H:%M:%S') — LLM local reposto"
      elif vlm_em_falta; then
        amarelo "$(date '+%H:%M:%S') — modelo de visão em baixo (apontar só pelo cosseno), a arrancar"
        arranca_vlm && verde "$(date '+%H:%M:%S') — modelo de visão reposto"
      fi
      sleep 60
    done ;;
  *)
    echo "verificação em $(date '+%H:%M:%S'):"; estado
    if fora_vivo && llm_vivo && ! vlm_em_falta; then verde "\nEstá tudo de pé — nada a fazer."; else
      echo; repara && { verde "Reposto:"; estado; } || { vermelho "Falhou:"; estado; exit 1; }
    fi ;;
esac
