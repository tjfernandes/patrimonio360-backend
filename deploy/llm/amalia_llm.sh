#!/bin/bash
# Serve o LLM da Amália localmente com vLLM — o mesmo modelo e a mesma revisão
# que a API remota serve (carminho/AMALIA-9B-50-DPO), mas na GPU desta máquina.
#
# Porquê: a API remota custava 3–8 s por resposta (rede + fila partilhada). Em
# local, com prefix caching e CUDA graphs, a resposta final passa a ~1–2 s e as
# chamadas JSON pequenas a ~0,1–0,2 s. Custa VRAM (~29 GB com a utilização por
# omissão), que cabe ao lado do backend (embeddings text8 + VL8, ~33 GB) numa
# RTX PRO 6000 de 96 GB.
#
# Segurança: bind só em 127.0.0.1 (o nginx do VPS nunca vê este porto); a
# revisão do modelo é pinada; os pesos vêm da cache HF partilhada (fora do repo)
# e o arranque corre em modo offline, sem ir à rede.
#
#   deploy/llm/amalia_llm.sh            arranca (se não estiver) e espera ficar pronto
#   deploy/llm/amalia_llm.sh --status   só diz se está a responder
#   deploy/llm/amalia_llm.sh --stop     pára o servidor
#
# Configuração por variáveis (com defaults):
#   AMALIA_LLM_MODEL, AMALIA_LLM_REVISION, AMALIA_LLM_PORT (8004),
#   AMALIA_LLM_GPU_UTIL (0.30), AMALIA_LLM_MAX_MODEL_LEN (16384),
#   AMALIA_LLM_MAX_NUM_SEQS (16), AMALIA_LLM_CONDA_ENV (hbim-llm — o ambiente
#   com vLLM 0.26 já usado no projeto HBIM)
set -u

MODEL="${AMALIA_LLM_MODEL:-carminho/AMALIA-9B-50-DPO}"
REVISION="${AMALIA_LLM_REVISION:-96586af2ca68570fc7e11c0578de312f8712401f}"
PORT="${AMALIA_LLM_PORT:-8004}"
GPU_UTIL="${AMALIA_LLM_GPU_UTIL:-0.30}"
MAX_LEN="${AMALIA_LLM_MAX_MODEL_LEN:-16384}"
MAX_SEQS="${AMALIA_LLM_MAX_NUM_SEQS:-16}"
ENV_NAME="${AMALIA_LLM_CONDA_ENV:-hbim-llm}"
LOG="${AMALIA_LLM_LOG:-$HOME/.amalia_llm.log}"
BASE="http://127.0.0.1:$PORT"

verde()    { printf '\033[92m%s\033[0m\n' "$1"; }
vermelho() { printf '\033[91m%s\033[0m\n' "$1"; }
amarelo()  { printf '\033[93m%s\033[0m\n' "$1"; }

pronto() { curl -sf -m 5 "$BASE/v1/models" 2>/dev/null | grep -q "\"$MODEL\""; }
pid_servidor() { pgrep -f "vllm serve $MODEL .*--port $PORT" | head -1; }

arranca() {
  if pronto; then verde "LLM local já está pronto em $BASE ($MODEL)"; return 0; fi
  if [ -n "$(pid_servidor)" ]; then
    amarelo "vLLM já está a arrancar (pid $(pid_servidor)) — a esperar…"
  else
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    # os scripts activate.d do cuda-nvcc referenciam variáveis sem default
    set +u; conda activate "$ENV_NAME"; set -u
    # triton/flashinfer compilam em JIT: usar o gcc e o CUDA do próprio ambiente
    export CC="${CC:-$CONDA_PREFIX/bin/gcc}"
    export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
    # WSL2: memória pinned/UVA desligada por omissão no vLLM
    export VLLM_WSL2_ENABLE_PIN_MEMORY=1
    # pesos e tokenizer já estão na cache: nunca ir à rede no arranque
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    amarelo "a arrancar vLLM: $MODEL@${REVISION:0:8} em $BASE (gpu util $GPU_UTIL, max len $MAX_LEN)…"
    setsid nohup vllm serve "$MODEL" \
      --revision "$REVISION" \
      --served-model-name "$MODEL" \
      --host 127.0.0.1 --port "$PORT" \
      --gpu-memory-utilization "$GPU_UTIL" \
      --max-model-len "$MAX_LEN" \
      --max-num-seqs "$MAX_SEQS" \
      > "$LOG" 2>&1 < /dev/null &
  fi
  for _ in $(seq 1 180); do
    if pronto; then verde "LLM local pronto em $BASE"; return 0; fi
    if [ -z "$(pid_servidor)" ]; then
      vermelho "o vLLM terminou durante o arranque — últimas linhas de $LOG:"; tail -20 "$LOG"; return 1
    fi
    sleep 5
  done
  vermelho "o vLLM não ficou pronto em 15 min — ver $LOG"; return 1
}

case "${1:---start}" in
  --status)
    if pronto; then verde "LLM local ok ($MODEL em $BASE)"; else vermelho "LLM local EM BAIXO ($BASE)"; exit 1; fi ;;
  --stop)
    p="$(pid_servidor)"
    if [ -z "$p" ]; then amarelo "não há vLLM a correr para $MODEL no porto $PORT"; exit 0; fi
    kill "$p" && sleep 5
    pkill -f "vllm serve $MODEL .*--port $PORT" 2>/dev/null
    verde "vLLM parado" ;;
  --start|*)
    arranca ;;
esac
