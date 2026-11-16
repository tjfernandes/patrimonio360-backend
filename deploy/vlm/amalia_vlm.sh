#!/bin/bash
# Serve o modelo de visão do «apontar e perguntar» (Qwen3-VL-8B, vLLM, API
# OpenAI) na GPU desta máquina, em 127.0.0.1:8005.
#
# É a ele que o backend pergunta, com POINT_VERIFY_ENABLED=true, qual das 3
# melhores candidatas é exatamente a peça apontada (app/services/point_verify.py,
# POINT_VERIFY_BASE_URL). Sem ele a responder o apontar continua a funcionar,
# mas só pelo cosseno: o registo traz «verify: unavailable».
# As ferramentas offline (tools/apontar, tools/percursos) usam o mesmo servidor.
#
# Memória: o modelo ocupa ~11,5 GB mais a cache KV. A cache é fixada em GB
# (AMALIA_VLM_KV_CACHE_GB, 8 por omissão → ~20 GB no total) em vez de ser uma
# fatia da GPU: com a AMÁLIA e o backend já carregados, o vLLM desconta da fatia
# a memória que os outros ocupam e morre no arranque com «No available memory
# for the cache blocks» (visto a 2026-09-29). Com a cache fixa arranca tanto com
# a GPU livre como ao lado da produção.
#
# Segurança: bind só em 127.0.0.1; a revisão do modelo é pinada; os pesos vêm
# da cache HF e o arranque corre em modo offline, sem ir à rede.
#
#   deploy/vlm/amalia_vlm.sh            arranca (se não estiver) e espera ficar pronto
#   deploy/vlm/amalia_vlm.sh --status   só diz se está a responder
#   deploy/vlm/amalia_vlm.sh --stop     pára o servidor
#
# Configuração por variáveis (com defaults):
#   AMALIA_VLM_MODEL, AMALIA_VLM_REVISION, AMALIA_VLM_PORT (8005),
#   AMALIA_VLM_KV_CACHE_GB (8), AMALIA_VLM_GPU_UTIL (0.30, ignorada pela cache
#   fixa), AMALIA_VLM_MAX_MODEL_LEN (12288), AMALIA_VLM_MAX_NUM_SEQS (32),
#   AMALIA_VLM_IMAGES_PER_PROMPT (3), AMALIA_VLM_CONDA_ENV (hbim-llm, o mesmo
#   ambiente com vLLM 0.26 do deploy/llm/amalia_llm.sh)
# Se mudares o modelo ou o porto, muda também POINT_VERIFY_MODEL e
# POINT_VERIFY_BASE_URL no .env do backend.
set -u

MODEL="${AMALIA_VLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct-FP8}"
REVISION="${AMALIA_VLM_REVISION:-9cdc6310a8cb770ce18efaf4e9935334512aee45}"
PORT="${AMALIA_VLM_PORT:-8005}"
KV_CACHE_GB="${AMALIA_VLM_KV_CACHE_GB:-8}"
GPU_UTIL="${AMALIA_VLM_GPU_UTIL:-0.30}"
# até 3 fotografias por pedido (~1 400 tokens de visão cada) + texto + resposta
MAX_LEN="${AMALIA_VLM_MAX_MODEL_LEN:-12288}"
MAX_SEQS="${AMALIA_VLM_MAX_NUM_SEQS:-32}"
IMAGES="${AMALIA_VLM_IMAGES_PER_PROMPT:-3}"
ENV_NAME="${AMALIA_VLM_CONDA_ENV:-hbim-llm}"
LOG="${AMALIA_VLM_LOG:-$HOME/.amalia_vlm.log}"
BASE="http://127.0.0.1:$PORT"

verde()    { printf '\033[92m%s\033[0m\n' "$1"; }
vermelho() { printf '\033[91m%s\033[0m\n' "$1"; }
amarelo()  { printf '\033[93m%s\033[0m\n' "$1"; }

pronto() { curl -sf -m 5 "$BASE/v1/models" 2>/dev/null | grep -q "\"$MODEL\""; }
pid_servidor() { pgrep -f "vllm serve $MODEL .*--port $PORT" | head -1; }

arranca() {
  if pronto; then verde "VLM já está pronto em $BASE ($MODEL)"; return 0; fi
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
    amarelo "a arrancar vLLM: $MODEL@${REVISION:0:8} em $BASE (cache KV de $KV_CACHE_GB GB, max len $MAX_LEN)…"
    setsid nohup vllm serve "$MODEL" \
      --revision "$REVISION" \
      --served-model-name "$MODEL" \
      --host 127.0.0.1 --port "$PORT" \
      --gpu-memory-utilization "$GPU_UTIL" \
      --kv-cache-memory-bytes "$((KV_CACHE_GB * 1024 * 1024 * 1024))" \
      --max-model-len "$MAX_LEN" \
      --limit-mm-per-prompt "{\"image\": $IMAGES}" \
      --max-num-seqs "$MAX_SEQS" \
      > "$LOG" 2>&1 < /dev/null &
  fi
  for _ in $(seq 1 180); do
    if pronto; then verde "VLM pronto em $BASE"; return 0; fi
    if [ -z "$(pid_servidor)" ]; then
      vermelho "o vLLM terminou durante o arranque — últimas linhas de $LOG:"; tail -20 "$LOG"; return 1
    fi
    sleep 5
  done
  vermelho "o vLLM não ficou pronto em 15 min — ver $LOG"; return 1
}

case "${1:---start}" in
  --status)
    if pronto; then verde "VLM ok ($MODEL em $BASE)"; else vermelho "VLM EM BAIXO ($BASE)"; exit 1; fi ;;
  --stop)
    p="$(pid_servidor)"
    if [ -z "$p" ]; then amarelo "não há vLLM a correr para $MODEL no porto $PORT"; exit 0; fi
    kill "$p" && sleep 5
    pkill -f "vllm serve $MODEL .*--port $PORT" 2>/dev/null
    verde "VLM parado" ;;
  --start|*)
    arranca ;;
esac
