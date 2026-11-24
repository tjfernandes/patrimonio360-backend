#!/bin/bash
# Mostra os passos do RAG de cada pedido, lidos do log do backend
# (~/.amalia_backend.log, o que o deploy/amalia_online.sh escreve): o que se
# leu do pedido, para onde foi encaminhado, o que se pesquisou e o que voltou,
# que cartões ficaram, cada chamada ao modelo com o tempo que levou, e a resposta.
#
#   deploy/amalia_logs.sh                  os últimos 5 pedidos
#   deploy/amalia_logs.sh -f               em direto: cada passo aparece quando acontece
#   deploy/amalia_logs.sh -l               índice, uma linha por pedido
#   deploy/amalia_logs.sh --rid 2d0636     um pedido (basta o início do rid do índice)
#   deploy/amalia_logs.sh -c -n 1          o último, com prompts e respostas por extenso
#   deploy/amalia_logs.sh -e               só os que tiveram avisos ou erros
#   deploy/amalia_logs.sh --arranque       o que o backend carregou ao arrancar
#   deploy/amalia_logs.sh --help           todas as opções
#
# Só lê o log: não toca no backend nem precisa de o reiniciar. O texto dos
# prompts, das respostas do modelo e da resposta final só fica no log com
# BACKEND_RAG_DEBUG_LLM_IO=true no .env (ver tools/registos/README.md).

BACKEND="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
# qualquer python3 serve (só biblioteca padrão); o do conda fica de recurso
PYTHON="$(command -v python3 2>/dev/null || echo "$HOME/miniconda3/envs/visitas_virtuais/bin/python")"
exec "$PYTHON" "$BACKEND/tools/registos/amalia_logs.py" "$@"
