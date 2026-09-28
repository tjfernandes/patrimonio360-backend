#!/bin/bash
# Ingestão de um museu, de ponta a ponta: dados -> índice -> visita.
#
# Junta num só comando o que estava espalhado por dois projetos irmãos
# (../raiz_scraper e ../patrimonio360_indexer) e nunca tinha sido escrito em
# lado nenhum: os comandos da Fase 2 só existiam no stdout dos logs.
#
# Passos (podem correr todos de seguida ou um de cada vez, com --passos):
#   raiz      build_raiz_index.py: os Excels de ../RaizAMALIA -> objetos.ndjson global
#   dados     raiz_scraper: imagens + NDJSON do museu (ou o Excel, com --fonte excel)
#   indexar   patrimonio360_indexer: embeddings e ingestão de artefactos, imagens e entidades
#   entidades só autores/conjuntos/exposições (a segunda metade do «indexar»)
#   pos       marcar_in_tour.py (peças da visita) e contagens finais
#
# «--passos tudo» é atalho para raiz,dados,indexar,pos. O passo «raiz» não entra
# por omissão: é global (reconstrói o ndjson dos quatro museus a partir do
# export relacional) e só faz falta quando o export do RAIZ muda.
#
# Por omissão SIMULA (o indexador também é dry-run por omissão): mostra o que
# faria sem escrever no índice. Só escreve com --executar.
#
#   tools/ingestao/ingerir_museu.sh --museu "Museu Nacional do Azulejo" --slug mnaz
#   tools/ingestao/ingerir_museu.sh --museu "…" --slug mnaz --executar
#   tools/ingestao/ingerir_museu.sh --museu "…" --slug mnaz --passos tudo --executar
#   tools/ingestao/ingerir_museu.sh --museu "…" --slug mnaz --passos dados
#   tools/ingestao/ingerir_museu.sh --museu "…" --slug mnaz --passos indexar,pos --executar
#
# Antes de correr: conda activate visitas_virtuais
# Ler o README.md desta pasta: há impedimentos conhecidos (GPU ocupada pelo
# vLLM, ../RaizAMALIA e ../Dados_MJ em falta, museu novo tem de ser declarado
# no indexador).
set -euo pipefail

AQUI="$(cd "$(dirname "$0")" && pwd)"
BACKEND="$(cd "$AQUI/../.." && pwd)"   # raiz do repositório do backend
# os projetos irmãos vivem ao lado deste repositório (~/Projects/AMALIA/);
# noutro sítio, indicar RAIZ_SCRAPER_DIR e INDEXER_DIR
RAIZ="${RAIZ_SCRAPER_DIR:-$(cd "$BACKEND/.." && pwd)/raiz_scraper}"
INDEXER="${INDEXER_DIR:-$(cd "$BACKEND/.." && pwd)/patrimonio360_indexer}"
SAIDA="$RAIZ/output"

MUSEU=""; SLUG=""; FONTE="raiz"; PASSOS="dados,indexar,pos"; EXECUTAR=0
XLSX=""; DADOS_MJ=""; LIMITE=""; TRABALHADORES=6

while [ $# -gt 0 ]; do
  case "$1" in
    --museu) MUSEU="$2"; shift 2 ;;
    --slug) SLUG="$2"; shift 2 ;;
    --fonte) FONTE="$2"; shift 2 ;;
    --passos) PASSOS="$2"; shift 2 ;;
    --executar) EXECUTAR=1; shift ;;
    --xlsx) XLSX="$2"; shift 2 ;;
    --dados-mj) DADOS_MJ="$2"; shift 2 ;;
    --limite) LIMITE="$2"; shift 2 ;;
    --workers) TRABALHADORES="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "argumento desconhecido: $1"; exit 2 ;;
  esac
done
[ -n "$MUSEU" ] || { echo "falta --museu \"Nome do Museu\""; exit 2; }

[ "$PASSOS" = "tudo" ] && PASSOS="raiz,dados,indexar,pos"
tem_passo() { echo ",$PASSOS," | grep -q ",$1,"; }
# Mesma regra do slug() do raiz_scraper e do tipo_slug() do indexador: tira os
# acentos e junta as palavras por _, mas a pontuação fica. «Casa-Museu Dr.
# Anastácio Gonçalves» -> casa-museu_dr._anastacio_goncalves, que é o nome da
# pasta que o scraper escreve e a chave do MUSEUM_NAME_TO_ID.
PASTA_MUSEU=$(python3 -c "
import sys, unicodedata
nome = unicodedata.normalize('NFKD', sys.argv[1])
print('_'.join(''.join(c for c in nome if not unicodedata.combining(c)).lower().split()))" "$MUSEU")
DIR_MUSEU="$SAIDA/raiz_index/museus/$PASTA_MUSEU"
NDJSON="$DIR_MUSEU/objetos_com_imagens.ndjson"
REGISTOS="$AQUI/registos/$(date +%Y%m%d-%H%M%S)-$PASTA_MUSEU"
mkdir -p "$REGISTOS"

if [ "$EXECUTAR" = "1" ]; then MODO="--ingest"; SIMULAR=""; else MODO="--dry-run"; SIMULAR="--simular"; fi
echo "museu    : $MUSEU  (pasta $PASTA_MUSEU${SLUG:+, museum_id $SLUG})"
echo "passos   : $PASSOS"
echo "modo     : $([ "$EXECUTAR" = 1 ] && echo 'A ESCREVER no índice' || echo 'simulação (usar --executar para escrever)')"
echo "registos : $REGISTOS"

indice_entidades() {  # indice_entidades <autores|conjuntos|exposicoes>
  # O indexador, sem --entity-index, escreve em cultural_heritage_authors/_sets/
  # _exhibitions — os índices antigos, de 2560 dimensões. O assistente lê os _v4,
  # de 4096, e a ingestão rebenta com «mapping dimension != embedding dimension».
  # O nome vem do .env do backend, que é quem manda no que o assistente consulta.
  local chave omissao valor
  case "$1" in
    autores)    chave=OPENSEARCH_INDEX_AUTOR;     omissao=cultural_heritage_authors_v4 ;;
    conjuntos)  chave=OPENSEARCH_INDEX_CONJUNTO;  omissao=cultural_heritage_sets_v4 ;;
    exposicoes) chave=OPENSEARCH_INDEX_EXPOSICAO; omissao=cultural_heritage_exhibitions_v4 ;;
    *) echo "tipo de entidade desconhecido: $1" >&2; return 1 ;;
  esac
  valor=$(sed -n "s/^${chave}=//p" "$BACKEND/.env" 2>/dev/null | tail -1 | tr -d "\"' " )
  echo "${valor:-$omissao}"
}

correr() {  # correr <nome-do-registo> <comando...>
  local nome="$1"; shift
  # %q para a linha ficar copiável: sem isto o nome do museu sai sem aspas
  echo; echo "--- $nome"; echo "    $(printf '%q ' "$@")"
  # O raiz_scraper e o build_raiz_index não têm modo de simulação: escrevem
  # sempre no disco. Em simulação mostra-se o comando e não se corre — só os
  # passos que sabem simular (indexador, marcar_in_tour) é que correm mesmo.
  if [ "$EXECUTAR" != "1" ] && [ "${NAO_SIMULAVEL:-0}" = "1" ]; then
    echo "    (simulação: não corri — este passo escreveria no disco)"
    return 0
  fi
  if ! "$@" > "$REGISTOS/$nome.log" 2>&1; then
    echo "    FALHOU — ver $REGISTOS/$nome.log"; tail -15 "$REGISTOS/$nome.log" | sed 's/^/    /'
    exit 1
  fi
  tail -3 "$REGISTOS/$nome.log" | sed 's/^/    /'
}

echo; echo "=== verificação prévia"
python3 "$AQUI/verificar_ambiente.py" --museu "$MUSEU" --fonte "$FONTE" || {
  echo "corrigir os impedimentos acima antes de continuar"; exit 1; }

# ----------------------------------------------------------------- raiz
if tem_passo raiz; then
  NAO_SIMULAVEL=1
  # Global aos quatro museus: lê os Excels de ../RaizAMALIA (o Movel vem do XML,
  # porque o xlsx tem ~2500 linhas partidas) e escreve o objetos.ndjson e os
  # índices de autores/conjuntos/exposições que o passo seguinte filtra por museu.
  # Sem --limite de propósito: o ficheiro é partilhado pelos quatro museus e um
  # ensaio truncado substituiria o bom (o script não tem modo de simulação).
  (cd "$RAIZ" && correr "0-raiz-index" python3 build_raiz_index.py)
  GLOBAL="$SAIDA/raiz_index/objetos.ndjson"
  if [ "$EXECUTAR" = "1" ]; then
    [ -s "$GLOBAL" ] || { echo "sem $GLOBAL no fim do passo «raiz»"; exit 1; }
    echo "    $(wc -l < "$GLOBAL") objetos em $GLOBAL (todos os museus)"
  fi
  NAO_SIMULAVEL=0
fi

# ---------------------------------------------------------------- dados
if tem_passo dados; then
  NAO_SIMULAVEL=1
  if [ "$FONTE" = "excel" ]; then
    # Museu manual: o Excel do inventário + imagens locais dão os 4 NDJSON de uma vez
    ARGS=(python3 build_mj_json.py --museu "$MUSEU")
    [ -n "$XLSX" ] && ARGS+=(--xlsx "$XLSX")
    [ -n "$DADOS_MJ" ] && ARGS+=(--dados-mj "$DADOS_MJ")
    (cd "$RAIZ" && correr "1-excel" "${ARGS[@]}")
  else
    # Museu do RAIZ: imagens a partir do export relacional, depois as entidades.
    # O objetos.ndjson global não se regenera aqui (precisa de ../RaizAMALIA).
    ARGS=(python3 scrape_raiz_images.py --museu "$MUSEU" --resume --workers "$TRABALHADORES")
    [ -n "$LIMITE" ] && ARGS+=(--limit "$LIMITE")
    (cd "$RAIZ" && correr "1-imagens" "${ARGS[@]}")
    (cd "$RAIZ" && correr "2-entidades" python3 build_museu_entidades.py --museu "$MUSEU")
  fi
  if [ "$EXECUTAR" = "1" ]; then
    [ -s "$NDJSON" ] || { echo "sem $NDJSON no fim do passo «dados»"; exit 1; }
    echo "    $(wc -l < "$NDJSON") objetos em $NDJSON"
  fi
  NAO_SIMULAVEL=0
fi

# -------------------------------------------------------------- indexar
if tem_passo indexar; then
  [ -s "$NDJSON" ] || { echo "falta $NDJSON: correr primeiro --passos dados"; exit 1; }
  COMUM=(python3 scripts/ingest_museum_opensearch.py --source "$NDJSON" --output-root "$SAIDA" "$MODO")
  [ -n "$LIMITE" ] && COMUM+=(--limit-artifacts "$LIMITE" --limit-images "$LIMITE")
  # O --source tem de ser sempre explícito: o default aponta para um NPZ antigo,
  # com campos em português, que o indexador de hoje já não entende.
  (cd "$INDEXER" && correr "3-artefactos" "${COMUM[@]}" --artifacts-only --embed-artifacts)
  (cd "$INDEXER" && correr "4-imagens" "${COMUM[@]}" --images-only --embed-images)
fi

# ------------------------------------------------------------ entidades
# Passo à parte para se poder repetir só esta metade: os artefactos e as
# imagens são a parte cara, e foi aqui que a primeira ingestão do CMAG parou.
if tem_passo indexar || tem_passo entidades; then
  for TIPO in autores conjuntos exposicoes; do
    FICHEIRO="$DIR_MUSEU/$TIPO.ndjson"
    if [ -s "$FICHEIRO" ]; then
      (cd "$INDEXER" && correr "5-$TIPO" python3 scripts/ingest_museum_opensearch.py \
        --entity-source "$FICHEIRO" --entity-index "$(indice_entidades "$TIPO")" \
        --output-root "$SAIDA" --embed-entities "$MODO")
    else
      echo "    (sem $TIPO.ndjson, salto)"
    fi
  done
fi

# ------------------------------------------------------------------ pos
if tem_passo pos; then
  if [ -n "$SLUG" ]; then
    # o in_tour é gravado na ingestão a partir dos POIs; isto apanha qualquer desvio
    correr "6-in-tour" python3 "$BACKEND/tools/poi_mapping/marcar_in_tour.py" "$SLUG" $SIMULAR
    # Só a secção das contagens, inteira. Um «tail -N» cortava a linha das peças
    # quando as contagens cresceram (as entidades acrescentaram três linhas) e
    # deixava no fim um «pronto a ingerir» que não faz sentido depois de ingerir.
    python3 "$AQUI/verificar_ambiente.py" --museu "$MUSEU" --fonte "$FONTE" --contagens "$SLUG" \
      | sed -n '/=== contagens/,/^$/p'
  else
    echo "    (sem --slug não sei o museum_id: salto o in_tour e as contagens)"
  fi
fi

echo; echo "concluído. Registos em $REGISTOS"
[ "$EXECUTAR" = "1" ] || echo "Isto foi uma simulação: repetir com --executar para escrever no índice."
