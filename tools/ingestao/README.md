# Ingestão e indexação de um museu

Como pôr um museu no índice que o assistente pesquisa: do inventário até ao
OpenSearch. O trabalho é feito por dois projetos irmãos — `../raiz_scraper`
(dados e imagens) e `../patrimonio360_indexer` (embeddings e indexação) — e
estes guiões só os chamam pela ordem certa, com os argumentos certos, e
guardam os registos.

Até aqui os comandos não estavam escritos em lado nenhum: tiveram de ser
reconstruídos a partir do stdout dos registos da Fase 2 (`.cache/phase2_logs/`)
e das docstrings dos scripts.

## Antes de começar

```bash
conda activate visitas_virtuais
cd ~/Projects/AMALIA/patrimonio360-backend
python tools/ingestao/verificar_ambiente.py --museu "Museu Nacional do Azulejo"
```

A verificação falha se houver algum impedimento e avisa sobre o resto. Vale a
pena ler o que ela diz sobre estes quatro pontos:

- **GPU.** Os embedders (Qwen3-Embedding-8B para texto e Qwen3-VL-Embedding-8B
  para imagem) pedem cerca de 30 GB. Se o vLLM da AMÁLIA e o backend estiverem
  a correr, não cabem: parar o backend (`pkill -f "uvicorn app.main:app"`) e o
  LLM (`deploy/llm/amalia_llm.sh --stop`) antes de indexar.
- **Museu novo.** O `museum_id` não vem dos dados: sai de `MUSEUM_NAME_TO_ID`,
  em `patrimonio360_indexer/scripts/ingest_museum_opensearch.py`, que hoje
  conhece `mnt`, `mnsr`, `mnaz`, `mj`, `cmag`, `mnc` e `mnaa`. Um museu que não
  esteja lá fica com `museum_id` vazio e nunca aparece nas respostas.
- **Disco.** As imagens de um museu ocupam alguns GB (o `output/` já vai em
  9 GB) e a partição está quase cheia.
- **Origem dos dados.** O export relacional está em `../RaizAMALIA` (os Excels
  mais o XML dos móveis) e é dele que sai o `objetos.ndjson`. Já `../Dados_MJ`
  (imagens do museu manual) **não existe** nesta máquina: o `--fonte excel` não
  se pode correr de novo, só reaproveitar o que está em `raiz_scraper/output`.

## Fazer tudo de seguida

```bash
tools/ingestao/ingerir_museu.sh --museu "Museu Nacional do Azulejo" --slug mnaz --passos tudo --executar
```

`--passos tudo` é `raiz,dados,indexar,pos`: reconstrói o `objetos.ndjson` a
partir dos Excels do RAIZ, descarrega as imagens do museu, gera os embeddings,
indexa e sincroniza o `in_tour`. Sem `--passos`, corre `dados,indexar,pos` e
reaproveita o `objetos.ndjson` que já existe — que é o caso normal, porque o
passo `raiz` é global aos quatro museus e só faz falta quando o export do RAIZ
muda.

**Por omissão simula**, como o próprio indexador: mostra o que faria sem
escrever. Só escreve com `--executar`. Os passos `raiz` e `dados` não sabem
simular (o `build_raiz_index.py` e o scraper escrevem sempre no disco), por
isso em simulação mostram o comando e não correm.

## Ou passo a passo

```bash
# 1. só os dados (imagens + NDJSON), que é o passo demorado
tools/ingestao/ingerir_museu.sh --museu "…" --slug mnaz --passos dados

# 2. mais tarde, com a GPU livre, os embeddings e o índice
tools/ingestao/ingerir_museu.sh --museu "…" --slug mnaz --passos indexar,pos --executar
```

| Passo | O que faz | Onde |
|---|---|---|
| `raiz` | os Excels de `../RaizAMALIA` → `objetos.ndjson` global e índices de entidades | `raiz_scraper` |
| `dados` | imagens e `objetos_com_imagens.ndjson` do museu, mais `autores/conjuntos/exposicoes.ndjson` | `raiz_scraper` |
| `indexar` | embeddings e ingestão de artefactos, imagens e entidades | `patrimonio360_indexer` |
| `entidades` | só autores/conjuntos/exposições — a segunda metade do `indexar`, para repetir sem refazer a parte cara | `patrimonio360_indexer` |
| `pos` | sincroniza o `in_tour` com os POIs e conta o que ficou no índice | este repositório |

Opções úteis: `--fonte excel` (museu manual, a partir do Excel do inventário,
com `--xlsx` e `--dados-mj`), `--limite N` (ensaio com poucos objetos),
`--workers N` (descargas em paralelo, por omissão 6).

O `--limite` não chega ao passo `raiz` de propósito: o `objetos.ndjson` é
partilhado pelos quatro museus e um ensaio truncado substituiria o bom.

Cada execução deixa os registos em `tools/ingestao/registos/<data>-<museu>/`,
um ficheiro por passo. Em caso de falha, o guião pára e mostra as últimas
linhas do registo do passo que falhou.

## O que corre por baixo

```
../RaizAMALIA ──build_raiz_index.py──► objetos.ndjson (4 museus)
                                                                   │
             scrape_raiz_images.py ──► objetos_com_imagens.ndjson  ─┤
             build_museu_entidades.py► autores/conjuntos/exposicoes │
                                                                   ▼
                        ingest_museum_opensearch.py --artifacts-only --embed-artifacts
                        ingest_museum_opensearch.py --images-only    --embed-images
                        ingest_museum_opensearch.py --entity-source …
                                                                   ▼
                        cultural_heritage_{artifacts,images,…}_v4 no OpenSearch
                                                                   ▼
                        tools/poi_mapping/marcar_in_tour.py  (peças com hotspot)
```

Notas que poupam tempo:

- O `--source` é sempre explícito. O valor por omissão do indexador aponta
  para um `.npz` antigo, com campos em português, que ele já não entende: daria
  documentos vazios sem dar erro.
- O índice é criado sozinho se não existir, a partir de
  `patrimonio360_indexer/mappings/museum_*_v4.json`. Se já existir com outra
  dimensão de vetores, o indexador recusa (as dimensões kNN são imutáveis).
- O `--entity-index` é sempre explícito, pelos nomes que estão no `.env` do
  backend (na raiz deste repositório). Sem ele, o indexador escreve nos índices de entidades
  antigos (`cultural_heritage_authors`, de 2560 dimensões) em vez dos `_v4`
  que o assistente lê, e a ingestão morre a meio com «mapping dimension !=
  embedding dimension» — depois de artefactos e imagens já terem entrado.
  Aconteceu na primeira ingestão do CMAG, a 2026-09-20; a verificação prévia
  passou a comparar as dimensões dos cinco índices antes de começar.
- As credenciais do OpenSearch vêm do `patrimonio360_indexer/.env`
  (`OPENSEARCH_URL`, `OPENSEARCH_USER`, `OPENSEARCH_PASSWORD`), não de
  argumentos. O `TOURS_DIR` desse ficheiro é o que liga o `in_tour` aos POIs.
- As imagens são referidas por caminho relativo ao `raiz_scraper/output`, que
  o guião passa como `--output-root`.

## Depois de indexar

1. `python tools/ingestao/verificar_ambiente.py --museu "…" --contagens mnaz`
   — quantos documentos ficaram e quantos estão marcados como na visita.
2. Reiniciar o backend, se o tiveres parado: `deploy/amalia_online.sh`.
3. Perguntar alguma coisa ao assistente nesse museu e confirmar que os
   resultados trazem peças com hotspot.

## Ficheiros

- `ingerir_museu.sh` — o orquestrador descrito acima.
- `verificar_ambiente.py` — verificação prévia e contagens finais.
- `registos/` — registos de cada execução (não versionados).

Ver também `tools/poi_mapping/` para o que liga as peças à visita: extração dos
POIs, validação dos números de inventário, sincronização do `in_tour`
(`marcar_in_tour.py`) e marcação manual de uma lista de números
(`definir_in_tour.py`).
