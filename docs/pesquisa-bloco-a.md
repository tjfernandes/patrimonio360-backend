# Pesquisa do chat — Bloco A (2026-11-16)

Primeiro bloco do plano de correção da pesquisa do chat no MNAA (`docs/plano-correcao-pesquisa-mnaa.md`).
Ramo `pesquisa-bloco-a`, criado a partir de `qwen-preprocessamento` (30beb35).
Nada disto está ligado em produção: as flags nascem desligadas e o `.env` não foi alterado.

## O que mudou

| N.º | Flag | O que faz | Onde |
|---|---|---|---|
| F1 | `CHAT_PRESELECT_EMPTY_IS_NONE` | A lista vazia do Qwen na escolha dos cartões passa a valer: zero cartões, sem «ver mais», regra `preselect_none` no registo. Erro, tempo esgotado e rótulos inválidos continuam em `preselect_failed` | `_preselect_cards` e a sua chamada em `chat_service.py` |
| F2 | `CHAT_NO_CARDS_ANSWER_PROMPT` | Numa pesquisa de texto sem nenhum cartão, a AMÁLIA recebe uma instrução curta de «não encontrei» em vez do prompt de pesquisa | `build_final_answer_prompt_v2` em `chat_prompts.py` |
| F3 | `CHAT_IN_TOUR_FROM_NAVIGATION` | No prompt da AMÁLIA, o cartão diz «Na visita virtual: sim» quando a peça tem alvo de navegação. Não escreve no índice, não muda os cartões do frontend nem os que o Qwen lê | `_artifact_results_as_prompt_docs` |
| F7 | sem flag (é um dado) | Alias «jeronimo» retirado do Mosteiro dos Jerónimos | `app/config/search_museums.json` e a lista por omissão |
| F9 | `CHAT_ROOM_BEFORE_ROUTE` | O nome completo de uma sala, sem mais palavras, ganha à regra dos percursos | `_room_request_precedes_route` |
| F10 | `CHAT_CATALOGUE_ANALYTICS_LLM_WORDING=false` (já existia) | As estatísticas saem com o texto do modelo fixo | só `.env` |
| F11 | `CHAT_STRIP_CARD_LABELS` | Retira da resposta os rótulos dos cartões copiados à letra | `_strip_card_labels` |
| F12 | `ROUTE_THEME_STRICT_TIME_MENTION` | Os séculos propostos pelo LLM do tema só entram quando o pedido fala de época | `merge_llm_facets` em `route_planning/theme.py` |

Testes: `tests/test_search_fixes_block_a.py` (31 testes). A suíte completa passa com 580 testes.

## Como foi medido

A sessão da checklist de 2026-11-16 (115 pedidos, com as mesmas conversas) foi repetida num processo à parte,
com o código deste ramo, o `.env` de produção lido sem alterações, embeddings em CPU, OpenSearch em leitura e
os dois modelos chamados como qualquer cliente. Uma vez com as flags desligadas e outra com as sete ligadas.
A produção não foi reiniciada.

### As 16 perguntas do bloco

| Pergunta | Flags desligadas | Flags ligadas |
|---|---|---|
| 93, 97, 100, 94, 98, 96, 92 | 8 cartões sem relação | 0 cartões, resposta de «não encontrei» ou de fora de âmbito |
| 27 «Há azulejos?» | inventa azulejos ou painéis em 3 de 3 | diz que não encontrou em 3 de 3 |
| 10 e 17 | dizem que a peça não está na visita, em 2 de 2 | nunca o dizem, em 4 de 4 |
| 6 | diz que não está na visita em 1 de 2 | nunca o diz, em 4 de 4 |
| 81 | não o disse em 2 execuções | nunca o diz, em 4 de 4 |
| 8 «Quem pintou o São Jerónimo?» | em produção ia ao Mosteiro dos Jerónimos | fica no MNAA, 828 Pint em 1.º, resposta com Dürer |
| 49 | «nove esculturas na zona da Sacristia» | «Há 99 peças…», o número do registo |
| 38 | sem rótulo nestas execuções | sem rótulo em 4 de 4 |
| 52 «Leva-me à Capela das Albertas.» | percurso de 19 paragens | resposta da sala, em 2 de 2 |

### As que passavam

Dos 49 pedidos da sessão que correspondem a perguntas com «OK» na checklist, 47 ficaram com o mesmo tipo de
resposta e os mesmos cartões. Os outros dois (65 e 68) ficaram com um cartão a menos. Repeti essas conversas duas vezes em
cada modo: a escolha do Qwen varia entre execuções com as flags desligadas (3 de 7 pedidos), por isso a
diferença é ruído e não efeito das flags.

### A instrução de zero cartões (F2)

Medida na AMÁLIA, 10 perguntas × 5 execuções por língua, temperatura 0,4. Ficou a terceira redação.

| | Prompt atual | Redação escolhida |
|---|---|---|
| PT, títulos inventados | 1/40 | 0/50 |
| PT, fala de «cartões» ou «instruções» | 17/40 | 5/50 (as 5 são a pergunta sobre o prompt de sistema) |
| PT, afirma que o museu tem o que não encontrou | não medido | 2/20 (a primeira redação: 5/20) |
| EN, títulos inventados | 4/40 | 0/50 |
| EN, fala de «cards» ou «instructions» | 28/40 | 5/50 |
| EN, horários ou preços inventados | não medido | 0/5 (a primeira redação inventava em 5/5) |

### A limpeza dos rótulos (F11)

Nas 282 respostas recolhidas, 6 traziam um rótulo. A forma mais comum é a citação entre aspas
(«Confirmado pelo cartão que indica "Tesouro Nacional: sim".»), não os parênteses. A limpeza trata as seis.

## O que fica por resolver

- **Com a F1, uma pesquisa falhada passa a dizer «não encontrei».** Acontece em «Há obras de Machado de Castro?» (35),
  «paineis de sao vicente de fora» (85), «Quero ver o quadro do Rafael Sanzio» e «Encontra pinturas renascentistas».
  O Qwen tem razão em rejeitar os 8 cartões, que não têm relação. A peça existe e a pesquisa não a devolve:
  é o Bloco B. Antes mostravam-se 8 cartões errados; agora diz-se que não há. Nenhuma das duas está certa.
- **«Is the Dürer a national treasure?» (42)** fica sem cartões mas a AMÁLIA recebe a nota dos Tesouros Nacionais
  e responde em português. É do Bloco B (pesquisa) e do Bloco C (língua).
- **Em inglês, «Are there any tiles?»** continua a responder «Yes, there are» em 5 de 5.
- **A resposta sem cartões remete para o site oficial** em cerca de 1 em cada 10 respostas que não são sobre horários.
- **Capela das Albertas (52)**: a resposta da sala diz que não há peças assinaladas e fala de «cartões abaixo»
  sem haver cartões. É a causa C16 do plano (peças por sala só dos hotspots), fora deste bloco.
- **Ruído do Qwen**: a mesma conversa, com as mesmas flags, deu escolhas de cartões diferentes em 3 de 7 pedidos.
  Qualquer medição de cartões precisa de pelo menos 3 execuções.

## Para ligar

Juntar ao `.env` e reiniciar o backend:

```
CHAT_PRESELECT_EMPTY_IS_NONE=true
CHAT_NO_CARDS_ANSWER_PROMPT=true
CHAT_IN_TOUR_FROM_NAVIGATION=true
CHAT_ROOM_BEFORE_ROUTE=true
CHAT_STRIP_CARD_LABELS=true
ROUTE_THEME_STRICT_TIME_MENTION=true
CHAT_CATALOGUE_ANALYTICS_LLM_WORDING=false
```

A F1 e a F2 ligam-se juntas: com a F1 sozinha, a AMÁLIA recebe o prompt de pesquisa sem cartões,
que é o caso em que inventa títulos.
