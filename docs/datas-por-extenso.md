# Datas das fichas por extenso (2026-11-17)

Uma flag nova, desligada por omissão. Não está ligada em produção.

## O problema

O export do RAIZ escreve os séculos como números: «16» é o século XVI, «18–19» os séculos XVIII a XIX, «-5» o século V a.C.
O campo `date_or_period` do índice guarda esse texto, e era ele que aparecia no cartão e que os modelos liam («Data: 16»).
No MNAA são 645 fichas (486 com um século, 159 com um intervalo de séculos); nos outros museus o formato é o mesmo.

Apareceu com «Qual é a peça mais antiga?»: o cartão da Píxide (940 Our) mostrava «-15».
Essa ficha tem ainda a data errada na origem (ver `patrimonio360_indexer`, PR #3); as duas coisas são independentes.

## O que mudou

| Flag | O que faz | Onde |
|---|---|---|
| `CHAT_READABLE_DATES` | O texto da data no formato do export é refeito a partir de `start_year`/`end_year`: «século XVI», «séculos XVIII–XIX», «século V a.C.», «590–580 a.C.» | `readable_catalogue_date` em `catalogue_analytics.py`; `_catalogue_date_text` em `chat_service.py` |

- Aplica-se aos cartões da resposta (`date_or_period` de `artifact_results` e das peças relacionadas) e aos cartões compactos dos prompts.
- Texto livre fica como está («Lisboa, c. 1715-1730», «Séc. XVI»), tal como as fichas sem anos no índice.
- O rótulo é sempre em português, como o título e a categoria da ficha.
- O frontend não muda: já mostra o `date_or_period` que recebe.

É uma regra e não uma decisão do modelo porque é uma conversão de formato com uma só resposta certa, sobre anos que o indexador já normalizou.

Testes: `tests/test_readable_dates.py` (14 testes). A suíte completa passa com 639.

## Como foi medido

A redação do que o Qwen lê é sensível, por isso a sessão da checklist de 2026-11-16 (115 pedidos) foi repetida fora da produção,
com a configuração de produção: uma vez sem a flag e duas com ela.

- Sem a flag, os cartões são os da medição do Bloco D nos 115 pedidos.
- Com a flag, as duas execuções dão os mesmos cartões nos 115 pedidos.
- Dos 628 cartões mostrados, 134 tinham a data em número de século; com a flag, nenhum.
- Os cartões escolhidos são os mesmos em 112 dos 115 pedidos.
- Das 49 perguntas «OK» da checklist, 48 ficam iguais.
- Tempo da sessão: 491 s sem a flag, 499 s e 495 s com ela.

Os três pedidos que mudam:

| Pergunta | Sem a flag | Com a flag |
|---|---|---|
| 69 «E onde estão na visita?» (as peças de prata) | 6 cartões: três de prata e três de ouro | 2 cartões, os dois de prata; fica de fora a Custódia 1 Our, também de prata |
| 72 «E cadeiras?» | 8 cadeiras, seis delas do mesmo conjunto | 8 cadeiras, todas diferentes |
| 81 «Show me Japanese folding screens.» | 6 biombos e uma caixa de alimentos | os 6 biombos |

## Para ligar

Juntar ao `.env` e reiniciar o backend:

```
CHAT_READABLE_DATES=true
```
