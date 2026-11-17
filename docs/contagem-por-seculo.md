# Contagem por século pelo campo do índice (2026-11-17)

Uma flag nova, desligada por omissão. Não está ligada em produção.

## O problema

«De que século é a maioria das peças?» contava com um histograma de `start_year` em intervalos de cem anos a começar em 1
(1701–1800). Depois de Cristo está certo. Antes de Cristo os intervalos ficam …, -199 a -100, -99 a 0, e o século I a.C. vai
de -100 a -1: o ano -100 caía no século II a.C. e o -500 no VI a.C.

## O que mudou

| Flag | O que faz | Onde |
|---|---|---|
| `CHAT_CATALOGUE_ANALYTICS_CENTURY_FIELD` | A contagem por século agrega pelo campo `start_century` das fichas (negativo antes de Cristo) | `_aggregate_artifacts_sync` em `opensearch_client.py`; a leitura das chaves em `chat_service.py` |

Precisa dos campos do século no índice, que estão no `cultural_heritage_artifacts_v4` desde 2026-11-17
(`patrimonio360_indexer`, `atualizar_datas_v4.py`). Uma ficha com `start_year` e sem `start_century` não entra na lista por século.

Testes: `tests/test_century_field_analytics.py` (5 testes). A suíte completa passa com 644.

## Como foi medido

A agregação corrida no índice de produção, só em leitura, sem e com a flag, nos sete museus.

- Depois de Cristo, as contagens são iguais em todos os museus, e a soma por século é igual ao número de fichas com data.
- Antes de Cristo:

| Museu | Sem a flag | Com a flag |
|---|---|---|
| MNAA | VI a.C. 2, II a.C. 1 | VI a.C. 1, V a.C. 1, I a.C. 1 |
| Soares dos Reis | XVI a.C. 1, XIII a.C. 1, VIII a.C. 2, VII a.C. 6, VI a.C. 1, V a.C. 15, IV a.C. 7 | XV a.C. 1, XII a.C. 1, VII a.C. 2, VI a.C. 6, V a.C. 1, IV a.C. 16, III a.C. 6 |
| Conímbriga | IX a.C. 1, VIII a.C. 8, VII a.C. 44, VI a.C. 1, V a.C. 4, IV a.C. 1, III a.C. 4, II a.C. 564, I a.C. 25 | VIII a.C. 1, VII a.C. 8, VI a.C. 44, V a.C. 1, IV a.C. 4, III a.C. 1, II a.C. 11, I a.C. 582 |

Em Conímbriga, 553 fichas que começam no século I a.C. (quase todas datadas «do século I a.C. ao V d.C.») estavam a ser
contadas no século II a.C.

## Por resolver: «século XVIII» na pergunta

Esta flag não mexe no filtro por século das perguntas («Quantas esculturas do século XVIII há?», «pinturas do século XV»).
Esse filtro procura entre 1700 e 1799, e no índice o século XVIII vai de 1701 a 1800. As fichas datadas só por século acabam
em 1700 ou 1800 e entram no século seguinte:

| MNAA, todas as fichas | Como está (X00–X99) | Pela convenção do índice (X01–X00) |
|---|---|---|
| século XV | 212 | 449 |
| século XVI | 1603 | 1278 |
| século XVIII | 1178 | 1010 |
| século XIX | 794 | 428 |

Trocar só os anos do filtro não chega. Um intervalo que começa num ano redondo («1900–1910»; no Museu do Traje há 444
fichas que começam em 1900 e continuam) passaria a contar no século XIX, e os campos do século dão-lhe hoje «séculos
XIX–XX». Antes de mexer no filtro é preciso decidir duas coisas: se um intervalo que começa em 1900 pertence ao século XX,
e se uma peça datada só «1500» (254 fichas no MNAA) é do século XV ou do XVI.
