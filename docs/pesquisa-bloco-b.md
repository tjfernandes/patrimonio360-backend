# Pesquisa do chat — Bloco B (2026-11-17)

Segundo bloco do plano de correção da pesquisa do chat no MNAA: o que a pesquisa devolve antes de o Qwen escolher os cartões.
Três flags novas, desligadas por omissão. Nada disto está ligado em produção.

## O que mudou

| N.º | Flag | O que faz | Onde |
|---|---|---|---|
| F5 | `CHAT_RETRIEVAL_QUERY_INSTRUCTION` | A pergunta é embebida com a instrução «query» do Qwen3-Embedding. O índice foi embebido como documento e a pergunta também o era | `QwenTextEmbedder.embed_query`, `EmbeddingProvider.embed_text(as_query=True)` e a chamada em `_retrieve_context` |
| F4 | `CHAT_RETRIEVAL_LEXICAL_METADATA_FIELDS` | O ramo lexical procura também no autor, na categoria, no material e no centro de fabrico. A cláusula que exige todas as palavras deixa de as exigir no mesmo campo | `LEXICAL_METADATA_FIELDS` e `_build_query_body` em `opensearch_client.py` |
| F6 | `CHAT_RETRIEVAL_KNN_PREFILTER` | Os filtros (museu, época, visita) vão dentro do kNN: os 1000 vizinhos são procurados já dentro do museu | `_build_query_body` |

Não exigem reindexar nem alteram o índice. A F5 só toca a pesquisa de texto do chat; percursos e pesquisa por imagem ficam como estavam.

Testes: `tests/test_search_fixes_block_b.py` (15 testes). A suíte completa passa com 595 testes.

## Como foi medido

Tudo fora da produção, com o código deste ramo, o `.env` de produção lido sem alterações, embeddings em CPU e OpenSearch em leitura.

1. **Pesquisa isolada**, pelo caminho de produção (`_retrieve_context`): 23 perguntas do MNAA e as 48 perguntas de texto do benchmark congelado
   (30 de metadados, com juízos completos; 18 de aparência, com juízos humanos sobre os 5 primeiros de sistemas antigos).
   A reescrita do Qwen é pedida uma vez por pergunta e partilhada por todas as variantes.
2. **Sessão da checklist** de 2026-11-16 (115 pedidos) repetida de ponta a ponta: flags desligadas, F4+F6, e as três (duas vezes).

### Benchmark, perguntas de metadados (30, juízos completos)

| Variante | MRR | recall@8 | nDCG@10 | Com relevante nos 8 primeiros |
|---|---|---|---|---|
| Hoje | 0,383 | 0,332 | 0,313 | 18/30 |
| F4 | 0,549 | 0,613 | 0,515 | 25/30 |
| F6 | 0,376 | 0,353 | 0,308 | 18/30 |
| F4+F6 | 0,594 | 0,643 | 0,558 | 25/30 |
| F5 | 0,502 | 0,466 | 0,430 | 20/30 |
| As três | 0,654 | 0,635 | 0,598 | 23/30 |

### Benchmark, perguntas de aparência (18, juízos parciais)

| Variante | MRR | recall@8 | Sem juízo nos 8 primeiros | Relevantes entre os julgados |
|---|---|---|---|---|
| Hoje | 0,931 | 0,735 | 3,1 | 0,69 |
| F4+F6 | 0,903 | 0,735 | 3,1 | 0,69 |
| As três | 0,833 | 0,651 | 4,1 | 0,76 |

Com a F5 o recall desce, mas entram mais peças que ninguém julgou (de 3,1 para 4,1 em 8) e, entre as julgadas, a proporção de relevantes sobe.
Com estes juízos não se distingue perda real de peças novas por julgar. É o único resultado contra a F5 e fica por esclarecer.

### MNAA, sessão da checklist de ponta a ponta

| Pergunta | Flags desligadas | F4+F6 | As três |
|---|---|---|---|
| 35 «Há obras de Machado de Castro?» | 0 cartões | 4 esculturas dele | 7 das 8 esculturas dele |
| 11 «Quero ver o quadro do Rafael.» | 1 cartão errado (121 Min) | igual | 568 Pint, o quadro de Rafael |
| «Quero ver o quadro do Rafael Sanzio» | 0 cartões | 568 Pint | 568 Pint |
| 42 «Is the Dürer a national treasure?» | 0 cartões | 0 cartões | 828 Pint |
| 85 «paineis de sao vicente de fora» | 0 cartões | 0 cartões | 1 cartão (1416 Pint) |
| 18 «Mostra-me pinturas do século XV.» | 4 cartões (5 recolhidos) | 8 pinturas | 8 pinturas |
| «Encontra pinturas renascentistas» | 0 cartões | 0 cartões | 5 pinturas |
| 36 «Mostra-me pintores flamengos.» | 1 cartão errado (Contador) | igual | 3 pinturas |
| 28 «Peças vindas do Japão.» | 3 cartões | 4 | 8, com os biombos |
| 21 «Procuro porcelana chinesa.» | 3 cartões | 2 | 6 |
| 30 «Que obras de Gregório Lopes há?» | 5 cartões | 8 | 8 |
| 33 «Obras do Zurbarán.» | 6 pinturas dele | 5 dele e 3 gravuras sem relação | 7 dele |

As duas repetições com as três flags deram os mesmos cartões nos 115 pedidos.

Posição da peça esperada na pesquisa isolada: nos 8 primeiros em 16 de 20 perguntas hoje, 17 com F4+F6 e 17 com as três
(a medida conta uma peça por pergunta; em «Machado de Castro» a peça escolhida fica em 17.º mas 7 dos 8 primeiros são dele).

### O que piora

- **6 «Fala-me dos Painéis de São Vicente.»** Hoje 7 cartões, 6 dos quais painéis. Com F4+F6, 4 painéis. Com as três, 5 cartões: 3 painéis,
  a Veneração de S. Vicente e o São Vicente atado à coluna. A causa é a F6: com os vizinhos procurados dentro do museu entram outras peças
  de São Vicente, e a pesquisa só entrega 8 fichas ao Qwen.
- **89 «Qual é o quadro mais famoso?»** Passa de 1 cartão a 0 com as três. Já era «Parcial»; é das sugestões sem assunto (Bloco D).
- **Benchmark de aparência**, descrito acima.
- **Benchmark de metadados**: com a F5, três perguntas perdem o relevante dos 8 primeiros («traje e adereços do século XXI»,
  «obras de Achilles & C.ª», «Cerâmica de Vila Nova de Gaia do século XVIII») e uma ganha-o. O total com relevante nos 8 primeiros fica em 23, contra 25 sem a F5.

## O que foi medido e não entra

- **F4b, o Qwen extrair o autor para um campo próprio.** Medi o limite superior, com o autor dado sem erro, em 9 perguntas de autor:
  obras do autor nos 8 primeiros, somadas. Hoje 15; F4+F6 22; as três 28; as três mais o campo de autor 30.
  Ganha 2 em 30 e obriga a mexer no prompt da reescrita, que é sensível. Sem a F5 o ganho seria maior (de 22 para 27).
- **Outra redação da instrução do embedding.** Três redações próprias («perguntas de visitantes de um museu», «pergunta ou descrição de um objeto»,
  «registos de catálogo») contra a do modelo: nenhuma melhor. Com elas, «Painéis de São Vicente» desce para 13.º ou 14.º (9.º com a do modelo)
  e o benchmark de aparência desce mais.
- **A reescrita da pesquisa pelo Qwen** é estável: 27 de 28 perguntas deram a mesma resposta em 5 execuções. A exceção é «Há azulejos?»,
  que devolve vazio (com maiúscula, 5 em 5) ou «azulejos» (com minúscula, 4 em 5). Com a F6 isso deixa de dar zero resultados:
  o ramo semântico passa a ter candidatos do museu. Com as três flags os 8 primeiros são peças de cerâmica, que o Qwen rejeita:
  a pergunta continua sem cartões, e numa das duas repetições a resposta voltou a afirmar que o museu tem azulejaria.

## Por resolver

- **Só 8 fichas chegam ao Qwen.** Em perguntas com muitas peças certas (os Painéis), qualquer ruído nos 8 primeiros tira cartões.
  Entregar mais fichas à escolha de cartões é a correção natural; não foi medida.
- **85** fica com um cartão, e não com os painéis.
- **19 e 36** continuam a depender de dados que as fichas não têm (material das esculturas, escola dos pintores).
- **84 e 67** (seguimentos) são do Bloco C.

## Para ligar

Juntar ao `.env` e reiniciar o backend:

```
CHAT_RETRIEVAL_QUERY_INSTRUCTION=true
CHAT_RETRIEVAL_LEXICAL_METADATA_FIELDS=true
CHAT_RETRIEVAL_KNN_PREFILTER=true
```

As três ligam-se juntas. F4+F6 sem a F5 deixa 11, 42 e 85 como estão e mete gravuras sem relação em «Obras do Zurbarán».
