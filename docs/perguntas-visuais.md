# Perguntas pelo aspeto: «estátuas brancas», «pinturas com barcos» (2026-11-19)

Cinco flags novas, todas desligadas por omissão. Em produção ficaram ligadas no mesmo dia. O frontend não muda.

| Flag | O que faz |
|---|---|
| `CAPTION_TEXT_UNCOVERED_MUSEUM_T2I` | Um museu sem nenhuma legenda no índice salta o ramo das legendas e usa o texto→imagem |
| `CAPTION_TEXT_QUERY_INSTRUCTION` | A pergunta vai às legendas com a instrução «query» do modelo de embeddings |
| `CHAT_VISUAL_CARD_FACTS` | Os cartões levam uma linha «Aspeto na fotografia» com o que a legenda diz da peça |
| `CHAT_VISUAL_CARD_CANDIDATES` | Quantas peças que só as legendas encontraram são julgadas pelo aspeto e juntas aos cartões (produção: 20) |
| `CHAT_VISUAL_INTENT_MODEL` | O modelo de pré-processamento diz se o pedido é visual quando as regras não veem sinal |

`CAPTION_TEXT_INDEX` passou a aceitar vários índices separados por vírgulas. O MNAA foi legendado neste dia e tem índice
próprio: `p360_legendas_mnaa_caption8b` (3632 peças, legenda_v6, Qwen3-VL-8B, 25 minutos, sem erros).

## O problema

«encontra estatuas brancas» no MNAA respondia «não encontrei», com zero cartões. Três causas encadeadas:

1. **O ramo visual não corria no MNAA.** O índice das legendas só tinha o MNT, o MNAz, o MNSR e o MJ. Para o MNAA
   devolvia lista vazia, e em `CAPTION_TEXT_MODE=replace` uma lista vazia conta como resposta: o texto→imagem não corria.
   O CMAG estava na mesma situação.
2. **As fichas não dizem a cor.** Das 299 esculturas do MNAA, 293 têm o material vazio.
3. **Quem escolhe os cartões só via as fichas.** O modelo de pré-processamento rejeitava os 15 candidatos por nenhum
   dizer «branco». E as peças que só o ramo visual encontra ficam, na fusão, atrás de todas as do texto (pesos 1,0 e
   0,7): nunca chegavam a ser candidatas.

Só o texto→imagem não chegava: para «estátuas brancas» trouxe sobretudo fólios de iluminuras (páginas brancas).

Havia ainda um defeito independente: a resposta da pré-seleção tinha 120 tokens de limite, que só chegam para 20
rótulos. Uma escolha maior saía cortada e contava como falha. O limite passou a crescer com o número de cartões.

## O que mudou

Numa pesquisa em que o ramo das legendas trouxe peças:

- **Os cartões levam o aspeto.** Para os candidatos da pré-seleção vai-se buscar a legenda estruturada de cada peça, e o
  cartão ganha a linha «Aspeto na fotografia: objeto…; materiais…; cores principais…; outras cores…; motivos…; forma…».
  A pré-seleção e a AMÁLIA veem a mesma linha.
- **As peças que só as legendas encontraram são julgadas à parte.** As primeiras N, pela ordem das legendas, vão numa
  segunda pergunta ao modelo de pré-processamento (`build_visual_pick_prompt`), com uma linha por peça (título e
  aspeto) e resposta «sim» ou «não» a cada uma. Corre em paralelo com a pré-seleção. As aceites juntam-se aos cartões
  escolhidos, depois dos da pesquisa de texto, até encher a página.
- **O modelo decide se o pedido é visual.** As regras de `retrieval/visual_intent.py` continuam a mandar quando veem
  sinal (cores, formas, «mostra-me»). Quando não veem, pergunta-se ao modelo (`build_visual_request_prompt`): só a
  resposta A liga o ramo visual.

A ordenação da pesquisa e os pesos da fusão não mudam.

## Medição

### O modelo a dizer se o pedido é visual

`tests/visual_intent_cases.py`, 85 mensagens do MNAA × 3 execuções (`scripts/medir_intencao_visual.py`):

| | Certos | Pelas regras, sozinhas |
|---|---|---|
| 40 pedidos pelo aspeto | 38 | 1 |
| 45 pedidos que não são visuais | 45 | 44 («Show me the Belém Monstrance» é apanhado pela regra «show me») |

As três execuções deram sempre a mesma letra. Falham «esculturas em madeira» e «há alguma coisa com elefantes?».
É a primeira redação; os casos não serviram para a afinar. Em produção: mediana de 69 ms, máximo de 99 ms.

### A escolha pelo aspeto

`tests/visual_pick_cases.json` tem 20 pedidos visuais com as listas reais do ramo das legendas (390 peças) e o que o
modelo aceitou (`scripts/medir_escolha_visual.py`). Não há resposta certa anotada; foram julgadas à vista.

| Redação | «estátuas totalmente brancas» | «esculturas em mármore branco» | «peças douradas» |
|---|---|---|---|
| Juntar as peças aos cartões da pré-seleção (35 cartões) | nenhuma, ou estátuas castanhas aceites | — | — |
| Pedir a lista das que correspondem | 3 de 22, as três todas brancas | 19 de 22, com madeira e marfim | 0 de 17 |
| «sim» ou «não» por peça (a que ficou) | 3 de 22, as mesmas | 6 de 22, todas de pedra branca | 8 de 17, todas de ouro |

Com 35 cartões inteiros o prompt chegava aos 11 230 tokens, contra um limite de 12 288 do modelo.
Em três corridas seguidas, 19 dos 20 pedidos deram a mesma escolha; o que mudou foi «peças de vidro transparente»
(11 e 6 peças). Noutra corrida «tecidos vermelhos com fios de ouro» deu 3 e depois 15. Demora 1,3 a 1,5 s de mediana.

### De ponta a ponta, no MNAA

36 perguntas contra o backend em produção, antes (sem legendas do MNAA, só com o texto→imagem) e depois:

| | Antes | Depois |
|---|---|---|
| 24 perguntas visuais sem nenhum cartão | 4 | 0 |
| 12 perguntas de controlo com os mesmos cartões | — | 12 |
| Falhas da pré-seleção | — | 0 |

As 12 de controlo são pedidos por nome, autor, época e conversa («Painéis de São Vicente», «Há obras de Bosch?»,
«pinturas do século XV», «Olá»…).

| Pergunta | Antes | Depois |
|---|---|---|
| encontra estatuas brancas | 0 | 3: São Roque, Santa Escolástica, São Vicente (branco como primeira cor) |
| encontra estatuas totalmente brancas | 0 | 2: Danaide e Apolo, de mármore |
| esculturas em mármore branco | 4, duas erradas | 4: Efebo, Apolo e Danaide, de mármore, e um Apolo de cerâmica |
| retratos de senhoras com colares de pérolas | 0 | 5 |
| peças de vidro transparente | 0 | 8 |
| quadros escuros com uma vela acesa | 1 | 5 |
| pinturas com barcos | 3 | 7 (inclui um desenho, uma gravura e um biombo com barcos) |
| móveis com embutidos de madrepérola | 3 | 8 |

Tempo de resposta depois: mediana de 4,9 s nas visuais e de 2,7 s nas de controlo. A diferença é sobretudo a escolha
pelo aspeto. Os tempos de antes foram medidos com o modelo de visão ocupado a legendar e não servem de comparação.

Nos outros museus fiz só uma verificação de fumo, sete perguntas (MNT, MNAz, MNSR, CMAG): todas responderam, sem erros.

## Limites

- **«estátuas brancas» não é exaustiva.** A pesquisa nas legendas ordena mais por «estátua» do que pela cor, e só as 20
  primeiras são julgadas. A Danaide estava em 23.º; o Efebo e o Apolo não vinham entre as devolvidas.
- **O modelo de 8B é permissivo no tipo de peça** («pinturas com barcos» aceita um biombo) e varia entre corridas em
  1 ou 2 pedidos em 20.
- **Pedidos só pelo material** («peças em marfim», «esculturas em madeira», «objetos de prata») não são lidos como
  visuais e ficam como estavam.
- **`CAPTION_TEXT_QUERY_INSTRUCTION` muda a ordem das legendas em todos os museus** e não foi medida no benchmark de
  2026-09-30 (a variante medida ia sem instrução). No MNAA, sem ela, as legendas traziam uma gravura de uma secretária
  e uma forma de bolos para «estátuas brancas»; com ela, só esculturas.
- **O CMAG continua sem legendas** e usa o texto→imagem.
- **Mais cerca de 1,5 s** nas perguntas visuais.

## Testes

`tests/test_visual_card_facts.py` (15 testes) e seis novos em `tests/test_caption_text_branch.py`. Suíte: 744 a passar.

## Para ligar

```
CAPTION_TEXT_INDEX=p360exp-1a2bf259341a-caption8b,p360_legendas_mnaa_caption8b
CAPTION_TEXT_UNCOVERED_MUSEUM_T2I=true
CAPTION_TEXT_QUERY_INSTRUCTION=true
CHAT_VISUAL_CARD_FACTS=true
CHAT_VISUAL_CARD_CANDIDATES=20
CHAT_VISUAL_INTENT_MODEL=true
```

Precisa de `CHAT_PRESELECT_CARDS` e de `MULTIMODAL_RETRIEVAL_MODE=intent`, que já estavam ligadas. Para legendar outro
museu: `scripts/legendar.py` e `scripts/indexar.py` do `patrimonio360_caption_indexer`, com `--origem vivo`, e
acrescentar o índice a `CAPTION_TEXT_INDEX`.
