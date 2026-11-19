# A resposta fala de todos os cartões (2026-11-19)

Duas flags novas, desligadas por omissão: `CHAT_ANSWER_ALL_CARDS` e `CHAT_AUTHOR_ONLY_NEW_ANSWER`. Em produção
ficaram ligadas no mesmo dia, com `LLM_MAX_TOKENS_TEXT` a passar de 450 para 700.

## O problema

A instrução da resposta pedia «destaca até três peças». O visitante via sete cartões e lia um texto sobre três.
Numa pergunta por um autor era pior: o texto era escrito para os cartões da pesquisa geral e só depois os cartões
passavam a ser só do autor (`CHAT_AUTHOR_ONLY_RESULTS`), por isso podia dizer «encontrei duas obras» ao lado de sete
cartões.

## O que mudou

- **`CHAT_ANSWER_ALL_CARDS`.** A frase «destaca até três peças pelo título» dá lugar a «fala de todas as peças dos
  cartões, uma a uma e pela ordem em que vêm: uma linha por peça, com o título e uma só frase curta». O mesmo na
  instrução das perguntas pelo aspeto. O resto do prompt não muda.
- **A garantia.** A AMÁLIA não cumpre sempre. Depois de a resposta estar escrita, os cartões que o texto não nomeia
  (nem pelo título nem pelo n.º de inventário) vão numa última linha: «Nos cartões estão também: **Fé** (60 Esc),
  **Gratidão** (62 Esc).» Com um só cartão não se acrescenta nada.
- **`CHAT_AUTHOR_ONLY_NEW_ANSWER`.** Quando os cartões mudam por serem restringidos ao autor (saem os de outros,
  entram mais dele), a resposta escreve-se outra vez, com o mesmo prompt final, sobre os cartões que ficam. Se os
  cartões não mudaram, o texto fica.

## Medição

`scripts/medir_resposta_visual.py --todos x2`: 20 pedidos do MNAA com os cartões reais (14 pelo aspeto e 6 gerais, de
3 a 8 cartões cada), 2 respostas da AMÁLIA real por pedido, com `LLM_MAX_TOKENS_TEXT=700`.

| | Até três (como estava) | Todos os cartões |
|---|---|---|
| Respostas que falam de todos os cartões | 7 de 40 | 40 de 40 |
| Cartões mencionados | 147 de 260 | 260 de 260 |
| Respostas que precisaram da última linha | — | 10 de 40 |
| Cortadas pelo limite de tokens | 0 | 1 |
| Tamanho mediano | 1000 letras | 1305 letras |
| Tempo mediano (máximo) | 4,1 s (10,7 s) | 5,4 s (9,5 s) |

Só com a instrução, sem a última linha, a AMÁLIA mencionava 239 de 260 cartões (27 de 40 respostas completas). Com o
limite de 450 tokens, 17 de 40 respostas saíam cortadas; por isso o limite subiu.

## Limites

- **A resposta é mais comprida e demora mais** cerca de 1,3 s na mediana.
- **Com streaming, o texto muda no fim.** A última linha e a resposta reescrita para o autor chegam no `reply` do
  evento `result`, depois de os tokens da primeira versão já terem passado. O frontend tem de mostrar o `reply` final.
- **A resposta reescrita custa mais uma geração** da AMÁLIA nas perguntas por um autor em que os cartões mudam.
- Só a pesquisa de texto leva a última linha. A pesquisa por imagem e por modelo 3D, os percursos e as listas de
  Tesouros Nacionais ficam como estavam.

## Testes

`tests/test_visual_card_facts.py` (`TestAnswerAllCards`) e `tests/test_author_only_results.py`
(`TestAuthorOnlyNewAnswer`).

## Para ligar

```
CHAT_ANSWER_ALL_CARDS=true
CHAT_AUTHOR_ONLY_NEW_ANSWER=true
LLM_MAX_TOKENS_TEXT=700
```
