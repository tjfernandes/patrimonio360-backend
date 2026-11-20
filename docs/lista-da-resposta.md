# A lista da resposta decidida no prompt e garantida no texto (2026-11-20)

Três flags novas, desligadas por omissão, todas por cima de `CHAT_ANSWER_ALL_CARDS`:
`CHAT_UNTITLED_CARD_BY_INVENTORY`, `CHAT_ANSWER_LIST_PLAN` e `CHAT_MISSING_CARDS_IN_LIST`.

## O problema

«Encontra obras de Machado de Castro» mostra sete cartões; um deles, a 584 Esc, não tem título no catálogo. O
cartão ia para o prompt como «Título: sem título» e a AMÁLIA ou saltava a peça (e a garantia de
`CHAT_ANSWER_ALL_CARDS` acrescentava «Nos cartões estão também: a peça 584 Esc.» depois do fecho), ou falava dela num
parágrafo à parte («A peça Santa Ana ensinando a Virgem a ler (Inv. 584 Esc) também está presente no museu…»), ou
juntava a descrição dela à linha da «Caridade». A resposta fechava muitas vezes com frases que os cartões não
diziam («Todas as obras confirmadas estão disponíveis nesta visita virtual»).

A AMÁLIA estava a decidir o que entra na lista, com que nome e em que formato. Isso passa a vir decidido.

## O que mudou

- **`CHAT_UNTITLED_CARD_BY_INVENTORY`.** Um cartão sem título chama-se pelo n.º de inventário: no prompt
  («Título: 584 Esc») e na última linha da garantia («**584 Esc**» em vez de «a peça 584 Esc»).
- **`CHAT_ANSWER_LIST_PLAN`.** Quando a instrução manda falar de todos os cartões e eles já estão escolhidos, o
  prompt final leva, antes dos cartões, o bloco «LISTA A ESCREVER»: uma linha começada por cartão, pela ordem dos
  cartões («1. Santa Teresa de Ávila –», … «7. 584 Esc –»; títulos repetidos levam o n.º entre parênteses). A
  AMÁLIA só acaba cada linha com uma frase.
- **A garantia.** Com a lista no prompt, o texto final é sempre a abertura da AMÁLIA e uma linha por cartão,
  numerada e pela ordem dos cartões. Cada linha é a que ela escreveu para esse nome (com ou sem marcador, com ou
  sem artigo); se não a escreveu, a frase é a primeira da descrição da ficha (em inglês fica só o nome: a ficha
  está em português). O que ela escrever depois da lista sai. Substitui a linha «Nos cartões estão também».
- **`CHAT_MISSING_CARDS_IN_LIST`.** Para as respostas que não levam a lista no prompt: os cartões por dizer entram
  na lista que a AMÁLIA escreveu, com o marcador, o negrito e o separador do último item, em vez de irem numa
  última linha. Sem lista no texto fica a linha de sempre.

## Medição

`scripts/medir_lista_da_resposta.py`, com os cartões reais e a AMÁLIA real (`LLM_MAX_TOKENS_TEXT=700`).

«Encontra obras de machado de castro», 7 cartões, 8 respostas:

| | Como estava | Com a lista no prompt |
|---|---|---|
| Cartões com linha própria, só a AMÁLIA | 23 de 56 | 55 de 56 |
| Respostas finais com os 7 em linha própria | 0 de 8 | 8 de 8 |

20 pedidos de controlo (6 gerais e 14 pelo aspeto, de 3 a 8 cartões), 2 respostas cada:

| | Como estava | Com a lista no prompt |
|---|---|---|
| Cartões com linha própria, só a AMÁLIA | 218 de 260 | 222 de 260 |
| Respostas finais com todos em linha própria | 26 de 40 | 40 de 40 |
| Cortadas pelo limite de tokens | 0 | 2 |
| Tamanho mediano | 1252 letras | 1107 letras |
| Tempo mediano (máximo) | 4,7 s (8,7 s) | 4,4 s (9,2 s) |

O que saiu depois da lista em 20 respostas vistas à mão: «Todas as obras confirmadas estão presentes na visita
virtual», «A ficha não indica se alguma destas peças é Tesouro Nacional» e parecidas; nenhuma dizia algo que os
cartões confirmassem.

Tentou-se e ficou de fora: um modelo com campos «< >» para preencher (a AMÁLIA copiava os «< >» e as respostas
saíam com o dobro do tamanho) e a lista no fim do prompt (a resposta abria com «Amália:»).

## Limites

- **A AMÁLIA ainda mistura factos entre cartões vizinhos**: descreve a «Caridade» com a Santa Ana da 584 Esc. A
  lista decidida não resolve isso; resolveria escrever cada linha num pedido só com o cartão dessa peça.
- **Quando ela ignora a lista e responde em prosa** (cerca de 15 % dos cartões no controlo), as linhas saem com a
  primeira frase da ficha, mais seca do que as dela. Numa pergunta como «Quem pintou o São Jerónimo?» fica a
  primeira frase dela, que responde, e depois a lista.
- **O fecho desaparece sempre**, mesmo quando era inofensivo.
- **A abertura continua a ser dela** (nas perguntas por um autor é a dos números do catálogo).
- **Com streaming, o texto muda no fim** quando a garantia mexe: o frontend tem de mostrar o `reply` final, como já
  acontecia com a última linha.
- Só a pesquisa de texto com os cartões escolhidos antes da resposta. Quando é a AMÁLIA a escolher os cartões na
  própria resposta, a lista não vai no prompt.

## Testes

`tests/test_answer_list_plan.py`.

## Para ligar

```
CHAT_UNTITLED_CARD_BY_INVENTORY=true
CHAT_ANSWER_LIST_PLAN=true
CHAT_MISSING_CARDS_IN_LIST=true
```
