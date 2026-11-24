# O pré-processamento pensa, a AMÁLIA reescreve (2026-11-20)

Flag `CHAT_DRAFT_THEN_REWRITE`, desligada por omissão.

Com `CHAT_CARD_SUMMARIES` a frase de cada peça vai no cartão dela e o texto fica só com a abertura: ver
`docs/frases-nos-cartoes.md` (2026-11-24).

## O problema

Numa resposta com lista, a AMÁLIA recebia um prompt de 6 600 caracteres (regras, a lista começada, os cartões com
todos os campos) e decidia o que dizer de cada peça. Daí vinham os remendos acumulados neste dia: a lista decidida no
prompt e a garantia que a refazia, o negrito posto depois, o que ela dizia da visita tirado à mão, o streaming que só
envia o que fica. Mesmo assim descrevia uma peça com a ficha da vizinha, escrevia «Não está presente na visita
virtual» de uma peça que estava, e respondia em prosa numa parte das respostas.

## O que mudou

Quem decide o que se diz passa a ser o pré-processamento; a AMÁLIA só dá a voz.

1. **O modelo de pré-processamento escreve o rascunho.** Uma frase muito breve por peça (no máximo 15 palavras),
   cada uma num pedido só com a ficha dessa peça, todos em paralelo; e, noutro pedido, a frase de abertura. As
   fichas não levam a linha da visita virtual. Numa resposta por autor a abertura é a dos números do catálogo e não
   se pede outra.
2. **O serviço formata.** A abertura e uma linha por cartão, pela ordem dos cartões: «N. **nome** – frase» (o nome é
   o título, ou o n.º de inventário quando a ficha não tem título; títulos repetidos levam o n.º entre parênteses).
3. **A AMÁLIA reescreve.** O prompt dela é o rascunho e quatro linhas: reescrever em português europeu, manter as
   linhas, a ordem, os números e o que está a negrito, não acrescentar nem tirar. Sem cartões e sem regras.
4. **O serviço confere, linha a linha e por ordem** (`app/services/draft_rewrite.py`). Fica a linha dela quando é a
   linha N do rascunho reescrita: o mesmo número, o mesmo nome a negrito e uma frase. Senão fica a do rascunho. O que
   ela escrever a mais (um fecho, peças inventadas) não entra. A abertura dela fica se for uma linha curta e sem
   peças; a do autor nunca é trocada.

O texto só cresce: uma linha decidida não muda. O streaming envia cada linha quando fica decidida, e o texto final é
exatamente o que foi enviado. Se a AMÁLIA estiver em baixo, a resposta é o rascunho. Se uma frase do rascunho falhar
ou demorar (`CHAT_DRAFT_TIMEOUT_SECONDS`), entra a primeira frase da descrição da ficha.

Vale para a pesquisa de texto com dois ou mais cartões já escolhidos antes da resposta. Com um cartão, com os cartões
escolhidos pela AMÁLIA ao escrever, e nas outras respostas, fica o prompt de sempre. Com a flag ligada, a lista no
prompt, a garantia, o negrito posto depois e o streaming do texto final deixam de correr nessas respostas: é o
rascunho que faz esse trabalho.

## Os prompts

`app/prompts/chat_prompts.py`: `build_draft_sentence_prompt` (uma ficha), `build_draft_opening_prompt` (todas as
fichas, para a abertura poder responder a «Quem pintou…?») e `build_rewrite_prompt`. Numa pergunta pelo aspeto, o
pedido não vai no prompt da frase, como já acontecia com a AMÁLIA (`CHAT_VISUAL_ANSWER_RULE`).

## Medição

Protótipo, com todas as fichas num só pedido ao modelo de pré-processamento (5 pedidos, 25 respostas): os 25
rascunhos vieram válidos, mas o modelo misturava peças vizinhas («Caridade – Santa Ana com a Virgem, uma criança ao
colo…», que é a 584 Esc). Com um pedido por ficha deixou de acontecer nas respostas vistas.

No serviço, com os dois modelos reais e o streaming ligado (`scripts/medir_lista_da_resposta.py`, variantes `sobrio`
— como estava em produção — e `rascunho`), 20 pedidos de controlo × 2:

| | Como estava | Rascunho e reescrita |
|---|---|---|
| Linhas da lista escritas pela AMÁLIA que ficam (de 260) | 213 | 246 |
| Respostas finais com todos os cartões na lista | 40 de 40 | 40 de 40 |
| Respostas cortadas pelo limite de tokens | 2 | 0 |
| Palavras por frase, mediana (máximo) | 24 (79) | 13 (32) |
| Tamanho mediano da resposta | 1249 letras | 896 letras |
| Tempo mediano (máximo) | 4,6 s (9,0 s) | 4,1 s (5,4 s) |
| Texto enviado igual ao texto final | 40 de 40 | 40 de 40 |

Nos dois pedidos por Machado de Castro, 14 respostas: texto enviado igual ao final em 14; em 3 a AMÁLIA não devolveu
a lista no formato e ficaram as linhas do rascunho.

### O gerúndio à brasileira, e um exemplo que saiu caro

As frases saíam com «segurando livro», «chamas saindo do peito». Contaram-se, nas linhas finais, as que têm um
gerúndio que a ficha da peça não tem e as que dizem «livro» sem a ficha o ter (20 pedidos × 2, 260 linhas, os dois
modelos reais):

| Instrução nos dois prompts | Gerúndio que não vem da ficha | «livro» que não vem da ficha |
|---|---|---|
| Nenhuma | 21 | não contado |
| «a segurar um livro», nunca «segurando um livro» | 5 a 7 | 31 |
| Sem gerúndio (formas em -ando, -endo, -indo): «a» com o infinitivo | 13 | 0 |

A segunda redação esteve em produção cerca de 20 minutos (PR #64): os modelos copiavam o objeto do exemplo e as
respostas inventavam livros («Cristo no Horto segura um livro», «Polvorinho… que suporta um livro»). Só se tinha
contado o gerúndio. Ficou a terceira, sem exemplo com objeto. Na medição das 288 linhas, o que estava em produção
antes deste modo tinha 60 linhas com gerúndio próprio.

## Limites

- **Ainda escapam gerúndios à brasileira**: 13 em 260 linhas, com as instruções nos dois prompts (ver acima).
- **A AMÁLIA enfeita a abertura** nas pesquisas que não são por autor («Bem-vindos ao Museu Nacional de Arte
  Antiga! Permita-me apresentar-lhe…») e às vezes acrescenta à frase o que não estava no rascunho. Não se limita
  (decisão de 2026-11-20).
- **Nada diz se a peça está na visita**, mesmo quando o visitante pergunta por ela: a resposta é a lista das peças
  que a pesquisa trouxe.
- **Em inglês** os prompts existem mas não foram medidos.
- O que foi medido foi o formato, o tamanho e o tempo. A exatidão de cada frase face à ficha foi vista à mão em
  cerca de 40 respostas, não contada.

## Testes

`tests/test_draft_rewrite.py`: a junção do rascunho com a reescrita, o streaming com qualquer corte dos tokens, o
rascunho com um pedido por ficha, e a pesquisa de ponta a ponta (incluindo a AMÁLIA em baixo).

## Para ligar

```
CHAT_DRAFT_THEN_REWRITE=true
```
