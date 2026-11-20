# Só peças do autor quando se pergunta por um autor (2026-11-19)

Uma flag nova, `CHAT_AUTHOR_ONLY_RESULTS`, desligada por omissão. Precisa de `CHAT_AUTHOR_CARD`. O frontend não muda.

## O problema

Com a ficha do autor na resposta, a primeira página de cartões é quase sempre só de peças dele, mas «ver mais resultados»
continuava pela pesquisa geral. Medido a 2026-11-19 na produção:

| Pergunta | Peças do autor | Total anunciado | O que vinha a seguir |
|---|---|---|---|
| «Há pinturas de Zurbarán?» | 8 | 1000 | da página 2 em diante, só peças de outros autores |
| «Quadros de Josefa de Óbidos» | 4 | 1000 | o mesmo |
| «Mostra-me obras de Gregório Lopes» | 19 | 1000 | a partir da página 3, iluminuras com «S. Gregório» no título |
| «Há esculturas de Machado de Castro?» | 7 | 1000 | um cartão de outro autor logo na página 1 |
| «Mostra-me as obras de Soares dos Reis.» (no museu com esse nome) | 84 | 44 | sete dos oito cartões da página 1 eram de outros autores |

A pesquisa é por texto e por significado, sem filtro de autor: depois das peças do autor vêm as que têm palavras parecidas.

## O que mudou

Quando a resposta leva a ficha do autor, faz-se mais uma pergunta ao modelo de pré-processamento
(`build_author_only_prompt`): para responder a esta mensagem, que peças se devem mostrar?

- A) só peças feitas por este artista (o visitante quer as obras dele ou saber quem foi)
- B) peças dele e de outro artista que a mensagem também nomeia
- C) peças de artistas parecidos, do mesmo tempo ou do mesmo género
- D) peças de outros artistas, e não dele

Só com a resposta A os cartões passam a ser só do autor:

1. A pesquisa que a resposta fez repete-se com os mesmos textos, o mesmo vetor e os mesmos filtros, mais o filtro dos ids do
   autor (`creator_ids`, todas as grafias que a ficha junta).
2. Na primeira página ficam os cartões que já eram dele, pela ordem em que estavam. Os lugares dos cartões que saem são
   para as peças dele que vêm a seguir na pesquisa. Uma página que o modelo de escolha dos cartões deixou mais curta não é
   aumentada, a não ser com `CHAT_AUTHOR_ONLY_FILL_PAGE` (2026-11-20): aí a página enche-se com peças dele até ao tamanho
   da página, os cartões escolhidos primeiro. Em «Encontra as obras de Machado de Castro neste museu?» a escolha ficou
   por 6 das 15 candidatas, a página por 6 cartões e a 7.ª peça dele (584 Esc, sem título) ia sozinha para «Ver mais
   resultados». A escolha é feita entre as candidatas da pesquisa geral; num pedido pelas obras de um autor as outras
   peças dele também contam, e já eram elas que vinham em «ver mais». Num pedido mais estreito («retratos de …») entram
   na página peças dele que a escolha não tinha posto, pela ordem da pesquisa.
3. `results_total` passa a ser o número de peças do autor que a pesquisa encontra, e `results_has_more` diz se há mais.
4. O pedido guardado para «ver mais resultados» fica com o filtro: as páginas seguintes são as peças dele que ainda não
   foram mostradas, sem repetições.
5. Quando a pesquisa de texto foi junta com a das legendas das imagens (pedido `multimodal_text`, com uma lista fixa de
   algumas dezenas de peças de vários autores), vale o mesmo: as páginas seguintes passam a vir da pesquisa de texto com
   o filtro do autor.

O texto da AMÁLIA não muda. Miniaturas e alvos de «Ver na tour» acompanham os cartões.

A decisão fica no modelo porque há mensagens que dizem o nome de um autor e não são só dele («obras de Cranach e de
Holbein», «pintores parecidos com Silva Porto», «que pinturas há sem ser as da Josefa de Óbidos?»). O filtro em si é uma
pesquisa por id de autor: não lê a mensagem nem tem casos escritos à mão.

Se o modelo ou o OpenSearch falharem, ou o passo demorar mais de oito segundos, a resposta sai como sairia sem a flag, e
o pedido guardado não é tocado.

## Como foi medido

Qwen3-VL-8B, temperatura 0. Casos em `tests/author_card_cases.py`: as 74 mensagens `COM_AUTOR` (têm de receber A) e 31
novas, `NAO_SO_DO_AUTOR`, que dizem o nome de um autor do museu mas pedem também, ou só, peças de outros (não podem
receber A). Seis museus.

### Redações da pergunta

Chamadas diretas ao modelo, com as palavras do nome como o serviço as tira da mensagem.

| Redação | Pedem um autor: A | Não só do autor: A (indevido) |
|---|---|---|
| «Que obras quer o visitante?», quatro respostas | 74 de 74 | 13 de 31 |
| «O visitante quer ver só obras de X?», sim ou não | 74 de 74 | 16 de 31 |
| «Que peças do museu se devem mostrar?», três respostas | 71 de 74 | 6 de 31 |
| «A mensagem pede peças de que artistas?», quatro respostas | 74 de 74 | 4 de 31 |
| «Que peças do museu se devem mostrar?», quatro respostas | 74 de 74 | 3 de 31 |
| A mesma, com «(o visitante quer as obras dele ou saber quem foi)» na resposta A (a que ficou) | 74 de 74 | 2 de 31 |

- Perguntar pelas peças a mostrar separa melhor do que perguntar pelo que o visitante quer: com «quer ver só obras de X?»
  o modelo diz que sim a «obras de X e de Y».
- Dar uma resposta própria a «outro artista que a mensagem também nomeia» é o que apanha os pedidos de dois autores.
- Sem o texto entre parênteses na resposta A, três «Quem foi X?» recebiam B.
- As duas que passam sem dever: «Mostra-me pintores contemporâneos do Vieira Lusitano» e «pintura portuguesa do tempo de
  Domingos Sequeira».

### Pelo código do serviço

Cada mensagem foi à API de produção (sem a flag). A resposta passou depois, num processo à parte, pelo código novo, com o
OpenSearch (só leitura), o Qwen e os alvos de navegação reais, e pediram-se três páginas. O vetor da pergunta fica na
memória da produção e não se consegue ler: no ensaio usou-se o vetor de texto de uma peça do autor. Serve para o filtro,
os totais e as páginas; a ordem das peças dentro do autor não é a que a produção dará.

| | Resultado |
|---|---|
| Mensagens que pedem um autor, com ficha | 68 de 74 (as outras seis não têm ficha e ficam como hoje) |
| Dessas, com os cartões restringidos ao autor | 68 de 68 |
| Cartões de outros autores nas três páginas, depois | 0 em 891 |
| Cartões repetidos entre páginas | 0 |
| `results_total` igual ao número de peças do autor | 68 de 68 |
| Antes: cartões de outros autores na página 1 | 31 em 398, em 14 mensagens |
| Antes: total anunciado maior do que as peças do autor | 67 de 68 (1000 em 65) |
| Mensagens que não são só do autor, com ficha | 19 de 31 |
| Dessas, restringidas sem dever | 1: «pintura portuguesa do tempo de Domingos Sequeira» |
| Tempo da ficha e dos cartões juntos | 0,35 s de mediana, 1,15 s no máximo |

- Em seis mensagens a página 1 fica mais curta, porque o autor tem menos peças do que os cartões que lá estavam: «Há
  esculturas de Machado de Castro?» passa de 8 para 7, «peças de Pierre Cardin» de 8 para 4, «obras de Rembrandt» de 2
  para 1.
- A que é restringida sem dever mostra só peças de Domingos Sequeira a quem pediu a pintura do tempo dele.

### Em produção, depois de ligar

Na primeira verificação em produção (2026-11-19), duas de sete perguntas por um autor ficaram por restringir: «Mostra-me
as obras de Soares dos Reis.» e «Mostra-me criações da Ana Salazar.». Nas duas a pesquisa tinha sido junta com a das
legendas, e esse pedido fica guardado como `multimodal_text`, que o código deixava passar. O ensaio de cima não o
apanhou porque simulava sempre um pedido de texto simples. Corrigido no ponto 5 de «O que mudou».

## Limites conhecidos

- Só há restrição quando há ficha do autor, ou seja, quando a pesquisa mostra pelo menos uma peça dele e o modelo
  reconhece o nome (`docs/ficha-do-autor.md`).
- Peças de uma oficina ou de um seguidor são de outro autor no catálogo: «Oficina de Nuno Gonçalves» não aparece quando se
  pede Nuno Gonçalves, nem «Oficina de Joaquim Machado de Castro» quando se pede Machado de Castro.
- Uma mensagem de seguimento sem o nome do autor («E desenhos?») não tem ficha e fica como hoje.
- O texto da AMÁLIA é escrito antes: pode referir uma peça de outro autor cujo cartão saiu.
- A ordem dentro do autor é a da pesquisa. «Pinturas de Domingos Sequeira» mostra primeiro as pinturas e depois as outras
  peças dele, não só pinturas.
- O total conta fichas do catálogo. Com fichas iguais juntas num cartão, o número de cartões pode ser menor.

## Para ligar

Juntar ao `.env` e reiniciar o backend:

```
CHAT_AUTHOR_ONLY_RESULTS=true
```
