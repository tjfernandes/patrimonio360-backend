# Perguntas para testar os campos enriquecidos (MNAA)

50 perguntas para experimentar na visita do Museu Nacional de Arte Antiga, que é o único museu com enriquecimento.
Cobrem as três coisas que mudaram a 24/11/2026: as contagens, as listagens e a pesquisa «tipo + o que se vê».

Os valores da coluna «Deve responder» são o que o backend em produção respondeu a 24/11/2026, com
`CHAT_ENRICHED_FIELDS` e `CHAT_ENRICHED_SEARCH_FILTER` ligadas. Mudam se o catálogo ou as listas mudarem; o que tem
de se manter é a ordem de grandeza e o tipo de peças nos cartões.

## Como confirmar o que o sistema fez

Depois de fazer uma pergunta, correr:

```bash
deploy/amalia_logs.sh -n 1
```

Quando os campos enriquecidos foram usados, aparece uma linha `chat.enriched_fields` com a leitura do pedido, por
exemplo `read_as=enr_tipo=cadeira  pieces=111`. Se a linha não aparecer, a pergunta seguiu pelo caminho de sempre.

Em cada tabela, a última coluna é para anotar: OK, Parcial ou Falhou.

## A. Contagens por tipo de peça

A resposta é imediata (menos de 1 segundo), começa por «Há N peças…» e traz 8 cartões desse tipo.

| N.º | Pergunta | Deve responder | Leitura no registo | Resultado |
|---|---|---|---|---|
| 1 | Quantas cadeiras há? | 111 | `enr_tipo=cadeira` | |
| 2 | Quantos cálices há? | 14 | `enr_tipo=cálice` | |
| 3 | Quantos copos há? | 108 | `enr_tipo=copo` | |
| 4 | Quantas esculturas há? | 299 | `enr_tipo=escultura` | |
| 5 | Quantos biombos há? | 6, com os seis cartões | `enr_tipo=biombo` | |
| 6 | Quantas peças de mobiliário tem o museu? | 320 | `enr_tipo=mobiliário` | |

O que confirmar: os cartões são todos do tipo pedido. Antes da alteração, «cadeiras» dava 157 e incluía desenhos e
pinturas com uma cadeira na cena.

## B. Contagens por material, técnica e origem

| N.º | Pergunta | Deve responder | Leitura no registo | Resultado |
|---|---|---|---|---|
| 7 | Quantas peças de prata há? | 290 | `enr_materiais=prata` | |
| 8 | Quantas peças em marfim há? | 103 | `enr_materiais=marfim` | |
| 9 | Quantas peças de porcelana há? | 31 | `enr_materiais=porcelana` | |
| 10 | Quantas pinturas a óleo há? | 329 | `enr_tecnicas=pintura a óleo` | |
| 11 | Quantas peças bordadas há? | 109 | `enr_tecnicas=bordado` | |
| 12 | Quantas peças da China há? | 66 | `enr_origem_pais=China` | |
| 13 | Quantas peças do Japão há? | 10 | `enr_origem_pais=Japão` | |

O que confirmar: na prata, os cartões são ourivesaria e não pratos de vidro ou de cerâmica (antes dava 372, porque a
pesquisa por palavras confundia «prata» com «prato»). Na China, não aparecem desenhos a tinta-da-china. No marfim
aparecem miniaturas pintadas sobre marfim, que antes ficavam de fora.

## C. Combinações, período e «a mais antiga»

| N.º | Pergunta | Deve responder | Leitura no registo | Resultado |
|---|---|---|---|---|
| 14 | Quantas peças de prata do século XVIII há? | 218, «entre 1700 e 1799» | `enr_materiais=prata` | |
| 15 | Quantos copos de vidro há? | 107 | `enr_tipo=copo E enr_tipo=vidro ou enr_materiais=vidro` | |
| 16 | Qual é a cadeira mais antiga? | «Cadeira de braços» (51 Mov), de 1470–1481, e as 24 mais antigas por ordem | `enr_tipo=cadeira` | |
| 17 | Qual é o cálice mais recente? | «Cálice» (346 Our), de 1768 | `enr_tipo=cálice` | |
| 18 | Quantas esculturas em madeira há? | 14 (ver «Limites conhecidos») | `enr_tipo=escultura E enr_materiais=madeira` | |

## D. Contagens por tema

| N.º | Pergunta | Deve responder | Leitura no registo | Resultado |
|---|---|---|---|---|
| 19 | Quantas peças religiosas há? | 1 804 | `enr_temas_palavras=religiosa` | |
| 20 | Quantas esculturas religiosas há? | 258, só esculturas nos cartões | `enr_tipo=escultura E enr_temas_palavras=religiosa` | |
| 21 | Quantos retratos há? | 137 | `enr_temas_palavras=retrato` | |
| 22 | Quantas peças com brasões há? | 346 | `enr_temas_palavras=brasão` | |
| 23 | Quantas pinturas com paisagens há? | 97, só pinturas nos cartões | `enr_tipo=pintura E enr_temas_palavras=paisagem` | |
| 24 | Quantas peças de mitologia há? | 178 | `enr_temas_palavras=mitologia` | |
| 25 | Quantas peças com batalhas há? | 180 | `enr_temas_palavras=batalha` | |
| 26 | Quantos manuscritos há? | 1 073 (os fólios iluminados) | `enr_tecnicas=manuscrito ou enr_temas_palavras=manuscrito` | |

O que confirmar: abrir dois ou três cartões e ver se o tema faz sentido. Heráldica (brasões), mitologia e guerra
(batalhas) entraram abaixo da fasquia da validação: é de esperar cerca de 1 cartão errado em cada 8.

## E. Contagens pelo que se vê na fotografia

| N.º | Pergunta | Deve responder | Leitura no registo | Resultado |
|---|---|---|---|---|
| 27 | Quantas pinturas com animais há? | 80 | `enr_tipo=pintura E enr_temas_palavras=animal ou enr_foto_palavras=animal` | |
| 28 | Quantas esculturas com inscrições há? | 19 | `enr_tipo=escultura E enr_foto_palavras=inscrição` | |
| 29 | Quantos desenhos com pessoas há? | 398 | `enr_tipo=desenho ou enr_tecnicas=desenho E enr_foto_palavras=pessoa` | |
| 30 | Quantas peças com animais há? | 626 | `enr_temas_palavras=animal ou enr_foto_palavras=animal` | |

Nas perguntas 27 e 30 contam as peças em que o modelo viu animais na fotografia e também aquelas cuja ficha os
descreve.

## F. Listagens

| N.º | Pergunta | Deve responder | Resultado |
|---|---|---|---|
| 31 | Que materiais há no museu? | «Sabe-se o material de 3348 das 3658 peças…», com ouro (1017), velino (739), papel (708), madeira (558), vidro (370), pergaminho (342), prata (290)… Sem cartões. | |
| 32 | Que técnicas há no museu? | «Sabe-se a técnica de 3479 das 3658 peças…», com aplicação de ouro (1441), pintura a têmpera (971), iluminura (923), desenho (448), pintura a óleo (329)… Sem cartões. | |

O que confirmar: os valores são nomes simples («prata», «madeira») e não as frases do catálogo («Prata fundida,
cinzelada e incisa»).

## G. Pesquisa «tipo + o que se vê»

Aqui a resposta é uma conversa normal, com texto da Amália e cartões. O que mudou é que os cartões são só do tipo
pedido e mostram o que se pede.

| N.º | Pergunta | Deve acontecer | Leitura no registo | Resultado |
|---|---|---|---|---|
| 33 | Pinturas com animais. | 7 cartões, todos pinturas: Regresso da caça, Passagem de um Vau, Grupo Familiar, Interior de Taberna, Natureza morta… Total 65. | `enr_tipo=pintura E enr_foto_palavras=animal` | |
| 34 | Mostra-me quadros com animais. | O mesmo conjunto de 65 («quadros» lê-se como pinturas). | idem | |
| 35 | Esculturas com animais. | 8 esculturas: Três meninos e uma cabra, Anunciação aos Pastores, Pastor, Cavalgada… Total 30. | `enr_tipo=escultura E enr_foto_palavras=animal` | |
| 36 | Pinturas com inscrições. | 8 pinturas com letras visíveis: Profeta Isaías, Vanitas, Profeta Miqueias… Total 70. | `enr_tipo=pintura E enr_foto_palavras=inscrição` | |
| 37 | Esculturas com inscrições. | Esculturas com inscrição; total 19. A 24/11 saiu só um cartão (a Lápide funerária de D. Justa). | `enr_tipo=escultura E enr_foto_palavras=inscrição` | |
| 38 | Desenhos com animais. | 8 desenhos com animais. Total 61. | `enr_tipo=desenho E enr_foto_palavras=animal` | |

O que confirmar: abrir as imagens e ver se o animal ou a inscrição lá está. Antes da alteração, «pinturas com
animais» mostrava uma pintura, uma iluminura e seis desenhos.

A pergunta 37 merece atenção: a pesquisa encontra as 19 esculturas, mas o modelo que escolhe os cartões ficou só com
uma. Se se repetir, é um problema da escolha de cartões e não do filtro.

## H. Perguntas que têm de ficar como estavam

O filtro não deve atuar em nenhuma destas: não aparece a linha `chat.enriched_fields`, e os cartões são os de sempre.

| N.º | Pergunta | Deve acontecer | Resultado |
|---|---|---|---|
| 39 | Mostra-me cadeiras. | Cadeiras, incluindo cadeiras de braços e uma duchesse. | |
| 40 | Peças em marfim. | Olifantes, saleiro, colher, cómodas com marfim. | |
| 41 | Peças vindas do Japão. | **Os biombos Namban têm de aparecer** (1638, 1639 e 1641 Mov), com a caixa lacada e os estribos. | |
| 42 | Pinturas com cães. | Pinturas e iluminuras com cães («cães» não é um dos três valores da fotografia). | |
| 43 | Tens tapeçarias? | Tapeçarias das séries A Arte da Guerra e Marco Aurélio. | |
| 44 | Há esculturas em madeira? | Esculturas, como hoje. | |
| 45 | Tentações de Santo Antão | As duas pinturas com esse título (1498 Pint e 1470 Pint). | |

A pergunta 41 é a mais importante deste grupo: a ficha dos biombos Namban não tem centro de produção, e um filtro
pela origem tirava-os da resposta. Foi por isso que a pesquisa não filtra pela origem nem pelo material.

## I. Limites conhecidos

Respostas que não estão certas, ou que surpreendem, e porquê. Servem para confirmar que o comportamento é o descrito
e não outro.

| N.º | Pergunta | Responde | Porquê | Resultado |
|---|---|---|---|---|
| 46 | Quantas esculturas em madeira há? | 14 | Das 299 esculturas, só 64 têm algum material na ficha. O número real é muito maior. | |
| 47 | Quantas peças de ouro há? | 1 017 | Conta os 887 fólios iluminados, cuja técnica é «pintura a têmpera e ouro». O material está certo, mas não é o que a pergunta quer. | |
| 48 | Quantas cadeiras bonitas há? | 0 | «Bonitas» não é valor de nenhuma lista: a pergunta segue pelo caminho antigo, que exige todas as palavras na ficha. | |
| 49 | Quantos azulejos há? | 3 | «Azulejo» não é um tipo do MNAA: caminho antigo, por palavras (três fichas mencionam azulejos). | |
| 50 | How many chairs are there? | 49 objects matching “chairs” | Os campos enriquecidos só se leem em português. Em inglês a contagem é a antiga, por palavras. | |

Outros limites que não cabem numa pergunta só:

- **«Quantas cadeiras há?» não conta as bergères e as duchesses** (5 peças), porque o tipo é a palavra do título.
- **«Que materiais há nas esculturas?»** não é reconhecida como listagem e vai para a pesquisa normal. Já era assim.
- **Os fólios iluminados não contam como pinturas** em «pinturas com animais»: o enriquecimento classifica-os como
  manuscritos.
- **Só o MNAA tem enriquecimento.** Noutro museu, todas estas perguntas respondem como antes.

O que cada campo é, como foi validado e o que ficou de fora está em `docs/enriquecimento-por-campos.md`.
