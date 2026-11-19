# Ficha do autor no chat (2026-11-18)

Uma flag nova, desligada por omissão. Não está ligada em produção. O frontend ainda não desenha o campo novo; o pedido para o
frontend está em `docs/ficha-do-autor-frontend-prompt.md`.

## O problema

Quando o visitante pergunta por um autor («Mostra-me obras de Gregório Lopes», «Quem foi Soares dos Reis?»), a resposta trazia
o texto e os cartões das peças, e nada sobre o autor. O índice de autores tem atividade, datas, locais e biografia para parte
deles, e esses dados não chegavam ao chat.

## O que mudou

| Flag | O que faz |
|---|---|
| `CHAT_AUTHOR_CARD` | Quando a mensagem diz o nome de um autor das peças mostradas, a resposta leva `author` com a ficha dele |
| `CHAT_AUTHOR_PORTRAITS` | A ficha leva o retrato do autor e os créditos da imagem, quando o autor tem retrato em `app/data/author_portraits.json` (`docs/retratos-dos-autores.md`) |
| `CHAT_AUTHOR_ONLY_RESULTS` | Com a ficha na resposta e o visitante a querer as obras desse autor, os cartões são só de peças dele, na primeira página e em «ver mais resultados» (`docs/so-pecas-do-autor.md`) |

O texto da resposta e os cartões são os mesmos com e sem `CHAT_AUTHOR_CARD`; só `CHAT_AUTHOR_ONLY_RESULTS` mexe nos cartões. O campo `author` vem em `POST /chat/messages`, no evento
`result` do stream e nas rotas de `regenerate`; nas outras respostas é `null`.

| Campo de `author` | De onde vem |
|---|---|
| `name` | O nome como o catálogo o escreve, pelas fichas das peças |
| `activity`, `birth_date`, `death_date`, `birth_place`, `death_place`, `biography`, `url` | Índice de autores |
| `birth_year`, `death_year` | O ano lido da data, quando ela tem um só ano de quatro dígitos |
| `artifact_count` | Contagem nas fichas das peças do museu da pesquisa |
| `in_tour_count` | Quantas dessas peças têm alvo de «Ver na tour» (hotspot ou peça localizada no panorama), com o critério dos cartões. Não usa o campo `in_tour` do índice, que está desatualizado. Nunca é menor do que o número de cartões do autor que a mesma resposta traz com alvo |
| `entity_ids` | Os ids do autor (mais de um quando o catálogo o tem em várias fichas) |
| `image_url` | Retrato, com `CHAT_AUTHOR_PORTRAITS`: ver «Retratos». É sempre um endereço absoluto http(s): o frontend ignora caminhos relativos e outros esquemas (em `image_url` e em `url`) e mostra as iniciais. `null` quando o autor não tem retrato ou com a flag desligada |
| `image_credit`, `image_license`, `image_source_url` | Créditos do retrato (autor da imagem, licença, página no Wikimedia Commons). `null` sem retrato |

## Como se decide

1. **O autor sai dos dados.** Entre os autores dos cartões da resposta, o que tem no nome mais palavras da mensagem. Uma
   palavra conta pelo que distingue («António» está em dezenas de nomes, «Pousão» num só). O que o catálogo junta ao nome entre
   parênteses (datas, locais) não conta. Sem cartões, ou sem palavras em comum, não há ficha nem pergunta ao modelo.
2. **O modelo de pré-processamento diz o que essas palavras são na mensagem**, numa pergunta com seis respostas por letra
   (`build_author_name_prompt`): artista ou oficina; santo ou quem está representado; rei ou figura histórica; lugar, sala ou
   museu; uma peça; não é um nome próprio. Só a primeira junta a ficha.

O passo 1 é uma procura na lista de autores das peças mostradas, não uma decisão. Fica fora do modelo porque o modelo, medido,
escolhe mal entre nomes parecidos: com «Gregório Lopes» em primeiro numa lista de seis, escolhia «Queirós, Gregório Francisco».

Grafias do mesmo autor ficam juntas quando o nome tem as mesmas palavras («Sequeira, Domingos António de» e «Sequeira,
Domingos António de (1768-1837)»): a ficha soma as peças das duas e fica com o primeiro valor preenchido de cada campo.

A ficha é um extra: se o OpenSearch ou o modelo falharem, ou o passo demorar mais de seis segundos, a resposta sai sem ela.

## Como foi medido

Qwen3-VL-8B, temperatura 0. Casos em `tests/author_card_cases.py`: 74 mensagens com o nome de um autor e 71 sem (santos,
reis, lugares, salas, o nome do museu, «autor desconhecido»), em seis museus.

### Redações da pergunta ao modelo

Chamadas diretas ao modelo, com o autor mais parecido de todo o museu (mais difícil do que em serviço, onde só contam os
autores dos cartões). As 129 mensagens reais são as da checklist do MNAA, do benchmark e dos casos de leitura.

| Redação | Com autor: certas | Sem autor: com ficha | Mensagens reais com ficha |
|---|---|---|---|
| Escolher o autor numa lista de seis, ou «nenhum» | 69 de 73 | 19 de 72 | 37 de 129 |
| «A mensagem diz o nome de algum destes autores?» | 28 de 73 | 1 de 72 | 3 de 129 |
| «O visitante pergunta por este autor?», com o nome do catálogo | 73 de 73 | 14 de 72 | 36 de 129 |
| «Na mensagem, «X» é o nome de quê?», com o nome do catálogo | 73 de 73 | 13 de 72 | 34 de 129 |
| A mesma, sem o nome do catálogo | 69 de 73 | 8 de 72 | 18 de 129 |
| A mesma, com a primeira e a última resposta reescritas (a que ficou) | 71 de 74 | 6 de 71 | 19 de 129 |

- Dizer ao modelo o nome que está no catálogo faz com que responda «é esse autor» a quase tudo.
- A redação final troca «autor de obras» por «oficina (quem fez obras)» e «não é um nome» por «não é um nome próprio»:
  mais duas certas e menos uma indevida. As mensagens com a palavra «autor» continuam a receber «Autor desconhecido».
- Nas cinco primeiras linhas «obras de Rembrandt» estava contada como sem autor; o museu tem peças dele e passou para o
  outro grupo.
- Das 19 mensagens reais com ficha na redação final, 16 dizem mesmo o nome de um autor; as outras três são «Que autor tem mais
  obras?», «Desse autor há mais?» e «Faz o mesmo mas só com pintura», com autores que em serviço não estariam nos cartões.

### Pelo código do serviço

Cada mensagem foi à API de produção (sem a flag) e à resposta aplicou-se `_author_card_for_response`, que é o que o backend
faz com a flag ligada.

| | Resultado |
|---|---|
| Mensagens com autor: ficha do autor certo | 68 de 74 |
| Mensagens sem autor: com ficha | 1 de 71 |
| Checklist do MNAA (115 mensagens, cartões da repetição guardada): com ficha | 15, todas com o nome do autor na mensagem |
| Tempo do passo quando pergunta ao modelo (76 de 145 mensagens) | 0,11 s de mediana, 0,79 s no máximo |
| Tempo do passo nas outras | abaixo de 0,03 s |

- As seis que falham: em quatro o modelo não reconhece o nome como de artista («E do Nuno Gonçalves, há alguma coisa?»,
  «Quem foi Frei Carlos?», «roupa da Fátima Lopes», «Do you have Mary Quant?»); em duas a pesquisa não trouxe nenhuma peça do
  autor («Who was Soares dos Reis?», sem cartões, e «Quadros de Silva Porto» na Casa-Museu). Ficam como hoje, sem ficha.
- A que recebe ficha sem dever: «Que peças são de autor desconhecido?» recebe a ficha de «Autor desconhecido».
- Nas 74 mensagens com autor, 395 dos 428 cartões são de peças dele, e em 63 são todos. A pior é «Mostra-me as obras de
  Soares dos Reis» no museu com esse nome: um cartão em oito (as palavras do nome do museu estão na bibliografia de muitas
  fichas).
- Das 68 fichas: 28 têm ano de nascimento, 16 têm biografia, 6 têm atividade, 67 têm ligação ao catálogo, nenhuma tem retrato.

## Retratos

O catálogo de origem não tem retratos de autores. Confirmado a 2026-11-18 em doze páginas de autor (as dos autores com
biografia e mais peças): só têm imagens das peças.

Com `CHAT_AUTHOR_PORTRAITS` (desligada por omissão) o retrato vem de `app/data/author_portraits.json`, que liga ids de
autores do catálogo a imagens do Wikimedia Commons: 340 fichas de autor, de 324 autores. Como foram escolhidos e
verificados, os números por museu e os créditos que o frontend tem de mostrar estão em `docs/retratos-dos-autores.md`.

## Seguimentos que não dizem o nome (2026-11-19)

Depois de se falar de uma peça, «Que mais obras existem deste autor?» mostrava as obras certas e não trazia a ficha:
o nome procurava-se na mensagem do visitante, e «deste autor» não tem nome nenhum. A leitura do seguimento já
resolvia o pedido para «obras do Machado de Castro», mas a ficha não o usava.

Com `CHAT_AUTHOR_CARD_FOLLOW_UP`, quando a mensagem não tem palavras do nome de nenhum autor das peças mostradas, o
nome procura-se no pedido resolvido que a pesquisa guardou. As duas perguntas ao modelo (se as palavras são o nome de
um artista; se o visitante quer só as obras dele) levam esse pedido em vez da mensagem. Uma mensagem que já diz um nome
fica exatamente como estava.

Medido a 2026-11-19 com o modelo real: oito pedidos resolvidos («obras do Machado de Castro», «Quem foi Machado de
Castro?», «What else did Dürer paint?»…) × 3 execuções, sempre «A» nas duas perguntas. Em produção, a conversa que deu
origem a isto (a Caridade, 63 Esc, e depois «Que mais obras existem deste autor?») passou a trazer a ficha de Machado
de Castro. Testes em `tests/test_author_only_results.py` (`TestAuthorCardOnFollowUp`).

## Limites conhecidos

- A ficha só aparece quando a pesquisa mostra pelo menos uma peça do autor.
- O nome é o do catálogo («Reis, António Soares dos», «Keil, Maria (Silves, 1914 - Lisboa 2012)»); não é reescrito.
- As datas são as do catálogo, em formatos variados («19/06/1731», «00/00/1884», «Século XV (?)»). O ano vem à parte.
- A maior parte dos autores só tem nome: no MNAA, 13 dos 450 têm biografia.
- Entradas do catálogo que não são pessoas («Autor desconhecido», «Não identificado») podem receber ficha.
- O texto da AMÁLIA não usa a biografia: «Quem foi Gregório Lopes?» continua a ser respondido a partir das peças.
- O modal de detalhe da peça (`/artifacts/{id}/detail-context`) lê os campos do autor pelos nomes antigos e, no índice v4,
  só devolve o nome, a biografia e a ligação. Não foi alterado aqui.

## Para ligar

Juntar ao `.env` e reiniciar o backend:

```
CHAT_AUTHOR_CARD=true
CHAT_AUTHOR_PORTRAITS=true   # opcional: retratos
CHAT_AUTHOR_ONLY_RESULTS=true   # opcional: só peças do autor nos cartões
CHAT_AUTHOR_CARD_FOLLOW_UP=true   # opcional: a ficha também em «e deste autor?»
```
