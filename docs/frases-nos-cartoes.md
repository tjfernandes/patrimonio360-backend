# O texto responde à pergunta, a frase de cada peça vai no cartão dela (2026-11-24)

Flag `CHAT_CARD_SUMMARIES`, desligada por omissão. Precisa de `CHAT_DRAFT_THEN_REWRITE` ligada (está, em produção):
é uma variante do rascunho descrito em `docs/rascunho-e-reescrita.md`.

## O que muda

Hoje, uma resposta com vários cartões é uma abertura e uma lista numerada, «N. **nome** – frase», que repete no
texto o que os cartões mostram. Com a flag:

1. **Cada cartão leva a sua frase.** `artifact_results[].card_summary` (`str | None`): texto simples, numa linha, sem
   markdown, sem o n.º de inventário, até 220 caracteres. O frontend passa à lista de uma coluna quando algum cartão
   da resposta traz o campo; sem ele fica a grelha.
2. **O texto é só a resposta.** `reply` é uma frase (às vezes duas), sem lista e sem negrito. Numa resposta por autor
   é a abertura dos números do catálogo, como hoje.
3. **«Ver mais resultados» também.** Os cartões de `POST /messages/results` levam `card_summary` quando os da
   primeira página levaram. O pedido do visitante e se a pergunta era pelo aspeto ficam guardados com a pesquisa
   (`card_summaries` no pedido guardado), para as frases das páginas seguintes se escreverem como as da primeira.
4. **A AMÁLIA não é chamada nestas respostas.** O texto é o que o modelo de pré-processamento escreveu. Com
   `CHAT_CARD_SUMMARIES_REWRITE_OPENING=true` a AMÁLIA reescreve a abertura; mediu pior (ver abaixo) e fica desligada.

Um cartão, um `card_summary`: peças do mesmo conjunto levam cada uma a sua frase, e um cartão com fichas juntas
(`group_size > 1`, `CHAT_COLLAPSE_SET_DUPLICATES`) leva uma só.

## Em que respostas se aplica

Aplica-se onde o rascunho já se aplicava: pesquisa de texto, primeira página, dois ou mais cartões já escolhidos
antes de a resposta ser escrita.

| Resposta | Com a flag |
|---|---|
| Pesquisa de texto com 2 ou mais cartões (por tema, autor, época, aspeto, n.º de inventário, noutro museu) | Texto curto e `card_summary` em todos os cartões |
| Pesquisa por autor com «só peças do autor» | O texto é a abertura dos números do catálogo; `card_summary` em todos os cartões |
| «Ver mais resultados» de uma dessas pesquisas | `card_summary` em todos os cartões da página |
| Pesquisa com um só cartão | Como hoje: o texto já é sobre a peça, sem `card_summary` |
| Pesquisa sem cartões, ou em que o pré-processamento rejeitou todos | Como hoje |
| Analítica do acervo («quantas…», «a mais antiga…», listagens de materiais e técnicas) | Como hoje: texto por regras, cartões em grelha |
| Tesouros Nacionais (a lista fechada) | Como hoje |
| Peças parecidas com uma peça (semelhança ancorada) | Como hoje |
| Percursos e perguntas sobre o sítio da visita | Como hoje |
| Descrição de uma peça selecionada, conversa sem pesquisa | Como hoje |
| Pesquisa por imagem, por modelo 3D, e «apontar» | Como hoje (outros fluxos) |
| Resposta pedida em JSON | Como hoje |
| «Ver mais resultados» de uma pesquisa feita sem a flag | Como hoje: a grelha continua grelha |

Nas respostas que ficam como hoje os cartões não trazem `card_summary`, e o frontend mostra a grelha.

## Os prompts

`app/prompts/chat_prompts.py`, os mesmos dois construtores, com um parâmetro:

- `build_draft_sentence_prompt(..., card_summary=True)`: o texto do cartão. Diz o que a peça é, em uma ou duas frases
  curtas, até 25 palavras, só com o que está na ficha; quando o pedido pergunta por uma coisa que a ficha responde
  (quem a fez, de quando é, de que é feita, que tipo de peça é), diz o que a ficha diz, mesmo não sendo o que foi
  pedido. Mantém as regras de sempre (sem título, sem n.º de inventário, sem visita, sem gerúndio) e a exceção das
  perguntas pelo aspeto: aí o pedido não vai no prompt (`CHAT_VISUAL_ANSWER_RULE`).
- `build_draft_opening_prompt(..., card_summaries=True)`: a resposta. Uma frase curta que vale para o conjunto das
  peças e não fala de nenhuma em particular; se o visitante perguntou quem, quando, onde ou se há, responde a isso.

Sem a flag os prompts são, letra a letra, os de antes.

Continua a não haver exemplos com objetos (o «livro» de 2026-11-20). O exemplo do gerúndio é só de verbos: «que
representa» em vez de «representando».

## O gerúndio

Sem a AMÁLIA a reescrever, as frases chegavam com mais gerúndio do que hoje: 30 em 260, contra 10 em 260 nas
respostas de hoje. O serviço passou a trocá-lo por «a» com o infinitivo, que é o que os prompts pedem: «um anjo
segurando uma coroa» fica «um anjo a segurar uma coroa» (`_without_gerund`). Só mexe em palavras em minúsculas, e
deixa como estão as que não são gerúndio ou que em Portugal se escrevem assim («quando», «tendo», «incluindo»,
«datando do século XVI», «destacando-se»). As 30 frases trocadas na medição foram lidas uma a uma e ficaram bem.

## Medição

`scripts/medir_frases_dos_cartoes.py x2`, a 2026-11-24, com os dois modelos reais e o streaming ligado. Os mesmos
20 pedidos do `docs/rascunho-e-reescrita.md` (cartões fixos), duas vezes cada: 40 respostas, 260 cartões.

| | Hoje (lista no texto) | Frases nos cartões | Com a AMÁLIA a reescrever a abertura |
|---|---|---|---|
| Tempo mediano (máximo) | 4,1 s (5,8 s) | 0,8 s (1,0 s) | 1,3 s (2,8 s) |
| Texto enviado por streaming igual ao final | 40 de 40 | 40 de 40 | 40 de 40 |
| Respostas com lista ou negrito no texto | 40 | 0 | 0 |
| Tamanho mediano do texto | 922 letras | 113 letras | 123 letras |
| Cartões com frase do modelo | não se aplica | 260 de 260 | 260 de 260 |
| Palavras por frase, mediana (máximo) | 14 | 20 (29) | 20 (29) |
| Letras por frase, mediana (máximo) | 82 | 122 (163) | 122 (163) |
| Frases com gerúndio que a ficha não tem | 10 | 0 | 0 |
| Frases com o n.º de inventário, ou a falar da visita | 0 | 0 | 0 |
| Frases com alguma palavra que a ficha não tem | 140 | 116 | 115 |

Nenhuma frase passou dos 220 caracteres; 18 das 260 passaram das 25 palavras (o máximo foi 29).

Mais cinco pedidos diretos, com cartões de tipos misturados («Há pinturas de Domingos Sequeira?», «Tens
tapeçarias?», «De que século são os biombos?», «Mostra-me esculturas em madeira.», «Há pinturas com cães?»), duas
vezes cada: 3,2 s hoje, 0,8 s com as frases nos cartões; streaming igual ao final em 10 de 10.

Em «Ver mais resultados», as frases de uma página de 8 cartões demoram entre 0,4 e 0,8 s. Hoje essa página não chama
nenhum modelo.

### As frases dizem o que a peça é, mesmo não sendo o pedido

Nos pedidos que nomeiam um tipo, 20 cartões eram de outro tipo: um desenho, uma gravura e um biombo em «pinturas com
barcos»; três desenhos em «Há pinturas de Domingos Sequeira?»; uma moldura em «esculturas em madeira». As 20 frases
dizem o que a peça é («É um desenho de Domingos Sequeira, de 1799–1800…», «É uma moldura dourada em madeira…»).

### Frases que dizem o que a ficha não tem

Contaram-se as frases com alguma palavra, ou número, que não está na ficha que o modelo viu nem no pedido. É uma
lista de candidatas: a maior parte são outras palavras para a mesma coisa («mulher» por «figura feminina»).

Nas 316 frases do modo novo (os 25 pedidos, duas vezes) há 227 diferentes. Dessas, 84 têm uma palavra que a ficha não
tem; 60 são só outra maneira de dizer; as outras 24 foram lidas contra a ficha:

- **5 dizem uma coisa errada:** «Bombo articulado» (é um biombo); uma senhora que «repousa sobre um tapete verde»
  (na ficha, a mão apoia-se a uma mesa coberta por um tapete); uma senhora «sentada» (a ficha não o diz); «cenas de
  porto» num biombo (a legenda diz «barco de grande porte»); decoração «sob vidro» (é sob o vidrado).
- **5 acrescentam uma leitura que a ficha não faz:** «a simbolizar a fugacidade da vida» (uma Vanitas), «em estilo
  oriental», «motivos tradicionais», «cenas de vida asiática», «cabelo solto».
- **14 estão certas:** dizem por outras palavras o que a ficha diz.

São 10 em 227 (4%). Nenhuma inventa um autor, uma data ou um material. As respostas de hoje têm mais candidatas (140
contra 116 nas mesmas 260) e não foram lidas uma a uma.

### A abertura

As 50 aberturas do modo novo têm no máximo 25 palavras; 48 são uma frase só. Nenhuma tem lista, negrito ou gerúndio.

- **6 dizem uma coisa que as fichas não suportam:** «os biombos são do século XVII» (três dos quatro são de
  1593–1601), duas vezes; «peças japonesas, do século XVII» (há peças dos séculos XVI–XVII), duas vezes; «de autores
  portugueses»; «todas em mármore branco» (uma é de cerâmica).
- **2 dizem títulos de peças,** contra a instrução.

Com a AMÁLIA a reescrever a abertura: mudou as 50, quase sempre trocando palavras por sinónimos; em 10 juntou uma
apresentação, o nome do museu ou um convite («Sou a Amália, a sua guia nesta visita virtual…», «mestre português do
Renascimento», que as fichas não dizem); 12 passaram a ter duas ou mais frases; e demora mais 0,45 s. Por isso fica
desligada.

## Limites

- **A abertura arredonda.** Responde ao conjunto das peças numa frase, e nas datas escolhe um século quando as peças
  não cabem num só. O cartão de cada peça tem a data certa.
- **A abertura dá por certo o que foi pedido.** Em «esculturas em mármore branco» diz que as há, mesmo havendo uma
  de cerâmica entre os cartões; é o cartão dessa peça que o diz.
- **Uma frase em cada 23 diz mais do que a ficha** (10 em 227), e a contagem só apanha o que usa uma palavra que a
  ficha não tem.
- **Nestas respostas já não fala a AMÁLIA.** O texto e as frases são do modelo de pré-processamento.
- **A conversa guardada fica só com a abertura.** As perguntas de seguimento («e a segunda?») leem os cartões
  mostrados, não o texto, e não mudam.
- **A abertura do autor acaba em dois pontos** («…em «Ver mais resultados»):»), como hoje; os cartões vêm a seguir.
- **Sem frase do modelo** (falha ou demora mais de `CHAT_DRAFT_TIMEOUT_SECONDS`) fica a primeira frase da descrição
  da ficha; sem descrição, o cartão fica sem `card_summary`.
- **Em inglês** os prompts existem e foram vistos num pedido só; a rede do gerúndio não atua.
- **Não muda** a pesquisa, os modelos, os embeddings nem os índices.

## Testes

`tests/test_card_summaries.py` (20 testes): o contrato, o rascunho sem lista, a frase que falha, as peças do mesmo
conjunto, o limite dos 220 caracteres, o gerúndio, os prompts, a pesquisa de ponta a ponta (com e sem a AMÁLIA, por
autor, com um cartão, com as flags desligadas) e «Ver mais resultados».

## Para ligar

```
CHAT_CARD_SUMMARIES=true
```

Precisa de reinício do backend e do frontend do ramo `lista-resultados-descricao`.
