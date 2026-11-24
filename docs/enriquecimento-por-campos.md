# Campos enriquecidos: regras sobre a ficha e temas (2026-11-24)

Uma flag nova, desligada por omissão. Não está ligada em produção.

## O problema

«Quantas cadeiras há?» contava as fichas em que a palavra «cadeira» aparece no título, na categoria, na técnica, no
material ou na descrição. No MNAA dava 157: entravam 33 desenhos, pinturas, esculturas e gravuras com uma cadeira
na cena. «Quantas peças da China há?» dava 64, das quais 55 eram desenhos e gravuras: «tinta-da-china», «laranja da
China». «Que materiais há?» listava o campo do material tal como o catálogo o escreve, que só está preenchido em 60%
das fichas do MNAA, com 120 formas diferentes («Madeira de carvalho», «madeira», «Papel avergoado»).

A ficha sabe mais do que o campo do material diz: «Prata fundida, cinzelada e incisa» está no campo da técnica.

## O que mudou

| Flag | O que faz |
|---|---|
| `CHAT_ENRICHED_FIELDS` | As contagens e as perguntas «a mais antiga» / «a mais recente» leem o assunto pelos campos enriquecidos, e as listagens de materiais e de técnicas mostram os valores normalizados |

Os campos enriquecidos estão num índice próprio, `OPENSEARCH_INDEX_ENRICHMENT` (por omissão
`cultural_heritage_enrichment_v1`): um documento por peça, com o mesmo `artifact_id` da ficha, sem vetores. O índice das
peças não é tocado.

| Campo | Lista | De onde vem |
|---|---|---|
| `enr_tipo` | fechada, em dois níveis («mobiliário», «cadeira») | categoria e palavra principal do título |
| `enr_materiais`, `enr_materiais_ficha` | fechada | campo do material, campo da técnica, categoria; nos objetos, também o título e a descrição |
| `enr_tecnicas`, `enr_tecnicas_ficha` | fechada | campo da técnica; nos objetos, também o título e a descrição |
| `enr_funcao` | fechada | regra sobre o tipo de objeto |
| `enr_origem_pais`, `enr_origem_regiao` | fechada | centro de produção |
| `enr_cores` | fechada | título e descrição |
| `enr_estilo` | fechada | só quando a ficha o escreve |
| `enr_tamanho` e as medidas em centímetros | cinco classes | campo das dimensões |
| `enr_temas`, `enr_temas_palavras` | fechada (nove temas) | o modelo de pré-processamento lê a ficha; ver «Temas» |

Os campos `_ficha` têm só o que veio de um campo estruturado da ficha. `enr_detalhe` guarda, para cada valor, a fonte
(`campo`, `texto` ou `regra`) e o excerto; não é pesquisável.

### Como o assunto é lido

O assunto de uma contagem («cadeiras», «esculturas madeira») é comparado com os valores que o museu tem em `enr_tipo`,
`enr_materiais`, `enr_tecnicas`, `enr_funcao`, `enr_origem_pais` e `enr_temas_palavras` (as palavras com que se pede cada
tema: «religiosas», «animais», «paisagens»). É uma comparação com dados, sem modelo
(`app/services/enriched_fields.py`):

- plurais e particípios («texteis» → «têxtil», «bordadas» → «bordado») e valores de várias palavras («pintura a óleo»);
- uma palavra que é valor em dois campos conta nos dois («tapeçaria» é tipo de objeto e é técnica);
- **todas** as palavras têm de ser valores. Se sobrar uma («cadeiras bonitas»), se o museu não tiver enriquecimento ou
  se o índice falhar, a contagem fica exatamente como sem a flag.

As peças designadas vão para a contagem que já existia como filtro por `artifact_id`, por isso o período («do século
XVIII») e o âmbito («nesta visita») continuam a aplicar-se. O registo traz `chat.enriched_fields` com o assunto, a
leitura (`enr_tipo=cadeira`) e o número de peças.

## Como foi medido

MNAA, 3 658 peças, contra o OpenSearch de produção e só em leitura. «Hoje» é a contagem por palavras; «Novo» é a dos
campos enriquecidos.

| Pergunta | Hoje | Novo | O que saiu ou entrou |
|---|---|---|---|
| Quantas cadeiras há? | 157 | 111 | saem 33 desenhos, pinturas, esculturas e gravuras com uma cadeira na cena e 8 móveis que só a mencionam (biombos, mesa, arca); saem também 5 bergères e duchesses, que são cadeiras de braços |
| Quantas peças de mobiliário tem o museu? | 376 | 320 | saem as gravuras de modelos de móveis |
| Quantas peças de prata há? | 372 | 290 | saem 40 pratos de vidro e de cerâmica (a pesquisa por palavras reduz «prata» e «prato» à mesma raiz), pinturas, esculturas e desenhos |
| Quantas peças em marfim há? | 93 | 103 | entram as miniaturas pintadas sobre marfim |
| Quantas peças da China há? | 64 | 66 | saem 58 peças, 55 delas desenhos e gravuras («tinta-da-china»); entram 60 com centro de produção na China |
| Quantos cálices há? | 38 | 14 | saem casulas, gravuras e iluminuras em que aparece um cálice |
| Quantos castiçais há? | 42 | 31 | saem desenhos e pinturas com um castiçal |
| Quantas colchas há? | 38 | 30 | saem panos de armar e uma pintura |
| Quantas esculturas há? | 331 | 299 | saem iluminuras e miniaturas |
| Quantas gravuras há? | 271 | 266 | saem desenhos que mencionam uma gravura; entram 22 gravuras sem categoria na ficha |
| Quantas peças de ouro há? | 1058 | 1017 | ver limites |
| Quantas esculturas em madeira há? | 32 | 14 | ver limites |
| Quantas tapeçarias há? / biombos / desenhos a carvão / peças de vidro | igual | igual | — |

Listagens:

| | Hoje | Novo |
|---|---|---|
| Materiais | 2 209 fichas, 120 formas: Velino (737), Papel (559), Pergaminho (286), Madeira de carvalho (174)… | 3 348 peças, 58 valores: ouro (1017), velino (739), papel (708), madeira (558), vidro (370), prata (290)… |
| Técnicas | 3 463 fichas, 1 051 formas: Pintura a têmpera e ouro (883), pintura a óleo (254), Buril (113)… | 3 479 peças, 49 valores: aplicação de ouro (1441), pintura a têmpera (971), iluminura (923), pintura a óleo (329)… |

## Validação

O Tiago conferiu, uma a uma, 200 peças do MNAA (repartidas por categoria, semente fixa) numa página com a fotografia,
a ficha e os valores atribuídos, marcando os errados: https://claude.ai/artifact/Jf3Pg1SQZXHtA5BD44To2y (coleção
`conferencias`). Demorou 42 minutos.

| Campo | Valores conferidos | Marcados errados | Certos |
|---|---|---|---|
| Tipo de objeto | 200 | 0 | 100% |
| Materiais | 303 | 2 | 99% |
| Técnicas | 470 | 1 | 100% |
| Função | 110 | 0 | 100% |
| Origem | 90 | 0 | 100% |
| Tamanho | 198 | 0 | 100% |
| Cores escritas na ficha | 242 | 0 | 100% |
| Temas, tal como o modelo os deu | 519 | 56 | 89% |

Os dois materiais errados são «ouro» em fólios de texto cuja técnica na ficha é «pintura a têmpera e ouro».

## Temas

`tools/enriquecimento/temas.py` pergunta ao modelo de pré-processamento, para cada um de 16 temas, se a peça o tem; em
cada «sim» o modelo copia da ficha o excerto que o mostra, e o «sim» só conta se o excerto existir na ficha. A
resposta vai presa a um esquema JSON. O que entra no índice decide-se depois, sem modelo, pelo que a validação mostrou
(só temas com pelo menos 95% de valores certos em pelo menos 15 conferidos):

| Tema | Como entra | Na validação |
|---|---|---|
| Religioso | modelo | 86 de 86 |
| Vida quotidiana | modelo | 37 de 38 |
| Animais | modelo | 29 de 30 |
| Paisagem e natureza | modelo | 19 de 19 |
| Ciência e ofícios | modelo | 18 de 18 |
| Flora e ornamento vegetal | modelo, só fora da categoria Pintura | 71 de 72 (na Pintura: 7 de 21) |
| Ornamento geométrico | modelo, só fora da categoria Pintura | 59 de 59 (na Pintura: 3 de 18) |
| Escrita e texto | regra: o tipo de objeto é um manuscrito | 20 de 20 (pelo modelo: 33 de 41) |
| Retrato | regra: o título diz «retrato» | 8 de 8 (pelo modelo: 17 de 21) |

Com estas regras ficam 350 temas em 188 das 200 peças, 3 marcados como errados. Este número é otimista: as regras
foram tiradas das mesmas 200 peças.

Três decisões que a validação pediu:

- **«Escrita e texto» só quando a própria peça é um escrito.** Um santo com um livro na mão ou uma legenda numa pintura
  não fazem da peça uma peça de escrita. Deixou de ser o modelo a dizê-lo.
- **Flora e ornamento geométrico são temas de decoração.** Num objeto são a decoração da peça; numa pintura ou num fólio
  iluminado o modelo dava-os por um pormenor da cena ou pelas linhas que guiam o texto.
- **«Retrato» só pelo título.** Os quatro erros vinham do modelo: um rosto entalhado numa cadeira, um busto num aparador.

Ficam de fora, à espera de uma instrução melhor: Arquitetura e cidade (30 de 34), Heráldica e poder (22 de 25), Guerra e
vida militar (9 de 11), Mitologia e alegoria (8 de 9), Oriente e contactos entre culturas (6 de 7), Mar e navegação (6
de 6) e Jogo e infância (3 de 4). O Tiago marcou ainda 15 temas em falta em 14 peças.

Pela fotografia (o modelo de visão a olhar para a imagem), também conferido mas ainda fora do índice: «tem pessoas»,
«tem animais» e «tem letras» certos em 599 de 600; a cor principal em 468 de 486.

## Limites

- **Só o MNAA tem enriquecimento.** As listas (`tools/enriquecimento/vocab.py`, versão `mnaa-1`) foram feitas com as
  fichas deste museu. Nos outros a flag não muda nada enquanto não forem construídos.
- **O tipo é a palavra do título.** As 5 bergères e duchesses do MNAA não contam como cadeiras. `enr_funcao=assento`
  junta 139 peças; falta uma hierarquia de tipos.
- **«Peças de ouro» inclui 887 fólios iluminados**, porque a técnica é «pintura a têmpera e ouro». O material está
  certo; não é o que um visitante espera da pergunta.
- **Esculturas em madeira: 14.** Das 299 esculturas, 293 têm o campo do material vazio e a descrição raramente o diz.
  As regras não inventam: esta pergunta continua por resolver.
- **Estilo (6%) e origem (33%)** ficam com a cobertura que a ficha dá.
- **`in_tour`** é copiado do índice das peças, onde está desatualizado (46 peças no MNAA para 760 na visita).
- **Os temas dependem do que a ficha diz.** Uma peça cuja descrição não fala do assunto fica sem tema.
- A pesquisa normal («mostra-me cadeiras») não usa ainda estes campos: só as contagens e as listagens.

## Construir e atualizar

```bash
conda activate visitas_virtuais
cd ~/Projects/AMALIA/patrimonio360-backend
python -m tools.enriquecimento.temas --museu mnaa                # pergunta ao modelo os temas das peças que faltam (retoma sozinho)
python -m tools.enriquecimento.construir --museu mnaa            # lê as fichas, aplica as regras, junta os temas, mostra a cobertura
python -m tools.enriquecimento.indexar --museu mnaa              # simula
python -m tools.enriquecimento.indexar --museu mnaa --executar   # cria o índice se faltar e escreve
```

As regras não precisam da GPU nem de parar o backend. Os temas usam o modelo de pré-processamento que já está de pé (o
mesmo que serve produção: cerca de 1 segundo por peça com 3 a 4 pedidos em paralelo). O backend guarda os valores do museu durante dez minutos; depois de
reindexar, as contagens mudam sozinhas nesse prazo.

Testes: `python -m unittest tests.test_enriched_fields`.
