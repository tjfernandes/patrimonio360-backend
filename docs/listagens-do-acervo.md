# Listagens do acervo (2026-11-17)

Uma flag nova, desligada por omissão. Não está ligada em produção.

## O problema

«Que autores há neste museu?», «que tipos de peças tem o museu?», «que estilos existem?» iam para a pesquisa normal: o visitante
recebia oito peças e um texto escrito a partir delas. Estas perguntas não pedem peças; pedem o que existe no museu inteiro.

## O que mudou

| Flag | O que faz |
|---|---|
| `CHAT_CATALOGUE_LISTINGS` | Reconhece o pedido de listagem, agrega o campo em todas as fichas do museu (ou da visita) e responde por modelo de texto, com contagens e sem cartões |

O que se lista:

| Pedido | De onde vem a resposta |
|---|---|
| autores | `creator_ids`, com o nome como o catálogo o escreve; os 15 com mais peças e o total de autores |
| tipos de peças | `category`, todas as categorias |
| materiais | `support_or_material`, os 15 mais frequentes |
| técnicas | `technique`, as 12 mais frequentes |
| locais de fabrico | `production_center`, os 15 mais frequentes |
| séculos ou épocas | a contagem por século que já existia na analítica |
| estilos | as fichas não registam estilo: a resposta diz isso e o que se consegue listar |

- «nesta visita» limita às peças da visita.
- Se a mensagem disser uma categoria do museu («que autores de pintura há?»), a listagem limita-se a ela.
- As respostas dizem em quantas fichas o campo está preenchido.
- Só para mensagens que não são seguimento; as respostas rápidas que já existiam (sala, analítica, cumprimentos) vêm primeiro.
- Se algum passo falhar, a mensagem segue para a pesquisa, como hoje.

## Como se reconhece o pedido

O modelo de pré-processamento responde a duas perguntas curtas (`build_listing_request_prompt` e `build_listing_guard_prompt`):

1. «Que resposta espera o visitante?», com nove opções por letra (peças; autores; tipos de peças; materiais; técnicas; locais;
   séculos; estilos; outra coisa).
2. Só se a primeira disser que é uma lista: «A mensagem nomeia alguma coisa?». Só passa se não nomear.

Depois, sem modelo, compara-se a mensagem com os valores que o catálogo desse museu tem no campo que se ia listar. Se a mensagem
disser um deles («peças vindas do Japão», «peças de seda»), o visitante quer as peças com esse valor e a mensagem segue para a
pesquisa. É uma comparação com dados e não uma decisão: o modelo foi medido primeiro e deixava passar 29 de 198 pesquisas deste
tipo.

## Como foi medido

Qwen3-VL-8B, temperatura 0. Casos em `tests/listing_request_cases.py`: 70 listagens (43 usadas para afinar, 27 escritas depois),
198 pedidos normais (69 escritos para parecerem listagens, mais as 129 mensagens distintas da checklist, do benchmark e dos casos
de leitura) e 198 pesquisas do tipo «que peças de X há?».

### Redações da primeira pergunta

| Redação | Listagens reconhecidas | Pedidos normais apanhados |
|---|---|---|
| Lista de características numa só chave | 46 de 54 | 124 de 169 |
| A mesma, com a restrição a parte das peças | 49 de 54 | 152 de 169 |
| «O que quer receber?» A/B/C e depois a característica | 18 de 54 | 1 de 169 |
| A/B/C com exemplos na opção das listas | 49 de 54 | 54 de 169 |
| Resposta esperada por letra, uma opção por listagem (a que ficou) | 47 de 54 | 15 de 169 |
| A mesma, com as opções reescritas («todos os…») | 26 de 54 | 7 de 169 |

Com a segunda pergunta, os pedidos normais apanhados descem de 15 para 3. Duas outras segundas perguntas não serviram: «a que se
refere a mensagem?» dizia «a uma característica» a 43 das 49 listagens, e pedir para copiar o que a mensagem nomeia devolvia
«autores» ou «técnicas».

### O nome do museu

Com «Museu Nacional do Traje» ou «do Azulejo» no prompt, «Que autores há neste museu?» era lida como pedido de peças; com o MNAA,
o Soares dos Reis e a Casa-Museu, como listagem. Os prompts ficaram sem o nome do museu, e a decisão depende só da mensagem.

### Resultado final

Com as duas perguntas e a comparação com os valores do catálogo, pelo código do serviço:

| | MNAA | Traje | Soares dos Reis |
|---|---|---|---|
| Listagens respondidas | 63 de 70 | 63 de 70 | 63 de 70 |
| Pedidos normais que acabam em listagem | 4 de 198 | 4 de 198 | 4 de 198 |
| «Que peças de X há?» que acabam em listagem | 3 de 198 | 7 de 198 | 7 de 198 |

- A comparação com os valores não tira nenhuma das 70 listagens.
- Seis das sete listagens que falham dizem «as peças» como sujeito («de que materiais são feitas as peças?», «as peças vêm de
  onde?»); a sétima é «Podes dizer-me os autores que há?». Seguem para a pesquisa, como hoje.
- Os quatro pedidos normais que escapam: «Há peças barrocas?», «Há pintores portugueses?», «Which paintings are from Flanders?»,
  «Are there any Flemish painters?».
- As pesquisas que escapam dizem uma palavra que o catálogo desse museu nunca escreve («linho» e «bronze» no Traje e no Soares
  dos Reis) ou um período por extenso («época medieval»).

### Pelo serviço inteiro

Fora da produção, com a configuração de produção e a flag ligada:

- 30 perguntas de listagem em três museus (dez em cada): todas respondidas como listagem, tempo mediano 0,7 s.
- 8 pesquisas normais nos mesmos museus: todas seguiram para a pesquisa.
- Repetição da checklist do MNAA (115 pedidos) com a flag ligada: nenhum pedido foi respondido como listagem.
- A mesma repetição com a flag desligada ficou a meio (a máquina reiniciou aos 63 pedidos). Nesses 63, 62 têm os mesmos cartões
  com e sem a flag. O outro («Quem pintou o São Jerónimo?») tem a mesma pesquisa e difere na escolha de cartões do modelo (11
  fichas em vez de 15), numa altura em que havia outras chamadas ao modelo a correr.
- Uma mensagem normal leva só a primeira pergunta ao modelo; a segunda só se faz quando a primeira diz que é uma lista. O custo em
  tempo por mensagem não ficou medido em separado: as duas repetições correram com carga diferente no modelo.

## Limites conhecidos

- A decisão é do Qwen, e erra: cerca de uma listagem em dez segue para a pesquisa e cerca de dois pedidos normais em cem recebem
  uma listagem.
- Os valores são os do catálogo, sem normalização: «Sequeira, Domingos António de» e «Sequeira, Domingos António de (1768-1837)»
  contam como dois autores; «Autor desconhecido» e «Não identificado» aparecem na lista; «La granja de St. Ildefonso» e «La Granja
  de St.Ildefonso» são dois locais.
- No Museu do Traje o material só está preenchido em 14 de 9645 fichas (o catálogo escreve-o na técnica), e a resposta diz isso.
- Perguntas limitadas por outra coisa que não a categoria («que escultores há?», «pintores portugueses») respondem pelo museu todo.
- Conjuntos e exposições ficaram de fora: essas opções faziam o modelo apanhar perguntas sobre salas e percursos.

## Para ligar

Juntar ao `.env` e reiniciar o backend:

```
CHAT_CATALOGUE_LISTINGS=true
```
