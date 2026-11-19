# Peça implícita pelo modelo (2026-11-19)

Uma flag nova, `CHAT_IMPLICIT_ARTIFACT_MODEL`, desligada por omissão. Precisa de `CHAT_IMPLICIT_ARTIFACT_CONTEXT`. O
frontend não muda.

## O problema

Depois de «Encontra os painéis de São Vicente de fora», o visitante perguntou «O que representa o Painel dos Frades?» e a
resposta falou do Painel da Relíquia («O Painel da Relíquia, ou Painel dos Frades, representa…»).

A pergunta nomeava a peça, mas uma regra trocou-a antes de chegar ao modelo. A peça implícita (`_resolve_implicit_artifact`)
procura um cartão visível cujo título comece pela palavra que vem depois de «o/a»: «Painel» bateu com «Painel da
Relíquia», o único cartão a começar assim. O Painel dos Frades estava no ecrã, na lista do conjunto por baixo do cartão,
mas o frontend só envia os cartões. A mensagem seguiu como «O que representa o Painel dos Frades? (refiro-me à peça «Painel
da Relíquia» (1362 Pint))», com a peça errada escolhida.

## O que mudou

Com a flag, a peça de que a mensagem fala resolve-se em três passos (`_resolve_implicit_artifact_with_model`):

1. **Procuras exatas, pelas regras.** A posição no ecrã («o segundo», «a última»), só entre os cartões, e o n.º de
   inventário, também entre as peças dos conjuntos.
2. **O nome de uma peça, pelo modelo.** O modelo de pré-processamento recebe a lista das peças mostradas, por letras e
   pela ordem do ecrã, e diz de qual fala a mensagem, ou de nenhuma (`build_implicit_artifact_prompt`). A lista tem os
   cartões e, a seguir ao primeiro cartão de cada conjunto, as outras peças desse conjunto. A peça aberta na visita, se não
   for um dos cartões, vai à parte com a letra X. Uma mensagem com verbo de pesquisa («procura…», «mostra…») não chega ao
   modelo.
3. **As regras de hoje, como rede.** Se o modelo não aponta nenhuma, ou falha, vale o que já havia: nome a seguir a um
   determinante e pergunta sem assunto próprio («de que é feito?»), que vai para a peça em foco.

As peças dos conjuntos vêm do backend: os conjuntos saem dos cartões que a conversa guardou e as peças de cada conjunto
leem-se do índice uma vez por conversa (até 12 por conjunto, três conjuntos).

`subject_artifact.resolved_by` passa a poder ser `model`. A pergunta segue, como antes, com «(refiro-me à peça …)».

### Porque não é tudo decidido pelo modelo

A ideia inicial era o modelo decidir tudo: peça nomeada, peça em foco ou pesquisa. Medido, não serve:

- Uma só pergunta para as três coisas acerta 50 a 61 de 68 (as regras de hoje, 61). Na melhor redação, seis dos sete erros
  são pesquisas ou outros assuntos atribuídos a uma peça («Peças de 1850», «O que é um tournure?», «Mostra-me pinturas de
  Pousão»).
- Uma pergunta só para «é sobre a peça em foco, uma pesquisa, ou outro assunto?» responde «sobre a peça» a «Encontra
  chapéus de mulher» e a «Vestidos de noiva».
- A dizer qual é a peça nomeada, o modelo é melhor do que a regra: resolve «o da relíquia», «a escultura 1082 Esc» e «E o
  Desterrado?», e distingue os painéis do conjunto.

Por isso o modelo ficou com a parte em que é melhor. A posição e o n.º de inventário ficam nas regras porque são procuras
exatas numa lista: não há nada a interpretar, e o modelo falhava «E o segundo?» em algumas redações.

## Como foi medido

Qwen3-VL-8B, temperatura 0. Casos em `tests/implicit_artifact_cases.py`: sete cenários de ecrã em quatro museus (cartões,
conjuntos, títulos repetidos, peça aberta na visita), cada caso com a peça esperada ou nenhuma. Três grupos:

- **Afinação** (68): serviu para escolher a redação. Inclui os 27 casos de afinação das regras (`test_implicit_artifact.py`).
- **Validação** (59): corrida só no fim. Inclui os 20 casos de validação das regras.
- **Confirmação** (26): escrita depois de tudo fechado. Duas escolhas (a letra X e a posição nas regras) foram feitas a
  olhar para a validação, e este grupo confirma que não ficaram afinadas a ela.

### Redações (casos de afinação, 68)

| | Certas |
|---|---|
| Regras de hoje | 61 |
| Regras de hoje, com as peças dos conjuntos na lista | 65 |
| Modelo decide tudo, lista numerada (4 redações) | 50 a 61 |
| Modelo diz a peça nomeada, lista por letras; o resto pelas regras (7 redações) | 60 a 66 |
| O mesmo, com «uma peça ou várias?» primeiro e as regras de hoje como rede | 67 a 68 |
| O mesmo, com a posição e o n.º de inventário pelas regras antes do modelo (a que ficou) | 67 |

- Com a lista numerada, o modelo respondia com o n.º de inventário («1363» em vez de «6»). Por letras deixa de acontecer,
  e a resposta com o n.º de inventário é aceite quando só uma peça o tem.
- Sem o campo «quantas» antes da letra, uma pesquisa no plural com as palavras de um cartão («Azulejos de padrão do
  século XVII») era lida como esse cartão.
- Juntar condições à resposta «nenhuma» faz o modelo recusar referências claras («E o segundo?», «Fala-me da Caridade»).
- Uma segunda pergunta de confirmação («uma peça ou várias?», quatro redações) não ajudou: a melhor trocava um erro por
  outro e a pior recusava quase tudo.

### Pelo código do serviço

`scripts/medir_peca_implicita.py`: cada caso passa por `_resolve_implicit_artifact` (hoje) e por
`_resolve_implicit_artifact_with_model`, com o Qwen real. Três execuções, iguais.

| Grupo | Casos | Regras de hoje | Com o modelo |
|---|---|---|---|
| Afinação | 68 | 61 | 67 |
| Validação | 59 | 52 | 57 |
| Confirmação | 26 | 20 | 25 |
| Total | 153 | 133 | 149 |

- Passam a estar certos 17 casos: 12 de peças de conjuntos («O que representa o Painel dos Frades?», «A cafeteira é de
  prata?», «1830 Our») e 5 de nomes que a regra não apanhava («o da relíquia é de quando?», «a escultura 1082 Esc é de que
  material?», «E o Desterrado?», «Fala-me do panorama de Lisboa», e «A Anunciação é de quem?» com a Anunciação aberta na
  visita).
- Passa a estar errado 1: «Retratos de família», com o cartão «Retrato de família» no ecrã, é lida como esse cartão.
- Ficam errados, como hoje, 3: «Há outros painéis no museu?» e «Há outras Anunciações?» (a regra da peça em foco) e
  «Quem a encomendou?» (a mesma regra, ao contrário).
- O modelo é chamado em 90 das 153 mensagens: 0,11 s de mediana, 0,51 s no máximo.

Com dados reais (a resposta de produção a «Encontra os painéis de São Vicente de fora», o OpenSearch em leitura e o Qwen):
«E o painel do infante?» dá o Painel do Infante, que é uma peça do conjunto sem cartão (hoje dá o Painel dos Frades); «o
dos pescadores é de quando?» dá o Painel dos Pescadores (hoje, nenhuma).

## Limites conhecidos

- Uma pesquisa no plural com as mesmas palavras do título de um cartão pode ser lida como esse cartão.
- A decisão «peça em foco ou pesquisa» continua nas regras, com os erros que já tinha.
- Com mais de 22 peças no ecrã o modelo não é chamado e valem as regras.
- As peças dos conjuntos só entram se a conversa ainda tiver os cartões guardados (perdem-se quando o backend reinicia).
- Mais uma pergunta ao modelo, de cerca de 0,1 s, em pouco mais de metade das mensagens com cartões no ecrã.
- Só este passo mudou. A leitura do pedido, o seguimento e as outras decisões da conversa continuam como estavam.

## Para ligar

Juntar ao `.env` e reiniciar o backend:

```
CHAT_IMPLICIT_ARTIFACT_MODEL=true
```
