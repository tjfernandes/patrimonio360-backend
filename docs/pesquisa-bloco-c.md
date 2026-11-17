# Pesquisa do chat — Bloco C (2026-11-17)

Terceiro bloco do plano de correção da pesquisa do chat no MNAA: o Qwen lê cada mensagem antes da pesquisa.
Duas flags novas, desligadas por omissão. Nada disto está ligado em produção.

## O que mudou

| Flag | O que faz | Onde |
|---|---|---|
| `CHAT_REQUEST_READING` | O modelo de pré-processamento lê a mensagem com a conversa anterior e devolve a língua, se é um seguimento e o pedido reescrito por inteiro | `build_request_reading_prompt`, `_read_request_with_llm` e `_apply_request_reading`, chamados a seguir a `_derive_context_policy` |
| `CHAT_CATALOGUE_ANALYTICS_STRICT_SUBJECT` | O assunto de uma estatística procura também no autor e exige todas as palavras | `_aggregate_artifacts_sync` em `opensearch_client.py` |

O que a leitura faz com cada campo:

- **Seguimento.** Substitui a decisão das expressões regulares. Sem conversa anterior nunca há seguimento.
- **Pedido.** Entra como reescrita e passa pelo guarda-corpo que já existia (`_resolve_contextual_retrieval_query`): todas as palavras de quatro ou mais
  letras têm de vir da mensagem, das últimas mensagens do visitante ou dos cartões do turno anterior (título, n.º de inventário, autores).
  Se o guarda-corpo recusar, o fluxo segue como hoje. As estatísticas também recebem o pedido reescrito.
- **Língua.** Se for português ou inglês e diferente da língua da interface, a resposta sai nessa língua. Só essa resposta; a interface não muda.

A leitura não corre com uma peça selecionada nem nos percursos. Se a chamada falhar ou a resposta vier mal formada, nada muda.

Testes: `tests/test_request_reading.py` (22 testes) e os 120 casos de medição em `tests/request_reading_cases.py`. A suíte completa passa com 617 testes.

## Como foi medido

### O prompt, com o modelo real

120 casos (90 da checklist com a conversa certa, 30 paráfrases de controlo), 3 execuções cada, com o prompt tal como está no código:

| | Checklist (90) | Controlo (30) |
|---|---|---|
| Seguimento | 90/90 | 30/30 |
| Língua | 90/90 | 30/30 |
| Pedido reescrito | 90/90 | 28/30 |
| Mesma resposta nas 3 execuções | 86/90 | 29/30 |

Latência da chamada: mediana 217 ms, percentil 90 de 263 ms, máximo 356 ms.
As duas falhas («Desse autor há mais?», «who is it by?») pedem o autor da peça anterior, e a conversa de teste só tinha títulos.
Em produção a conversa dada ao modelo leva o autor de cada cartão.

### A sessão da checklist, de ponta a ponta

115 pedidos repetidos fora da produção, com as flags dos Blocos A e B ligadas: uma vez sem a leitura e duas com ela.

| Pergunta | Sem a leitura | Com a leitura |
|---|---|---|
| 67 «Quem o pintou e quando?» | 8 retratos sem relação; responde Cranach | Pesquisa «Quem pintou as Tentações de Santo Antão e quando?»; responde Bosch, 1505–1506 |
| 84 «And who made it?» | 8 peças de Santo André | Pesquisa «Who made the Tentações de Santo Antão?»; responde Bosch, em inglês |
| 42 «Is the Dürer a national treasure?» | Certa, mas em português | Em inglês |
| «Show me the custódia de Belém» | Em português | Em inglês |
| 66 «Qual delas é a mais antiga?» | «Não encontrei peças… que correspondem a "delas"» | «Visitação» (4 Pint), 1520–1525, entre as 23 obras de Gregório Lopes |

- A 66 precisa das duas flags. Só com a leitura, a estatística respondia «Missa de São Gregório»: procurava «gregorio lopes» sem o autor e bastava-lhe uma das palavras.
- Das 49 perguntas «OK» da checklist, 46 ficaram iguais em tipo de resposta e cartões. As três que mudam são seguimentos que já funcionavam
  (65 «E do século XVI?», 68 «Só as de prata.», 72 «E cadeiras?»): continuam certos, com o pedido escrito de outra forma e cartões um pouco diferentes
  (a 68 passa de 7 para 4 cartões, todos de prata).
- A leitura reescreveu a pesquisa em 7 dos 115 pedidos.
- As duas execuções com a leitura deram os mesmos cartões em 114 de 115 pedidos. A exceção é a 69 «E onde estão na visita?» (2 e 5 cartões).
- Tempo total da sessão: 467 s sem a leitura, 483 s e 476 s com ela.
- As outras oito respostas de estatística da sessão ficaram iguais com a segunda flag (os assuntos têm uma só palavra).

## O que não entra neste bloco

- **Ajuste de percurso (64 «Encurta o percurso para 5 peças.»).** O `adjust_route` só sabe encurtar, limitar a uma sala e trocar uma paragem, e é chamado pelo widget.
  Ligar-lhe um pedido em texto («para 5 peças», «tira a segunda») pede um desenho próprio. O campo `ajuste_percurso` ficou fora do prompt.
- **Sugestões sem assunto** («coisas bonitas para ver», «Qual é o quadro mais famoso?»). Continuam como estavam (Bloco D).

## Por resolver

- **O guarda-corpo recusa reescritas com uma palavra nova de quatro ou mais letras** («obras de Bosch» quando ninguém disse «obras»). Nesses casos fica o comportamento de hoje.
- **«E a peça mais recente?»** e outros seguimentos de estatística dependem de a reescrita trazer o assunto; não foram medidos de ponta a ponta além da 66.
- **69** varia entre execuções.
- Na 84 a resposta diz que as duas pinturas são Tesouro Nacional; é texto da AMÁLIA, não da leitura.

## Para ligar

Juntar ao `.env` e reiniciar o backend:

```
CHAT_REQUEST_READING=true
CHAT_CATALOGUE_ANALYTICS_STRICT_SUBJECT=true
```
