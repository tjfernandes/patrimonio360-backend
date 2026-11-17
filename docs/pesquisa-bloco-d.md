# Pesquisa do chat — Bloco D, primeira parte (2026-11-17)

O Bloco D do plano junta o que ficou «por desenhar e medir». Esta primeira parte é uma só alteração: dar mais fichas a escolher ao Qwen.
Uma flag nova, desligada por omissão. Não está ligada em produção.

## O que mudou

| Flag | O que faz | Onde |
|---|---|---|
| `CHAT_PRESELECT_CARDS_CANDIDATES` | Quantas fichas da pesquisa o modelo de pré-processamento vê para escolher os cartões da primeira página. 0 (omissão): as 8 da página, como antes. Com 15: escolhe entre as 15 primeiras e mostram-se as escolhidas, pela ordem da pesquisa, até encher a página | `_preselect_candidate_count`, `_cap_preselected_cards` e a chamada de `_preselect_cards` em `chat_service.py` |

- O «ver mais» salta as fichas que a primeira página já mostrou (as escolhidas para lá da 8.ª posição).
- Se a escolha falhar (erro, tempo esgotado, resposta inválida), ficam as 8 fichas da página, como hoje.
- O prompt da escolha é o mesmo; só recebe mais cartões. A AMÁLIA continua a ver só os escolhidos.

Testes: `tests/test_search_fixes_block_d.py` (8 testes). A suíte completa passa com 625 testes.

## Como foi medido

A sessão da checklist de 2026-11-16 (115 pedidos) repetida fora da produção, com as flags dos Blocos A, B e C ligadas:
uma vez com 8 fichas e duas com 15. As duas com 15 deram os mesmos cartões nos 115 pedidos.

| Pergunta | 8 fichas | 15 fichas |
|---|---|---|
| 6 «Fala-me dos Painéis de São Vicente.» | 5 cartões, 3 painéis | 8 cartões, os 6 painéis |
| 85 «paineis de sao vicente de fora» | 1 cartão (Veneração de S. Vicente) | 4 painéis |
| 87 «aquela peça de ouro grande, tipo torre, com muitas figurinhas» | 1 cartão (Relicário) | 4 cartões, com a Custódia de Belém em 1.º |
| 29 «Pinturas com animais.» | 1 cartão | 3 cartões |
| 36 «Mostra-me pintores flamengos.» | 3 cartões | 5 cartões |
| 21, 22, 19, «pinturas renascentistas», «retratos masculinos» | 5 ou 6 cartões | 8 cartões |
| 89 «Qual é o quadro mais famoso?» | 0 cartões | 1 cartão |

- Em 72 dos 115 pedidos o modelo viu mais de 8 fichas. Escolheu em média 6,2; em 26 escolheu mais de 8 e mostraram-se as 8 primeiras.
- Listas vazias: 12 com 8 fichas e 12 com 15.
- Tempo da escolha: de 218 ms para 384 ms em média. A sessão inteira: 470 s com 8 fichas, 495 s com 15.
- Das 49 perguntas «OK» da checklist, 44 ficaram iguais. As cinco que mudam continuam certas, com mais ou menos um cartão
  (31, 33, 68, 72, 82).

### O que piora

- **91 «Quero ver pinturas flamengas com santos, mas não as do Bosch.»** Passa de 6 cartões a 0: com 15 fichas à frente, o modelo diz que nenhuma corresponde.
  As fichas não têm a escola do pintor, por isso nenhuma das seis anteriores estava confirmada como flamenga; ainda assim, a resposta deixa de mostrar pinturas de santos.
- **79 «What are the must-see pieces here?»** e **56 «Como vou daqui à Pintura Europeia?»** já falhavam; passam a mostrar 8 cartões sem relação em vez de 2 e 4.
- **68 «Só as de prata.»** Mostra 8 cartões, mas o texto da AMÁLIA fala de «apenas uma peça». É o texto, não a escolha.

## O resto do Bloco D: o que falta e porquê

Nada disto foi feito nesta parte.

| Tema | Perguntas | O que trava |
|---|---|---|
| Sugestões sem assunto | 4, 79, 88, 89, 58 | Reconhecer o pedido falhou quatro vezes. A quarta, medida agora: tratar como sugestão só quando o Qwen o assinala **e** a escolha de cartões vem vazia. Não serve: das quatro sugestões da checklist, só a 4 fica sem cartões; a 79 e a 88 ficam com 8 cartões sem relação e a 89 com 1 |
| Navegação fora da lista de palavras | 56 | As três redações de intenção medidas no plano não acertaram esta pergunta e estragavam outras |
| Peças por sala | 54 | As peças sem hotspot não têm `overlay_id` nem panorama; mostrá-las numa sala mexe no formato da resposta e no frontend |
| Ajuste de percurso por texto | 64 | O `adjust_route` só encurta, limita a uma sala e troca uma paragem; «para 5 peças» e «tira a segunda» pedem ações novas |
| Dados das fichas | 19, 36, 91 | Material das esculturas e escola dos pintores não estão no índice. Enriquecer é escrever no índice |

## Para ligar

Juntar ao `.env` e reiniciar o backend:

```
CHAT_PRESELECT_CARDS_CANDIDATES=15
```
