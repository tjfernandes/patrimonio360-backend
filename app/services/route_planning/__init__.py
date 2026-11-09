"""Percursos temáticos gerados em conversa (CHAT_ROUTE_PLANNING, 2026-11-09).

O visitante pede «faz-me um percurso sobre azul-cobalto no século XVII» e a
Amália devolve um percurso de 3 a 10 paragens pela visita virtual, ordenado
pelo caminho mais curto no grafo da visita a partir de onde o visitante está,
com um cartão por paragem, o alvo de navegação («Ver na tour»), o porquê de
cada peça e o que olhar. Depois o frontend pede a narração de cada paragem com
a intenção `describe_artifact` e `metadata.route_stop`.

Dados (offline, tools/percursos/): grafo da visita `poi_tours/tour-graph-<museu>.json`,
paragens `tour-stops-<museu>.ndjson` enriquecidas pelo VLM e indexadas em
`cultural_heritage_tour_stops_v1` (indexer). Nada aqui corre com a flag
desligada e nada toca no retrieval, nos prompts nem nos índices do chat atual.

Módulos: `graph` (grafo e distâncias), `theme` (intenção por regras e facetas
do tema), `search` (consultas ao índice de paragens), `candidates` (os quatro
ramos + RRF), `planner` (escolha e ordem das paragens, determinístico),
`store` (cache dos percursos por route_id), `service` (resposta do chat,
ajustes e narração).
"""
