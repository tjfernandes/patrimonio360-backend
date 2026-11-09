# Percursos temáticos — dados (Fase 0 e Fase 1)

Tudo o que está aqui é offline e só **lê** o OpenSearch. O chat só usa estes
dados atrás de `CHAT_ROUTE_PLANNING=true` (ver `app/services/route_planning/`).

## Fase 0 — grafo e paragens

```
python tools/percursos/extrair_grafo_visita.py mnt mnsr mnaz mj cmag mnaa
python tools/percursos/construir_paragens.py   mnt mnsr mnaz mj cmag mnaa
```

- `poi_tours/tour-graph-<museu>.json` — nós (panoramas: `panorama_key`,
  `playlist_index`, `label`, `subtitle`, `room`, `component`), arestas
  (`source`, `target`, `kinds` ⊆ {adjacent, hotspot, inferred}, `yaw`, `pitch`,
  `distance`, `overlay_id`, `enabled`, `inferred`) e `overlays` (direção de cada
  hotspot no seu panorama). As seis visitas são uma só componente ligada
  (o MNAA tem um panorama órfão fora da playlist).
- `poi_tours/tour-stops-<museu>.ndjson` — um registo por (peça, panorama);
  `poi_tours/tour-stops-<museu>.report.json` — cobertura. Campos do registo:

| campo | significado |
|---|---|
| `stop_id` | `<museu>:<artifact_id>@<panorama_key>` |
| `museum_slug`, `artifact_id`, `inventory_number`, `inventory_key`, `title` | a peça (`inventory_key` = n.º sem maiúsculas/espaços/pontuação) |
| `panorama_key`, `room`, `playlist_index`, `panorama_known` | onde está na visita |
| `placements[]` | `{kind: hotspot\|object, overlay_id\|object_id, yaw, pitch, bbox, polygon, area_fraction, placement_score, accepted_by}` — hotspots primeiro, depois por área |
| `kinds`, `placement_count` | tipos presentes neste panorama; em quantos panoramas a peça aparece |
| `best_placement` | `true` no panorama que melhor mostra a peça (hotspot primeiro, depois a maior `area_fraction`); uma vez por peça |
| `national_treasure` | do mapa dos POIs |
| `catalogue_found`, `catalogue{}` | cópia da ficha `_v4`: title, description, technique, support_or_material, date_or_period, start_year, end_year, start_century, end_century, creators, production_center, category, super_category, inventory_type, dimensions_cm, in_tour, image_count, image_paths, image_ids, detail_url |

Objetos: com verificação VLM no ficheiro (`accepted_by` presente) entram os
`vlm`/`hotspot` com score ≥ 0,70; sem verificação (os outros 5 museus, base de
27/10) entram os de cosseno ≥ 0,80 (o limiar do «Ver na tour» por objetos).

## Fase 1 — atributos visuais (VLM) e índice `cultural_heritage_tour_stops_v1`

```
python tools/percursos/enriquecer_paragens.py mnaz --amostra 20 --csv   # validação
python tools/percursos/enriquecer_paragens.py mnaz                      # todas as peças
```

Vocabulário fechado em `app/config/visual_vocabulary.json`. Saída
`poi_tours/tour-stops-visual-<museu>.ndjson`, **um registo por peça** (o
`best_placement`), com este contrato (é o que o indexer lê):

```json
{
  "museum_slug": "mnaz", "artifact_id": "raiz:movel:228345", "inventory_number": "MNAz 5969 Az",
  "stop_id": "mnaz:raiz:movel:228345@panorama_…",
  "inputs": {"photos": ["Images/…/x.jpg"], "in_situ": {"stop_id": "…", "panorama_key": "…", "kind": "object|hotspot", "crop_path": "tmp/percursos/mnaz/<inventory_key>.jpg"} },
  "visual": {
    "colors": [{"name": "azul", "share": 0.6}, {"name": "branco", "share": 0.4}],
    "motifs": ["vegetalista"], "iconography": ["cena religiosa"],
    "figures": {"people": 2, "animals": ["cão"]},
    "text_inscriptions": "legenda em latim" , "materials_visual": ["cerâmica vidrada"],
    "techniques_visual": ["pintura sob vidrado"], "style_period_visual": ["barroco"],
    "scale_class": "parede", "condition_visible": "lacunas",
    "one_line": "Painel azul e branco com cena de caça.",
    "look_for": ["o cão no canto inferior esquerdo"],
    "in_situ": {"lighting": "natural lateral", "neighbors": "outros painéis", "visibility": "boa"}
  },
  "reconciled": {
    "pigment_hints": [{"name": "azul-cobalto", "evidence": "catalogue+visual", "confidence": 0.8}],
    "style_period": ["barroco"],
    "lay_keywords": ["azul e branco", "azulejos antigos"]
  },
  "model": "Qwen/Qwen3-VL-8B-Instruct-FP8", "prompt_version": "percursos-v1",
  "generated_at": "2026-11-09T12:00:00+00:00", "errors": []
}
```

Todos os valores de listas fechadas vêm do vocabulário (em PT, minúsculas);
`look_for` e `one_line` são observações do que se vê, nunca factos
históricos; a passagem B nunca contradiz a data do cartão.

Indexer (`patrimonio360_indexer`): `mappings/museum_tour_stops_v1.json` e
`scripts/ingest_tour_stops.py --stops-source … --visual-source …`. Um documento
por registo de `tour-stops`, `_id = stop_id`, com os campos acima em
`catalogue.*` e `visual.*` (listas fechadas como keyword), `enriched_text`
(title, one_line, motifs, iconography, cores, lay_keywords, description) e
`enriched_embedding` (Qwen3-Embedding-8B, 4096-d, l2, sem instrução — como o
`text_embedding` do `_v4`, para a query do backend ser compatível). Catálogo e
VLM ficam sempre em campos separados. Criar o índice é uma escrita no
OpenSearch: só com autorização.

## Fase 1 — validação e avaliação

- `--amostra 20 --csv` por museu → `tools/percursos/validacao/amostra-<museu>.csv`
  (uma linha por peça e por campo de lista fechada, para dois anotadores).
- `benchmarks/percursos/temas.json`: 30 temas × {mnaz, mnaa}, relevância 0/1/2
  por paragem (anotação do Tiago); `tools/percursos/avaliar_temas.py` dá nDCG@8
  e recall@20 em quatro variantes (texto original, texto→imagem, atributos VLM,
  híbrido RRF).
