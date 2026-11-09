"""Consultas ao índice de paragens (ROUTE_STOPS_INDEX, `cultural_heritage_tour_stops_v1`).

Só leitura. Quatro ramos sobre as paragens `best_placement` do museu, com os
mesmos filtros duros (museu, `best_placement`, janela de anos quando o tema
é temporal, sala quando o ajuste pede «só esta sala»):

  lexical     BM25 em enriched_text, catalogue.description, title, lay_keywords
  semantic    kNN em enriched_embedding (Qwen3-Embedding-8B do tema)
  image       texto→imagem no índice de imagens (Qwen3-VL-Embedding-8B),
              restrito às peças com paragem; melhor imagem por peça
  attributes  cláusulas `should` com boost nas listas fechadas do VLM e do
              catálogo que batem nas facetas do tema

Cada ramo devolve [(artifact_id, score, doc)] por ordem. Os documentos vêm
sem o vetor. As funções são síncronas (o cliente opensearch-py é síncrono) e
o serviço corre-as em thread, como o OpenSearchGateway faz.
"""
from __future__ import annotations

from typing import Any

from app.services.route_planning.theme import ThemeFacets

STOP_SOURCE_FIELDS = [
    "stop_id", "museum_slug", "museum_id", "artifact_id", "inventory_number", "inventory_key", "title", "panorama_key",
    "room", "playlist_index", "best_placement", "placement_count", "kinds", "national_treasure", "placement",
    "catalogue_found", "catalogue.title", "catalogue.description", "catalogue.technique", "catalogue.support_or_material",
    "catalogue.date_or_period", "catalogue.start_year", "catalogue.end_year", "catalogue.start_century", "catalogue.end_century",
    "catalogue.creators", "catalogue.category", "catalogue.super_category", "catalogue.production_center", "catalogue.image_count",
    "visual.color_names", "visual.colors", "visual.motifs", "visual.iconography", "visual.materials_visual", "visual.techniques_visual",
    "visual.style_period_visual", "visual.scale_class", "visual.figures_people", "visual.figures_animals", "visual.one_line",
    "visual.look_for", "pigment_names", "style_period", "lay_keywords",
]

# faceta do tema -> campos do índice onde bate (keyword, dobrados pelo normalizer) e boost
ATTRIBUTE_FIELDS: dict[str, list[tuple[str, float]]] = {
    "colors": [("visual.color_names", 3.0), ("lay_keywords", 1.5)],
    "pigments": [("pigment_names", 4.0), ("visual.color_names", 1.0)],
    "motifs": [("visual.motifs", 3.0), ("lay_keywords", 1.0)],
    "iconography": [("visual.iconography", 3.0), ("lay_keywords", 1.0)],
    "materials": [("visual.materials_visual", 2.5), ("catalogue.support_or_material", 2.0), ("catalogue.technique", 1.0), ("catalogue.category", 1.0)],
    "techniques": [("visual.techniques_visual", 2.5), ("catalogue.technique", 2.5)],
    "styles": [("visual.style_period_visual", 2.0), ("style_period", 3.0)],
}


class RouteStopSearch:
    def __init__(self, client: Any, *, stops_index: str, images_index: str) -> None:
        self.client = client
        self.stops_index = stops_index
        self.images_index = images_index

    # ------------------------------------------------------------ filtros
    @staticmethod
    def hard_filters(*, museum_slug: str, facets: ThemeFacets | None, year_tolerance: int, room: str | None = None) -> list[dict[str, Any]]:
        filters: list[dict[str, Any]] = [{"term": {"museum_slug": museum_slug}}, {"term": {"best_placement": True}}]
        window = facets.year_window(year_tolerance) if facets is not None else None
        if window is not None:
            start, end = window
            # a peça sobrepõe-se à janela: começou antes do fim e acabou depois do início
            filters.append({"range": {"catalogue.start_year": {"lte": end}}})
            filters.append({"range": {"catalogue.end_year": {"gte": start}}})
        if room:
            filters.append({"term": {"room": room}})
        if facets is not None and getattr(facets, "national_treasure", False):
            filters.append({"term": {"national_treasure": True}})
        categories = [c for c in (getattr(facets, "categories", None) or []) if c] if facets is not None else []
        if categories:
            # match no subcampo de texto: apanha «Escultura/Arquitectura» e «escultura funerária: tumulária»
            filters.append({"bool": {"should": [{"match": {"catalogue.category.text": {"query": c, "operator": "and"}}} for c in categories], "minimum_should_match": 1}})
        return filters

    @staticmethod
    def _hits(response: dict[str, Any]) -> list[tuple[str, float, dict[str, Any]]]:
        out: list[tuple[str, float, dict[str, Any]]] = []
        for hit in (response.get("hits") or {}).get("hits") or []:
            source = hit.get("_source") or {}
            artifact_id = str(source.get("artifact_id") or "").strip()
            if artifact_id:
                out.append((artifact_id, float(hit.get("_score") or 0.0), source))
        return out

    # ------------------------------------------------------------- ramos
    LEXICAL_FIELDS = ["title^3", "enriched_text^2", "enriched_text.no_stem", "lay_keywords.text^2", "catalogue.description", "visual.one_line^2"]
    # só o texto do catálogo (variante «texto original» da avaliação, tools/percursos/avaliar_temas.py)
    CATALOGUE_FIELDS = ["title^3", "catalogue.description", "catalogue.technique.text", "catalogue.category.text", "catalogue.support_or_material.text"]
    # só o que o VLM viu (variante «atributos VLM»)
    VISUAL_FIELDS = ["visual.one_line^2", "visual.look_for", "lay_keywords.text^2"]

    def lexical(self, *, query_text: str, filters: list[dict[str, Any]], size: int, fields: list[str] | None = None) -> list[tuple[str, float, dict[str, Any]]]:
        text = (query_text or "").strip()
        if not text:
            return []
        body = {
            "size": size,
            "_source": STOP_SOURCE_FIELDS,
            "query": {
                "bool": {
                    "filter": filters,
                    "must": [{
                        "multi_match": {
                            "query": text,
                            "type": "best_fields",
                            "fields": list(fields or self.LEXICAL_FIELDS),
                            "operator": "or",
                        }
                    }],
                }
            },
        }
        return self._hits(self.client.search(index=self.stops_index, body=body))

    def semantic(self, *, embedding: list[float], filters: list[dict[str, Any]], size: int) -> list[tuple[str, float, dict[str, Any]]]:
        if not embedding:
            return []
        body = {
            "size": size,
            "_source": STOP_SOURCE_FIELDS,
            "query": {"bool": {"filter": filters, "must": [{"knn": {"enriched_embedding": {"vector": embedding, "k": max(size * 3, 50)}}}]}},
        }
        return self._hits(self.client.search(index=self.stops_index, body=body))

    def attributes(self, *, facets: ThemeFacets, filters: list[dict[str, Any]], size: int) -> list[tuple[str, float, dict[str, Any]]]:
        should: list[dict[str, Any]] = []
        for kind, fields in ATTRIBUTE_FIELDS.items():
            values = [v for v in getattr(facets, kind, []) if v]
            if not values:
                continue
            for field_name, boost in fields:
                should.append({"terms": {field_name: values, "boost": boost}})
        if not should:
            return []
        body = {
            "size": size,
            "_source": STOP_SOURCE_FIELDS,
            "query": {"bool": {"filter": filters, "should": should, "minimum_should_match": 1}},
        }
        return self._hits(self.client.search(index=self.stops_index, body=body))

    def all_filtered(self, *, filters: list[dict[str, Any]], size: int) -> list[tuple[str, float, dict[str, Any]]]:
        """Todas as paragens que passam os filtros duros, pela ordem da visita (ramo «treasures»)."""
        body = {"size": size, "_source": STOP_SOURCE_FIELDS, "query": {"bool": {"filter": filters}}, "sort": [{"playlist_index": "asc"}]}
        hits = self._hits(self.client.search(index=self.stops_index, body=body))
        return [(a, 1.0, d) for a, _s, d in hits]

    def artifact_ids(self, *, filters: list[dict[str, Any]], limit: int = 5000) -> list[str]:
        """As peças com paragem que passam os filtros (para restringir o ramo de imagem)."""
        body = {
            "size": 0,
            "query": {"bool": {"filter": filters}},
            "aggs": {"ids": {"terms": {"field": "artifact_id", "size": limit}}},
        }
        response = self.client.search(index=self.stops_index, body=body)
        return [str(b["key"]) for b in ((response.get("aggregations") or {}).get("ids") or {}).get("buckets") or []]

    def image(self, *, embedding: list[float], museum_id: str, artifact_ids: list[str], size: int) -> list[tuple[str, float, dict[str, Any]]]:
        """Texto→imagem: melhor fotografia por peça, só entre as peças com paragem."""
        if not embedding or not artifact_ids:
            return []
        body = {
            "size": max(size * 4, 60),
            "_source": ["artifact_id", "image_id", "local_path", "inventory_number"],
            "query": {
                "bool": {
                    "filter": [{"term": {"museum_id": museum_id}}, {"terms": {"artifact_id": artifact_ids}}],
                    "must": [{"knn": {"visual_embedding": {"vector": embedding, "k": max(size * 6, 120)}}}],
                }
            },
        }
        best: dict[str, tuple[float, dict[str, Any]]] = {}
        for artifact_id, score, source in self._hits(self.client.search(index=self.images_index, body=body)):
            if artifact_id not in best or score > best[artifact_id][0]:
                best[artifact_id] = (score, source)
        ranked = sorted(best.items(), key=lambda item: -item[1][0])[:size]
        return [(artifact_id, score, source) for artifact_id, (score, source) in ranked]

    # ------------------------------------------------------------- fichas
    def stops_by_artifact(self, *, museum_slug: str, artifact_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not artifact_ids:
            return {}
        body = {
            "size": len(artifact_ids),
            "_source": STOP_SOURCE_FIELDS,
            "query": {"bool": {"filter": [{"term": {"museum_slug": museum_slug}}, {"term": {"best_placement": True}}, {"terms": {"artifact_id": artifact_ids}}]}},
        }
        return {artifact_id: source for artifact_id, _score, source in self._hits(self.client.search(index=self.stops_index, body=body))}

    def rooms(self, *, museum_slug: str) -> list[str]:
        body = {"size": 0, "query": {"bool": {"filter": [{"term": {"museum_slug": museum_slug}}, {"term": {"best_placement": True}}]}},
                "aggs": {"rooms": {"terms": {"field": "room", "size": 200}}}}
        response = self.client.search(index=self.stops_index, body=body)
        return [str(b["key"]) for b in ((response.get("aggregations") or {}).get("rooms") or {}).get("buckets") or []]
