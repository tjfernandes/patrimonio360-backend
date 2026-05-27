from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import lru_cache
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from app.core.config import Settings, get_settings
from app.query_planning import (
    CompiledOpenSearchDSL,
    QueryExecutionResult,
    QueryPlan,
    execute_query,
)


def _import_opensearch() -> Any:
    try:
        from opensearchpy import OpenSearch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency 'opensearch-py'. Install backend requirements first."
        ) from exc
    return OpenSearch


logger = logging.getLogger(__name__)
EVENT_LABELS: dict[str, str] = {
    "opensearch.retrieve.body": "Body completo enviado ao OpenSearch para retrieval.",
    "opensearch.retrieve.response": "Resposta de retrieval recebida do OpenSearch.",
    "opensearch.image_retrieve.body": "Body completo enviado ao OpenSearch para image retrieval.",
    "opensearch.image_retrieve.response": "Resposta de image retrieval recebida do OpenSearch.",
    "opensearch.image_fetch_by_artifact.body": "Body completo enviado ao OpenSearch para fetch de imagens por artifact_id.",
    "opensearch.image_fetch_by_artifact.response": "Resposta de fetch de imagens por artifact_id recebida do OpenSearch.",
    "opensearch.artifact_fetch.body": "Body completo enviado ao OpenSearch para fetch por artifact_id.",
    "opensearch.artifact_fetch.response": "Resposta de fetch por artifact_id recebida do OpenSearch.",
    "opensearch.structured.body": "Body completo enviado ao OpenSearch para query estruturada.",
    "opensearch.structured.response": "Resposta de query estruturada recebida do OpenSearch.",
}


@dataclass(slots=True)
class OpenSearchRetrievalPage:
    results: list[dict[str, Any]]
    total: int


class OpenSearchGateway:
    """OpenSearch adapter used by chat retrieval."""

    _ALLOWED_SORT_FIELDS = {
        "inventory_number",
        "title.keyword",
        "category",
        "date_or_period",
    }

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: Any | None = None

    def _log(self, level: int, event: str, **fields: object) -> None:
        if self.settings.LOG_JSON:
            payload = {"event": event, **fields}
            prefix = EVENT_LABELS.get(event, event)
            if self.settings.LOG_JSON_PRETTY:
                logger.log(
                    level,
                    f"{prefix}\n"
                    + json.dumps(
                        payload,
                        ensure_ascii=False,
                        default=str,
                        indent=max(self.settings.LOG_JSON_INDENT, 0),
                    ),
                )
            else:
                logger.log(
                    level,
                    f"{prefix} " + json.dumps(payload, ensure_ascii=False, default=str),
                )
            return

        details = " ".join(f"{key}={value}" for key, value in fields.items())
        logger.log(level, f"{event} {details}".strip())

    def _resolve_opensearch_endpoint(self) -> tuple[str, str, int, str]:
        host_value = (self.settings.OPENSEARCH_HOST or "").strip()
        if not host_value:
            raise RuntimeError("OPENSEARCH_HOST is required for retrieval.")

        default_scheme = self.settings.OPENSEARCH_SCHEME or (
            "https" if self.settings.opensearch_use_ssl_resolved else "http"
        )

        # Allows both:
        #   localhost:9200
        #   https://localhost:9200
        #   https://motion-notes.di.fct.unl.pt:443/opensearch
        parse_value = host_value
        if "://" not in parse_value:
            parse_value = f"{default_scheme}://{parse_value}"

        parsed = urlparse(parse_value)

        if not parsed.hostname:
            raise RuntimeError(f"Invalid OPENSEARCH_HOST: {host_value}")

        scheme = parsed.scheme or default_scheme

        port = parsed.port
        if port is None:
            port = self.settings.OPENSEARCH_PORT or (443 if scheme == "https" else 80)

        url_prefix = parsed.path.strip("/")

        return parsed.hostname, scheme, port, url_prefix

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client

        OpenSearch = _import_opensearch()
        host, scheme, port, url_prefix = self._resolve_opensearch_endpoint()

        host_config: dict[str, Any] = {
            "host": host,
            "port": port,
            "scheme": scheme,
        }

        if url_prefix:
            host_config["url_prefix"] = url_prefix

        client_kwargs: dict[str, Any] = {
            "hosts": [host_config],
            "use_ssl": scheme == "https",
            "verify_certs": self.settings.OPENSEARCH_VERIFY_CERTS,
            "ssl_show_warn": self.settings.OPENSEARCH_SSL_SHOW_WARN,
        }

        if self.settings.OPENSEARCH_USERNAME and self.settings.OPENSEARCH_PASSWORD:
            client_kwargs["http_auth"] = (
                self.settings.OPENSEARCH_USERNAME,
                self.settings.OPENSEARCH_PASSWORD,
            )

        self._client = OpenSearch(**client_kwargs)
        return self._client

    def _build_filter_clauses(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        filters: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        clauses: list[dict[str, Any]] = []

        resolved_museum_id = (museum_id or "").strip()
        if resolved_museum_id:
            clauses.append({"term": {"museum_id": resolved_museum_id}})
        elif museum_slug.startswith("museum_"):
            clauses.append({"term": {"museum_id": museum_slug}})

        if not filters:
            return clauses

        keyword_fields = {
            "artifact_id",
            "museum_id",
            "category",
            "super_category",
            "creator",
            "date_or_period",
            "support_or_material",
            "technique",
            "origin_history",
            "incorporation",
            "production_center",
            "inventory_number",
            "detail_type",
            "source_system",
        }
        numeric_fields = {
            "image_count",
            "image_order",
            "width",
            "height",
        }
        legacy_keyword_aliases = {
            "inventory": "inventory_number",
            "support": "support_or_material",
            "historical_origin": "origin_history",
            "manufacturer_location": "production_center",
            "museum_name": "museum",
            "location": "production_center",
        }

        for key, value in filters.items():
            if value is None:
                continue
            resolved_key = legacy_keyword_aliases.get(key, key)

            if resolved_key in keyword_fields:
                if isinstance(value, list) and value:
                    clauses.append({"terms": {resolved_key: value}})
                elif isinstance(value, (str, int, float, bool)):
                    clauses.append({"term": {resolved_key: value}})
                continue

            if resolved_key in numeric_fields:
                if isinstance(value, dict):
                    range_payload = {
                        op: val
                        for op, val in value.items()
                        if op in {"gt", "gte", "lt", "lte"} and isinstance(val, (int, float))
                    }
                    if range_payload:
                        clauses.append({"range": {resolved_key: range_payload}})
                elif isinstance(value, (int, float)):
                    clauses.append({"term": {resolved_key: value}})
                continue

            if key.endswith("_gte") and isinstance(value, (int, float)):
                field = key[:-4]
                if field in numeric_fields:
                    clauses.append({"range": {field: {"gte": value}}})
                continue

            if key.endswith("_lte") and isinstance(value, (int, float)):
                field = key[:-4]
                if field in numeric_fields:
                    clauses.append({"range": {field: {"lte": value}}})
                continue

        return clauses

    def _build_sort(self, sort: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not sort:
            return []

        sort_payload: list[dict[str, Any]] = []
        for field, value in sort.items():
            if field not in self._ALLOWED_SORT_FIELDS:
                continue

            if isinstance(value, str):
                order = value.lower()
            elif isinstance(value, dict):
                order = str(value.get("order", "")).lower()
            else:
                order = ""

            if order not in {"asc", "desc"}:
                continue

            sort_payload.append({field: {"order": order}})

        return sort_payload

    def _build_image_filter_clauses(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
    ) -> list[dict[str, Any]]:
        resolved_museum_id = (museum_id or "").strip()
        if resolved_museum_id:
            return [{"term": {"museum_id": resolved_museum_id}}]
        if museum_slug.startswith("museum_"):
            return [{"term": {"museum_id": museum_slug}}]
        if museum_slug:
            return [{"term": {"museum_id": museum_slug}}]
        return []

    def _build_in_tour_boost_clause(self) -> dict[str, Any] | None:
        boost = float(getattr(self.settings, "CHAT_IN_TOUR_BOOST", 0) or 0)
        if boost <= 0:
            return None
        return {"term": {"in_tour": {"value": True, "boost": boost}}}

    def _is_in_tour(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes"}
        if isinstance(value, (int, float)):
            return bool(value)
        return False

    def _artifact_candidate_size(self, top_k: int) -> int:
        base = max(top_k, 1)
        boost = float(getattr(self.settings, "CHAT_IN_TOUR_BOOST", 0) or 0)
        if boost <= 0:
            return base
        return min(150, max(base * 4, base + 20))

    def _total_hits_value(self, response: dict[str, Any]) -> int:
        total_obj = response.get("hits", {}).get("total", 0)
        if isinstance(total_obj, dict):
            raw_value = total_obj.get("value", 0)
        else:
            raw_value = total_obj
        try:
            return max(int(raw_value), 0)
        except (TypeError, ValueError):
            return 0

    def _prioritize_in_tour_results(
        self,
        results: list[dict[str, Any]],
        *,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not results:
            return []
        preferred: list[dict[str, Any]] = []
        others: list[dict[str, Any]] = []
        for item in results:
            if self._is_in_tour(item.get("in_tour")):
                preferred.append(item)
            else:
                others.append(item)
        ranked = preferred + others
        return ranked[: max(top_k, 1)]

    def _resolve_inventory_number(self, source: dict[str, Any]) -> str:
        return str(source.get("inventory_number") or source.get("inventory") or "").strip()

    def _resolve_search_text(self, source: dict[str, Any]) -> str:
        return str(source.get("search_text") or source.get("full_text") or "").strip()

    def _resolve_museum_name(self, source: dict[str, Any]) -> str:
        return str(source.get("museum") or source.get("museum_name") or "").strip()

    def _resolve_support_or_material(self, source: dict[str, Any]) -> str:
        return str(source.get("support_or_material") or source.get("support") or "").strip()

    def _resolve_origin_history(self, source: dict[str, Any]) -> str:
        return str(source.get("origin_history") or source.get("historical_origin") or "").strip()

    def _resolve_production_center(self, source: dict[str, Any]) -> str:
        return str(source.get("production_center") or source.get("manufacturer_location") or "").strip()

    def _artifact_payload_from_source(
        self,
        *,
        source: dict[str, Any],
        score: Any,
        snippet: str,
    ) -> dict[str, Any]:
        inventory_number = self._resolve_inventory_number(source)
        search_text = self._resolve_search_text(source)
        museum_name = self._resolve_museum_name(source)
        support_or_material = self._resolve_support_or_material(source)
        origin_history = self._resolve_origin_history(source)
        production_center = self._resolve_production_center(source)

        payload: dict[str, Any] = {
            "score": score,
            "artifact_id": source.get("artifact_id"),
            "in_tour": self._is_in_tour(source.get("in_tour")),
            "inventory_number": inventory_number or None,
            "title": source.get("title"),
            "museum_id": source.get("museum_id"),
            "museum": museum_name or None,
            "category": source.get("category"),
            "super_category": source.get("super_category"),
            "creator": source.get("creator"),
            "date_or_period": source.get("date_or_period"),
            "support_or_material": support_or_material or None,
            "technique": source.get("technique"),
            "origin_history": origin_history or None,
            "incorporation": source.get("incorporation"),
            "production_center": production_center or None,
            "description": self._truncate(source.get("description"), max_chars=900),
            "search_text": search_text or None,
            "detail_type": source.get("detail_type"),
            "detail_url": source.get("detail_url"),
            "snippet": snippet,
            "image_ids": source.get("image_ids"),
            "image_file_ids": source.get("image_file_ids"),
            "image_paths": source.get("image_paths"),
            "image_urls": source.get("image_urls"),
            "image_count": source.get("image_count"),
            # Legacy compatibility aliases.
            "inventory": inventory_number or None,
            "museum_name": museum_name or None,
            "full_text": search_text or None,
            "support": support_or_material or None,
            "historical_origin": origin_history or None,
            "manufacturer_location": production_center or None,
            "location": production_center or None,
        }
        return payload

    def _resolve_original_image_name(self, source: dict[str, Any]) -> str:
        local_path = str(source.get("local_path") or "").strip()
        if local_path:
            sanitized = local_path.replace("\\", "/")
            if sanitized.strip("/"):
                return sanitized.strip("/")

        source_url = str(source.get("source_url") or "").strip()
        if source_url:
            parsed = urlparse(source_url)
            name = Path(parsed.path).name.strip()
            if name:
                return name

        image_id = str(source.get("image_id") or source.get("id") or "").strip()
        if image_id:
            return f"{image_id.replace(':', '_')}.jpg"
        return ""

    def _image_payload_from_source(self, *, score: Any, source: dict[str, Any]) -> dict[str, Any]:
        image_id = str(source.get("image_id") or source.get("id") or "").strip()
        local_path = str(source.get("local_path") or "").strip() or None
        source_url = str(source.get("source_url") or "").strip() or None
        original_image_name = self._resolve_original_image_name(source)
        inventory_number = self._resolve_inventory_number(source)
        image_key = image_id or local_path or source_url or None

        return {
            "score": score,
            "id": image_id or None,
            "image_id": image_id or None,
            "artifact_id": str(source.get("artifact_id") or "").strip() or None,
            "in_tour": self._is_in_tour(source.get("in_tour")),
            "museum_id": source.get("museum_id"),
            "local_path": local_path,
            "source_url": source_url,
            "artifact_title": source.get("artifact_title"),
            "inventory_number": inventory_number or None,
            "caption": source.get("caption"),
            "alt_text": source.get("alt_text"),
            # Legacy compatibility aliases.
            "image_key": image_key,
            "original_image_name": original_image_name or None,
            "inventory": inventory_number or None,
        }

    def _resolve_knn_k(
        self,
        *,
        from_offset: int,
        size: int,
        retrieval_window_size: int | None,
        minimum: int = 1,
    ) -> int:
        if retrieval_window_size is not None:
            return max(int(retrieval_window_size), int(size), int(minimum), 1)
        return max((int(from_offset) + int(size)) * 3, int(size), int(minimum), 1)

    def _build_query_body(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        query_text: str,
        lexical_query: str | None,
        query_embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None,
        sort: dict[str, Any] | None,
        from_offset: int = 0,
        size_override: int | None = None,
        pagination_depth: int | None = None,
        retrieval_window_size: int | None = None,
    ) -> dict[str, Any]:
        from_value = max(int(from_offset), 0)
        size = (
            max(int(size_override), 1)
            if size_override is not None
            else self._artifact_candidate_size(top_k)
        )
        retrieval_window = (
            max(int(retrieval_window_size), 1)
            if retrieval_window_size is not None
            else None
        )
        knn_k = self._resolve_knn_k(
            from_offset=from_value,
            size=size,
            retrieval_window_size=retrieval_window,
        )
        filter_clauses = self._build_filter_clauses(
            museum_slug=museum_slug,
            museum_id=museum_id,
            filters=filters,
        )
        in_tour_boost_clause = self._build_in_tour_boost_clause()
        hybrid_queries: list[dict[str, Any]] = []
        embedding_only = self.settings.CHAT_RETRIEVAL_EMBEDDING_ONLY

        if query_embedding:
            knn_query: dict[str, Any] = {
                "knn": {
                    "text_embedding": {
                        "vector": query_embedding,
                        "k": knn_k,
                    }
                }
            }
            if filter_clauses or in_tour_boost_clause:
                bool_query: dict[str, Any] = {"must": [knn_query]}
                if filter_clauses:
                    bool_query["filter"] = filter_clauses
                if in_tour_boost_clause:
                    bool_query["should"] = [in_tour_boost_clause]
                hybrid_queries.append({"bool": bool_query})
            else:
                hybrid_queries.append(knn_query)

        lexical_query_value = (lexical_query or query_text or "").strip()
        if lexical_query_value and not embedding_only:
            multi_match_query: dict[str, Any] = {
                "multi_match": {
                    "query": lexical_query_value,
                    "fields": [
                        "search_text^4",
                        "title^3",
                        "description^2",
                        "inventory_number^1.8",
                        "inventory_number.text^1.8",
                        "category.text^1.5",
                        "super_category.text^1.3",
                        "support_or_material.text^1.2",
                        "technique.text^1.2",
                        "origin_history^1.1",
                        "production_center.text^1.1",
                        "incorporation.text^1.1",
                        "museum.text",
                    ],
                    "type": "best_fields",
                    "operator": "or",
                }
            }
            if filter_clauses or in_tour_boost_clause:
                bool_query = {"must": [multi_match_query]}
                if filter_clauses:
                    bool_query["filter"] = filter_clauses
                if in_tour_boost_clause:
                    bool_query["should"] = [in_tour_boost_clause]
                hybrid_queries.append({"bool": bool_query})
            else:
                hybrid_queries.append(multi_match_query)

        if not hybrid_queries:
            fallback_query: dict[str, Any] = {"match_all": {}}
            if filter_clauses or in_tour_boost_clause:
                bool_query = {"must": [fallback_query]}
                if filter_clauses:
                    bool_query["filter"] = filter_clauses
                if in_tour_boost_clause:
                    bool_query["should"] = [in_tour_boost_clause]
                hybrid_queries = [{"bool": bool_query}]
            else:
                hybrid_queries = [fallback_query]

        body: dict[str, Any] = {
            "size": size,
            "track_total_hits": True,
            "_source": [
                "artifact_id",
                "inventory_number",
                "title",
                "museum_id",
                "museum",
                "category",
                "super_category",
                "creator",
                "date_or_period",
                "support_or_material",
                "technique",
                "origin_history",
                "incorporation",
                "production_center",
                "description",
                "search_text",
                "detail_type",
                "detail_url",
                "in_tour",
                "image_ids",
                "image_file_ids",
                "image_paths",
                "image_urls",
                "image_count",
            ],
            "query": {"hybrid": {"queries": hybrid_queries}},
            "highlight": {
                "pre_tags": ["<em>"],
                "post_tags": ["</em>"],
                "fields": {"search_text": {"number_of_fragments": 1, "fragment_size": 220}},
            },
        }
        if from_value:
            body["from"] = from_value
        resolved_pagination_depth = pagination_depth
        if resolved_pagination_depth is None and retrieval_window is not None:
            resolved_pagination_depth = retrieval_window
        if resolved_pagination_depth is not None and not embedding_only:
            hybrid_query = body.get("query", {}).get("hybrid")
            if isinstance(hybrid_query, dict):
                hybrid_query["pagination_depth"] = max(int(resolved_pagination_depth), size, 1)

        if embedding_only:
            # Embeddings-only mode: avoid hybrid/BM25 entirely and run strict KNN + filters.
            knn_query = {
                "knn": {
                    "text_embedding": {
                        "vector": query_embedding,
                        "k": knn_k,
                    }
                }
            }
            if filter_clauses or in_tour_boost_clause:
                bool_query = {"must": [knn_query]}
                if filter_clauses:
                    bool_query["filter"] = filter_clauses
                if in_tour_boost_clause:
                    bool_query["should"] = [in_tour_boost_clause]
                body["query"] = {"bool": bool_query}
            else:
                body["query"] = knn_query

        sort_payload = self._build_sort(sort)
        if sort_payload:
            body["sort"] = sort_payload

        return body

    def _truncate(self, value: Any, *, max_chars: int) -> str:
        text = str(value or "").strip()
        if len(text) <= max_chars:
            return text
        return f"{text[:max_chars].rstrip()}..."

    def _artifact_results_from_hits(self, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for hit in hits:
            source = hit.get("_source", {}) or {}
            highlight = hit.get("highlight", {}) or {}
            highlight_search_text = highlight.get("search_text", [])
            snippet = ""
            if isinstance(highlight_search_text, list) and highlight_search_text:
                snippet = self._truncate(highlight_search_text[0], max_chars=260)
            elif source.get("description"):
                snippet = self._truncate(source.get("description"), max_chars=260)
            else:
                snippet = self._truncate(source.get("search_text"), max_chars=260)

            results.append(
                self._artifact_payload_from_source(
                    source=source,
                    score=hit.get("_score"),
                    snippet=snippet,
                )
            )
        return results

    def _search_relevant_context_sync(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        query_text: str,
        lexical_query: str | None,
        query_embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None,
        sort: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        client = self._ensure_client()
        body = self._build_query_body(
            museum_slug=museum_slug,
            museum_id=museum_id,
            query_text=query_text,
            lexical_query=lexical_query,
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
            sort=sort,
        )
        self._log(
            logging.INFO,
            "opensearch.retrieve.body",
            body=body,
        )

        search_kwargs: dict[str, Any] = {
            "index": self.settings.OPENSEARCH_INDEX_ARTIFACT,
            "body": body,
        }
        if not self.settings.CHAT_RETRIEVAL_EMBEDDING_ONLY:
            search_kwargs["search_pipeline"] = "nlp-search-pipeline"

        response = client.search(**search_kwargs)
        hits = response.get("hits", {}).get("hits", [])
        total_obj = response.get("hits", {}).get("total", {})
        total_value = total_obj.get("value") if isinstance(total_obj, dict) else total_obj
        top_1_hit: dict[str, Any] | None = None
        if hits:
            top_source = hits[0].get("_source", {}) or {}
            top_1_hit = {
                "score": hits[0].get("_score"),
                "artifact_id": top_source.get("artifact_id"),
                "inventory_number": self._resolve_inventory_number(top_source),
                "title": top_source.get("title"),
                "museum_id": top_source.get("museum_id"),
                "category": top_source.get("category"),
                "production_center": self._resolve_production_center(top_source),
            }
        self._log(
            logging.INFO,
            "opensearch.retrieve.response",
            index=self.settings.OPENSEARCH_INDEX_ARTIFACT,
            museum_slug=museum_slug,
            museum_id=museum_id,
            took_ms=response.get("took"),
            timed_out=response.get("timed_out"),
            hits_returned=len(hits),
            hits_total=total_value,
            top_1_hit=top_1_hit,
        )
        results: list[dict[str, Any]] = []
        results = self._artifact_results_from_hits(hits)

        return self._prioritize_in_tour_results(results, top_k=top_k)

    def _search_relevant_context_page_sync(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        query_text: str,
        lexical_query: str | None,
        query_embedding: list[float],
        from_offset: int,
        page_size: int,
        filters: dict[str, Any] | None,
        sort: dict[str, Any] | None,
        retrieval_window_size: int | None = None,
    ) -> OpenSearchRetrievalPage:
        client = self._ensure_client()
        body = self._build_query_body(
            museum_slug=museum_slug,
            museum_id=museum_id,
            query_text=query_text,
            lexical_query=lexical_query,
            query_embedding=query_embedding,
            top_k=page_size,
            filters=filters,
            sort=sort,
            from_offset=from_offset,
            size_override=page_size,
            pagination_depth=(
                max(int(retrieval_window_size), 1)
                if retrieval_window_size is not None
                else max(int(from_offset), 0) + max(int(page_size), 1)
            ),
            retrieval_window_size=retrieval_window_size,
        )
        self._log(logging.INFO, "opensearch.retrieve.body", body=body)

        search_kwargs: dict[str, Any] = {
            "index": self.settings.OPENSEARCH_INDEX_ARTIFACT,
            "body": body,
        }
        if not self.settings.CHAT_RETRIEVAL_EMBEDDING_ONLY:
            search_kwargs["search_pipeline"] = "nlp-search-pipeline"

        response = client.search(**search_kwargs)
        hits = response.get("hits", {}).get("hits", [])
        total_value = self._total_hits_value(response)
        self._log(
            logging.INFO,
            "opensearch.retrieve.response",
            index=self.settings.OPENSEARCH_INDEX_ARTIFACT,
            museum_slug=museum_slug,
            museum_id=museum_id,
            took_ms=response.get("took"),
            timed_out=response.get("timed_out"),
            hits_returned=len(hits),
            hits_total=total_value,
            from_offset=from_offset,
            page_size=page_size,
        )
        results = self._artifact_results_from_hits(hits)
        return OpenSearchRetrievalPage(results=results, total=total_value)

    async def search_relevant_context(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        query_text: str,
        lexical_query: str | None = None,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        sort: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self._search_relevant_context_sync,
            museum_slug=museum_slug,
            museum_id=museum_id,
            query_text=query_text,
            lexical_query=lexical_query,
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
            sort=sort,
        )

    async def search_relevant_context_page(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        query_text: str,
        lexical_query: str | None = None,
        query_embedding: list[float],
        from_offset: int,
        page_size: int,
        filters: dict[str, Any] | None = None,
        sort: dict[str, Any] | None = None,
        retrieval_window_size: int | None = None,
    ) -> OpenSearchRetrievalPage:
        return await asyncio.to_thread(
            self._search_relevant_context_page_sync,
            museum_slug=museum_slug,
            museum_id=museum_id,
            query_text=query_text,
            lexical_query=lexical_query,
            query_embedding=query_embedding,
            from_offset=from_offset,
            page_size=page_size,
            filters=filters,
            sort=sort,
            retrieval_window_size=retrieval_window_size,
        )

    def _image_results_from_hits(self, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for hit in hits:
            source = hit.get("_source", {}) or {}
            artifact_id = str(source.get("artifact_id") or "").strip()
            if not artifact_id:
                continue
            payload = self._image_payload_from_source(score=hit.get("_score"), source=source)
            if payload.get("artifact_id"):
                results.append(payload)
        return results

    def _search_similar_images_sync(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        image_embedding: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        client = self._ensure_client()
        requested_top_k = max(top_k, 1)
        size = self._artifact_candidate_size(requested_top_k)
        in_tour_boost_clause = self._build_in_tour_boost_clause()
        bool_query: dict[str, Any] = {
            "filter": self._build_image_filter_clauses(
                museum_slug=museum_slug,
                museum_id=museum_id,
            ),
            "must": [
                {
                    "knn": {
                        "visual_embedding": {
                            "vector": image_embedding,
                            "k": max(size * 3, size),
                        }
                    }
                }
            ],
        }
        if in_tour_boost_clause:
            bool_query["should"] = [in_tour_boost_clause]

        body: dict[str, Any] = {
            "size": size,
            "_source": [
                "image_id",
                "artifact_id",
                "museum_id",
                "local_path",
                "source_url",
                "artifact_title",
                "inventory_number",
                "caption",
                "alt_text",
                "in_tour",
            ],
            "query": {"bool": bool_query},
        }
        self._log(
            logging.INFO,
            "opensearch.image_retrieve.body",
            body=body,
        )
        response = client.search(
            index=self.settings.OPENSEARCH_INDEX_IMAGE,
            body=body,
        )
        hits = response.get("hits", {}).get("hits", [])
        total_obj = response.get("hits", {}).get("total", {})
        total_value = total_obj.get("value") if isinstance(total_obj, dict) else total_obj
        top_1_hit: dict[str, Any] | None = None
        if hits:
            top_source = hits[0].get("_source", {}) or {}
            top_1_hit = {
                "score": hits[0].get("_score"),
                "artifact_id": top_source.get("artifact_id"),
                "image_id": top_source.get("image_id"),
                "museum_id": top_source.get("museum_id"),
            }
        self._log(
            logging.INFO,
            "opensearch.image_retrieve.response",
            index=self.settings.OPENSEARCH_INDEX_IMAGE,
            museum_slug=museum_slug,
            museum_id=museum_id,
            took_ms=response.get("took"),
            timed_out=response.get("timed_out"),
            hits_returned=len(hits),
            hits_total=total_value,
            top_1_hit=top_1_hit,
        )

        results = self._image_results_from_hits(hits)
        return self._prioritize_in_tour_results(results, top_k=requested_top_k)

    def _search_similar_images_page_sync(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        image_embedding: list[float],
        from_offset: int,
        page_size: int,
        retrieval_window_size: int | None = None,
    ) -> OpenSearchRetrievalPage:
        client = self._ensure_client()
        from_value = max(int(from_offset), 0)
        size = max(int(page_size), 1)
        retrieval_window = (
            max(int(retrieval_window_size), 1)
            if retrieval_window_size is not None
            else None
        )
        knn_k = self._resolve_knn_k(
            from_offset=from_value,
            size=size,
            retrieval_window_size=retrieval_window,
        )
        in_tour_boost_clause = self._build_in_tour_boost_clause()
        bool_query: dict[str, Any] = {
            "filter": self._build_image_filter_clauses(
                museum_slug=museum_slug,
                museum_id=museum_id,
            ),
            "must": [
                {
                    "knn": {
                        "visual_embedding": {
                            "vector": image_embedding,
                            "k": knn_k,
                        }
                    }
                }
            ],
        }
        if in_tour_boost_clause:
            bool_query["should"] = [in_tour_boost_clause]

        body: dict[str, Any] = {
            "from": from_value,
            "size": size,
            "track_total_hits": True,
            "_source": [
                "image_id",
                "artifact_id",
                "museum_id",
                "local_path",
                "source_url",
                "artifact_title",
                "inventory_number",
                "caption",
                "alt_text",
                "in_tour",
            ],
            "query": {"bool": bool_query},
        }
        self._log(logging.INFO, "opensearch.image_retrieve.body", body=body)
        response = client.search(
            index=self.settings.OPENSEARCH_INDEX_IMAGE,
            body=body,
        )
        hits = response.get("hits", {}).get("hits", [])
        total_value = self._total_hits_value(response)
        self._log(
            logging.INFO,
            "opensearch.image_retrieve.response",
            index=self.settings.OPENSEARCH_INDEX_IMAGE,
            museum_slug=museum_slug,
            museum_id=museum_id,
            took_ms=response.get("took"),
            timed_out=response.get("timed_out"),
            hits_returned=len(hits),
            hits_total=total_value,
            from_offset=from_value,
            page_size=size,
        )
        return OpenSearchRetrievalPage(
            results=self._image_results_from_hits(hits),
            total=total_value,
        )

    def _search_similar_images_multi_sync(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        image_embeddings: list[list[float]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        embeddings = [embedding for embedding in image_embeddings if embedding]
        if not embeddings:
            return []

        client = self._ensure_client()
        requested_top_k = max(top_k, 1)
        size = self._artifact_candidate_size(requested_top_k)
        filter_clauses = self._build_image_filter_clauses(
            museum_slug=museum_slug,
            museum_id=museum_id,
        )
        in_tour_boost_clause = self._build_in_tour_boost_clause()

        should_clauses: list[dict[str, Any]] = []
        for embedding in embeddings:
            should_clauses.append(
                {
                    "knn": {
                        "visual_embedding": {
                            "vector": embedding,
                            "k": max(30, size * 3),
                            "boost": 3.0,
                        }
                    }
                }
            )

        bool_query: dict[str, Any] = {
            "filter": filter_clauses,
            "must": [{"bool": {"should": should_clauses, "minimum_should_match": 1}}],
        }
        if in_tour_boost_clause:
            bool_query["should"] = [in_tour_boost_clause]

        body: dict[str, Any] = {
            "size": size,
            "_source": [
                "image_id",
                "artifact_id",
                "museum_id",
                "local_path",
                "source_url",
                "artifact_title",
                "inventory_number",
                "caption",
                "alt_text",
                "in_tour",
            ],
            "query": {"bool": bool_query},
        }
        self._log(
            logging.INFO,
            "opensearch.image_retrieve.body",
            body=body,
        )
        response = client.search(
            index=self.settings.OPENSEARCH_INDEX_IMAGE,
            body=body,
        )
        hits = response.get("hits", {}).get("hits", [])
        total_obj = response.get("hits", {}).get("total", {})
        total_value = total_obj.get("value") if isinstance(total_obj, dict) else total_obj
        top_1_hit: dict[str, Any] | None = None
        if hits:
            top_source = hits[0].get("_source", {}) or {}
            top_1_hit = {
                "score": hits[0].get("_score"),
                "artifact_id": top_source.get("artifact_id"),
                "image_id": top_source.get("image_id"),
                "museum_id": top_source.get("museum_id"),
            }
        self._log(
            logging.INFO,
            "opensearch.image_retrieve.response",
            index=self.settings.OPENSEARCH_INDEX_IMAGE,
            museum_slug=museum_slug,
            museum_id=museum_id,
            took_ms=response.get("took"),
            timed_out=response.get("timed_out"),
            hits_returned=len(hits),
            hits_total=total_value,
            top_1_hit=top_1_hit,
        )

        results = self._image_results_from_hits(hits)
        return self._prioritize_in_tour_results(results, top_k=requested_top_k)

    def _search_similar_images_multi_page_sync(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        image_embeddings: list[list[float]],
        from_offset: int,
        page_size: int,
        retrieval_window_size: int | None = None,
    ) -> OpenSearchRetrievalPage:
        embeddings = [embedding for embedding in image_embeddings if embedding]
        if not embeddings:
            return OpenSearchRetrievalPage(results=[], total=0)

        client = self._ensure_client()
        from_value = max(int(from_offset), 0)
        size = max(int(page_size), 1)
        retrieval_window = (
            max(int(retrieval_window_size), 1)
            if retrieval_window_size is not None
            else None
        )
        knn_k = self._resolve_knn_k(
            from_offset=from_value,
            size=size,
            retrieval_window_size=retrieval_window,
            minimum=30,
        )
        filter_clauses = self._build_image_filter_clauses(
            museum_slug=museum_slug,
            museum_id=museum_id,
        )
        in_tour_boost_clause = self._build_in_tour_boost_clause()

        should_clauses: list[dict[str, Any]] = []
        for embedding in embeddings:
            should_clauses.append(
                {
                    "knn": {
                        "visual_embedding": {
                            "vector": embedding,
                            "k": knn_k,
                            "boost": 3.0,
                        }
                    }
                }
            )

        bool_query: dict[str, Any] = {
            "filter": filter_clauses,
            "must": [{"bool": {"should": should_clauses, "minimum_should_match": 1}}],
        }
        if in_tour_boost_clause:
            bool_query["should"] = [in_tour_boost_clause]

        body: dict[str, Any] = {
            "from": from_value,
            "size": size,
            "track_total_hits": True,
            "_source": [
                "image_id",
                "artifact_id",
                "museum_id",
                "local_path",
                "source_url",
                "artifact_title",
                "inventory_number",
                "caption",
                "alt_text",
                "in_tour",
            ],
            "query": {"bool": bool_query},
        }
        self._log(logging.INFO, "opensearch.image_retrieve.body", body=body)
        response = client.search(
            index=self.settings.OPENSEARCH_INDEX_IMAGE,
            body=body,
        )
        hits = response.get("hits", {}).get("hits", [])
        total_value = self._total_hits_value(response)
        self._log(
            logging.INFO,
            "opensearch.image_retrieve.response",
            index=self.settings.OPENSEARCH_INDEX_IMAGE,
            museum_slug=museum_slug,
            museum_id=museum_id,
            took_ms=response.get("took"),
            timed_out=response.get("timed_out"),
            hits_returned=len(hits),
            hits_total=total_value,
            from_offset=from_value,
            page_size=size,
        )
        return OpenSearchRetrievalPage(
            results=self._image_results_from_hits(hits),
            total=total_value,
        )

    async def search_similar_images(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        image_embedding: list[float],
        top_k: int = 6,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self._search_similar_images_sync,
            museum_slug=museum_slug,
            museum_id=museum_id,
            image_embedding=image_embedding,
            top_k=top_k,
        )

    async def search_similar_images_page(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        image_embedding: list[float],
        from_offset: int,
        page_size: int,
        retrieval_window_size: int | None = None,
    ) -> OpenSearchRetrievalPage:
        return await asyncio.to_thread(
            self._search_similar_images_page_sync,
            museum_slug=museum_slug,
            museum_id=museum_id,
            image_embedding=image_embedding,
            from_offset=from_offset,
            page_size=page_size,
            retrieval_window_size=retrieval_window_size,
        )

    async def search_similar_images_multi(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        image_embeddings: list[list[float]],
        top_k: int = 6,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self._search_similar_images_multi_sync,
            museum_slug=museum_slug,
            museum_id=museum_id,
            image_embeddings=image_embeddings,
            top_k=top_k,
        )

    async def search_similar_images_multi_page(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        image_embeddings: list[list[float]],
        from_offset: int,
        page_size: int,
        retrieval_window_size: int | None = None,
    ) -> OpenSearchRetrievalPage:
        return await asyncio.to_thread(
            self._search_similar_images_multi_page_sync,
            museum_slug=museum_slug,
            museum_id=museum_id,
            image_embeddings=image_embeddings,
            from_offset=from_offset,
            page_size=page_size,
            retrieval_window_size=retrieval_window_size,
        )

    def _fetch_images_by_artifact_ids_sync(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        artifact_ids: list[str],
        per_artifact: int = 1,
        max_total: int = 24,
    ) -> list[dict[str, Any]]:
        unique_ids: list[str] = []
        seen: set[str] = set()
        for artifact_id in artifact_ids:
            cleaned = (artifact_id or "").strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            unique_ids.append(cleaned)

        if not unique_ids:
            return []

        per_artifact_value = max(per_artifact, 1)
        max_total_value = max(max_total, per_artifact_value)
        if per_artifact_value == 1:
            candidate_size = min(max_total_value, len(unique_ids))
        else:
            candidate_size = min(
                500,
                max(
                    max_total_value,
                    len(unique_ids) * per_artifact_value * 4,
                    len(unique_ids),
                ),
            )

        client = self._ensure_client()
        filter_clauses: list[dict[str, Any]] = [{"terms": {"artifact_id": unique_ids}}]
        filter_clauses.extend(
            self._build_image_filter_clauses(
                museum_slug=museum_slug,
                museum_id=museum_id,
            )
        )
        body: dict[str, Any] = {
            "size": max(candidate_size, 1),
            "_source": [
                "image_id",
                "artifact_id",
                "museum_id",
                "local_path",
                "source_url",
                "artifact_title",
                "inventory_number",
                "caption",
                "alt_text",
                "in_tour",
            ],
            "query": {
                "bool": {
                    "filter": filter_clauses,
                }
            },
        }
        if per_artifact_value == 1:
            body["collapse"] = {"field": "artifact_id"}
        self._log(
            logging.INFO,
            "opensearch.image_fetch_by_artifact.body",
            body=body,
        )
        response = client.search(
            index=self.settings.OPENSEARCH_INDEX_IMAGE,
            body=body,
        )
        hits = response.get("hits", {}).get("hits", [])
        total_obj = response.get("hits", {}).get("total", {})
        total_value = total_obj.get("value") if isinstance(total_obj, dict) else total_obj
        self._log(
            logging.INFO,
            "opensearch.image_fetch_by_artifact.response",
            index=self.settings.OPENSEARCH_INDEX_IMAGE,
            museum_slug=museum_slug,
            museum_id=museum_id,
            took_ms=response.get("took"),
            timed_out=response.get("timed_out"),
            hits_returned=len(hits),
            hits_total=total_value,
        )

        by_artifact: dict[str, list[dict[str, Any]]] = {artifact_id: [] for artifact_id in unique_ids}
        seen_keys_by_artifact: dict[str, set[str]] = {
            artifact_id: set() for artifact_id in unique_ids
        }
        for hit in hits:
            source = hit.get("_source", {}) or {}
            payload = self._image_payload_from_source(score=hit.get("_score"), source=source)
            artifact_id = str(payload.get("artifact_id") or "").strip()
            if not artifact_id:
                continue
            if artifact_id not in by_artifact:
                continue
            if len(by_artifact[artifact_id]) >= per_artifact_value:
                continue
            dedupe_key = str(
                payload.get("image_id")
                or payload.get("local_path")
                or payload.get("source_url")
                or payload.get("original_image_name")
                or ""
            ).strip()
            if not dedupe_key:
                continue
            if dedupe_key in seen_keys_by_artifact[artifact_id]:
                continue
            seen_keys_by_artifact[artifact_id].add(dedupe_key)
            by_artifact[artifact_id].append(payload)

        ordered: list[dict[str, Any]] = []
        for artifact_id in unique_ids:
            for image_payload in by_artifact.get(artifact_id, []):
                ordered.append(image_payload)
                if len(ordered) >= max_total_value:
                    return ordered
        return ordered

    async def fetch_images_by_artifact_ids(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        artifact_ids: list[str],
        per_artifact: int = 1,
        max_total: int = 24,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self._fetch_images_by_artifact_ids_sync,
            museum_slug=museum_slug,
            museum_id=museum_id,
            artifact_ids=artifact_ids,
            per_artifact=per_artifact,
            max_total=max_total,
        )

    def _fetch_artifacts_by_ids_sync(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        artifact_ids: list[str],
        top_k: int,
    ) -> list[dict[str, Any]]:
        unique_ids: list[str] = []
        seen: set[str] = set()
        for artifact_id in artifact_ids:
            cleaned = (artifact_id or "").strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            unique_ids.append(cleaned)

        if not unique_ids:
            return []

        client = self._ensure_client()
        size = min(max(top_k, 1), len(unique_ids))
        filter_clauses: list[dict[str, Any]] = [{"terms": {"artifact_id": unique_ids}}]
        filter_clauses.extend(
            self._build_image_filter_clauses(
                museum_slug=museum_slug,
                museum_id=museum_id,
            )
        )

        body: dict[str, Any] = {
            "size": size,
            "_source": [
                "artifact_id",
                "inventory_number",
                "title",
                "museum_id",
                "museum",
                "category",
                "super_category",
                "creator",
                "date_or_period",
                "support_or_material",
                "technique",
                "origin_history",
                "incorporation",
                "production_center",
                "description",
                "search_text",
                "detail_type",
                "detail_url",
                "image_ids",
                "image_file_ids",
                "image_paths",
                "image_urls",
                "image_count",
            ],
            "query": {
                "bool": {
                    "filter": filter_clauses,
                }
            },
        }
        self._log(
            logging.INFO,
            "opensearch.artifact_fetch.body",
            body=body,
        )
        response = client.search(
            index=self.settings.OPENSEARCH_INDEX_ARTIFACT,
            body=body,
        )
        hits = response.get("hits", {}).get("hits", [])
        total_obj = response.get("hits", {}).get("total", {})
        total_value = total_obj.get("value") if isinstance(total_obj, dict) else total_obj
        self._log(
            logging.INFO,
            "opensearch.artifact_fetch.response",
            index=self.settings.OPENSEARCH_INDEX_ARTIFACT,
            museum_slug=museum_slug,
            museum_id=museum_id,
            took_ms=response.get("took"),
            timed_out=response.get("timed_out"),
            hits_returned=len(hits),
            hits_total=total_value,
        )

        by_id: dict[str, dict[str, Any]] = {}
        for hit in hits:
            source = hit.get("_source", {}) or {}
            artifact_id = str(source.get("artifact_id") or "").strip()
            if not artifact_id:
                continue
            snippet = ""
            if source.get("description"):
                snippet = self._truncate(source.get("description"), max_chars=260)
            else:
                snippet = self._truncate(source.get("search_text"), max_chars=260)

            by_id[artifact_id] = self._artifact_payload_from_source(
                source=source,
                score=hit.get("_score"),
                snippet=snippet,
            )

        ordered: list[dict[str, Any]] = []
        for artifact_id in unique_ids:
            payload = by_id.get(artifact_id)
            if payload is not None:
                ordered.append(payload)
            if len(ordered) >= size:
                break
        return ordered

    def _fetch_artifacts_by_inventory_numbers_sync(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        inventory_numbers: list[str],
        top_k: int,
    ) -> list[dict[str, Any]]:
        unique_inventories: list[str] = []
        seen: set[str] = set()
        for inventory_number in inventory_numbers:
            cleaned = str(inventory_number or "").strip()
            if not cleaned:
                continue
            seen_key = cleaned.casefold()
            if seen_key in seen:
                continue
            seen.add(seen_key)
            unique_inventories.append(cleaned)

        if not unique_inventories:
            return []

        resolved_museum_id = str(museum_id or "").strip()
        if not resolved_museum_id:
            self._log(
                logging.WARNING,
                "opensearch.artifact_fetch_by_inventory.skipped",
                museum_slug=museum_slug,
                reason="missing_museum_id",
                inventory_count=len(unique_inventories),
            )
            return []

        client = self._ensure_client()
        size = min(max(top_k, 1), len(unique_inventories))
        inventory_terms: list[str] = []
        seen_terms: set[str] = set()
        for value in unique_inventories:
            for candidate in (value, value.casefold()):
                if candidate and candidate not in seen_terms:
                    seen_terms.add(candidate)
                    inventory_terms.append(candidate)

        filter_clauses: list[dict[str, Any]] = [{"term": {"museum_id": resolved_museum_id}}]

        body: dict[str, Any] = {
            "size": size,
            "_source": [
                "artifact_id",
                "inventory_number",
                "title",
                "museum_id",
                "museum",
                "category",
                "super_category",
                "creator",
                "date_or_period",
                "support_or_material",
                "technique",
                "origin_history",
                "incorporation",
                "production_center",
                "description",
                "search_text",
                "detail_type",
                "detail_url",
                "image_ids",
                "image_file_ids",
                "image_paths",
                "image_urls",
                "image_count",
            ],
            "query": {
                "bool": {
                    "filter": filter_clauses,
                    "should": [
                        {"terms": {"inventory_number": inventory_terms}},
                        {
                            "multi_match": {
                                "query": " ".join(unique_inventories),
                                "fields": ["inventory_number.text^2"],
                                "operator": "or",
                            }
                        },
                    ],
                    "minimum_should_match": 1,
                }
            },
        }
        self._log(
            logging.INFO,
            "opensearch.artifact_fetch_by_inventory.body",
            body=body,
        )
        response = client.search(
            index=self.settings.OPENSEARCH_INDEX_ARTIFACT,
            body=body,
        )
        hits = response.get("hits", {}).get("hits", [])
        total_obj = response.get("hits", {}).get("total", {})
        total_value = total_obj.get("value") if isinstance(total_obj, dict) else total_obj
        self._log(
            logging.INFO,
            "opensearch.artifact_fetch_by_inventory.response",
            index=self.settings.OPENSEARCH_INDEX_ARTIFACT,
            museum_slug=museum_slug,
            museum_id=museum_id,
            took_ms=response.get("took"),
            timed_out=response.get("timed_out"),
            hits_returned=len(hits),
            hits_total=total_value,
        )

        by_inventory: dict[str, dict[str, Any]] = {}
        for hit in hits:
            source = hit.get("_source", {}) or {}
            inventory_number = self._resolve_inventory_number(source)
            if not inventory_number:
                continue
            snippet = ""
            if source.get("description"):
                snippet = self._truncate(source.get("description"), max_chars=260)
            else:
                snippet = self._truncate(source.get("search_text"), max_chars=260)

            by_inventory[inventory_number.casefold()] = self._artifact_payload_from_source(
                source=source,
                score=hit.get("_score"),
                snippet=snippet,
            )

        ordered: list[dict[str, Any]] = []
        for inventory_number in unique_inventories:
            payload = by_inventory.get(inventory_number.casefold())
            if payload is not None:
                ordered.append(payload)
            if len(ordered) >= size:
                break
        return ordered

    async def fetch_artifacts_by_ids(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        artifact_ids: list[str],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self._fetch_artifacts_by_ids_sync,
            museum_slug=museum_slug,
            museum_id=museum_id,
            artifact_ids=artifact_ids,
            top_k=top_k,
        )

    async def fetch_artifacts_by_inventory_numbers(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        inventory_numbers: list[str],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self._fetch_artifacts_by_inventory_numbers_sync,
            museum_slug=museum_slug,
            museum_id=museum_id,
            inventory_numbers=inventory_numbers,
            top_k=top_k,
        )

    def _execute_structured_query_sync(
        self,
        *,
        plan: QueryPlan,
        dsl: CompiledOpenSearchDSL,
    ) -> QueryExecutionResult:
        client = self._ensure_client()
        self._log(
            logging.INFO,
            "opensearch.structured.body",
            endpoint=dsl.endpoint,
            index=dsl.index,
            operation=plan.operation,
            body=dsl.body,
        )
        result = execute_query(plan=plan, dsl=dsl, client=client)
        self._log(
            logging.INFO,
            "opensearch.structured.response",
            endpoint=dsl.endpoint,
            index=dsl.index,
            operation=plan.operation,
            total=result.total,
            count=result.count,
            exists=result.exists,
            groups=len(result.groups),
            items=len(result.items),
        )
        return result

    async def execute_structured_query(
        self,
        *,
        plan: QueryPlan,
        dsl: CompiledOpenSearchDSL,
    ) -> QueryExecutionResult:
        return await asyncio.to_thread(
            self._execute_structured_query_sync,
            plan=plan,
            dsl=dsl,
        )


@lru_cache(maxsize=1)
def get_opensearch_gateway() -> OpenSearchGateway:
    return OpenSearchGateway(get_settings())
