from __future__ import annotations

import asyncio
from functools import lru_cache
import json
import logging
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


class OpenSearchGateway:
    """OpenSearch adapter used by chat retrieval."""

    _ALLOWED_SORT_FIELDS = {
        "initial_year",
        "final_year",
        "initial_century",
        "final_century",
        "title.keyword",
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

    def _resolve_host_and_scheme(self) -> tuple[str, str]:
        host_value = (self.settings.OPENSEARCH_HOST or "").strip()
        if not host_value:
            raise RuntimeError("OPENSEARCH_HOST is required for retrieval.")

        if "://" in host_value:
            parsed = urlparse(host_value)
            if not parsed.hostname:
                raise RuntimeError(f"Invalid OPENSEARCH_HOST: {host_value}")
            scheme = parsed.scheme or ("https" if self.settings.opensearch_use_ssl_resolved else "http")
            return parsed.hostname, scheme

        scheme = self.settings.OPENSEARCH_SCHEME or (
            "https" if self.settings.opensearch_use_ssl_resolved else "http"
        )
        return host_value, scheme

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client

        OpenSearch = _import_opensearch()
        host, scheme = self._resolve_host_and_scheme()
        client_kwargs: dict[str, Any] = {
            "hosts": [
                {
                    "host": host,
                    "port": self.settings.OPENSEARCH_PORT,
                    "scheme": scheme,
                }
            ],
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

    def ensure_ready_sync(self) -> bool:
        client = self._ensure_client()
        ping = getattr(client, "ping", None)
        if callable(ping):
            return bool(ping())
        return True

    async def ensure_ready(self) -> bool:
        return await asyncio.to_thread(self.ensure_ready_sync)

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
            "support",
            "location",
            "historical_origin",
            "incorporation",
            "manufacturer_location",
        }
        numeric_fields = {
            "initial_year",
            "final_year",
            "initial_century",
            "final_century",
            "height",
            "width",
            "depth",
            "length",
            "diameter",
            "thickness",
        }

        for key, value in filters.items():
            if value is None:
                continue

            if key in keyword_fields:
                if isinstance(value, list) and value:
                    clauses.append({"terms": {key: value}})
                elif isinstance(value, (str, int, float, bool)):
                    clauses.append({"term": {key: value}})
                continue

            if key in numeric_fields:
                if isinstance(value, dict):
                    range_payload = {
                        op: val
                        for op, val in value.items()
                        if op in {"gt", "gte", "lt", "lte"} and isinstance(val, (int, float))
                    }
                    if range_payload:
                        clauses.append({"range": {key: range_payload}})
                elif isinstance(value, (int, float)):
                    clauses.append({"term": {key: value}})
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
        boost = float(self.settings.CHAT_IN_TOUR_BOOST)
        if boost <= 0:
            return None
        return {
            "constant_score": {
                "filter": {"term": {"in_tour": True}},
                "boost": boost,
            }
        }

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
    ) -> dict[str, Any]:
        size = max(top_k, 1)
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
                        #"k": max(size * 3, size),
                        "k": 100,
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
                        "full_text^4",
                        "title^3",
                        "description^2",
                        "inventory^1.8",
                        "category^1.5",
                        "support^1.2",
                        "location^1.2",
                        "historical_origin^1.1",
                        "manufacturer_location^1.1",
                        "museum_name",
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
            "_source": [
                "artifact_id",
                "inventory",
                "title",
                "museum_id",
                "museum_name",
                "location",
                "category",
                "initial_year",
                "final_year",
                "description",
                "full_text",
            ],
            "query": {"hybrid": {"queries": hybrid_queries}},
            "highlight": {
                "pre_tags": ["<em>"],
                "post_tags": ["</em>"],
                "fields": {"full_text": {"number_of_fragments": 1, "fragment_size": 220}},
            },
        }

        if embedding_only:
            # Embeddings-only mode: avoid hybrid/BM25 entirely and run strict KNN + filters.
            knn_query = {
                "knn": {
                    "text_embedding": {
                        "vector": query_embedding,
                        "k": max(size * 3, size),
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
                "inventory": top_source.get("inventory"),
                "title": top_source.get("title"),
                "museum_id": top_source.get("museum_id"),
                "category": top_source.get("category"),
                "location": top_source.get("location"),
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

        for hit in hits:
            source = hit.get("_source", {}) or {}
            highlight = hit.get("highlight", {}) or {}
            highlight_full_text = highlight.get("full_text", [])
            snippet = ""
            if isinstance(highlight_full_text, list) and highlight_full_text:
                snippet = self._truncate(highlight_full_text[0], max_chars=260)
            elif source.get("description"):
                snippet = self._truncate(source.get("description"), max_chars=260)
            else:
                snippet = self._truncate(source.get("full_text"), max_chars=260)

            results.append(
                {
                    "score": hit.get("_score"),
                    "artifact_id": source.get("artifact_id"),
                    "inventory": source.get("inventory"),
                    "title": source.get("title"),
                    "museum_id": source.get("museum_id"),
                    "museum_name": source.get("museum_name"),
                    "location": source.get("location"),
                    "category": source.get("category"),
                    "initial_year": source.get("initial_year"),
                    "final_year": source.get("final_year"),
                    "description": self._truncate(source.get("description"), max_chars=900),
                    "snippet": snippet,
                }
            )

        return results

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

    def _search_similar_images_sync(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        image_embedding: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        client = self._ensure_client()
        size = max(top_k, 1)
        body: dict[str, Any] = {
            "size": size,
            "_source": [
                "id",
                "artifact_id",
                "image_key",
                "museum_id",
                "original_image_name",
            ],
            "query": {
                "bool": {
                    "filter": self._build_image_filter_clauses(
                        museum_slug=museum_slug,
                        museum_id=museum_id,
                    ),
                    "must": [
                        {
                            "knn": {
                                "multimodal_embedding": {
                                    "vector": image_embedding,
                                    "k": max(size * 3, size),
                                }
                            }
                        }
                    ],
                }
            },
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
                "image_key": top_source.get("image_key"),
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

        results: list[dict[str, Any]] = []
        for hit in hits:
            source = hit.get("_source", {}) or {}
            original_image_name = str(source.get("original_image_name") or "").strip()
            artifact_id = str(source.get("artifact_id") or "").strip()
            if not original_image_name or not artifact_id:
                continue
            results.append(
                {
                    "score": hit.get("_score"),
                    "id": source.get("id"),
                    "artifact_id": artifact_id,
                    "image_key": source.get("image_key"),
                    "museum_id": source.get("museum_id"),
                    "original_image_name": original_image_name,
                }
            )
        return results

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
        size = max(top_k, 1)
        filter_clauses = self._build_image_filter_clauses(
            museum_slug=museum_slug,
            museum_id=museum_id,
        )

        should_clauses: list[dict[str, Any]] = []
        for index, embedding in enumerate(embeddings):
            boost = 2.0 if index == 0 else 1.0
            should_clauses.append(
                {
                    "knn": {
                        "multimodal_embedding": {
                            "vector": embedding,
                            "k": max(30, size * 3),
                            "boost": boost,
                        }
                    }
                }
            )

        body: dict[str, Any] = {
            "size": size,
            "_source": [
                "id",
                "artifact_id",
                "image_key",
                "museum_id",
                "original_image_name",
            ],
            "query": {
                "bool": {
                    "filter": filter_clauses,
                    "should": should_clauses,
                    "minimum_should_match": 1,
                }
            },
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
                "image_key": top_source.get("image_key"),
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

        results: list[dict[str, Any]] = []
        for hit in hits:
            source = hit.get("_source", {}) or {}
            original_image_name = str(source.get("original_image_name") or "").strip()
            artifact_id = str(source.get("artifact_id") or "").strip()
            if not original_image_name or not artifact_id:
                continue
            results.append(
                {
                    "score": hit.get("_score"),
                    "id": source.get("id"),
                    "artifact_id": artifact_id,
                    "image_key": source.get("image_key"),
                    "museum_id": source.get("museum_id"),
                    "original_image_name": original_image_name,
                }
            )
        return results

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
                "id",
                "artifact_id",
                "image_key",
                "museum_id",
                "original_image_name",
            ],
            "query": {
                "bool": {
                    "filter": filter_clauses,
                }
            },
        }
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
        seen_names_by_artifact: dict[str, set[str]] = {
            artifact_id: set() for artifact_id in unique_ids
        }
        for hit in hits:
            source = hit.get("_source", {}) or {}
            artifact_id = str(source.get("artifact_id") or "").strip()
            original_image_name = str(source.get("original_image_name") or "").strip()
            if not artifact_id or not original_image_name:
                continue
            if artifact_id not in by_artifact:
                continue
            if len(by_artifact[artifact_id]) >= per_artifact_value:
                continue
            if original_image_name in seen_names_by_artifact[artifact_id]:
                continue
            seen_names_by_artifact[artifact_id].add(original_image_name)
            by_artifact[artifact_id].append(
                {
                    "score": hit.get("_score"),
                    "id": source.get("id"),
                    "artifact_id": artifact_id,
                    "image_key": source.get("image_key"),
                    "museum_id": source.get("museum_id"),
                    "original_image_name": original_image_name,
                }
            )

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
                "inventory",
                "title",
                "museum_id",
                "museum_name",
                "location",
                "category",
                "initial_year",
                "final_year",
                "description",
                "full_text",
            ],
            "query": {
                "bool": {
                    "filter": filter_clauses,
                }
            },
        }
        #self._log(
            #logging.INFO,
            #"opensearch.artifact_fetch.body",
            #body=body,
        #)
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
                snippet = self._truncate(source.get("full_text"), max_chars=260)

            by_id[artifact_id] = {
                "score": hit.get("_score"),
                "artifact_id": artifact_id,
                "inventory": source.get("inventory"),
                "title": source.get("title"),
                "museum_id": source.get("museum_id"),
                "museum_name": source.get("museum_name"),
                "location": source.get("location"),
                "category": source.get("category"),
                "initial_year": source.get("initial_year"),
                "final_year": source.get("final_year"),
                "description": self._truncate(source.get("description"), max_chars=900),
                "snippet": snippet,
            }

        ordered: list[dict[str, Any]] = []
        for artifact_id in unique_ids:
            payload = by_id.get(artifact_id)
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
