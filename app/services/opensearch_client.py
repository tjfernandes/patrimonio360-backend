from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
import time
from typing import Any
import unicodedata
from urllib.parse import urlparse

from app.core.config import Settings, get_settings
from app.core.logging import log_event

logger = logging.getLogger(__name__)

NO_STEM_AND_FIELDS = [
    "title.no_stem^10",
    "description.no_stem^3",
]

NO_STEM_PHRASE_FIELDS = [
    "title.no_stem^8",
    "description.no_stem^2",
]

NO_STEM_OR_FIELDS = [
    "title.no_stem^3",
    "description.no_stem^1",
]

_RETRIEVAL_BOOST_ALIASES_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "retrieval_boost_aliases.json"
)


@lru_cache(maxsize=1)
def _load_retrieval_boost_aliases() -> dict[str, list[dict[str, Any]]]:
    try:
        payload = json.loads(_RETRIEVAL_BOOST_ALIASES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}

    loaded: dict[str, list[dict[str, Any]]] = {}
    for group in ("category", "support_or_material", "technique"):
        entries = payload.get(group)
        if isinstance(entries, list):
            loaded[group] = [entry for entry in entries if isinstance(entry, dict)]
    return loaded


def _fold_boost_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", (text or "").casefold())
    folded = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", folded, flags=re.UNICODE).strip()


def _boost_tokens(text: str) -> list[str]:
    folded = _fold_boost_text(text)
    return folded.split() if folded else []


def _contains_token_sequence(tokens: list[str], alias_tokens: list[str]) -> bool:
    if not tokens or not alias_tokens or len(alias_tokens) > len(tokens):
        return False
    alias_size = len(alias_tokens)
    return any(
        tokens[index : index + alias_size] == alias_tokens
        for index in range(len(tokens) - alias_size + 1)
    )


def _coerce_boost(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _import_opensearch() -> Any:
    try:
        from opensearchpy import OpenSearch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency 'opensearch-py'. Install backend requirements first."
        ) from exc
    return OpenSearch




@dataclass(slots=True)
class OpenSearchRetrievalPage:
    results: list[dict[str, Any]]
    total: int
    query_body: dict[str, Any] | None = None


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
        log_event(
            logger,
            logging.INFO,
            "opensearch.client.create",
            host=host,
            scheme=scheme,
            port=port,
            url_prefix=url_prefix or None,
            verify_certs=self.settings.OPENSEARCH_VERIFY_CERTS,
            has_auth=bool(self.settings.OPENSEARCH_USERNAME and self.settings.OPENSEARCH_PASSWORD),
        )

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

    async def ensure_ready(self) -> bool:
        started_at = time.perf_counter()
        log_event(logger, logging.INFO, "opensearch.ready.start")
        try:
            client = self._ensure_client()
            await asyncio.to_thread(client.info)
        except Exception as exc:
            duration_ms = (time.perf_counter() - started_at) * 1000
            log_event(
                logger,
                logging.ERROR,
                "opensearch.ready.error",
                duration_ms=round(duration_ms, 1),
                error=exc,
            )
            raise
        duration_ms = (time.perf_counter() - started_at) * 1000
        log_event(
            logger,
            logging.INFO,
            "opensearch.ready.finish",
            duration_ms=round(duration_ms, 1),
        )
        return True

    async def _to_thread_logged(
        self,
        event: str,
        func: Any,
        *,
        log_fields: dict[str, Any] | None = None,
        result_size: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        fields = dict(log_fields or {})
        started_at = time.perf_counter()
        log_event(logger, logging.INFO, f"{event}.start", **fields)
        try:
            result = await asyncio.to_thread(func, **kwargs)
        except Exception as exc:
            duration_ms = (time.perf_counter() - started_at) * 1000
            log_event(
                logger,
                logging.ERROR,
                f"{event}.error",
                duration_ms=round(duration_ms, 1),
                error=exc,
                **fields,
            )
            raise
        duration_ms = (time.perf_counter() - started_at) * 1000
        if result_size is not None:
            try:
                fields["result_count"] = result_size(result)
            except Exception:
                pass
        if isinstance(result, OpenSearchRetrievalPage):
            fields["result_count"] = len(result.results)
            fields["total"] = result.total
        log_event(
            logger,
            logging.INFO,
            f"{event}.finish",
            duration_ms=round(duration_ms, 1),
            **fields,
        )
        return result

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
            "creators",
            "creator_ids",
            "sets",
            "set_ids",
            "set_numbers",
            "exhibitions",
            "exhibition_ids",
            "exhibition_types",
            "date_or_period",
            "support_or_material",
            "technique",
            "origin_history",
            "incorporation",
            "production_center",
            "inventory_number",
            "detail_type",
            "tipo_inventario",
            "source_system",
        }

        numeric_fields = {
            "image_count",
            "image_order",
            "width",
            "height",
            "exhibition_count",
            "bibliography_count",
            "start_year",
            "end_year",
        }
        boolean_fields = {
            "in_tour",
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

            if key == "_temporal_interval":
                if not isinstance(value, dict):
                    continue

                query_start_year = value.get("start_year")
                query_end_year = value.get("end_year")

                if (
                    not isinstance(query_start_year, int)
                    or isinstance(query_start_year, bool)
                    or not isinstance(query_end_year, int)
                    or isinstance(query_end_year, bool)
                ):
                    continue

                # Proteção contra intervalos invertidos.
                if query_start_year > query_end_year:
                    query_start_year, query_end_year = (
                        query_end_year,
                        query_start_year,
                    )

                clauses.append(
                    {
                        "bool": {
                            "should": [
                                # Caso 1:
                                # Documento com start_year e end_year.
                                #
                                # Há sobreposição quando:
                                # document.start_year <= query.end_year
                                # AND
                                # document.end_year >= query.start_year
                                {
                                    "bool": {
                                        "filter": [
                                            {
                                                "range": {
                                                    "start_year": {
                                                        "lte": query_end_year,
                                                    }
                                                }
                                            },
                                            {
                                                "range": {
                                                    "end_year": {
                                                        "gte": query_start_year,
                                                    }
                                                }
                                            },
                                        ]
                                    }
                                },

                                # Caso 2:
                                # Documento com apenas start_year.
                                #
                                # O start_year é tratado como uma data pontual
                                # e tem de estar dentro do intervalo pesquisado.
                                {
                                    "bool": {
                                        "filter": [
                                            {
                                                "range": {
                                                    "start_year": {
                                                        "gte": query_start_year,
                                                        "lte": query_end_year,
                                                    }
                                                }
                                            }
                                        ],
                                        "must_not": [
                                            {
                                                "exists": {
                                                    "field": "end_year",
                                                }
                                            }
                                        ],
                                    }
                                },
                            ],
                            "minimum_should_match": 1,
                        }
                    }
                )
                continue

            resolved_key = legacy_keyword_aliases.get(key, key)

            if resolved_key in boolean_fields:
                if isinstance(value, bool):
                    clauses.append({"term": {resolved_key: value}})
                continue

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
                        if (
                            op in {"gt", "gte", "lt", "lte"}
                            and isinstance(val, (int, float))
                            and not isinstance(val, bool)
                        )
                    }

                    if range_payload:
                        clauses.append(
                            {
                                "range": {
                                    resolved_key: range_payload,
                                }
                            }
                        )

                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    clauses.append(
                        {
                            "term": {
                                resolved_key: value,
                            }
                        }
                    )

                continue

            if (
                key.endswith("_gte")
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                field = key[:-4]

                if field in numeric_fields:
                    clauses.append(
                        {
                            "range": {
                                field: {
                                    "gte": value,
                                }
                            }
                        }
                    )

                continue

            if (
                key.endswith("_lte")
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                field = key[:-4]

                if field in numeric_fields:
                    clauses.append(
                        {
                            "range": {
                                field: {
                                    "lte": value,
                                }
                            }
                        }
                    )

                continue

        return clauses


    @staticmethod
    def _temporal_known_year_matches_interval(
        *,
        document_start_year: int | None,
        document_end_year: int | None,
        query_start_year: int,
        query_end_year: int,
    ) -> bool:
        """
        Verifica se a cronologia conhecida de um documento corresponde
        ao intervalo cronológico da query.

        Regras:
        - start_year + end_year: verificar sobreposição de intervalos;
        - apenas start_year: tratar como ano pontual;
        - sem start_year: não corresponde.

        Por contrato dos dados, end_year não existe sem start_year.
        """

        if query_start_year > query_end_year:
            query_start_year, query_end_year = (
                query_end_year,
                query_start_year,
            )

        if document_start_year is None:
            return False

        if document_end_year is None:
            return query_start_year <= document_start_year <= query_end_year

        return (
            document_start_year <= query_end_year
            and document_end_year >= query_start_year
        )


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

    def _build_in_tour_boost_clause(self, *, boost: float | None = None) -> dict[str, Any] | None:
        # boost=None -> usa o boost de texto (CHAT_IN_TOUR_BOOST). Pesquisas por
        # imagem/modelo passam boost=IMAGE_IN_TOUR_BOOST.
        resolved = self._configured_query_boost() if boost is None else float(boost or 0)
        if resolved <= 0:
            return None
        return {"term": {"in_tour": {"value": True, "boost": resolved}}}

    def _configured_query_boost(self) -> float:
        """Boost de in_tour para pesquisas de texto."""
        return float(getattr(self.settings, "CHAT_IN_TOUR_BOOST", 0) or 0)

    def _configured_image_boost(self) -> float:
        """Boost de in_tour para pesquisas por imagem/modelo 3D."""
        return float(getattr(self.settings, "IMAGE_IN_TOUR_BOOST", 0) or 0)

    def _is_in_tour(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes"}
        if isinstance(value, (int, float)):
            return bool(value)
        return False

    def _artifact_candidate_size(self, top_k: int, *, boost: float | None = None) -> int:
        base = max(top_k, 1)
        resolved = self._configured_query_boost() if boost is None else float(boost or 0)
        if resolved <= 0:
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

    def _as_list_of_strings(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        out: list[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                out.append(text)
        return out

    def _coerce_int(self, value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(value.strip())
        return None

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

        creators = self._as_list_of_strings(source.get("creators"))
        # creator (string legada) -> popular creators se vier sem o array novo.
        legacy_creator = str(source.get("creator") or "").strip()
        if not creators and legacy_creator:
            creators = [legacy_creator]

        payload: dict[str, Any] = {
            "score": score,
            "artifact_id": source.get("artifact_id"),
            "tipo_inventario": str(source.get("tipo_inventario") or "").strip() or None,
            "in_tour": self._is_in_tour(source.get("in_tour")),
            "inventory_number": inventory_number or None,
            "title": source.get("title"),
            "museum_id": source.get("museum_id"),
            "museum": museum_name or None,
            "category": source.get("category"),
            "super_category": source.get("super_category"),
            # Autores (novo array) + string legada para back-compat.
            "creators": creators,
            "creator_ids": self._as_list_of_strings(source.get("creator_ids")),
            "creator": legacy_creator or (creators[0] if creators else None),
            "date_or_period": source.get("date_or_period"),
            "start_year": self._coerce_int(source.get("start_year")),
            "end_year": self._coerce_int(source.get("end_year")),
            "support_or_material": support_or_material or None,
            "technique": source.get("technique"),
            # Conjuntos.
            "sets": self._as_list_of_strings(source.get("sets")),
            "set_ids": self._as_list_of_strings(source.get("set_ids")),
            "set_numbers": self._as_list_of_strings(source.get("set_numbers")),
            # Exposicoes (fisica + online no mesmo array com tipo).
            "exhibitions": self._as_list_of_strings(source.get("exhibitions")),
            "exhibition_ids": self._as_list_of_strings(source.get("exhibition_ids")),
            "exhibition_types": self._as_list_of_strings(source.get("exhibition_types")),
            "exhibition_count": self._coerce_int(source.get("exhibition_count")),
            # Bibliografia (so como texto pesquisavel).
            "bibliography": str(source.get("bibliography") or "").strip() or None,
            "bibliography_count": self._coerce_int(source.get("bibliography_count")),
            "origin_history": origin_history or None,
            "incorporation": source.get("incorporation"),
            "production_center": production_center or None,
            "description": str(source.get("description") or "").strip() or None,
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
            "image_order": self._coerce_int(source.get("image_order")),
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

    def _matched_retrieval_boost_clauses(
        self,
        *,
        query_text: str,
        lexical_query: str | None,
    ) -> list[dict[str, Any]]:
        boosts = self._matched_retrieval_boosts(
            query_text=query_text,
            lexical_query=lexical_query,
        )
        clauses: list[dict[str, Any]] = []

        for boost in boosts:
            kind = boost.get("kind")
            field = str(boost.get("field") or "").strip()
            boost_value = boost.get("boost")
            if kind == "term" and field == "category":
                value = str(boost.get("value") or "").strip()
                if not value:
                    continue
                clauses.append(
                    {
                        "term": {
                            "category": {
                                "value": value,
                                "boost": boost_value,
                            }
                        }
                    }
                )
            elif kind == "match" and field:
                query = str(boost.get("query") or "").strip()
                if not query:
                    continue
                clauses.append(
                    {
                        "match": {
                            field: {
                                "query": query,
                                "boost": boost_value,
                            }
                        }
                    }
                )

        return clauses

    def matched_retrieval_boosts(
        self,
        *,
        query_text: str,
        lexical_query: str | None,
    ) -> list[dict[str, Any]]:
        return [
            {
                key: value
                for key, value in boost.items()
                if key in {"group", "kind", "field", "query", "value", "boost", "matched_alias"}
            }
            for boost in self._matched_retrieval_boosts(
                query_text=query_text,
                lexical_query=lexical_query,
            )
        ]

    def _matched_retrieval_boosts(
        self,
        *,
        query_text: str,
        lexical_query: str | None,
    ) -> list[dict[str, Any]]:
        tokens = _boost_tokens(f"{query_text or ''} {lexical_query or ''}")
        if not tokens:
            return []

        config = _load_retrieval_boost_aliases()
        boosts: list[dict[str, Any]] = []

        def matched_alias(aliases: list[Any]) -> str | None:
            for alias in aliases:
                alias_text = str(alias)
                if _contains_token_sequence(tokens, _boost_tokens(alias_text)):
                    return alias_text
            return None

        def add_text_match_boosts(*, group: str, field: str, fallback_boost: float) -> None:
            seen_queries: set[str] = set()
            for entry in config.get(group, []):
                query = str(entry.get("query") or "").strip()
                aliases = entry.get("aliases")
                if not query or not isinstance(aliases, list) or query in seen_queries:
                    continue
                alias = matched_alias(aliases)
                if not alias:
                    continue
                seen_queries.add(query)
                boost = _coerce_boost(entry.get("boost"), fallback_boost)
                boosts.append(
                    {
                        "group": group,
                        "kind": "match",
                        "field": field,
                        "query": query,
                        "boost": boost,
                        "matched_alias": alias,
                    }
                )

        seen_categories: set[str] = set()
        for entry in config.get("category", []):
            value = str(entry.get("value") or "").strip()
            aliases = entry.get("aliases")
            if not value or not isinstance(aliases, list) or value in seen_categories:
                continue
            alias = matched_alias(aliases)
            if not alias:
                continue
            seen_categories.add(value)
            boost = _coerce_boost(entry.get("boost"), 1.0)
            boosts.append(
                {
                    "group": "category",
                    "kind": "term",
                    "field": "category",
                    "value": value,
                    "boost": boost,
                    "matched_alias": alias,
                }
            )

        add_text_match_boosts(
            group="support_or_material",
            field="support_or_material.text",
            fallback_boost=1.0,
        )
        add_text_match_boosts(
            group="technique",
            field="technique.text",
            fallback_boost=1.0,
        )

        return boosts

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
        knn_k = 1000
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
        retrieval_boost_clauses = self._matched_retrieval_boost_clauses(
            query_text=query_text,
            lexical_query=lexical_query_value,
        )
        if lexical_query_value and not embedding_only:
            lexical_bool_query: dict[str, Any] = {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": lexical_query_value,
                                "fields": NO_STEM_AND_FIELDS,
                                "type": "best_fields",
                                "operator": "and",
                                "boost": 5,
                            }
                        },
                        {
                            "multi_match": {
                                "query": lexical_query_value,
                                "fields": NO_STEM_PHRASE_FIELDS,
                                "type": "phrase",
                                "boost": 4,
                            }
                        },
                        {
                            "multi_match": {
                                "query": lexical_query_value,
                                "fields": NO_STEM_OR_FIELDS,
                                "type": "best_fields",
                                "operator": "or",
                                "minimum_should_match": "2<75%",
                                "boost": 0.3,
                            }
                        },
                    ],
                    "minimum_should_match": 1,
                }
            }
            if filter_clauses or in_tour_boost_clause or retrieval_boost_clauses:
                bool_query = {"must": [lexical_bool_query]}
                if filter_clauses:
                    bool_query["filter"] = filter_clauses
                should_clauses: list[dict[str, Any]] = []
                if in_tour_boost_clause:
                    should_clauses.append(in_tour_boost_clause)
                should_clauses.extend(retrieval_boost_clauses)
                if should_clauses:
                    bool_query["should"] = should_clauses
                hybrid_queries.append({"bool": bool_query})
            else:
                hybrid_queries.append(lexical_bool_query)

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
                "tipo_inventario",
                "inventory_number",
                "title",
                "museum_id",
                "museum",
                "category",
                "super_category",
                "creator",
                "creators",
                "creator_ids",
                "sets",
                "set_ids",
                "set_numbers",
                "exhibitions",
                "exhibition_ids",
                "exhibition_types",
                "exhibition_count",
                "bibliography",
                "bibliography_count",
                "date_or_period",
                "start_year",
                "end_year",
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

        search_kwargs: dict[str, Any] = {
            "index": self.settings.OPENSEARCH_INDEX_ARTIFACT,
            "body": body,
        }
        if not self.settings.CHAT_RETRIEVAL_EMBEDDING_ONLY:
            search_kwargs["search_pipeline"] = "nlp-search-pipeline"

        response = client.search(**search_kwargs)
        hits = response.get("hits", {}).get("hits", [])
        total_value = self._total_hits_value(response)
        results = self._artifact_results_from_hits(hits)
        return OpenSearchRetrievalPage(
            results=results,
            total=total_value,
            query_body=dict(search_kwargs),
        )

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
        return await self._to_thread_logged(
            "opensearch.search_relevant_context",
            self._search_relevant_context_sync,
            log_fields={
                "museum_slug": museum_slug,
                "museum_id": museum_id,
                "top_k": top_k,
                "has_filters": bool(filters),
                "has_sort": bool(sort),
                "embedding_dim": len(query_embedding),
            },
            result_size=len,
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
        return await self._to_thread_logged(
            "opensearch.search_relevant_context_page",
            self._search_relevant_context_page_sync,
            log_fields={
                "museum_slug": museum_slug,
                "museum_id": museum_id,
                "from_offset": from_offset,
                "page_size": page_size,
                "retrieval_window_size": retrieval_window_size,
                "has_filters": bool(filters),
                "has_sort": bool(sort),
                "embedding_dim": len(query_embedding),
            },
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
        image_boost = self._configured_image_boost()
        size = self._artifact_candidate_size(requested_top_k, boost=image_boost)
        in_tour_boost_clause = self._build_in_tour_boost_clause(boost=image_boost)
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
                "image_order",
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
        in_tour_boost_clause = self._build_in_tour_boost_clause(boost=self._configured_image_boost())
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
                "image_order",
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
        search_kwargs: dict[str, Any] = {
            "index": self.settings.OPENSEARCH_INDEX_IMAGE,
            "body": body,
        }
        response = client.search(**search_kwargs)
        hits = response.get("hits", {}).get("hits", [])
        total_value = self._total_hits_value(response)
        return OpenSearchRetrievalPage(
            results=self._image_results_from_hits(hits),
            total=total_value,
            query_body=dict(search_kwargs),
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
        image_boost = self._configured_image_boost()
        size = self._artifact_candidate_size(requested_top_k, boost=image_boost)
        filter_clauses = self._build_image_filter_clauses(
            museum_slug=museum_slug,
            museum_id=museum_id,
        )
        in_tour_boost_clause = self._build_in_tour_boost_clause(boost=image_boost)

        should_clauses: list[dict[str, Any]] = []
        for embedding in embeddings:
            should_clauses.append(
                {
                    "knn": {
                        "visual_embedding": {
                            "vector": embedding,
                            "k": max(30, size * 3),
                            # Ponderacao por imagem (modelo 3D -> varias vistas),
                            # nao confundir com o boost de in_tour.
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
                "image_order",
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
        in_tour_boost_clause = self._build_in_tour_boost_clause(boost=self._configured_image_boost())

        should_clauses: list[dict[str, Any]] = []
        for embedding in embeddings:
            should_clauses.append(
                {
                    "knn": {
                        "visual_embedding": {
                            "vector": embedding,
                            "k": knn_k,
                            # Ponderacao por imagem (modelo 3D -> varias vistas),
                            # nao confundir com o boost de in_tour.
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
                "image_order",
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
        search_kwargs = {
            "index": self.settings.OPENSEARCH_INDEX_IMAGE,
            "body": body,
        }
        response = client.search(**search_kwargs)
        hits = response.get("hits", {}).get("hits", [])
        total_value = self._total_hits_value(response)
        return OpenSearchRetrievalPage(
            results=self._image_results_from_hits(hits),
            total=total_value,
            query_body=dict(search_kwargs),
        )

    async def search_similar_images(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        image_embedding: list[float],
        top_k: int = 6,
    ) -> list[dict[str, Any]]:
        return await self._to_thread_logged(
            "opensearch.search_similar_images",
            self._search_similar_images_sync,
            log_fields={
                "museum_slug": museum_slug,
                "museum_id": museum_id,
                "top_k": top_k,
                "embedding_dim": len(image_embedding),
            },
            result_size=len,
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
        return await self._to_thread_logged(
            "opensearch.search_similar_images_page",
            self._search_similar_images_page_sync,
            log_fields={
                "museum_slug": museum_slug,
                "museum_id": museum_id,
                "from_offset": from_offset,
                "page_size": page_size,
                "retrieval_window_size": retrieval_window_size,
                "embedding_dim": len(image_embedding),
            },
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
        return await self._to_thread_logged(
            "opensearch.search_similar_images_multi",
            self._search_similar_images_multi_sync,
            log_fields={
                "museum_slug": museum_slug,
                "museum_id": museum_id,
                "top_k": top_k,
                "embedding_count": len(image_embeddings),
                "embedding_dim": len(image_embeddings[0]) if image_embeddings else 0,
            },
            result_size=len,
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
        return await self._to_thread_logged(
            "opensearch.search_similar_images_multi_page",
            self._search_similar_images_multi_page_sync,
            log_fields={
                "museum_slug": museum_slug,
                "museum_id": museum_id,
                "from_offset": from_offset,
                "page_size": page_size,
                "retrieval_window_size": retrieval_window_size,
                "embedding_count": len(image_embeddings),
                "embedding_dim": len(image_embeddings[0]) if image_embeddings else 0,
            },
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
                "image_order",
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
            "sort": [
                {
                    "image_order": {
                        "order": "asc",
                        "missing": "_last",
                        "unmapped_type": "long",
                    }
                }
            ],
        }
        if per_artifact_value == 1:
            body["collapse"] = {"field": "artifact_id"}
        response = client.search(
            index=self.settings.OPENSEARCH_INDEX_IMAGE,
            body=body,
        )
        hits = response.get("hits", {}).get("hits", [])
        total_obj = response.get("hits", {}).get("total", {})
        total_value = total_obj.get("value") if isinstance(total_obj, dict) else total_obj

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

        def _image_order_sort_key(payload: dict[str, Any]) -> tuple[int, int]:
            image_order = payload.get("image_order")
            if isinstance(image_order, int):
                return (0, image_order)
            return (1, 0)

        ordered: list[dict[str, Any]] = []
        for artifact_id in unique_ids:
            for image_payload in sorted(
                by_artifact.get(artifact_id, []),
                key=_image_order_sort_key,
            ):
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
        return await self._to_thread_logged(
            "opensearch.fetch_images_by_artifact_ids",
            self._fetch_images_by_artifact_ids_sync,
            log_fields={
                "museum_slug": museum_slug,
                "museum_id": museum_id,
                "artifact_count": len(artifact_ids),
                "per_artifact": per_artifact,
                "max_total": max_total,
            },
            result_size=len,
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
                "tipo_inventario",
                "inventory_number",
                "title",
                "museum_id",
                "museum",
                "category",
                "super_category",
                "creator",
                "creators",
                "creator_ids",
                "sets",
                "set_ids",
                "set_numbers",
                "exhibitions",
                "exhibition_ids",
                "exhibition_types",
                "exhibition_count",
                "bibliography",
                "bibliography_count",
                "date_or_period",
                "start_year",
                "end_year",
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
            "query": {
                "bool": {
                    "filter": filter_clauses,
                }
            },
        }
        response = client.search(
            index=self.settings.OPENSEARCH_INDEX_ARTIFACT,
            body=body,
        )
        hits = response.get("hits", {}).get("hits", [])
        total_obj = response.get("hits", {}).get("total", {})
        total_value = total_obj.get("value") if isinstance(total_obj, dict) else total_obj

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
                "tipo_inventario",
                "inventory_number",
                "title",
                "museum_id",
                "museum",
                "category",
                "super_category",
                "creator",
                "creators",
                "creator_ids",
                "sets",
                "set_ids",
                "set_numbers",
                "exhibitions",
                "exhibition_ids",
                "exhibition_types",
                "exhibition_count",
                "bibliography",
                "bibliography_count",
                "date_or_period",
                "start_year",
                "end_year",
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
        response = client.search(
            index=self.settings.OPENSEARCH_INDEX_ARTIFACT,
            body=body,
        )
        hits = response.get("hits", {}).get("hits", [])
        total_obj = response.get("hits", {}).get("total", {})
        total_value = total_obj.get("value") if isinstance(total_obj, dict) else total_obj

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

    def _search_artifacts_by_inventory_candidates_once(
        self,
        *,
        client: Any,
        inventory_numbers: list[str],
        top_k: int,
        museum_id: str | None,
    ) -> list[dict[str, Any]]:
        inventory_terms: list[str] = []
        seen_terms: set[str] = set()
        for value in inventory_numbers:
            cleaned = str(value or "").strip()
            if not cleaned:
                continue
            for candidate in (cleaned, cleaned.casefold()):
                if candidate and candidate not in seen_terms:
                    seen_terms.add(candidate)
                    inventory_terms.append(candidate)

        if not inventory_terms:
            return []

        should_clauses: list[dict[str, Any]] = []
        for candidate in inventory_terms:
            should_clauses.append({"term": {"inventory_number": candidate}})
        for candidate in inventory_numbers:
            cleaned = str(candidate or "").strip()
            if cleaned:
                should_clauses.append({"match": {"inventory_number.text": cleaned}})

        bool_query: dict[str, Any] = {
            "must": [
                {
                    "bool": {
                        "should": should_clauses,
                        "minimum_should_match": 1,
                    }
                }
            ]
        }
        resolved_museum_id = str(museum_id or "").strip()
        if resolved_museum_id:
            bool_query["filter"] = [{"term": {"museum_id": resolved_museum_id}}]

        body: dict[str, Any] = {
            "size": max(int(top_k), 1),
            "_source": [
                "artifact_id",
                "tipo_inventario",
                "inventory_number",
                "title",
                "museum_id",
                "museum",
                "category",
                "super_category",
                "creator",
                "creators",
                "creator_ids",
                "sets",
                "set_ids",
                "set_numbers",
                "exhibitions",
                "exhibition_ids",
                "exhibition_types",
                "exhibition_count",
                "bibliography",
                "bibliography_count",
                "date_or_period",
                "start_year",
                "end_year",
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
            "query": {"bool": bool_query},
        }

        response = client.search(
            index=self.settings.OPENSEARCH_INDEX_ARTIFACT,
            body=body,
        )
        hits = response.get("hits", {}).get("hits", [])
        return self._artifact_results_from_hits(hits)

    def _search_artifacts_by_inventory_candidates_sync(
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

        client = self._ensure_client()
        size = min(max(top_k, 1), 5)
        resolved_museum_id = str(museum_id or "").strip()
        return self._search_artifacts_by_inventory_candidates_once(
            client=client,
            inventory_numbers=unique_inventories,
            top_k=size,
            museum_id=resolved_museum_id or None,
        )

    async def fetch_artifacts_by_ids(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        artifact_ids: list[str],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        return await self._to_thread_logged(
            "opensearch.fetch_artifacts_by_ids",
            self._fetch_artifacts_by_ids_sync,
            log_fields={
                "museum_slug": museum_slug,
                "museum_id": museum_id,
                "id_count": len(artifact_ids),
                "top_k": top_k,
            },
            result_size=len,
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
        return await self._to_thread_logged(
            "opensearch.fetch_artifacts_by_inventory_numbers",
            self._fetch_artifacts_by_inventory_numbers_sync,
            log_fields={
                "museum_slug": museum_slug,
                "museum_id": museum_id,
                "inventory_count": len(inventory_numbers),
                "top_k": top_k,
            },
            result_size=len,
            museum_slug=museum_slug,
            museum_id=museum_id,
            inventory_numbers=inventory_numbers,
            top_k=top_k,
        )

    async def search_artifacts_by_inventory_candidates(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        inventory_numbers: list[str],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        return await self._to_thread_logged(
            "opensearch.search_artifacts_by_inventory_candidates",
            self._search_artifacts_by_inventory_candidates_sync,
            log_fields={
                "museum_slug": museum_slug,
                "museum_id": museum_id,
                "inventory_count": len(inventory_numbers),
                "top_k": top_k,
            },
            result_size=len,
            museum_slug=museum_slug,
            museum_id=museum_id,
            inventory_numbers=inventory_numbers,
            top_k=top_k,
        )

    # --------------------------------------------------------------------- #
    # Entidades relacionais (autores / conjuntos / exposicoes).
    # --------------------------------------------------------------------- #
    _ENTITY_INDEX_ATTR = {
        "autor": "OPENSEARCH_INDEX_AUTOR",
        "conjunto": "OPENSEARCH_INDEX_CONJUNTO",
        "exposicao": "OPENSEARCH_INDEX_EXPOSICAO",
    }
    # Campo do artifact_doc que liga ao indice da entidade, por tipo.
    _ENTITY_ARTIFACT_REF_FIELD = {
        "autor": "creator_ids",
        "conjunto": "set_ids",
        "exposicao": "exhibition_ids",
    }
    _ENTITY_SOURCE_FIELDS = {
        "autor": [
            "entity_id", "tipo_entidade", "name", "atividade",
            "data_nascimento", "data_obito", "local_nascimento", "local_obito",
            "biografia", "biography", "url", "museums", "n_objetos", "objetos",
        ],
        "conjunto": [
            "entity_id", "tipo_entidade", "name", "num_conjunto",
            "historial", "descricao", "url", "museums", "n_objetos", "objetos",
        ],
        "exposicao": [
            "entity_id", "tipo_entidade", "name", "tipo_exposicao",
            "local", "ano_inicial", "ano_final", "texto", "ficha_tecnica",
            "url", "museums", "n_objetos", "objetos",
        ],
    }

    def _entity_index(self, tipo: str) -> str:
        attr = self._ENTITY_INDEX_ATTR.get(tipo)
        if not attr:
            raise RuntimeError(f"Tipo de entidade desconhecido: {tipo}")
        return getattr(self.settings, attr)

    def _fetch_entities_by_ids_sync(
        self,
        *,
        tipo: str,
        entity_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Devolve docs do indice da entidade pelos entity_ids dados, na mesma ordem."""
        unique_ids: list[str] = []
        seen: set[str] = set()
        for value in entity_ids:
            cleaned = str(value or "").strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            unique_ids.append(cleaned)
        if not unique_ids:
            return []

        client = self._ensure_client()
        index_name = self._entity_index(tipo)
        source_fields = self._ENTITY_SOURCE_FIELDS.get(tipo, ["entity_id", "name"])
        body: dict[str, Any] = {
            "size": len(unique_ids),
            "_source": source_fields,
            "query": {"terms": {"entity_id": unique_ids}},
        }
        response = client.search(index=index_name, body=body)
        hits = response.get("hits", {}).get("hits", []) or []

        by_id: dict[str, dict[str, Any]] = {}
        for hit in hits:
            source = hit.get("_source", {}) or {}
            eid = str(source.get("entity_id") or "").strip()
            if eid:
                by_id[eid] = source

        ordered: list[dict[str, Any]] = []
        for eid in unique_ids:
            entry = by_id.get(eid)
            if entry is not None:
                ordered.append(entry)
        return ordered

    async def fetch_entities_by_ids(
        self,
        *,
        tipo: str,
        entity_ids: list[str],
    ) -> list[dict[str, Any]]:
        return await self._to_thread_logged(
            "opensearch.fetch_entities_by_ids",
            self._fetch_entities_by_ids_sync,
            log_fields={
                "tipo": tipo,
                "entity_count": len(entity_ids),
            },
            result_size=len,
            tipo=tipo,
            entity_ids=entity_ids,
        )

    def _fetch_authors_by_ids_sync(
        self,
        *,
        author_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Devolve autores pelo _id do indice cultural_heritage_authors."""
        unique_ids: list[str] = []
        seen: set[str] = set()
        for value in author_ids:
            cleaned = str(value or "").strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            unique_ids.append(cleaned)
        if not unique_ids:
            return []

        client = self._ensure_client()
        index_name = self._entity_index("autor")
        body: dict[str, Any] = {
            "size": len(unique_ids),
            "_source": self._ENTITY_SOURCE_FIELDS["autor"],
            "query": {"ids": {"values": unique_ids}},
        }
        response = client.search(index=index_name, body=body)
        hits = response.get("hits", {}).get("hits", []) or []

        by_id: dict[str, dict[str, Any]] = {}
        for hit in hits:
            source = dict(hit.get("_source", {}) or {})
            doc_id = str(hit.get("_id") or "").strip()
            entity_id = str(source.get("entity_id") or "").strip()
            if not source.get("entity_id") and doc_id:
                source["entity_id"] = doc_id
            if doc_id:
                by_id[doc_id] = source
            if entity_id:
                by_id[entity_id] = source

        ordered: list[dict[str, Any]] = []
        for author_id in unique_ids:
            entry = by_id.get(author_id)
            if entry is not None:
                ordered.append(entry)
        return ordered

    async def fetch_authors_by_ids(
        self,
        *,
        author_ids: list[str],
    ) -> list[dict[str, Any]]:
        return await self._to_thread_logged(
            "opensearch.fetch_authors_by_ids",
            self._fetch_authors_by_ids_sync,
            log_fields={"author_count": len(author_ids)},
            result_size=len,
            author_ids=author_ids,
        )

    def _fetch_authors_by_names_sync(
        self,
        *,
        names: list[str],
        top_k: int,
    ) -> list[dict[str, Any]]:
        unique_names: list[str] = []
        seen: set[str] = set()
        for value in names:
            cleaned = str(value or "").strip()
            key = cleaned.casefold()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            unique_names.append(cleaned)
        if not unique_names:
            return []

        should_clauses: list[dict[str, Any]] = []
        for name in unique_names:
            should_clauses.extend(
                [
                    {
                        "match_phrase": {
                            "name": {
                                "query": name,
                                "boost": self._configured_query_boost(),
                            }
                        }
                    },
                    {
                        "multi_match": {
                            "query": name,
                            "fields": ["name^4", "biografia^0.8", "biography^0.8", "atividade^0.5"],
                            "operator": "and",
                        }
                    },
                ]
            )

        client = self._ensure_client()
        index_name = self._entity_index("autor")
        size = max(min(int(top_k), 50), 1)
        body: dict[str, Any] = {
            "size": size,
            "_source": self._ENTITY_SOURCE_FIELDS["autor"],
            "query": {
                "bool": {
                    "should": should_clauses,
                    "minimum_should_match": 1,
                }
            },
        }
        response = client.search(index=index_name, body=body)
        hits = response.get("hits", {}).get("hits", []) or []

        results: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for hit in hits:
            source = hit.get("_source", {}) or {}
            entity_id = str(source.get("entity_id") or "").strip()
            key = entity_id or str(source.get("name") or "").strip().casefold()
            if key and key in seen_ids:
                continue
            if key:
                seen_ids.add(key)
            results.append(source)
        return results

    async def fetch_authors_by_names(
        self,
        *,
        names: list[str],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        return await self._to_thread_logged(
            "opensearch.fetch_authors_by_names",
            self._fetch_authors_by_names_sync,
            log_fields={
                "name_count": len(names),
                "top_k": top_k,
            },
            result_size=len,
            names=names,
            top_k=top_k,
        )

    def _fetch_artifacts_by_entity_sync(
        self,
        *,
        tipo: str,
        entity_id: str,
        museum_slug: str,
        museum_id: str | None,
        top_k: int,
        from_offset: int = 0,
        exclude_artifact_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Artefactos ligados a uma entidade. Devolve (resultados, total)."""
        ref_field = self._ENTITY_ARTIFACT_REF_FIELD.get(tipo)
        if not ref_field:
            raise RuntimeError(f"Tipo de entidade desconhecido: {tipo}")
        eid = str(entity_id or "").strip()
        if not eid:
            return [], 0
        # Para entidade=conjunto/exposicao, set_ids guarda o id NAO prefixado
        # (ex.: "389"), porque vem direto do build_raiz_index. O entity_id da
        # entidade no indexer e "conjunto:389". Tentamos ambos.
        prefix_to_strip = f"{tipo}:"
        bare_id = eid[len(prefix_to_strip):] if eid.startswith(prefix_to_strip) else eid
        # Exposicoes vem como "exposicao:fisica:8892" ou "exposicao:online:12"
        # e o artifact guarda em exhibition_ids "fisica:8892"/"online:12" -- a
        # diferenca da chave compoe-se ao retirar "exposicao:".
        ref_values = [bare_id]
        if eid != bare_id:
            ref_values.append(eid)

        client = self._ensure_client()
        size = max(int(top_k), 1)
        from_value = max(int(from_offset), 0)
        filter_clauses: list[dict[str, Any]] = [{"terms": {ref_field: ref_values}}]
        filter_clauses.extend(
            self._build_image_filter_clauses(museum_slug=museum_slug, museum_id=museum_id)
        )
        must_not: list[dict[str, Any]] = []
        if exclude_artifact_id:
            must_not.append({"term": {"artifact_id": exclude_artifact_id}})

        body: dict[str, Any] = {
            "from": from_value,
            "size": size,
            "track_total_hits": True,
            "_source": [
                "artifact_id", "tipo_inventario", "inventory_number", "title",
                "museum_id", "museum", "category", "super_category",
                "creators", "creator_ids", "date_or_period",
                "description", "search_text", "detail_type", "detail_url",
                "in_tour",
                "image_ids", "image_file_ids", "image_paths", "image_urls", "image_count",
            ],
            "query": {
                "bool": {
                    "filter": filter_clauses,
                    **({"must_not": must_not} if must_not else {}),
                }
            },
            "sort": [
                # Prioriza objetos in_tour, depois ordem natural.
                {"in_tour": {"order": "desc", "missing": "_last"}},
                {"_score": {"order": "desc"}},
            ],
        }
        response = client.search(index=self.settings.OPENSEARCH_INDEX_ARTIFACT, body=body)
        hits = response.get("hits", {}).get("hits", []) or []
        total_value = self._total_hits_value(response)
        results = self._artifact_results_from_hits(hits)
        return results, total_value

    async def fetch_artifacts_by_entity(
        self,
        *,
        tipo: str,
        entity_id: str,
        museum_slug: str,
        museum_id: str | None,
        top_k: int,
        from_offset: int = 0,
        exclude_artifact_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        return await self._to_thread_logged(
            "opensearch.fetch_artifacts_by_entity",
            self._fetch_artifacts_by_entity_sync,
            log_fields={
                "tipo": tipo,
                "museum_slug": museum_slug,
                "museum_id": museum_id,
                "top_k": top_k,
                "from_offset": from_offset,
                "has_exclude": bool(exclude_artifact_id),
            },
            result_size=lambda result: len(result[0]) if result else 0,
            tipo=tipo,
            entity_id=entity_id,
            museum_slug=museum_slug,
            museum_id=museum_id,
            top_k=top_k,
            from_offset=from_offset,
            exclude_artifact_id=exclude_artifact_id,
        )

@lru_cache(maxsize=1)
def get_opensearch_gateway() -> OpenSearchGateway:
    return OpenSearchGateway(get_settings())
