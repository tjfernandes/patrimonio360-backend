"""Resposta do chat para a intenção `plan_route`, ajustes e narração das paragens.

Entra por `maybe_answer_route` (chamado por ChatService.handle_message só com
CHAT_ROUTE_PLANNING): intenção explícita em metadata.intent ou por regras
(theme.is_route_request). O fluxo é

  tema (regras + LLM opcional) → candidatos (4 ramos + RRF, candidates.py)
  → planeador determinístico (planner.py) → cartões das paragens (fichas do
  índice de peças, como os artifact_results) → alvo de navegação por paragem
  (hotspot ou objeto, como o «Ver na tour») → RoutePlan na resposta + cache
  (store.py) + registo `route_planned`.

Sem candidatos suficientes: `route: null` e uma resposta a dizer quantas peças
há e a propor alargar ao museu ou deixar cair uma faceta. A narração de cada
paragem reutiliza `_answer_describe_artifact` com o prompt de
route_prompts.build_route_stop_prompt (route_stop_prompt_builder).
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Any, Callable
from uuid import uuid4

from app.core.logging import log_event
from app.prompts.route_prompts import build_route_stop_prompt, build_theme_prompt, route_payload_summary
from app.schemas.chat import ArtifactResult, ChatMessageRequest, ChatMessageResponse, ResponseFormatObject, RoutePlan, TourNavigationTarget
from app.services.chat_i18n import normalize_language
from app.services.llm_service import LLMServiceError
from app.services.panorama_objects import OBJECT_OVERLAY_PREFIX
from app.services.query_logger import utc_timestamp
from app.services.route_planning import planner
from app.services.route_planning.candidates import Candidate, find_candidates
from app.services.route_planning.graph import TourGraph, get_tour_graph_store
from app.services.route_planning.search import RouteStopSearch
from app.services.route_planning.store import StoredRoute, get_route_store
from app.services.route_planning.theme import ThemeFacets, fold, is_route_request, load_vocabulary, merge_llm_facets, rule_facets, theme_title

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Acesso aos serviços do chat (com recurso para os testes)
# --------------------------------------------------------------------------- #


def _settings(chat: Any) -> Any:
    return chat.settings


def _planner_settings(chat: Any) -> planner.PlannerSettings:
    s = _settings(chat)
    return planner.PlannerSettings(
        default_stops=int(getattr(s, "ROUTE_DEFAULT_STOPS", 6)),
        min_stops=int(getattr(s, "ROUTE_MIN_STOPS", 3)),
        max_stops=int(getattr(s, "ROUTE_MAX_STOPS", 10)),
        minutes_per_stop=float(getattr(s, "ROUTE_MINUTES_PER_STOP", 2.5)),
        min_fusion_score=float(getattr(s, "ROUTE_MIN_FUSION_SCORE", 0.012)),
        min_candidates=int(getattr(s, "ROUTE_MIN_CANDIDATES", 3)),
        diversity_weight=float(getattr(s, "ROUTE_DIVERSITY_WEIGHT", 0.35)),
        path_cost_weight=float(getattr(s, "ROUTE_PATH_COST_WEIGHT", 0.02)),
    )


def _route_search(chat: Any) -> RouteStopSearch | None:
    """A pesquisa no índice de paragens; `chat._route_search` sobrepõe-se (testes)."""
    injected = getattr(chat, "_route_search", None)
    if injected is not None:
        return injected
    gateway = getattr(chat, "opensearch_gateway", None)
    ensure = getattr(gateway, "_ensure_client", None)
    if ensure is None:
        return None
    try:
        client = ensure()
    except Exception as exc:  # noqa: BLE001
        log_event(logger, logging.WARNING, "route.search.client_error", error=exc)
        return None
    s = _settings(chat)
    search = RouteStopSearch(client, stops_index=str(getattr(s, "ROUTE_STOPS_INDEX", "cultural_heritage_tour_stops_v1")), images_index=str(s.OPENSEARCH_INDEX_IMAGE))
    chat._route_search = search
    return search


def _graph(chat: Any, museum_slug: str) -> TourGraph | None:
    store = getattr(chat, "_route_graph_store", None)
    return (store if store is not None else get_tour_graph_store()).get(museum_slug)


def _store(chat: Any):
    store = getattr(chat, "_route_store", None)  # uma RouteStore vazia é falsa (__len__): testar por None
    return store if store is not None else get_route_store()


def get_stored_route(route_id: str) -> StoredRoute | None:
    return get_route_store().get(route_id)


def _museum_id(chat: Any, payload: ChatMessageRequest) -> str:
    resolver = getattr(chat, "_resolve_museum_id", None)
    value = None
    if resolver is not None:
        try:
            value = resolver(payload)
        except Exception:  # noqa: BLE001
            value = None
    return str(value or payload.museum_id or payload.museum_slug).strip()


def _start_panorama(metadata: dict[str, Any] | None) -> str | None:
    location = (metadata or {}).get("tour_location")
    if isinstance(location, dict):
        key = str(location.get("panorama_key") or location.get("panoramaKey") or "").strip()
        return key or None
    return None


# --------------------------------------------------------------------------- #
# Entrada
# --------------------------------------------------------------------------- #


async def maybe_answer_route(chat: Any, payload: ChatMessageRequest, *, status_cb=None) -> ChatMessageResponse | None:
    """Resposta de percurso se o pedido o for (metadata.intent ou regras); senão None."""
    metadata = payload.metadata if isinstance(payload.metadata, dict) else {}
    explicit = str(metadata.get("intent") or "").strip().lower() == "plan_route"
    if not explicit and not is_route_request(payload.message or ""):
        return None
    return await answer_route(chat, payload, status_cb=status_cb, reason="metadata_intent" if explicit else "rules")


async def answer_route(chat: Any, payload: ChatMessageRequest, *, status_cb=None, reason: str = "rules") -> ChatMessageResponse:
    started = time.monotonic()
    conversation_id = payload.conversation_id or str(uuid4())
    query_id = str(uuid4())
    metadata = payload.metadata if isinstance(payload.metadata, dict) else {}
    state = chat.session_store.load_or_create(conversation_id=conversation_id, museum_slug=payload.museum_slug)
    language = chat._sync_state_language(state, payload.language) if hasattr(chat, "_sync_state_language") else normalize_language(payload.language)
    chat.session_store.append_turn(state, role="user", text=payload.message)
    await _emit(chat, status_cb, "status.analyzing_request", language)
    museum_id = _museum_id(chat, payload)
    start_panorama = _start_panorama(metadata)
    log_event(logger, logging.INFO, "chat.message.route", conversation_id=conversation_id, query_id=query_id, mode="rag", intent="plan_route", fast_path="plan_route", reason=reason)

    facets = await interpret_theme(chat, payload.message, museum_name=payload.museum_name, language=language)
    await _emit(chat, status_cb, "status.searching_collection", language)
    plan_dict, info = await plan_route(
        chat, museum_slug=payload.museum_slug, museum_id=museum_id, museum_name=payload.museum_name, language=language,
        facets=facets, start_panorama=start_panorama, conversation_id=conversation_id,
    )
    reply = _reply_text(plan_dict, facets, info, language=language, museum_name=payload.museum_name)
    chat.session_store.append_turn(state, role="assistant", text=route_payload_summary(plan_dict) if plan_dict else reply)
    state.intent = "plan_route"
    chat.session_store.save(state)

    artifact_results: list[ArtifactResult] = []
    if plan_dict:
        artifact_results = [RoutePlan.model_validate(plan_dict).stops[i].artifact for i in range(len(plan_dict["stops"]))]
    response = ChatMessageResponse(
        conversation_id=conversation_id,
        query_id=query_id,
        response_format=payload.response_format or ResponseFormatObject(type="text"),
        reply=reply,
        artifact_results=artifact_results,
        navigation_targets=[],
        results_page=1,
        results_page_size=len(artifact_results),
        results_total=len(artifact_results),
        route=RoutePlan.model_validate(plan_dict) if plan_dict else None,
    )
    await _log_planned(chat, payload=payload, conversation_id=conversation_id, query_id=query_id, language=language, facets=facets,
                       plan=plan_dict, info=info, latency_ms=(time.monotonic() - started) * 1000, reason=reason)
    await _emit(chat, status_cb, "status.answer_ready", language)
    return response


async def _emit(chat: Any, status_cb, key: str, language: str | None) -> None:
    emitter = getattr(chat, "_emit_status", None)
    if emitter is None or status_cb is None:
        return
    try:
        await emitter(status_cb, key, language=language)
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# Tema
# --------------------------------------------------------------------------- #


async def interpret_theme(chat: Any, message: str, *, museum_name: str | None, language: str | None) -> ThemeFacets:
    vocabulary = load_vocabulary()
    facets = rule_facets(message, vocabulary=vocabulary)
    if not bool(getattr(_settings(chat), "ROUTE_THEME_LLM", True)):
        return facets
    llm = getattr(chat, "llm_service", None)
    if llm is None:
        return facets
    system_prompt, user_prompt = build_theme_prompt(user_message=message, museum_name=museum_name, vocabulary=vocabulary, language=language)
    try:
        response = await llm.generate(message=user_prompt, response_format=ResponseFormatObject(type="json_object"), system_prompt=system_prompt)
    except (LLMServiceError, Exception) as exc:  # noqa: BLE001 — sem LLM ficam as regras
        log_event(logger, logging.WARNING, "route.theme.llm_error", error=exc)
        return facets
    parsed = getattr(response, "parsed_json", None)
    if parsed is None:
        text = getattr(response, "text", "") or ""
        try:
            import json

            parsed = json.loads(text[text.find("{"): text.rfind("}") + 1]) if "{" in text else None
        except Exception:  # noqa: BLE001
            parsed = None
    return merge_llm_facets(facets, parsed if isinstance(parsed, dict) else None, vocabulary=vocabulary)


# --------------------------------------------------------------------------- #
# Planeamento
# --------------------------------------------------------------------------- #


async def plan_route(
    chat: Any,
    *,
    museum_slug: str,
    museum_id: str,
    museum_name: str | None,
    language: str | None,
    facets: ThemeFacets,
    start_panorama: str | None,
    conversation_id: str | None,
    candidates: list[Candidate] | None = None,
    k_override: int | None = None,
    room: str | None = None,
    exclude_artifacts: set[str] | None = None,
    parent_route_id: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Percurso (dict do RoutePlan) ou None, e o diagnóstico (contagens, latências)."""
    s = _settings(chat)
    info: dict[str, Any] = {"latency_ms": {}}
    theme_text = facets.free_text or facets.theme_text
    if candidates is None:
        search = _route_search(chat)
        if search is None:
            info["error"] = "sem pesquisa de paragens"
            return None, info
        embedding_provider = getattr(chat, "embedding_provider", None)
        embed_text = getattr(embedding_provider, "embed_text", None) if embedding_provider is not None else None
        embed_image_text = getattr(embedding_provider, "embed_multimodal_text_query", None) if embedding_provider is not None else None
        weights = {
            "treasures": 1.0,
            "lexical": float(getattr(s, "ROUTE_WEIGHT_LEXICAL", 1.0)),
            "semantic": float(getattr(s, "ROUTE_WEIGHT_SEMANTIC", 1.0)),
            "image": float(getattr(s, "ROUTE_WEIGHT_IMAGE", 0.7)),
            "attributes": float(getattr(s, "ROUTE_WEIGHT_ATTRIBUTES", 1.0)),
        }
        t0 = time.monotonic()
        candidates, diagnostics = await find_candidates(
            search=search, facets=facets, museum_slug=museum_slug, museum_id=museum_id, theme_text=theme_text,
            embed_text=embed_text, embed_image_text=embed_image_text, weights=weights,
            rrf_k=int(getattr(s, "ROUTE_RRF_K", 60)), branch_size=int(getattr(s, "ROUTE_BRANCH_SIZE", 40)),
            year_tolerance=int(getattr(s, "ROUTE_YEAR_TOLERANCE", 25)), limit=int(getattr(s, "ROUTE_CANDIDATES", 40)), room=room,
        )
        info["latency_ms"]["candidates"] = round((time.monotonic() - t0) * 1000, 1)
        info["branches"] = diagnostics.get("branches")
    pool = [c for c in candidates if not exclude_artifacts or c.artifact_id not in exclude_artifacts]
    if room:
        pool = [c for c in pool if fold(str(c.stop.get("room") or "")) == fold(room)]
    info["candidate_count"] = len(pool)
    graph = _graph(chat, museum_slug)
    settings = _planner_settings(chat)
    t0 = time.monotonic()
    stops, plan_info = planner.plan(
        [c.as_planner_dict() for c in pool], graph=graph, start_panorama=start_panorama, settings=settings,
        stops_requested=k_override or facets.stops_requested, time_budget_min=None if k_override else facets.time_budget_min,
        temporal=facets.is_temporal(), language=language,
    )
    info.update(plan_info)
    info["latency_ms"]["planner"] = round((time.monotonic() - t0) * 1000, 1)
    if stops is None:
        return None, info  # sem percurso não há nada a guardar

    t0 = time.monotonic()
    cards = await _artifact_cards(chat, museum_slug=museum_slug, museum_id=museum_id, stops=stops)
    info["latency_ms"]["cards"] = round((time.monotonic() - t0) * 1000, 1)
    route_id = f"route-{uuid4().hex[:12]}"
    plan_dict = _plan_payload(
        route_id=route_id, museum_slug=museum_slug, museum_name=museum_name, language=language, facets=facets, theme_text=theme_text,
        stops=stops, cards=cards, estimated_minutes=int(plan_info.get("estimated_minutes") or len(stops) * 3),
        candidate_count=len(pool), start_panorama=start_panorama,
    )
    stored = StoredRoute(
        route_id=route_id, museum_slug=museum_slug, museum_id=museum_id, museum_name=museum_name, language=language or "pt",
        conversation_id=conversation_id, theme_text=theme_text, facets=facets.as_dict(), plan=plan_dict,
        candidates=[{k: v for k, v in c.as_planner_dict().items() if k != "stop"} | {"stop": c.stop} for c in candidates],
        start_panorama=start_panorama, parent_route_id=parent_route_id,
    )
    stored.candidates_obj = candidates  # type: ignore[attr-defined]  # para os ajustes, sem reconstruir
    _store(chat).put(stored)
    return plan_dict, info


def navigation_target_for(stop: dict[str, Any]) -> TourNavigationTarget | None:
    """Alvo «Ver na tour» da paragem: o hotspot, ou o objeto (overlay sintético + yaw/pitch)."""
    placement = stop.get("placement") if isinstance(stop.get("placement"), dict) else None
    if placement is None:
        placements = stop.get("placements") or []
        placement = placements[0] if placements and isinstance(placements[0], dict) else None
    panorama_key = str(stop.get("panorama_key") or "").strip()
    inventory = str(stop.get("inventory_number") or "").strip()
    if not placement or not panorama_key or not inventory:
        return None
    kind = str(placement.get("kind") or "hotspot")
    if kind == "hotspot":
        overlay_id = str(placement.get("overlay_id") or "").strip()
        if not overlay_id:
            return None
        return TourNavigationTarget(
            overlay_id=overlay_id, panorama_key=panorama_key, inventory_id=inventory, location=stop.get("room"), title=stop.get("title"),
            source="hotspot", yaw=_round(placement.get("yaw")), pitch=_round(placement.get("pitch")),
        )
    object_id = str(placement.get("object_id") or "").strip()
    if not object_id:
        return None
    return TourNavigationTarget(
        overlay_id=f"{OBJECT_OVERLAY_PREFIX}{object_id}", panorama_key=panorama_key, inventory_id=inventory, location=stop.get("room"),
        title=stop.get("title"), source="object", yaw=_round(placement.get("yaw")), pitch=_round(placement.get("pitch")),
        score=_round(placement.get("placement_score"), 3),
    )


def _round(value: Any, digits: int = 2) -> float | None:
    try:
        return round(float(value), digits) if value is not None else None
    except (TypeError, ValueError):
        return None


def _minimal_card(stop: dict[str, Any], museum_id: str) -> ArtifactResult:
    catalogue = stop.get("catalogue") or {}
    return ArtifactResult(
        artifact_id=str(stop.get("artifact_id") or ""), inventory_number=stop.get("inventory_number"), title=stop.get("title") or catalogue.get("title"),
        museum_id=museum_id, category=catalogue.get("category"), super_category=catalogue.get("super_category"),
        creators=list(catalogue.get("creators") or []), date_or_period=catalogue.get("date_or_period"),
        start_year=catalogue.get("start_year"), end_year=catalogue.get("end_year"), support_or_material=catalogue.get("support_or_material"),
        technique=catalogue.get("technique"), description=catalogue.get("description"), in_tour=True,
        image_count=catalogue.get("image_count"), national_treasure=bool(stop.get("national_treasure")),
    )


async def _artifact_cards(chat: Any, *, museum_slug: str, museum_id: str, stops: list[planner.PlannedStop]) -> dict[str, ArtifactResult]:
    """Cartões iguais aos artifact_results para as peças das paragens (e dos focos)."""
    wanted: list[str] = []
    stop_docs: dict[str, dict[str, Any]] = {}
    for stop in stops:
        for candidate in [stop.candidate, *stop.also_here]:
            artifact_id = str(candidate.get("artifact_id") or "")
            if artifact_id and artifact_id not in stop_docs:
                wanted.append(artifact_id)
                stop_docs[artifact_id] = candidate.get("stop") or {}
    cards: dict[str, ArtifactResult] = {}
    gateway = getattr(chat, "opensearch_gateway", None)
    fetch = getattr(gateway, "fetch_artifacts_by_ids", None)
    build = getattr(chat, "_build_artifact_results", None)
    if fetch is not None and build is not None and wanted:
        try:
            docs = await fetch(museum_slug=museum_slug, museum_id=museum_id, artifact_ids=wanted, top_k=len(wanted))
            results = await build(museum_slug=museum_slug, museum_id=museum_id, artifact_docs=docs, max_images_per_artifact=1)
            attach = getattr(chat, "_attach_national_treasure", None)
            if attach is not None:
                try:
                    attach(results, fallback_museum_id=museum_id)
                except Exception:  # noqa: BLE001
                    pass
            for result in results:
                cards[result.artifact_id] = result
        except Exception as exc:  # noqa: BLE001 — sem fichas ficam os cartões mínimos das paragens
            log_event(logger, logging.WARNING, "route.cards.error", error=exc)
    for artifact_id in wanted:
        if artifact_id not in cards:
            cards[artifact_id] = _minimal_card(stop_docs[artifact_id], museum_id)
        else:
            cards[artifact_id].in_tour = True
    return cards


def _plan_payload(
    *, route_id: str, museum_slug: str, museum_name: str | None, language: str | None, facets: ThemeFacets, theme_text: str,
    stops: list[planner.PlannedStop], cards: dict[str, ArtifactResult], estimated_minutes: int, candidate_count: int, start_panorama: str | None,
) -> dict[str, Any]:
    en = normalize_language(language) == "en"
    payload_stops: list[dict[str, Any]] = []
    for index, stop in enumerate(stops, start=1):
        candidate = stop.candidate
        stop_doc = candidate.get("stop") or {}
        target = navigation_target_for(stop_doc)
        card = cards.get(str(candidate.get("artifact_id") or ""))
        if target is None or card is None:
            continue
        also_here = []
        for other in stop.also_here:
            other_doc = other.get("stop") or {}
            also_here.append({
                "artifact_id": other.get("artifact_id"), "inventory_number": other_doc.get("inventory_number"),
                "title": other_doc.get("title"), "navigation_target": (nt.model_dump() if (nt := navigation_target_for(other_doc)) else None),
                "why": list(other.get("why") or []),
            })
        payload_stops.append({
            "order": index,
            "stop_id": str(candidate.get("stop_id") or stop_doc.get("stop_id") or ""),
            "artifact": card.model_dump(),
            "navigation_target": target.model_dump(),
            "room": candidate.get("room"),
            "why": list(candidate.get("why") or []),
            "look_for": list(candidate.get("look_for") or [])[:3],
            "transition": stop.transition,
            "also_here": also_here,
            "score": round(float(candidate.get("fusion_score") or 0.0), 4),
        })
    title = theme_title(facets, language)
    n = len(payload_stops)
    if en:
        intro = f"{n} stops, about {estimated_minutes} min: {theme_text} across the tour of {museum_name or 'the museum'}."
    else:
        intro = f"{n} paragens, cerca de {estimated_minutes} min: {theme_text} pela visita {('ao ' if museum_name else 'ao museu') + (museum_name or '')}".rstrip() + "."
    return {
        "route_id": route_id,
        "title": title,
        "intro": intro,
        "theme": {"text": theme_text, "facets": facets.labels()},
        "estimated_minutes": estimated_minutes,
        "stops": payload_stops,
        "museum_slug": museum_slug,
        "candidate_count": candidate_count,
        "start_panorama_key": start_panorama,
    }


def _reply_text(plan: dict[str, Any] | None, facets: ThemeFacets, info: dict[str, Any], *, language: str | None, museum_name: str | None) -> str:
    en = normalize_language(language) == "en"
    theme = plan["theme"]["text"] if plan else (facets.free_text or facets.theme_text)
    if plan and plan["stops"]:
        first = plan["stops"][0]
        room = first.get("room")
        n = len(plan["stops"])
        if en:
            where = f", starting in {room}" if room else ""
            return (f"I have planned a route with {n} stops about \"{theme}\"{where}. Each stop has a \"View in tour\" button; "
                    f"when you get there, I will tell you about the piece.")
        where = f", a começar {('pela ' if room.lower().startswith(('sala', 'capela', 'exposicao', 'exposição')) else 'por ') + room}" if room else ""
        return (f"Preparei um percurso com {n} paragens sobre «{theme}»{where}. Cada paragem tem o botão «Ver na tour»; "
                f"quando lá chegar, conto-lhe mais sobre a peça.")
    count = int(info.get("relevant_count") or 0)
    labels = [f["label"] for f in facets.labels() if f["kind"] not in {"theme"}]
    drop = labels[-1] if len(labels) > 1 else None
    if en:
        if count == 0:
            base = f"I could not find pieces in the tour that match \"{theme}\"."
        else:
            base = f"I only found {count} piece{'s' if count != 1 else ''} in the tour matching \"{theme}\" — too few for a route."
        options = f" I can widen the search to the whole museum{(' or drop «' + drop + '»') if drop else ''}: which do you prefer?"
        return base + options
    if count == 0:
        base = f"Não encontrei peças na visita que encaixem em «{theme}»."
    else:
        base = f"Só encontrei {count} peça{'s' if count != 1 else ''} na visita que encaix{'em' if count != 1 else 'e'} em «{theme}» — poucas para um percurso."
    options = f" Posso alargar a pesquisa a todo o museu{(' ou deixar cair «' + drop + '»') if drop else ''}: o que prefere?"
    return base + options


# --------------------------------------------------------------------------- #
# Ajustes
# --------------------------------------------------------------------------- #


async def adjust_route(chat: Any, *, route_id: str, action: str, room: str | None = None, stop_index: int | None = None, language: str | None = None) -> dict[str, Any] | None:
    stored = _store(chat).get(route_id)
    if stored is None or not stored.plan:
        return None
    candidates = getattr(stored, "candidates_obj", None)
    if candidates is None:
        candidates = [Candidate(artifact_id=c["artifact_id"], stop=c["stop"], fusion_score=float(c["fusion_score"]), sources=c.get("sources") or {},
                                why=list(c.get("why") or []), look_for=list(c.get("look_for") or [])) for c in stored.candidates]
    facets = ThemeFacets(**{k: (tuple(v) if k == "year_range" and v else v) for k, v in stored.facets.items()})
    n = len(stored.plan.get("stops") or [])
    k_override: int | None = None
    exclude: set[str] = set()
    only_room = None
    if action == "shorter":
        k_override = max(int(getattr(_settings(chat), "ROUTE_MIN_STOPS", 3)), int(math.ceil(n * 0.6)))
        if k_override >= n:
            k_override = max(int(getattr(_settings(chat), "ROUTE_MIN_STOPS", 3)), n - 1)
    elif action == "only_room":
        only_room = (room or "").strip() or None
        k_override = n
    elif action == "swap_stop":
        if stop_index is None or not 0 <= stop_index < n:
            return None
        stop = stored.plan["stops"][stop_index]
        exclude.add(str((stop.get("artifact") or {}).get("artifact_id") or ""))
        for focus in stop.get("also_here") or []:
            exclude.add(str(focus.get("artifact_id") or ""))
        k_override = n
    else:
        return None
    plan_dict, _info = await plan_route(
        chat, museum_slug=stored.museum_slug, museum_id=stored.museum_id or stored.museum_slug, museum_name=stored.museum_name,
        language=language or stored.language, facets=facets, start_panorama=stored.start_panorama, conversation_id=stored.conversation_id,
        candidates=candidates, k_override=k_override, room=only_room, exclude_artifacts=exclude or None, parent_route_id=route_id,
    )
    if plan_dict:
        await _log_event(chat, {
            "event_type": "route_adjusted", "timestamp": utc_timestamp(), "parent_route_id": route_id, "route_id": plan_dict["route_id"],
            "action": action, "room": only_room, "stop_index": stop_index, "stops": len(plan_dict["stops"]),
        })
    return plan_dict


# --------------------------------------------------------------------------- #
# Narração
# --------------------------------------------------------------------------- #


def route_stop_prompt_builder(chat: Any, *, metadata: dict[str, Any] | None, selected: dict[str, Any] | None, language: str | None, museum_name: str | None) -> Callable[[str], str] | None:
    """Construtor do prompt da paragem para `_answer_describe_artifact`, se o pedido vier com metadata.route_stop."""
    route_stop = (metadata or {}).get("route_stop")
    if not isinstance(route_stop, dict):
        return None
    stored = _store(chat).get(str(route_stop.get("route_id") or ""))
    if stored is None or not stored.plan:
        return None
    stops = stored.plan.get("stops") or []
    index = route_stop.get("stop_index")
    stop = None
    if isinstance(index, int) and 0 <= index < len(stops):
        stop = stops[index]
    if stop is None and route_stop.get("stop_id"):
        stop = next((s for s in stops if s.get("stop_id") == route_stop.get("stop_id")), None)
        index = stops.index(stop) if stop else None
    if stop is None:
        return None
    # a peça narrada pode ser um foco secundário da paragem: look_for só se for a principal
    selected_id = str((selected or {}).get("artifact_id") or "")
    main_id = str((stop.get("artifact") or {}).get("artifact_id") or "")
    look_for = list(stop.get("look_for") or []) if not selected_id or selected_id == main_id else []
    next_transition = stops[index + 1].get("transition") if isinstance(index, int) and index + 1 < len(stops) else None

    def build(card_context: str) -> str:
        return build_route_stop_prompt(
            museum_name=museum_name or stored.museum_name, theme_text=stored.theme_text, position=(index or 0) + 1, total=len(stops),
            card_context=card_context, look_for=look_for, next_transition=next_transition, language=language or stored.language,
        )

    return build


async def log_route_stop_narrated(chat: Any, *, metadata: dict[str, Any] | None, conversation_id: str | None) -> None:
    route_stop = (metadata or {}).get("route_stop")
    if not isinstance(route_stop, dict):
        return
    await _log_event(chat, {
        "event_type": "route_stop_narrated", "timestamp": utc_timestamp(), "conversation_id": conversation_id,
        "route_id": route_stop.get("route_id"), "stop_index": route_stop.get("stop_index"), "stop_id": route_stop.get("stop_id"),
        "session_id": (metadata or {}).get("session_id"),
    })


# --------------------------------------------------------------------------- #
# Registos
# --------------------------------------------------------------------------- #


async def _log_event(chat: Any, event: dict[str, Any]) -> None:
    query_logger = getattr(chat, "query_logger", None)
    log = getattr(query_logger, "log_event", None)
    if log is None:
        return
    try:
        await log(event)
    except Exception:  # noqa: BLE001
        pass


async def _log_planned(chat: Any, *, payload: ChatMessageRequest, conversation_id: str, query_id: str, language: str | None, facets: ThemeFacets,
                       plan: dict[str, Any] | None, info: dict[str, Any], latency_ms: float, reason: str) -> None:
    metadata = payload.metadata if isinstance(payload.metadata, dict) else {}
    await _log_event(chat, {
        "event_type": "route_planned",
        "timestamp": utc_timestamp(),
        "session_id": metadata.get("session_id"),
        "conversation_id": conversation_id,
        "query_id": query_id,
        "museum_slug": payload.museum_slug,
        "language": language,
        "raw_query": payload.message,
        "reason": reason,
        "route_id": plan["route_id"] if plan else None,
        "theme": facets.free_text or facets.theme_text,
        "facets": facets.as_dict(),
        "k": info.get("k"),
        "candidate_count": info.get("candidate_count"),
        "relevant_count": info.get("relevant_count"),
        "branches": info.get("branches"),
        "chosen_stop_ids": [s["stop_id"] for s in (plan or {}).get("stops") or []],
        "fusion_scores": [s.get("score") for s in (plan or {}).get("stops") or []],
        "start_panorama_key": (plan or {}).get("start_panorama_key") or _start_panorama(metadata),
        "estimated_minutes": (plan or {}).get("estimated_minutes"),
        "latency_ms": {**(info.get("latency_ms") or {}), "total": round(latency_ms, 1)},
        "status": "ok" if plan else "no_route",
    })
    emit = getattr(chat, "_emit_query_log", None)
    if emit is None:
        return
    try:
        await emit(
            payload=payload, conversation_id=conversation_id, query_id=query_id, language=language, raw_query=payload.message,
            resolved_query=facets.free_text or facets.theme_text, lexical_query=None, embedding_query=None, route="plan_route",
            retrieval_mode="route", filters_applied={"facets": facets.as_dict()}, boosts_applied=None,
            retrieved_artifacts=[{"rank": i + 1, "artifact_id": s["artifact"]["artifact_id"], "inventory_number": s["artifact"].get("inventory_number"),
                                  "title": s["artifact"].get("title"), "score": s.get("score")} for i, s in enumerate((plan or {}).get("stops") or [])],
            shown_artifacts=[{"artifact_id": s["artifact"]["artifact_id"], "inventory_number": s["artifact"].get("inventory_number"), "stop_id": s["stop_id"]}
                             for s in (plan or {}).get("stops") or []],
            navigation_targets=[{"inventory_number": s["navigation_target"]["inventory_id"], "has_navigation_target": True,
                                 "panorama_id": s["navigation_target"]["panorama_key"], "overlay_id": s["navigation_target"]["overlay_id"],
                                 "source": s["navigation_target"].get("source")} for s in (plan or {}).get("stops") or []],
            latency_retrieval_ms=(info.get("latency_ms") or {}).get("candidates"), latency_llm_ms=None, latency_total_ms=round(latency_ms, 1),
            status="ok", error=None, extra={"fast_path": "plan_route", "route_id": (plan or {}).get("route_id"), "route_status": "ok" if plan else "no_route"},
        )
    except Exception as exc:  # noqa: BLE001
        log_event(logger, logging.WARNING, "route.log.error", error=exc)


__all__ = [
    "adjust_route", "answer_route", "get_stored_route", "interpret_theme", "log_route_stop_narrated", "maybe_answer_route",
    "navigation_target_for", "plan_route", "route_stop_prompt_builder",
]
