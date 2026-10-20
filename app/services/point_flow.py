"""O «Apontar e perguntar» sobre o ChatService (intenção point_and_ask):
vista → recorte → pesquisa → decisão → describe_artifact.

Os eventos SSE `point` (captured, segmented, candidates, decided) saem pelo
mesmo canal dos status e dos tokens.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Awaitable, Callable
from uuid import uuid4

from app.core.logging import log_event
from app.schemas.chat import ChatMessageRequest, ChatMessageResponse, ResponseFormatObject
from app.services.chat_i18n import translate
from app.services.point_ask import Candidate, PointOutcome, PointRequest, compute_ray, get_point_ask_service

logger = logging.getLogger(__name__)

StatusCallback = Callable[[dict[str, Any]], Awaitable[None]]


def _point_service(service: Any):
    return getattr(service, "point_ask_service", None) or get_point_ask_service()


async def _emit_point(status_cb: StatusCallback | None, payload: dict[str, Any]) -> None:
    if status_cb is None:
        return
    try:
        await status_cb({"type": "point", "payload": payload})
    except Exception:  # noqa: BLE001 — o canal nunca trava o pedido
        pass


def _first_token_recorder(status_cb: StatusCallback | None, timings: dict[str, float], started: float) -> StatusCallback | None:
    """Envolve o canal para registar quando sai o primeiro token do texto."""
    if status_cb is None:
        return None
    seen = False

    async def wrapped(event: dict[str, Any]) -> None:
        nonlocal seen
        if not seen and event.get("type") == "token":
            seen = True
            timings["llm_first_token"] = (time.monotonic() - started) * 1000
        await status_cb(event)

    return wrapped


async def answer_point_and_ask(
    service: Any,
    payload: ChatMessageRequest,
    request: PointRequest,
    *,
    status_cb: StatusCallback | None = None,
) -> ChatMessageResponse:
    point = _point_service(service)
    settings = service.settings
    threshold = float(getattr(settings, "POINT_MATCH_THRESHOLD", 0.8))
    conversation_id = payload.conversation_id or str(uuid4())
    query_id = str(uuid4())
    started = time.monotonic()
    state = service.session_store.load_or_create(conversation_id=conversation_id, museum_slug=payload.museum_slug)
    language = service._sync_state_language(state, payload.language)
    museum_slug = payload.museum_slug
    museum_id = service._resolve_museum_id(payload)
    nav = service.tour_navigation_service
    outcome = PointOutcome(request=request)
    log_event(
        logger, logging.INFO, "chat.point.start", conversation_id=conversation_id, query_id=query_id,
        museum_slug=museum_slug, panorama_key=request.panorama_key, request_id=request.request_id,
        hotspot_overlay_id=request.hotspot_overlay_id,
    )

    # 1. a vista
    await service._emit_status(status_cb, "status.point_segmenting", language=language)
    try:
        outcome.ray = compute_ray(request)
        outcome.view = await point.render(request, museum_slug=museum_slug)
        outcome.timings_ms["projection"] = outcome.view.elapsed_ms
        # 2. o recorte
        await point.cut(outcome, museum_slug=museum_slug)
    except Exception as exc:
        outcome.error = f"{type(exc).__name__}: {exc}"
        log_event(logger, logging.ERROR, "chat.point.view.error", conversation_id=conversation_id, query_id=query_id, error=exc)
        outcome.timings_ms["decided"] = (time.monotonic() - started) * 1000
        await _emit_point(status_cb, outcome.decided_payload())
        return await _finish_without_piece(
            service, payload, state, outcome, status_cb=status_cb, conversation_id=conversation_id, query_id=query_id,
            language=language, reply=translate("message.point_unavailable", language), threshold=threshold, started=started,
            candidate_results=([], []),
        )
    await _emit_point(status_cb, {"stage": "captured", "request_id": request.request_id, "crop_url": outcome.crop_url})
    await _emit_point(status_cb, {
        "stage": "segmented", "request_id": request.request_id, "cutout_url": outcome.cutout_url,
        "mask": [list(p) for p in outcome.mask_polygon],
    })

    # 3. a pesquisa (ou o hotspot)
    await service._emit_status(status_cb, "status.point_searching", language=language)
    search_started = time.monotonic()
    docs_by_id: dict[str, dict[str, Any]] = {}
    hotspot_inventories = (
        nav.overlay_inventories(museum_slug=museum_slug, museum_id=museum_id, overlay_id=request.hotspot_overlay_id)
        if request.hotspot_overlay_id else []
    )
    if hotspot_inventories:
        try:
            docs = await service.opensearch_gateway.fetch_artifacts_by_inventory_numbers(
                museum_slug=museum_slug, museum_id=museum_id, inventory_numbers=hotspot_inventories, top_k=len(hotspot_inventories),
            )
        except Exception as exc:
            log_event(logger, logging.WARNING, "chat.point.hotspot_fetch.error", error=exc)
            docs = []
        wanted = {nav.normalize_inventory_key(i) for i in hotspot_inventories}
        docs = [d for d in docs if nav.normalize_inventory_key(str(d.get("inventory_number") or "")) in wanted]
        for doc in docs:
            docs_by_id[str(doc.get("artifact_id"))] = doc
            outcome.candidates.append(Candidate(
                artifact_id=str(doc.get("artifact_id")), inventory_number=str(doc.get("inventory_number") or "") or None,
                title=str(doc.get("title") or "") or None, score=0.99, image_name=None, in_tour=True, in_panorama=True, raw_score=None,
            ))
        outcome.hotspot_match = bool(docs)
    if not outcome.candidates:
        try:
            embedding = await service.embedding_provider.embed_multimodal_image_bytes(image_bytes=outcome.crop_jpeg or b"", text=None)
            outcome.timings_ms["embedding"] = (time.monotonic() - search_started) * 1000
            knn_started = time.monotonic()
            page = await service.opensearch_gateway.search_similar_images_page(
                museum_slug=museum_slug, museum_id=museum_id, image_embedding=embedding,
                from_offset=0, page_size=max(int(settings.POINT_SEARCH_K), 3), retrieval_window_size=None,
            )
            outcome.timings_ms["knn"] = (time.monotonic() - knn_started) * 1000
            outcome.candidates = point.rank(
                list(page.results),
                in_tour_keys=nav.tour_inventory_keys(museum_slug=museum_slug, museum_id=museum_id),
                panorama_keys=nav.panorama_inventory_keys(museum_slug=museum_slug, museum_id=museum_id, panorama_key=request.panorama_key),
                normalize_key=nav.normalize_inventory_key,
            )
        except Exception as exc:
            outcome.error = f"{type(exc).__name__}: {exc}"
            log_event(logger, logging.ERROR, "chat.point.search.error", conversation_id=conversation_id, query_id=query_id, error=exc)
    outcome.timings_ms["search"] = (time.monotonic() - search_started) * 1000
    await _fill_image_names(service, outcome, docs_by_id, museum_slug=museum_slug, museum_id=museum_id)
    await _emit_point(status_cb, {"stage": "candidates", "request_id": request.request_id, "candidates": [c.payload() for c in outcome.candidates]})

    # 4. a decisão: acima do limiar, com certeza; entre o chão e o limiar,
    # apresenta-se a mesma peça mas sem certeza («parece ser…»); abaixo, nada
    floor = float(getattr(settings, "POINT_LOW_CONFIDENCE_FLOOR", 0.45))
    if outcome.candidates and outcome.candidates[0].score >= threshold:
        outcome.chosen, outcome.confidence = outcome.candidates[0], "high"
    elif outcome.candidates and outcome.candidates[0].score >= min(floor, threshold):
        outcome.chosen, outcome.confidence = outcome.candidates[0], "low"
    outcome.timings_ms["decided"] = (time.monotonic() - started) * 1000
    await _emit_point(status_cb, outcome.decided_payload())
    log_event(
        logger, logging.INFO, "chat.point.decided", conversation_id=conversation_id, query_id=query_id,
        chosen=outcome.chosen.artifact_id if outcome.chosen else None,
        score=round(outcome.candidates[0].score, 3) if outcome.candidates else None, hotspot_match=outcome.hotspot_match,
        candidates=len(outcome.candidates), elapsed_ms=round(outcome.timings_ms["decided"], 1),
    )
    candidate_results = await _candidate_results(service, outcome, docs_by_id, museum_slug=museum_slug, museum_id=museum_id)
    point_log = outcome.log_payload(threshold)

    if outcome.chosen is None:
        return await _finish_without_piece(
            service, payload, state, outcome, status_cb=status_cb, conversation_id=conversation_id, query_id=query_id,
            language=language, reply=translate("message.point_not_found", language), threshold=threshold, started=started,
            candidate_results=candidate_results, point_log=point_log,
        )

    # 5. a peça escolhida segue o fluxo «Ver na tour»
    chosen = outcome.chosen
    describe_payload = payload.model_copy(update={
        "conversation_id": conversation_id,
        "metadata": {**(payload.metadata or {}), "intent": "describe_artifact", "selected_artifact": {
            "artifact_id": chosen.artifact_id, "inventory_number": chosen.inventory_number, "title": chosen.title, "source": "point_and_ask",
        }},
    })
    decided_at = time.monotonic()
    response = await service._answer_describe_artifact(
        describe_payload,
        selected={"artifact_id": chosen.artifact_id, "inventory_number": chosen.inventory_number, "title": chosen.title},
        status_cb=_first_token_recorder(status_cb, point_log["timings_ms"], decided_at),
        fast_path="point_and_ask",
        log_extra={"point": point_log},
        prompt_subject=" ".join(p for p in (chosen.title, f"({chosen.inventory_number})" if chosen.inventory_number else None) if p) or None,
        uncertain_percent=int(round(chosen.score * 100)) if outcome.confidence == "low" else None,
    )
    artifact_results, navigation_targets = candidate_results
    if artifact_results:
        response.artifact_results = artifact_results
        response.navigation_targets = navigation_targets
        response.results_page_size = len(artifact_results)
        response.results_total = len(artifact_results)
    response.point = outcome.decided_payload()
    return response


async def _fill_image_names(service: Any, outcome: PointOutcome, docs_by_id: dict[str, dict[str, Any]], *, museum_slug: str, museum_id: str | None) -> None:
    """image_name = a primeira imagem da peça (o frontend faz a miniatura com /api/v1/chat/images/…)."""
    missing = [c.artifact_id for c in outcome.candidates if not c.image_name]
    if not missing:
        return
    try:
        hits = await service.opensearch_gateway.fetch_images_by_artifact_ids(
            museum_slug=museum_slug, museum_id=museum_id, artifact_ids=missing, per_artifact=1, max_total=len(missing),
        )
    except Exception as exc:
        log_event(logger, logging.WARNING, "chat.point.image_fetch.error", error=exc)
        return
    by_artifact: dict[str, str] = {}
    for hit in hits or []:
        artifact_id = str(hit.get("artifact_id") or "").strip()
        name = str(hit.get("local_path") or hit.get("original_image_name") or "").replace("\\", "/").strip("/")
        if artifact_id and name and artifact_id not in by_artifact:
            by_artifact[artifact_id] = name
    for candidate in outcome.candidates:
        if not candidate.image_name:
            candidate.image_name = by_artifact.get(candidate.artifact_id)


async def _candidate_results(service: Any, outcome: PointOutcome, docs_by_id: dict[str, dict[str, Any]], *, museum_slug: str, museum_id: str | None):
    """Cartões dos candidatos (o escolhido primeiro) e os alvos de navegação dos que têm hotspot."""
    if not outcome.candidates:
        return [], []
    wanted = [c.artifact_id for c in outcome.candidates if c.artifact_id not in docs_by_id]
    if wanted:
        try:
            fetched = await service.opensearch_gateway.fetch_artifacts_by_ids(
                museum_slug=museum_slug, museum_id=museum_id, artifact_ids=wanted, top_k=len(wanted),
            )
        except Exception as exc:
            log_event(logger, logging.WARNING, "chat.point.candidates_fetch.error", error=exc)
            fetched = []
        for doc in fetched or []:
            docs_by_id[str(doc.get("artifact_id"))] = doc
    docs = [docs_by_id[c.artifact_id] for c in outcome.candidates if c.artifact_id in docs_by_id]
    if not docs:
        return [], []
    artifact_results = await service._build_artifact_results(museum_slug=museum_slug, museum_id=museum_id, artifact_docs=docs)
    navigation_targets = service._resolve_navigation_targets(museum_slug=museum_slug, museum_id=museum_id, docs=docs)
    return artifact_results, navigation_targets


async def _finish_without_piece(
    service: Any, payload: ChatMessageRequest, state: Any, outcome: PointOutcome, *, status_cb: StatusCallback | None,
    conversation_id: str, query_id: str, language: str, reply: str, threshold: float, started: float,
    candidate_results, point_log: dict[str, Any] | None = None,
) -> ChatMessageResponse:
    artifact_results, navigation_targets = candidate_results
    service.session_store.append_turn(state, role="user", text=payload.message)
    service.session_store.append_turn(state, role="assistant", text=reply)
    service.session_store.save(state)
    await service._emit_status(status_cb, "status.answer_ready", language=language)
    point_log = point_log or outcome.log_payload(threshold)
    await service._emit_query_log(
        payload=payload, conversation_id=conversation_id, query_id=query_id, language=language,
        raw_query=(payload.message or "").strip(), resolved_query=(payload.message or "").strip(),
        lexical_query=None, embedding_query=None, route="rag", retrieval_mode="point_and_ask",
        filters_applied={"filters": {}, "sort": {}, "target_museum": None},
        boosts_applied={"point_in_tour_boost": service.settings.POINT_IN_TOUR_BOOST, "point_panorama_boost": service.settings.POINT_PANORAMA_BOOST},
        retrieved_artifacts=[{"artifact_id": c.artifact_id, "inventory_number": c.inventory_number, "title": c.title, "score": c.score} for c in outcome.candidates],
        shown_artifacts=service._log_shown_artifact_entries(artifact_results),
        navigation_targets=service._log_navigation_target_entries(navigation_targets=navigation_targets, artifact_results=artifact_results),
        latency_retrieval_ms=outcome.timings_ms.get("search"), latency_llm_ms=None,
        latency_total_ms=(time.monotonic() - started) * 1000,
        status="error" if outcome.error and not outcome.candidates else "ok", error=outcome.error,
        opensearch_query=None, extra={"fast_path": "point_and_ask", "point": point_log},
    )
    return ChatMessageResponse(
        conversation_id=conversation_id, query_id=query_id, response_format=payload.response_format or ResponseFormatObject(type="text"),
        reply=reply, model_hint="deterministic:point_and_ask", artifact_results=artifact_results, navigation_targets=navigation_targets,
        results_page=1, results_page_size=len(artifact_results), results_total=len(artifact_results), results_has_more=False,
        point=outcome.decided_payload(),
    )
