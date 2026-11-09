"""Endpoints auxiliares dos percursos temáticos (CHAT_ROUTE_PLANNING).

  GET  /api/v1/chat/tour/routes/{route_id}          o percurso da cache (route_id da resposta do chat)
  POST /api/v1/chat/tour/routes/{route_id}/adjust   {action: shorter|only_room|swap_stop, room?, stop_index?}
                                                    -> percurso novo, com route_id novo

Com a flag desligada respondem 404, como se não existissem. O frontend ainda
não os usa (2026-11-09).
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.schemas.chat import RoutePlan
from app.services.chat_service import ChatService, get_chat_service

router = APIRouter()


class RouteAdjustRequest(BaseModel):
    action: Literal["shorter", "only_room", "swap_stop"]
    room: str | None = Field(default=None, description="Sala, para only_room (nome tal como no percurso).")
    stop_index: int | None = Field(default=None, ge=0, description="Índice (0-based) da paragem a trocar, para swap_stop.")
    language: str | None = None


def _enabled() -> None:
    if not bool(getattr(get_settings(), "CHAT_ROUTE_PLANNING", False)):
        raise HTTPException(status_code=404, detail="Not found")


@router.get("/tour/routes/{route_id}", response_model=RoutePlan)
async def get_route(route_id: str) -> RoutePlan:
    _enabled()
    from app.services.route_planning.service import get_stored_route

    stored = get_stored_route(route_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="Percurso desconhecido ou expirado.")
    return RoutePlan.model_validate(stored.plan)


@router.post("/tour/routes/{route_id}/adjust", response_model=RoutePlan)
async def adjust_route(route_id: str, body: RouteAdjustRequest, service: ChatService = Depends(get_chat_service)) -> RoutePlan:
    _enabled()
    from app.services.route_planning.service import adjust_route as _adjust

    if body.action == "only_room" and not (body.room or "").strip():
        raise HTTPException(status_code=422, detail="only_room precisa de room.")
    if body.action == "swap_stop" and body.stop_index is None:
        raise HTTPException(status_code=422, detail="swap_stop precisa de stop_index.")
    plan = await _adjust(service, route_id=route_id, action=body.action, room=body.room, stop_index=body.stop_index, language=body.language)
    if plan is None:
        raise HTTPException(status_code=404, detail="Percurso desconhecido, expirado ou sem paragens suficientes para o ajuste.")
    return RoutePlan.model_validate(plan)
