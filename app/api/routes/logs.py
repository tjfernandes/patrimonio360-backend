import logging
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.logging import log_event
from app.services.query_logger import QueryLogger, get_query_logger, utc_timestamp

router = APIRouter()
logger = logging.getLogger(__name__)

FRONTEND_EVENT_TYPES = {
    "assistant_opened",
    "assistant_closed",
    "message_sent",
    "answer_received",
    "artifact_card_opened",
    "artifact_context_selected",
    "artifact_context_cleared",
    "see_in_tour_clicked",
    "navigation_command_sent",
    "navigation_completed",
    "tour_location_changed",
    "artifact_info_opened",
    "artifact_info_closed",
    "tour_window_opened",
    "tour_window_closed",
    "task_started",
    "task_completed",
    "feedback_clicked",
    "error_shown",
}


class FrontendEventPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    event_type: str = Field(..., min_length=1)
    timestamp: str | None = None
    session_id: str | None = None
    conversation_id: str | None = None
    query_id: str | None = None
    participant_id: str | None = None
    task_id: str | None = None
    tour_id: str | None = None
    language: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, value: str) -> str:
        if value not in FRONTEND_EVENT_TYPES:
            raise ValueError(f"unsupported frontend event_type: {value}")
        return value


@router.post("/events")
async def post_frontend_event(
    payload: FrontendEventPayload,
    query_logger: QueryLogger = Depends(get_query_logger),
) -> dict[str, str]:
    event = payload.model_dump(mode="json")
    event["timestamp"] = event.get("timestamp") or utc_timestamp()
    await query_logger.log_frontend_event(event)
    log_event(
        logger,
        logging.INFO,
        "frontend_event.logged",
        event_type=payload.event_type,
        conversation_id=payload.conversation_id,
        query_id=payload.query_id,
        task_id=payload.task_id,
    )
    return {"status": "ok"}
