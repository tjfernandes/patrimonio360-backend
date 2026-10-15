from fastapi import APIRouter, WebSocket

from app.core.config import get_settings
from app.services.query_logger import get_query_logger
from app.services.speech_dictation import get_speech_engine, origin_allowed, run_dictation_session

router = APIRouter()


@router.websocket("/stream")
async def speech_stream(websocket: WebSocket) -> None:
    """Ditado por voz (protocolo em app/services/speech_dictation.py)."""
    settings = get_settings()
    if not origin_allowed(websocket.headers.get("origin"), settings.cors_allow_origins_list):
        await websocket.close(code=1008)  # antes do accept: o handshake leva 403
        return
    engine = get_speech_engine()
    if not engine.ready:
        # sem ditado no servidor: recusar a ligação (sem «ready») faz o frontend
        # passar ao reconhecimento do browser
        await websocket.close(code=1013)
        return
    await websocket.accept()
    await run_dictation_session(websocket, engine, settings, get_query_logger())
