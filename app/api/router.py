from fastapi import APIRouter

from app.api.routes import chat, health, logs, speech, tour_routes

api_router = APIRouter()
api_router.include_router(health.router, tags=["system"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
# percursos temáticos (CHAT_ROUTE_PLANNING; 404 com a flag desligada)
api_router.include_router(tour_routes.router, prefix="/chat", tags=["chat"])
api_router.include_router(logs.router, prefix="/logs", tags=["logs"])
api_router.include_router(speech.router, prefix="/speech", tags=["speech"])
