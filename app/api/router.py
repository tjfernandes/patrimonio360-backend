from fastapi import APIRouter

from app.api.routes import chat, health, logs

api_router = APIRouter()
api_router.include_router(health.router, tags=["system"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(logs.router, prefix="/logs", tags=["logs"])
