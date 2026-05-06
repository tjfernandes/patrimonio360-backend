import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import get_settings
from app.services.warmup import warmup_chat_stack


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    
    app = FastAPI(
        title="Patrimonio360 Embed API",
        version="1.0.0",
        description="API for chat integration.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins_list,
        allow_methods=settings.cors_allow_methods_list,
        allow_headers=settings.cors_allow_headers_list,
        allow_credentials=False,
    )

    @app.get("/health", tags=["system"])
    async def root_health() -> dict[str, str]:
        return {"status": "ok", "mode": settings.APP_ENV}

    @app.on_event("startup")
    async def startup_prewarm() -> None:
        if not settings.CHAT_PREWARM_ON_STARTUP:
            return
        try:
            await warmup_chat_stack(
                include_multimodal=settings.CHAT_PREWARM_INCLUDE_MULTIMODAL,
                include_reranker=settings.CHAT_PREWARM_INCLUDE_RERANKER,
                include_multiview_worker=settings.CHAT_PREWARM_INCLUDE_MULTIVIEW_WORKER,
            )
        except Exception:
            logging.getLogger(__name__).exception("chat stack prewarm failed during startup")

    app.include_router(api_router, prefix=settings.API_PREFIX)
    return app


app = create_app()
