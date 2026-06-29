import logging
import time
from uuid import uuid4

from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import get_settings
from app.core.logging import (
    configure_backend_logging,
    log_event,
    reset_request_id,
    set_request_id,
)

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_backend_logging(settings)
    log_event(
        logger,
        logging.INFO,
        "app.configure",
        env=settings.APP_ENV,
        api_prefix=settings.API_PREFIX,
        log_level=settings.BACKEND_LOG_LEVEL,
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

    @app.on_event("startup")
    async def startup_logging_and_prewarm() -> None:
        log_event(
            logger,
            logging.INFO,
            "app.startup",
            env=settings.APP_ENV,
            prewarm=settings.CHAT_PREWARM_ON_STARTUP,
        )
        if not settings.CHAT_PREWARM_ON_STARTUP:
            return
        from app.services.warmup import warmup_chat_stack

        await warmup_chat_stack(
            include_multimodal=settings.CHAT_PREWARM_INCLUDE_MULTIMODAL,
            include_multiview_worker=settings.CHAT_PREWARM_INCLUDE_MULTIVIEW_WORKER,
        )

    @app.middleware("http")
    async def operational_request_logging(request: Request, call_next):
        if not settings.BACKEND_ACCESS_LOG_ENABLED:
            return await call_next(request)

        path = request.url.path
        is_healthcheck = path in {
            "/health",
            f"{settings.API_PREFIX}/health",
            f"{settings.API_PREFIX}/chat/health",
        }
        if is_healthcheck and not settings.BACKEND_LOG_HEALTHCHECKS:
            return await call_next(request)

        request_id = request.headers.get("x-request-id") or uuid4().hex[:12]
        token = set_request_id(request_id)
        started_at = time.perf_counter()
        client_host = request.client.host if request.client else None
        log_event(
            logger,
            logging.INFO,
            "http.request.start",
            method=request.method,
            path=path,
            client=client_host,
        )
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - started_at) * 1000
            logger.exception(
                "http.request.error method=%s path=%s duration_ms=%.1f",
                request.method,
                path,
                duration_ms,
            )
            raise
        finally:
            if "response" not in locals():
                reset_request_id(token)

        duration_ms = (time.perf_counter() - started_at) * 1000
        response.headers["X-Request-ID"] = request_id
        log_event(
            logger,
            logging.INFO,
            "http.request.finish",
            method=request.method,
            path=path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 1),
        )
        reset_request_id(token)
        return response

    @app.get("/health", tags=["system"])
    async def root_health() -> dict[str, str]:
        return {"status": "ok", "mode": settings.APP_ENV}

    app.include_router(api_router, prefix=settings.API_PREFIX)

    return app


app = create_app()
