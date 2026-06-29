
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    
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

    app.include_router(api_router, prefix=settings.API_PREFIX)

    return app


app = create_app()
