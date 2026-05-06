from __future__ import annotations

import atexit
import asyncio
import base64
from dataclasses import dataclass
from functools import lru_cache
import json
import logging
import os
from pathlib import Path
import subprocess
import threading
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class MultiviewRenderError(RuntimeError):
    """Raised when the persistent multiview worker fails."""


@dataclass(slots=True)
class RenderedModelView:
    name: str
    png_bytes: bytes


class PersistentMultiviewRenderer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._stdout_thread: threading.Thread | None = None
        atexit.register(self.close)

    @property
    def _base_url(self) -> str:
        return f"http://{self.settings.MULTIVIEW_WORKER_HOST}:{self.settings.MULTIVIEW_WORKER_PORT}"

    @property
    def _worker_dir(self) -> Path:
        return (Path(__file__).resolve().parents[2] / "multiview_worker").resolve()

    def _http_json(
        self,
        *,
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        timeout: float,
    ) -> dict[str, Any]:
        body: bytes | None = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(url=url, data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=timeout) as response:  # noqa: S310
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:  # pragma: no cover - runtime-only path
            detail = exc.read().decode("utf-8", errors="ignore")
            raise MultiviewRenderError(
                f"Multiview worker request failed ({exc.code}): {detail or exc.reason}"
            ) from exc
        except URLError as exc:  # pragma: no cover - runtime-only path
            raise MultiviewRenderError(f"Multiview worker unreachable: {exc}") from exc

    def _is_worker_healthy_locked(self) -> bool:
        try:
            payload = self._http_json(
                method="GET",
                url=f"{self._base_url}/health",
                timeout=1.5,
            )
        except MultiviewRenderError:
            return False
        return str(payload.get("status", "")).strip().lower() == "ok"

    def _stream_worker_output(self, process: subprocess.Popen[str]) -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            text = line.rstrip()
            if text:
                logger.info("[multiview-worker] %s", text)

    def _spawn_worker_locked(self) -> None:
        worker_dir = self._worker_dir
        if not worker_dir.exists():
            raise MultiviewRenderError(f"Multiview worker directory not found: {worker_dir}")

        package_json = worker_dir / "package.json"
        if not package_json.exists():
            raise MultiviewRenderError(f"Missing multiview worker package.json: {package_json}")

        command = [
            "node",
            "server.js",
            "--host",
            self.settings.MULTIVIEW_WORKER_HOST,
            "--port",
            str(self.settings.MULTIVIEW_WORKER_PORT),
        ]

        try:
            process = subprocess.Popen(
                command,
                cwd=str(worker_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError as exc:  # pragma: no cover - runtime-only path
            raise MultiviewRenderError(
                "Node.js not found. Install Node.js and run npm install in backend/multiview_worker."
            ) from exc

        self._process = process
        self._stdout_thread = threading.Thread(
            target=self._stream_worker_output,
            args=(process,),
            daemon=True,
            name="multiview-worker-stdout",
        )
        self._stdout_thread.start()

    def _ensure_worker_running_sync(self) -> None:
        with self._lock:
            if self._is_worker_healthy_locked():
                return

            if self._process is None or self._process.poll() is not None:
                self._spawn_worker_locked()

            deadline = time.monotonic() + max(self.settings.MULTIVIEW_WORKER_START_TIMEOUT_SECONDS, 5.0)
            while time.monotonic() < deadline:
                if self._is_worker_healthy_locked():
                    return
                if self._process is not None and self._process.poll() is not None:
                    raise MultiviewRenderError(
                        "Multiview worker exited during startup. Check backend/multiview_worker dependencies."
                    )
                time.sleep(0.5)

        raise MultiviewRenderError("Timed out waiting for multiview worker to become healthy.")

    def render_views_sync(
        self,
        *,
        model_bytes: bytes,
        file_name: str,
        views: int,
        skip_views: int,
        target_view_count: int,
    ) -> list[RenderedModelView]:
        if not model_bytes:
            raise MultiviewRenderError("Cannot render empty model bytes.")

        self._ensure_worker_running_sync()
        extension = Path(file_name).suffix or ".bin"
        temp_fd, temp_path = tempfile.mkstemp(prefix="p360_model_", suffix=extension)
        try:
            with os.fdopen(temp_fd, "wb") as stream:
                stream.write(model_bytes)
                stream.flush()
            payload = {
                "file_name": file_name,
                "file_path": temp_path,
                "views": max(1, views),
                "skip_views": max(0, skip_views),
                "target_view_count": max(target_view_count, views + skip_views),
                "size": max(128, int(self.settings.MULTIVIEW_RENDER_SIZE)),
                "background": self.settings.MULTIVIEW_RENDER_BACKGROUND,
                "fov": int(self.settings.MULTIVIEW_RENDER_FOV),
                "dpr": float(self.settings.MULTIVIEW_RENDER_DPR),
                "strategy": self.settings.MULTIVIEW_RENDER_STRATEGY,
                "oversample": int(self.settings.MULTIVIEW_RENDER_OVERSAMPLE),
                "orbit_margin": float(self.settings.MULTIVIEW_RENDER_ORBIT_MARGIN),
                "ensure_top": bool(self.settings.MULTIVIEW_RENDER_ENSURE_TOP),
                "delay_ms": int(self.settings.MULTIVIEW_RENDER_DELAY_MS),
            }
            response = self._http_json(
                method="POST",
                url=f"{self._base_url}/render",
                payload=payload,
                timeout=max(self.settings.MULTIVIEW_WORKER_START_TIMEOUT_SECONDS, 60.0),
            )
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass
        images = response.get("images")
        if not isinstance(images, list) or not images:
            raise MultiviewRenderError("Multiview worker returned no rendered images.")

        rendered_views: list[RenderedModelView] = []
        for index, entry in enumerate(images):
            if not isinstance(entry, dict):
                continue
            png_base64 = str(entry.get("pngBase64") or "").strip()
            if not png_base64:
                continue
            name = str(entry.get("name") or f"view-{skip_views + index}.png").strip()
            rendered_views.append(
                RenderedModelView(
                    name=name,
                    png_bytes=base64.b64decode(png_base64),
                )
            )

        if not rendered_views:
            raise MultiviewRenderError("Multiview worker returned empty PNG payloads.")

        try:
            self._persist_last_views(source_file_name=file_name, rendered_views=rendered_views)
        except Exception:  # pragma: no cover - debug-only path
            logger.exception("Failed to persist latest multiview renders.")

        return rendered_views

    def _persist_last_views(
        self,
        *,
        source_file_name: str,
        rendered_views: list[RenderedModelView],
    ) -> None:
        if not self.settings.MULTIVIEW_SAVE_LAST_VIEWS:
            return

        target_dir = self.settings.multiview_last_views_dir_resolved
        target_dir.mkdir(parents=True, exist_ok=True)

        for child in target_dir.iterdir():
            if child.is_file():
                try:
                    child.unlink()
                except OSError:
                    logger.warning("Could not remove stale file in %s: %s", target_dir, child.name)

        views_meta: list[dict[str, Any]] = []
        for index, view in enumerate(rendered_views, start=1):
            safe_name = (Path(view.name).name or f"view-{index}.png").strip()
            output_name = f"{index:02d}_{safe_name}"
            output_path = target_dir / output_name
            output_path.write_bytes(view.png_bytes)
            views_meta.append(
                {
                    "input_name": view.name,
                    "saved_as": output_name,
                    "bytes": len(view.png_bytes),
                }
            )

        metadata = {
            "source_file_name": source_file_name,
            "saved_at_epoch": int(time.time()),
            "view_count": len(rendered_views),
            "views": views_meta,
        }
        (target_dir / "_meta.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved latest multiview renders: dir=%s views=%s", target_dir, len(rendered_views))

    async def render_views(
        self,
        *,
        model_bytes: bytes,
        file_name: str,
        views: int,
        skip_views: int,
        target_view_count: int,
    ) -> list[RenderedModelView]:
        return await asyncio.to_thread(
            self.render_views_sync,
            model_bytes=model_bytes,
            file_name=file_name,
            views=views,
            skip_views=skip_views,
            target_view_count=target_view_count,
        )

    async def ensure_worker_running(self) -> None:
        await asyncio.to_thread(self._ensure_worker_running_sync)

    def close(self) -> None:
        with self._lock:
            if self._process is None:
                return
            process = self._process
            self._process = None
            if process.poll() is None:  # pragma: no branch - runtime-only path
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()


@lru_cache(maxsize=1)
def get_multiview_renderer() -> PersistentMultiviewRenderer:
    return PersistentMultiviewRenderer(get_settings())
