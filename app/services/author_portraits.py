"""Retratos dos autores (CHAT_AUTHOR_PORTRAITS).

O catálogo de origem não tem retratos. O ficheiro app/data/author_portraits.json
liga ids de autores do catálogo a imagens da Wikimedia Commons, escolhidas fora
do backend (docs/retratos-dos-autores.md). Aqui só se lê o ficheiro.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def absolute_http_url(value: Any) -> str | None:
    """O endereço, se for absoluto http(s); senão None. O frontend ignora o resto."""
    text = str(value or "").strip()
    return text if text.startswith(("https://", "http://")) and " " not in text else None


class AuthorPortraitStore:
    """Lê o ficheiro dos retratos e volta a lê-lo quando muda no disco."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._mtime: float | None = None
        self._portraits: dict[str, dict[str, Any]] = {}

    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            self._mtime, self._portraits = None, {}
            return self._portraits
        if mtime != self._mtime:
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                portraits = data.get("portraits") if isinstance(data, dict) else None
                self._portraits = portraits if isinstance(portraits, dict) else {}
            except (OSError, ValueError) as exc:
                logger.warning("author_portraits.load_error path=%s error=%s", self.path, exc)
                self._portraits = {}
            self._mtime = mtime
        return self._portraits

    def portrait_for(self, author_ids: list[str]) -> dict[str, str | None] | None:
        """O retrato do primeiro destes ids que tiver um, com os créditos; None se nenhum tiver."""
        portraits = self._load()
        for author_id in author_ids:
            entry = portraits.get(str(author_id or "").strip())
            if not isinstance(entry, dict):
                continue
            image_url = absolute_http_url(entry.get("image_url"))
            if image_url is None:
                continue

            def text(key: str) -> str | None:
                return str(entry.get(key) or "").strip() or None

            return {
                "image_url": image_url,
                "image_credit": text("credit"),
                "image_license": text("license"),
                "image_source_url": absolute_http_url(entry.get("page_url")),
            }
        return None
