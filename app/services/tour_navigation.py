from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import re
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings


_TOUR_ROOMS_PATH = Path(__file__).resolve().parents[1] / "data" / "tour_rooms.json"


@lru_cache(maxsize=1)
def _load_room_descriptions() -> dict[str, dict[str, dict[str, str]]]:
    """Descrições curadas por museu -> zona (dobrada) -> {description, source}."""
    try:
        payload = json.loads(_TOUR_ROOMS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    loaded: dict[str, dict[str, dict[str, str]]] = {}
    for museum, rooms in (payload.items() if isinstance(payload, dict) else []):
        if not isinstance(rooms, dict) or str(museum).startswith("_"):
            continue
        entries: dict[str, dict[str, str]] = {}
        for room, meta in rooms.items():
            if not isinstance(meta, dict):
                continue
            description = str(meta.get("description") or "").strip()
            if description:
                entries[TourNavigationService._fold_room(room)] = {
                    "description": description,
                    "source": str(meta.get("source") or "").strip(),
                }
        loaded[str(museum).strip().lower()] = entries
    return loaded



@dataclass(frozen=True)
class _TourNavigationEntry:
    overlay_id: str
    panorama_key: str
    inventory_id: str
    location: str | None = None
    title: str | None = None


class TourNavigationService:
    """Resolves tour navigation targets from inventory ids using poi_tours JSON files."""

    _INVENTORY_SPLIT_RE = re.compile(r"[;,\n|]+")
    _NON_ALNUM_RE = re.compile(r"[^A-Z0-9]+")

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._inventory_index: dict[str, dict[str, list[_TourNavigationEntry]]] = {}
        # Índices por posição na visita (CHAT_TOUR_LOCATION_ANSWERS): panorama
        # -> overlays desse panorama; sala (texto dobrado) -> overlays da sala.
        self._panorama_index: dict[str, dict[str, list[_TourNavigationEntry]]] = {}
        self._room_index: dict[str, dict[str, list[_TourNavigationEntry]]] = {}
        # sala (dobrada) -> nome de apresentação e panoramas, pela ordem do ficheiro
        self._rooms: dict[str, dict[str, dict[str, Any]]] = {}
        self._loaded_museums: set[str] = set()

    @staticmethod
    def _fold_room(value: str | None) -> str:
        import unicodedata

        normalized = unicodedata.normalize("NFKD", (value or "").casefold())
        folded = "".join(char for char in normalized if not unicodedata.combining(char))
        return re.sub(r"[^a-z0-9]+", " ", folded).strip()

    def _normalize_inventory(self, raw: str) -> str:
        cleaned = (raw or "").upper().strip()
        if not cleaned:
            return ""
        return self._NON_ALNUM_RE.sub("", cleaned)

    def _extract_candidates(self, inventory_text: str) -> list[str]:
        raw = (inventory_text or "").strip()
        if not raw:
            return []

        candidates: list[str] = [raw]
        without_parenthesis = re.sub(r"\([^)]*\)", " ", raw).strip()
        if without_parenthesis and without_parenthesis != raw:
            candidates.append(without_parenthesis)
        for token in self._INVENTORY_SPLIT_RE.split(raw):
            value = token.strip()
            if value:
                candidates.append(value)

        seen: set[str] = set()
        unique: list[str] = []
        for value in candidates:
            normalized = self._normalize_inventory(value)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique.append(value)
        return unique

    def _mapping_file_candidates(self, museum_slug: str, museum_id: str | None) -> list[Path]:
        base_dir = self.settings.poi_tours_dir_resolved
        values: list[str] = []

        for value in (museum_slug, museum_id):
            cleaned = (value or "").strip().lower()
            if cleaned and cleaned not in values:
                values.append(cleaned)

        return [base_dir / f"panorama-overlays-inventory-{value}.json" for value in values]

    def _ensure_loaded(self, museum_slug: str, museum_id: str | None) -> None:
        museum_key = (museum_slug or "").strip().lower()
        if not museum_key:
            return
        if museum_key in self._loaded_museums:
            return

        index: dict[str, list[_TourNavigationEntry]] = {}
        panorama_index: dict[str, list[_TourNavigationEntry]] = {}
        room_index: dict[str, list[_TourNavigationEntry]] = {}
        rooms: dict[str, dict[str, Any]] = {}
        loaded_any = False

        for file_path in self._mapping_file_candidates(museum_slug=museum_slug, museum_id=museum_id):
            if not file_path.exists() or not file_path.is_file():
                continue
            try:
                payload = json.loads(file_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, list):
                continue

            loaded_any = True
            for item in payload:
                if not isinstance(item, dict):
                    continue
                overlay_id = str(item.get("overlayId") or "").strip()
                panorama_key = str(item.get("panoramaKey") or "").strip()
                if not overlay_id or not panorama_key:
                    continue

                inventory_ids = item.get("inventoryIds")
                if isinstance(inventory_ids, str):
                    inventory_list = [inventory_ids]
                elif isinstance(inventory_ids, list):
                    inventory_list = [str(value).strip() for value in inventory_ids if str(value).strip()]
                else:
                    inventory_list = []

                location = str(item.get("location") or "").strip() or None
                title = str(item.get("title") or "").strip() or None
                # Uma zona sem peças (Pátio) continua a ser uma zona da visita.
                if location:
                    room = rooms.setdefault(
                        self._fold_room(location), {"room": location, "panorama_keys": []}
                    )
                    if panorama_key not in room["panorama_keys"]:
                        room["panorama_keys"].append(panorama_key)
                if not inventory_list:
                    continue

                for inventory_id in inventory_list:
                    entry = _TourNavigationEntry(
                        overlay_id=overlay_id,
                        panorama_key=panorama_key,
                        inventory_id=inventory_id,
                        location=location,
                        title=title,
                    )
                    panorama_index.setdefault(panorama_key, []).append(entry)
                    if location:
                        room_index.setdefault(self._fold_room(location), []).append(entry)
                    for candidate in self._extract_candidates(inventory_id):
                        normalized = self._normalize_inventory(candidate)
                        if not normalized:
                            continue
                        index.setdefault(normalized, []).append(entry)

        if loaded_any:
            self._inventory_index[museum_key] = index
            self._panorama_index[museum_key] = panorama_index
            self._room_index[museum_key] = room_index
            self._rooms[museum_key] = rooms
        else:
            self._inventory_index[museum_key] = {}
            self._panorama_index[museum_key] = {}
            self._room_index[museum_key] = {}
            self._rooms[museum_key] = {}
        self._loaded_museums.add(museum_key)

    _ROOM_TYPE_WORDS = frozenset({"sala", "salas", "capela", "claustro", "piso", "zona", "galeria", "room", "chapel"})
    _ROOM_LINK_WORDS = frozenset({"da", "de", "do", "das", "dos", "d", "of", "the"})

    def list_rooms(self, *, museum_slug: str, museum_id: str | None) -> list[dict[str, Any]]:
        """Zonas da visita pela ordem do ficheiro: nome, panoramas e n.º de peças."""
        self._ensure_loaded(museum_slug=museum_slug, museum_id=museum_id)
        museum_key = (museum_slug or "").strip().lower()
        result: list[dict[str, Any]] = []
        descriptions = _load_room_descriptions().get(museum_key) or _load_room_descriptions().get(
            (museum_id or "").strip().lower(), {}
        )
        for folded, room in self._rooms.get(museum_key, {}).items():
            entries = self._room_index.get(museum_key, {}).get(folded, [])
            curated = descriptions.get(folded, {})
            inventories: list[str] = []
            seen: set[str] = set()
            for entry in entries:
                normalized = self._normalize_inventory(entry.inventory_id)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    inventories.append(entry.inventory_id)
            result.append(
                {
                    "room": room["room"],
                    "folded": folded,
                    "panorama_keys": list(room["panorama_keys"]),
                    "inventories": inventories,
                    "description": curated.get("description") or None,
                    "source": curated.get("source") or None,
                    "targets": [
                        {
                            "overlay_id": e.overlay_id,
                            "panorama_key": e.panorama_key,
                            "inventory_id": e.inventory_id,
                            "location": e.location,
                            "title": e.title,
                        }
                        for e in entries
                    ],
                }
            )
        return result

    def find_room(self, *, museum_slug: str, museum_id: str | None, text: str) -> dict[str, Any] | None:
        """A zona da visita referida no texto («e a sala cantonal?»).

        Casa pelo nome completo dobrado («sala cantonal») ou pelo núcleo sem
        a palavra de tipo e as ligações («cantonal», «musica»), desde que o
        núcleo tenha 4+ letras e não seja só numérico («Sala 1.01» exige o
        nome completo). Devolve a sala com a correspondência mais longa e os
        tokens do texto que a cobriram.
        """
        tokens = self._fold_room(text).split()
        if not tokens:
            return None
        best: dict[str, Any] | None = None
        for room in self.list_rooms(museum_slug=museum_slug, museum_id=museum_id):
            name_tokens = room["folded"].split()
            candidates = [name_tokens]
            core = [t for t in name_tokens if t not in self._ROOM_TYPE_WORDS and t not in self._ROOM_LINK_WORDS]
            if core and core != name_tokens and any(len(t) >= 4 and not t.isdigit() for t in core):
                candidates.append(core)
            for candidate in candidates:
                size = len(candidate)
                for start in range(len(tokens) - size + 1):
                    if tokens[start:start + size] == candidate:
                        if best is None or size > best["matched_size"]:
                            best = {**room, "matched_size": size, "matched_tokens": set(candidate)}
                        break
        return best

    def describe_location(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        panorama_key: str | None,
        room: str | None,
    ) -> dict[str, Any]:
        """Peças assinaladas na posição atual da visita.

        Prefere o panorama (posição exata); sem panorama conhecido, ou sem
        peças nele, cai para a sala. `known` diz se a posição existe nos
        ficheiros da visita; `room` é o nome da sala resolvido.
        """
        self._ensure_loaded(museum_slug=museum_slug, museum_id=museum_id)
        museum_key = (museum_slug or "").strip().lower()
        panorama_entries = self._panorama_index.get(museum_key, {}).get((panorama_key or "").strip(), [])
        resolved_room = (room or "").strip() or None
        if not resolved_room:
            resolved_room = next((e.location for e in panorama_entries if e.location), None)
        room_entries = self._room_index.get(museum_key, {}).get(self._fold_room(resolved_room), []) if resolved_room else []

        def unique_inventories(entries: list[_TourNavigationEntry]) -> list[str]:
            seen: set[str] = set()
            ordered: list[str] = []
            for entry in entries:
                normalized = self._normalize_inventory(entry.inventory_id)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    ordered.append(entry.inventory_id)
            return ordered

        scope = "panorama" if panorama_entries else ("room" if room_entries else "none")
        chosen = panorama_entries or room_entries
        return {
            "known": bool(panorama_entries or room_entries),
            "scope": scope,
            "room": resolved_room,
            "panorama_key": (panorama_key or "").strip() or None,
            "inventories": unique_inventories(chosen),
            "room_inventory_count": len(unique_inventories(room_entries)),
            "targets": [
                {
                    "overlay_id": e.overlay_id,
                    "panorama_key": e.panorama_key,
                    "inventory_id": e.inventory_id,
                    "location": e.location,
                    "title": e.title,
                }
                for e in chosen
            ],
        }

    def resolve_targets(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        inventories: list[str],
        limit: int = 6,
    ) -> list[dict[str, Any]]:
        self._ensure_loaded(museum_slug=museum_slug, museum_id=museum_id)
        museum_key = (museum_slug or "").strip().lower()
        index = self._inventory_index.get(museum_key, {})
        if not index:
            return []

        results: list[dict[str, Any]] = []
        seen_pairs: set[tuple[str, str]] = set()
        max_results = max(limit, 1)

        for inventory_text in inventories:
            for candidate in self._extract_candidates(inventory_text):
                normalized = self._normalize_inventory(candidate)
                if not normalized:
                    continue
                entries = index.get(normalized, [])
                for entry in entries:
                    key = (entry.overlay_id, entry.panorama_key)
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    results.append(
                        {
                            "overlay_id": entry.overlay_id,
                            "panorama_key": entry.panorama_key,
                            "inventory_id": entry.inventory_id,
                            "location": entry.location,
                            "title": entry.title,
                        }
                    )
                    if len(results) >= max_results:
                        return results
        return results


@lru_cache(maxsize=1)
def get_tour_navigation_service() -> TourNavigationService:
    return TourNavigationService(get_settings())
