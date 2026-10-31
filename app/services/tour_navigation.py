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
        # Lugares da visita: os da própria visita 3DVista (app/data/tour_locations.json,
        # pela ordem da visita) fundidos com as salas dos ficheiros poi_tours.
        self._places: dict[str, list[dict[str, Any]]] = {}
        self._panorama_place: dict[str, dict[str, int]] = {}
        self._tour_locations: dict[str, Any] | None = None
        # número como a visita o anuncia (normalizado) -> números do catálogo
        self._corrected_index: dict[str, dict[str, list[str]]] = {}
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
        corrected: dict[str, list[str]] = {}
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

                # `inventoryIdsOriginal`: o que a visita anuncia quando não é o
                # número do catálogo (só com um original: vários não se sabe a
                # que peça corresponde cada um)
                originals = item.get("inventoryIdsOriginal")
                if isinstance(originals, list) and len(originals) == 1:
                    original_key = self._normalize_inventory(str(originals[0]))
                    catalogue_keys = {self._normalize_inventory(value) for value in inventory_list}
                    if original_key and original_key not in catalogue_keys:
                        corrected.setdefault(original_key, list(inventory_list))

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
            self._corrected_index[museum_key] = corrected
        else:
            self._inventory_index[museum_key] = {}
            self._panorama_index[museum_key] = {}
            self._room_index[museum_key] = {}
            self._rooms[museum_key] = {}
        self._build_places(museum_key=museum_key, museum_id=museum_id, poi_rooms=rooms)
        self._loaded_museums.add(museum_key)

    def overlay_inventories(self, *, museum_slug: str, museum_id: str | None, overlay_id: str) -> list[str]:
        """N.ºs do catálogo do hotspot `overlay_id` («Apontar e perguntar»: o toque caiu num hotspot)."""
        self._ensure_loaded(museum_slug=museum_slug, museum_id=museum_id)
        museum_key = (museum_slug or "").strip().lower()
        wanted = (overlay_id or "").strip()
        found: list[str] = []
        for entries in self._panorama_index.get(museum_key, {}).values():
            for entry in entries:
                if entry.overlay_id == wanted and entry.inventory_id not in found:
                    found.append(entry.inventory_id)
        return found

    def panorama_room(self, *, museum_slug: str, museum_id: str | None, panorama_key: str) -> str | None:
        """Nome da zona de um panorama: o lugar da visita a que pertence, senão
        a `location` dos seus hotspots; None se o panorama é desconhecido."""
        self._ensure_loaded(museum_slug=museum_slug, museum_id=museum_id)
        museum_key = (museum_slug or "").strip().lower()
        panorama = (panorama_key or "").strip()
        place_index = self._panorama_place.get(museum_key, {}).get(panorama)
        if place_index is not None:
            return str(self._places[museum_key][place_index]["room"])
        return next((e.location for e in self._panorama_index.get(museum_key, {}).get(panorama, []) if e.location), None)

    def panorama_inventory_keys(self, *, museum_slug: str, museum_id: str | None, panorama_key: str) -> set[str]:
        """N.ºs (normalizados) das peças com hotspot no panorama."""
        self._ensure_loaded(museum_slug=museum_slug, museum_id=museum_id)
        museum_key = (museum_slug or "").strip().lower()
        entries = self._panorama_index.get(museum_key, {}).get((panorama_key or "").strip(), [])
        return {self._normalize_inventory(e.inventory_id) for e in entries if self._normalize_inventory(e.inventory_id)}

    def tour_inventory_keys(self, *, museum_slug: str, museum_id: str | None) -> set[str]:
        """N.ºs (normalizados) de todas as peças com hotspot na visita do museu."""
        self._ensure_loaded(museum_slug=museum_slug, museum_id=museum_id)
        museum_key = (museum_slug or "").strip().lower()
        return set(self._inventory_index.get(museum_key, {}).keys())

    def normalize_inventory_key(self, inventory: str | None) -> str:
        return self._normalize_inventory(inventory or "")

    def corrected_inventories(self, *, museum_slug: str, museum_id: str | None, inventory: str) -> list[str]:
        """Números do catálogo para um número tal como a visita o anuncia.

        Vazio quando o mapa dos POIs não tem correspondência: o número da
        visita já é o do catálogo, ou não se conhece.
        """
        self._ensure_loaded(museum_slug=museum_slug, museum_id=museum_id)
        museum_key = (museum_slug or "").strip().lower()
        return list(self._corrected_index.get(museum_key, {}).get(self._normalize_inventory(inventory), []))

    def _tour_places_for(self, museum_key: str, museum_id: str | None) -> list[dict[str, Any]]:
        if self._tour_locations is None:
            path = getattr(self.settings, "tour_locations_path_resolved", None)
            try:
                payload = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
            except Exception:
                payload = {}
            self._tour_locations = payload if isinstance(payload, dict) else {}
        for key in (museum_key, (museum_id or "").strip().lower()):
            places = self._tour_locations.get(key)
            if isinstance(places, list):
                return [p for p in places if isinstance(p, dict) and p.get("room") and p.get("panorama_keys")]
        return []

    def _build_places(self, *, museum_key: str, museum_id: str | None, poi_rooms: dict[str, dict[str, Any]]) -> None:
        """Funde os lugares da visita 3DVista com as salas dos poi_tours.

        A visita dá todos os lugares navegáveis (Pátio, Jardim, túmulos…) pela
        ordem da visita; os poi_tours dão as peças. Uma sala dos poi_tours junta-se
        ao lugar com o mesmo nome dobrado ou com panoramas em comum («Sala
        Chinioiserie» nos poi_tours = «Sala Chinoiserie» na visita), ou a um
        lugar de uma só palavra que é o início do seu nome («Escadaria» /
        «Escadaria Monumental»); sem correspondência, entra no fim da lista.
        """
        places: list[dict[str, Any]] = [
            {
                "room": str(tour_place["room"]).strip(),
                "aliases": [self._fold_room(tour_place["room"])],
                "panorama_keys": [str(k) for k in tour_place["panorama_keys"]],
            }
            for tour_place in self._tour_places_for(museum_key, museum_id)
        ]
        for folded, poi_room in poi_rooms.items():
            poi_panoramas = set(poi_room["panorama_keys"])
            head = folded.split()[0] if folded else ""
            target = (
                next((p for p in places if folded in p["aliases"]), None)
                or next((p for p in places if poi_panoramas & set(p["panorama_keys"])), None)
            )
            if target is None and head and len(folded.split()) > 1:
                singles = [p for p in places if p["aliases"][0] == head]
                target = singles[0] if len(singles) == 1 else None
            if target is None:
                places.append(
                    {"room": poi_room["room"], "aliases": [folded], "panorama_keys": list(poi_room["panorama_keys"])}
                )
                continue
            if folded not in target["aliases"]:
                target["aliases"].append(folded)
            for key in poi_room["panorama_keys"]:
                if key not in target["panorama_keys"]:
                    target["panorama_keys"].append(key)
        panorama_place: dict[str, int] = {}
        for index, place in enumerate(places):
            for key in place["panorama_keys"]:
                panorama_place.setdefault(key, index)
        self._places[museum_key] = places
        self._panorama_place[museum_key] = panorama_place

    # Palavras que dizem o tipo de zona e ligações: não identificam um lugar sozinhas.
    _ROOM_TYPE_WORDS = frozenset({"sala", "salas", "piso", "zona", "zonas", "galeria", "room", "area", "espaco"})
    _ROOM_LINK_WORDS = frozenset({"da", "de", "do", "das", "dos", "e", "a", "o", "of", "the", "and"})

    def _place_entries(self, museum_key: str, place: dict[str, Any]) -> list[_TourNavigationEntry]:
        seen: set[tuple[str, str]] = set()
        entries: list[_TourNavigationEntry] = []
        for alias in place["aliases"]:
            for entry in self._room_index.get(museum_key, {}).get(alias, []):
                key = (entry.overlay_id, entry.inventory_id)
                if key not in seen:
                    seen.add(key)
                    entries.append(entry)
        return entries

    def list_rooms(self, *, museum_slug: str, museum_id: str | None) -> list[dict[str, Any]]:
        """Lugares da visita pela ordem da visita: nome, panoramas, peças e descrição curada."""
        self._ensure_loaded(museum_slug=museum_slug, museum_id=museum_id)
        museum_key = (museum_slug or "").strip().lower()
        result: list[dict[str, Any]] = []
        descriptions = _load_room_descriptions().get(museum_key) or _load_room_descriptions().get(
            (museum_id or "").strip().lower(), {}
        )
        for place in self._places.get(museum_key, []):
            entries = self._place_entries(museum_key, place)
            curated = next((descriptions[a] for a in place["aliases"] if a in descriptions), {})
            inventories: list[str] = []
            seen: set[str] = set()
            for entry in entries:
                normalized = self._normalize_inventory(entry.inventory_id)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    inventories.append(entry.inventory_id)
            result.append(
                {
                    "room": place["room"],
                    "folded": place["aliases"][0],
                    "aliases": list(place["aliases"]),
                    "panorama_keys": list(place["panorama_keys"]),
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

    def _significant_tokens(self, folded_name: str) -> list[str]:
        return [
            t for t in folded_name.split()
            if t not in self._ROOM_LINK_WORDS and (len(t) >= 2 or t.isdigit())
        ]

    def match_rooms(self, *, museum_slug: str, museum_id: str | None, text: str) -> dict[str, Any] | None:
        """Lugares da visita referidos no texto, com empate = ambiguidade.

        Pontuação por nome (e aliases): todas as palavras significativas do
        nome presentes no texto = correspondência completa (vale sempre mais);
        só algumas, desde que uma delas seja distintiva (4+ letras, não é tipo
        de zona nem número) = parcial. «Sala 1.01» exige o número; «claustro»
        casa «Claustro Térreo» e «Claustro Superior» (dois candidatos).
        Devolve {"candidates": [...], "matched_tokens": {...}} ou None.
        """
        tokens = set(self._fold_room(text).split())
        if not tokens:
            return None
        scored: list[tuple[float, dict[str, Any], set[str]]] = []
        for room in self.list_rooms(museum_slug=museum_slug, museum_id=museum_id):
            best_score, best_tokens = 0.0, set()
            for name in [room["folded"], *room["aliases"]]:
                significant = self._significant_tokens(name)
                if not significant:
                    continue
                matched = [t for t in significant if t in tokens]
                if matched and len(matched) == len(significant):
                    score = 10.0 + len(significant)
                elif any(t not in self._ROOM_TYPE_WORDS and not t.isdigit() and len(t) >= 4 for t in matched):
                    score = float(len(matched))
                else:
                    continue
                if score > best_score:
                    best_score, best_tokens = score, set(matched)
            if best_score > 0:
                scored.append((best_score, room, best_tokens))
        if not scored:
            return None
        top = max(score for score, _room, _tokens in scored)
        winners = [(room, matched) for score, room, matched in scored if score == top]
        return {
            "candidates": [room for room, _matched in winners],
            "matched_tokens": set().union(*(matched for _room, matched in winners)),
            "full": top >= 10.0,
        }

    def find_room(self, *, museum_slug: str, museum_id: str | None, text: str) -> dict[str, Any] | None:
        """O lugar da visita referido no texto, quando é só um."""
        match = self.match_rooms(museum_slug=museum_slug, museum_id=museum_id, text=text)
        if not match or len(match["candidates"]) != 1:
            return None
        return {**match["candidates"][0], "matched_tokens": match["matched_tokens"]}

    def describe_location(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        panorama_key: str | None,
        room: str | None,
    ) -> dict[str, Any]:
        """Peças assinaladas na posição atual da visita.

        Responde pela sala inteira (peças do panorama atual primeiro); sem
        sala conhecida fica pelo panorama. `known` diz se a posição existe nos
        ficheiros da visita; `room` é o nome da sala resolvido.
        """
        self._ensure_loaded(museum_slug=museum_slug, museum_id=museum_id)
        museum_key = (museum_slug or "").strip().lower()
        panorama = (panorama_key or "").strip()
        panorama_entries = self._panorama_index.get(museum_key, {}).get(panorama, [])
        places = self._places.get(museum_key, [])
        place_index = self._panorama_place.get(museum_key, {}).get(panorama)
        place = places[place_index] if place_index is not None else None
        # O tour reporta o título do panorama («Salão Nobre 123», «02_Igreja_003_…»);
        # o nome da zona vem do lugar da visita, depois dos poi_tours, e só por
        # fim do que o tour mandou (se não for um nome de ficheiro).
        resolved_room = place["room"] if place else next((e.location for e in panorama_entries if e.location), None)
        reported = (room or "").strip() or None
        if not resolved_room and reported:
            if "_" in reported:
                guessed = self.find_room(museum_slug=museum_slug, museum_id=museum_id, text=reported)
                resolved_room = guessed["room"] if guessed else None
                place = next((p for p in places if guessed and p["room"] == guessed["room"]), None)
            else:
                resolved_room = reported
                folded_reported = self._fold_room(reported)
                place = next((p for p in places if folded_reported in p["aliases"]), None)
        if place is not None:
            room_entries = self._place_entries(museum_key, place)
        else:
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

        # «Onde estou?» responde pela sala inteira: uma sala tem vários
        # panoramas e cada um só mostra alguns hotspots, por isso listar só os
        # do panorama dava respostas diferentes dentro da mesma sala. As peças
        # deste ponto vêm primeiro; o resto da sala a seguir. Só sem sala
        # conhecida é que a resposta fica pelo panorama.
        if room_entries:
            scope = "room"
            here = {id(e) for e in panorama_entries}
            chosen = list(panorama_entries) + [e for e in room_entries if id(e) not in here]
        else:
            scope = "panorama" if panorama_entries else "none"
            chosen = panorama_entries
        return {
            "known": bool(panorama_entries or room_entries or place is not None),
            "scope": scope,
            "room": resolved_room,
            "panorama_key": (panorama_key or "").strip() or None,
            "inventories": unique_inventories(chosen),
            "panorama_inventories": unique_inventories(panorama_entries),
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
