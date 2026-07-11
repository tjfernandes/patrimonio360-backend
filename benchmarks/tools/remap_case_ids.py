"""Remap legacy benchmark-case artifact IDs to the live index ID scheme.

The historical ``benchmark_cases.json`` targets IDs such as ``artifact_100000``
that predate the RAIZ reindex (live IDs look like ``raiz:movel:76202``). No
surviving index carries the legacy scheme, and the benchmark asset images were
re-encoded (no byte-identity with ``backend/Images``), so the remap combines
three tiers, highest precision first:

Tier A — model_3d cases: notes carry ``inventory <number>`` -> exact lookup by
    ``inventory_number`` + museum.
Tier B — text cases: notes carry the artifact title (optionally ``Title, year``)
    -> accepted only when the title (+year) matches exactly ONE artifact in the
    case museum.
Tier C — image cases: perceptual dHash of the asset against every collection
    image of the case museum (``Images/<museum_dir>/obj_<inv>/...``) -> accepted
    only when the best Hamming distance <= threshold and the runner-up is
    clearly worse. The matched file also yields the live ``image_id`` used for
    leave-self-out (``exclude_image_ids``).

All tiers feed one old_id -> new_id mapping shared by every case (text cases
inherit mappings discovered via their paired image cases and vice versa).
Unmapped cases are disabled and listed in the report for manual curation.

Read-only with respect to OpenSearch. Never modifies the input cases file.

Usage (from ``backend/``):
    python -m benchmarks.tools.remap_case_ids \
        --cases benchmarks/cases/benchmark_cases.json \
        --images-root Images \
        --output benchmarks/cases/benchmark_cases_live_ids.json \
        --multimodal-output benchmarks/cases/benchmark_cases_multimodal.json \
        --report benchmarks/cases/id_remap_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

from app.core.config import get_settings
from app.services.opensearch_client import OpenSearchGateway

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
_OBJ_DIR_PATTERN = re.compile(r"^obj_(?P<inventory>.+)$")
_NOTES_INVENTORY_PATTERN = re.compile(r"inventory\s+(?P<inventory>[\w./-]+)", re.IGNORECASE)
_NOTES_TITLE_YEAR_PATTERN = re.compile(
    r"^(?P<title>[^,]+?),\s*(?:c\.\s*)?(?P<year>\d{4})(?:[-/]\d{2,4})?\.?$"
)

_MUSEUM_ID_TO_DIR = {
    "mnt": "museu_nacional_do_traje",
    "mnaz": "museu_nacional_do_azulejo",
    "mj": "mosteiro_dos_jeronimos",
    "mnsr": "museu_nacional_soares_dos_reis",
}

# Notes that are commentary, not titles.
_NON_TITLE_NOTES_PREFIXES = (
    "named work",
    "novo case",
    "same artifact",
    "theme-driven",
    "distinct iconographic",
    "scaffold multi",
    "imagem de benchmark",
)


def _dhash(image: Any, *, hash_size: int = 8) -> int:
    """Difference hash (hash_size^2 bits); robust to re-encode/resize, not to re-photograph."""
    grayscale = image.convert("L").resize((hash_size + 1, hash_size), 2)  # 2 = BILINEAR
    pixels = list(grayscale.getdata())
    bits = 0
    for row in range(hash_size):
        for col in range(hash_size):
            left = pixels[row * (hash_size + 1) + col]
            right = pixels[row * (hash_size + 1) + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _load_pil():
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Pillow is required for perceptual matching.") from exc
    return Image


def _dhash_file(path: Path, *, hash_size: int = 8) -> int | None:
    Image = _load_pil()
    try:
        with Image.open(path) as image:
            return _dhash(image, hash_size=hash_size)
    except Exception:
        return None


def _hash_museum_images(
    images_root: Path, museum_dir: str, cache_path: Path | None, *, hash_size: int = 8
) -> dict[str, int]:
    """Return {relative_local_path: dhash} for one museum directory."""
    cache: dict[str, Any] = {}
    if cache_path is not None and cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    cache_key = f"{museum_dir}@{hash_size}"
    if cache_key in cache:
        hashes = {key: int(value) for key, value in cache[cache_key].items()}
        print(f"[remap] reusing dhash cache for {museum_dir} ({len(hashes)} files)")
        return hashes

    museum_path = images_root / museum_dir
    hashes: dict[str, int] = {}
    count = 0
    for path in museum_path.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        digest = _dhash_file(path, hash_size=hash_size)
        if digest is None:
            continue
        relative = path.relative_to(images_root.parent).as_posix()
        hashes[relative] = digest
        count += 1
        if count % 2000 == 0:
            print(f"[remap] dhashed {count} files in {museum_dir}...")
    print(f"[remap] dhashed {count} files in {museum_dir}")

    if cache_path is not None:
        cache[cache_key] = {key: str(value) for key, value in hashes.items()}
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return hashes


def _parse_matched_path(relative_path: str) -> tuple[str | None, str | None]:
    parts = Path(relative_path).parts
    museum_id = None
    inventory = None
    dir_to_id = {value: key for key, value in _MUSEUM_ID_TO_DIR.items()}
    for part in parts:
        if part in dir_to_id:
            museum_id = dir_to_id[part]
        match = _OBJ_DIR_PATTERN.match(part)
        if match:
            inventory = match.group("inventory")
    return museum_id, inventory


class LiveIndexResolver:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.gateway = OpenSearchGateway(self.settings)
        self.client = self.gateway._ensure_client()

    def _search(self, index: str, body: dict[str, Any]) -> list[dict[str, Any]]:
        response = self.client.search(index=index, body=body)
        return [hit["_source"] for hit in response.get("hits", {}).get("hits", [])]

    def artifact_by_inventory(
        self, *, inventory_number: str, museum_id: str
    ) -> dict[str, Any] | None:
        hits = self._search(
            self.settings.OPENSEARCH_INDEX_ARTIFACT,
            {
                "size": 2,
                "query": {
                    "bool": {
                        "should": [
                            {"term": {"inventory_number": inventory_number.lower()}},
                            {"match": {"inventory_number.text": inventory_number}},
                        ],
                        "minimum_should_match": 1,
                        "filter": [{"term": {"museum_id": museum_id}}],
                    }
                },
                "_source": [
                    "artifact_id",
                    "title",
                    "category",
                    "inventory_number",
                    "museum_id",
                ],
            },
        )
        return hits[0] if hits else None

    def artifacts_by_title(
        self, *, title: str, museum_id: str, year: int | None
    ) -> list[dict[str, Any]]:
        must: list[dict[str, Any]] = [{"match_phrase": {"title": title}}]
        filters: list[dict[str, Any]] = [{"term": {"museum_id": museum_id}}]
        if year is not None:
            filters.append(
                {
                    "bool": {
                        "should": [
                            {"term": {"start_year": year}},
                            {
                                "bool": {
                                    "must": [
                                        {"range": {"start_year": {"lte": year}}},
                                        {"range": {"end_year": {"gte": year}}},
                                    ]
                                }
                            },
                        ],
                        "minimum_should_match": 1,
                    }
                }
            )
        return self._search(
            self.settings.OPENSEARCH_INDEX_ARTIFACT,
            {
                "size": 5,
                "query": {"bool": {"must": must, "filter": filters}},
                "_source": [
                    "artifact_id",
                    "title",
                    "category",
                    "inventory_number",
                    "museum_id",
                    "start_year",
                ],
            },
        )

    def artifact_by_id(self, *, artifact_id: str) -> dict[str, Any] | None:
        hits = self._search(
            self.settings.OPENSEARCH_INDEX_ARTIFACT,
            {
                "size": 1,
                "query": {"term": {"artifact_id": artifact_id}},
                "_source": [
                    "artifact_id",
                    "title",
                    "category",
                    "inventory_number",
                    "museum_id",
                ],
            },
        )
        return hits[0] if hits else None

    def disambiguate_by_query(
        self,
        *,
        candidate_ids: list[str],
        query_text: str,
        museum_id: str,
    ) -> tuple[dict[str, Any] | None, float | None, float | None]:
        """Rank title-candidates by BM25 of the case query over search_text."""
        hits_raw = self.client.search(
            index=self.settings.OPENSEARCH_INDEX_ARTIFACT,
            body={
                "size": len(candidate_ids),
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query_text,
                                    "fields": ["search_text", "description", "title"],
                                }
                            }
                        ],
                        "filter": [
                            {"term": {"museum_id": museum_id}},
                            {"terms": {"artifact_id": candidate_ids}},
                        ],
                    }
                },
                "_source": [
                    "artifact_id",
                    "title",
                    "category",
                    "inventory_number",
                    "museum_id",
                ],
            },
        )
        hits = hits_raw.get("hits", {}).get("hits", [])
        if not hits:
            return None, None, None
        top_score = float(hits[0].get("_score") or 0.0)
        second_score = float(hits[1].get("_score") or 0.0) if len(hits) > 1 else 0.0
        return hits[0]["_source"], top_score, second_score

    def image_by_local_path(self, *, local_path: str) -> dict[str, Any] | None:
        hits = self._search(
            self.settings.OPENSEARCH_INDEX_IMAGE,
            {
                "size": 1,
                "query": {"term": {"local_path": local_path}},
                "_source": ["image_id", "artifact_id", "inventory_number", "local_path"],
            },
        )
        return hits[0] if hits else None


def _load_cases(cases_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload, list(payload.get("cases") or [])
    return {"schema_version": 1, "cases": payload}, list(payload)


def _resolve_case_asset(case: dict[str, Any], base_dir: Path, key: str) -> Path | None:
    raw = case.get(key)
    if not raw:
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    return candidate if candidate.exists() else None


def _notes_title_and_year(notes: str) -> tuple[str, int | None] | None:
    cleaned = (notes or "").strip()
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if any(lowered.startswith(prefix) for prefix in _NON_TITLE_NOTES_PREFIXES):
        return None
    match = _NOTES_TITLE_YEAR_PATTERN.match(cleaned)
    if match:
        year_raw = match.group("year")
        return match.group("title").strip(), int(year_raw)
    # Bare title (no comma/year) — only accept short, name-like notes.
    if "," not in cleaned and len(cleaned) <= 60:
        return cleaned, None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--images-root", required=True, help="e.g. backend/Images")
    parser.add_argument("--output", required=True)
    parser.add_argument("--multimodal-output", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--hash-cache", default=None)
    parser.add_argument("--dhash-threshold", type=int, default=10)
    parser.add_argument("--dhash-margin", type=int, default=4)
    parser.add_argument("--dhash-size", type=int, default=8,
                        help="dHash grid size (8 -> 64-bit; 16 -> 256-bit for fine textures).")
    args = parser.parse_args()

    cases_path = Path(args.cases).resolve()
    images_root = Path(args.images_root).resolve()
    if not images_root.exists():
        print(f"[remap] images root not found: {images_root}", file=sys.stderr)
        return 2

    payload, raw_cases = _load_cases(cases_path)
    base_dir = cases_path.parent
    hash_cache = Path(args.hash_cache).resolve() if args.hash_cache else None
    resolver = LiveIndexResolver()

    id_mapping: dict[str, str] = {}
    mapping_evidence: dict[str, dict[str, Any]] = {}
    case_image_matches: dict[str, dict[str, Any]] = {}
    problems: list[dict[str, Any]] = []

    def _record_mapping(old_id: str, artifact: dict[str, Any], *, tier: str, case_id: str) -> None:
        new_id = artifact["artifact_id"]
        existing = id_mapping.get(old_id)
        if existing and existing != new_id:
            problems.append(
                {
                    "case_id": case_id,
                    "old_id": old_id,
                    "conflict": [existing, new_id],
                    "tier": tier,
                }
            )
            return
        id_mapping[old_id] = new_id
        mapping_evidence[old_id] = {
            "new_id": new_id,
            "tier": tier,
            "case_id": case_id,
            "title": artifact.get("title"),
            "inventory_number": artifact.get("inventory_number"),
        }

    # ---- Tier A: model_3d notes carry "inventory <number>" -------------------
    for case in raw_cases:
        if case.get("mode") != "model_3d":
            continue
        old_target = (case.get("target_artifact") or "").strip()
        notes = case.get("notes") or ""
        match = _NOTES_INVENTORY_PATTERN.search(notes)
        if not old_target or not match:
            continue
        inventory = match.group("inventory").rstrip(".")
        artifact = resolver.artifact_by_inventory(
            inventory_number=inventory, museum_id=case["museum_id"]
        )
        if artifact is None:
            problems.append(
                {
                    "case_id": case.get("case_id"),
                    "old_id": old_target,
                    "reason": f"inventory '{inventory}' not found in live index",
                    "tier": "A",
                }
            )
            continue
        _record_mapping(old_target, artifact, tier="A_inventory", case_id=str(case.get("case_id")))

    # ---- Tier B: text-case notes carry a unique title ------------------------
    for case in raw_cases:
        if case.get("mode") not in {"text_single", "text_multi"}:
            continue
        old_target = (case.get("target_artifact") or "").strip()
        if not old_target or old_target in id_mapping:
            continue
        parsed = _notes_title_and_year(case.get("notes") or "")
        if parsed is None:
            continue
        title, year = parsed
        hits = resolver.artifacts_by_title(
            title=title, museum_id=case["museum_id"], year=year
        )
        exact = [hit for hit in hits if (hit.get("title") or "").strip().lower() == title.lower()]
        candidates = exact or hits
        if len(candidates) == 1:
            _record_mapping(
                old_target, candidates[0], tier="B_title", case_id=str(case.get("case_id"))
            )
        elif candidates:
            # Tier B2: disambiguate with the case query text (BM25 over search_text).
            query_text = (case.get("query") or "").strip()
            resolved = False
            if query_text:
                winner, top_score, second_score = resolver.disambiguate_by_query(
                    candidate_ids=[str(hit.get("artifact_id")) for hit in candidates],
                    query_text=query_text,
                    museum_id=case["museum_id"],
                )
                if (
                    winner is not None
                    and top_score
                    and (second_score in (None, 0.0) or top_score >= 1.6 * second_score)
                ):
                    _record_mapping(
                        old_target,
                        winner,
                        tier="B2_title_query",
                        case_id=str(case.get("case_id")),
                    )
                    resolved = True
            if not resolved:
                problems.append(
                    {
                        "case_id": case.get("case_id"),
                        "old_id": old_target,
                        "reason": f"title '{title}' ambiguous ({len(candidates)} candidates)",
                        "candidates": [
                            {
                                "artifact_id": hit.get("artifact_id"),
                                "title": hit.get("title"),
                                "inventory_number": hit.get("inventory_number"),
                            }
                            for hit in candidates
                        ],
                        "tier": "B",
                    }
                )

    # ---- Tier C: perceptual match for image cases ----------------------------
    museum_hashes: dict[str, dict[str, int]] = {}
    for case in raw_cases:
        if case.get("mode") != "image":
            continue
        case_id = str(case.get("case_id"))
        old_target = (case.get("target_artifact") or "").strip()
        asset_path = _resolve_case_asset(case, base_dir, "image_path")
        if asset_path is None:
            problems.append({"case_id": case_id, "reason": "asset image missing", "tier": "C"})
            continue
        museum_id = case["museum_id"]
        museum_dir = _MUSEUM_ID_TO_DIR.get(museum_id)
        if museum_dir is None:
            problems.append(
                {"case_id": case_id, "reason": f"unknown museum '{museum_id}'", "tier": "C"}
            )
            continue
        if museum_dir not in museum_hashes:
            museum_hashes[museum_dir] = _hash_museum_images(
                images_root, museum_dir, hash_cache, hash_size=args.dhash_size
            )
        asset_hash = _dhash_file(asset_path, hash_size=args.dhash_size)
        if asset_hash is None:
            problems.append({"case_id": case_id, "reason": "asset image unreadable", "tier": "C"})
            continue
        def _obj_dir(relative: str) -> str:
            for part in Path(relative).parts:
                if part.startswith("obj_"):
                    return part
            return relative

        best: tuple[int, str] | None = None
        second_other: int | None = None  # runner-up from a DIFFERENT artifact dir
        for relative, digest in museum_hashes[museum_dir].items():
            distance = _hamming(asset_hash, digest)
            if best is None or distance < best[0]:
                if best is not None and _obj_dir(best[1]) != _obj_dir(relative):
                    second_other = (
                        best[0]
                        if second_other is None or best[0] < second_other
                        else second_other
                    )
                best = (distance, relative)
            elif _obj_dir(relative) != _obj_dir(best[1]) and (
                second_other is None or distance < second_other
            ):
                second_other = distance
        if best is None or best[0] > args.dhash_threshold or (
            second_other is not None and second_other - best[0] < args.dhash_margin
        ):
            problems.append(
                {
                    "case_id": case_id,
                    "old_id": old_target,
                    "reason": "no unambiguous perceptual match",
                    "best_distance": None if best is None else best[0],
                    "best_path": None if best is None else best[1],
                    "second_other_artifact_distance": second_other,
                    "tier": "C",
                }
            )
            continue
        matched_relative = best[1]
        # Join via the live image doc (authoritative artifact_id); the path-derived
        # inventory is only a fallback (directory names mangle spaces/case).
        image_doc = resolver.image_by_local_path(local_path=matched_relative)
        artifact = None
        if image_doc and image_doc.get("artifact_id"):
            artifact = resolver.artifact_by_id(artifact_id=image_doc["artifact_id"])
        if artifact is None:
            parsed_museum, inventory = _parse_matched_path(matched_relative)
            if inventory:
                artifact = resolver.artifact_by_inventory(
                    inventory_number=inventory.replace("_", " "),
                    museum_id=parsed_museum or museum_id,
                )
        if artifact is None:
            problems.append(
                {
                    "case_id": case_id,
                    "old_id": old_target,
                    "reason": "matched image but artifact not in live index",
                    "matched": matched_relative,
                    "tier": "C",
                }
            )
            continue
        if old_target:
            _record_mapping(old_target, artifact, tier="C_perceptual", case_id=case_id)
        case_image_matches[case_id] = {
            "matched_local_path": matched_relative,
            "distance": best[0],
            "second_other_artifact_distance": second_other,
            "live_artifact": artifact,
            "live_image_id": (image_doc or {}).get("image_id"),
        }

    # ---- Rewrite cases --------------------------------------------------------
    def _is_live_id(value: str) -> bool:
        return value.startswith(("raiz:", "manual:"))

    remapped_cases: list[dict[str, Any]] = []
    unmapped_cases: list[str] = []
    for case in raw_cases:
        new_case = dict(case)
        case_id = str(case.get("case_id"))
        old_target = (case.get("target_artifact") or "").strip()
        unmapped = False
        if old_target:
            if old_target in id_mapping:
                new_case["target_artifact"] = id_mapping[old_target]
            elif not _is_live_id(old_target):
                unmapped = True
        relevant = case.get("relevant_artifacts") or []
        if relevant:
            new_relevant = []
            for item in relevant:
                item_clean = (item or "").strip()
                if item_clean in id_mapping:
                    new_relevant.append(id_mapping[item_clean])
                else:
                    new_relevant.append(item_clean)
                    if item_clean and not _is_live_id(item_clean):
                        unmapped = True
            new_case["relevant_artifacts"] = new_relevant
        match = case_image_matches.get(case_id)
        if match and match.get("live_image_id"):
            new_case["exclude_image_ids"] = [match["live_image_id"]]
        if unmapped:
            new_case["enabled"] = False
            new_case["notes"] = (
                f"{case.get('notes') or ''} [unmapped_legacy_id: needs manual ground truth]"
            ).strip()
            unmapped_cases.append(case_id)
        remapped_cases.append(new_case)

    output_payload = dict(payload)
    output_payload["cases"] = remapped_cases
    output_payload["notes"] = (
        f"{payload.get('notes') or ''} Remapped to live raiz IDs by "
        "benchmarks.tools.remap_case_ids; leave-self-out via exclude_image_ids."
    ).strip()
    output_path = Path(args.output).resolve()
    output_path.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[remap] wrote {len(remapped_cases)} cases -> {output_path}")

    # ---- Starter multimodal cases ---------------------------------------------
    if args.multimodal_output:
        text_query_by_target: dict[str, str] = {}
        for case in remapped_cases:
            if case.get("mode") == "text_single" and case.get("enabled", True):
                target = case.get("target_artifact")
                if target and case.get("query") and _is_live_id(str(target)):
                    text_query_by_target.setdefault(target, case["query"])

        multimodal_cases: list[dict[str, Any]] = []
        for case in remapped_cases:
            if case.get("mode") != "image" or not case.get("enabled", True):
                continue
            case_id = str(case.get("case_id"))
            target = str(case.get("target_artifact") or "")
            if not _is_live_id(target):
                continue
            match = case_image_matches.get(case_id, {})
            live = match.get("live_artifact") or {}
            t2i_query = text_query_by_target.get(target)
            if t2i_query:
                multimodal_cases.append(
                    {
                        "case_id": f"T2I_{case_id}",
                        "museum_id": case["museum_id"],
                        "mode": "text_to_image",
                        "query": t2i_query,
                        "target_artifact": target,
                        "exclude_image_ids": case.get("exclude_image_ids", []),
                        "notes": "Auto-derivado do caso text_single com o mesmo alvo.",
                    }
                )
            descriptor = (live.get("category") or live.get("title") or "").strip()
            if descriptor:
                multimodal_cases.append(
                    {
                        "case_id": f"IT_{case_id}",
                        "museum_id": case["museum_id"],
                        "mode": "image_text",
                        "image_path": case.get("image_path"),
                        "query": f"peças de {descriptor.lower()} semelhantes a esta imagem",
                        "target_artifact": target,
                        "exclude_image_ids": case.get("exclude_image_ids", []),
                        "notes": (
                            "AUTO-GERADO (curar manualmente: substituir por restrição "
                            "composicional real)."
                        ),
                    }
                )
        multimodal_payload = {
            "schema_version": payload.get("schema_version", 1),
            "notes": (
                "Starter multimodal cases (text_to_image / image_text) auto-derived by "
                "benchmarks.tools.remap_case_ids. image_text queries are placeholders "
                "and MUST be manually curated into real compositional constraints."
            ),
            "cases": multimodal_cases,
        }
        multimodal_path = Path(args.multimodal_output).resolve()
        multimodal_path.write_text(
            json.dumps(multimodal_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"[remap] wrote {len(multimodal_cases)} starter multimodal cases -> {multimodal_path}"
        )

    report = {
        "cases_total": len(raw_cases),
        "id_mapping": mapping_evidence,
        "image_case_matches": case_image_matches,
        "problems": problems,
        "disabled_unmapped_cases": unmapped_cases,
    }
    if args.report:
        report_path = Path(args.report).resolve()
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[remap] report -> {report_path}")
    print(
        f"[remap] mapped {len(id_mapping)} legacy IDs "
        f"(A/B/C tiers); {len(problems)} problems; "
        f"{len(unmapped_cases)} cases disabled pending manual ground truth"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
