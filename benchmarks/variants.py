from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VariantSpec:
    name: str
    description: str
    supported: bool
    unsupported_reason: str | None = None


VARIANT_SPECS: dict[str, VariantSpec] = {
    "full": VariantSpec(
        name="full",
        description="Runs the benchmark against the current production retrieval pipeline.",
        supported=True,
    ),
    "no_rewriting": VariantSpec(
        name="no_rewriting",
        description=(
            "Alias of the current production retrieval behavior. "
            "Text retrieval already uses the raw user wording in the current pipeline."
        ),
        supported=True,
    ),
    "bm25_only": VariantSpec(
        name="bm25_only",
        description=(
            "Lexical-only baseline: runs the production BM25 clause set (no_stem multi_match "
            "variants + filters + boosts) without the kNN branch or the hybrid pipeline. "
            "Uses the raw case query (no LLM rewriting) for isolation."
        ),
        supported=True,
    ),
    "dense_only": VariantSpec(
        name="dense_only",
        description=(
            "Dense-only baseline: pure text_embedding kNN with production filters, without "
            "the BM25 branch or the hybrid pipeline. Uses the raw case query (no LLM rewriting)."
        ),
        supported=True,
    ),
    "hybrid": VariantSpec(
        name="hybrid",
        description="Placeholder for an explicit hybrid baseline.",
        supported=False,
        unsupported_reason=(
            "The current pipeline already defines the OpenSearch hybrid query shape. "
            "A separate benchmark variant would require changing that query."
        ),
    ),
    "no_filtering": VariantSpec(
        name="no_filtering",
        description="Placeholder for a future no-museum-filter baseline.",
        supported=False,
        unsupported_reason=(
            "Museum filtering is part of the current production retrieval behavior and is "
            "kept unchanged in this benchmark."
        ),
    ),
}


def get_variant_spec(name: str) -> VariantSpec:
    try:
        return VARIANT_SPECS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown benchmark variant '{name}'. Expected one of {sorted(VARIANT_SPECS)}."
        ) from exc


def resolve_variants(names: list[str]) -> list[VariantSpec]:
    if not names:
        return [VARIANT_SPECS["full"]]
    return [get_variant_spec(name) for name in names]

