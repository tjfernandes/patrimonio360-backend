# -*- coding: utf-8 -*-
"""Avaliação da recuperação de paragens por tema (percursos temáticos, Fase 1.5).

    python tools/percursos/avaliar_temas.py pool            # pool a anotar: benchmarks/percursos/pool-<museu>.csv
    python tools/percursos/avaliar_temas.py eval            # nDCG@8 e recall@20 por variante, com as anotações
    python tools/percursos/avaliar_temas.py eval --museus mnaz --temas t01 t02

Temas em benchmarks/percursos/temas.json (30 × {mnaz, mnaa}); relevância
graduada 0/1/2 por paragem em `judgments[<museu>][<id do tema>][<stop_id>]`,
anotada à mão (o que não está anotado conta 0). Quatro variantes, todas
sobre as paragens `best_placement` do índice cultural_heritage_tour_stops_v1
(só leitura), com os mesmos filtros duros (museu; século quando o tema o diz):

  texto       BM25 só no texto do catálogo (title, description, técnica, categoria, material)
  imagem      texto→imagem (Qwen3-VL-Embedding-8B) nas fotografias das peças com paragem
  atributos   listas fechadas do VLM que batem nas facetas do tema + BM25 só no que o VLM escreveu
  hibrido     o que o planeador usa: lexical + semântico + imagem + atributos, Weighted RRF (ROUTE_WEIGHT_*)

`pool` junta o top-20 de cada variante por tema e museu num CSV com título,
sala, n.º de inventário, campos visuais e uma coluna `relevancia` vazia; as
anotações copiam-se depois para o temas.json (ou dão-se com --anotacoes CSV,
colunas tema,museu,stop_id,relevancia). Precisa do índice de paragens já
criado e dos embedders (GPU) para as variantes imagem e híbrido; sem GPU
corre só `--variantes texto atributos`.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.services.retrieval.fusion import BranchHit, weighted_rrf  # noqa: E402
from app.services.route_planning.search import RouteStopSearch  # noqa: E402
from app.services.route_planning.theme import rule_facets  # noqa: E402

TEMAS = ROOT / "benchmarks" / "percursos" / "temas.json"
VARIANTES = ("texto", "imagem", "atributos", "hibrido")


def ndcg(ranked: list[str], gains: dict[str, int], k: int) -> float:
    dcg = sum((2 ** gains.get(stop_id, 0) - 1) / math.log2(i + 2) for i, stop_id in enumerate(ranked[:k]))
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall(ranked: list[str], gains: dict[str, int], k: int) -> float:
    relevant = {s for s, g in gains.items() if g > 0}
    if not relevant:
        return 0.0
    return len(relevant & set(ranked[:k])) / len(relevant)


class Runner:
    def __init__(self, *, variantes: list[str]) -> None:
        from app.services.opensearch_client import OpenSearchGateway

        self.settings = get_settings()
        client = OpenSearchGateway(self.settings)._ensure_client()
        self.search = RouteStopSearch(client, stops_index=str(getattr(self.settings, "ROUTE_STOPS_INDEX", "cultural_heritage_tour_stops_v1")), images_index=self.settings.OPENSEARCH_INDEX_IMAGE)
        self.variantes = variantes
        self.embeddings = None
        if any(v in variantes for v in ("imagem", "hibrido")):
            from app.services.embeddings import EmbeddingProvider

            self.embeddings = EmbeddingProvider(self.settings)
        self.weights = {
            "lexical": float(getattr(self.settings, "ROUTE_WEIGHT_LEXICAL", 1.0)), "semantic": float(getattr(self.settings, "ROUTE_WEIGHT_SEMANTIC", 1.0)),
            "image": float(getattr(self.settings, "ROUTE_WEIGHT_IMAGE", 0.7)), "attributes": float(getattr(self.settings, "ROUTE_WEIGHT_ATTRIBUTES", 1.0)),
        }

    async def rankings(self, *, museum: str, text: str, size: int = 40) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
        facets = rule_facets(text)
        filters = RouteStopSearch.hard_filters(museum_slug=museum, facets=facets, year_tolerance=int(getattr(self.settings, "ROUTE_YEAR_TOLERANCE", 25)))
        query = " ".join(p for p in [facets.free_text or text, *facets.colors, *facets.pigments, *facets.motifs, *facets.iconography] if p)
        docs: dict[str, dict[str, Any]] = {}
        out: dict[str, list[str]] = {}

        def keep(hits):
            ids = []
            for _artifact, _score, doc in hits:
                stop_id = str(doc.get("stop_id") or "")
                if stop_id and stop_id not in ids:
                    ids.append(stop_id)
                    docs.setdefault(stop_id, doc)
            return ids

        text_hits = self.search.lexical(query_text=query, filters=filters, size=size, fields=RouteStopSearch.CATALOGUE_FIELDS)
        if "texto" in self.variantes:
            out["texto"] = keep(text_hits)
        attr_hits = self.search.attributes(facets=facets, filters=filters, size=size) if facets.has_visual() else []
        visual_text_hits = self.search.lexical(query_text=query, filters=filters, size=size, fields=RouteStopSearch.VISUAL_FIELDS)
        if "atributos" in self.variantes:
            fused = weighted_rrf({"attributes": [BranchHit(artifact_id=str(d.get("stop_id")), rank=i + 1, score=s) for i, (_a, s, d) in enumerate(attr_hits)],
                                  "visual_text": [BranchHit(artifact_id=str(d.get("stop_id")), rank=i + 1, score=s) for i, (_a, s, d) in enumerate(visual_text_hits)]},
                                 weights={"attributes": 1.0, "visual_text": 1.0}, rrf_k=60)
            for _a, _s, d in attr_hits + visual_text_hits:
                docs.setdefault(str(d.get("stop_id")), d)
            out["atributos"] = [r.artifact_id for r in sorted(fused, key=lambda r: (-r.fusion_score, r.artifact_id))][:size]
        image_hits = []
        semantic_hits = []
        if self.embeddings is not None:
            eligible = self.search.artifact_ids(filters=filters)
            image_embedding = await self.embeddings.embed_multimodal_text_query(query)
            image_hits = self.search.image(embedding=image_embedding, museum_id=museum, artifact_ids=eligible, size=size)
            if "hibrido" in self.variantes:
                text_embedding = await self.embeddings.embed_text(query)
                semantic_hits = self.search.semantic(embedding=text_embedding, filters=filters, size=size)
        if "imagem" in self.variantes:
            by_artifact = self.search.stops_by_artifact(museum_slug=museum, artifact_ids=[a for a, _s, _d in image_hits])
            ids = []
            for artifact_id, _score, _doc in image_hits:
                stop = by_artifact.get(artifact_id)
                if stop and stop.get("stop_id") not in ids:
                    ids.append(stop["stop_id"])
                    docs.setdefault(stop["stop_id"], stop)
            out["imagem"] = ids
        if "hibrido" in self.variantes:
            lexical_hits = self.search.lexical(query_text=query, filters=filters, size=size)
            by_artifact = self.search.stops_by_artifact(museum_slug=museum, artifact_ids=[a for a, _s, _d in image_hits])
            branches = {
                "lexical": [BranchHit(artifact_id=a, rank=i + 1, score=s) for i, (a, s, _d) in enumerate(lexical_hits)],
                "semantic": [BranchHit(artifact_id=a, rank=i + 1, score=s) for i, (a, s, _d) in enumerate(semantic_hits)],
                "image": [BranchHit(artifact_id=a, rank=i + 1, score=s) for i, (a, s, _d) in enumerate(image_hits)],
                "attributes": [BranchHit(artifact_id=a, rank=i + 1, score=s) for i, (a, s, _d) in enumerate(attr_hits)],
            }
            fused = weighted_rrf(branches, weights=self.weights, rrf_k=int(getattr(self.settings, "ROUTE_RRF_K", 60)))
            artifact_to_stop = {str(d.get("artifact_id")): str(d.get("stop_id")) for d in docs.values()}
            artifact_to_stop.update({a: s["stop_id"] for a, s in by_artifact.items()})
            for _a, _s, d in lexical_hits + semantic_hits + attr_hits:
                docs.setdefault(str(d.get("stop_id")), d)
                artifact_to_stop[str(d.get("artifact_id"))] = str(d.get("stop_id"))
            for a, s in by_artifact.items():
                docs.setdefault(s["stop_id"], s)
            ids = []
            for r in sorted(fused, key=lambda r: (-r.fusion_score, r.artifact_id)):
                stop_id = artifact_to_stop.get(r.artifact_id)
                if stop_id and stop_id not in ids:
                    ids.append(stop_id)
            out["hibrido"] = ids[:size]
        return out, docs


def load_temas() -> dict[str, Any]:
    return json.loads(TEMAS.read_text(encoding="utf-8"))


async def cmd_pool(args: argparse.Namespace) -> int:
    temas = load_temas()
    runner = Runner(variantes=args.variantes)
    for museum in args.museus:
        rows: list[dict[str, Any]] = []
        for theme in temas["themes"]:
            if args.temas and theme["id"] not in args.temas:
                continue
            rankings, docs = await runner.rankings(museum=museum, text=theme["text"], size=20)
            pooled: list[str] = []
            for variante in VARIANTES:
                for stop_id in rankings.get(variante, []):
                    if stop_id not in pooled:
                        pooled.append(stop_id)
            existing = ((temas.get("judgments") or {}).get(museum) or {}).get(theme["id"]) or {}
            for stop_id in pooled:
                doc = docs.get(stop_id) or {}
                visual = doc.get("visual") or {}
                rows.append({
                    "tema": theme["id"], "tema_texto": theme["text"], "museu": museum, "stop_id": stop_id,
                    "inventario": doc.get("inventory_number"), "titulo": doc.get("title"), "sala": doc.get("room"),
                    "data": (doc.get("catalogue") or {}).get("date_or_period"), "categoria": (doc.get("catalogue") or {}).get("category"),
                    "cores": ", ".join(visual.get("color_names") or []), "motivos": ", ".join(visual.get("motifs") or []),
                    "iconografia": ", ".join(visual.get("iconography") or []), "uma_linha": visual.get("one_line"),
                    "variantes": ";".join(v for v in VARIANTES if stop_id in rankings.get(v, [])),
                    "relevancia": existing.get(stop_id, ""),
                })
            print(f"{museum} {theme['id']} «{theme['text']}»: pool {len(pooled)} (" + ", ".join(f"{v} {len(rankings.get(v, []))}" for v in VARIANTES if v in rankings) + ")")
        out = TEMAS.parent / f"pool-{museum}.csv"
        with out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["tema", "museu", "stop_id", "relevancia"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"escrito {out} ({len(rows)} linhas)")
    return 0


def load_annotations_csv(path: Path) -> dict[str, dict[str, dict[str, int]]]:
    judgments: dict[str, dict[str, dict[str, int]]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            value = str(row.get("relevancia") or "").strip()
            if value not in {"0", "1", "2"}:
                continue
            judgments.setdefault(row["museu"], {}).setdefault(row["tema"], {})[row["stop_id"]] = int(value)
    return judgments


async def cmd_eval(args: argparse.Namespace) -> int:
    temas = load_temas()
    judgments = temas.get("judgments") or {}
    if args.anotacoes:
        for museum, per_theme in load_annotations_csv(Path(args.anotacoes)).items():
            for theme_id, values in per_theme.items():
                judgments.setdefault(museum, {}).setdefault(theme_id, {}).update(values)
    runner = Runner(variantes=args.variantes)
    results: dict[str, Any] = {"museums": {}, "variantes": args.variantes}
    for museum in args.museus:
        per_variant = {v: {"ndcg8": [], "recall20": []} for v in args.variantes}
        evaluated = 0
        for theme in temas["themes"]:
            if args.temas and theme["id"] not in args.temas:
                continue
            gains = {k: int(v) for k, v in ((judgments.get(museum) or {}).get(theme["id"]) or {}).items()}
            if not any(g > 0 for g in gains.values()):
                continue
            evaluated += 1
            rankings, _docs = await runner.rankings(museum=museum, text=theme["text"], size=40)
            line = [f"{museum} {theme['id']:4} {theme['text'][:34]:34}"]
            for variante in args.variantes:
                ranked = rankings.get(variante, [])
                n8, r20 = ndcg(ranked, gains, 8), recall(ranked, gains, 20)
                per_variant[variante]["ndcg8"].append(n8)
                per_variant[variante]["recall20"].append(r20)
                line.append(f"{variante} {n8:.2f}/{r20:.2f}")
            print("  ".join(line))
        summary = {v: {"ndcg8": round(sum(m["ndcg8"]) / len(m["ndcg8"]), 3) if m["ndcg8"] else None,
                       "recall20": round(sum(m["recall20"]) / len(m["recall20"]), 3) if m["recall20"] else None,
                       "temas": len(m["ndcg8"])} for v, m in per_variant.items()}
        results["museums"][museum] = summary
        print(f"{museum}: {evaluated} temas anotados")
        for variante, metrics in summary.items():
            print(f"  {variante:10} nDCG@8 {metrics['ndcg8']}  recall@20 {metrics['recall20']}")
    out = TEMAS.parent / "resultados.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"escrito {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("pool", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--museus", nargs="+", default=["mnaz", "mnaa"])
        p.add_argument("--temas", nargs="*", default=None)
        p.add_argument("--variantes", nargs="+", default=list(VARIANTES), choices=VARIANTES)
        if name == "eval":
            p.add_argument("--anotacoes", help="CSV com colunas tema,museu,stop_id,relevancia (o pool preenchido)")
    args = parser.parse_args()
    return asyncio.run(cmd_pool(args) if args.cmd == "pool" else cmd_eval(args))


if __name__ == "__main__":
    sys.exit(main())
