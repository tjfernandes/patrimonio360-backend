"""Percursos temáticos (CHAT_ROUTE_PLANNING): grafo, tema, planeador, resposta do chat.

    python -m unittest tests.test_route_planning
"""

from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path

from app.schemas.chat import ChatMessageRequest, RoutePlan
from app.services.route_planning import planner
from app.services.route_planning.graph import TourGraph
from app.services.route_planning.candidates import Candidate, explain
from app.services.route_planning.store import RouteStore
from app.services.route_planning.theme import ThemeFacets, is_route_request, merge_llm_facets, rule_facets, strip_request_words, theme_title, unexplained_words
from app.services.route_planning import service as route_service
from tests.test_tour_location import _Base

VOCAB = {
    "colors": ["azul", "branco", "amarelo", "verde", "dourado"],
    "pigment_hints": ["azul-cobalto", "cobalto=azul-cobalto", "manganês"],
    "motifs": ["vegetalista", "geométrico", "animais", "flores"],
    "iconography": ["cena de caça", "religiosa", "mitológica"],
    "materials_visual": ["cerâmica vidrada", "madeira", "prata"],
    "techniques_visual": ["pintura sob vidrado", "talha dourada"],
    "style_period_visual": ["barroco", "rococó", "art déco"],
}

# Visita de teste: três salas em linha (A: p1,p2; B: p3; C: p4,p5), ligações só entre vizinhos.
GRAPH = {
    "museum_slug": "mnt",
    "nodes": [
        {"panorama_key": "p1", "playlist_index": 0, "room": "Sala A"},
        {"panorama_key": "p2", "playlist_index": 1, "room": "Sala A"},
        {"panorama_key": "p3", "playlist_index": 2, "room": "Sala B"},
        {"panorama_key": "p4", "playlist_index": 3, "room": "Sala C"},
        {"panorama_key": "p5", "playlist_index": 4, "room": "Sala C"},
        {"panorama_key": "p9", "playlist_index": 5, "room": "Anexo"},
    ],
    "edges": [
        {"source": "p1", "target": "p2", "kinds": ["adjacent"]}, {"source": "p2", "target": "p1", "kinds": ["adjacent"]},
        {"source": "p2", "target": "p3", "kinds": ["adjacent"]}, {"source": "p3", "target": "p2", "kinds": ["adjacent"]},
        {"source": "p3", "target": "p4", "kinds": ["adjacent"]}, {"source": "p4", "target": "p3", "kinds": ["adjacent"]},
        {"source": "p4", "target": "p5", "kinds": ["adjacent"]},  # só num sentido
    ],
}


def _stop(artifact: str, pano: str, room: str, *, kind: str = "hotspot", century: int = 17, category: str = "Cerâmica", colors=("azul", "branco"), look_for=("o cão no canto",)):
    placement = {"kind": kind, "overlay_id": f"ov-{artifact}", "yaw": 10.0, "pitch": -2.0} if kind == "hotspot" else {"kind": "object", "object_id": f"{pano}:1", "yaw": 33.3, "pitch": -4.4, "placement_score": 0.87}
    return {
        "stop_id": f"mnt:{artifact}@{pano}", "museum_slug": "mnt", "artifact_id": artifact, "inventory_number": artifact.upper(), "title": f"Peça {artifact}",
        "panorama_key": pano, "room": room, "playlist_index": int(pano[1]) - 1, "best_placement": True, "placement": placement,
        "catalogue": {"category": category, "start_year": (century - 1) * 100 + 10, "end_year": (century - 1) * 100 + 60, "start_century": century, "end_century": century,
                      "date_or_period": f"séc. {century}", "description": "Painel azul e branco", "image_count": 1},
        "visual": {"color_names": list(colors), "motifs": ["vegetalista"], "look_for": list(look_for)},
        "pigment_names": ["azul-cobalto"] if "azul" in colors else [],
    }


class TestGraph(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = TourGraph(GRAPH)

    def test_hops_and_rooms(self) -> None:
        self.assertEqual(self.graph.hops("p1", "p4"), 3)
        self.assertEqual(self.graph.hops("p1", "p1"), 0)
        self.assertEqual(self.graph.rooms_crossed("p1", "p4"), 2)
        self.assertEqual(self.graph.rooms_crossed("p1", "p2"), 0)
        # p5 só se chega num sentido: o caminho de volta usa o grafo sem sentido
        self.assertEqual(self.graph.hops("p5", "p4"), 1)
        # nó sem ligações: inalcançável
        self.assertIsNone(self.graph.hops("p1", "p9"))

    def test_transition_text(self) -> None:
        self.assertEqual(self.graph.transition_text("p1", "p1"), "Aqui mesmo")
        self.assertEqual(self.graph.transition_text("p1", "p2"), "Mesmo ao lado")
        self.assertEqual(self.graph.transition_text("p1", "p3"), "Na sala seguinte")
        self.assertEqual(self.graph.transition_text("p1", "p4"), "Duas salas adiante")
        self.assertEqual(self.graph.transition_text("p1", "p4", language="en"), "Two rooms ahead")
        self.assertEqual(self.graph.transition_text(None, "p4"), "Em Sala C")
        self.assertEqual(self.graph.transition_text("p1", "p9"), "Noutra zona da visita (Anexo)")


class TestTheme(unittest.TestCase):
    def test_request_detection(self) -> None:
        for text in ["Faz-me um percurso sobre azul-cobalto no século XVII", "Quero um roteiro de pintura religiosa", "Leva-me a ver as peças douradas",
                     "mostra-me os têxteis pela visita", "Plan a route about animals", "Can you give me a guided tour of the baroque pieces?", "itinerário barroco"]:
            self.assertTrue(is_route_request(text), text)
        for text in ["Onde estou?", "Que peças há nesta sala?", "mostra-me azulejos", "vestidos do século XVIII", "ver na tour"]:
            self.assertFalse(is_route_request(text), text)

    def test_rule_facets(self) -> None:
        facets = rule_facets("Faz-me um percurso de 5 paragens sobre azul-cobalto e animais no século XVII", vocabulary=VOCAB)
        self.assertEqual(facets.pigments, ["azul-cobalto"])
        self.assertEqual(facets.colors, [])  # «azul» dentro de azul-cobalto não conta duas vezes
        self.assertEqual(facets.motifs, ["animais"])
        self.assertEqual(facets.centuries, [17])
        self.assertEqual(facets.stops_requested, 5)
        self.assertEqual(facets.year_window(25), (1575, 1724))
        self.assertTrue(facets.is_temporal() and facets.has_visual())
        labels = [(f["kind"], f["label"]) for f in facets.labels()]
        self.assertIn(("pigment", "azul-cobalto"), labels)
        self.assertIn(("century", "século XVII"), labels)
        self.assertEqual(facets.free_text, "azul-cobalto animais século XVII")

    def test_centuries_variants(self) -> None:
        self.assertEqual(rule_facets("obras dos séculos XVI a XVIII", vocabulary=VOCAB).centuries, [16, 17, 18])
        self.assertEqual(rule_facets("peças do século 18", vocabulary=VOCAB).centuries, [18])
        self.assertEqual(rule_facets("percurso seiscentista", vocabulary=VOCAB).centuries, [17])
        facets = rule_facets("um percurso de meia hora sobre flores", vocabulary=VOCAB)
        self.assertEqual(facets.time_budget_min, 30)
        self.assertEqual(facets.motifs, ["flores"])
        self.assertEqual(rule_facets("entre 1600 e 1650", vocabulary=VOCAB).year_range, (1600, 1650))

    def test_strip_request_words(self) -> None:
        self.assertEqual(strip_request_words("Faz-me um percurso sobre azul-cobalto no século XVII"), "azul-cobalto século XVII")
        self.assertEqual(strip_request_words("Quero um roteiro de pintura religiosa"), "pintura religiosa")

    def test_categories_from_request(self) -> None:
        facets = rule_facets("Leva-me a ver pintura flamenga do século XVI", vocabulary=VOCAB)
        self.assertEqual(facets.categories, ["Pintura"])
        self.assertEqual(facets.centuries, [16])
        self.assertEqual([f["label"] for f in facets.labels()], ["século XVI", "Pintura"])
        self.assertEqual(rule_facets("Quero ver esculturas e móveis", vocabulary=VOCAB).categories, ["Escultura", "Mobiliário"])
        self.assertEqual(rule_facets("Faz-me um percurso sobre azul-cobalto", vocabulary=VOCAB).categories, [])
        self.assertEqual(unexplained_words("Faz-me um percurso de pinturas do século XVI", vocabulary=VOCAB), [])

    def test_duration_phrases_leave_the_theme(self) -> None:
        facets = rule_facets("Quero um percurso de meia hora sobre retratos", vocabulary=VOCAB)
        self.assertEqual(facets.time_budget_min, 30)
        self.assertEqual(facets.free_text, "retratos")
        self.assertEqual(strip_request_words("Faz-me um percurso de uma hora sobre prata"), "prata")

    def test_llm_paraphrase_only_titles(self) -> None:
        message = "Faz-me um percurso sobre astronautas"
        facets = rule_facets(message, vocabulary=VOCAB)
        merged = merge_llm_facets(facets, {"theme_text": "Astronautas e a exploração espacial", "facets": {}}, vocabulary=VOCAB, message=message)
        self.assertEqual(merged.free_text, "astronautas")                       # a pesquisa fica com as palavras do visitante
        self.assertEqual(theme_title(merged), "Astronautas e a exploração espacial")

    def test_unexplained_words(self) -> None:
        # tudo explicado pelas regras: o LLM não é chamado
        self.assertEqual(unexplained_words("Faz-me um percurso sobre azul-cobalto no século XVII", vocabulary=VOCAB), [])
        self.assertEqual(unexplained_words("Faz-me um percurso pelos tesouros nacionais", vocabulary=VOCAB), [])
        self.assertEqual(unexplained_words("Quero um percurso de 20 minutos entre 1600 e 1700 sobre flores", vocabulary=VOCAB), [])
        # palavras leigas ou fora do vocabulário: vale a pena perguntar ao LLM
        self.assertEqual(unexplained_words("Leva-me a ver azulejos com bichos", vocabulary=VOCAB), ["azulejos", "bichos"])
        self.assertEqual(unexplained_words("Faz-me um percurso de 15 minutos sobre vestidos com flores", vocabulary=VOCAB), ["vestidos"])

    def test_merge_llm_only_vocabulary(self) -> None:
        facets = rule_facets("percurso sobre louça", vocabulary=VOCAB)
        merged = merge_llm_facets(facets, {"theme_text": "louça azul", "facets": {"colors": ["Azul", "púrpura"], "centuries": [17, 99], "styles": ["barroco"]}, "stops_requested": 4}, vocabulary=VOCAB)
        self.assertEqual(merged.colors, ["azul"])        # «púrpura» não está no vocabulário
        self.assertEqual(merged.centuries, [17])          # 99 fora
        self.assertEqual(merged.styles, ["barroco"])
        self.assertEqual(merged.stops_requested, 4)
        self.assertEqual(merged.llm_theme_text, "louça azul")
        self.assertEqual(merged.free_text, "louça")                      # a pesquisa fica com as palavras do visitante
        self.assertEqual(merged.source, "rules+llm")

    def test_merge_llm_grounded_in_message(self) -> None:
        # com o pedido em mão e regras com resultado, o LLM só acrescenta o que o visitante disse:
        # «azul» está em «azul-cobalto»; geométrico, animais, barroco e o século XVIII caem
        message = "Faz-me um percurso sobre azul-cobalto no século XVII"
        facets = rule_facets(message, vocabulary=VOCAB)
        merged = merge_llm_facets(facets, {"facets": {"colors": ["azul", "branco"], "motifs": ["geométrico", "animais"], "styles": ["barroco"], "centuries": [18]}}, vocabulary=VOCAB, message=message)
        self.assertEqual(merged.colors, ["azul"])
        self.assertEqual(merged.pigments, ["azul-cobalto"])
        self.assertEqual(merged.motifs, [])
        self.assertEqual(merged.styles, [])
        self.assertEqual(merged.centuries, [17])

    def test_merge_llm_nothing_anchored_adds_nothing(self) -> None:
        # «astronautas» não ancora nada: sem facetas, o percurso acaba em «não há» em vez de estrelas
        message = "Faz-me um percurso sobre astronautas"
        facets = rule_facets(message, vocabulary=VOCAB)
        merged = merge_llm_facets(facets, {"theme_text": "astronautas", "facets": {"motifs": ["geométrico"], "materials": ["madeira"]}}, vocabulary=VOCAB, message=message)
        self.assertFalse(merged.has_visual())
        self.assertEqual(merged.free_text, "astronautas")

    def test_merge_llm_lay_phrasing_with_budget(self) -> None:
        # com orçamento (desligado por omissão), a tradução entra: no máximo dois valores
        # sem apoio por faceta, três no total, e sem inventar época quando o pedido não fala dela
        message = "Leva-me a ver coisas com bichos"
        facets = rule_facets(message, vocabulary=VOCAB)
        self.assertFalse(facets.has_visual())
        merged = merge_llm_facets(
            facets,
            {"facets": {"colors": ["azul", "branco"], "motifs": ["animais", "flores", "geométrico"], "iconography": ["cena de caça"], "centuries": [17]}},
            vocabulary=VOCAB, message=message, ungrounded_budget=3,
        )
        self.assertEqual(merged.motifs, ["animais", "flores"])      # 2 por faceta
        self.assertEqual(merged.iconography, ["cena de caça"])      # 3 no total…
        self.assertEqual(merged.colors, [])                         # …e as cores ficam para o fim
        self.assertEqual(merged.centuries, [])

    def test_merge_llm_theme_text_anchors_lay_translation(self) -> None:
        # a paráfrase do LLM («azulejos com animais») ancora «animais»; o resto cai
        message = "Leva-me a ver azulejos com bichos"
        facets = rule_facets(message, vocabulary=VOCAB)
        merged = merge_llm_facets(
            facets,
            {"theme_text": "azulejos com animais", "facets": {"colors": ["azul", "branco"], "motifs": ["animais", "flores"], "iconography": ["cena de caça"]}},
            vocabulary=VOCAB, message=message,
        )
        self.assertEqual(merged.motifs, ["animais"])
        self.assertEqual(merged.colors, [])
        self.assertEqual(merged.iconography, [])
        self.assertEqual(merged.llm_theme_text, "azulejos com animais")

    def test_labels_without_duplicates(self) -> None:
        facets = ThemeFacets(colors=["dourado"], techniques=["dourado", "talha dourada"])
        self.assertEqual([f["label"] for f in facets.labels()], ["dourado", "talha dourada"])

    def test_merge_llm_stem_grounding_blocks_translations(self) -> None:
        # «douradas» ancora «dourado» e «talha dourada» não (falta «talha»); havendo algo ancorado,
        # as traduções do LLM (prata, religiosa) já não entram
        message = "Quero um percurso sobre coisas douradas"
        facets = rule_facets(message, vocabulary=VOCAB)
        merged = merge_llm_facets(facets, {"facets": {"colors": ["dourado"], "materials": ["prata"], "techniques": ["talha dourada"], "iconography": ["religiosa"]}}, vocabulary=VOCAB, message=message)
        self.assertEqual(merged.colors, ["dourado"])
        self.assertEqual(merged.materials, [])
        self.assertEqual(merged.techniques, [])
        self.assertEqual(merged.iconography, [])


class TestPlanner(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = TourGraph(GRAPH)
        self.settings = planner.PlannerSettings(min_candidates=3, min_stops=3, max_stops=10, default_stops=6)

    def _cands(self):
        return [
            {"artifact_id": "a", "stop_id": "s-a", "panorama_key": "p4", "room": "Sala C", "playlist_index": 3, "fusion_score": 0.030, "category": "Cerâmica", "start_century": 17},
            {"artifact_id": "b", "stop_id": "s-b", "panorama_key": "p1", "room": "Sala A", "playlist_index": 0, "fusion_score": 0.028, "category": "Cerâmica", "start_century": 17},
            {"artifact_id": "c", "stop_id": "s-c", "panorama_key": "p3", "room": "Sala B", "playlist_index": 2, "fusion_score": 0.026, "category": "Pintura", "start_century": 18},
            {"artifact_id": "d", "stop_id": "s-d", "panorama_key": "p4", "room": "Sala C", "playlist_index": 3, "fusion_score": 0.020, "category": "Cerâmica", "start_century": 17},
            {"artifact_id": "e", "stop_id": "s-e", "panorama_key": "p5", "room": "Sala C", "playlist_index": 4, "fusion_score": 0.005, "category": "Cerâmica", "start_century": 17},
        ]

    def test_choose_k(self) -> None:
        self.assertEqual(planner.choose_k(stops_requested=None, time_budget_min=None, settings=self.settings), 6)
        # sem pedido explícito, k cresce com os candidatos relevantes: round(n × 0,6) entre 8 e 20
        default = planner.PlannerSettings()
        self.assertEqual(planner.choose_k(stops_requested=None, time_budget_min=None, settings=default, relevant_count=5), 8)
        self.assertEqual(planner.choose_k(stops_requested=None, time_budget_min=None, settings=default, relevant_count=12), 8)
        self.assertEqual(planner.choose_k(stops_requested=None, time_budget_min=None, settings=default, relevant_count=20), 12)
        self.assertEqual(planner.choose_k(stops_requested=None, time_budget_min=None, settings=default, relevant_count=80), 20)
        # pedidos explícitos não mudam com os candidatos
        self.assertEqual(planner.choose_k(stops_requested=5, time_budget_min=None, settings=default, relevant_count=80), 5)
        self.assertEqual(planner.choose_k(stops_requested=None, time_budget_min=10, settings=default, relevant_count=80), 4)
        self.assertEqual(planner.choose_k(stops_requested=2, time_budget_min=None, settings=self.settings), 3)
        self.assertEqual(planner.choose_k(stops_requested=15, time_budget_min=None, settings=self.settings), 10)
        # por omissão o teto é 50: «uma hora» dá 24 paragens, 60 pedidas ficam em 50
        default = planner.PlannerSettings()
        self.assertEqual(default.max_stops, 50)
        self.assertEqual(planner.choose_k(stops_requested=None, time_budget_min=60, settings=default), 24)
        self.assertEqual(planner.choose_k(stops_requested=60, time_budget_min=None, settings=default), 50)

    def test_order_many_stops_follows_the_corridor(self) -> None:
        # 14 panoramas em linha (q0…q13): com mais de 10 paragens a ordem vem do vizinho
        # mais próximo + 2-opt e tem de ser a do corredor, a partir do visitante em q0
        keys = [f"q{i}" for i in range(14)]
        graph = TourGraph({
            "museum_slug": "mnt",
            "nodes": [{"panorama_key": k, "playlist_index": i, "room": "Corredor"} for i, k in enumerate(keys)],
            "edges": [e for a, b in zip(keys, keys[1:]) for e in ({"source": a, "target": b, "kinds": ["adjacent"]}, {"source": b, "target": a, "kinds": ["adjacent"]})],
        })
        shuffled = [keys[i] for i in (7, 2, 11, 0, 13, 5, 9, 1, 12, 4, 8, 3, 10, 6)]
        stops = [planner.PlannedStop(candidate={"artifact_id": k, "panorama_key": k, "playlist_index": int(k[1:])}) for k in shuffled]
        ordered = planner.order_stops(stops, graph=graph, start_panorama="q0")
        self.assertEqual([s.panorama_key for s in ordered], keys)
        self.assertEqual(planner.choose_k(stops_requested=None, time_budget_min=20, settings=self.settings), 8)

    def test_plan_orders_by_shortest_path_and_groups_panorama(self) -> None:
        stops, info = planner.plan(self._cands(), graph=self.graph, start_panorama="p1", settings=self.settings, stops_requested=None, time_budget_min=None, temporal=True, language="pt")
        self.assertIsNotNone(stops)
        assert stops is not None
        # «e» fica abaixo do limiar de relevância; a, b, c, d são relevantes; d junta-se a a (mesmo panorama)
        self.assertEqual(info["relevant_count"], 4)
        order = [s.candidate["artifact_id"] for s in stops]
        self.assertEqual(order, ["b", "c", "a"])  # p1 -> p3 -> p4, a partir de p1
        self.assertEqual([o["artifact_id"] for o in stops[2].also_here], ["d"])
        self.assertEqual([s.transition for s in stops], ["Aqui mesmo", "Na sala seguinte", "Na sala seguinte"])
        self.assertGreaterEqual(info["estimated_minutes"], 3 * 2)

    def test_plan_none_with_few_candidates(self) -> None:
        stops, info = planner.plan(self._cands()[:2], graph=self.graph, start_panorama="p1", settings=self.settings, stops_requested=None, time_budget_min=None, temporal=False, language="pt")
        self.assertIsNone(stops)
        self.assertEqual(info["relevant_count"], 2)

    def test_held_karp_is_optimal(self) -> None:
        dist = [[0, 1, 9, 9], [1, 0, 1, 9], [9, 1, 0, 1], [9, 9, 1, 0]]
        self.assertEqual(planner._held_karp(dist, [0, 9, 9, 9]), [0, 1, 2, 3])
        self.assertEqual(planner._held_karp(dist, [9, 9, 9, 0]), [3, 2, 1, 0])

    def test_without_graph_uses_playlist_order(self) -> None:
        stops, _ = planner.plan(self._cands(), graph=None, start_panorama=None, settings=self.settings, stops_requested=4, time_budget_min=None, temporal=False, language="pt")
        assert stops is not None
        self.assertEqual([s.candidate["artifact_id"] for s in stops], ["b", "c", "a"])  # d agrupado com a


class TestExplain(unittest.TestCase):
    def test_why(self) -> None:
        facets = rule_facets("percurso sobre azul-cobalto e animais no século XVII", vocabulary=VOCAB)
        stop = _stop("x", "p1", "Sala A")
        self.assertEqual(explain(stop, facets), ["azul-cobalto", "séc. XVII"])


class _FakeSearch:
    """Pesquisa de paragens em memória: lexical devolve todas por ordem; os outros ramos vazios."""

    def __init__(self, stops: list[dict]) -> None:
        self.stops = stops
        self.calls: list[str] = []

    def lexical(self, *, query_text, filters, size):
        self.calls.append("lexical")
        return [(s["artifact_id"], 10.0 - i, s) for i, s in enumerate(self.stops[:size])]

    def semantic(self, *, embedding, filters, size):
        self.calls.append("semantic")
        return [(s["artifact_id"], 0.9 - i * 0.01, s) for i, s in enumerate(reversed(self.stops[:size]))]

    def attributes(self, *, facets, filters, size):
        self.calls.append("attributes")
        return [(s["artifact_id"], 5.0, s) for s in self.stops if "azul" in (s.get("visual") or {}).get("color_names", [])][:size]

    def artifact_ids(self, *, filters, limit=5000):
        return [s["artifact_id"] for s in self.stops]

    def image(self, *, embedding, museum_id, artifact_ids, size):
        self.calls.append("image")
        return []

    def stops_by_artifact(self, *, museum_slug, artifact_ids):
        return {s["artifact_id"]: s for s in self.stops if s["artifact_id"] in artifact_ids}


class _Embeddings:
    async def embed_text(self, text):
        return [0.1] * 4

    async def embed_multimodal_text_query(self, text):
        return [0.2] * 4


class _ThemeLLM:
    def __init__(self, parsed):
        self.parsed = parsed
        self.calls = 0

    async def generate(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs

        class _R:
            parsed_json = self.parsed
            text = json.dumps(self.parsed)
            model = "fake"
        return _R()


class TestService(_Base):
    def setUp(self) -> None:
        super().setUp()
        (Path(self._tmp.name) / "tour-graph-mnt.json").write_text(json.dumps(GRAPH), encoding="utf-8")
        self.stops = [
            _stop("a", "p4", "Sala C"), _stop("b", "p1", "Sala A", kind="object"), _stop("c", "p3", "Sala B", category="Pintura", century=18, colors=("verde",)),
            _stop("d", "p4", "Sala C"), _stop("e", "p5", "Sala C", colors=("amarelo",)),
        ]

    def _route_service(self, *, enabled=True, llm=None, **overrides):
        values = {"CHAT_TOUR_LOCATION_ANSWERS": True, "CHAT_ROUTE_PLANNING": enabled, "ROUTE_THEME_LLM": llm is not None, "ROUTE_MIN_FUSION_SCORE": 0.0}
        values.update(overrides)
        settings = self._settings(**values)
        service = self._service(settings, llm=llm)
        service.embedding_provider = _Embeddings()
        service._route_search = _FakeSearch(self.stops)
        from app.services.route_planning.graph import TourGraphStore

        service._route_graph_store = TourGraphStore(self._tmp.name, check_interval=0.0)
        service._route_store = RouteStore(600)
        return service

    def test_flag_off_does_not_plan(self) -> None:
        service = self._route_service(enabled=False)
        # com a flag desligada o pedido segue o caminho normal: o router (LLM) é chamado e falha
        from tests.test_tour_location import _RouterCalled

        with self.assertRaises(_RouterCalled):
            asyncio.run(service.handle_message(self._request("Faz-me um percurso sobre azul-cobalto no século XVII")))

    def test_plan_route_response_contract(self) -> None:
        service = self._route_service()
        response = asyncio.run(service.handle_message(self._request(
            "Faz-me um percurso sobre azul-cobalto no século XVII", tour_location={"panorama_key": "p1", "room": "Sala A"})))
        self.assertIsNotNone(response.route)
        route = response.route
        assert route is not None
        self.assertTrue(route.route_id.startswith("route-"))
        self.assertEqual(route.theme.text, "azul-cobalto século XVII")
        self.assertIn({"kind": "pigment", "label": "azul-cobalto"}, [f.model_dump() for f in route.theme.facets])
        # «c» (séc. XVIII, verde) e «e» (amarelo) não batem nas facetas do tema, mas o lexical devolve-as: ficam atrás
        orders = [s.order for s in route.stops]
        self.assertEqual(orders, list(range(1, len(route.stops) + 1)))
        first = route.stops[0]
        self.assertEqual(first.navigation_target.panorama_key, "p1")
        self.assertEqual(first.navigation_target.source, "object")
        self.assertTrue(first.navigation_target.overlay_id.startswith("object:"))
        self.assertEqual(first.transition, "Aqui mesmo")
        self.assertEqual(first.why, ["azul-cobalto", "séc. XVII"])
        self.assertEqual(first.look_for, ["o cão no canto"])
        self.assertTrue(first.artifact.in_tour)
        last_c = next(s for s in route.stops if s.artifact.artifact_id == "a")
        self.assertEqual([f.artifact_id for f in last_c.also_here], ["d"])
        self.assertIn("Ver na tour", response.reply)
        self.assertEqual(len(response.artifact_results), len(route.stops))
        self.assertEqual(response.navigation_targets, [])
        events = [e for e in service.query_logger.events if e.get("event_type") == "route_planned"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["route_id"], route.route_id)
        self.assertEqual(events[0]["chosen_stop_ids"], [s.stop_id for s in route.stops])
        # ficou na cache, com o mesmo conteúdo
        stored = service._route_store.get(route.route_id)
        self.assertEqual(RoutePlan.model_validate(stored.plan).route_id, route.route_id)

    def test_explicit_intent_and_llm_theme(self) -> None:
        llm = _ThemeLLM({"theme_text": "faiança azul", "facets": {"colors": ["azul"]}, "stops_requested": 3})
        service = self._route_service(llm=llm)
        # «bichos» não está no vocabulário: o LLM é chamado para traduzir
        response = asyncio.run(service.handle_message(self._request("peças de faiança com bichos", metadata={"session_id": "s1", "intent": "plan_route"})))
        self.assertEqual(llm.calls, 1)
        # o JSON do tema não cabe nos 200 tokens do chat: vai com orçamento próprio
        self.assertEqual(llm.kwargs.get("max_tokens"), 600)
        assert response.route is not None
        self.assertEqual(response.route.title, "Faiança azul")
        self.assertEqual(response.route.theme.text, "faiança bichos")
        self.assertLessEqual(len(response.route.stops), 3)

    def test_llm_skipped_when_rules_explain_everything(self) -> None:
        llm = _ThemeLLM({"theme_text": "azul", "facets": {"colors": ["azul", "branco"], "motifs": ["geométrico"]}})
        service = self._route_service(llm=llm)
        response = asyncio.run(service.handle_message(self._request("Faz-me um percurso sobre azul-cobalto")))
        self.assertEqual(llm.calls, 0)
        assert response.route is not None
        self.assertEqual([f.label for f in response.route.theme.facets], ["azul-cobalto"])

    def test_stop_card_without_title_gets_fallback(self) -> None:
        service = self._route_service()
        for stop in self.stops:
            stop["title"] = None
            stop["catalogue"]["title"] = None
        response = asyncio.run(service.handle_message(self._request("Faz-me um percurso sobre azul-cobalto")))
        assert response.route is not None
        for stop in response.route.stops:
            self.assertTrue(stop.artifact.title)
            self.assertIn("inv.", stop.artifact.title)

    def test_no_route_when_few_candidates(self) -> None:
        service = self._route_service()
        service._route_search = _FakeSearch(self.stops[:2])
        response = asyncio.run(service.handle_message(self._request("Faz-me um percurso sobre astronautas")))
        self.assertIsNone(response.route)
        self.assertIn("2 peças", response.reply)
        self.assertIn("alargar", response.reply)
        self.assertEqual(response.artifact_results, [])
        events = [e for e in service.query_logger.events if e.get("event_type") == "route_planned"]
        self.assertEqual(events[0]["status"], "no_route")

    def test_adjust_shorter_and_swap(self) -> None:
        service = self._route_service()
        response = asyncio.run(service.handle_message(self._request("Faz-me um percurso sobre azul-cobalto", tour_location={"panorama_key": "p1"})))
        assert response.route is not None
        n = len(response.route.stops)
        shorter = asyncio.run(route_service.adjust_route(service, route_id=response.route.route_id, action="shorter"))
        assert shorter is not None
        self.assertLess(len(shorter["stops"]), n)
        self.assertNotEqual(shorter["route_id"], response.route.route_id)
        swapped = asyncio.run(route_service.adjust_route(service, route_id=response.route.route_id, action="swap_stop", stop_index=0))
        assert swapped is not None
        first_before = response.route.stops[0].artifact.artifact_id
        self.assertNotIn(first_before, [s["artifact"]["artifact_id"] for s in swapped["stops"]])
        only = asyncio.run(route_service.adjust_route(service, route_id=response.route.route_id, action="only_room", room="Sala C"))
        # só a Sala C tem a, d, e: três candidatos -> percurso de uma paragem com focos, ou None se o planeador não chegar aos 3
        if only is not None:
            self.assertTrue(all(s["room"] == "Sala C" for s in only["stops"]))
        self.assertIsNone(asyncio.run(route_service.adjust_route(service, route_id="route-nope", action="shorter")))

    def test_route_stop_prompt_builder(self) -> None:
        service = self._route_service()
        response = asyncio.run(service.handle_message(self._request("Faz-me um percurso sobre azul-cobalto no século XVII", tour_location={"panorama_key": "p1"})))
        assert response.route is not None
        metadata = {"route_stop": {"route_id": response.route.route_id, "stop_index": 0, "stop_id": response.route.stops[0].stop_id}}
        selected = {"artifact_id": response.route.stops[0].artifact.artifact_id}
        builder = route_service.route_stop_prompt_builder(service, metadata=metadata, selected=selected, language="pt", museum_name="Museu Nacional do Traje")
        assert builder is not None
        prompt = builder("[doc_1] Título: Peça b")
        self.assertIn("paragem 1 de", prompt)
        self.assertIn("azul-cobalto século XVII", prompt)
        self.assertIn("o cão no canto", prompt)
        self.assertIn("CARTÃO:", prompt)
        self.assertIsNone(route_service.route_stop_prompt_builder(service, metadata={"route_stop": {"route_id": "x"}}, selected=None, language="pt", museum_name=None))

    def test_describe_artifact_uses_route_prompt(self) -> None:
        from tests.test_tour_location import _WordingLLM

        llm = _WordingLLM("Este painel…")
        service = self._route_service(llm=llm, CHAT_DESCRIBE_ARTIFACT_INTENT=True, ROUTE_THEME_LLM=False)
        response = asyncio.run(service.handle_message(self._request("Faz-me um percurso sobre azul-cobalto", tour_location={"panorama_key": "p1"})))
        assert response.route is not None
        metadata = {"session_id": "s1", "intent": "describe_artifact", "selected_artifact": {"artifact_id": "a-5262", "inventory_number": "5262", "title": "Leque"},
                    "route_stop": {"route_id": response.route.route_id, "stop_index": 1}}
        answer = asyncio.run(service.handle_message(self._request("Leque (5262)", metadata=metadata)))
        self.assertEqual(answer.reply, "Este painel…")
        self.assertTrue(any("paragem 2 de" in p for p in llm.prompts), llm.prompts)
        narrated = [e for e in service.query_logger.events if e.get("event_type") == "route_stop_narrated"]
        self.assertEqual(len(narrated), 1)
        self.assertEqual(narrated[0]["stop_index"], 1)

    def test_route_stop_without_selected_artifact_narrates_instead_of_replanning(self) -> None:
        # o frontend reenviou o texto do pedido original só com metadata.route_stop (caso real de 10/11):
        # tem de sair a narração da paragem, não um segundo percurso
        from tests.test_tour_location import _WordingLLM

        llm = _WordingLLM("Repare neste painel…")
        service = self._route_service(llm=llm, CHAT_DESCRIBE_ARTIFACT_INTENT=True, ROUTE_THEME_LLM=False)
        response = asyncio.run(service.handle_message(self._request("Faz-me um percurso sobre azul-cobalto", tour_location={"panorama_key": "p1"})))
        assert response.route is not None
        first = response.route.stops[0]
        metadata = {"session_id": "s1", "route_stop": {"route_id": response.route.route_id, "stop_index": 0, "stop_id": first.stop_id}}
        answer = asyncio.run(service.handle_message(self._request("Faz-me um percurso sobre azul-cobalto", metadata=metadata)))
        self.assertIsNone(answer.route)
        # a peça «a» não existe no gateway falso, por isso a resposta é a de «ficha não encontrada»
        # do fluxo describe_artifact — o que importa é que não foi planeado um segundo percurso
        self.assertIn("ficha", answer.reply)
        planned = [e for e in service.query_logger.events if e.get("event_type") == "route_planned"]
        self.assertEqual(len(planned), 1)
        narrated = [e for e in service.query_logger.events if e.get("event_type") == "route_stop_narrated"]
        self.assertEqual(len(narrated), 1)

    def test_route_request_text_with_describe_intent_is_not_replanned(self) -> None:
        service = self._route_service(CHAT_DESCRIBE_ARTIFACT_INTENT=True)
        metadata = {"session_id": "s1", "intent": "describe_artifact", "selected_artifact": {"artifact_id": "a", "inventory_number": "A", "title": "Peça a"}}
        from app.services.route_planning import service as route_service
        result = asyncio.run(route_service.maybe_answer_route(service, self._request("Faz-me um percurso sobre azul-cobalto", metadata=metadata)))
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
