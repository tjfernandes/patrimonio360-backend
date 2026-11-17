"""Listagens do acervo (CHAT_CATALOGUE_LISTINGS): «que autores há?», «que tipos de peças tem o museu?».

O modelo de pré-processamento aqui é um guião; a redação dos dois prompts mede-se
com o modelo real (tests/listing_request_cases.py).

    python -m unittest tests.test_catalogue_listings
"""
from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from app.core.config import Settings
from app.prompts.query_planner_prompts import build_listing_guard_prompt, build_listing_request_prompt
from app.services.catalogue_analytics import LISTING_FACETS, mentions_tour_scope
from app.services.opensearch_client import OpenSearchGateway
from tests.test_tour_location import _Base, _RouterCalled


class _Preprocess:
    """Responde às duas perguntas das listagens pelo que o teste dita."""

    def __init__(self, resposta: str = "A", nomeia: str = "B") -> None:
        self.resposta, self.nomeia, self.asked = resposta, nomeia, []

    async def generate(self, *, message, **_kwargs):
        if "Que resposta espera o visitante?" in message:
            self.asked.append("resposta")
            return SimpleNamespace(parsed_json={"resposta": self.resposta})
        if "A mensagem nomeia alguma coisa?" in message:
            self.asked.append("nomeia")
            return SimpleNamespace(parsed_json={"nomeia": self.nomeia})
        raise _RouterCalled("outra chamada ao modelo")


class _ListingGateway:
    VALUES = {
        "creator_ids": [("Gonçalves, Nuno", 12), ("Lopes, Gregório", 9)],
        # o próprio museu assina réplicas: «museu» não pode contar como nome de autor
        "creators": [("museu nacional do traje", 299), ("goncalves, nuno", 12), ("lopes, gregorio", 9)],
        "category": [("Pintura", 1465), ("Escultura", 299), ("Ourivesaria", 294)],
        "support_or_material": [("Madeira de carvalho", 174), ("Tela", 168)],
        "technique": [("Pintura a óleo", 254), ("Buril", 113), ("tafeta de seda", 9), ("cetim de seda", 8), ("veludo de seda", 7),
                      ("damasco de seda", 6), ("sarja de seda", 5)],
        "production_center": [("Portugal", 238), ("Japão", 12), ("Países Baixos", 4)],
    }

    def __init__(self) -> None:
        self.facet_calls: list[dict] = []
        self.artifact_calls: list[dict] = []

    async def aggregate_facet(self, **kwargs):
        self.facet_calls.append(kwargs)
        values = list(self.VALUES[kwargs["field"]])
        if kwargs.get("labels", True) and kwargs["field"] == "technique":
            values = values[:2]   # o que se mostra; o vocabulário (labels=False) vê todos
        return {"values": values, "distinct": 451 if kwargs["field"] == "creator_ids" else len(values), "filled": 2014, "total": 3658, "query_body": {}}

    async def aggregate_artifacts(self, **kwargs):
        self.artifact_calls.append(kwargs)
        return {"docs": [], "total_all": 100, "total_dated": 90, "total_attributed": 0, "centuries": [(1701, 30), (1801, 60)], "creators": [], "query_body": {}}

    async def fetch_images_by_artifact_ids(self, **_kwargs):
        return []

    async def fetch_artifacts_by_inventory_numbers(self, **_kwargs):
        return []


class TestListings(_Base):
    def _service_with(self, preprocess: _Preprocess, **flags):
        settings = self._settings(**{"CHAT_TOUR_LOCATION_ANSWERS": True, "CHAT_CATALOGUE_ANALYTICS": True, "CHAT_DETERMINISTIC_SMALLTALK": True,
                                     "CHAT_CATALOGUE_LISTINGS": True, **flags})
        service = self._service(settings)      # _FailingLLM: o LLM do texto final nunca pode ser chamado
        service._preprocess_llm_service = preprocess
        service.opensearch_gateway = _ListingGateway()
        return service

    def _ask(self, service, message: str, **kwargs):
        return asyncio.run(service.handle_message(self._request(message, **kwargs)))

    def test_off_by_default_the_model_is_not_asked(self) -> None:
        self.assertFalse(Settings(_env_file=None).CHAT_CATALOGUE_LISTINGS)
        preprocess = _Preprocess("B")
        service = self._service_with(preprocess, CHAT_CATALOGUE_LISTINGS=False)
        with self.assertRaises(_RouterCalled):
            self._ask(service, "Que autores há neste museu?")
        self.assertEqual(preprocess.asked, [])

    def test_creators(self) -> None:
        preprocess = _Preprocess("B")
        service = self._service_with(preprocess)
        response = self._ask(service, "Que autores há neste museu?")
        self.assertEqual(response.reply, "Nas fichas do museu há 451 autores identificados; 2014 das 3658 peças têm autor indicado. "
                                         "Os 2 com mais peças: Gonçalves, Nuno (12), Lopes, Gregório (9).")
        self.assertEqual(response.artifact_results, [])
        self.assertEqual(preprocess.asked, ["resposta", "nomeia"])
        call = service.opensearch_gateway.facet_calls[-1]
        self.assertEqual((call["field"], call["in_tour"], call["category"]), ("creator_ids", False, None))

    def test_categories_in_the_tour(self) -> None:
        service = self._service_with(_Preprocess("C"))
        response = self._ask(service, "Que tipos de peças há nesta visita?")
        self.assertEqual(response.reply, "As peças desta visita virtual dividem-se por 3 categorias: Pintura (1465), Escultura (299), Ourivesaria (294).")
        self.assertTrue(service.opensearch_gateway.facet_calls[-1]["in_tour"])

    def test_materials_techniques_and_places(self) -> None:
        for letra, mensagem, comeco in (
            ("D", "Que materiais existem?", "O material ou suporte está indicado em 2014 das 3658 fichas do museu, pelas palavras do catálogo. Os 2 mais frequentes: Madeira de carvalho (174), Tela (168)."),
            ("E", "Que técnicas existem?", "A técnica está indicada em 2014 das 3658 fichas do museu, pelas palavras do catálogo (2 formas diferentes). As 2 mais frequentes: Pintura a óleo (254), Buril (113)."),
            # «países» não é o valor «Países Baixos»: só a frase inteira conta
            ("F", "De que países são as peças deste museu?", "O local de fabrico está indicado em 2014 das 3658 fichas do museu. Os 3 mais frequentes: Portugal (238), Japão (12), Países Baixos (4)."),
        ):
            response = self._ask(self._service_with(_Preprocess(letra)), mensagem)
            self.assertEqual(response.reply, comeco)

    def test_styles_are_not_in_the_records(self) -> None:
        service = self._service_with(_Preprocess("H"))
        response = self._ask(service, "Que estilos existem neste museu?")
        self.assertIn("não registam o estilo das peças", response.reply)
        self.assertEqual(service.opensearch_gateway.facet_calls, [])

    def test_centuries_use_the_analytics_count(self) -> None:
        service = self._service_with(_Preprocess("G"))
        response = self._ask(service, "Que épocas estão representadas?")
        self.assertIn("Por século: XVIII 30, XIX 60.", response.reply)
        self.assertTrue(service.opensearch_gateway.artifact_calls[-1]["centuries"])

    def test_a_category_named_in_the_message_limits_the_listing(self) -> None:
        service = self._service_with(_Preprocess("B"))
        response = self._ask(service, "que autores de pintura há?")
        self.assertTrue(response.reply.startswith("Nas fichas de Pintura do museu há 451 autores identificados"))
        self.assertEqual(service.opensearch_gateway.facet_calls[-1]["category"], "pintura")

    def test_the_guard_sends_named_things_to_the_search(self) -> None:
        preprocess = _Preprocess("D", nomeia="A")
        service = self._service_with(preprocess)
        with self.assertRaises(_RouterCalled):
            self._ask(service, "Há esculturas em madeira?")
        self.assertEqual(preprocess.asked, ["resposta", "nomeia"])
        self.assertEqual(service.opensearch_gateway.facet_calls, [])

    def test_pieces_are_not_listings_and_the_guard_is_not_asked(self) -> None:
        preprocess = _Preprocess("A")
        with self.assertRaises(_RouterCalled):
            self._ask(self._service_with(preprocess), "Que obras de Gregório Lopes há?")
        self.assertEqual(preprocess.asked, ["resposta"])

    def test_a_value_of_the_list_in_the_message_goes_to_the_search(self) -> None:
        # o modelo deixou passar; a mensagem diz um dos locais que a lista teria
        with self.assertRaises(_RouterCalled):
            self._ask(self._service_with(_Preprocess("F")), "Que peças vieram do Japão?")
        with self.assertRaises(_RouterCalled):
            self._ask(self._service_with(_Preprocess("D")), "Tens esculturas de madeira?")
        # «seda» não é a primeira palavra de nenhum valor, mas aparece em cinco técnicas
        with self.assertRaises(_RouterCalled):
            self._ask(self._service_with(_Preprocess("D")), "Que peças de seda há?")
        with self.assertRaises(_RouterCalled):
            self._ask(self._service_with(_Preprocess("B")), "Há obras de Gregório Lopes?")

    def test_a_century_in_the_message_goes_to_the_search(self) -> None:
        service = self._service_with(_Preprocess("G"))
        with self.assertRaises(_RouterCalled):
            self._ask(service, "Tens peças do século XIX?")
        self.assertEqual(service.opensearch_gateway.artifact_calls, [])

    def test_an_answer_outside_the_letters_goes_to_the_search(self) -> None:
        for resposta in ("I", "", "Z"):
            with self.assertRaises(_RouterCalled):
                self._ask(self._service_with(_Preprocess(resposta)), "Que autores há neste museu?")

    def test_earlier_fast_paths_come_first(self) -> None:
        preprocess = _Preprocess("B")
        service = self._service_with(preprocess)
        self._ask(service, "que autor tem mais obras no museu?")   # analítica por regras
        self.assertEqual(preprocess.asked, [])


class _FacetClient:
    def __init__(self) -> None:
        self.bodies: list[dict] = []

    def search(self, *, index, body):
        self.bodies.append(body)
        field = body["aggs"]["valores"]["terms"]["field"]
        if field == "creator_ids":
            buckets = [{"key": "raiz:autor:1", "doc_count": 12, "ficha": {"hits": {"hits": [
                {"_source": {"creator_ids": ["raiz:autor:9", "raiz:autor:1"], "creators": ["Outro", "Gonçalves, Nuno"]}}]}}}]
        else:
            buckets = [
                {"key": "tela", "doc_count": 99, "ficha": {"hits": {"hits": [{"_source": {field: "Tela"}}]}}},
                {"key": "tela.", "doc_count": 69, "ficha": {"hits": {"hits": [{"_source": {field: "Tela."}}]}}},
                {"key": "", "doc_count": 1449, "ficha": {"hits": {"hits": [{"_source": {field: ""}}]}}},
                {"key": "ceramica", "doc_count": 45, "ficha": {"hits": {"hits": [{"_source": {field: "Cerâmica"}}]}}},
            ]
        return {"hits": {"total": {"value": 3658}, "hits": []},
                "aggregations": {"valores": {"buckets": buckets}, "distintos": {"value": 120}, "preenchidas": {"doc_count": 2209}}}


class TestGateway(unittest.TestCase):
    def _gateway(self) -> tuple[OpenSearchGateway, _FacetClient]:
        gateway = OpenSearchGateway(Settings(_env_file=None, OPENSEARCH_INDEX_ARTIFACT="arts"))
        fake = _FacetClient()
        gateway._client = fake
        return gateway, fake

    def test_values_keep_the_catalogue_spelling_and_merge_trailing_punctuation(self) -> None:
        gateway, fake = self._gateway()
        result = asyncio.run(gateway.aggregate_facet(museum_slug="mnaa", museum_id="mnaa", field="support_or_material", in_tour=True, category="pintura"))
        self.assertEqual(result["values"], [("Tela", 168), ("Cerâmica", 45)])
        self.assertEqual((result["distinct"], result["filled"], result["total"]), (120, 2209, 3658))
        filters = fake.bodies[0]["query"]["bool"]["filter"]
        self.assertIn({"term": {"museum_id": "mnaa"}}, filters)
        self.assertIn({"term": {"in_tour": True}}, filters)
        self.assertIn({"term": {"category": "pintura"}}, filters)
        self.assertEqual(fake.bodies[0]["size"], 0)

    def test_creators_come_back_by_name(self) -> None:
        gateway, _fake = self._gateway()
        result = asyncio.run(gateway.aggregate_facet(museum_slug="mnaa", museum_id="mnaa", field="creator_ids"))
        self.assertEqual(result["values"], [("Gonçalves, Nuno", 12)])

    def test_without_labels_the_keys_come_back_and_no_record_is_fetched(self) -> None:
        gateway, fake = self._gateway()
        result = asyncio.run(gateway.aggregate_facet(museum_slug="mnaa", museum_id="mnaa", field="technique", size=3000, labels=False))
        self.assertEqual(result["values"], [("tela", 168), ("ceramica", 45)])
        self.assertNotIn("aggs", fake.bodies[0]["aggs"]["valores"])
        self.assertEqual(fake.bodies[0]["aggs"]["valores"]["terms"], {"field": "technique", "size": 3000})


class TestMeasuredWording(unittest.TestCase):
    """A redação foi medida; estes testes avisam quem lhe mexer."""

    def test_request_prompt(self) -> None:
        prompt = build_listing_request_prompt(message="Que autores há?")
        # sem o nome do museu: com ele, a decisão do modelo mudava de museu para museu
        self.assertTrue(prompt.startswith("Mensagem de um visitante do museu:\n«Que autores há?»\n\nQue resposta espera o visitante?\n"))
        self.assertIn("A) peças do museu (uma ou várias), ou informação sobre uma peça\n", prompt)
        self.assertIn("F) os locais ou países de onde as peças do museu vêm\n", prompt)
        self.assertTrue(prompt.endswith('I) outra coisa\nResponde só com JSON: {"resposta": "A" a "I"}'))
        self.assertEqual(sorted(LISTING_FACETS), list("BCDEFGH"))

    def test_guard_prompt(self) -> None:
        prompt = build_listing_guard_prompt(message="Há esculturas em madeira?")
        self.assertTrue(prompt.startswith("Mensagem de um visitante do museu:\n«Há esculturas em madeira?»\n\n"))
        self.assertIn("A mensagem nomeia alguma coisa?\n", prompt)
        self.assertIn("B) não: só pergunta quais existem (que autores, que tipos de peças, que materiais, que estilos…)\n", prompt)
        self.assertTrue(prompt.endswith('Responde só com JSON: {"nomeia": "A", "B" ou "C"}'))

    def test_tour_scope(self) -> None:
        self.assertTrue(mentions_tour_scope("Que tipos de peças há nesta visita?"))
        self.assertFalse(mentions_tour_scope("Que tipos de peças tem o museu?"))


if __name__ == "__main__":
    unittest.main()
