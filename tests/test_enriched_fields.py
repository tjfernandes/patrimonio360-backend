"""Contagens e listagens pelos campos enriquecidos (CHAT_ENRICHED_FIELDS) e as regras que os produzem.

    python -m unittest tests.test_enriched_fields
"""
from __future__ import annotations

import asyncio
import unittest

from app.core.config import Settings
from app.services import enriched_fields as E
from app.services.opensearch_client import OpenSearchGateway
from tests.test_catalogue_analytics import DOCS, TestFastPath, _AnalyticsGateway
from tools.enriquecimento import fotografia as F
from tools.enriquecimento import regras as R
from tools.enriquecimento import temas as T

VOCABULARY = {
    "enr_tipo": ["mobiliário", "cadeira", "escultura", "pintura", "têxtil", "tapeçaria", "ourivesaria", "painel"],
    "enr_materiais": ["prata", "prata dourada", "madeira", "marfim", "cerâmica"],
    "enr_tecnicas": ["tapeçaria", "pintura a óleo", "bordado"],
    "enr_funcao": ["serviço de mesa"],
    "enr_origem_pais": ["China", "Japão"],
}


class TestMatchSubject(unittest.TestCase):
    def test_one_word_plural_and_accents(self) -> None:
        self.assertEqual(E.match_subject("cadeiras", VOCABULARY), [{"enr_tipo": ["cadeira"]}])
        self.assertEqual(E.match_subject("mobiliario", VOCABULARY), [{"enr_tipo": ["mobiliário"]}])
        self.assertEqual(E.match_subject("texteis", VOCABULARY), [{"enr_tipo": ["têxtil"]}])
        self.assertEqual(E.match_subject("paineis", VOCABULARY), [{"enr_tipo": ["painel"]}])
        self.assertEqual(E.match_subject("japao", VOCABULARY), [{"enr_origem_pais": ["Japão"]}])

    def test_every_word_has_to_be_a_value(self) -> None:
        self.assertEqual(E.match_subject("esculturas madeira", VOCABULARY), [{"enr_tipo": ["escultura"]}, {"enr_materiais": ["madeira"]}])
        self.assertIsNone(E.match_subject("esculturas bonitas", VOCABULARY))   # «bonitas» não é valor de nenhuma lista
        self.assertIsNone(E.match_subject("vestidos noiva", VOCABULARY))
        self.assertIsNone(E.match_subject("", VOCABULARY))
        self.assertIsNone(E.match_subject("cadeiras", {}))

    def test_longest_phrase_first(self) -> None:
        self.assertEqual(E.match_subject("prata dourada", VOCABULARY), [{"enr_materiais": ["prata dourada"]}])
        self.assertEqual(E.match_subject("pintura a oleo", VOCABULARY), [{"enr_tecnicas": ["pintura a óleo"]}])
        # o assunto chega da analítica sem palavras de ligação e, muitas vezes, no plural
        self.assertEqual(E.match_subject("pinturas oleo", VOCABULARY), [{"enr_tecnicas": ["pintura a óleo"]}])
        self.assertEqual(E.match_subject("esculturas em madeira", VOCABULARY), [{"enr_tipo": ["escultura"]}, {"enr_materiais": ["madeira"]}])
        self.assertEqual(E.match_subject("servico de mesa", VOCABULARY), [{"enr_funcao": ["serviço de mesa"]}])

    def test_participles_and_words_that_must_not_be_bent(self) -> None:
        vocabulary = {"enr_tecnicas": ["bordado"], "enr_tipo": ["prato", "sobre madeira", "pintura"], "enr_materiais": ["prata", "madeira"]}
        self.assertEqual(E.match_subject("bordadas", vocabulary), [{"enr_tecnicas": ["bordado"]}])
        self.assertEqual(E.match_subject("prata", vocabulary), [{"enr_materiais": ["prata"]}])           # nunca «prato»
        self.assertEqual(E.match_subject("madeira", vocabulary), [{"enr_materiais": ["madeira"]}])       # nunca «sobre madeira»
        self.assertEqual(E.match_subject("pinturas sobre madeira", vocabulary), [{"enr_tipo": ["pintura"]}, {"enr_tipo": ["sobre madeira"]}])

    def test_themes_are_asked_by_their_everyday_words(self) -> None:
        vocabulary = {"enr_tipo": ["pintura"], "enr_temas_palavras": T.PALAVRAS["Religioso"] + T.PALAVRAS["Animais"] + T.PALAVRAS["Paisagem e natureza"]}
        self.assertEqual(E.match_subject("religiosas", vocabulary), [{"enr_temas_palavras": ["religiosa"]}])
        self.assertEqual(E.match_subject("pinturas animais", vocabulary), [{"enr_tipo": ["pintura"]}, {"enr_temas_palavras": ["animal"]}])
        self.assertEqual(E.match_subject("paisagens", vocabulary), [{"enr_temas_palavras": ["paisagem"]}])
        self.assertEqual(E.match_subject("arte sacra", vocabulary), [{"enr_temas_palavras": ["arte sacra"]}])

    def test_a_word_can_be_a_value_in_two_fields(self) -> None:
        self.assertEqual(E.match_subject("tapecarias", VOCABULARY), [{"enr_tipo": ["tapeçaria"], "enr_tecnicas": ["tapeçaria"]}])
        self.assertEqual(E.describe(E.match_subject("tapecarias", VOCABULARY)), "enr_tipo=tapeçaria ou enr_tecnicas=tapeçaria")


class _Indices:
    def __init__(self, exists: bool = True) -> None:
        self._exists = exists

    def exists(self, *, index):
        return self._exists


class _EnrichmentClient:
    def __init__(self, *, exists: bool = True) -> None:
        self.indices, self.calls = _Indices(exists), []

    def search(self, *, index, body):
        self.calls.append((index, body))
        if "aggs" not in body:
            return {"hits": {"total": {"value": 2}, "hits": [{"_id": "a1"}, {"_id": "a2"}]}}
        bucket = lambda key, n, field, label: {"key": key, "doc_count": n, "ficha": {"hits": {"hits": [{"_source": {field: [label, "outro"]}}]}}}
        aggregations = {}
        for name, definition in body["aggs"].items():
            if "terms" in definition:
                field = definition["terms"]["field"]
                aggregations[name] = {"buckets": [bucket("ceramica", 53, field, "cerâmica"), bucket("prata", 290, field, "prata")]}
        aggregations.update({"distintos": {"value": 58}, "preenchidas": {"doc_count": 3347}})
        return {"hits": {"total": {"value": 3658}, "hits": []}, "aggregations": aggregations}


class TestGateway(unittest.TestCase):
    def _gateway(self, **kwargs):
        gateway = OpenSearchGateway(Settings(_env_file=None, OPENSEARCH_INDEX_ENRICHMENT="enr"))
        gateway._client = _EnrichmentClient(**kwargs)
        return gateway, gateway._client

    def test_vocabulary_keeps_the_accents_and_survives_a_missing_index(self) -> None:
        gateway, client = self._gateway()
        vocabulary = asyncio.run(gateway.enriched_vocabulary(museum_id="mnaa", fields=("enr_materiais",)))
        self.assertEqual(vocabulary, {"enr_materiais": ["cerâmica", "prata"]})
        self.assertEqual(client.calls[0][0], "enr")
        self.assertEqual(client.calls[0][1]["query"], {"term": {"museum_id": "mnaa"}})
        gateway, client = self._gateway(exists=False)
        self.assertEqual(asyncio.run(gateway.enriched_vocabulary(museum_id="mnaa", fields=("enr_tipo",))), {"enr_tipo": []})
        self.assertEqual(client.calls, [])

    def test_ids_need_every_clause_and_any_field_of_each(self) -> None:
        gateway, client = self._gateway()
        clauses = [{"enr_tipo": ["escultura"]}, {"enr_tipo": ["tapeçaria"], "enr_tecnicas": ["tapeçaria"]}]
        self.assertEqual(asyncio.run(gateway.enriched_artifact_ids(museum_id="mnaa", clauses=clauses)), ["a1", "a2"])
        filters = client.calls[0][1]["query"]["bool"]["filter"]
        self.assertEqual(filters[0], {"term": {"museum_id": "mnaa"}})
        self.assertEqual(filters[1], {"bool": {"should": [{"terms": {"enr_tipo": ["escultura"]}}], "minimum_should_match": 1}})
        self.assertEqual(len(filters[2]["bool"]["should"]), 2)

    def test_facet_has_the_shape_of_the_listing(self) -> None:
        gateway, client = self._gateway()
        result = asyncio.run(gateway.aggregate_enriched_facet(museum_id="mnaa", field="enr_materiais", in_tour=True, category="escultura"))
        self.assertEqual(result["values"], [("prata", 290), ("cerâmica", 53)])
        self.assertEqual((result["distinct"], result["filled"], result["total"]), (58, 3347, 3658))
        filters = client.calls[0][1]["query"]["bool"]["filter"]
        self.assertIn({"term": {"in_tour": True}}, filters)
        self.assertIn({"term": {"category": "escultura"}}, filters)


class _EnrichedGateway(_AnalyticsGateway):
    def __init__(self, *args, vocabulary=None, ids=("a-9", "a-5262"), **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vocabulary, self.ids, self.enriched_calls = vocabulary if vocabulary is not None else VOCABULARY, list(ids), []

    async def enriched_vocabulary(self, *, museum_id, fields):
        self.enriched_calls.append(("vocabulary", museum_id))
        return {field: self.vocabulary.get(field, []) for field in fields}

    async def enriched_artifact_ids(self, *, museum_id, clauses):
        self.enriched_calls.append(("ids", museum_id, clauses))
        return list(self.ids)


class TestCountByEnrichedFields(TestFastPath):
    """Herda os testes da analítica: com a flag ligada, o que não é valor das listas fica exatamente como era."""

    def _service_with(self, gateway, **flags):
        return super()._service_with(gateway, **{"CHAT_ENRICHED_FIELDS": True, **flags})

    def test_count_reads_the_subject_as_an_object_type(self) -> None:
        gateway = _EnrichedGateway(DOCS, total_all=2, total_dated=2)
        service = self._service_with(gateway)
        response = asyncio.run(service.handle_message(self._request("quantas cadeiras há?")))
        call = gateway.calls[0]
        self.assertEqual(call["subject"], "")                                   # já não é pesquisa por palavras
        self.assertEqual(call["filters"]["artifact_id"], ["a-9", "a-5262"])     # são as peças que o enriquecimento designa
        self.assertEqual(gateway.enriched_calls[1][2], [{"enr_tipo": ["cadeira"]}])
        self.assertTrue(response.reply.startswith("Há 2 peças do museu que correspondem a «cadeiras»"), response.reply)

    def test_period_and_tour_still_apply(self) -> None:
        gateway = _EnrichedGateway(DOCS, total_all=1, total_dated=1)
        asyncio.run(self._service_with(gateway).handle_message(self._request("quantas esculturas em madeira do século XVIII há nesta visita?")))
        call = gateway.calls[0]
        self.assertEqual(gateway.enriched_calls[1][2], [{"enr_tipo": ["escultura"]}, {"enr_materiais": ["madeira"]}])
        self.assertEqual((call["subject"], call["in_tour"]), ("", True))
        self.assertEqual(call["filters"]["_temporal_interval"]["start_year"], 1700)

    def test_no_piece_counts_zero(self) -> None:
        gateway = _EnrichedGateway([], total_all=0, total_dated=0, ids=())
        asyncio.run(self._service_with(gateway).handle_message(self._request("quantas cadeiras há?")))
        self.assertEqual(gateway.calls[0]["filters"]["artifact_id"], ["(nenhuma)"])

    def test_a_word_outside_the_lists_keeps_the_old_count(self) -> None:
        gateway = _EnrichedGateway(DOCS, total_all=3, total_dated=3)
        asyncio.run(self._service_with(gateway).handle_message(self._request("quantas cadeiras bonitas há?")))
        self.assertEqual(gateway.calls[0]["subject"], "cadeiras bonitas")
        self.assertNotIn("artifact_id", gateway.calls[0]["filters"])
        self.assertEqual([c[0] for c in gateway.enriched_calls], ["vocabulary"])

    def test_museum_without_enrichment_keeps_the_old_count(self) -> None:
        gateway = _EnrichedGateway(DOCS, vocabulary={})
        asyncio.run(self._service_with(gateway).handle_message(self._request("quantas cadeiras há?")))
        self.assertEqual(gateway.calls[0]["subject"], "cadeiras")

    def test_flag_off_never_asks_the_enrichment(self) -> None:
        gateway = _EnrichedGateway(DOCS)
        asyncio.run(self._service_with(gateway, CHAT_ENRICHED_FIELDS=False).handle_message(self._request("quantas cadeiras há?")))
        self.assertEqual((gateway.calls[0]["subject"], gateway.enriched_calls), ("cadeiras", []))

    def test_a_failing_enrichment_keeps_the_old_count(self) -> None:
        class _Broken(_EnrichedGateway):
            async def enriched_artifact_ids(self, **_kwargs):
                raise RuntimeError("índice em baixo")

        gateway = _Broken(DOCS)
        asyncio.run(self._service_with(gateway).handle_message(self._request("quantas cadeiras há?")))
        self.assertEqual(gateway.calls[0]["subject"], "cadeiras")


def _ficha(**campos):
    return {"artifact_id": "x", "museum_id": "mnaa", "title": "", "category": "", "technique": "", "support_or_material": "",
            "production_center": "", "description": "", "origin_history": "", "dimensions_cm": "", **campos}


def _valores(lista, fonte=None):
    return {v["valor"] for v in lista if fonte is None or v["fonte"] == fonte}


class TestRegras(unittest.TestCase):
    """O que o piloto de 2026-11-24 ensinou, para não voltar a acontecer."""

    def test_material_from_the_technique_field(self) -> None:
        enr = R.enriquecer(_ficha(title="Leiteira", category="Ourivesaria", technique="Prata fundida, cinzelada e incisa"))
        self.assertEqual(_valores(enr["materiais"], "campo"), {"prata"})
        self.assertEqual(_valores(enr["tecnicas"], "campo"), {"fundição", "cinzelado", "incisão"})
        self.assertEqual(enr["tipo_objeto"]["valor"], "ourivesaria › leiteira")
        self.assertEqual(_valores(enr["funcao"]), {"serviço de mesa"})

    def test_parent_value_comes_with_the_child(self) -> None:
        enr = R.enriquecer(_ficha(title="São Pedro", category="Pintura", technique="Pintura a óleo", support_or_material="Madeira de carvalho"))
        self.assertEqual(_valores(enr["materiais"]), {"madeira › carvalho", "madeira"})
        self.assertEqual(R.documento(_ficha(), enr)["enr_materiais"], ["carvalho", "madeira"])
        self.assertEqual(enr["tipo_objeto"]["valor"], "pintura › sobre madeira")

    def test_gothic_letters_are_not_a_style(self) -> None:
        enr = R.enriquecer(_ficha(title="Fólio de Texto", category="Pintura", support_or_material="Velino", technique="Pintura a têmpera e ouro",
                                  description="O texto é escrito em latim com caracteres góticos."))
        self.assertEqual(enr["estilo"], [])
        self.assertEqual(enr["tipo_objeto"]["valor"], "manuscrito iluminado › fólio")
        self.assertIn("iluminura", _valores(enr["tecnicas"], "regra"))
        enr = R.enriquecer(_ficha(title="Cálice", category="Ourivesaria", description="Cálice gótico com nó facetado."))
        self.assertEqual(_valores(enr["estilo"]), {"gótico"})

    def test_the_rim_of_a_plate_is_not_a_colour(self) -> None:
        enr = R.enriquecer(_ficha(title="Prato", category="Cerâmica", description="Prato com bordo ondulado, decoração a azul sobre branco."))
        self.assertEqual(_valores(enr["cores"]), {"azul", "branco"})

    def test_a_painting_description_is_about_the_scene(self) -> None:
        enr = R.enriquecer(_ficha(title="Virgem", category="Pintura", technique="Pintura a óleo",
                                  description="A Virgem, com coroa de ouro e manto de seda bordado, senta-se num trono de pedra."))
        self.assertEqual(_valores(enr["materiais"]), set())
        self.assertEqual(_valores(enr["tecnicas"]), {"pintura a óleo"})

    def test_in_a_sculpture_only_the_opening_presents_the_piece(self) -> None:
        longa = "Escultura em madeira policromada. " + "A santa veste túnica comprida. " * 6 + "Está sentada numa pedra."
        enr = R.enriquecer(_ficha(title="Santa", category="Escultura", description=longa))
        self.assertEqual(_valores(enr["materiais"]), {"madeira"})

    def test_upholstery_is_not_the_sculpture_technique(self) -> None:
        enr = R.enriquecer(_ficha(title="Cadeira", category="Mobiliário", description="Cadeira com espaldar entalhado e assento estofado."))
        self.assertEqual(_valores(enr["tecnicas"]), {"talha"})
        self.assertEqual(enr["tipo_objeto"]["valor"], "mobiliário › cadeira")
        self.assertEqual(_valores(enr["funcao"]), {"assento"})

    def test_the_object_name_is_not_always_the_first_word(self) -> None:
        tipo = lambda titulo, categoria: R.enriquecer(_ficha(title=titulo, category=categoria))["tipo_objeto"]["valor"]
        self.assertEqual(tipo("Grande copo", "Vidros"), "vidro › copo")
        self.assertEqual(tipo("Par de castiçais", "Ourivesaria"), "ourivesaria › castiçal")
        self.assertEqual(tipo("Nossa Senhora", "Ourivesaria"), "ourivesaria")       # sem nome de objeto: fica a categoria

    def test_drawings_get_no_colours_from_the_description(self) -> None:
        enr = R.enriquecer(_ficha(title="Retrato", category="Desenho", technique="Desenho a carvão", description="Homem com lenço branco ao pescoço."))
        self.assertEqual(enr["cores"], [])

    def test_origin_and_size(self) -> None:
        enr = R.enriquecer(_ficha(title="Saleiro", category="Vidros", production_center="La Granja de St. Ildefonso", dimensions_cm="Alt. 2,3 x Larg. 10, 4 x Prof. 8,6 cm"))
        self.assertEqual([(o["pais"], o["regiao"], o["fonte"]) for o in enr["origem"]], [("Espanha", "La Granja de San Ildefonso", "campo")])
        self.assertEqual((enr["medidas"]["largura_cm"], enr["tamanho"]), (10.4, "pequeno (10 a 30 cm)"))
        self.assertEqual(_valores(enr["materiais"], "campo"), {"vidro"})          # a categoria diz o material
        self.assertEqual(R.enriquecer(_ficha(production_center="Desconhecido."))["origem"], [])

    def test_document_for_the_index(self) -> None:
        ficha = _ficha(artifact_id="raiz:movel:1", inventory_number="1 Our", title="Cálice", category="Ourivesaria", technique="Prata dourada", in_tour=True, start_year=1500)
        doc = R.documento(ficha, R.enriquecer(ficha))
        self.assertEqual((doc["artifact_id"], doc["museum_id"], doc["in_tour"], doc["start_year"]), ("raiz:movel:1", "mnaa", True, 1500))
        self.assertEqual(doc["enr_tipo"], ["ourivesaria", "cálice"])
        self.assertEqual(doc["enr_materiais_ficha"], ["prata dourada", "prata"])
        self.assertEqual(doc["enr_funcao"], ["litúrgico"])
        self.assertEqual(doc["enr_detalhe"]["materiais"][0]["onde"], "técnica")


def _sim(*chaves, excerto):
    return {chave: {"presente": "sim" if chave in chaves else "não", "excerto": excerto if chave in chaves else ""} for chave, _, _ in T.TEMAS}


class TestTemas(unittest.TestCase):
    """As regras de aceitação que saíram das 200 peças conferidas a 2026-11-24."""

    def _temas(self, ficha, saida):
        enr = R.enriquecer(ficha)
        return {t["valor"]: t["fonte"] for t in T.aceitar(ficha, saida, enr["tipo_objeto"]["valor"] if enr["tipo_objeto"] else None)}

    def test_a_yes_needs_its_excerpt_in_the_record(self) -> None:
        ficha = _ficha(title="São Pedro", category="Escultura", description="São Pedro esculpido de pé, com as chaves na mão.")
        self.assertEqual(self._temas(ficha, _sim("religioso", excerto="São Pedro esculpido de pé")), {"Religioso": "texto"})
        self.assertEqual(self._temas(ficha, _sim("religioso", excerto="um santo com o seu cão")), {})        # o excerto não está na ficha
        self.assertEqual(self._temas(ficha, None), {})

    def test_themes_accepted_as_they_are(self) -> None:
        # decisão de 2026-11-24: os sete temas abaixo da fasquia entram tal como o modelo os dá
        ficha = _ficha(title="Terrina", category="Ourivesaria", description="De cada lado vêem-se as Armas Reais Portuguesas, entre folhas de acanto.")
        saida = _sim("heraldica", "flora", "arquitetura", excerto="Armas Reais Portuguesas, entre folhas de acanto")
        self.assertEqual(set(self._temas(ficha, saida)), {"Flora e ornamento vegetal", "Heráldica e poder", "Arquitetura e cidade"})
        self.assertTrue(set(T.NOME.values()) - {"Escrita e texto", "Retrato"} - T.SO_EM_OBJETOS <= T.ACEITES_DO_MODELO)
        self.assertEqual(set(T.PALAVRAS), set(T.NOME.values()))

    def test_decoration_themes_stay_out_of_paintings(self) -> None:
        saida = _sim("flora", "geometrico", "religioso", excerto="com decoração fitomórfica")
        folio = _ficha(title="Fólio de Texto", category="Pintura", support_or_material="Velino", technique="Pintura a têmpera e ouro",
                       description="Sete letras iluminadas, com decoração fitomórfica.")
        self.assertEqual(self._temas(folio, saida), {"Religioso": "texto", "Escrita e texto": "regra"})
        jarra = _ficha(title="Jarra", category="Vidros", description="Jarra pintada com decoração fitomórfica.")
        self.assertEqual(set(self._temas(jarra, saida)), {"Flora e ornamento vegetal", "Ornamento geométrico", "Religioso"})

    def test_writing_only_when_the_piece_is_a_written_thing(self) -> None:
        pintura = _ficha(title="São Francisco", category="Pintura", technique="Pintura a óleo", description="O santo segura um livro entreaberto com a mão esquerda.")
        self.assertEqual(self._temas(pintura, _sim("escrita", excerto="segura um livro entreaberto")), {})
        folio = _ficha(title="Horas da Virgem", category="Pintura", support_or_material="Velino", technique="Pintura a têmpera e ouro")
        self.assertEqual(self._temas(folio, _sim(excerto="")), {"Escrita e texto": "regra"})

    def test_portrait_only_from_the_title(self) -> None:
        cadeira = _ficha(title="Cadeira", category="Mobiliário", description="No cachaço, um rosto feminino com longos cabelos.")
        self.assertEqual(self._temas(cadeira, _sim("retrato", excerto="rosto feminino com longos cabelos")), {})
        retrato = _ficha(title="Auto-retrato de D. Maria", category="Pintura")
        self.assertEqual(self._temas(retrato, None), {"Retrato": "campo"})
        self.assertEqual(T.palavras(T.aceitar(retrato, None, "pintura")), ["retrato"])

    def test_what_the_photo_shows(self) -> None:
        saida = {"cores_principais": ["vermelho"], "cores": ["vermelho"], "motivos": ["cenas narrativas"], "temas": ["Ciência e ofícios"],
                 "tem_figura_humana": "sim", "tem_animais": "não", "tem_inscricao": "sim"}
        self.assertEqual(F.aceitar(saida), ["pessoas", "letras"])        # só os três campos confirmados na validação
        self.assertEqual(F.aceitar({"tem_animais": "não se sabe"}), [])
        self.assertEqual(F.aceitar(None), [])
        self.assertEqual(F.palavras(["animais"]), ["animal", "bicho"])
        self.assertEqual(F.IMPRESSAO, "f419f58168ca3e9d")                   # a instrução conferida; mudá-la refaz a cache
        # «pinturas com animais»: o animal tanto pode vir do tema (ficha) como da fotografia
        vocabulary = {"enr_tipo": ["pintura"], "enr_temas_palavras": T.PALAVRAS["Animais"], "enr_foto_palavras": F.PALAVRAS["animais"]}
        self.assertEqual(E.match_subject("pinturas animais", vocabulary),
                         [{"enr_tipo": ["pintura"]}, {"enr_temas_palavras": ["animal"], "enr_foto_palavras": ["animal"]}])

    def test_the_prompt_is_the_one_that_was_validated(self) -> None:
        # mudar a instrução ou as definições muda as respostas: a validação deixa de valer e a cache é refeita
        self.assertEqual(T.IMPRESSAO, "5256e961bcfd5581")


if __name__ == "__main__":
    unittest.main()
