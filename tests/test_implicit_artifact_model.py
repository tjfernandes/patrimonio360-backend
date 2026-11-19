"""Peça implícita pelo modelo (CHAT_IMPLICIT_ARTIFACT_MODEL): quando a mensagem diz o nome de
uma peça do ecrã, quem diz qual é passa a ser o modelo, com a lista das peças mostradas (os
cartões e as outras peças dos conjuntos deles).

O modelo de pré-processamento aqui é um guião; a redação do prompt mede-se com o modelo real
(tests/implicit_artifact_cases.py).

    python -m unittest tests.test_implicit_artifact_model
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.core.config import Settings
from app.prompts.query_planner_prompts import build_implicit_artifact_prompt
from app.schemas.chat import ArtifactResult, ChatMessageResponse
from tests.test_tour_location import _Base

SET_ID, SET_NAME = "15007", "Painéis de São Vicente de Fora"
CARDS = [
    ArtifactResult(artifact_id="v1", inventory_number="1550 Esc", title="São Vicente"),
    ArtifactResult(artifact_id="v2", inventory_number="1082 Esc", title="São Vicente"),
    ArtifactResult(artifact_id="p2", inventory_number="1362 Pint", title="Painel da Relíquia", set_ids=[SET_ID], sets=[SET_NAME]),
]
VISIBLE = [{"artifact_id": card.artifact_id, "inventory_number": card.inventory_number, "title": card.title} for card in CARDS]
FIRST = {**VISIBLE[0], "source": "first_result"}
OPEN = {"artifact_id": "h1", "inventory_number": "1211 Pint", "title": "Anunciação", "source": "tour_hotspot"}
# as seis peças do conjunto, como o índice as devolve (o cartão incluído)
MEMBERS = [
    {"artifact_id": "p1", "inventory_number": "1361 Pint", "title": "Painel do Infante"},
    {"artifact_id": "p2", "inventory_number": "1362 Pint", "title": "Painel da Relíquia"},
    {"artifact_id": "p3", "inventory_number": "1363 Pint", "title": "Painel dos Frades"},
    {"artifact_id": "p4", "inventory_number": "1364 Pint", "title": "Painel do Arcebispo"},
    {"artifact_id": "p5", "inventory_number": "1365 Pint", "title": "Painel dos Cavaleiros"},
    {"artifact_id": "p6", "inventory_number": "1366 Pint", "title": "Painel dos Pescadores"},
]
# a lista que o modelo vê: A v1, B v2, C p2, D p1, E p3, F p4, G p5, H p6


class _Preprocess:
    def __init__(self, answer: str | None = "nenhuma", how_many: str = "uma") -> None:
        self.answer, self.how_many, self.prompts = answer, how_many, []

    async def generate(self, *, message, **_kwargs):
        self.prompts.append(message)
        if self.answer is None:
            raise RuntimeError("modelo em baixo")
        return SimpleNamespace(parsed_json={"quantas": self.how_many, "peca": self.answer})


class _SetGateway:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls, self.fail = [], fail

    async def fetch_artifacts_by_entity(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("OpenSearch em baixo")
        return [dict(member) for member in MEMBERS], len(MEMBERS)


class TestImplicitArtifactByModel(_Base):
    def _service_with(self, preprocess: _Preprocess, *, flag: bool = True, stored: bool = True, fail_sets: bool = False):
        service = self._service(self._settings(CHAT_IMPLICIT_ARTIFACT_CONTEXT=True, CHAT_IMPLICIT_ARTIFACT_MODEL=flag))
        service._preprocess_llm_service = preprocess
        service.opensearch_gateway = _SetGateway(fail=fail_sets)
        if stored:
            # como o serviço: os cartões da última resposta ficam guardados na conversa
            state = service.session_store.load_or_create(conversation_id="c1", museum_slug="mnaa")
            service._cache_last_retrieval_results(state=state, artifact_results=CARDS, image_matches=[], navigation_targets=[],
                                                  default_page_size=8, retrieval_request={"kind": "text"})
            service.session_store.save(state)
        self.seen = {}

        async def answered(payload, *, status_cb=None):
            self.seen = {"metadata": payload.metadata, "message": payload.message}
            return ChatMessageResponse(reply="ok")

        service._handle_message_with_selection = answered
        return service

    def _ask(self, service, message: str, *, visible=VISIBLE, context=FIRST):
        request = self._request(message, museum_slug="mnaa", museum_id="mnaa", conversation_id="c1",
                                museum_name="Museu Nacional de Arte Antiga")
        request.metadata.update({"visible_artifacts": visible, "context_artifact": context})
        response = asyncio.run(service.handle_message(request))
        subject = response.subject_artifact
        return (subject.artifact_id, subject.resolved_by) if subject is not None else (None, None)

    def test_off_by_default_the_rules_decide(self) -> None:
        self.assertFalse(Settings(_env_file=None).CHAT_IMPLICIT_ARTIFACT_MODEL)
        preprocess = _Preprocess("E")
        service = self._service_with(preprocess, flag=False)
        # a regra do título só vê os cartões: «Painel» bate com o único que começa assim
        self.assertEqual(self._ask(service, "O que representa o Painel dos Frades?"), ("p2", "title"))
        self.assertEqual(preprocess.prompts, [])
        self.assertEqual(service.opensearch_gateway.calls, [])

    def test_the_model_names_a_piece_of_the_set(self) -> None:
        preprocess = _Preprocess("E")
        service = self._service_with(preprocess)
        self.assertEqual(self._ask(service, "O que representa o Painel dos Frades?"), ("p3", "model"))
        # a pergunta segue a dizer de que peça se trata, e a peça fica escolhida
        self.assertEqual(self.seen["message"], "O que representa o Painel dos Frades? (refiro-me à peça «Painel dos Frades» (1363 Pint))")
        self.assertEqual(self.seen["metadata"]["selected_artifact"], {"artifact_id": "p3", "inventory_number": "1363 Pint",
                                                                      "title": "Painel dos Frades", "source": "implicit_model"})
        # o modelo viu os três cartões e, a seguir ao do conjunto, as outras cinco peças dele
        prompt = preprocess.prompts[0]
        self.assertIn("A) «São Vicente» (n.º 1550 Esc)\nB) «São Vicente» (n.º 1082 Esc)\n"
                      f"C) «Painel da Relíquia» (n.º 1362 Pint) — do conjunto «{SET_NAME}»\n"
                      f"D) «Painel do Infante» (n.º 1361 Pint) — do conjunto «{SET_NAME}»\n"
                      f"E) «Painel dos Frades» (n.º 1363 Pint) — do conjunto «{SET_NAME}»\n", prompt)
        self.assertIn(f"H) «Painel dos Pescadores» (n.º 1366 Pint) — do conjunto «{SET_NAME}»\n\nMensagem do visitante:\n"
                      "«O que representa o Painel dos Frades?»", prompt)
        self.assertNotIn("X)", prompt)
        call = service.opensearch_gateway.calls[0]
        self.assertEqual((call["tipo"], call["entity_id"], call["museum_id"]), ("conjunto", SET_ID, "mnaa"))

    def test_position_and_inventory_stay_exact_lookups(self) -> None:
        preprocess = _Preprocess("A")
        service = self._service_with(preprocess)
        self.assertEqual(self._ask(service, "E o segundo?"), ("v2", "ordinal"))
        # «o último» é o último cartão, não a última peça do conjunto
        self.assertEqual(self._ask(service, "O último é de quando?"), ("p2", "ordinal"))
        # o n.º de inventário de uma peça do conjunto também se encontra
        self.assertEqual(self._ask(service, "1365 Pint"), ("p5", "inventory"))
        self.assertEqual(preprocess.prompts, [])

    def test_a_search_is_not_shown_to_the_model(self) -> None:
        preprocess = _Preprocess("E")
        service = self._service_with(preprocess)
        self.assertEqual(self._ask(service, "Procura painéis com frades"), (None, None))
        self.assertEqual(self._ask(service, "Mostra-me o Painel dos Frades"), (None, None))
        self.assertEqual(preprocess.prompts, [])

    def test_when_the_model_names_none_the_rules_of_today_apply(self) -> None:
        for answer in ("nenhuma", "", "Z", "1999 Pint", None):
            with self.subTest(answer=answer):
                preprocess = _Preprocess(answer, how_many="várias")
                service = self._service_with(preprocess)
                # pergunta sem assunto próprio: a peça em foco, pelas regras
                self.assertEqual(self._ask(service, "De que é feito?"), ("v1", "context"))
                self.assertEqual(self.seen["message"], "De que é feito?")
                # pesquisa: sem peça
                self.assertEqual(self._ask(service, "Pinturas flamengas do século XVI"), (None, None))
                self.assertEqual(len(preprocess.prompts), 2)

    def test_the_inventory_number_as_an_answer_is_accepted(self) -> None:
        service = self._service_with(_Preprocess("1364 Pint"))
        self.assertEqual(self._ask(service, "E o do arcebispo?"), ("p4", "model"))

    def test_requests_answered_elsewhere_do_not_reach_the_model(self) -> None:
        preprocess = _Preprocess("C")
        service = self._service_with(preprocess)
        for message in ("Onde estou?", "Olá"):
            self.assertEqual(self._ask(service, message), (None, None), message)
        self.assertEqual(preprocess.prompts, [])
        # sem cartões nem peça em foco não há nada a resolver
        self.assertEqual(self._ask(service, "O que representa o Painel dos Frades?", visible=[], context=None), (None, None))
        self.assertEqual(preprocess.prompts, [])

    def test_the_piece_open_in_the_tour_has_its_own_letter(self) -> None:
        preprocess = _Preprocess("X")
        service = self._service_with(preprocess)
        subject = self._ask(service, "A Anunciação é de quem?", context=OPEN)
        self.assertEqual(subject, ("h1", "model"))
        self.assertIn(f"H) «Painel dos Pescadores» (n.º 1366 Pint) — do conjunto «{SET_NAME}»\n"
                      "Fora desta lista, o visitante tem aberta na visita a peça:\nX) «Anunciação» (n.º 1211 Pint)\n\n", preprocess.prompts[0])
        # e a posição continua a contar só os cartões
        self.assertEqual(self._ask(service, "E o último?", context=OPEN), ("p2", "ordinal"))

    def test_the_set_is_read_once_per_conversation(self) -> None:
        service = self._service_with(_Preprocess("E"))
        self._ask(service, "O que representa o Painel dos Frades?")
        self._ask(service, "E o painel do infante?")
        self.assertEqual(len(service.opensearch_gateway.calls), 1)

    def test_without_stored_cards_or_with_the_index_down_only_the_cards_are_listed(self) -> None:
        for stored, fail_sets in ((False, False), (True, True)):
            with self.subTest(stored=stored):
                preprocess = _Preprocess("C")
                service = self._service_with(preprocess, stored=stored, fail_sets=fail_sets)
                self.assertEqual(self._ask(service, "O painel da relíquia é de quando?"), ("p2", "model"))
                self.assertIn("C) «Painel da Relíquia» (n.º 1362 Pint)", preprocess.prompts[0])
                self.assertNotIn("D)", preprocess.prompts[0])

    def test_a_long_screen_is_left_to_the_rules(self) -> None:
        many = [{"artifact_id": f"m{n}", "inventory_number": str(1000 + n), "title": f"Peça {n}"} for n in range(30)]
        preprocess = _Preprocess("A")
        service = self._service_with(preprocess, stored=False)
        self.assertEqual(self._ask(service, "De que é feito?", visible=many, context={**many[0], "source": "first_result"}), ("m0", "context"))
        self.assertEqual(preprocess.prompts, [])

    def test_the_prompt(self) -> None:
        prompt = build_implicit_artifact_prompt(
            entities=[{"title": "Fé", "inventory_number": "60 Esc", "set_name": "Devoção"}, {"title": None, "inventory_number": "584 Esc"}],
            open_piece=None, message="E a Fé?",
        )
        self.assertIn("A) «Fé» (n.º 60 Esc) — do conjunto «Devoção»\nB) peça sem título (n.º 584 Esc)\n\nMensagem do visitante:\n«E a Fé?»", prompt)
        self.assertTrue(prompt.endswith('{"quantas": "uma" ou "várias", "peca": "letra" ou "nenhuma"}'))
