"""Intenção «describe_artifact» (CHAT_DESCRIBE_ARTIFACT_INTENT): o frontend diz
qual é a peça («Ver na tour»), não há pesquisa, e a resposta é só sobre ela.

    python -m unittest tests.test_describe_artifact_intent
"""
from __future__ import annotations

import asyncio
import unittest

from app.services.chat_service import ChatService
from tests.test_tour_location import _Base, _Gateway, _LLMReply

SAPATOS = {
    "artifact_id": "a-4270", "inventory_number": "4270", "title": "Sapatos (par)/Feminino", "museum_id": "mnt",
    "category": "Calçado", "date_or_period": "1790-1800", "support_or_material": "Seda; Pele",
    "description": "Par de sapatos de seda bordada a fio de prata, com salto forrado.", "image_count": 1, "in_tour": True,
}


class _IdGateway(_Gateway):
    def __init__(self, docs_by_id: dict | None = None) -> None:
        super().__init__()
        self.docs_by_id = docs_by_id if docs_by_id is not None else {"a-4270": SAPATOS}
        self.id_calls: list[list[str]] = []

    async def fetch_artifacts_by_ids(self, *, museum_slug, museum_id, artifact_ids, top_k):
        self.id_calls.append(list(artifact_ids))
        return [self.docs_by_id[i] for i in artifact_ids if i in self.docs_by_id]

    async def fetch_images_by_artifact_ids(self, *, artifact_ids, **_kwargs):
        return [
            {"artifact_id": i, "image_id": f"img-{i}", "local_path": f"Images/x/{i}.jpg",
             "inventory_number": self.docs_by_id[i]["inventory_number"], "museum_id": "mnt", "score": 1.0}
            for i in artifact_ids if i in self.docs_by_id
        ]


class _DescribeLLM:
    def __init__(self, text: str | None = "Um par de sapatos de seda do fim do século XVIII, bordados a prata.") -> None:
        self.text, self.prompts, self.system_prompts = text, [], []

    async def generate(self, *, message, system_prompt=None, **_kwargs):
        self.prompts.append(message)
        self.system_prompts.append(system_prompt)
        if self.text is None:
            from app.services.llm_service import LLMServiceError
            raise LLMServiceError("LLM em baixo")
        reply = _LLMReply(self.text)
        reply.model = "fake"
        return reply


class _SearchCalled(RuntimeError):
    pass


class TestDescribeArtifactIntent(_Base):
    POI_BY_MUSEUM = {"mnt": [
        {"overlayId": "ov-sapatos", "panoramaKey": "pan_calcado", "inventoryIds": ["4270"], "location": "Sala do Calçado", "title": "Sapatos"},
    ]}
    SELECTED = {
        "artifact_id": "a-4270", "inventory_number": "4270", "title": "Sapatos (par)/Feminino",
        "museum_slug": "mnt", "source": "see_in_tour",
    }

    def _describe_service(self, llm=None, gateway=None, **flags):
        settings = self._settings(**{
            "CHAT_DESCRIBE_ARTIFACT_INTENT": True, "CHAT_SELECTED_ARTIFACT_AUTO_RELEASE": True,
            "CHAT_COMPACT_ANSWER_CONTEXT": True, **flags,
        })
        service = self._service(settings, llm=llm or _DescribeLLM())
        service.opensearch_gateway = gateway or _IdGateway()

        async def _no_search(*_args, **_kwargs):
            raise _SearchCalled("a pesquisa não devia correr")

        service._handle_message_with_selection = _no_search
        return service

    def _describe_request(self, **kwargs):
        request = self._request("Sapatos (par)/Feminino (4270)", **kwargs)
        request.metadata.update({"intent": "describe_artifact", "selected_artifact": dict(self.SELECTED)})
        return request

    def test_answers_about_the_piece_without_searching(self) -> None:
        llm = _DescribeLLM()
        service = self._describe_service(llm=llm)
        response = asyncio.run(service.handle_message(self._describe_request()))
        self.assertEqual(response.reply, llm.text)
        self.assertEqual(service.opensearch_gateway.id_calls, [["a-4270"]])
        self.assertEqual([a.artifact_id for a in response.artifact_results], ["a-4270"])
        self.assertEqual([m.artifact_id for m in response.image_matches], ["a-4270"])
        self.assertEqual([t.inventory_id for t in response.navigation_targets], ["4270"])
        self.assertFalse(response.results_has_more)
        self.assertEqual(response.results_total, 1)
        self.assertEqual(response.selected_artifact_status, "used")
        self.assertEqual(response.model_hint, "fake")

    def test_prompt_is_the_backend_one_with_the_card_only(self) -> None:
        llm = _DescribeLLM()
        service = self._describe_service(llm=llm)
        asyncio.run(service.handle_message(self._describe_request()))
        prompt = llm.prompts[0]
        self.assertIn("acabou de chegar a esta peça", prompt)
        self.assertIn("PEÇA: Sapatos (par)/Feminino (4270)", prompt)
        self.assertIn("[doc_1] Título: Sapatos (par)/Feminino | Inventário: 4270", prompt)
        self.assertIn("bordada a fio de prata", prompt)
        self.assertEqual(prompt.count("[doc_1] Título"), 1)
        self.assertNotIn("[doc_2]", prompt)
        self.assertNotIn("CARTÕES CONFIRMADOS", prompt)

    def test_english_request_gets_the_english_prompt(self) -> None:
        llm = _DescribeLLM("A pair of silk shoes.")
        service = self._describe_service(llm=llm)
        asyncio.run(service.handle_message(self._describe_request(language="en")))
        self.assertIn("has just reached this object", llm.prompts[0])
        self.assertIn("OBJECT: Sapatos (par)/Feminino (4270)", llm.prompts[0])
        self.assertIn("English", llm.system_prompts[0] or "")

    def test_falls_back_to_inventory_when_the_id_is_gone(self) -> None:
        gateway = _IdGateway(docs_by_id={})   # por id não há nada; por inventário há o 19663 (DOCS do teste vizinho)
        service = self._describe_service(gateway=gateway)
        request = self._describe_request()
        request.metadata["selected_artifact"] = {"artifact_id": "a-antigo", "inventory_number": "19663", "title": "Barrete"}
        response = asyncio.run(service.handle_message(request))
        self.assertEqual(gateway.inventory_calls, [["19663"]])
        self.assertEqual([a.artifact_id for a in response.artifact_results], ["a-19663"])

    def test_unknown_piece_says_so_without_cards(self) -> None:
        llm = _DescribeLLM()
        service = self._describe_service(llm=llm, gateway=_IdGateway(docs_by_id={}))
        request = self._describe_request()
        request.metadata["selected_artifact"] = {"artifact_id": "a-nada", "title": "Nada"}
        response = asyncio.run(service.handle_message(request))
        self.assertIn("Não encontrei a ficha", response.reply)
        self.assertEqual(response.artifact_results, [])
        self.assertEqual(llm.prompts, [])
        self.assertEqual(response.model_hint, "deterministic:describe_artifact")

    def test_llm_failure_keeps_the_card(self) -> None:
        service = self._describe_service(llm=_DescribeLLM(text=None))
        response = asyncio.run(service.handle_message(self._describe_request()))
        self.assertIn("LLM indisponivel", response.reply)
        self.assertEqual([a.artifact_id for a in response.artifact_results], ["a-4270"])
        self.assertEqual(service.query_logger.events[-1]["status"], "error")

    def test_query_log_has_no_participant_and_marks_the_path(self) -> None:
        service = self._describe_service()
        asyncio.run(service.handle_message(self._describe_request()))
        event = service.query_logger.events[-1]
        self.assertEqual(event["retrieval_mode"], "describe_artifact")
        self.assertEqual(event["fast_path"], "describe_artifact")
        self.assertIsNone(event["participant_id"])
        self.assertIsNone(event["task_id"])
        self.assertEqual([a["artifact_id"] for a in event["shown_artifacts"]], ["a-4270"])

    def test_opening_greeting_is_stripped(self) -> None:
        strip = ChatService._strip_opening_greeting
        self.assertEqual(strip("Olá, veja a *Fuga para o Egipto*, azulejo de 1730."), "A *Fuga para o Egipto*, azulejo de 1730.")
        self.assertEqual(strip("Olá! Este painel é de 1730."), "Este painel é de 1730.")
        self.assertEqual(strip("Hello, look at this panel."), "This panel.")
        self.assertEqual(strip("Este painel é de 1730."), "Este painel é de 1730.")
        self.assertEqual(strip("Olá a todos os visitantes."), "Olá a todos os visitantes.")
        self.assertEqual(strip(""), "")

    def test_greeting_from_the_llm_is_removed_end_to_end(self) -> None:
        service = self._describe_service(llm=_DescribeLLM("Olá, veja este par de sapatos de seda."))
        response = asyncio.run(service.handle_message(self._describe_request()))
        self.assertEqual(response.reply, "Este par de sapatos de seda.")

    def test_flag_off_goes_to_the_normal_path(self) -> None:
        service = self._describe_service(CHAT_DESCRIBE_ARTIFACT_INTENT=False)
        with self.assertRaises(_SearchCalled):
            asyncio.run(service.handle_message(self._describe_request()))

    def test_other_intents_and_missing_selection_go_to_the_normal_path(self) -> None:
        service = self._describe_service()
        request = self._describe_request()
        request.metadata["intent"] = "ask"
        with self.assertRaises(_SearchCalled):
            asyncio.run(service.handle_message(request))
        request = self._describe_request()
        del request.metadata["selected_artifact"]
        with self.assertRaises(_SearchCalled):
            asyncio.run(service.handle_message(request))


if __name__ == "__main__":
    unittest.main()
