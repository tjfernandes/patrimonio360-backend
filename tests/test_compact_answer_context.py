"""Quick win PR-01 (CHAT_COMPACT_ANSWER_CONTEXT): cartões compactos e prompt
final v2. Desligado, o contexto e o prompt são os históricos.

    python -m unittest tests.test_compact_answer_context
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import Settings  # noqa: E402
from app.prompts.chat_prompts import build_final_answer_prompt_v2  # noqa: E402
from app.schemas.chat import ArtifactResult, ChatMessageRequest  # noqa: E402
from app.services.chat_service import ChatService  # noqa: E402
from app.services.chat_session_store import ChatSessionState  # noqa: E402


def _service(compact: bool) -> ChatService:
    service = object.__new__(ChatService)
    service.settings = Settings(_env_file=None, CHAT_COMPACT_ANSWER_CONTEXT=compact,
                                QUERY_LOG_ENABLED=False, BACKEND_LOG_ENABLED=False)
    return service


def _results() -> list[ArtifactResult]:
    long_description = "Vestido de noiva em seda branca com bordados. " * 12
    return [
        ArtifactResult(
            artifact_id="art-1", inventory_number="MNT 1234", title="Vestido de noiva",
            category="Traje", super_category="Traje e adereços", creators=["Autor Desconhecido"],
            date_or_period="1890", support_or_material="Seda", technique="Bordado",
            description=long_description, in_tour=True, museum="Museu Nacional do Traje",
        ),
        ArtifactResult(artifact_id="art-2", title="Capa", start_year=1700, end_year=1750, in_tour=False),
    ]


class TestCompactCards(unittest.TestCase):
    def test_compact_context_shape(self) -> None:
        service = _service(True)
        context = service._build_visible_results_retrieval_context(
            artifact_results=_results(), image_matches=[], page=1, page_size=8, total=2,
            prefer_artifact_results=True, include_image_matches_section=False,
        )
        self.assertIn("visible_results_count: 2", context)
        self.assertIn("[doc_1] Título: Vestido de noiva | Inventário: MNT 1234", context)
        self.assertIn("Categoria: Traje (Traje e adereços) | Data: 1890 | Materiais: Seda | Técnica: Bordado | Autor: Autor Desconhecido", context)
        self.assertIn("Na visita virtual: sim", context)
        self.assertIn("[doc_2] Título: Capa | Inventário: não indicado", context)
        self.assertIn("Data: 1700–1750", context)
        self.assertIn("Na visita virtual: não", context)
        self.assertNotIn("art-1", context)
        self.assertNotIn("artifact_id", context)
        self.assertNotIn("{", context)
        description_line = next(line for line in context.splitlines() if "Descrição:" in line)
        self.assertLessEqual(len(description_line), len("    Descrição: ") + 282)
        self.assertTrue(description_line.endswith("…"))

    def test_flag_off_keeps_json_context(self) -> None:
        service = _service(False)
        context = service._build_visible_results_retrieval_context(
            artifact_results=_results(), image_matches=[], page=1, page_size=8, total=2,
            prefer_artifact_results=True, include_image_matches_section=False,
        )
        self.assertIn('[doc_1] {"inventory_number": "MNT 1234", "title": "Vestido de noiva"', context)
        self.assertNotIn("Título:", context)

    def test_sanitizer_alignment_with_compact_cards(self) -> None:
        service = _service(True)
        docs = service._artifact_results_as_prompt_docs(_results())
        reply = service._sanitize_assistant_reply("Veja doc_2 e [doc_1].", docs=docs, language="pt")
        self.assertNotIn("doc_", reply)
        self.assertIn("Capa", reply)
        self.assertIn("Vestido de noiva", reply)


class TestFinalPromptV2(unittest.TestCase):
    def test_pt_prompt_contains_guards_and_cards(self) -> None:
        prompt = build_final_answer_prompt_v2(
            museum_name="Museu Nacional do Traje", intent="search", mode="rag", rolling_summary="",
            filters_state={"category": "traje", "empty": None}, user_message="Encontra vestidos de noiva",
            rewritten_query="vestidos de noiva", retrieval_context="visible_results_count: 2\n[doc_1] Título: X",
            language="pt",
        )
        self.assertIn("És a Amália", prompt)
        self.assertIn("visible_results_count", prompt)
        self.assertIn("Filtros ativos: {\"category\": \"traje\"}", prompt)
        self.assertIn("PERGUNTA: Encontra vestidos de noiva", prompt)
        self.assertIn("PEDIDO INTERPRETADO: vestidos de noiva", prompt)
        self.assertIn("TIPO DE RESPOSTA: pesquisa", prompt)
        self.assertTrue(prompt.rstrip().endswith("[doc_1] Título: X"))
        self.assertNotIn("museum_slug", prompt)
        self.assertNotIn("rolling_summary:", prompt)

    def test_no_cards_and_selected_and_conversation_kinds(self) -> None:
        empty = build_final_answer_prompt_v2(
            museum_name=None, intent="fallback", mode="llm_only", rolling_summary="antes falámos de capas",
            filters_state={}, user_message="explica melhor", rewritten_query="explica melhor",
            retrieval_context="", use_history_for_answer=True, language="pt",
        )
        self.assertIn("TIPO DE RESPOSTA: conversa", empty)
        self.assertIn("CARTÕES CONFIRMADOS:\n(nenhum)", empty)
        self.assertIn("Conversa anterior", empty)
        selected = build_final_answer_prompt_v2(
            museum_name="MNSR", intent="selected_artifact_question", mode="rag", rolling_summary="",
            filters_state={}, user_message="quem fez esta peça?", rewritten_query="quem fez esta peça?",
            retrieval_context="[doc_1] Título: Mesa", language="en",
        )
        self.assertIn("ANSWER TYPE: selected object", selected)
        self.assertIn("You are Amália", selected)

    def test_service_selects_v2_only_with_flag(self) -> None:
        payload = ChatMessageRequest(museum_slug="mnt", museum_id="mnt", museum_name="MNT",
                                     language="pt", message="vestidos")
        state = ChatSessionState(conversation_id="c", museum_slug="mnt", language="pt")
        decision = {"mode": "rag", "intent": "search", "rewritten_query": "vestidos"}
        on = _service(True)._build_final_prompt(
            payload=payload, state=state, router_decision=decision, retrieval_context="[doc_1] Título: V",
            effective_filters={}, effective_sort={}, use_history_for_answer=False, museum_name="MNT")
        off = _service(False)._build_final_prompt(
            payload=payload, state=state, router_decision=decision, retrieval_context="[doc_1] Título: V",
            effective_filters={}, effective_sort={}, use_history_for_answer=False, museum_name="MNT")
        self.assertIn("És a Amália", on)
        self.assertIn("Es um assistente virtual para uma visita 360 ao museu.", off)


if __name__ == "__main__":
    unittest.main()
