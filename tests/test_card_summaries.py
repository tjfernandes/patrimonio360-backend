"""O texto responde à pergunta e a frase de cada peça vai no cartão dela (CHAT_CARD_SUMMARIES).
docs/frases-nos-cartoes.md.

    python -m unittest tests.test_card_summaries
"""
from __future__ import annotations

import asyncio
import json
import re

from pydantic import ValidationError

from app.api.routes.logs import FrontendEventPayload
from app.core.config import Settings
from app.prompts.chat_prompts import build_draft_opening_prompt, build_draft_sentence_prompt
from app.schemas.chat import ArtifactResult, ChatResultsPageRequest, ResponseFormatObject
from app.services.draft_rewrite import AnswerDraft, DraftRewriteMerger, merge_rewrite
from app.services.llm_service import LLMResponse, LLMServiceError
from tests.test_answer_list_plan import CARDS, _card
from tests.test_author_before_answer import _Flow
from tests.test_draft_rewrite import _FlowThinker, _Thinker
from tests.test_tour_location import _Base

AUTHOR_OPENING = "O museu tem 19 peças de Gregório Lopes. Os cartões mostram as primeiras 8 (as restantes estão em «Ver mais resultados»):"
MODEL_OPENING = "Encontrei estas obras do pintor."


class TestContract(_Base):
    def test_off_by_default(self) -> None:
        settings = Settings(_env_file=None)
        self.assertFalse(settings.CHAT_CARD_SUMMARIES)
        self.assertFalse(settings.CHAT_CARD_SUMMARIES_REWRITE_OPENING)
        self.assertIsNone(ArtifactResult(artifact_id="a").card_summary)
        self.assertIn("card_summary", ArtifactResult(artifact_id="a").model_dump())

    def test_the_frontend_event_objects_loaded_is_accepted(self) -> None:
        self.assertEqual(FrontendEventPayload(event_type="objects_loaded").event_type, "objects_loaded")
        with self.assertRaises(ValidationError):
            FrontendEventPayload(event_type="evento_que_nao_existe")


class TestDraftWithSummaries(_Base):
    def _service_with(self, thinker, **settings):
        service = self._service(self._settings(CHAT_COMPACT_ANSWER_CONTEXT=True, CHAT_UNTITLED_CARD_BY_INVENTORY=True,
                                               CHAT_IN_TOUR_FROM_NAVIGATION=True, **settings))
        service.preprocess_llm_service = thinker
        return service

    def _draft(self, thinker, cards=CARDS, **kwargs) -> AnswerDraft:
        return asyncio.run(self._service_with(thinker)._draft_answer(
            user_message="esculturas de barro", cards=cards, language="pt", card_summaries=True, **kwargs))

    def test_the_sentences_go_to_the_cards_and_the_text_is_the_opening(self) -> None:
        thinker = _Thinker()
        draft = self._draft(thinker)
        self.assertEqual(draft.summaries, ["frase da peça Fé.", "frase da peça Caridade.", "frase da peça 584 Esc."])
        self.assertEqual((draft.lines, draft.text), ([], "Encontrei estas esculturas."))
        per_piece = [prompt for prompt in thinker.prompts if "FICHA:\n" in prompt]
        self.assertEqual(len(per_piece), 3)   # um pedido por ficha, só com ela
        for prompt in per_piece:
            self.assertEqual(prompt.count("Título: "), 1)
            self.assertIn("no máximo 25 palavras ao todo", prompt)
            self.assertNotIn("Na visita virtual", prompt)
        opening = [prompt for prompt in thinker.prompts if "FICHAS:\n" in prompt]
        self.assertEqual(len(opening), 1)
        self.assertIn("Tu escreves só a resposta ao pedido", opening[0])
        self.assertEqual((draft.info["card_summaries"], draft.info["sentences_from_model"]), (3, 3))

    def test_a_sentence_that_fails_is_the_first_of_the_record(self) -> None:
        draft = self._draft(_Thinker(fail={"Caridade"}))
        self.assertEqual(draft.summaries[1], "Grupo escultórico com uma figura feminina e duas crianças, encomendado por D. Maria I.")
        self.assertEqual(draft.info["sentences_from_model"], 2)
        # sem frase do modelo nem descrição na ficha, o cartão fica sem texto
        self.assertEqual(self._draft(_Thinker(fail={"Sem descrição"}), [_card("x", "1 Esc", "Sem descrição"), CARDS[0]]).summaries,
                         [None, "frase da peça Fé."])

    def test_the_authors_opening_is_the_text(self) -> None:
        thinker = _Thinker()
        draft = self._draft(thinker, fixed_opening="O museu tem 3 peças de X. São estas:")
        self.assertEqual((draft.text, draft.fixed_opening, len(thinker.prompts)), ("O museu tem 3 peças de X. São estas:", True, 3))

    def test_pieces_of_the_same_set_get_one_sentence_each_and_a_grouped_card_gets_one(self) -> None:
        same_set = [ArtifactResult(artifact_id=f"p{n}", inventory_number=f"{n} Cer", title=f"Prato {n}", set_ids=["c:1"], sets=["Serviço"])
                    for n in (1, 2, 3)]
        grouped = ArtifactResult(artifact_id="g", inventory_number="9 Cer", title="Chávena", group_size=3,
                                 group_inventory_numbers=["9 Cer", "10 Cer", "11 Cer"], group_artifact_ids=["g", "h", "i"])
        thinker = _Thinker()
        draft = self._draft(thinker, [*same_set, grouped])
        self.assertEqual(draft.summaries, ["frase da peça Prato 1.", "frase da peça Prato 2.", "frase da peça Prato 3.", "frase da peça Chávena."])
        self.assertEqual(sum("FICHA:\n" in prompt for prompt in thinker.prompts), 4)

    def test_a_card_text_fits_in_three_lines(self) -> None:
        service = self._service_with(_Thinker())
        short = "Pintura a óleo sobre madeira, do século XVI."
        self.assertEqual(service._card_summary_text(f" **{short[:-1]}** "), short)
        two = "Desenho a pena e aguada, do séc. XVII, atribuído a um mestre flamengo. " + "Mostra uma paisagem com um rio e figuras na margem, " * 4
        self.assertEqual(service._card_summary_text(two), "Desenho a pena e aguada, do séc. XVII, atribuído a um mestre flamengo.")
        self.assertLessEqual(len(service._card_summary_text("Frase curta. " * 40)), 220)
        # nem a primeira frase cabe: fica a da ficha
        self.assertEqual(service._card_summary_text("palavra " * 60), "")
        self.assertEqual(service._card_summary_text("Linha\numa só."), "Linha uma só.")

    def test_a_gerund_that_escapes_becomes_the_infinitive(self) -> None:
        service = self._service_with(_Thinker())
        fixed = service._without_gerund
        self.assertEqual(fixed("Um anjo segurando uma coroa, com chamas saindo do peito."), "Um anjo a segurar uma coroa, com chamas a sair do peito.")
        self.assertEqual(fixed("Pintura a óleo, representando São Jerónimo e erguendo-se sobre uma base."),
                         "Pintura a óleo, a representar São Jerónimo e a erguer-se sobre uma base.")
        # nomes, o início da frase e o que não é gerúndio (ou se escreve assim em Portugal) ficam como estão
        for text in ("Retrato de Fernando II, quando era príncipe.", "Tendo sido feita em Goa, é uma peça do século XVI.",
                     "Serviço de chá, incluindo bule e açucareiro, datando do século XVIII.", "Um bando de aves num fundo lindo.",
                     "Estuário com navios, destacando-se um grande navio no primeiro plano."):
            self.assertEqual(fixed(text), text)
        self.assertEqual(service._card_summary_text("Santa Ana ensinando a Virgem a ler"), "Santa Ana a ensinar a Virgem a ler.")

    def test_prompts(self) -> None:
        sentence = build_draft_sentence_prompt(user_message="pinturas com barcos", card="FICHA-DE-TESTE", card_summary=True)
        self.assertIn("«pinturas com barcos»", sentence)
        self.assertIn("mesmo quando não é o que foi pedido", sentence)
        self.assertIn("só com o que está na ficha", sentence)
        self.assertIn("nunca com o gerúndio", sentence)
        self.assertIn("não fales da visita virtual", sentence)
        self.assertTrue(sentence.endswith('Responde só com JSON: {"frase": "..."}\n\nFICHA:\nFICHA-DE-TESTE'))
        # pergunta pelo aspeto: o pedido não vai, nem a instrução que fala dele (CHAT_VISUAL_ANSWER_RULE)
        visual = build_draft_sentence_prompt(user_message=None, card="x", card_summary=True)
        self.assertNotIn("O visitante pediu", visual)
        self.assertNotIn("o pedido", visual)
        opening = build_draft_opening_prompt(user_message="Quem pintou as Tentações?", cards="FICHA-DE-TESTE", card_summaries=True)
        self.assertIn("Se o visitante perguntou quem, quando, onde ou se há, a frase responde a isso", opening)
        self.assertIn("não fale de nenhuma em particular", opening)
        self.assertIn("Não escrevas títulos de peças", opening)
        self.assertIn("nunca com o gerúndio", opening)
        self.assertTrue(opening.endswith('{"abertura": "..."}\n\nFICHAS:\nFICHA-DE-TESTE'))
        # sem exemplos com objetos: o modelo copiava-os para as frases (docs/rascunho-e-reescrita.md)
        for prompt in (sentence, visual, opening):
            self.assertNotIn("livro", prompt)
            self.assertNotIn("por exemplo", prompt)
        self.assertIn("at most 25 words in all", build_draft_sentence_prompt(user_message="boats", card="x", language="en", card_summary=True))
        self.assertIn("speaks of none in particular", build_draft_opening_prompt(user_message="boats", cards="x", language="en", card_summaries=True))
        # sem a flag os prompts são os de sempre
        self.assertIn("uma frase com no máximo 15 palavras", build_draft_sentence_prompt(user_message="retratos", card="x"))
        self.assertIn("que as apresente em resposta ao pedido", build_draft_opening_prompt(user_message="retratos", cards="x"))

    def test_a_draft_without_lines_is_only_its_opening(self) -> None:
        draft = AnswerDraft(opening="Sim, há pinturas com barcos.", names=[], lines=[], summaries=["a", "b"])
        self.assertEqual(merge_rewrite(draft, "Claro que sim, há pinturas com barcos!\n\n1. **Peça** – inventada.\nBoa visita!"),
                         ("Claro que sim, há pinturas com barcos!", 0))
        # uma abertura dela que já enumera peças, ou nada: fica a do rascunho
        self.assertEqual(merge_rewrite(draft, "1. **Peça** – inventada.")[0], "Sim, há pinturas com barcos.")
        self.assertEqual(merge_rewrite(draft, "Há a **Peça** e outras.")[0], "Sim, há pinturas com barcos.")
        self.assertEqual(merge_rewrite(draft, "")[0], "Sim, há pinturas com barcos.")
        merger = DraftRewriteMerger(draft)
        merger.feed("Claro que sim.")
        self.assertEqual((merger.text(), merger.opening_rewritten), ("Claro que sim.", True))


class TestInTheSearch(_Flow):
    ALL = [f"doc_{n}" for n in range(1, 9)]
    FLAGS = dict(CHAT_ANSWER_ALL_CARDS=True, CHAT_ANSWER_LIST_PLAN=True, CHAT_UNTITLED_CARD_BY_INVENTORY=True, CHAT_ANSWER_BOLD_NAMES=True,
                 CHAT_AUTHOR_ONLY_FILL_PAGE=True, CHAT_TOUR_PRESENCE_ONLY_IF_ASKED=True, CHAT_STREAM_COMMITTED_LIST=True,
                 CHAT_DRAFT_THEN_REWRITE=True, CHAT_CARD_SUMMARIES=True)
    OTHERS = "Mostra-me obras de Gregório Lopes e de outros pintores."

    def test_the_text_answers_and_each_card_carries_its_sentence(self) -> None:
        pre = _FlowThinker(cards=self.ALL, only="B")
        service, llm = self._flow(pre, "isto não devia ser pedido", **self.FLAGS)
        response, streamed = self._ask(service, self.OTHERS)
        # o texto é só a abertura do pré-processamento: sem lista, sem negrito, e a AMÁLIA não é chamada
        self.assertEqual(response.reply, MODEL_OPENING)
        self.assertEqual(streamed, response.reply)
        self.assertEqual(llm.prompts, [])
        self.assertEqual(response.model_hint, service.settings.llm_model_resolved)
        self.assertEqual([card.card_summary for card in response.artifact_results],
                         [f"frase de Peça {c}." for c in ["l1", "o1", "l2", "o2", "l3", "o3", "l4", "o4"]])
        self.assertEqual(sum("FICHA:\n" in p for p in pre.prompts), 8)
        self.assertEqual(sum("FICHAS:\n" in p for p in pre.prompts), 1)
        state = service.session_store.get(response.conversation_id)
        self.assertEqual(state.history[-1].text, response.reply)
        # os cartões guardados para a conversa seguinte são os mesmos; a frase fica só na resposta
        stored = state.paged_results_by_request_id[response.results_request_id]["retrieval_request"]
        self.assertEqual(stored["card_summaries"], {"question": self.OTHERS, "visual": False})

    def test_an_answer_by_author_keeps_the_catalogue_numbers(self) -> None:
        service, llm = self._flow(_FlowThinker(cards=self.ALL), "isto não devia ser pedido", **self.FLAGS,
                                  CHAT_CARD_SUMMARIES_REWRITE_OPENING=True)
        response, streamed = self._ask(service)
        self.assertEqual(response.reply, AUTHOR_OPENING)
        self.assertEqual(streamed, response.reply)
        self.assertEqual(llm.prompts, [])   # a abertura do autor nunca é trocada: nada há para reescrever
        self.assertEqual(response.model_hint, "deterministic:author_counts")
        self.assertEqual([card.card_summary for card in response.artifact_results], [f"frase de Peça l{n}." for n in range(1, 9)])

    def test_the_final_model_may_rewrite_the_opening_and_nothing_else(self) -> None:
        wrote = "Com todo o gosto, aqui ficam as obras do pintor.\n\n1. **Peça l1** – inventada.\n\nBoa visita!"
        service, llm = self._flow(_FlowThinker(cards=self.ALL, only="B"), wrote, **self.FLAGS, CHAT_CARD_SUMMARIES_REWRITE_OPENING=True)
        response, streamed = self._ask(service, self.OTHERS)
        self.assertEqual(len(llm.prompts), 1)
        self.assertTrue(llm.prompts[0].endswith(f"TEXTO:\n{MODEL_OPENING}"))
        self.assertEqual(response.reply, "Com todo o gosto, aqui ficam as obras do pintor.")
        self.assertEqual(streamed, response.reply)
        self.assertEqual(response.artifact_results[0].card_summary, "frase de Peça l1.")

    def test_when_the_final_model_is_down_the_opening_is_the_drafts(self) -> None:
        service, llm = self._flow(_FlowThinker(cards=self.ALL, only="B"), "x", **self.FLAGS, CHAT_CARD_SUMMARIES_REWRITE_OPENING=True)

        async def down(**_kwargs):
            raise LLMServiceError("AMÁLIA em baixo")

        llm.generate_stream = down
        llm.generate = down
        response, streamed = self._ask(service, self.OTHERS)
        self.assertEqual(response.reply, MODEL_OPENING)
        self.assertEqual(streamed, response.reply)
        self.assertEqual(response.artifact_results[0].card_summary, "frase de Peça l1.")

    def test_flag_off_the_list_stays_in_the_text(self) -> None:
        wrote = "Aqui ficam.\n" + "\n".join(f"{n}. **Peça l{n}** – reescrita {n}." for n in range(1, 9))
        service, llm = self._flow(_FlowThinker(cards=self.ALL), wrote, **{**self.FLAGS, "CHAT_CARD_SUMMARIES": False})
        response, _streamed = self._ask(service)
        self.assertIn("1. **Peça l1** – reescrita 1.", response.reply)
        self.assertEqual({card.card_summary for card in response.artifact_results}, {None})
        stored = service.session_store.get(response.conversation_id).paged_results_by_request_id[response.results_request_id]
        self.assertNotIn("card_summaries", stored["retrieval_request"])

    def test_without_the_draft_flag_nothing_changes(self) -> None:
        service, llm = self._flow(_FlowThinker(cards=self.ALL), "Encontrei:\n1. Peça l1 – a.", **{**self.FLAGS, "CHAT_DRAFT_THEN_REWRITE": False})
        response, _streamed = self._ask(service)
        self.assertIn("CARTÕES CONFIRMADOS:", llm.prompts[0])
        self.assertEqual({card.card_summary for card in response.artifact_results}, {None})

    def test_one_card_stays_as_it_was(self) -> None:
        service, llm = self._flow(_FlowThinker(cards=["doc_1"], name="B"), "A Peça l1 é uma pintura.", **self.FLAGS)
        response, _streamed = self._ask(service, "Mostra-me a peça da rua Gregório Lopes.")
        self.assertEqual(len(response.artifact_results), 1)
        self.assertIn("CARTÕES CONFIRMADOS:", llm.prompts[0])   # o prompt de sempre: o texto já é sobre a peça
        self.assertIsNone(response.artifact_results[0].card_summary)
        self.assertEqual(response.reply, "A Peça l1 é uma pintura.")

    def _page(self, service, response, page: int = 2):
        return asyncio.run(service.get_results_page(ChatResultsPageRequest(
            museum_slug="mnaa", museum_id="mnaa", conversation_id=response.conversation_id, results_page=page,
            results_request_id=response.results_request_id)))

    def test_see_more_results_carries_the_sentences_too(self) -> None:
        pre = _FlowThinker(cards=self.ALL, only="B")
        service, _llm = self._flow(pre, "x", **self.FLAGS)
        response, _streamed = self._ask(service, self.OTHERS)
        before = len(pre.prompts)
        page = self._page(service, response)
        self.assertEqual(len(page.artifact_results), 8)
        self.assertEqual([card.card_summary for card in page.artifact_results],
                         [f"frase de {card.title}." for card in page.artifact_results])
        asked = pre.prompts[before:]
        self.assertEqual(len(asked), 8)   # uma frase por cartão; a abertura não se pede outra vez
        self.assertTrue(all("FICHA:\n" in prompt and f"«{self.OTHERS}»" in prompt for prompt in asked))
        # a terceira página também (o pedido vem guardado com a pesquisa)
        self.assertTrue(all(card.card_summary for card in self._page(service, response, 3).artifact_results))

    def test_see_more_results_falls_back_to_the_record_when_the_model_is_down(self) -> None:
        pre = _FlowThinker(cards=self.ALL, only="B")
        service, _llm = self._flow(pre, "x", **self.FLAGS)
        response, _streamed = self._ask(service, self.OTHERS)

        async def down(**_kwargs):
            raise RuntimeError("modelo em baixo")

        pre.generate = down
        page = self._page(service, response)
        self.assertEqual(len(page.artifact_results), 8)
        self.assertEqual({card.card_summary for card in page.artifact_results}, {None})   # estas fichas de teste não têm descrição

    def test_see_more_results_stays_a_grid_when_the_first_page_was_one(self) -> None:
        pre = _FlowThinker(cards=self.ALL, only="B")
        wrote = "Aqui ficam.\n" + "\n".join(f"{n}. **Peça** – x." for n in range(1, 9))
        service, _llm = self._flow(pre, wrote, **{**self.FLAGS, "CHAT_CARD_SUMMARIES": False})
        response, _streamed = self._ask(service, self.OTHERS)
        before = len(pre.prompts)
        service.settings.CHAT_CARD_SUMMARIES = True   # ligada a meio de uma conversa: a pesquisa antiga fica como estava
        page = self._page(service, response)
        self.assertEqual({card.card_summary for card in page.artifact_results}, {None})
        self.assertEqual(len(pre.prompts), before)
