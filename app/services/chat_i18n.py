from typing import Any, Literal

ChatLanguage = Literal["pt", "en"]
TranslationValue = str | dict[str, str]


_TRANSLATIONS: dict[str, dict[ChatLanguage, TranslationValue]] = {
    "status.analyzing_request": {
        "pt": "A analisar pedido",
        "en": "Analyzing request",
    },
    "status.searching_collection": {
        "pt": "A procurar artefactos no acervo",
        "en": "Searching the collection",
    },
    "status.artifacts_found": {
        "pt": {
            "one": "Encontrado 1 artefacto",
            "other": "Encontrados {artifact_count} artefactos",
        },
        "en": {
            "one": "Found 1 artifact",
            "other": "Found {artifact_count} artifacts",
        },
    },
    "status.generating_final_answer": {
        "pt": "A gerar resposta final",
        "en": "Generating final answer",
    },
    "status.answer_ready": {
        "pt": "Resposta pronta",
        "en": "Answer ready",
    },
    "status.analyzing_image": {
        "pt": "A analisar imagem enviada",
        "en": "Analyzing uploaded image",
    },
    "status.preparing_model": {
        "pt": "A preparar modelo 3D",
        "en": "Preparing 3D model",
    },
    "status.generating_model_views": {
        "pt": "A gerar vistas do modelo 3D",
        "en": "Generating 3D model views",
    },
    "error.no_history_regenerate": {
        "pt": "Conversa sem historico para regenerar.",
        "en": "There is no conversation history to regenerate.",
    },
    "error.no_user_message_regenerate": {
        "pt": "Nao existe mensagem de utilizador para regenerar resposta.",
        "en": "There is no user message to regenerate a reply from.",
    },
    "error.image_processing_failed": {
        "pt": "Nao consegui processar a imagem enviada. Tenta novamente com outra imagem ou formato diferente.",
        "en": "I could not process the uploaded image. Try again with another image or a different format.",
    },
    "error.model_processing_failed": {
        "pt": "Nao consegui processar o modelo 3D enviado. Tenta novamente com um ficheiro .glb, .gltf ou .obj valido.",
        "en": "I could not process the uploaded 3D model. Try again with a valid .glb, .gltf, or .obj file.",
    },
    "error.llm_unavailable": {
        "pt": "LLM indisponivel em desenvolvimento: {error}. Verifica LLM_PROVIDER/LLM_BASE_URL/model no backend .env.",
        "en": "LLM unavailable in dev: {error}. Check LLM_PROVIDER/LLM_BASE_URL/model settings in backend .env.",
    },
    "error.metadata_invalid": {
        "pt": "metadata invalido: {error}",
        "en": "Invalid metadata: {error}",
    },
    "error.response_format_invalid": {
        "pt": "response_format invalido.",
        "en": "Invalid response_format.",
    },
    "error.empty_image_upload": {
        "pt": "Imagem vazia no upload.",
        "en": "Uploaded image is empty.",
    },
    "error.empty_model_upload": {
        "pt": "Modelo 3D vazio no upload.",
        "en": "Uploaded 3D model is empty.",
    },
    "error.model_format_invalid": {
        "pt": "Formato 3D invalido. Usa {allowed}.",
        "en": "Invalid 3D format. Use {allowed}.",
    },
    "message.default_image_query": {
        "pt": "Analisa a imagem e identifica a peca mais provavel no museu.",
        "en": "Analyze the image and identify the most likely object in the museum.",
    },
    "message.default_model_query": {
        "pt": "Analisa este modelo 3D e identifica a peca mais provavel no museu.",
        "en": "Analyze this 3D model and identify the most likely object in the museum.",
    },
    "message.results_page_fallback": {
        "pt": "Aqui estao mais resultados encontrados para esta pesquisa.",
        "en": "Here are more results found for this search.",
    },
    "message.smalltalk_reply": {
        "pt": (
            "Olá! Posso ajudar a encontrar peças do museu e a explorá-las nesta visita. "
            "Diga-me o que procura: um objeto, um tema, uma época ou um autor."
        ),
        "en": (
            "Hello! I can help you find objects in the museum and explore them in this tour. "
            "Tell me what you are looking for: an object, a theme, a period or a maker."
        ),
    },
    "message.smalltalk_thanks_reply": {
        "pt": "De nada! Se quiser, continue a explorar: diga-me uma peça, um tema, uma época ou um autor.",
        "en": "You're welcome! Keep exploring if you like: name an object, a theme, a period or a maker.",
    },
    "message.smalltalk_bye_reply": {
        "pt": "Até à próxima! Boa visita.",
        "en": "See you next time! Enjoy the tour.",
    },
    "message.tour_location_unknown": {
        "pt": (
            "Ainda não sei em que ponto da visita está: a visita virtual não me indicou a sua posição. "
            "Avance para um panorama ou abra uma peça e volte a perguntar. Posso, entretanto, procurar peças no acervo."
        ),
        "en": (
            "I don't know where you are yet: the virtual tour hasn't reported your position. "
            "Move to a panorama or open an object and ask again. Meanwhile, I can search the collection for you."
        ),
    },
    "message.tour_location_here": {
        "pt": "Está em «{room}».",
        "en": "You are in \"{room}\".",
    },
    "message.tour_location_here_unnamed": {
        "pt": "Está num ponto da visita sem nome atribuído.",
        "en": "You are at a point of the tour without an assigned name.",
    },
    "message.tour_location_pieces": {
        "pt": {
            "one": "Neste ponto da visita está assinalada 1 peça: {pieces}. Abra o cartão para ver a ficha ou peça-me detalhes.",
            "other": "Neste ponto da visita estão assinaladas {count} peças: {pieces}. Abra um cartão para ver a ficha ou peça-me detalhes sobre uma delas.",
        },
        "en": {
            "one": "At this point of the tour there is 1 highlighted object: {pieces}. Open the card to see the record or ask me for details.",
            "other": "At this point of the tour there are {count} highlighted objects: {pieces}. Open a card to see the record or ask me for details about one of them.",
        },
    },
    "message.tour_location_room_pieces": {
        "pt": {
            "one": "Nesta sala a visita assinala 1 peça: {pieces}. Abra o cartão para ver a ficha ou peça-me detalhes.",
            "other": "Nesta sala a visita assinala {count} peças: {pieces}. Abra um cartão para ver a ficha ou peça-me detalhes sobre uma delas.",
        },
        "en": {
            "one": "In this room the tour highlights 1 object: {pieces}. Open the card to see the record or ask me for details.",
            "other": "In this room the tour highlights {count} objects: {pieces}. Open a card to see the record or ask me for details about one of them.",
        },
    },
    "message.tour_room_intro": {
        "pt": "«{room}» faz parte da visita virtual.",
        "en": "\"{room}\" is part of the virtual tour.",
    },
    "message.tour_room_go": {
        "pt": "Pode ir até lá com o botão abaixo.",
        "en": "You can go there with the button below.",
    },
    "message.tour_room_unknown": {
        "pt": "Não encontro nenhuma zona da visita com esse nome. Estas são as zonas da visita; escolha uma e levo-o até lá.",
        "en": "I can't find a tour area with that name. These are the tour areas; pick one and I will take you there.",
    },
    "message.tour_room_choices": {
        "pt": "Há {count} zonas da visita com esse nome. Escolha uma e levo-o até lá.",
        "en": "There are {count} tour areas with that name. Pick one and I will take you there.",
    },
    "message.tour_rooms_list": {
        "pt": {
            "one": "A visita virtual tem 1 zona. Pode ir até lá com o botão abaixo.",
            "other": "A visita virtual tem {count} zonas. Escolha uma e levo-o até lá.",
        },
        "en": {
            "one": "The virtual tour has 1 area. You can go there with the button below.",
            "other": "The virtual tour has {count} areas. Pick one and I will take you there.",
        },
    },
    "message.tour_location_no_pieces": {
        "pt": "Esta zona da visita não tem peças assinaladas. Diga-me o que procura e pesquiso no acervo do museu.",
        "en": "This area of the tour has no highlighted objects. Tell me what you are looking for and I will search the museum's collection.",
    },
    "message.tour_pieces_list": {
        "pt": {
            "one": "A visita virtual tem 1 peça assinalada. O cartão está abaixo: abra-o para ver a ficha ou pergunte-me por ela.",
            "other": "A visita virtual tem {count} peças assinaladas, em {rooms} zonas. Os cartões estão abaixo, pela ordem da visita: abra um para ver a ficha ou pergunte-me por uma delas.",
        },
        "en": {
            "one": "The virtual tour has 1 highlighted object. Its card is below: open it to see the record or ask me about it.",
            "other": "The virtual tour has {count} highlighted objects, in {rooms} areas. The cards are below, in tour order: open one to see the record or ask me about it.",
        },
    },
    "message.tour_pieces_none": {
        "pt": "Esta visita virtual não tem peças assinaladas. Diga-me o que procura e pesquiso no acervo do museu.",
        "en": "This virtual tour has no highlighted objects. Tell me what you are looking for and I will search the museum's collection.",
    },
    "message.describe_artifact_not_found": {
        "pt": "Não encontrei a ficha desta peça no acervo, por isso não a consigo apresentar. Pergunte-me por outra peça ou diga-me o que procura.",
        "en": "I could not find this object's record in the collection, so I cannot present it. Ask me about another object or tell me what you are looking for.",
    },
    "message.results_page_template": {
        "pt": {
            "one": "Página {page}: mais 1 peça desta pesquisa ({total} no total). Abra o cartão para ver a ficha ou peça-me detalhes.",
            "other": "Página {page}: mais {count} peças desta pesquisa ({total} no total). Abra um cartão para ver a ficha ou peça-me detalhes sobre uma delas.",
        },
        "en": {
            "one": "Page {page}: 1 more object for this search ({total} in total). Open the card to see the record or ask me for details.",
            "other": "Page {page}: {count} more objects for this search ({total} in total). Open a card to see the record or ask me for details about one of them.",
        },
    },
    "llm.final_language_guard": {
        "pt": "A resposta final ao utilizador deve estar integralmente em portugues.",
        "en": "The final user-facing answer must be entirely in English.",
    },
    "sanitizer.artifact_label": {
        "pt": "a peca",
        "en": "the artifact",
    },
    "sanitizer.context_label": {
        "pt": "contexto",
        "en": "context",
    },
    "sanitizer.collection_artifact": {
        "pt": "a peca do acervo",
        "en": "the artifact from the collection",
    },
    "sanitizer.titled_artifact": {
        "pt": "a peca \"{title}\"",
        "en": "the artifact \"{title}\"",
    },
    "sanitizer.titled_inventory_artifact": {
        "pt": "a peca \"{title}\" ({inventory})",
        "en": "the artifact \"{title}\" ({inventory})",
    },
    "sanitizer.inventory_artifact": {
        "pt": "a peca de inventario {inventory}",
        "en": "the artifact with inventory {inventory}",
    },
}


class _SafeFormatDict(dict[str, object]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def normalize_language(language: str | None) -> ChatLanguage:
    return "en" if (language or "").strip().lower() == "en" else "pt"


def translate(key: str, language: str | None, **params: Any) -> str:
    resolved_language = normalize_language(language)
    by_language = _TRANSLATIONS.get(key)
    if by_language is None:
        return key

    value = by_language.get(resolved_language) or by_language["pt"]
    if isinstance(value, dict):
        count_value = params.get("count", params.get("artifact_count"))
        try:
            count = int(count_value)
        except (TypeError, ValueError):
            count = 0
        template = value["one"] if count == 1 else value["other"]
    else:
        template = value

    return template.format_map(_SafeFormatDict(params))
