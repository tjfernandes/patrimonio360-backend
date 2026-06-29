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
