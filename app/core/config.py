from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _parse_csv(raw: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    return values or ["*"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    APP_ENV: str = "development"
    # Post-submission deployment candidates are opt-in; the current stack stays intact.
    RETRIEVAL_PROFILE_PATH: str = ""
    CHAT_ENABLE_EFFICIENCY_GUARDS: bool = False
    CHAT_BOUNDED_GPU_EXECUTION: bool = False
    CHAT_SERIALIZE_CONVERSATIONS: bool = False
    QUERY_LOG_COMPACT_VECTORS: bool = False
    # Quick wins pós-submissão (branch feature/post-submission-quick-wins).
    # Todos opt-in; desligados reproduzem byte a byte o comportamento do estudo.
    CHAT_DETERMINISTIC_SMALLTALK: bool = False      # saudação pura sem router nem LLM
    CHAT_TEMPLATE_RESULTS_PAGE_REPLY: bool = False  # «ver mais» sem chamada LLM
    CHAT_STREAM_FINAL_ANSWER: bool = False          # tokens da resposta final por SSE
    CHAT_COMPACT_ANSWER_CONTEXT: bool = False       # cartões compactos + prompt final PT v2
    CHAT_INVENTORY_GRAMMAR_V2: bool = False         # «793 Mob MNSR» reconhecido como inventário
    # off | log — verifica títulos/inventários citados | repair — log + repõe
    # inventários abreviados que correspondem a um único cartão («71 CMP/MNSR»)
    CHAT_GROUNDING_CHECK: str = "off"
    # Router: "llm" = chamada JSON ao 9B em todos os turnos (2,7–2,9 s, estudo);
    # "deterministic" = política de contexto decide (rag salvo small talk;
    # refine se follow-up; filtros só por extração determinística). Nos 146
    # turnos de texto do estudo o router LLM nunca produziu filters_delta e as
    # 3 vezes em que escolheu llm_only fora de saudações estava errado.
    CHAT_ROUTER_MODE: str = "llm"
    # «Onde estou?» / «o que há nesta sala?»: resposta determinística a partir
    # da posição na visita (metadata.tour_location enviado pelo widget) e dos
    # ficheiros poi_tours, sem pesquisa nem LLM. Desligado = vai ao router.
    CHAT_TOUR_LOCATION_ANSWERS: bool = False
    # Peça selecionada («Perguntar», hotspot aberto): um pedido que não é sobre ela
    # (nova pesquisa, «onde estou?») liberta-a em vez de a forçar como contexto.
    CHAT_SELECTED_ARTIFACT_AUTO_RELEASE: bool = False
    # Peça implícita: sem botões nem etiquetas, a pergunta resolve-se para a peça de que
    # o visitante fala (ordinal, n.º de inventário ou nome de um cartão visível; senão o
    # hotspot aberto ou o primeiro resultado) quando não traz assunto próprio.
    CHAT_IMPLICIT_ARTIFACT_CONTEXT: bool = False
    # Com a anterior: quando a mensagem diz o nome de uma peça do ecrã, quem diz qual é passa a
    # ser o modelo de pré-processamento, com a lista das peças mostradas (os cartões e as outras
    # peças dos conjuntos deles). A posição («o segundo») e o n.º de inventário continuam a ser
    # procuras exatas, e as regras de hoje ficam como rede quando o modelo não aponta nenhuma.
    # Medido em tests/implicit_artifact_cases.py (docs/peca-implicita-pelo-modelo.md).
    CHAT_IMPLICIT_ARTIFACT_MODEL: bool = False
    # Com o anterior: a frase é redigida pelo LLM a partir dos factos da
    # visita (sala e peças fixas; cartões trancados); template se falhar.
    CHAT_TOUR_CONTEXT_LLM_WORDING: bool = False
    # Intenção explícita do frontend: ao carregar em «Ver na tour» num cartão, o
    # pedido traz metadata.intent = "describe_artifact" e a peça em
    # metadata.selected_artifact. Sem pesquisa: a ficha vai buscar-se pelo
    # artifact_id e a resposta é uma apresentação curta da peça, só dela.
    CHAT_DESCRIBE_ARTIFACT_INTENT: bool = False
    # O modelo escolhe os cartões: termina a resposta com «[[cartões: doc_1, doc_3]]»
    # (retirado do texto) e só essas peças, pela ordem dele, ficam em
    # artifact_results, image_matches e navigation_targets. Um pedido por nome
    # ou n.º de inventário mostra só as peças que correspondem («egito» ≈
    # «Egipto»), sem «ver mais». Sem bloco, os cartões ficam como hoje.
    CHAT_MODEL_SELECTS_CARDS: bool = False
    # Pré-processamento num segundo modelo (2026-11-16): com a flag ligada, todas
    # as chamadas JSON antes da resposta (router, datas, seguimento, reescrita da
    # pesquisa, tema dos percursos, escolha dos cartões) vão ao modelo de
    # CHAT_PREPROCESS_LLM_* (o Qwen3-VL-8B do deploy/vlm/amalia_vlm.sh) e a AMÁLIA
    # (LLM_*) só escreve o texto final. Se esse servidor recusar a ligação, o
    # mesmo pedido vai à AMÁLIA, como antes.
    CHAT_PREPROCESS_LLM_ENABLED: bool = False
    CHAT_PREPROCESS_LLM_BASE_URL: str = "http://127.0.0.1:8005/v1"
    CHAT_PREPROCESS_LLM_MODEL: str = "Qwen/Qwen3-VL-8B-Instruct-FP8"
    CHAT_PREPROCESS_LLM_API_KEY: str = ""
    CHAT_PREPROCESS_LLM_TIMEOUT_SECONDS: float = 20.0
    # O modelo de pré-processamento escolhe, antes da resposta, os cartões que
    # correspondem ao pedido; a AMÁLIA só vê esses e deixa de escrever o bloco
    # «[[cartões: …]]» (CHAT_MODEL_SELECTS_CARDS fica como recurso quando a
    # escolha falha). Medido a 2026-11-16 com as 8 fichas da «Custódia de Belém»:
    # 39/42 certos, contra 20–27/48 do bloco da AMÁLIA.
    CHAT_PRESELECT_CARDS: bool = False
    CHAT_PRESELECT_CARDS_TIMEOUT_SECONDS: float = 4.0
    # Quantas fichas da pesquisa o modelo de pré-processamento vê para escolher
    # os cartões da primeira página. 0 (ou um valor até ao tamanho da página):
    # só as da página, como antes. Com 15, escolhe entre as 15 primeiras e
    # mostram-se as escolhidas, até encher a página; o «ver mais» salta as que
    # já foram mostradas. Sem escolha utilizável ficam as fichas da página.
    CHAT_PRESELECT_CARDS_CANDIDATES: int = 0
    # A data das fichas por extenso. O export do RAIZ escreve os séculos como
    # números («16», «18–19», «-5») e era isso que ia para os cartões e para os
    # modelos («Data: 16»). Com a flag, o texto nesse formato é refeito a partir
    # de start_year/end_year: «século XVI», «séculos XVIII–XIX», «século V a.C.»,
    # «590–580 a.C.». Texto livre («Lisboa, c. 1715-1730») fica como está.
    CHAT_READABLE_DATES: bool = False
    # Correção da pesquisa do chat, Bloco A (docs/plano-correcao-pesquisa-mnaa.md,
    # 2026-11-16). Cada flag é independente e, desligada, nada muda.
    # F1: a lista vazia do modelo de pré-processamento («nenhum cartão
    # corresponde») passa a valer: zero cartões, regra «preselect_none». Erro,
    # tempo esgotado ou JSON sem lista válida continuam em «preselect_failed».
    # Na corrida de 2026-11-16 a lista vazia saiu em 14 pedidos, sempre com razão.
    CHAT_PRESELECT_EMPTY_IS_NONE: bool = False
    # F2: numa pesquisa sem cartões a AMÁLIA recebe uma instrução curta de «não
    # encontrei» em vez do prompt de pesquisa. Medido em 10 perguntas × 5 por
    # língua: títulos inventados 0/50 (antes 1/40 em PT e 4/40 em EN); menções a
    # «cartões/base de dados» 17/40 -> 5/50 em PT e 28/40 -> 5/50 em EN.
    CHAT_NO_CARDS_ANSWER_PROMPT: bool = False
    # F3: «Na visita virtual: sim» no cartão do prompt quando a peça tem alvo de
    # navegação («Ver na tour»), mesmo que o in_tour do índice diga que não (no
    # MNAA: 46 marcadas, 760 na visita). Só muda o texto dado à AMÁLIA: os cartões
    # do pré-processamento e os devolvidos ao frontend ficam como estavam.
    CHAT_IN_TOUR_FROM_NAVIGATION: bool = False
    # F9: o nome completo de uma sala, sem mais nada no pedido, ganha à regra dos
    # percursos («Leva-me à Capela das Albertas» vai à sala, não cria um percurso).
    CHAT_ROOM_BEFORE_ROUTE: bool = False
    # F11: retira da resposta os rótulos dos cartões copiados à letra
    # («(Tesouro Nacional: sim)», «Na visita virtual: não»), como já se faz a [doc_N].
    CHAT_STRIP_CARD_LABELS: bool = False
    # --- Pesquisa do chat, Bloco B (docs/pesquisa-bloco-b.md) ---
    # F5: a pergunta de texto é embebida com a instrução «query» do
    # Qwen3-Embedding (o modelo distingue pergunta de documento; o índice foi
    # embebido como documento e a pergunta também o era). Só a pesquisa de texto
    # do chat; percursos e pesquisa por imagem ficam como estão.
    CHAT_RETRIEVAL_QUERY_INSTRUCTION: bool = False
    # F4: o ramo lexical passa a procurar também no autor, na categoria, no
    # material e no centro de fabrico (só título e descrição até aqui). A
    # cláusula que exige todas as palavras deixa de as exigir no mesmo campo.
    CHAT_RETRIEVAL_LEXICAL_METADATA_FIELDS: bool = False
    # F6: os filtros (museu, época, visita) vão dentro do kNN, para os 1000
    # vizinhos serem procurados já dentro do museu e não no índice dos sete.
    CHAT_RETRIEVAL_KNN_PREFILTER: bool = False
    # --- Pesquisa do chat, Bloco C (docs/pesquisa-bloco-c.md) ---
    # O modelo de pré-processamento lê cada mensagem de texto com a conversa
    # anterior e devolve a língua, se é um seguimento e o pedido reescrito por
    # inteiro («Quem o pintou?» -> «Quem pintou as Tentações de Santo Antão?»).
    # A decisão de seguimento deixa de ser das expressões regulares; o pedido
    # reescrito continua a passar pelo guarda-corpo das reescritas. Mais uma
    # chamada por mensagem (mediana de 0,2 a 0,3 s). Não corre com uma peça
    # selecionada nem nos percursos.
    CHAT_REQUEST_READING: bool = False
    # O assunto de uma estatística («a mais antiga de Gregório Lopes») procura
    # também no autor e exige todas as palavras, repartidas pelos campos. Sem
    # isto bastava uma («Gregório» apanhava as peças de São Gregório) e o autor
    # não contava. Assuntos de uma palavra («esculturas») ficam iguais.
    CHAT_CATALOGUE_ANALYTICS_STRICT_SUBJECT: bool = False
    # Fichas iguais do mesmo conjunto («Botões (dez)», 38195/1 a 38195/10: título e
    # descrição iguais, só o n.º muda) juntam-se num cartão, o da ficha mais bem
    # posicionada (ou a que está na visita), com group_size e os n.ºs das outras.
    # A pesquisa pede FETCH_FACTOR vezes mais candidatos para a página continuar
    # cheia; o «ver mais» pede desde o início e corta depois de juntar.
    CHAT_COLLAPSE_SET_DUPLICATES: bool = False
    CHAT_COLLAPSE_FETCH_FACTOR: int = 3
    # Listagens do acervo: «que autores há?», «que tipos de peças tem o museu?»,
    # «de que materiais são as peças?», «que estilos existem?». Uma pesquisa
    # devolve oito peças; estas perguntas querem o que existe no museu inteiro.
    # O modelo de pré-processamento diz se a mensagem é uma destas e de quê
    # (duas perguntas curtas, medidas em tests/listing_request_cases.py); o
    # OpenSearch agrega o campo em todas as fichas e a resposta sai por modelo de
    # texto, sem cartões. Só para mensagens que não são seguimento.
    CHAT_CATALOGUE_LISTINGS: bool = False
    # Ficha do autor: quando a mensagem diz o nome de um autor das peças
    # mostradas («obras de Gregório Lopes», «quem foi Soares dos Reis?»), a
    # resposta leva `author` com os dados do catálogo (atividade, datas,
    # biografia, n.º de peças). O autor sai das fichas; o modelo de
    # pré-processamento só diz se as palavras da mensagem são o nome de um
    # artista (uma pergunta curta, medida em tests/author_card_cases.py). O
    # texto e os cartões não mudam.
    CHAT_AUTHOR_CARD: bool = False
    # A ficha do autor também num seguimento que não diz o nome («Que mais obras
    # existem deste autor?», depois de se falar de uma peça dele). O nome
    # procura-se então no pedido que a leitura do seguimento resolveu («obras do
    # Machado de Castro»), e só quando a mensagem não tem nome nenhum.
    CHAT_AUTHOR_CARD_FOLLOW_UP: bool = False
    # Autor da peça em foco: com uma peça à frente no minichat
    # (`metadata.context_artifact`), «Que mais obras existem deste autor?» passa a
    # ser «obras de <autor dessa peça>», mesmo que o nome nunca tenha aparecido na
    # conversa. O autor lê-se da ficha da peça; o modelo de pré-processamento diz
    # se a mensagem pede as outras obras dele (build_focused_author_prompt).
    CHAT_FOCUSED_ARTIFACT_AUTHOR: bool = False
    # Retrato do autor na ficha (image_url e créditos), lido de AUTHOR_PORTRAITS_PATH.
    # Os retratos vêm da Wikidata/Wikimedia Commons (docs/retratos-dos-autores.md).
    CHAT_AUTHOR_PORTRAITS: bool = False
    AUTHOR_PORTRAITS_PATH: str | None = None
    # Quando a resposta leva a ficha do autor e o visitante quer as obras desse
    # autor (mais uma pergunta curta ao modelo de pré-processamento, medida em
    # tests/author_card_cases.py), os cartões passam a ser só de peças dele, na
    # primeira página e nas seguintes («ver mais resultados»). Precisa de
    # CHAT_AUTHOR_CARD. O texto da resposta não muda.
    CHAT_AUTHOR_ONLY_RESULTS: bool = False
    # Com CHAT_AUTHOR_ONLY_RESULTS: quando os cartões mudam (saem os de outros
    # autores, entram mais do autor), a resposta escreve-se outra vez sobre os
    # cartões que ficam. Sem isto o texto falava de peças que já não estavam à
    # vista, ou de menos peças do que os cartões. Custa mais uma geração da AMÁLIA.
    CHAT_AUTHOR_ONLY_NEW_ANSWER: bool = False
    # O texto da resposta fala de todas as peças dos cartões mostrados, uma a uma,
    # em vez de destacar até três (docs/todos-os-cartoes.md).
    CHAT_ANSWER_ALL_CARDS: bool = False
    # Analítica do acervo por regras: «qual é a peça mais antiga/recente?»,
    # «quantas peças há nesta visita?», «de que século é a maioria?», «que autor
    # tem mais obras?». Calculado no OpenSearch (datas normalizadas a 2026-10-12);
    # resposta por template com os factos e cartões, sem LLM, salvo com a
    # segunda flag, que manda o LLM redigir a partir dos mesmos factos.
    CHAT_CATALOGUE_ANALYTICS: bool = False
    CHAT_CATALOGUE_ANALYTICS_CARDS: int = 24
    CHAT_CATALOGUE_ANALYTICS_LLM_WORDING: bool = False
    # «De que século é a maioria das peças?» conta pelo campo start_century do
    # índice em vez de um histograma de start_year. O histograma corta os séculos
    # em …, -199, -99, 1, 101, …, o que está certo depois de Cristo e errado antes:
    # punha o ano -500 no século VI a.C. e o -100 no II a.C. Depois de Cristo o
    # resultado é o mesmo. Precisa dos campos do século no índice
    # (patrimonio360_indexer, atualizar_datas_v4.py).
    CHAT_CATALOGUE_ANALYTICS_CENTURY_FIELD: bool = False
    # Tesouros Nacionais (o nationalTreasure dos mapas dos POIs em poi_tours/,
    # relidos quando mudam). O campo national_treasure dos cartões vai sempre;
    # esta flag liga os pedidos («tesouros nacionais de pintura» filtra pela
    # lista; «isto é tesouro nacional?» responde pela lista) e a marca nos
    # cartões do prompt, com a regra de nunca afirmar um tesouro fora dela.
    CHAT_NATIONAL_TREASURES: bool = False
    # EXPERIMENTAL, manter desligado: medido em 2026-09-15 sem ganho de tempo
    # (o 9B enche filters_delta com filtros inventados). Ver PLANO_MELHORIAS §3.
    CHAT_ROUTER_COMPACT_SCHEMA: bool = False        # router pede 5 campos em vez de 12
    API_PREFIX: str = "/api/v1"

    BACKEND_LOG_ENABLED: bool = True
    BACKEND_LOG_LEVEL: str = "INFO"
    BACKEND_ACCESS_LOG_ENABLED: bool = True
    BACKEND_LOG_HEALTHCHECKS: bool = False
    BACKEND_RAG_DEBUG_ENABLED: bool = True
    BACKEND_RAG_DEBUG_MAX_CHARS: int = 40000

    CORS_ALLOW_ORIGINS: str = "*"
    CORS_ALLOW_METHODS: str = "*"
    CORS_ALLOW_HEADERS: str = "*"

    OPENSEARCH_HOST: str | None = None
    OPENSEARCH_PORT: int = 9200
    OPENSEARCH_USERNAME: str | None = None
    OPENSEARCH_PASSWORD: str | None = None
    OPENSEARCH_SCHEME: str = "https"
    OPENSEARCH_USE_SSL: bool | None = None
    OPENSEARCH_VERIFY_CERTS: bool = False
    OPENSEARCH_SSL_SHOW_WARN: bool = False

    # Defaults reflect the live production indexes; .env still overrides.
    OPENSEARCH_INDEX_ARTIFACT: str = "cultural_heritage_artifacts"
    OPENSEARCH_INDEX_IMAGE: str = "cultural_heritage_images"
    OPENSEARCH_INDEX_MUSEUM: str = "cultural_heritage_museums"
    # Indices das entidades relacionais (autores, conjuntos, exposicoes).
    # Geridos pelo patrimonio360_indexer; usados para enriquecer o modal de
    # artefacto com a info da entidade + lista de outros objetos relacionados.
    OPENSEARCH_INDEX_AUTOR: str = "cultural_heritage_authors"
    OPENSEARCH_INDEX_CONJUNTO: str = "cultural_heritage_sets"
    OPENSEARCH_INDEX_EXPOSICAO: str = "cultural_heritage_exhibitions"
    # Top N de artefactos por conjunto / exposicao a mostrar no modal.
    CHAT_RELATED_ARTIFACTS_TOP_K: int = 24

    # Defaults reflect the production embedding stack (Qwen3-Embedding-4B /
    # Qwen3-VL-Embedding-2B); .env still overrides.
    ARTIFACT_TEXT_EMBEDDING_DIMENSION: int = 2560
    ARTIFACT_MULTIMODAL_EMBEDDING_DIMENSION: int = 2048
    MUSEUM_TEXT_EMBEDDING_DIMENSION: int = 2560
    IMAGE_MULTIMODAL_EMBEDDING_DIMENSION: int = 2048

    QWEN_TEXT_EMBEDDING_MODEL_ID: str = "Qwen/Qwen3-Embedding-4B"
    QWEN_MULTIMODAL_EMBEDDING_MODEL_ID: str = "Qwen/Qwen3-VL-Embedding-2B"
    # Optional pinned HF revisions (snapshot hashes). Empty = unpinned, which
    # preserves the pre-existing loading behavior; the v4/8B deploy pins these
    # to the snapshots used by the index rebuild (see indexer .env.example).
    QWEN_TEXT_EMBEDDING_MODEL_REVISION: str = ""
    QWEN_MULTIMODAL_EMBEDDING_MODEL_REVISION: str = ""
    USE_OPENROUTER_BGE_M3: bool = False
    OPENROUTER_API_KEY: str | None = None
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_BGE_MODEL: str = "baai/bge-m3"
    # Backward compatibility with previous backend env naming.
    TEXT_EMBEDDING_MODEL: str | None = None
    MULTIMODAL_EMBEDDING_MODEL: str | None = None
    EMBEDDING_PREFER_BF16: bool = True
    EMBEDDING_MAX_LENGTH: int = 2048
    TEXT_EMBEDDING_BATCH_SIZE: int = 2
    MULTIMODAL_TEXT_EMBEDDING_BATCH_SIZE: int = 8
    MULTIMODAL_IMAGE_EMBEDDING_BATCH_SIZE: int = 4

    LLM_PROVIDER: str = "openai_compatible"
    LLM_BASE_URL: str = "https://amalia.novasearch.org/vlm/chat/completions"
    LLM_API_KEY: str | None = None
    LLM_MODEL: str = "carminho/AMALIA-9B-50-DPO"
    # Backward-compatible legacy settings; ignored when LLM_MODEL is set.
    LLM_MODEL_TEXT: str | None = None
    LLM_MODEL_JSON: str | None = None
    LLM_TEMPERATURE_TEXT: float = 0.4
    LLM_TEMPERATURE_JSON: float = 0.0
    # 0 means no server-side max token cap from this API layer.
    LLM_MAX_TOKENS: int = 0
    LLM_MAX_TOKENS_TEXT: int | None = None
    LLM_MAX_TOKENS_JSON: int | None = None
    LLM_TIMEOUT_SECONDS: float = 45.0
    # LLM local (vLLM, deploy/llm/amalia_llm.sh) com a API remota como reserva:
    # se o servidor primário recusar a ligação (em baixo, a arrancar) o mesmo
    # pedido segue para o fallback. Vazio = sem fallback (comportamento antigo).
    LLM_CONNECT_TIMEOUT_SECONDS: float = 5.0
    LLM_FALLBACK_BASE_URL: str = ""
    LLM_FALLBACK_MODEL: str = ""
    LLM_FALLBACK_API_KEY: str | None = None

    CHAT_HISTORY_WINDOW: int = 8
    CHAT_SESSION_TTL_SECONDS: int = 3600
    CHAT_ROLLING_SUMMARY_MAX_CHARS: int = 600
    CHAT_ENABLE_RAG: bool = True
    CHAT_ENABLE_LLM_LEXICAL_QUERY: bool = True
    CHAT_PREWARM_ON_STARTUP: bool = False
    CHAT_PREWARM_INCLUDE_MULTIMODAL: bool = True
    CHAT_PREWARM_INCLUDE_MULTIVIEW_WORKER: bool = False
    CHAT_USE_QUERY_EMBEDDINGS: bool = True
    CHAT_RETRIEVAL_EMBEDDING_ONLY: bool = False
    CHAT_RETRIEVAL_CANDIDATES: int = 15
    CHAT_RETRIEVAL_TOP_K: int = 5
    CHAT_RETRIEVAL_RESULTS_PAGE_SIZE: int = 10
    CHAT_INCLUDE_ORIGIN_HISTORY_IN_LLM_CONTEXT: bool = True
    CHAT_RELATED_ARTIFACTS_PAGE_SIZE: int = 10
    CHAT_RETRIEVAL_PAGINATION_WINDOW: int = 150
    CHAT_IN_TOUR_BOOST: float = 5
    # Legacy (sem efeito desde a Fase 3/E7A): as queries visuais deixaram de
    # transportar preferencia in_tour; a preferencia passa a ser pos-fusao
    # (MULTIMODAL_IN_TOUR_MARGIN, Etapa 10). Declarado para paridade de .env.
    IMAGE_IN_TOUR_BOOST: float = 5
    CHAT_IMAGE_RETRIEVAL_TOP_K: int = 6
    CHAT_IMAGE_ARTIFACT_TOP_K: int = 5
    CHAT_IMAGE_RETRIEVAL_PAGINATION_WINDOW: int = 150
    CHAT_IMAGE_DEFAULT_MESSAGE: str = "Analisa a imagem e identifica a peça mais provável no museu."
    CHAT_MODEL_DEFAULT_MESSAGE: str = "Analisa este modelo 3D e identifica a peça mais provável no museu."
    CHAT_MODEL_FIRST_PASS_VIEWS: int = 3
    CHAT_MODEL_TOTAL_VIEWS: int = 5
    CHAT_MODEL_LOW_CONFIDENCE_SCORE_THRESHOLD: float = 0.35
    CHAT_MODEL_CACHE_SIZE: int = 12

    # --- Fase 3: retrieval multimodal de produção (tudo atrás de flags) ---
    # off    -> comportamento atual, sem ramo visual nem alteração de ranking.
    # intent -> ramo texto→imagem só quando a intenção visual é identificada.
    # always -> ramo texto→imagem em todas as pesquisas textuais (avaliação).
    MULTIMODAL_RETRIEVAL_MODE: str = "off"
    MULTIMODAL_RRF_K: int = 60
    MULTIMODAL_ARTIFACT_WEIGHT: float = 1.0
    MULTIMODAL_IMAGE_WEIGHT: float = 0.7
    # Floor de score do ramo visual antes da fusão. Calibrado 2026-07-13 com
    # dados reais (12 positivos com alvo + 12 queries factuais como negativos):
    # positivos observados >= 0.6677; amostra visual >= 0.645; negativos top-1
    # mediana 0.682/max 0.712 — o espaço cross-modal é estreito e o score tem
    # fraco poder separador, por isso o floor corta apenas a cauda fraca
    # (< 0.64) e a qualidade é garantida pelo router + pesos RRF por rank.
    MULTIMODAL_MIN_IMAGE_SCORE: float = 0.64
    MULTIMODAL_IMAGE_TOP_K: int = 30
    MULTIMODAL_DEBUG: bool = False
    # Pesos da fusão imagem+texto (Etapa 8): a semelhança visual (i2i) é o
    # ramo principal; o texto refina via t2i (MULTIMODAL_IMAGE_WEIGHT) e/ou
    # pesquisa documental de artefactos com peso reduzido.
    MULTIMODAL_I2I_WEIGHT: float = 1.0
    MULTIMODAL_IMAGE_TEXT_ARTIFACT_WEIGHT: float = 0.5
    # E9: ramos múltiplos numa mensagem usam _msearch (erros por ramo isolados;
    # fallback automático para queries separadas em falha transport-level).
    MULTIMODAL_USE_MSEARCH: bool = True
    # E10: preferência in_tour APÓS a fusão — um resultado in_tour sobe no
    # máximo uma posição quando a diferença de fusion_score é <= margem.
    # 0.0 desliga a política. Nunca altera scores nem introduz candidatos.
    MULTIMODAL_IN_TOUR_MARGIN: float = 0.0
    # Ramo conjunto imagem+texto (imagem carregada + texto do utilizador num
    # ÚNICO embedding Qwen3-VL contra visual_embedding). Só corre quando a
    # mensagem tem imagem E texto não vazio; nunca substitui os ramos atuais.
    MULTIMODAL_JOINT_IMAGE_TEXT_ENABLED: bool = False
    MULTIMODAL_JOINT_IMAGE_TEXT_TOP_K: int = 30
    MULTIMODAL_JOINT_IMAGE_TEXT_WEIGHT: float = 1.0
    # Posição do rerank VL na rota de imagem:
    # local  -> só o pool i2i pré-fusão (comportamento histórico).
    # global -> pool fundido pós-RRF (todas as origens; recomendado).
    # both   -> ambos (ablação). off -> nenhum, mesmo com o flag VL ligado.
    # CHAT_ENABLE_VL_RERANKING=false continua a desligar tudo (master switch).
    MULTIMODAL_VL_RERANK_STAGE: str = "local"
    MULTIMODAL_GLOBAL_VL_RERANK_TOP_N: int = 16
    # Experiência das legendas (../patrimonio360_caption_indexer): nas perguntas
    # escritas com intenção visual, um ramo texto→legendas (kNN no campo
    # caption_embedding, o mesmo Qwen3-Embedding-8B do ramo de texto). Vazio
    # desliga. replace -> as legendas ocupam o lugar do ramo texto→imagem (a
    # variante A medida a 2026-09-30); add -> correm os dois. Se o ramo das
    # legendas falhar, fica o texto→imagem. As fotografias não mudam.
    CAPTION_TEXT_INDEX: str = ""
    CAPTION_TEXT_MODE: str = "replace"
    CAPTION_TEXT_WEIGHT: float = 0.7
    CAPTION_TEXT_TOP_K: int = 30
    # O índice das legendas tem todos os museus e o filtro de museu vem depois
    # do kNN, por isso o k é largo (como na avaliação: --k 1000).
    CAPTION_TEXT_KNN_K: int = 1000
    # CAPTION_TEXT_INDEX aceita vários índices separados por vírgulas (o do corpus
    # do benchmark e os dos museus legendados depois, como p360_legendas_mnaa_caption8b).
    # Nem todos os museus têm legendas (a 2026-11-19 falta o CMAG; o MNAA foi
    # legendado nesse dia). Em replace, um museu sem legendas ficava sem ramo visual
    # nenhum: lista vazia não é falha, e o texto→imagem não corria. Com esta
    # flag, um museu sem nenhuma legenda no índice salta o ramo das legendas e
    # usa o texto→imagem, como antes da experiência. Um museu com legendas que
    # devolva zero resultados fica como estava (baseline).
    CAPTION_TEXT_UNCOVERED_MUSEUM_T2I: bool = False
    # A pergunta escrita é embebida com a instrução «query» do Qwen3-Embedding
    # antes de ir às legendas, como já acontece no ramo das fichas
    # (CHAT_RETRIEVAL_QUERY_INSTRUCTION). Desligada, vai sem instrução, como na
    # variante medida a 2026-09-30.
    CAPTION_TEXT_QUERY_INSTRUCTION: bool = False
    # Perguntas pelo aspeto («estátuas brancas», «peças com cavalos»), 2026-11-19.
    # As fichas quase nunca dizem a cor nem o que se vê (no MNAA, 293 das 299
    # esculturas não têm material), e o modelo que escolhe os cartões só via as
    # fichas: rejeitava tudo. Cada flag é independente e, desligada, nada muda.
    # CHAT_VISUAL_CARD_FACTS: numa pesquisa em que o ramo das legendas trouxe
    # peças, os cartões dados ao modelo de pré-processamento e à AMÁLIA levam
    # uma linha «Aspeto na fotografia: …» com o que a legenda da peça diz
    # (objeto, materiais, cores, motivos, forma).
    CHAT_VISUAL_CARD_FACTS: bool = False
    # Com CHAT_VISUAL_CARD_FACTS: quantas peças que só o ramo das legendas
    # encontrou são também julgadas, pela ordem das legendas. Na fusão ficam atrás
    # de todas as do texto e nunca chegavam a ser vistas. São julgadas à parte,
    # só pelo aspeto (build_visual_pick_prompt, «sim» ou «não» por peça), em
    # paralelo com a pré-seleção, e as aceites juntam-se aos cartões escolhidos.
    # Com 20, a pergunta demora cerca de 1,5 s. Precisa de CHAT_PRESELECT_CARDS.
    CHAT_VISUAL_CARD_CANDIDATES: int = 0
    # CHAT_VISUAL_INTENT_MODEL: quando as regras (retrieval/visual_intent.py)
    # não veem sinal visual, o modelo de pré-processamento diz se o visitante
    # descreve as peças pelo aspeto (build_visual_request_prompt); se sim, o
    # ramo visual corre. As regras que já acertam continuam a mandar.
    CHAT_VISUAL_INTENT_MODEL: bool = False
    # CHAT_VISUAL_ANSWER_RULE: quando os cartões da resposta levam o aspeto, a
    # AMÁLIA não recebe o texto do pedido: só lhe é dito que o visitante descreveu
    # o aspeto das peças e que deve dizer o que cada cartão mostra. Com o pedido à
    # vista, atribuía a cada peça o que o visitante pediu («um colar subtil» numa
    # mulher só com brincos): 21 de 70 respostas, contra 4 de 70 sem ele.
    CHAT_VISUAL_ANSWER_RULE: bool = False

    @field_validator("CAPTION_TEXT_MODE")
    @classmethod
    def _validate_caption_text_mode(cls, value: str) -> str:
        normalized = (value or "").strip().lower() or "replace"
        if normalized not in {"replace", "add"}:
            raise ValueError(f"CAPTION_TEXT_MODE must be one of ['add', 'replace']; got {value!r}")
        return normalized

    @field_validator("CHAT_ROUTER_MODE")
    @classmethod
    def _validate_router_mode(cls, value: str) -> str:
        normalized = (value or "").strip().lower() or "llm"
        if normalized not in {"llm", "deterministic"}:
            raise ValueError(
                f"CHAT_ROUTER_MODE must be one of ['deterministic', 'llm']; got {value!r}"
            )
        return normalized

    @field_validator("OPENSEARCH_USE_SSL", mode="before")
    @classmethod
    def _blank_use_ssl_is_auto(cls, value: object) -> object:
        # O .env.example documenta «vazio = automático», mas o pydantic rejeitava
        # a string vazia e um .env copiado do exemplo não arrancava.
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("CHAT_GROUNDING_CHECK")
    @classmethod
    def _validate_grounding_check(cls, value: str) -> str:
        normalized = (value or "").strip().lower() or "off"
        if normalized not in {"off", "log", "repair"}:
            raise ValueError(
                f"CHAT_GROUNDING_CHECK must be one of ['log', 'off', 'repair']; got {value!r}"
            )
        return normalized

    @field_validator("MULTIMODAL_VL_RERANK_STAGE")
    @classmethod
    def _validate_multimodal_vl_rerank_stage(cls, value: str) -> str:
        normalized = (value or "").strip().lower()
        allowed = {"off", "local", "global", "both"}
        if normalized not in allowed:
            raise ValueError(
                f"MULTIMODAL_VL_RERANK_STAGE must be one of {sorted(allowed)}; got {value!r}"
            )
        return normalized

    @field_validator("MULTIMODAL_JOINT_IMAGE_TEXT_TOP_K")
    @classmethod
    def _validate_multimodal_joint_top_k(cls, value: int) -> int:
        if not 1 <= value <= 500:
            raise ValueError(
                f"MULTIMODAL_JOINT_IMAGE_TEXT_TOP_K must be within [1, 500]; got {value}"
            )
        return value

    @field_validator("MULTIMODAL_GLOBAL_VL_RERANK_TOP_N")
    @classmethod
    def _validate_multimodal_global_rerank_top_n(cls, value: int) -> int:
        if not 1 <= value <= 100:
            raise ValueError(
                f"MULTIMODAL_GLOBAL_VL_RERANK_TOP_N must be within [1, 100]; got {value}"
            )
        return value

    @field_validator("MULTIMODAL_IN_TOUR_MARGIN")
    @classmethod
    def _validate_multimodal_in_tour_margin(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"MULTIMODAL_IN_TOUR_MARGIN must be within [0, 1]; got {value}"
            )
        return value

    @field_validator("MULTIMODAL_RETRIEVAL_MODE")
    @classmethod
    def _validate_multimodal_mode(cls, value: str) -> str:
        normalized = (value or "").strip().lower()
        allowed = {"off", "intent", "always"}
        if normalized not in allowed:
            raise ValueError(
                f"MULTIMODAL_RETRIEVAL_MODE must be one of {sorted(allowed)}; got {value!r}"
            )
        return normalized

    @field_validator("MULTIMODAL_RRF_K")
    @classmethod
    def _validate_multimodal_rrf_k(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"MULTIMODAL_RRF_K must be >= 1; got {value}")
        return value

    @field_validator(
        "MULTIMODAL_ARTIFACT_WEIGHT",
        "MULTIMODAL_IMAGE_WEIGHT",
        "MULTIMODAL_I2I_WEIGHT",
        "MULTIMODAL_IMAGE_TEXT_ARTIFACT_WEIGHT",
        "MULTIMODAL_JOINT_IMAGE_TEXT_WEIGHT",
    )
    @classmethod
    def _validate_multimodal_weights(cls, value: float) -> float:
        if value < 0:
            raise ValueError(f"multimodal fusion weights must be >= 0; got {value}")
        return value

    @field_validator("MULTIMODAL_MIN_IMAGE_SCORE")
    @classmethod
    def _validate_multimodal_min_image_score(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"MULTIMODAL_MIN_IMAGE_SCORE must be within [0, 1]; got {value}"
            )
        return value

    @field_validator("MULTIMODAL_IMAGE_TOP_K")
    @classmethod
    def _validate_multimodal_image_top_k(cls, value: int) -> int:
        if not 1 <= value <= 500:
            raise ValueError(
                f"MULTIMODAL_IMAGE_TOP_K must be within [1, 500]; got {value}"
            )
        return value

    # Vocabulário leigo -> catálogo (app/config/lay_vocabulary.json): frases do
    # catálogo como alternativa no ramo lexical. Desligado = query histórica.
    CHAT_LAY_VOCABULARY_EXPANSION: bool = False
    IMAGE_ASSET_ROOT: str | None = None
    # Miniaturas `?w=` em /chat/images (ver app/services/image_thumbnails.py).
    # Desligado: ignora `w` e serve sempre o original, sem Cache-Control.
    CHAT_IMAGE_THUMBNAILS: bool = False
    IMAGE_THUMBNAIL_CACHE_DIR: str | None = None
    CHAT_IMAGE_CACHE_MAX_AGE_SECONDS: int = 86400
    POI_TOURS_DIR: str | None = None
    # Lugares das visitas 3DVista (scripts/build_tour_locations.py); vazio = app/data/tour_locations.json
    TOUR_LOCATIONS_PATH: str | None = None
    MULTIVIEW_WORKER_HOST: str = "127.0.0.1"
    MULTIVIEW_WORKER_PORT: int = 3101
    MULTIVIEW_WORKER_START_TIMEOUT_SECONDS: float = 30.0
    # Render HTTP timeout. Kept at the historical 60s floor by default so a
    # broken/hung render fails FAST and the worker recovers, rather than making
    # the user wait. Operators can raise it once GPU-accelerated rendering (or a
    # known-good Chromium) makes genuinely-slow renders viable.
    MULTIVIEW_RENDER_TIMEOUT_SECONDS: float = 60.0
    MULTIVIEW_RENDER_SIZE: int = 512
    MULTIVIEW_RENDER_BACKGROUND: str = "#FFFFFF"
    MULTIVIEW_RENDER_FOV: int = 35
    MULTIVIEW_RENDER_DPR: float = 1.0
    MULTIVIEW_RENDER_STRATEGY: str = "adaptive"
    MULTIVIEW_RENDER_OVERSAMPLE: int = 320
    MULTIVIEW_RENDER_ORBIT_MARGIN: float = 1.35
    MULTIVIEW_RENDER_ENSURE_TOP: bool = True
    MULTIVIEW_RENDER_DELAY_MS: int = 8
    MULTIVIEW_SAVE_LAST_VIEWS: bool = True
    MULTIVIEW_LAST_VIEWS_DIR: str = "tmp/multiview_last_views"

    # Structured backend_query JSONL logging for offline evaluation/paper analysis.
    QUERY_LOG_ENABLED: bool = True
    QUERY_LOG_PATH: str = "logs/evaluation/backend_queries.jsonl"
    FRONTEND_EVENT_LOG_PATH: str = "logs/evaluation/frontend_events.jsonl"

    # «Apontar e perguntar» (metadata.intent point_and_ask):
    # a vista reconstrói-se das faces do cubo da visita, o SAM 3 recorta o
    # objeto tocado, o Qwen3-VL-Embedding procura-o no índice de imagens e o
    # escolhido segue o fluxo describe_artifact. Desligado, a intenção segue
    # como uma mensagem normal.
    POINT_AND_ASK_ENABLED: bool = False
    POINT_TOURS_BASE_URL: str = "https://patrimonio360.webredirect.org:5500"
    # tiles das faces: vazio = tmp/visitas (a mesma pasta do obter_poi_mapping.sh)
    POINT_TILES_CACHE_DIR: str = ""
    POINT_FACE_LEVEL: int = 1           # nível 1 = 3×3 tiles de 512 px (1536 px por face)
    POINT_FACE_TILES: int = 3
    POINT_VIEW_WIDTH: int = 1024
    # o recorte faz-se numa segunda vista, apertada à volta do objeto e com as
    # faces no nível mais fino (0 = 6×6 tiles, 3072 px), para o embedding ver o
    # objeto com resolução e não um recorte de 140 px da vista geral
    POINT_CROP_FACE_LEVEL: int = 0
    POINT_CROP_FACE_TILES: int = 6
    POINT_CROP_WIDTH: int = 640
    POINT_SAM_MODEL: str = "facebook/sam3"
    POINT_SAM_DEVICE: str = "cuda"
    # SAM 3 por conceito: a peça inteira que contém o ponto («artwork» dá o
    # quadro; o ponto só dava o cisne pintado). Sem instância, cai no ponto.
    POINT_CONCEPT_PROMPTS: str = "artwork,furniture,tapestry,garment"
    POINT_CONCEPT_THRESHOLD: float = 0.3
    POINT_CROP_MARGIN: float = 0.25     # folga da vista apertada à volta da máscara
    # score normalizado 0..1 (cosseno do Qwen3-VL-Embedding) a partir do qual a peça se dá por identificada com
    # certeza; abaixo apresenta-se na mesma, sem certeza («parece ser…»). Com pelo menos um candidato nunca
    # se diz «não encontrei» (pedido do utilizador, 2026-10-20); o chão só serve se se quiser voltar a isso.
    POINT_MATCH_THRESHOLD: float = 0.70
    POINT_LOW_CONFIDENCE_FLOOR: float = 0.0
    # Sem boosts (1.0): o score é só o cosseno. Medido a 2026-10-19 no MNT: o ×1,15
    # «na visita» pôs um «Elemento de Vestido» com hotspot (0,75) à frente do vestido
    # certo sem hotspot (0,84); o printscreen, sem boosts, acertava.
    POINT_IN_TOUR_BOOST: float = 1.0
    POINT_PANORAMA_BOOST: float = 1.0
    POINT_CANDIDATES: int = 3
    POINT_SEARCH_K: int = 24            # imagens pedidas ao kNN antes de agrupar por peça
    POINT_ASSET_TTL_SECONDS: int = 600  # crop/cutout em memória, por request_id, para a legenda
    # pasta de depuração: guarda vista, crop e cutout de cada toque (só em testes)
    POINT_DEBUG_DIR: str = ""
    # O modelo de visão confirma a peça entre as melhores candidatas (POINT_VERIFY_TOP_K):
    # confirma uma → é essa, com certeza; nenhuma → «não encontro esta peça no acervo» e os
    # cartões das parecidas. A escolha do VLM só vale se a candidata tiver cosseno ≥
    # POINT_VERIFY_MIN_SCORE ou hotspot neste panorama (medido 2026-10-28: as escolhas
    # erradas eram decoração do edifício abaixo de 0,6). Sem VLM a responder, o fluxo é o de sempre.
    POINT_VERIFY_ENABLED: bool = False
    POINT_VERIFY_BASE_URL: str = "http://127.0.0.1:8005/v1"
    POINT_VERIFY_MODEL: str = "Qwen/Qwen3-VL-8B-Instruct-FP8"
    POINT_VERIFY_API_KEY: str = ""
    POINT_VERIFY_TIMEOUT_SECONDS: float = 6.0
    POINT_VERIFY_TOP_K: int = 3
    POINT_VERIFY_MIN_SCORE: float = 0.60
    POINT_VERIFY_MIN_CONFIDENCE: float = 0.6
    # «Ver na tour» também para as peças sem hotspot que a indexação offline
    # localizou num panorama (poi_tours/panorama-objects-<museu>.json). Só as
    # localizações definitivas: confirmadas pelo VLM ou por hotspot, ou aceites
    # só pelo cosseno com score ≥ TOUR_OBJECT_TARGETS_MIN_SCORE (os ficheiros
    # dos museus ainda sem verificação, 2026-10-27). O alvo leva yaw/pitch e a
    # visita vira a câmara para a peça (2026-10-31).
    TOUR_OBJECT_TARGETS_ENABLED: bool = False
    TOUR_OBJECT_TARGETS_MIN_SCORE: float = 0.80
    # Percursos temáticos gerados em conversa (2026-11-09, app/services/route_planning/).
    # Tudo atrás de CHAT_ROUTE_PLANNING: com a flag desligada nenhuma resposta muda.
    CHAT_ROUTE_PLANNING: bool = False
    # F12 do Bloco A: os séculos e anos propostos pelo LLM do tema só entram
    # quando o pedido fala de época (século, anos, um ano de 4 dígitos). Desligada,
    # qualquer dígito serve e «30 minutos» deixava entrar um século inventado.
    ROUTE_THEME_STRICT_TIME_MENTION: bool = False
    ROUTE_STOPS_INDEX: str = "cultural_heritage_tour_stops_v1"   # paragens enriquecidas (indexer)
    ROUTE_THEME_LLM: bool = True          # o LLM completa as facetas que as regras não apanharam (JSON)
    ROUTE_THEME_MAX_TOKENS: int = 600     # o JSON do tema não cabe nos LLM_MAX_TOKENS_JSON do chat (200 cortava-o)
    ROUTE_CANDIDATES: int = 120           # candidatos depois da fusão (chega para 50 paragens + focos)
    ROUTE_BRANCH_SIZE: int = 120          # resultados pedidos a cada ramo
    ROUTE_RRF_K: int = 60
    ROUTE_WEIGHT_LEXICAL: float = 1.0     # BM25 em enriched_text + descrição
    ROUTE_WEIGHT_SEMANTIC: float = 1.0    # kNN em enriched_embedding
    ROUTE_WEIGHT_IMAGE: float = 0.7       # texto→imagem, só nas peças com paragem
    ROUTE_WEIGHT_ATTRIBUTES: float = 1.0  # listas fechadas do VLM/catálogo que batem nas facetas
    ROUTE_MIN_FUSION_SCORE: float = 0.012 # ≈ um ramo de peso 1 na posição 20; abaixo não é «relevante»
    ROUTE_MIN_CANDIDATES: int = 3         # menos do que isto -> route: null e explicação
    ROUTE_DEFAULT_STOPS: int = 8          # sem número nem duração: mínimo do k por omissão
    ROUTE_DEFAULT_FRACTION: float = 0.6   # … k = round(candidatos relevantes × fração), entre DEFAULT_STOPS e DEFAULT_MAX_STOPS
    ROUTE_DEFAULT_MAX_STOPS: int = 20
    ROUTE_MIN_STOPS: int = 3
    ROUTE_MAX_STOPS: int = 50             # teto; há menos quando o tema não dá para tantas
    ROUTE_MINUTES_PER_STOP: float = 2.5
    ROUTE_YEAR_TOLERANCE: int = 25        # folga dos filtros duros de século/ano
    ROUTE_DIVERSITY_WEIGHT: float = 0.35  # bónus por sala/categoria/século novos na escolha gulosa
    ROUTE_PATH_COST_WEIGHT: float = 0.02  # custo por salto no grafo a partir da última paragem
    ROUTE_TTL_SECONDS: int = 7200         # cache dos percursos por route_id

    # Ditado por voz no servidor (WebSocket /api/v1/speech/stream, faster-whisper).
    # Desligado: o modelo não carrega e o WebSocket recusa a ligação, e o
    # frontend passa ao reconhecimento do browser.
    SPEECH_STT_ENABLED: bool = False
    SPEECH_MODEL: str = "large-v3-turbo"
    SPEECH_DEVICE: str = "cuda"
    SPEECH_COMPUTE_TYPE: str = "float16"
    # réplicas do modelo na GPU que transcrevem em paralelo (sessões em fila)
    SPEECH_NUM_WORKERS: int = 2
    SPEECH_MAX_SESSIONS: int = 6
    SPEECH_MAX_SESSION_SECONDS: float = 120.0
    SPEECH_PARTIAL_INTERVAL_MS: int = 400
    SPEECH_ENDPOINT_SILENCE_MS: int = 600
    # troço mais longo do que isto: confirmam-se as palavras estáveis e continua-se
    SPEECH_MAX_SEGMENT_SECONDS: float = 12.0
    SPEECH_NO_SPEECH_THRESHOLD: float = 0.6
    # vazio: speech_sessions.jsonl na pasta do QUERY_LOG_PATH (só metadados e texto final, nunca áudio)
    SPEECH_LOG_PATH: str = ""

    # Reranker (second-stage; disabled by default). These keys existed in .env but
    # were silently dropped by extra="ignore" until declared here — declaring them
    # is what makes PairwiseRerankerService constructible at all.
    CHAT_ENABLE_RERANKING: bool = False
    RERANKER_MODEL_ID: str = "Qwen/Qwen3-Reranker-4B"
    RERANKER_MODEL_REVISION: str | None = None
    RERANKER_MAX_LENGTH: int = 1024
    RERANKER_BATCH_SIZE: int = 10
    RERANKER_INSTRUCTION: str = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )
    RERANKER_PREFER_BF16: bool = True
    # Candidate pool reranked on the first page of text retrieval. When reranking
    # is enabled the OpenSearch request is widened to this size and truncated back
    # to the original page size after reordering.
    RERANKER_CANDIDATES: int = 24

    # Visual reranker (second-stage for image retrieval; disabled by default).
    # Scores (query image/text, candidate image+caption) pairs with Qwen3-VL-Reranker.
    CHAT_ENABLE_VL_RERANKING: bool = False
    VL_RERANKER_MODEL_ID: str = "Qwen/Qwen3-VL-Reranker-8B"
    VL_RERANKER_MODEL_REVISION: str | None = None
    VL_RERANKER_INSTRUCTION: str = (
        "Given a user query (image and/or text), retrieve museum artifact images "
        "that match the queried object"
    )
    VL_RERANKER_BATCH_SIZE: int = 4
    VL_RERANKER_CANDIDATES: int = 16
    # Candidate images are downscaled to this max side (px) before scoring.
    VL_RERANKER_MAX_IMAGE_SIDE: int = 1024

    # Declared for .env parity (currently without consumers in app code).
    CHAT_ENABLE_STRUCTURED_QUERY_PLANNING: bool = False
    CHAT_ANALYTICS_PLANNER_MIN_CONFIDENCE: float = 0.55
    CHAT_ANALYTICS_LIST_TOP_K: int = 8
    LOG_JSON: bool = False
    LOG_JSON_PRETTY: bool = False
    LOG_JSON_INDENT: int = 2
    LOG_LEVEL: str = "INFO"
    LOG_CHAT_MESSAGES: bool = False
    LOG_CHAT_STATE_HISTORY: bool = False
    DEBUG_EMBEDDINGS: bool = False

    @property
    def cors_allow_origins_list(self) -> list[str]:
        return _parse_csv(self.CORS_ALLOW_ORIGINS)

    @property
    def cors_allow_methods_list(self) -> list[str]:
        return _parse_csv(self.CORS_ALLOW_METHODS)

    @property
    def cors_allow_headers_list(self) -> list[str]:
        return _parse_csv(self.CORS_ALLOW_HEADERS)

    @property
    def opensearch_use_ssl_resolved(self) -> bool:
        if self.OPENSEARCH_USE_SSL is not None:
            return self.OPENSEARCH_USE_SSL
        return bool(self.OPENSEARCH_USERNAME and self.OPENSEARCH_PASSWORD)

    @property
    def llm_base_url_resolved(self) -> str:
        return self.LLM_BASE_URL.rstrip("/")

    @property
    def llm_openai_base_url_resolved(self) -> str:
        """Accept full chat-completions URL and convert it to OpenAI client base_url."""
        base = self.llm_base_url_resolved
        suffixes = ("/chat/completions",)
        for suffix in suffixes:
            if base.endswith(suffix):
                return base[: -len(suffix)].rstrip("/")
        return base

    @property
    def llm_fallback_openai_base_url_resolved(self) -> str | None:
        base = (self.LLM_FALLBACK_BASE_URL or "").strip().rstrip("/")
        if not base:
            return None
        if base.endswith("/chat/completions"):
            base = base[: -len("/chat/completions")].rstrip("/")
        return base

    @property
    def llm_model_resolved(self) -> str:
        if self.LLM_MODEL and self.LLM_MODEL.strip():
            return self.LLM_MODEL.strip()
        if self.LLM_MODEL_TEXT and self.LLM_MODEL_TEXT.strip():
            return self.LLM_MODEL_TEXT.strip()
        if self.LLM_MODEL_JSON and self.LLM_MODEL_JSON.strip():
            return self.LLM_MODEL_JSON.strip()
        return "amalia-llm/AMALIA-9B-0626-DPO"

    @property
    def llm_auth_header(self) -> dict[str, str]:
        if not self.LLM_API_KEY:
            return {}
        return {"Authorization": f"Bearer {self.LLM_API_KEY}"}

    @property
    def text_embedding_model_resolved(self) -> str:
        if self.TEXT_EMBEDDING_MODEL and self.TEXT_EMBEDDING_MODEL.strip():
            return self.TEXT_EMBEDDING_MODEL.strip()
        return self.QWEN_TEXT_EMBEDDING_MODEL_ID.strip()

    @property
    def multimodal_embedding_model_resolved(self) -> str:
        if self.MULTIMODAL_EMBEDDING_MODEL and self.MULTIMODAL_EMBEDDING_MODEL.strip():
            return self.MULTIMODAL_EMBEDDING_MODEL.strip()
        return self.QWEN_MULTIMODAL_EMBEDDING_MODEL_ID.strip()

    @property
    def reranker_model_resolved(self) -> str:
        raw = (self.RERANKER_MODEL_ID or "").strip()
        return raw or "Qwen/Qwen3-Reranker-4B"

    @property
    def reranker_model_revision_resolved(self) -> str | None:
        raw = (self.RERANKER_MODEL_REVISION or "").strip()
        return raw or None

    @property
    def vl_reranker_model_resolved(self) -> str:
        raw = (self.VL_RERANKER_MODEL_ID or "").strip()
        return raw or "Qwen/Qwen3-VL-Reranker-8B"

    @property
    def vl_reranker_model_revision_resolved(self) -> str | None:
        raw = (self.VL_RERANKER_MODEL_REVISION or "").strip()
        return raw or None

    @property
    def openrouter_base_url_resolved(self) -> str:
        return self.OPENROUTER_BASE_URL.strip().rstrip("/")

    @property
    def openrouter_bge_model_resolved(self) -> str:
        raw = self.OPENROUTER_BGE_MODEL.strip()
        if raw:
            return raw
        return "baai/bge-m3"

    @property
    def image_asset_root_resolved(self) -> Path | None:
        raw = (self.IMAGE_ASSET_ROOT or "").strip()
        if not raw:
            default_root = (Path(__file__).resolve().parents[2] / "Images").resolve()
            if default_root.exists():
                return default_root
            return None
        return Path(raw).expanduser().resolve()

    @property
    def image_thumbnail_cache_dir_resolved(self) -> Path:
        raw = (self.IMAGE_THUMBNAIL_CACHE_DIR or "").strip()
        if raw:
            return Path(raw).expanduser().resolve()
        return (Path(__file__).resolve().parents[2] / "tmp" / "image_thumbs").resolve()

    @property
    def tour_locations_path_resolved(self) -> Path:
        raw = (self.TOUR_LOCATIONS_PATH or "").strip()
        if raw:
            return Path(raw).expanduser().resolve()
        return (Path(__file__).resolve().parents[1] / "data" / "tour_locations.json").resolve()

    @property
    def author_portraits_path_resolved(self) -> Path:
        raw = (self.AUTHOR_PORTRAITS_PATH or "").strip()
        if raw:
            return Path(raw).expanduser().resolve()
        return (Path(__file__).resolve().parents[1] / "data" / "author_portraits.json").resolve()

    @property
    def poi_tours_dir_resolved(self) -> Path:
        raw = (self.POI_TOURS_DIR or "").strip()
        if raw:
            return Path(raw).expanduser().resolve()
        return (Path(__file__).resolve().parents[2] / "poi_tours").resolve()

    @property
    def multiview_last_views_dir_resolved(self) -> Path:
        raw = (self.MULTIVIEW_LAST_VIEWS_DIR or "").strip()
        if not raw:
            raw = "tmp/multiview_last_views"
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (Path(__file__).resolve().parents[2] / path).resolve()
        return path

    @property
    def query_log_path_resolved(self) -> Path:
        raw = (self.QUERY_LOG_PATH or "").strip()
        if not raw:
            raw = "logs/evaluation/backend_queries.jsonl"
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (Path(__file__).resolve().parents[2] / path).resolve()
        return path

    @property
    def point_tiles_cache_dir_resolved(self) -> Path:
        raw = (self.POINT_TILES_CACHE_DIR or "").strip()
        if raw:
            return Path(raw).expanduser().resolve()
        return (Path(__file__).resolve().parents[2] / "tmp" / "visitas").resolve()

    @property
    def speech_log_path_resolved(self) -> Path:
        raw = (self.SPEECH_LOG_PATH or "").strip()
        if not raw:
            return self.query_log_path_resolved.parent / "speech_sessions.jsonl"
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (Path(__file__).resolve().parents[2] / path).resolve()
        return path

    @property
    def frontend_event_log_path_resolved(self) -> Path:
        raw = (self.FRONTEND_EVENT_LOG_PATH or "").strip()
        if not raw:
            raw = "logs/evaluation/frontend_events.jsonl"
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (Path(__file__).resolve().parents[2] / path).resolve()
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    from app.services.retrieval_profile import apply_profile
    return apply_profile(Settings())
