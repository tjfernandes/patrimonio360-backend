# Património 360 — backend

API do assistente de museus do Património 360. É uma aplicação FastAPI que:

- responde às perguntas dos visitantes sobre as coleções, com pesquisa híbrida no OpenSearch (texto e imagem) e respostas do LLM AMALIA;
- pesquisa por fotografia e por modelo 3D;
- liga as respostas às visitas virtuais 3DVista: «ver na visita», «onde estou?» e a peça aberta na visita.

O frontend está em [tjfernandes/patrimonio360-frontend](https://github.com/tjfernandes/patrimonio360-frontend), e o site em produção é https://patrimonio360-amalia-frontend.web.app. Uma mudança no contrato da API (as rotas `/api/v1/chat/...` e o formato das respostas) tem de ser combinada com esse repositório.

## Em produção

```text
browser
  └─► nginx no VPS, https://patrimonio360.webredirect.org:8081
        └─► túnel SSH inverso (autossh): VPS 127.0.0.1:8001 → esta máquina 127.0.0.1:8001
              └─► uvicorn app.main:app em 127.0.0.1:8001 (WSL Ubuntu 24.04, conda visitas_virtuais)
                    ├─► vLLM local em 127.0.0.1:8004: carminho/AMALIA-9B-50-DPO (conda hbim-llm)
                    │     └─ se falhar: a API remota em LLM_FALLBACK_BASE_URL
                    └─► OpenSearch em https://patrimonio360.webredirect.org:9201 (índices v4)
```

- O uvicorn só escuta em `127.0.0.1`. Não há nada na porta 8000 e o backend não está exposto na Tailscale (`100.79.39.115`). De fora, só se chega pelo nginx do VPS.
- Os embedders correm dentro do processo do backend, na GPU, e ocupam cerca de 33 GB: Qwen3-Embedding-8B para texto e Qwen3-VL-Embedding-8B para imagem, com vetores de 4096 dimensões. O vLLM ocupa cerca de 29 GB.
- Os rerankers estão desligados em produção (`CHAT_ENABLE_RERANKING` e `CHAT_ENABLE_VL_RERANKING` a `false`).
- Índices que o backend lê:

  | variável | índice |
  |---|---|
  | `OPENSEARCH_INDEX_ARTIFACT` | `cultural_heritage_artifacts_v4` |
  | `OPENSEARCH_INDEX_IMAGE` | `cultural_heritage_images_v4` |
  | `OPENSEARCH_INDEX_AUTOR` | `cultural_heritage_authors_v4` |
  | `OPENSEARCH_INDEX_CONJUNTO` | `cultural_heritage_sets_v4` |
  | `OPENSEARCH_INDEX_EXPOSICAO` | `cultural_heritage_exhibitions_v4` |
  | `OPENSEARCH_INDEX_MUSEUM` | `cultural_heritage_museums_v1` |

  Os índices v4 têm os sete museus: mnt, mnaz, mj, mnsr, cmag, mnc e mnaa.

### O nginx da porta 8081, no VPS

O `server` da porta 8081 passa tudo para `http://127.0.0.1:8001`, que é a ponta do túnel. Usa `proxy_buffering off` por causa do streaming SSE, `proxy_read_timeout 300s` e `client_max_body_size 100M`. Há exceções a que o próprio nginx responde:

- `/health`, `/api/v1/health`, `/docs`, `/redoc` e `/openapi.json` dão 404, de propósito;
- `/api/v1/chat/health` dá 204 com cabeçalhos CORS.

Por isso, o estado do processo vê-se nesta máquina, com `curl http://127.0.0.1:8001/health`.

## Como corre nesta máquina

O supervisor `deploy/amalia_online.sh` arranca e vigia os três elos: o LLM local, o uvicorn e o túnel.

```bash
cd ~/Projects/AMALIA/patrimonio360-backend
deploy/amalia_online.sh            # verifica e repara o que estiver em baixo
deploy/amalia_online.sh --status   # só diagnostica
deploy/amalia_online.sh --vigiar   # vigia de 60 em 60 s e repara sozinho
```

O supervisor lança o uvicorn a partir da raiz do repositório, porque o `config.py` lê o `.env` da pasta atual:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Para reiniciar só o backend, que demora cerca de 2 minutos entre os imports e o pré-aquecimento dos embedders:

```bash
pkill -f "uvicorn app.main:app --host 127.0.0.1 --port 8001"
deploy/amalia_online.sh
```

O LLM local tem o seu próprio script, `deploy/llm/amalia_llm.sh`, com `--status` e `--stop`.

Registos:

- `~/.amalia_backend.log` (uvicorn), `~/.amalia_tunnel.log` (autossh) e `~/.amalia_llm.log` (vLLM);
- `logs/production/`: os pedidos e os eventos do frontend (`QUERY_LOG_PATH` e `FRONTEND_EVENT_LOG_PATH`);
- `logs/evaluation/`: os registos do estudo com utilizadores, que não se mexem.

Depois de um reinício, confirmar:

```bash
curl -s http://127.0.0.1:8001/health
curl -s -o /dev/null -w "%{http_code}\n" -X OPTIONS https://patrimonio360.webredirect.org:8081/api/v1/chat/messages/stream -H "Origin: https://patrimonio360-amalia-frontend.web.app" -H "Access-Control-Request-Method: POST"
```

O primeiro devolve `{"status":"ok",…}` e o segundo tem de dar `200`. O `/ready` diz quando o pré-aquecimento acabou e devolve 503 até lá.

## Fluxo de trabalho

- `main` é o que está em produção, e o código nesta máquina é sempre igual ao GitHub.
- Cada tarefa tem um ramo próprio e um PR. O CI (`compilar`) tem de passar, e o merge é sempre feito pelo utilizador. Não há commits diretos no `main`.
- Depois do merge, nesta máquina:

  ```bash
  cd ~/Projects/AMALIA/patrimonio360-backend
  git switch main
  git pull --ff-only
  pkill -f "uvicorn app.main:app --host 127.0.0.1 --port 8001"
  deploy/amalia_online.sh
  ```

- O `git push` usa HTTPS com a sessão do `gh`. A chave SSH do GitHub desta máquina tem frase-passe e não serve para scripts.

## Correr localmente

É preciso Python 3.11, uma GPU NVIDIA com CUDA 12.8 para os embedders e Node 20 ou mais recente para o `multiview_worker`, que faz a pesquisa por modelo 3D.

```bash
conda create -n visitas_virtuais python=3.11
conda activate visitas_virtuais
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128 torchvision==0.23.0+cu128
pip install -r requirements.txt
(cd multiview_worker && npm install)
cp .env.example .env    # e preencher as credenciais
uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

- O `torch` e o `torchvision` instalam-se antes do `requirements.txt`.
- Sem o vLLM local, aponta-se `LLM_BASE_URL` para a API remota.
- As imagens das peças vêm de `Images/` (`IMAGE_ASSET_ROOT`), que não está no git.

Testes e benchmark offline:

```bash
python -m unittest discover -s tests
python -m benchmarks.run --cases benchmarks/cases/benchmark_cases.json --variant full
```

O benchmark escreve em `benchmark_runs/`, que fica fora do git.

## Variáveis de ambiente

O `.env.example` tem todas as chaves do `.env` de produção, com os mesmos valores e comentários, e as credenciais vazias. Estas são as credenciais a preencher:

| variável | para quê |
|---|---|
| `OPENSEARCH_USERNAME`, `OPENSEARCH_PASSWORD` | acesso ao OpenSearch |
| `LLM_API_KEY` | chave do LLM; serve também para a API de reserva se `LLM_FALLBACK_API_KEY` ficar vazio |
| `OPENROUTER_API_KEY` | só com `USE_OPENROUTER_BGE_M3=true` |

Grupos de variáveis:

- **Aplicação:** `APP_ENV`, `API_PREFIX` (`/api/v1`) e `CORS_*`. Hoje `CORS_ALLOW_ORIGINS=*`, o que deixa passar as pré-visualizações do Firebase (`patrimonio360-amalia-frontend--pr-N-….web.app`). Se o CORS for apertado, esses domínios têm de continuar a ser aceites.
- **OpenSearch:** `OPENSEARCH_HOST`, `OPENSEARCH_PORT`, as credenciais e os `OPENSEARCH_INDEX_*`.
- **Modelos:** `QWEN_*`, `EMBEDDING_*`, `MULTIMODAL_*`, `VL_*` e `RERANKER_*`.
- **LLM:** `LLM_BASE_URL` (o vLLM local), `LLM_FALLBACK_BASE_URL` (a API de reserva) e os restantes `LLM_*`.
- **Assistente:** `CHAT_*`. Cada comportamento tem um comentário no `.env.example`.
- **Ficheiros:** `IMAGE_ASSET_ROOT`, `POI_TOURS_DIR` (vazio usa `poi_tours/`), `MULTIVIEW_LAST_VIEWS_DIR`, `QUERY_LOG_PATH` e `FRONTEND_EVENT_LOG_PATH`.
- **Registos:** `LOG_*` e `BACKEND_LOG_*`.

## Rotas

Com `API_PREFIX=/api/v1`:

| método | caminho | o quê |
|---|---|---|
| GET | `/health`, `/ready` | estado do processo e fim do pré-aquecimento (só nesta máquina) |
| GET | `/api/v1/health` | estado da API (só nesta máquina) |
| GET | `/api/v1/chat/health` | estado do chat; pelo VPS responde o próprio nginx, com CORS |
| POST | `/api/v1/chat/messages` | pergunta de texto, com a resposta completa |
| POST | `/api/v1/chat/messages/stream` | o mesmo em SSE, com os estados e a resposta final; é o que o frontend usa |
| POST | `/api/v1/chat/messages/results` | mais resultados da mesma pesquisa |
| POST | `/api/v1/chat/messages/regenerate` e `/regenerate/stream` | voltar a gerar a última resposta |
| POST | `/api/v1/chat/messages/image` e `/image/stream` | pesquisa por fotografia (multipart) |
| POST | `/api/v1/chat/messages/model` e `/model/stream` | pesquisa por modelo 3D, `.glb`, `.gltf` ou `.obj` (multipart) |
| GET | `/api/v1/chat/artifacts/{id}/full` | ficha completa de uma peça |
| GET | `/api/v1/chat/artifacts/{id}/detail-context` | contexto da peça aberta, para as perguntas sobre ela |
| GET | `/api/v1/chat/artifacts/by-inventory/full` | ficha pelo número de inventário |
| GET | `/api/v1/chat/artifacts/related` | peças relacionadas, por páginas |
| GET | `/api/v1/chat/images/{image_ref}` | imagem de uma peça, a partir de `IMAGE_ASSET_ROOT` |
| POST | `/api/v1/logs/events` | eventos do frontend, gravados em `FRONTEND_EVENT_LOG_PATH` |

## O que não está no git

Estes ficheiros ficam só na pasta onde o backend corre (ver o `.gitignore`):

| caminho | tamanho | porquê |
|---|---|---|
| `.env` | — | credenciais |
| `Images/` | 12 GB | imagens das peças, servidas por `/api/v1/chat/images/` |
| `logs/` | 79 MB | registos do estudo e de produção |
| `benchmarks/assets/` | 60 MB | modelos 3D e fotografias das perguntas do benchmark |
| `multiview_worker/node_modules/` | 52 MB | dependências do Node (`npm install`) |
| `tmp/` | 37 MB | últimas vistas geradas pelo multiview |
| `benchmark_runs/` | 1 MB | resultados do benchmark, descartáveis |

Para montar um clone novo a partir de uma cópia que já funciona (`ORIGEM`), sem gastar disco com as imagens:

```bash
cp -p "$ORIGEM/.env" .
cp -al "$ORIGEM/Images" Images                 # ligações rígidas: não duplica os 12 GB
cp -al "$ORIGEM/benchmarks/assets" benchmarks/assets
cp -a "$ORIGEM/logs" "$ORIGEM/tmp" "$ORIGEM/benchmark_runs" .
cp -a "$ORIGEM/multiview_worker/node_modules" multiview_worker/
```

## Estrutura

- `app/`: a aplicação. Tem as rotas em `api/routes/`, a lógica em `services/`, as definições em `core/config.py`, o vocabulário leigo e os museus pesquisáveis em `config/*.json`, os lugares das visitas em `data/tour_locations.json` e os prompts em `prompts/`.
- `poi_tours/`: os hotspots de cada visita 3DVista (`panorama-overlays-inventory-<museu>.json`), que dão o «ver na visita» e o «onde estou?». Existem para cmag, mj, mnaz, mnsr e mnt.
- `scripts/build_tour_locations.py`: gera `app/data/tour_locations.json` a partir das visitas publicadas. Volta a correr-se sempre que uma visita é republicada.
- `multiview_worker/`: serviço Node com puppeteer que desenha vistas de um modelo 3D para a pesquisa por modelo.
- `benchmarks/`: o benchmark offline.
- `tests/`: os testes unitários.
- `tools/ingestao/`: ingestão de um museu de ponta a ponta, do RAIZ ao índice e à visita. Usa os projetos irmãos `../raiz_scraper` e `../patrimonio360_indexer`, e está explicada em `tools/ingestao/README.md`.
- `tools/poi_mapping/`: números de inventário dos POIs e campo `in_tour` no índice.
- `deploy/amalia_online.sh` e `deploy/llm/amalia_llm.sh`: o arranque em produção, descrito acima.
- `deploy/tours/`: instala o modo «focus» e as pontes nas visitas do VPS (`/root/tours`, porta 5500). A cópia oficial de `tour-focus-artifact.js` está em `tour-scripts/`, no repositório do frontend. Hoje as duas são iguais; se a do frontend mudar, copia-se para aqui antes de instalar.
- `docs/`: notas de metodologia sobre a paginação no OpenSearch e o benchmark de modelos da fase 0B.
- `Dockerfile`: a variante para Cloud Run (CUDA 12.8, `uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --proxy-headers`). Não é o que corre nesta máquina.
