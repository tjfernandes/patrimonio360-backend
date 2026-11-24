# Ver os passos do RAG nos logs

`deploy/amalia_logs.sh` lê o log do backend (`~/.amalia_backend.log`) e mostra
cada pedido de conversa por passos: leitura do pedido, encaminhamento,
pesquisa, cartões, resposta. Só lê o ficheiro: não toca no backend e funciona
com qualquer `python3` (só biblioteca padrão).

```bash
deploy/amalia_logs.sh                  # os últimos 5 pedidos
deploy/amalia_logs.sh -f               # em direto: cada passo aparece quando acontece
deploy/amalia_logs.sh -f -l            # em direto, uma linha por pedido quando acaba
deploy/amalia_logs.sh -l               # índice: hora, rid, duração, estado, rota, pergunta
deploy/amalia_logs.sh --rid 2d0636     # um pedido (basta o início do rid do índice)
deploy/amalia_logs.sh -c -n 1          # o último, com prompts e respostas por extenso
deploy/amalia_logs.sh -p bosch -n 0    # todos os que falam de «bosch» (pergunta ou resposta)
deploy/amalia_logs.sh -e               # só os que tiveram avisos, erros ou ficaram sem fim
deploy/amalia_logs.sh --conversa 65f9  # os pedidos de uma conversa
deploy/amalia_logs.sh --desde 14:30    # a partir de uma hora
deploy/amalia_logs.sh --arranque       # o que o backend carregou ao arrancar
deploy/amalia_logs.sh --ficheiro logs/archive/amalia_backend_20261120_1901_antes-rascunho-e-reescrita.log -l
```

Em direto (`-f`), uma chamada ao modelo ou ao OpenSearch sai quando acaba, com a
duração; se levar mais de 0,8 s sai antes um aviso de que está em curso. O
rodapé com os tempos sai 2 s depois do fim do pedido. Com `-l` ou com um filtro
(`-e`, `-p`, `--conversa`), cada pedido sai de uma vez quando acaba. Se houver
dois pedidos ao mesmo tempo, as linhas de cada um vêm marcadas com o `rid`.

No terminal a saída passa pelo `less` quando não cabe no ecrã (`--sem-pager`
desliga); `--cor` força as cores para `| less -R`.

## O que se vê em cada pedido

- **Cabeçalho**: hora, pergunta, museu, língua, rota (`modo/intenção`), estado,
  duração, `rid`, conversa e `query_id`.
- **Passos**, pela ordem em que aconteceram, com o tempo desde o início do pedido:
  - cada chamada ao modelo numa linha — o nome do passo (`leitura do pedido`,
    `reescrita da pesquisa`, `pré-seleção dos cartões`, `rascunho …`,
    `resposta final`, …), o modelo, tokens de entrada→saída, duração e, na
    resposta final, quando chegou a primeira palavra;
  - a decisão do router e a pergunta que seguiu para a pesquisa;
  - o plano da pesquisa (texto para o BM25, texto para o vetor, filtros,
    ordenação) e cada ida ao OpenSearch com duração e número de resultados;
  - os primeiros documentos recuperados, com pontuação;
  - a pré-seleção e os cartões que o visitante vê;
  - o rascunho, o prompt final e a resposta.
- **Tempos**: somas por modelo, OpenSearch e embeddings, e o total do pedido.

Com `-c` aparecem por extenso os prompts (instruções de sistema e prompt), as
respostas do modelo, o corpo enviado ao OpenSearch e os blocos JSON.

## O que tem de estar ligado no `.env`

| Variável | Valor | Para quê |
|---|---|---|
| `BACKEND_LOG_ENABLED` | `true` (omissão) | sem isto não há log |
| `BACKEND_RAG_DEBUG_ENABLED` | `true` (omissão) | decisão do router, plano, resultados, cartões |
| `BACKEND_RAG_DEBUG_LLM_IO` | `true` (omissão: `false`) | texto dos prompts e das respostas do modelo (`rag.llm_call`) e a resposta enviada ao visitante (`rag.answer`) |

Sem `BACKEND_RAG_DEBUG_LLM_IO` veem-se os passos, as decisões, os documentos
e os tempos; dos textos só o tamanho. Com ela o log cresce cerca de 30 a 60 kB
por pedido e passa a guardar as respostas por extenso (as perguntas já lá
estavam). Um texto maior do que metade de `BACKEND_RAG_DEBUG_MAX_CHARS` fica
cortado no log, e o visualizador diz o tamanho original.

## Limites

- O `deploy/amalia_online.sh` recomeça `~/.amalia_backend.log` em cada
  arranque: o que se quer guardar copia-se antes para `logs/archive/` (e lê-se
  com `--ficheiro`).
- Os logs anteriores a esta ferramenta leem-se na mesma, mas sem o nome dos
  passos nas chamadas ao modelo, sem milésimos na hora e sem evento de fecho
  nos percursos e no apontar sem peça escolhida (o estado aparece como `—`).
- Os tempos do rodapé são somas: chamadas em paralelo contam todas, por isso a
  soma pode passar o total do pedido.

## Quando se acrescenta um passo ao backend

- Uma chamada nova ao modelo leva `step="nome_do_passo"` em
  `generate(...)`/`generate_stream(...)`; o nome aparece tal e qual (com
  espaços) e a fase sai de `PHASE_BY_STEP` em `amalia_logs.py`.
- Um evento novo (`log_event`) aparece sem mais nada, como `nome  chave=valor`;
  para o pôr num passo, junta-se o prefixo a `PHASE_BY_EVENT`.
- O formato da linha está em `BACKEND_LOG_FORMAT` (`app/core/logging.py`);
  `tests/test_amalia_logs.py` escreve com o backend e lê com o visualizador.
