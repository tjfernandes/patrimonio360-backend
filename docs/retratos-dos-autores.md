# Retratos dos autores

A ficha do autor (`docs/ficha-do-autor.md`) pode trazer o retrato do autor. O catálogo de origem não tem retratos:
vêm da Wikidata e do Wikimedia Commons, escolhidos fora do backend e guardados em `app/data/author_portraits.json`.
O backend só lê esse ficheiro.

Atrás da flag `CHAT_AUTHOR_PORTRAITS`, desligada por omissão. Com a flag desligada a resposta é igual à de antes.

## O que o frontend recebe

Em `author`, quando o autor tem retrato:

| Campo | O que é |
|---|---|
| `image_url` | Endereço absoluto https da imagem (330 px de largura, ou o original quando é mais estreito), em `thumb.wikimedia.org` ou `upload.wikimedia.org` |
| `image_credit` | Autor da imagem, como está no Commons. Pode vir `null` |
| `image_license` | Licença, por exemplo «Public domain» ou «CC BY-SA 4.0» |
| `image_source_url` | Página da imagem no Commons |

Sem retrato, os quatro campos vêm `null` e a ficha mostra as iniciais, como hoje.

**Os créditos têm de aparecer.** 79 das 340 imagens têm licenças que obrigam a indicar o autor e a licença, com
ligação à origem (Creative Commons BY e BY-SA, e uma Free Art License). Basta uma linha pequena junto do retrato,
por exemplo «Foto: José Rafael Botelho, CC BY-SA 3.0», com ligação a `image_source_url`. As outras 261 estão em
domínio público, CC0 ou sem restrições.

Nenhuma destas licenças tem custo nem pede registo: são licenças livres, e a única obrigação é a menção do autor e
da licença. O texto para a sessão do frontend está em `docs/retratos-dos-autores-frontend-prompt.md`.

O crédito vem como está no Commons e nem sempre é um nome limpo («Unknown author», «portuguese»). Não foi corrigido
à mão.

## Como foram escolhidos

O pedido era encontrar o retrato e confirmar que a pessoa encontrada é de facto o autor do catálogo. Os scripts estão
em `scripts/retratos_dos_autores/`.

1. **Autores** (`exportar_autores.py`): os 1228 autores dos seis museus, com o que o catálogo diz de cada um (datas,
   atividade, biografia) e o contexto das peças (categorias, séculos, títulos, museus). Só 190 têm algum ano de
   nascimento ou morte.
2. **Nome e tipo** (`passo1_nomes.py`): o Qwen escreve o nome pela ordem normal e diz o que o autor é. 1126 são uma
   pessoa ou uma atribuição a uma pessoa; os outros são oficinas, círculos, empresas, anónimos ou vários autores, e
   não recebem retrato.
3. **Candidatos** (`passo2_wikidata.py`): procura do nome na Wikidata, ficando só com pessoas. 811 autores têm
   alguma pessoa candidata e 533 têm alguma com imagem.
4. **É a mesma pessoa?** (`passo3_mesmo.py`): o Qwen vê os dois lados (catálogo e Wikidata) e responde sim, não,
   não dá para saber, ou «é uma oficina, não a pessoa». Decide todas as pessoas candidatas de cada autor, para se
   saber quando aceita mais de uma. 4396 pares decididos, 702 aceites.
5. **A imagem é um retrato?** (`passo4_imagens.py`): o Qwen vê a imagem e diz se é o retrato de uma só pessoa, um
   grupo, ou uma obra, objeto ou edifício.
6. **Revisão à mão** (`rejeitados.json`, `escolhidos.json`): todos os pares aceites foram vistos um a um, com a
   imagem ao lado dos dados do catálogo e da Wikidata.
7. **Ficheiro** (`passo5_ficheiro.py`): junta tudo e escreve `app/data/author_portraits.json`.

### Quanto acerta a pergunta «é a mesma pessoa?»

Medido em 539 pares de que se sabe a resposta pelas datas (249 com nascimento e morte iguais, 290 com mais de dez
anos de diferença nas duas):

| | Aceita os iguais | Aceita os diferentes |
|---|---|---|
| Com as datas do catálogo à vista | 244 de 249 | 0 de 290 |
| Com as datas do catálogo escondidas | 227 de 249 | 18 de 290 |

Com datas no catálogo a pergunta é segura. Sem datas engana-se em homónimos e familiares (Jan Brueghel pai e filho,
quatro «Pierre Petit», Costa Motta tio e sobrinho). Como a maior parte dos autores não tem datas, ficaram duas
defesas: um autor em que o Qwen aceita mais de uma pessoa fica sem retrato até alguém escolher, e a revisão à mão.

### O que a revisão à mão encontrou

O Qwen aceitou 401 pares com imagem, de 390 autores.

- **20 autores rejeitados**, com o motivo em `rejeitados.json`:
  - pessoa errada (3): «A. J. Pereira», gravador, ligado a um arcebispo de Évora; «Le Brun» ligado a
    Jean-Baptiste-Pierre Lebrun quando as gravuras são segundo Charles Le Brun; «Silva Porto, António (pai)» ligado
    ao filho pintor;
  - oficinas, círculos e peças «a partir de» que o passo 2 tinha dado como pessoa (8): «Oficina de Joaquim Machado
    de Castro», «Oficina de Vasco Fernandes», «Círculo de Giulio Romano», «Bibiena, círculo dos Galli» e quatro
    «a partir de»;
  - nome que não chega (2): «Della Robbia» (uma família de escultores) e «Domingos Vandelli ?»;
  - pessoa certa mas imagem que não serve (7): a imagem na Wikidata é uma obra do autor e não o seu retrato
    (Josefa de Óbidos, Edoardo Gioja) ou mostra mais de uma pessoa (Rodolfo Pinto do Couto, Lucien Lelong,
    Louis Gauffier, Thomas Germain, Pere Serra).
- **30 autores resolvidos**, com o motivo em `escolhidos.json`: o Qwen tinha aceitado também entradas vazias da
  Wikidata com o mesmo nome (quase sempre investigadores sem datas nem profissão) ou um familiar. Exemplos:
  Ana Salazar, Fátima Lopes, François Boucher, Grão Vasco, «David Teniers, o Moço» (o II e não o III).
- Ficaram por resolver, sem retrato, os autores com mais de uma pessoa aceite em que não deu para escolher com
  segurança ou em que a imagem não serve: Luís de Morales
  (a imagem é uma pintura), Charles Heath, Johannes Möller, José Lucas, entre outros.
- A imagem de Sarah Afonso tem 46 px de largura e ficou de fora.
- Nuno Gonçalves tinha ficado de fora porque a imagem que a Wikidata lhe dá é um «possível autorretrato», recortado
  dos Painéis de São Vicente, com 119 px de largura. Entrou a 2026-11-19 por decisão do responsável do projeto. A
  ficha mostra-a como retrato, sem essa ressalva.

A pergunta sobre a imagem deixa passar obras do próprio autor que retratam outra pessoa e alguns retratos de grupo:
foi a revisão à mão que os apanhou.

## Números

340 fichas de autor do catálogo com retrato, de 324 autores (o catálogo tem o mesmo autor escrito de várias
maneiras) e 313 pessoas diferentes.

| Museu | Autores com retrato | Peças com autor cobertas |
|---|---|---|
| Museu Nacional de Arte Antiga | 146 de 441 | 342 de 2116 |
| Museu Nacional Soares dos Reis | 79 de 251 | 498 de 898 |
| Museu Nacional do Azulejo | 43 de 245 | 1253 de 3238 |
| Museu Nacional do Traje | 40 de 245 | 451 de 1625 |
| Casa-Museu Dr. Anastácio Gonçalves | 38 de 83 | 388 de 496 |
| Mosteiro dos Jerónimos | 1 de 6 | 55 de 64 |

Prova de cada retrato, no campo `evidence` do ficheiro: 155 fichas com as datas de nascimento e morte iguais nos
dois lados, 20 com datas parciais e 165 sem datas no catálogo (nome, época das peças, profissão e a revisão à mão).

Autores com muitas peças que ficaram sem retrato, por não se ter encontrado a pessoa na Wikidata, por ela não ter
imagem ou por a imagem não servir: António de Holanda (atribuído), João Machado Costa, Natércia Costa, José António
Tenente, Maria Gambina, Hein Semke, Manuel Cargaleiro (a imagem é uma fotografia de grupo), Querubim Lapa,
Gregório Lopes.

## Limites

- As imagens são servidas pelo Wikimedia, a partir do browser de quem visita. Se um ficheiro for apagado ou mudar de
  nome no Commons, o retrato deixa de aparecer. Para não depender disso seria preciso guardar cópias no nosso
  servidor, o que não foi feito.
- Os retratos de autores antigos são os que a tradição lhes atribui (autorretratos presumidos, gravuras póstumas).
  A revisão confirmou que a pessoa é a mesma e que a imagem é a que a Wikidata dá a essa pessoa; não confirmou a
  autenticidade de cada retrato.
- O ficheiro é uma fotografia do catálogo de 2026-11-18. Autores novos no catálogo ficam sem retrato até se correr
  outra vez o processo.

## Refazer

Na pasta do backend, com o Qwen local ligado (porta 8005) e acesso de leitura ao OpenSearch:

```bash
export RETRATOS_PASTA=~/.cache/patrimonio360-retratos
PYTHONPATH=. python scripts/retratos_dos_autores/exportar_autores.py $RETRATOS_PASTA/autores.json
cd scripts/retratos_dos_autores
python passo1_nomes.py
python passo2_wikidata.py
python passo3_mesmo.py medir      # opcional: repete a medição
python passo3_mesmo.py decidir
python passo4_imagens.py
python passo4c_pequenas.py        # miniaturas das imagens estreitas, se o passo 4 as deixou por descarregar
python passo4_imagens.py          # outra vez, para o Qwen ver as que chegaram
python passo5_ficheiro.py rever   # tabela e folhas de contacto para a revisão à mão
python passo5_ficheiro.py escrever ../../app/data/author_portraits.json
```

Todos os passos são retomáveis. O ficheiro é relido pelo backend quando muda no disco, sem reinício.

O Wikimedia limita os clientes sem identificação: 10 pedidos por minuto na API da Wikidata e do Commons, e muito
menos no servidor dos ficheiros originais (`upload.wikimedia.org`), que responde 429 durante minutos. Por isso a
procura usa o serviço de consultas SPARQL em lotes de 20 nomes, e a verificação usa as miniaturas de
`thumb.wikimedia.org`.
