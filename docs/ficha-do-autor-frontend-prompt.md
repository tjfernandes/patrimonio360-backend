# Prompt para o frontend: ficha do autor no chat

Cola o texto abaixo na sessão do repositório `patrimonio360-frontend`.

---

Quero que o chat da AMÁLIA mostre uma **ficha do autor** quando o visitante pergunta por um autor em concreto («Mostra-me obras de Gregório Lopes», «Quem foi Soares dos Reis?»). O backend já envia os dados; falta desenhá-los.

Trabalha em duas fases. **Não integres nada no chat antes de eu escolher uma maquete.**

## Fase 1: maquetes numa página de debug

1. Cria uma página de debug que só existe em desenvolvimento e na pré-visualização do PR (segue o que o repositório já usar para páginas de debug; se não houver nada, uma rota `/debug/ficha-autor`). Não pode ficar acessível ao visitante em produção.
2. Nessa página, desenha **três propostas diferentes** do bloco do autor, com componentes reais e os estilos do chat, não imagens. Têm de ser diferentes na estrutura, não só na cor. Pontos de partida, que podes trocar por melhores:
   - **A, cabeçalho compacto:** retrato pequeno à esquerda, nome, uma linha de atividade e datas, contagens; biografia fechada.
   - **B, ficha:** um cartão do tamanho dos cartões das peças, com o retrato em cima e o texto por baixo, a abrir a fila dos cartões.
   - **C, faixa de uma linha:** só nome, datas e contagens, que se abre ao toque para mostrar o resto.
3. Cada proposta aparece com **todos os estados de dados** da secção «Dados de exemplo», uns por baixo dos outros, e sempre com três ou quatro cartões de peças por baixo (o componente de cartão que o chat já usa, com dados fictícios), para se ver o conjunto.
4. A página tem um seletor de largura (telemóvel, ~380 px; painel do chat em computador) e o modo claro/escuro, se a interface o tiver.
5. Mostra também os casos que costumam partir o desenho: nome muito comprido, biografia de vários parágrafos, `image_url` que dá erro ao carregar, tudo a `null` exceto o nome.
6. No fim, diz-me em duas ou três linhas por proposta o que ganha e o que perde, **qual recomendas e porquê**, e dá-me o endereço da página na pré-visualização. Depois espera pela minha escolha.

## Fase 2: integrar a proposta escolhida

1. Integra no chat só a proposta que eu escolher, com os ajustes que eu pedir.
2. A página de debug fica, só com a proposta escolhida e os estados de dados, para testar alterações futuras sem backend.
3. Apaga o código das propostas que não ficaram.

## O que o backend envia

A resposta de `POST /api/v1/chat/messages` e o evento final `result` de `POST /api/v1/chat/messages/stream` (e as duas rotas de `regenerate`) passam a ter um campo novo, `author`. É `null` na maior parte das respostas. Quando vem preenchido, tem esta forma:

```json
"author": {
  "entity_ids": ["raiz:autor:59837"],
  "name": "Reis, António Soares dos",
  "activity": "Escultor",
  "birth_date": "1847",
  "death_date": "1889",
  "birth_year": 1847,
  "death_year": 1889,
  "birth_place": "Mafamude, Vila Nova de Gaia",
  "death_place": "St. Marinha, Vila Nova de Gaia",
  "biography": "António Manuel Soares dos Reis nasceu em 1847 na freguesia de Mafamude, Vila Nova de Gaia. Em 1861, com 13 anos de idade, iniciou os estudos de desenho na Academia Portuense de Belas Artes…",
  "url": "https://raiz.museusemonumentos.pt/DetalhesAutor?authorId=59837",
  "image_url": null,
  "artifact_count": 84,
  "in_tour_count": 3
}
```

| Campo | O que é |
|---|---|
| `name` | O nome como o catálogo o escreve. Vem sempre. Pode vir como «Apelido, Nome» ou com datas entre parênteses («Keil, Maria (Silves, 1914 - Lisboa 2012)»): mostra-o tal como vem. |
| `activity` | «Pintor», «Escultor»… Muitas vezes `null`. |
| `birth_year`, `death_year` | O ano, quando o backend o conseguiu ler. |
| `birth_date`, `death_date` | A data como o catálogo a escreve, em formatos variados («1847», «19/06/1731», «00/00/1884», «Século XV (?)»). Usa-as só quando o ano respetivo é `null`. |
| `birth_place`, `death_place` | Locais. Muitas vezes `null`. |
| `biography` | Texto do catálogo, sem formatação, por vezes com vários parágrafos. Poucos autores têm biografia. |
| `url` | Página do autor no catálogo de origem. |
| `image_url` | Retrato do autor. **Hoje vem sempre `null`** (o catálogo não tem retratos); o campo existe para o frontend ficar pronto. |
| `artifact_count` | Quantas peças deste autor o museu tem. |
| `in_tour_count` | Quantas dessas estão na visita virtual. |
| `entity_ids` | Ids internos do autor. Não se mostram. |

Qualquer campo exceto `name`, `artifact_count`, `in_tour_count` e `entity_ids` pode vir `null`. O caso mais comum é só nome, contagens e `url`.

As peças do autor não vêm dentro de `author`: são os cartões de sempre, em `artifact_results`, com imagem. Quando há `author`, pelo menos um desses cartões é de uma peça dele (normalmente todos).

## Dados de exemplo para as maquetes

Os quatro primeiros são respostas reais do backend; os dois últimos são inventados para testar o desenho.

1. **Completo:** o JSON de cima (Soares dos Reis), com a biografia alongada para três parágrafos.
2. **Mínimo, o mais comum:** `{"entity_ids": ["raiz:autor:46230", "raiz:autor:46373"], "name": "Gregório Lopes", "url": "https://raiz.museusemonumentos.pt/DetalhesAutor?authorId=46230", "artifact_count": 19, "in_tour_count": 1}`, o resto a `null`.
3. **Datas e locais, sem atividade:** `"name": "Machado de Castro, Joaquim"`, `"birth_date": "19/06/1731"`, `"death_date": "17/11/1822"`, `"birth_year": 1731`, `"death_year": 1822`, `"birth_place": "Coimbra"`, `"death_place": "Lisboa"`, com biografia, `"artifact_count": 7`, `"in_tour_count": 0`.
4. **Data mal escrita no catálogo:** `"name": "Pousão, Henrique César de Araújo"`, `"birth_date": "01/01/1859"`, `"death_date": "00/00/1884"`, `"birth_year": 1859`, `"death_year": 1884`, `"birth_place": "Vila Viçosa"`, `"death_place": "Vila Viçosa"`, com biografia, `"artifact_count": 80`, `"in_tour_count": 1`.
5. **Nome comprido e sem ano (inventado):** `"name": "Vieira Lusitano, Francisco Vieira de Matos, o (1699-1783)"`, `"birth_date": "Século XVII (?)"`, anos a `null`, `"artifact_count": 1`, `"in_tour_count": 0`.
6. **Com retrato (inventado):** o exemplo 1 com `image_url` a apontar para uma imagem de teste do próprio repositório, e uma variante com um endereço que dá erro.

## O que o bloco tem de mostrar, em qualquer proposta

1. O retrato, se `image_url` vier preenchido. Sem retrato, ou se a imagem falhar, um marcador neutro com as iniciais ou um ícone de pessoa: nunca uma imagem partida, nunca uma peça a fazer de retrato.
2. O nome.
3. A atividade e as datas, só com o que existir: «Escultor · 1847–1889», «1731–1822», «Pintor»; só com nascimento, «n. 1847»; só com morte, «m. 1889».
4. Os locais, se existirem: «Coimbra – Lisboa».
5. A contagem: «84 peças no museu» e, se `in_tour_count` > 0, «3 nesta visita» (singular quando for 1).
6. A biografia, se existir, fechada ou cortada a três ou quatro linhas, com «ver mais» para abrir o resto.
7. Uma ligação discreta «Ver no catálogo» para `url`, a abrir noutro separador.

## Regras da integração (fase 2)

1. O bloco aparece quando `author` não é `null`, junto dos cartões dessa resposta e por baixo do texto da AMÁLIA.
2. Os cartões das peças continuam sem alterações, com as imagens e o «Ver na tour» de sempre. Não escondas nem reordenes cartões por causa da ficha.
3. O bloco pertence à mensagem em que veio. Fica no histórico da conversa como os cartões ficam, e as mensagens seguintes não o apagam.
4. A página seguinte de resultados (`POST /api/v1/chat/messages/results`) não traz `author`: o bloco da primeira página mantém-se.
5. Textos do bloco nas línguas que a interface já tem: «peças no museu», «nesta visita», «ver mais», «Ver no catálogo».
6. Em ecrã estreito o bloco não pode empurrar os cartões para fora do ecrã.
7. Sem `author`, ou com o campo ausente (backend sem a funcionalidade ligada), tudo fica como está. Trata `undefined` como `null`.
8. Não faças pedidos novos ao backend para a ficha: vem tudo na resposta.

## Para testar com o backend

O backend só envia `author` com a flag `CHAT_AUTHOR_CARD` ligada. Perguntas que devem trazer ficha, no Museu Nacional de Arte Antiga: «Mostra-me obras de Gregório Lopes» (só nome e contagens), «Há esculturas de Machado de Castro?» (datas, locais e biografia), «Tens obras do Domingos Sequeira?». No Museu Nacional Soares dos Reis: «Quem foi Henrique Pousão?» (com biografia). Perguntas que não devem trazer: «Quem pintou o São Jerónimo?», «Peças vindas do Japão», «Que autores há neste museu?».
