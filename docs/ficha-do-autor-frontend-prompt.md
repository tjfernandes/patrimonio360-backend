# Prompt para o frontend: ficha do autor no chat

Cola o texto abaixo na sessão do repositório `patrimonio360-frontend`.

---

Quero que o chat da AMÁLIA mostre uma **ficha do autor** quando o visitante pergunta por um autor em concreto («Mostra-me obras de Gregório Lopes», «Quem foi Soares dos Reis?»). O backend já envia os dados; falta desenhá-los.

## O que o backend envia

A resposta de `POST /api/v1/chat/messages` e o evento final `result` de `POST /api/v1/chat/messages/stream` (e as duas rotas de `regenerate`) passam a ter um campo novo, `author`. É `null` na maior parte das respostas. Quando vem preenchido, tem esta forma:

```json
"author": {
  "entity_ids": ["raiz:autor:68153"],
  "name": "Machado de Castro, Joaquim",
  "activity": null,
  "birth_date": "19/06/1731",
  "death_date": "17/11/1822",
  "birth_year": 1731,
  "death_year": 1822,
  "birth_place": "Coimbra",
  "death_place": "Lisboa",
  "biography": "Texto do catálogo, pode ter vários parágrafos…",
  "url": "https://raiz.museusemonumentos.pt/DetalhesAutor?authorId=68153",
  "image_url": null,
  "artifact_count": 7,
  "in_tour_count": 0
}
```

| Campo | O que é |
|---|---|
| `name` | O nome como o catálogo o escreve. Vem sempre. Pode vir como «Apelido, Nome» ou com datas entre parênteses («Keil, Maria (Silves, 1914 - Lisboa 2012)»): mostra-o tal como vem. |
| `activity` | «Pintor», «Escultor»… Muitas vezes `null`. |
| `birth_year`, `death_year` | O ano, quando o backend o conseguiu ler. |
| `birth_date`, `death_date` | A data como o catálogo a escreve, em formatos variados («1847», «19/06/1731», «1922-03-05», «Século XV (?)»). Usa-as só quando o ano respetivo é `null`. |
| `birth_place`, `death_place` | Locais. Muitas vezes `null`. |
| `biography` | Texto do catálogo, sem formatação. Poucos autores têm biografia. |
| `url` | Página do autor no catálogo de origem. |
| `image_url` | Retrato do autor. **Hoje vem sempre `null`** (o catálogo não tem retratos); o campo existe para o frontend ficar pronto. |
| `artifact_count` | Quantas peças deste autor o museu tem. |
| `in_tour_count` | Quantas dessas estão na visita virtual. |
| `entity_ids` | Ids internos do autor. Não se mostram. |

Qualquer campo exceto `name`, `artifact_count`, `in_tour_count` e `entity_ids` pode vir `null`. O caso mais comum é só nome, contagens e `url`.

As peças do autor não vêm dentro de `author`: são os cartões de sempre, em `artifact_results`, com imagem. Quando há `author`, pelo menos um desses cartões é de uma peça dele (normalmente todos).

## O que quero ver

1. Quando `author` não é `null`, um **bloco do autor por cima dos cartões** dessa resposta, por baixo do texto da AMÁLIA.
2. No bloco:
   - o retrato, se `image_url` vier preenchido; sem retrato, um marcador neutro com as iniciais ou um ícone de pessoa (nunca uma imagem partida, nunca uma peça a fazer de retrato);
   - o nome;
   - uma linha com a atividade e as datas, só com o que existir: «Escultor · 1847–1889», «1731–1822», «Pintor»; só com nascimento, «n. 1847»; só com morte, «m. 1889»;
   - os locais, se existirem: «Coimbra – Lisboa»;
   - a contagem: «7 peças no museu» e, se `in_tour_count` > 0, «3 nesta visita» (singular quando for 1);
   - a biografia, se existir, cortada a três ou quatro linhas com «ver mais» para abrir o resto;
   - uma ligação discreta «Ver no catálogo» para `url`, a abrir noutro separador.
3. Os cartões das peças continuam por baixo, sem alterações, com as imagens e o «Ver na tour» de sempre.
4. O bloco pertence à mensagem em que veio. Fica no histórico da conversa como os cartões ficam, e as mensagens seguintes não o apagam.
5. A página seguinte de resultados (`POST /api/v1/chat/messages/results`) não traz `author`: o bloco da primeira página mantém-se.
6. Textos do bloco nas línguas que a interface já tem: «peças no museu», «nesta visita», «ver mais», «Ver no catálogo».
7. Em ecrã estreito o bloco não pode empurrar os cartões para fora do ecrã: retrato pequeno ao lado do nome, biografia fechada por omissão.

## O que não muda

- Sem `author` (ou com o campo ausente, se o backend ainda não tiver a funcionalidade ligada), tudo fica como está. Trata `undefined` como `null`.
- O texto da resposta e os cartões são os mesmos com e sem ficha. Não escondas nem reordenes cartões por causa dela.
- Não faças pedidos novos ao backend para a ficha: vem tudo na resposta.

## Para testar

O backend só envia `author` com a flag `CHAT_AUTHOR_CARD` ligada. Perguntas que devem trazer ficha, no Museu Nacional de Arte Antiga: «Mostra-me obras de Gregório Lopes» (só nome e contagens), «Há esculturas de Machado de Castro?» (datas e locais), «Tens obras do Domingos Sequeira?». No Museu Nacional Soares dos Reis: «Quem foi Henrique Pousão?» (com biografia). Perguntas que não devem trazer: «Quem pintou o São Jerónimo?», «Peças vindas do Japão», «Que autores há neste museu?».

Para desenhar sem backend, usa o JSON de exemplo acima e uma variante mínima: `{"entity_ids": ["x"], "name": "Gregório Lopes", "artifact_count": 18, "in_tour_count": 1}` com o resto a `null`.
