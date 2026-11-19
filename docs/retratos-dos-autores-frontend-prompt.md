# Prompt para o frontend: retrato e créditos na ficha do autor

Cola o texto abaixo na sessão do repositório `patrimonio360-frontend`.

---

A ficha do autor do chat vai passar a receber retratos. O backend envia em `author.image_url` uma imagem do Wikimedia Commons e, com ela, três campos novos com os créditos da imagem. A ficha já desenha o retrato quando `image_url` vem preenchido; falta **mostrar os créditos** e confirmar que o retrato fica bem com as imagens reais.

Não há nada a pagar nem contas a criar: são imagens em domínio público ou com licenças livres (Creative Commons), carregadas diretamente do Wikimedia. Cerca de um quarto delas pede em troca que se indique o autor da imagem e a licença, com ligação à origem. É isso que a linha de créditos resolve.

## O que o backend envia

Em `author`, quando o autor tem retrato (os outros campos da ficha não mudam):

```json
"author": {
  "name": "Keil, Maria (Silves, 1914 - Lisboa 2012)",
  "image_url": "https://thumb.wikimedia.org/wikipedia/commons/thumb/9/90/Maria_Keil_1955.jpg/330px-Maria_Keil_1955.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=thumbnail",
  "image_credit": "José Rafael Botelho",
  "image_license": "CC BY-SA 3.0",
  "image_source_url": "https://commons.wikimedia.org/wiki/File:Maria_Keil_1955.jpg"
}
```

| Campo | O que é |
|---|---|
| `image_url` | O retrato. Endereço absoluto https em `thumb.wikimedia.org` ou `upload.wikimedia.org`. A imagem tem 330 px de largura, ou menos quando o original é mais pequeno |
| `image_credit` | Autor da imagem, como está no Wikimedia Commons. Texto simples, até 120 caracteres. Pode vir `null`, e nem sempre é um nome limpo («Unknown author», «Upload by Carlos Botelho»): mostra-o como vem |
| `image_license` | Licença: «Public domain», «CC BY-SA 4.0», «CC0»… |
| `image_source_url` | Página da imagem no Wikimedia Commons |

Sem retrato os quatro campos vêm `null` (ou não vêm, com um backend mais antigo: trata `undefined` como `null`) e a ficha mostra as iniciais, como hoje.

## O que fazer

1. **Linha de créditos.** Sempre que a ficha mostra o retrato, mostra também os créditos, em texto pequeno e discreto: «Imagem: José Rafael Botelho · CC BY-SA 3.0». A linha inteira é uma ligação para `image_source_url`, a abrir noutro separador (`rel="noopener noreferrer"`).
   - `image_credit` a `null`: «Imagem: CC BY-SA 3.0».
   - `image_source_url` a `null`: o mesmo texto, sem ligação.
   - `image_credit` e `image_license` os dois a `null`: sem linha.
   - A regra é a mesma para todas as imagens. Não decidas pela licença se mostras ou não.
   - O crédito é texto, nunca HTML. Se não couber numa linha, corta com reticências e põe o texto completo em `title`.
   - Se a imagem falhar e a ficha mostrar as iniciais, não mostres os créditos.
2. **Onde.** Junto do retrato ou no fundo da ficha, onde ficar melhor no desenho que já existe. Se em ecrã estreito não houver espaço, os créditos podem ficar atrás de um «i» pequeno no canto do retrato, que os abre com um toque. Têm de se alcançar com um toque: só ao passar o rato não chega.
3. **O retrato com imagens reais.** As proporções variam (quase todas mais altas do que largas, algumas ao baixo) e há imagens pequenas, até 119 px de largura. Usa `object-fit: cover`, com o enquadramento puxado para cima (os rostos estão na parte de cima), e não desenhes o retrato maior do que já é na ficha. `alt`: «Retrato de {name}».
4. **Carregamento.** A imagem vem diretamente do Wikimedia num `<img>`, com `loading="lazy"`. Não a copies para o repositório nem a passes por um servidor nosso. Se o site tiver uma Content-Security-Policy com `img-src`, acrescenta `https://upload.wikimedia.org` e `https://thumb.wikimedia.org`.
5. **Textos** nas línguas que a interface já tem: «Imagem:», «Retrato de».
6. **Página de debug da ficha.** Acrescenta os exemplos de baixo, para se verem os créditos em todos os estados.
7. Não faças pedidos novos ao backend: vem tudo na resposta.

## Dados de exemplo

Todos reais, do ficheiro de retratos do backend.

1. **Licença que obriga a créditos:** o JSON de cima (Maria Keil).
2. **Domínio público:** `"name": "Sequeira, Domingos António de"`, `"image_url": "https://thumb.wikimedia.org/wikipedia/commons/thumb/c/c8/Domingos_Ant%C3%B3nio_Sequeira.jpg/330px-Domingos_Ant%C3%B3nio_Sequeira.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=thumbnail"`, `"image_credit": "Domingos Sequeira"`, `"image_license": "Public domain"`, `"image_source_url": "https://commons.wikimedia.org/wiki/File:Domingos_Ant%C3%B3nio_Sequeira.jpg"`.
3. **Sem autor da imagem:** `"name": "Ludovice, João-Frederico (atrib.)"`, `"image_url": "https://thumb.wikimedia.org/wikipedia/commons/thumb/5/59/Jo%C3%A3o_Frederico_Ludovice.JPG/330px-Jo%C3%A3o_Frederico_Ludovice.JPG?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=thumbnail"`, `"image_credit": null`, `"image_license": "CC BY-SA 3.0"`, `"image_source_url": "https://commons.wikimedia.org/wiki/File:Jo%C3%A3o_Frederico_Ludovice.JPG"`.
4. **Imagem pequena (119 × 245 px):** `"name": "Nuno Gonçalves"`, `"image_url": "https://upload.wikimedia.org/wikipedia/commons/4/4f/Possible_self-portrait_of_Nuno_Goncalves_%28from_St_Vincent_Panels%29.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=thumbnail_unscaled"`, `"image_credit": "Nuno Gonçalves"`, `"image_license": "Public domain"`, `"image_source_url": "https://commons.wikimedia.org/wiki/File:Possible_self-portrait_of_Nuno_Goncalves_(from_St_Vincent_Panels).jpg"`.
5. **Sem retrato:** `"name": "Gregório Lopes"`, os quatro campos a `null`.
6. **Imagem que falha:** o exemplo 1 com um `image_url` que dá erro. Iniciais, sem créditos.

## Para testar com o backend

O backend só envia retratos com a flag `CHAT_AUTHOR_PORTRAITS` ligada. No Museu Nacional de Arte Antiga: «Tens obras do Domingos Sequeira?» (retrato em domínio público), «Mostra-me obras de Nuno Gonçalves» (imagem pequena), «Mostra-me obras de Gregório Lopes» (sem retrato). No Museu Nacional do Azulejo: «Fala-me da Maria Keil» (licença com créditos).
