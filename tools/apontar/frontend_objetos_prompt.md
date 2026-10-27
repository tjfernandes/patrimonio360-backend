# Prompt para o frontend: «Que peças há aqui?» (objetos pré-indexados por panorama)

> Para dar ao agente do repositório `patrimonio360-frontend`. Experiência do backend no ramo `experimento/objetos-panorama` (sem PR para main). O backend de teste serve-a em qualquer instância desse ramo; em produção só depois de o ramo ser fundido.

## Contexto

O «Apontar e perguntar» obriga o visitante a tocar numa peça e, se ela não está no índice, a resposta é «não encontrei». Em alternativa, o backend passou a ter, para cada panorama, a lista das peças que **já identificou offline com confiança**, com a posição angular e o contorno em graus. O frontend assinala-as no panorama. O visitante nunca toca em nada que o sistema não saiba identificar.

Endpoint, já a funcionar:

```
GET /api/v1/chat/tour/objects?museum_slug=<slug>&panorama_key=<panorama_key>
Cache-Control: public, max-age=300
```

Resposta:

```json
{
  "museum_slug": "mnaa",
  "panorama_key": "panorama_83D8666B_91ED_0BA2_41D9_9A5E8973B146",
  "indexed": true,
  "generated_at": "2026-10-26T13:11:34+00:00",
  "segmented": 73,
  "objects": [
    {
      "objectId": "panorama_83D8666B_…:1",
      "artifactId": "raiz:movel:251226",
      "inventoryNumber": "1874 Pint",
      "title": "Fuga para o Egipto",
      "score": 0.889,
      "imageName": "Images/museu_nacional_de_arte_antiga/obj_1874_pint/fileid_309214_1.jpg",
      "yaw": 2.45,
      "pitch": 3.63,
      "bbox": {"yaw_min": -4.3, "yaw_max": 9.05, "pitch_min": -4.96, "pitch_max": 11.95},
      "polygon": [[-3.41, 11.95], [-2.9, 11.9], "… 40 a 400 vértices [yaw, pitch] em graus, pela ordem do contorno"],
      "hotspotOverlayId": "overlay_C29262E1_9F52_29DC_41CB_60D97463C7D0"
    }
  ]
}
```

- `yaw` cresce para a direita e `pitch` para cima, em graus, **na mesma convenção da câmara que já enviam em `metadata.point.camera`** no «Apontar e perguntar». O yaw de um vértice pode sair de [-180, 180] (os contornos são contínuos à volta do centro); a projeção abaixo usa seno e cosseno, por isso não importa.
- `score` é o cosseno 0..1 da correspondência com a ficha (≥ 0,75 por construção).
- `imageName` é o que já usam para as miniaturas: `GET /api/v1/chat/images/<imageName>?w=…`.
- `hotspotOverlayId` vem preenchido quando a peça já tem hotspot neste panorama; `null` quando não tem.
- Panorama sem indexação: HTTP 200 com `indexed: false` e `objects: []`. Museu sem ficheiro: HTTP 404. Trata ambos como «sem peças identificadas aqui», sem erro ao visitante.

## O que implementar

### 1. Pedir a lista quando o panorama muda

- No evento de mudança de panorama do 3DVista (o mesmo onde já lêem a `panorama_key` para o `metadata.point`), fazer o `GET` acima e guardar a resposta numa cache por `panorama_key` (sessão inteira; o backend já manda 5 minutos de cache HTTP).
- Pedido em curso cancelado se o panorama mudar entretanto (`AbortController`).

### 2. Botão «Peças identificadas aqui»

- Um botão na barra da visita ou do chat, ao lado do «Apontar». Com a lista carregada, mostra o número: «3 peças identificadas aqui». Com `indexed: false` ou lista vazia: botão desativado e tooltip «Sem peças identificadas neste ponto».
- Clicar liga e desliga o **modo de destaque**. Mudar de panorama mantém o modo ligado e troca a lista.

### 3. Modo de destaque: contornos no panorama

- Um `<canvas>` a cobrir o viewport do player, por cima dele, com `pointer-events: none` (os cliques resolvem-se no ponto 5).
- Em cada frame (ou em cada evento de câmara do 3DVista, se for suficiente para acompanhar o arrastar), ler `yaw`, `pitch` e `hfov` da câmara e `width`/`height` do viewport, como já fazem para o `metadata.point.camera`, e projetar cada vértice de cada `polygon` com a função abaixo. Desenhar o contorno (linha de 3 px, cor de destaque, preenchimento a 10 % de opacidade) e uma etiqueta junto ao topo da caixa com o `title` (ou «Peça nº <inventoryNumber>» se não houver título). Durante a experiência, mostrar também o `score` em percentagem, pequeno, para avaliarmos a confiança; sai depois.
- Um vértice atrás da câmara (`null` na função) parte o contorno: desenhar só os segmentos entre vértices visíveis consecutivos, ou saltar o objeto se tiver menos de 3 vértices visíveis.
- O contorno é a peça na vista geral: num quadro é a moldura, numa escultura a figura. Pode cobrir só parte de um objeto grande (uma tapeçaria).

Projeção, a mesma do backend (`app/services/point_view.py`, `point_in_view`):

```js
// cam = {yaw, pitch, hfov, width, height}; ângulos em graus, width/height em px do viewport.
// Devolve {x, y} em px do viewport, ou null se a direção está atrás da câmara.
function projectDirection(cam, yawDeg, pitchDeg) {
  const d2r = Math.PI / 180;
  const p = pitchDeg * d2r, y = yawDeg * d2r;
  const dx = Math.cos(p) * Math.sin(y), dy = Math.sin(p), dz = Math.cos(p) * Math.cos(y);
  const cy = Math.cos(cam.yaw * d2r), sy = Math.sin(cam.yaw * d2r);
  const cp = Math.cos(cam.pitch * d2r), sp = Math.sin(cam.pitch * d2r);
  const x1 = cy * dx - sy * dz, y1 = dy, z1 = sy * dx + cy * dz;   // desfaz o yaw da câmara
  const x2 = x1, y2 = cp * y1 - sp * z1, z2 = sp * y1 + cp * z1;    // desfaz o pitch da câmara
  if (z2 <= 1e-6) return null;
  const f = (cam.width / 2) / Math.tan((cam.hfov * d2r) / 2);
  return { x: cam.width / 2 + (x2 / z2) * f, y: cam.height / 2 - (y2 / z2) * f };
}
```

Validação rápida: com a câmara em `yaw 2.41, pitch -2.5, hfov 100` e viewport `999×759`, a Fuga para o Egipto (`yaw 2.45, pitch 3.63`) tem de cair perto do centro, uns 60 px acima.

### 4. Setas para as peças fora do ecrã

- Para cada objeto cujo centro (`yaw`, `pitch`) não cai dentro do viewport (projeção `null` ou fora dos limites), desenhar uma seta na margem do viewport a apontar para ele, com o `title` abreviado.
- Direção da seta: usar `x2` e `y2` do cálculo acima (as coordenadas da direção no referencial da câmara, antes da divisão por `z2`): o ângulo no ecrã é `Math.atan2(-y2, x2)`. Colocar a seta no ponto onde a semirreta do centro do viewport nesse ângulo cruza a margem (com 24 px de folga) e rodá-la para esse ângulo. Funciona para peças atrás da câmara: a seta aponta para o lado por onde é mais curto rodar.
- Peças que estão na vista têm contorno e não têm seta. Quando o visitante roda a câmara e a peça entra, a seta dá lugar ao contorno.

### 5. Clique numa peça

- Como o canvas não recebe cliques, no `click` do viewport (ou no vosso bloqueio de cliques do 3DVista, se o modo de destaque o reutilizar) testar se o ponto está dentro de algum contorno projetado (teste ponto-em-polígono em px). Com o ponto 3 a desenhar a cada frame, basta guardar os polígonos projetados do último frame.
- Clique dentro de um contorno: enviar ao chat a mensagem de «Ver na tour» que já existe, com `metadata.intent = "describe_artifact"` e `metadata.selected_artifact = {artifact_id, inventory_number, title, source: "panorama_objects"}` (os campos do objeto). Não há pesquisa nem modelo de visão: o backend responde com a ficha. Se preferirem, abrir primeiro o diálogo de detalhe da peça (a miniatura vem de `imageName`) e só depois a mensagem.
- Clique numa seta: rodar a câmara para o `yaw`/`pitch` do objeto, com a animação que já usam para ir a um hotspot. Se a API da câmara não estiver acessível, pelo menos realçar a seta.
- Objetos com `hotspotOverlayId`: podem reutilizar o clique do próprio hotspot, se isso der uma experiência mais coerente; caso contrário tratar como os outros.

### 6. O que não fazer

- Não chamar o endpoint a cada frame nem a cada movimento da câmara: uma vez por panorama.
- Não bloquear a navegação do 3DVista no modo de destaque (ao contrário do modo «Apontar»), exceto o clique dentro de um contorno.
- Não mostrar nada para `indexed: false`: nem mensagem de erro, nem contornos vazios.
- Não editar os scripts das visitas publicadas no VPS (`script_general.js`, `script_mobile.js`).

### 7. Como testar

Backend de teste do ramo numa instância local (o `backendBaseUrl` do frontend a apontar para ela). Todos os panoramas dos 6 museus com visita estão indexados (885 panoramas, 2823 objetos; 251 panoramas sem nenhum, onde o botão fica desativado). Casos para verificar:

- Sala 52 do MNAA (`panorama_83D8666B_91ED_0BA2_41D9_9A5E8973B146`): 4 contornos, Fuga para o Egipto (yaw 2,4), Triunfo das Artes (yaw 39), Retrato do Escultor francês Jacques Buirette (yaw 87) e Túmulo Alegórico de Joseph Addison (yaw −154). Virar a câmara de costas para a Fuga e confirmar que a seta aponta para o lado mais curto.
- Sala das Bandeiras do MNT (`panorama_CFEC4CF2_E3BA_2A2E_41EB_E931E9C5BBD6`): 6, entre eles o Vestido de criança 3546 e o Retrato de família 26510.
- Um panorama do MJ sem objetos: botão desativado, sem erro.
- Clique dentro de um contorno abre a ficha da peça certa; clique fora não faz nada de novo.
