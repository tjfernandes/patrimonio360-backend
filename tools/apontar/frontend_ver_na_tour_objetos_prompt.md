# «Ver na tour» para peças sem hotspot — o que muda no frontend

Contexto (2026-10-31). O backend passou a dar alvo de navegação também às peças que
não têm hotspot na visita mas que a indexação offline localizou num panorama
(`poi_tours/panorama-objects-<museu>.json`, só localizações definitivas). Sem
nenhuma alteração no frontend o botão **«Ver na tour» já aparece** nesses cartões e
a visita **vai ao panorama certo**, mas a câmara não se vira para a peça, porque o
widget hoje só reenvia `overlayId`/`panoramaKey`/`inventoryId`. O que se pede é
passar também a direção.

## O que o backend envia

Cada entrada de `navigation_targets` (resposta do chat, páginas de resultados,
`navigation_target` dos relacionados no modal) pode trazer campos novos:

```json
{
  "overlay_id": "object:panorama_0A348E91_1B9C_4603_41B8_6523A676892B:0",
  "panorama_key": "panorama_0A348E91_1B9C_4603_41B8_6523A676892B",
  "inventory_id": "1183 Mov",
  "location": "Sala 1",
  "title": "Mesa de uma aba",
  "source": "object",
  "yaw": 110.88,
  "pitch": -16.32,
  "score": 0.825
}
```

- `source`: `"hotspot"` (como até aqui; `yaw`/`pitch`/`score` vêm a `null`) ou
  `"object"` (peça sem hotspot).
- Num alvo `"object"` o `overlay_id` é **sintético** (`object:<object_id>`), não
  existe na visita. Está lá porque o widget e a visita exigem um overlayId não vazio.
- `yaw`/`pitch` em graus, na convenção dos hotspots do 3DVista e da câmara que já
  enviam em `metadata.point.camera`: yaw cresce para a direita, pitch para cima.
- `score`: cosseno da identificação (0–1), só para diagnóstico.

## Alterações pedidas

1. `src/embed/services/chatApi.ts` — `RawNavigationTarget` e
   `normalizeNavigationTargetEntry`: ler `source`, `yaw`, `pitch`, `score` e guardá-los
   no `ChatNavigationTarget` (`src/embed/types.ts`): `source?: 'hotspot' | 'object'`,
   `yaw?: number`, `pitch?: number`, `score?: number`. Só aceitar `yaw`/`pitch` quando
   forem números finitos.
2. `src/embed/services/tourBridge.ts` — `navigateToArtifactInTour`: quando o payload
   tiver `yaw` e `pitch` numéricos, incluí-los na mensagem `navigateToArtifact`
   (`{ ..., yaw, pitch }`). O script da visita (`tour-focus-artifact.js` ≥ 2026-10-31.1)
   usa-os quando não encontra o overlay: vai ao panorama e vira a câmara para lá, sem
   pulsação de ícone (não há ícone).
3. `src/embed/components/TourAssistantEmbed.tsx` — `handleNavigateToTarget` (e o
   deep-link entre museus, `buildCrossMuseumEmbedUrl`/parsing dos parâmetros `yaw`,
   `pitch`): passar `yaw`/`pitch` do alvo para o payload. O `mode` continua `'focus'`
   (o alvo tem overlayId).
4. Opcional, para distinguir: no botão «Ver na tour» de um alvo `source === 'object'`
   mostrar um sufixo discreto (p. ex. «≈ localização»), porque a visita não destaca
   nenhum ícone ao chegar. Não mostrar o `score`.
5. Nos eventos de interação (`interactionLogger.ts`) acrescentar `source` do alvo ao
   `navigation_command_sent`, para medirmos quantos cliques vêm dos alvos novos.

## Como testar

- MNAA, perguntar por uma peça sem hotspot localizada (p. ex. «mesa de uma aba» →
  1183 Mov): o cartão tem «Ver na tour»; ao clicar, a visita muda para o panorama e,
  com o script novo instalado no VPS, a câmara fica virada para a mesa.
- Uma peça com hotspot (p. ex. Fuga para o Egipto, 1874 Pint) continua igual: panorama
  + câmara no hotspot + ícone a pulsar.
- Peça que não está em nenhum mapa: sem botão, como antes.

Qualquer dúvida sobre os dados: `GET /api/v1/chat/tour/objects?museum_slug=mnaa&panorama_key=…`
lista os objetos localizados num panorama, com a mesma convenção de yaw/pitch.
