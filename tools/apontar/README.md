# «Apontar e perguntar»

O visitante toca num ponto do panorama e o backend diz que peça é. O frontend manda `metadata.intent = "point_and_ask"` com `metadata.point` (panorama, câmara, viewport, ponto 0..1, hotspot próximo); o backend responde pelo SSE de sempre mais os eventos `point` (`captured`, `segmented`, `candidates`, `decided`). Liga-se com `POINT_AND_ASK_ENABLED=true`.

## O que acontece a cada toque

1. **Vista.** `app/services/point_view.py` projeta a vista retilínea da câmara a partir das faces do cubo da visita publicada (`media/<panorama>_0/{f,b,l,r,u,d}/<nível>/<row>_<col>.webp`, tiles de 512 px). Os tiles ficam em cache em `tmp/visitas/<museu>/media/`. ~120 ms com a cache quente.
2. **Recorte.** `app/services/point_segmenter.py`: SAM 3 por conceito («artwork», «furniture», «tapestry», «garment») dá a peça inteira que contém o ponto; sem instância, o SAM por ponto. Depois projeta-se uma vista apertada à volta da máscara com as faces no nível mais fino (3072 px): é essa que vai ao embedding e, escurecida fora da máscara, ao visitante. ~0,35 s.
3. **Pesquisa.** O recorte passa pelo Qwen3-VL-Embedding e vai ao kNN do índice de imagens, só neste museu. Score = cosseno, sem boosts (os ×1,15/×1,3 da visita e do panorama foram desligados a 2026-10-19: viravam resultados certos). Se o toque caiu num hotspot, a peça dele é o candidato 1 com 0,99, sem pesquisa.
4. **Decisão.** Acima de `POINT_MATCH_THRESHOLD` (0,80) a peça segue o fluxo `describe_artifact`; abaixo, «Não encontrei esta peça no acervo» e o frontend oferece «Descreve-me o que vês» (`describe_point_crop`), que pede ao modelo de visão (vLLM na :8005, `servir_vlm.sh` do caption_indexer) uma descrição do recorte.

Os recortes ficam em memória 10 minutos: `GET /api/v1/chat/point/<request_id>/{crop,cutout,view}.jpg`.

## Ferramentas

- `tools/apontar/descarregar_faces.py [museu…] [--completo]` aquece a cache das faces. Sem ela o primeiro toque num panorama demora 1 a 3 s a mais (tiles do VPS).
- `scripts/testar_apontar.py` simula o toque do frontend e mede cada fase; com `--expect-inventory` ou `--expect-none` verifica o resultado.

## Medido (2026-10-16, RTX 6000, cache quente)

Toque no centro da «Fuga para o Egipto» (Sala 52 do MNAA): projeção 184 ms, SAM 263 ms, vista apertada 102 ms, embedding 91 ms, kNN 121 ms; `decided` aos 0,79 s; primeiro token aos 0,94 s. Em 24 toques no centro das peças com hotspot do MNAA (câmara no hotspot, +5° de pitch), a peça certa foi o 1.º candidato em 15; a maioria das falhas são toques que não caem na peça (o ícone do hotspot fica abaixo dela). Scores das peças certas: mediana 1,0, mínimo 0,53; dos candidatos errados: mediana 0,61.
