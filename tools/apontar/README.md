# «Apontar e perguntar»

O visitante toca num ponto do panorama e o backend diz que peça é. O frontend manda `metadata.intent = "point_and_ask"` com `metadata.point` (panorama, câmara, viewport, ponto 0..1, hotspot próximo); o backend responde pelo SSE de sempre mais os eventos `point` (`captured`, `segmented`, `candidates`, `decided`). Liga-se com `POINT_AND_ASK_ENABLED=true`.

## O que acontece a cada toque

1. **Vista.** `app/services/point_view.py` projeta a vista retilínea da câmara a partir das faces do cubo da visita publicada (`media/<panorama>_0/{f,b,l,r,u,d}/<nível>/<row>_<col>.webp`, tiles de 512 px). Os tiles ficam em cache em `tmp/visitas/<museu>/media/`. ~120 ms com a cache quente.
2. **Recorte.** `app/services/point_segmenter.py`: SAM 3 por conceito («artwork», «furniture», «tapestry», «garment») dá a peça inteira que contém o ponto; sem instância, o SAM por ponto. Depois projeta-se uma vista apertada à volta da máscara com as faces no nível mais fino (3072 px): é essa que vai ao embedding e, escurecida fora da máscara, ao visitante. ~0,35 s.
3. **Pesquisa.** O recorte passa pelo Qwen3-VL-Embedding e vai ao kNN do índice de imagens, só neste museu. Score = cosseno, sem boosts (os ×1,15/×1,3 da visita e do panorama foram desligados a 2026-10-19: viravam resultados certos). Se o toque caiu num hotspot, a peça dele é o candidato 1 com 0,99, sem pesquisa.
4. **Decisão.** Com `POINT_VERIFY_ENABLED=true` e o modelo de visão a responder (vLLM na :8005), é ele que decide entre as 3 melhores candidatas (`app/services/point_verify.py`): vê o recorte e uma grelha com as fotos das fichas, marcadas A, B, C, e diz qual é exatamente a mesma peça. Confirma uma → é essa, com certeza (`confidence: "high"` se o cosseno chega ao limiar, `"verified"` se não); nenhuma → «Não encontro esta peça no acervo. As mais parecidas que tenho são estas.» com os cartões das candidatas (`confidence: "none"`). A escolha só vale se a candidata tiver cosseno ≥ `POINT_VERIFY_MIN_SCORE` (0,6) ou hotspot neste panorama: medido offline a 2026-10-28, as escolhas erradas eram decoração do edifício abaixo de 0,6. O toque num hotspot não passa pelo VLM. A resposta do VLM vai em `point.verify` e no registo. Sem VLM a responder, ou com a flag desligada: acima de `POINT_MATCH_THRESHOLD` (0,70) a peça segue o fluxo `describe_artifact` com certeza; abaixo segue na mesma, mas o texto começa por «Não tenho a certeza, mas parece ser…» (`confidence: "low"`). Com pelo menos um candidato nunca se diz «não encontrei»; sem candidatos (ou abaixo de `POINT_LOW_CONFIDENCE_FLOOR`, 0 por omissão) é «Não encontrei esta peça no acervo».

Os recortes ficam em memória 10 minutos, só para a legenda os mostrar: `GET /api/v1/chat/point/<request_id>/{crop,cutout,view}.jpg`. Um pedido com outra intenção qualquer segue como uma mensagem normal.

## Experiência: objetos pré-indexados por panorama («que peças há aqui?», 2026-10-26)

Em vez de o visitante apontar e arriscar um «não encontrei», o backend sabe de antemão que peças identifica em cada panorama e onde estão. `tools/apontar/indexar_objetos_panorama.py` percorre 16 vistas por panorama (yaw de 45° em 45°, hfov 90°, pitches 0 e −30), recorta todas as instâncias do SAM 3 por conceito (`artwork, painting, sculpture, furniture, tapestry, garment, vase`), projeta a vista apertada de cada uma no nível fino, passa-a pelo Qwen3-VL-Embedding e vai ao kNN do índice de imagens. Fica quem tem cosseno ≥ 0,75 e ≥ 0,05 à frente da segunda peça; a mesma peça vista de duas vistas, ou duas peças no mesmo sítio, ficam reduzidas à de melhor score. Guarda centro (yaw, pitch), caixa e contorno em graus em `poi_tours/panorama-objects-<museu>.json`; `GET /api/v1/chat/tour/objects?museum_slug=&panorama_key=` serve-os (`app/services/panorama_objects.py`), com o `overlayId` do hotspot quando a peça já tem um. O prompt do frontend está em `tools/apontar/frontend_objetos_prompt.md`.

Medido em 5 panoramas (3 do MNAA, 2 do MNT, 20 a 130 s cada, ao lado da produção): 644 recortes, 24 peças identificadas, 23 certas à vista (recorte ao lado da imagem da ficha); a duvidosa é um pormenor de um quadro a 0,761. A Fuga para o Egipto ficou a yaw 2,4°, onde o visitante lhe tinha tocado. Das 23 peças com hotspot nesses panoramas encontrou 7: as que faltam são sobretudo objetos pequenos em vitrinas (botões, bolsas, cartola, bacia e gomil) e peças que nunca foram a melhor correspondência de nenhum recorte.

## Ferramentas

- `tools/apontar/descarregar_faces.py [museu…] [--completo]` aquece a cache das faces. Sem ela o primeiro toque num panorama demora 1 a 3 s a mais (tiles do VPS).
- `scripts/testar_apontar.py` simula o toque do frontend e mede cada fase; com `--expect-inventory` ou `--expect-none` verifica o resultado.

## Medido (2026-10-16, RTX 6000, cache quente)

Toque no centro da «Fuga para o Egipto» (Sala 52 do MNAA): projeção 184 ms, SAM 263 ms, vista apertada 102 ms, embedding 91 ms, kNN 121 ms; `decided` aos 0,79 s; primeiro token aos 0,94 s. Em 24 toques no centro das peças com hotspot do MNAA (câmara no hotspot, +5° de pitch), a peça certa foi o 1.º candidato em 15; a maioria das falhas são toques que não caem na peça (o ícone do hotspot fica abaixo dela). Scores das peças certas: mediana 1,0, mínimo 0,53; dos candidatos errados: mediana 0,61.
