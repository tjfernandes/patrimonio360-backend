# POIs das visitas: mapeamento hotspot → número de inventário

O backend liga as peças às visitas 3DVista através de `poi_tours/panorama-overlays-inventory-<museu>.json`. Esse ficheiro dá o «ver na visita», o «onde estou?» e o campo `in_tour` no índice. Quando uma visita é republicada, os ids `overlay_*` e `panorama_*` podem mudar e o ficheiro fica desatualizado. O sintoma é o «ver na visita» deixar de levar a lado nenhum, ou POIs novos nunca aparecerem.

Correr tudo a partir da raiz do repositório, com o Python do backend (`~/miniconda3/envs/visitas_virtuais/bin/python`), que lê o `.env` da raiz para chegar ao OpenSearch.

## Gerar o mapa de um museu

As visitas só existem no VPS. Este script descarrega o `script_general.js` da visita publicada para `tmp/visitas/<museu>/` e gera o mapa sem browser:

```bash
tools/poi_mapping/obter_poi_mapping.sh mnaa
```

- Escreve em `poi_tours/panorama-overlays-inventory-<museu>.json`. Como é um ficheiro versionado, faz-se num ramo próprio e entra por PR.
- O nome da zona de cada POI é o `subtitle` do panorama onde está, lido dos `locale/*.txt` da visita. No MNAA os blocos de JS dizem «Sala 37», «Sala 64», e os `subtitle` dizem a coleção. O valor do bloco fica em `locationOriginal`. Para manter o valor do bloco, passa-se `--location-from js` ao `extract_poi_mapping.py`.
- Os lugares da visita para o «onde estou?» e para a lista de zonas vêm de `scripts/build_tour_locations.py`, que se corre outra vez sempre que uma visita é republicada. No CMAG e no MNAA também usa o `subtitle`.
- Com `--so-comparar`, escreve só em `tmp/visitas/<museu>/` e mostra o que mudaria.
- Se já houver um mapa, mostra os POIs novos, os que desapareceram e os inventários alterados, e mantém as listas de números que o export novo colapsou numa só string.

Por baixo corre o `extract_poi_mapping.py`, que também aceita uma pasta de export local.

## Tesouros Nacionais: `nationalTreasure`

Cada entrada do mapa tem sempre `"nationalTreasure": true | false`. É a única fonte da marca: o backend lê-a para o campo `national_treasure` dos cartões, para os pedidos sobre tesouros e para o `GET /api/v1/chat/tour/pois`. Os ficheiros são relidos quando mudam, sem reiniciar.

- Uma entrada é tesouro quando a página do POI que o hotspot abre tem o selo visível `<b style="color: goldenrod"> Tesouro Nacional </b>`. A página é `iFrames/generated-html/<sala>/<peça>.html` (`iframes/` no MNT), e chega-se a ela pelo `url` do `PopupWebFrameBehaviour` no `locale/*.txt`.
- O campo `"treasure": "Tesouro Nacional"` do JSON embutido em todas as páginas é uma constante do modelo e não conta.
- A mesma peça em vários hotspots fica com o mesmo valor.
- O `obter_poi_mapping.sh` descarrega também as páginas dos POIs, e o `extract_poi_mapping.py` escreve o campo em cada entrada que gera.
- Para atualizar só a marca sem regenerar o mapa, o que mantém as correções feitas à mão, corre-se:

  ```bash
  tools/poi_mapping/obter_poi_mapping.sh mnaa --so-comparar
  ~/miniconda3/envs/visitas_virtuais/bin/python tools/poi_mapping/marcar_tesouros_poi.py mnaa
  ```

  - Sem museus, faz todos os de `poi_tours/`. Tem `--simular`.
  - Com `--acrescentar-novos`, junta os POIs da visita que o mapa ainda não tem, com os números tal como a visita os anuncia. Esses números confirmam-se depois no catálogo, como em baixo.

## Validar os números contra o índice

```bash
~/miniconda3/envs/visitas_virtuais/bin/python tools/poi_mapping/validate_tour.py tmp/visitas/mnaa --museum mnaa
```

Para cada hotspot diz se o número existe tal e qual, se existe escrito de outra forma, ou se não existe. Só faz leituras no OpenSearch. Sai com código 1 se houver números em falta. Aceita também o `.vtp` do projeto 3DVista.

Escreve em `FIXES/<museu>/`, que não vai para o git:

- `nao_existem_no_raiz.xlsx`: números que não existem no Raiz, para o museu confirmar;
- `sem_hotspot.xlsx`: fichas da visita sem hotspot, para quem faz a visita;
- `mal_escritos.xlsx`: hotspots com o número em formato errado, e o que deve ficar;
- `validos.xlsx`: os que estão bem, com o título do Raiz para confirmar a olho;
- `validacao.json`: o resultado completo.

Com `--fix`, corrige no ficheiro de origem o que é mecânico e deixa uma cópia `.bak-<data>`. Numa pasta de `tmp/visitas/` isso só muda a cópia local: a correção tem de ir depois para o projeto 3DVista, que é quem republica a visita.

Outras verificações:

- `validar_inventarios_poi.py <museu>` confirma no índice os números do ficheiro de `poi_tours/` e corrige os que resolve, como `MNAz_185_Az` para `MNAz 185 Az`. Tem `--simular`.
- `validate_poi_mapping.py poi_tours/panorama-overlays-inventory-<museu>.json --museum <museu>` só lista os números que não existem no catálogo, e sai com código 1.

## Número errado na visita: corrige-se no mapa, nunca nos ficheiros da visita

Não se editam o `script_general.js` nem o `script_mobile.js` de uma visita publicada. O player do 3DVista verifica a integridade desses ficheiros, e qualquer alteração faz aparecer a marca de água «3DVista Academic» em toda a visita. Aconteceu a 18/09 e a 29/09/2026. Por isso, não se usa o `--fix` do `validate_tour.py` nem o `fix_smart_quotes.py` numa visita publicada.

Quando a visita anuncia um número que não é o do catálogo, a correspondência escreve-se no mapa dos POIs:

```json
{
  "inventoryIds": ["MNAz 6991/2 Az"],
  "inventoryIdsOriginal": ["MNAz 6991 Az"],
  "inventoryIdsNota": "na visita está «MNAz 6991 Az», que não existe no catálogo"
}
```

O backend traduz o número quando a visita avisa que um hotspot foi aberto. A correção definitiva faz-se no projeto do 3DVista, antes de exportar.

O `obter_poi_mapping.sh` gera o mapa a partir da visita e não conhece estas correspondências. Depois de o correr, comparam-se as entradas com `inventoryIdsOriginal` do mapa anterior e repõem-se as que ainda fizerem falta.

## Depois de o mapa entrar no `main`

```bash
~/miniconda3/envs/visitas_virtuais/bin/python tools/poi_mapping/marcar_in_tour.py mnaa
```

Sincroniza o `in_tour` do índice com os POIs e escreve só os documentos que estão errados. Tem `--simular`. Sem museu, faz mnt, mnsr, mnaz e mj.

A seguir reinicia-se o backend. Se a visita for nova, instalam-se nela os scripts de ponte com `deploy/tours/instalar_focus_no_vps.sh <museu>`.

`definir_in_tour.py` liga ou desliga o `in_tour` à mão, para uma lista de números. A próxima corrida do `marcar_in_tour.py` desfaz essa marca.

## Antes de exportar do 3DVista

- `audit_vtp.py <projeto.vtp>` cruza os hotspots do projeto com as fichas em `Conteudos/`: hotspots sem bloco JS, blocos com aspas curvas ou SN sem número, fichas sem hotspot. Não precisa do backend.
- `fix_vtp_unicode.py` corrige os caminhos com acentos decompostos dos projetos `.vtp` vindos de macOS, que o 3DVista no Windows dá como ficheiros em falta.
- `fix_smart_quotes.py` troca as aspas curvas nos scripts de um export, que partem o aviso ao assistente quando se abre um POI.
- `listar_acoes_partidas.py` lista as ações partidas das visitas publicadas, como em `acoes-a-corrigir-3dvista.csv`.

## O caminho antigo, pelo browser

`overlay-inventory-mapper.js` cola-se na consola da página da visita e descarrega o mesmo ficheiro. Serve para confirmar um caso duvidoso. O `obter_poi_mapping.sh` substitui-o no caso normal.
