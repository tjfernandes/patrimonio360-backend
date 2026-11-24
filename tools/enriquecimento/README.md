# Enriquecimento das fichas por regras

Lê as fichas de um museu no índice das peças, aplica regras sem modelo e escreve os campos novos (tipo de objeto,
materiais, técnicas, função, origem, cores, estilo, tamanho) num índice próprio, com um documento por peça. O que são
os campos, como o backend os usa e o que foi medido está em `docs/enriquecimento-por-campos.md`.

| Ficheiro | O que faz |
|---|---|
| `vocab.py` | as listas fechadas e os padrões que levam o texto da ficha até elas |
| `regras.py` | `enriquecer(ficha)`: os valores de uma ficha, cada um com a fonte e o excerto; `documento(...)`: o que vai para o índice |
| `temas.py` | pergunta ao modelo de pré-processamento os temas de cada peça (resposta presa a um esquema, excerto verificado na ficha) e decide, sem modelo, quais entram; as regras de aceitação vêm da validação de 2026-11-24 |
| `construir.py` | lê as fichas (só leitura), escreve `.cache/<museu>.jsonl` e mostra a cobertura e o que ficou sem correspondência |
| `indexar.py` | escreve no índice de enriquecimento; simula por omissão, recusa outros índices e não apaga nenhum |

## Acrescentar um museu

1. `python -m tools.enriquecimento.construir --museu <id>` e ler o fim da saída: os valores dos campos material, técnica
   e centro de produção que nenhuma regra apanhou.
2. Acrescentar a `vocab.py` o que faltar (um museu de traje ou de arqueologia tem tipos e materiais que o MNAA não tem)
   e mudar `VERSAO`.
3. Conferir uma amostra à mão antes de indexar: os valores vindos de campos da ficha e, à parte, os tirados do título e
   da descrição, que é onde os erros aparecem.
4. `python -m tools.enriquecimento.temas --museu <id>` para os temas, e voltar a correr o `construir.py`. As regras de
   aceitação dos temas foram validadas só no MNAA: noutro museu convém conferir uma amostra antes de os usar.
5. `python -m tools.enriquecimento.indexar --museu <id> --executar`.

## Regras que custaram erros

- Numa pintura, num desenho ou numa gravura, a descrição fala do que está representado: materiais e técnicas só dos
  campos da ficha.
- Numa escultura só o início da descrição apresenta a peça; o resto conta a cena («sentada numa pedra»).
- «Caracteres góticos» é a letra de um manuscrito, não o estilo da peça.
- «Bordo» é a borda de um prato, não a cor vinoso; «laranja da China» é a fruta.
- «Estofado» numa cadeira é o assento, não a técnica de escultura.
