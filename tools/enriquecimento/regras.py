"""Enriquecimento de uma ficha só com regras, sem modelo.

`enriquecer(ficha)` recebe a ficha como está no índice das peças e devolve os campos
novos, cada valor com a fonte:

  campo — um campo estruturado da ficha di-lo (material, técnica, categoria, centro de produção, dimensões)
  texto — está escrito no título ou na descrição; guarda-se o excerto
  regra — deduzido de outros valores (a função sai do tipo de objeto)

A ficha manda: nada aqui vem de imagens nem de modelos. O que se aprendeu no piloto
(2026-11-24) está nas regras: numa pintura, num desenho ou numa gravura a descrição
fala do que está representado, e não da matéria da peça; numa escultura só a primeira
frase apresenta a peça; «caracteres góticos» é a letra de um manuscrito e não um estilo.
"""
from __future__ import annotations

import re
from typing import Any

from app.services.enriched_fields import word_forms

from . import vocab as V

DUAS_DIMENSOES = {"pintura", "desenho", "gravura"}
# técnicas que numa descrição têm outro sentido (figuras «em vulto», «escrito a», um estofo de cadeira)
SO_NO_CAMPO = {"escultura de vulto", "manuscrito", "desenho", "gravação", "decoração pintada", "moldagem", "cozedura", "modelação"}
LETRA_GOTICA = re.compile(r"(?:caracteres|letras?|escrita|caligrafia|minusculas?|maiusculas?) (?:em )?gotic\w+")
ANTES_DO_NOME = {"grande", "grandes", "pequeno", "pequena", "pequenos", "pequenas", "par", "de", "dois", "duas", "conjunto"}
MEDIDAS = (("Alt", "altura_cm"), ("Larg", "largura_cm"), ("Comp", "comprimento_cm"), ("Prof", "profundidade_cm"), ("Di[aâ]m", "diametro_cm"))


def _achar(lista, texto: Any, fonte: str, onde: str, saida: list[dict], vistos: set[str]) -> None:
    normalizado = V.n(texto)
    for valor, padrao in lista:
        m = padrao.search(normalizado)
        if m and valor not in vistos:
            vistos.add(valor)
            saida.append({"valor": valor, "fonte": fonte, "onde": onde, "excerto": normalizado[max(0, m.start() - 25): m.end() + 25]})


def _com_pais(saida: list[dict]) -> list[dict]:
    """«madeira › carvalho» traz também «madeira», com a mesma fonte."""
    vistos = {v["valor"] for v in saida}
    for v in list(saida):
        pai = v["valor"].split(" › ")[0]
        if pai != v["valor"] and pai not in vistos:
            vistos.add(pai)
            saida.append({**v, "valor": pai})
    return saida


def tipo_objeto(ficha: dict[str, Any]) -> dict[str, str] | None:
    cat, titulo = V.n(ficha.get("category")), V.n(ficha.get("title"))
    tecnica, material = V.n(ficha.get("technique")), V.n(ficha.get("support_or_material"))
    def t(valor: str, onde: str, fonte: str = "campo") -> dict[str, str]:
        return {"valor": valor, "fonte": fonte, "onde": onde}
    if cat == "pintura":
        if re.search(r"velino|pergaminho", material) or (V.MANUSCRITO_TITULO.search(titulo) and "tempera" in tecnica):
            return t("manuscrito iluminado › fólio", "material, técnica e título")
        if "triptico" in titulo + " " + V.n(ficha.get("description"))[:300]:
            return t("pintura › tríptico", "título/descrição", "texto")
        if "tela" in material + " " + tecnica:
            return t("pintura › sobre tela", "material/técnica")
        if re.search(r"madeira|carvalho|castanho|choupo|pinho", material + " " + tecnica):
            return t("pintura › sobre madeira", "material/técnica")
        if "cobre" in material + " " + tecnica:
            return t("pintura › sobre cobre", "material/técnica")
        return t("pintura", "categoria")
    if cat in ("desenho", "gravura"):
        return t(cat, "categoria")
    if cat == "escultura":
        if "relevo" in tecnica:
            return t("escultura › relevo", "técnica")
        if "vulto" in tecnica:
            return t("escultura › vulto", "técnica")
        return t("escultura", "categoria")
    ramo = V.RAMO_DA_CATEGORIA.get(cat)
    # a palavra principal do título, saltando o que vem antes dela («Grande copo», «Par de castiçais»)
    palavras = [p for p in re.findall(r"[a-z]+", titulo)[:4] if p not in ANTES_DO_NOME][:1]
    folha = next((V.TIPO_POR_CHAVE[forma] for forma in word_forms(palavras[0]) if forma in V.TIPO_POR_CHAVE), None) if palavras else None
    if folha:
        return t(f"{ramo or 'objeto'} › {folha}", "título e categoria")
    if ramo and cat:
        return t(ramo, "categoria")
    return None


def medidas(texto: Any) -> tuple[dict[str, float], str | None]:
    limpo = str(texto or "").replace(", ", ",")
    achados: dict[str, float] = {}
    for rotulo, chave in MEDIDAS:
        m = re.search(rotulo + r"\.?\s*([0-9]+(?:[.,][0-9]+)?)", limpo)
        if m:
            achados[chave] = float(m.group(1).replace(",", "."))
    if not achados:
        return {}, None
    maior = max(achados.values())
    if "mm" in limpo and "cm" not in limpo:
        maior /= 10
    return achados, next(nome for limite, nome in V.CLASSES_TAMANHO if maior < limite)


def origem(ficha: dict[str, Any]) -> list[dict[str, Any]]:
    saida: list[dict[str, Any]] = []

    def procurar(texto: str, fonte: str, onde: str) -> None:
        vistos = {(o["pais"], o["regiao"]) for o in saida}
        for padrao, pais, regiao in V.LUGARES:
            m = padrao.search(texto)
            if not m or (pais, regiao) in vistos or (regiao is None and any(o["pais"] == pais for o in saida)):
                continue
            vistos.add((pais, regiao))
            saida.append({"pais": pais, "regiao": regiao, "fonte": fonte, "onde": onde, "incerto": "?" in texto,
                          "excerto": texto[max(0, m.start() - 25): m.end() + 25]})

    centro = V.n(ficha.get("production_center"))
    if centro and not V.SEM_LUGAR.search(centro):
        procurar(centro, "campo", "centro de produção")
    if not saida and V.n(ficha.get("category")) not in DUAS_DIMENSOES:
        # só o título e o início da descrição: «Contador indo-português», «Prato de porcelana chinesa»
        procurar(V.n(ficha.get("title")) + " . " + V.n(ficha.get("description"))[:220], "texto", "título/descrição")
    return saida


def funcao(ficha: dict[str, Any], tipo: str | None) -> list[str]:
    cat, titulo = V.n(ficha.get("category")), V.n(ficha.get("title"))
    palavra = V.n((tipo or "").split(" › ")[-1])
    saida = [nome for nome, palavras in V.FUNCAO_POR_TIPO.items() if palavra in palavras]
    if tipo and tipo.startswith("manuscrito iluminado"):
        saida.append("devocional")
    elif cat == "escultura" and V.RELIGIOSO_TITULO.search(titulo):
        saida.append("devocional")
    elif cat == "desenho" and re.match(r"estudos?|academia", titulo):
        saida.append("estudo ou preparatório")
    return saida


def enriquecer(ficha: dict[str, Any]) -> dict[str, Any]:
    cat = V.n(ficha.get("category"))
    titulo, descricao = str(ficha.get("title") or ""), str(ficha.get("description") or "")
    objeto = cat not in DUAS_DIMENSOES
    # numa escultura só o início da descrição apresenta a peça; o resto conta a cena
    descricao_da_peca = descricao[:160] if cat == "escultura" else descricao

    materiais: list[dict] = []
    vistos: set[str] = set()
    _achar(V.MATERIAIS, ficha.get("support_or_material"), "campo", "material", materiais, vistos)
    _achar(V.MATERIAIS, ficha.get("technique"), "campo", "técnica", materiais, vistos)
    if cat == "vidros" and "vidro" not in vistos:
        vistos.add("vidro")
        materiais.append({"valor": "vidro", "fonte": "campo", "onde": "categoria", "excerto": "vidros"})
    if objeto:
        _achar(V.MATERIAIS, titulo + " . " + descricao_da_peca, "texto", "título/descrição", materiais, vistos)

    tecnicas: list[dict] = []
    vistos = set()
    _achar(V.TECNICAS, ficha.get("technique"), "campo", "técnica", tecnicas, vistos)
    if objeto:
        fora = SO_NO_CAMPO | (set() if cat == "escultura" else {"estofado"})
        _achar([(v, p) for v, p in V.TECNICAS if v not in fora], titulo + " . " + descricao_da_peca, "texto", "título/descrição", tecnicas, vistos)
    em_velino = any(m["valor"] in ("velino", "pergaminho") for m in materiais)
    if em_velino and "pintura a têmpera" in vistos and "iluminura" not in vistos:
        tecnicas.append({"valor": "iluminura", "fonte": "regra", "onde": "têmpera sobre velino ou pergaminho", "excerto": ""})

    estilos: list[dict] = []
    texto_estilo = LETRA_GOTICA.sub(" ", V.n(titulo + " . " + descricao + " . " + str(ficha.get("origin_history") or "")))
    _achar(V.ESTILOS, texto_estilo, "texto", "título/descrição/historial", estilos, set())

    cores: list[dict] = []
    if cat not in ("desenho", "gravura"):  # aí a descrição diz as cores do que está representado
        _achar(V.CORES, titulo + " . " + descricao, "texto", "título/descrição", cores, set())

    tipo = tipo_objeto(ficha)
    numeros, classe = medidas(ficha.get("dimensions_cm"))
    return {
        "versao": V.VERSAO,
        "tipo_objeto": tipo,
        "materiais": _com_pais(materiais),
        "tecnicas": _com_pais(tecnicas),
        "estilo": estilos,
        "origem": origem(ficha),
        "cores": cores,
        "funcao": [{"valor": f, "fonte": "regra", "onde": "tipo e categoria"} for f in funcao(ficha, tipo["valor"] if tipo else None)],
        "medidas": numeros,
        "tamanho": classe,
    }


def _folhas(valores: list[dict], so_fonte: str | None = None) -> list[str]:
    """Os valores como se pesquisam: «madeira › carvalho» fica «carvalho» (e «madeira» já lá está à parte)."""
    saida: list[str] = []
    for v in valores:
        if so_fonte and v["fonte"] != so_fonte:
            continue
        folha = v["valor"].split(" › ")[-1]
        if folha not in saida:
            saida.append(folha)
    return saida


def documento(ficha: dict[str, Any], enr: dict[str, Any]) -> dict[str, Any]:
    """O documento do índice de enriquecimento: os campos novos em forma pesquisável, mais os campos da ficha
    de que as contagens e os filtros precisam (museu, visita, datas), para uma só consulta responder."""
    tipo = enr["tipo_objeto"]["valor"] if enr["tipo_objeto"] else None
    doc: dict[str, Any] = {
        "artifact_id": ficha.get("artifact_id"), "museum_id": ficha.get("museum_id"), "inventory_number": ficha.get("inventory_number"),
        "title": ficha.get("title"), "category": ficha.get("category"), "in_tour": bool(ficha.get("in_tour")),
        "start_year": ficha.get("start_year"), "end_year": ficha.get("end_year"),
        "enr_versao": enr["versao"],
        "enr_tipo": [p for p in (tipo.split(" › ") if tipo else [])],
        "enr_tipo_caminho": tipo,
        "enr_materiais": _folhas(enr["materiais"]), "enr_materiais_ficha": _folhas(enr["materiais"], "campo"),
        "enr_tecnicas": _folhas(enr["tecnicas"]), "enr_tecnicas_ficha": _folhas(enr["tecnicas"], "campo"),
        "enr_estilo": _folhas(enr["estilo"]), "enr_cores": _folhas(enr["cores"]), "enr_funcao": _folhas(enr["funcao"]),
        "enr_origem_pais": list(dict.fromkeys(o["pais"] for o in enr["origem"])),
        "enr_origem_regiao": list(dict.fromkeys(o["regiao"] for o in enr["origem"] if o["regiao"])),
        "enr_tamanho": enr["tamanho"],
        "enr_detalhe": enr,
    }
    for chave, valor in enr["medidas"].items():
        doc[f"enr_{chave}"] = valor
    return {k: v for k, v in doc.items() if v is not None}
