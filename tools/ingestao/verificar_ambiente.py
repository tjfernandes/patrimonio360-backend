"""Verificações antes (e depois) de ingerir um museu.

Antes de gastar horas de scraping e de GPU, confirma o que costuma falhar:
os projetos e o ambiente, as credenciais do OpenSearch, o museu conhecido pelo
indexador, os ficheiros de origem, o espaço em disco e a GPU livre.

    python tools/ingestao/verificar_ambiente.py --museu "Museu Nacional do Azulejo"
    python tools/ingestao/verificar_ambiente.py --museu "…" --contagens mnaz

Sai com 0 se estiver tudo pronto e 1 se houver algum impedimento (os avisos
não fazem falhar). Corre com o Python do ambiente `visitas_virtuais`.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

ASSISTANT = Path(__file__).resolve().parents[2]
RAIZ = ASSISTANT.parent / "raiz_scraper"
INDEXER = ASSISTANT.parent / "patrimonio360_indexer"
RAIZAMALIA = ASSISTANT.parent / "RaizAMALIA"
SAIDA = RAIZ / "output"

# o que o build_raiz_index.py lê (os móveis vêm do XML, não do Movel.xlsx)
LIVROS_RAIZ = (
    "Movel.xlsx", "Documento.xlsx", "Imaterial.xlsx",
    "Autor.xlsx", "Conjunto.xlsx", "Exposicao.xlsx", "ExposicaoOnline.xlsx", "Bibliografia.xlsx",
    "AutorElemento.xlsx", "ConjuntoElemento.xlsx", "ExposicaoElemento.xlsx",
    "ExposicaoOnlineElemento.xlsx", "BibliografiaElemento.xlsx",
)

problemas: list[str] = []
avisos: list[str] = []


def ok(texto: str) -> None:
    print(f"  ok      {texto}")


def aviso(texto: str) -> None:
    avisos.append(texto)
    print(f"  aviso   {texto}")


def erro(texto: str) -> None:
    problemas.append(texto)
    print(f"  FALHA   {texto}")


def pasta_do_museu(nome: str) -> str:
    """Cópia exata do `slug()` do raiz_scraper e do `tipo_slug()` do indexador.

    Só tira acentos e junta as palavras por _: a pontuação FICA. «Casa-Museu
    Dr. Anastácio Gonçalves» dá `casa-museu_dr._anastacio_goncalves`, com o
    hífen e o ponto. Trocar a pontuação por _ (o que isto fazia antes) dá o
    mesmo resultado nos quatro museus atuais, que não têm pontuação nenhuma, e
    um nome errado em qualquer museu que tenha: a pasta dos dados não seria
    encontrada e a chave do MUSEUM_NAME_TO_ID não bateria certo.
    """
    sem_acentos = unicodedata.normalize("NFKD", nome or "")
    sem_acentos = "".join(c for c in sem_acentos if not unicodedata.combining(c))
    return "_".join(sem_acentos.lower().split())


def ler_env(ficheiro: Path) -> dict[str, str]:
    valores: dict[str, str] = {}
    if not ficheiro.exists():
        return valores
    for linha in ficheiro.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if not linha or linha.startswith("#") or "=" not in linha:
            continue
        chave, valor = linha.split("=", 1)
        valores[chave.strip()] = valor.strip().strip('"').strip("'")
    return valores


def museus_conhecidos() -> dict[str, str]:
    """`MUSEUM_NAME_TO_ID` do indexador: um museu novo tem de ser lá acrescentado."""
    ficheiro = INDEXER / "scripts" / "ingest_museum_opensearch.py"
    if not ficheiro.exists():
        return {}
    texto = ficheiro.read_text(encoding="utf-8")
    bloco = re.search(r"MUSEUM_NAME_TO_ID[^{]*\{(.*?)\}", texto, re.S)
    if not bloco:
        return {}
    return {nome: ident for nome, ident in re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', bloco.group(1))}


def verificar(museu: str, fonte: str) -> None:
    print("\n=== projetos e ambiente")
    for pasta in (RAIZ, INDEXER):
        (ok if pasta.exists() else erro)(f"{pasta}")
    ambiente = os.environ.get("CONDA_DEFAULT_ENV") or Path(sys.executable).parents[1].name
    (ok if ambiente == "visitas_virtuais" else aviso)(f"ambiente: {ambiente} (esperado visitas_virtuais)")

    print("\n=== OpenSearch (indexer/.env)")
    env = ler_env(INDEXER / ".env")
    for chave in ("OPENSEARCH_URL", "OPENSEARCH_USER", "OPENSEARCH_PASSWORD"):
        (ok if env.get(chave) or os.environ.get(chave) else erro)(f"{chave} definido")
    if env.get("OPENSEARCH_URL"):
        ok(f"destino: {env['OPENSEARCH_URL']}")
    for chave in ("OPENSEARCH_ARTIFACT_INDEX", "OPENSEARCH_IMAGE_INDEX", "TOURS_DIR"):
        valor = env.get(chave) or os.environ.get(chave)
        (ok if valor else aviso)(f"{chave} = {valor or '(por omissão)'}")

    print("\n=== museu")
    conhecidos = museus_conhecidos()
    # as chaves do indexador são o slug do nome (tipo_slug), não o nome escrito
    ident = conhecidos.get(pasta_do_museu(museu)) or conhecidos.get(museu)
    if ident:
        ok(f"«{museu}» -> museum_id «{ident}» no indexador")
    else:
        erro(
            f"«{museu}» não está em MUSEUM_NAME_TO_ID "
            f"({', '.join(sorted(conhecidos.values())) or 'vazio'}): os documentos ficariam sem museum_id. "
            "Acrescentar em patrimonio360_indexer/scripts/ingest_museum_opensearch.py"
        )
    pasta = SAIDA / "raiz_index" / "museus" / pasta_do_museu(museu)
    ok(f"pasta de dados: {pasta}")

    print("\n=== ficheiros de origem")
    if not SAIDA.exists():
        erro(f"{SAIDA} não existe (é o --output-root do indexador)")
    if fonte == "raiz":
        objetos = SAIDA / "raiz_index" / "objetos.ndjson"
        if RAIZAMALIA.exists():
            faltam = [nome for nome in LIVROS_RAIZ if not (RAIZAMALIA / nome).exists()]
            (ok if not faltam else erro)(
                f"{RAIZAMALIA.name}: {len(LIVROS_RAIZ) - len(faltam)}/{len(LIVROS_RAIZ)} Excels do export"
                + (f" (faltam: {', '.join(faltam)})" if faltam else "")
            )
            # o Movel.xlsx tem ~2500 linhas partidas: o build_raiz_index lê os móveis do XML
            xml = next(RAIZAMALIA.glob("raizonline_objetos*.xml"), None)
            (ok if xml else erro)(f"XML dos móveis {xml.name if xml else '(em falta: sem ele o passo «raiz» não corre)'}")
        else:
            aviso(f"{RAIZAMALIA} não existe: não se pode correr o passo «raiz», só reusar o objetos.ndjson que houver")
        if objetos.exists():
            ok(f"{objetos.name} ({objetos.stat().st_size / 2**20:.0f} MB, todos os museus)")
        elif RAIZAMALIA.exists():
            aviso(f"{objetos.name} ainda não existe: correr o passo «raiz» para o construir dos Excels")
        else:
            erro(f"{objetos} não existe e não há ../RaizAMALIA para o construir")
    else:
        excel = next(RAIZ.glob("Invent*.xlsx"), None)
        (ok if excel else erro)(f"Excel do inventário em {RAIZ}")
        dados = RAIZ.parent / "Dados_MJ"
        (ok if dados.exists() else erro)(f"{dados} (imagens locais do museu manual)")
    ndjson = pasta / "objetos_com_imagens.ndjson"
    if ndjson.exists():
        linhas = sum(1 for _ in ndjson.open(encoding="utf-8"))
        ok(f"{ndjson.name} já existe ({linhas} objetos): o passo «dados» pode ser saltado")
    else:
        aviso(f"{ndjson.name} ainda não existe: é preciso correr o passo «dados»")

    print("\n=== máquina")
    livre = shutil.disk_usage(RAIZ).free / 2**30
    (ok if livre > 30 else aviso)(f"disco livre: {livre:.0f} GB (as imagens de um museu ocupam alguns GB)")
    try:
        saida = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip().splitlines()[0]
        usada, total = (int(x) for x in saida.split(","))
        livre_gpu = (total - usada) / 1024
        mensagem = f"GPU: {livre_gpu:.0f} GB livres de {total / 1024:.0f} GB"
        if livre_gpu < 30:
            aviso(mensagem + " — os embedders 8B pedem ~30 GB; parar o vLLM e o backend antes de indexar")
        else:
            ok(mensagem)
    except Exception as exc:  # sem GPU ou sem nvidia-smi
        aviso(f"não consegui ler a GPU ({exc}); a indexação com embeddings precisa de CUDA")


def ligar_backend():
    """Cliente do OpenSearch e definições, pelas contas do backend.

    É o backend quem consulta o índice em produção, por isso é a configuração
    dele que diz onde a ingestão tem de escrever. (As credenciais do .env do
    indexador deram 401 a 2026-09-20.)
    """
    sys.path.insert(0, str(ASSISTANT / "backend"))
    cwd = Path.cwd()
    os.chdir(ASSISTANT / "backend")
    from app.core.config import get_settings  # noqa: PLC0415
    from opensearchpy import OpenSearch  # noqa: PLC0415

    s = get_settings()
    os.chdir(cwd)
    cliente = OpenSearch(
        hosts=[{"host": s.OPENSEARCH_HOST.replace("https://", "").replace("http://", ""),
                "port": int(s.OPENSEARCH_PORT)}],
        http_auth=(s.OPENSEARCH_USERNAME, s.OPENSEARCH_PASSWORD),
        use_ssl=True, verify_certs=False, ssl_show_warn=False, timeout=60,
    )
    return cliente, s


def dimensao(cliente, indice: str) -> int | None:
    """Dimensão dos vetores knn de um índice (None se não existir)."""
    if not cliente.indices.exists(index=indice):
        return None
    mapa = cliente.indices.get_mapping(index=indice)[indice]["mappings"].get("properties", {})
    dims = {p.get("dimension") for p in mapa.values() if p.get("type") == "knn_vector"}
    return max(dims) if dims else None


def verificar_indices() -> None:
    """Os índices onde se vai escrever são mesmo os que o assistente lê?

    O indexador, sem --entity-index, escreve nos índices antigos de 2560
    dimensões e rebenta a meio com «mapping dimension != embedding dimension»
    (aconteceu na 1.ª ingestão do CMAG, a 2026-09-20, depois de artefactos e
    imagens já terem entrado). Confirmar as dimensões antes custa um segundo.
    """
    print("\n=== índices (os que o backend lê)")
    try:
        cliente, s = ligar_backend()
    except Exception as exc:
        erro(f"não consegui ligar ao OpenSearch: {exc}")
        return
    referencia = dimensao(cliente, s.OPENSEARCH_INDEX_ARTIFACT)
    entidades = (s.OPENSEARCH_INDEX_AUTOR, s.OPENSEARCH_INDEX_CONJUNTO, s.OPENSEARCH_INDEX_EXPOSICAO)
    for indice in (s.OPENSEARCH_INDEX_ARTIFACT, s.OPENSEARCH_INDEX_IMAGE, *entidades):
        dim = dimensao(cliente, indice)
        if dim is None:
            aviso(f"{indice}: não existe (será criado na ingestão, a partir de mappings/museum_*_v4.json)")
        elif indice in entidades and referencia and dim != referencia:
            # as entidades usam o mesmo modelo de texto dos artefactos
            erro(f"{indice}: vetores de {dim} dimensões, mas os artefactos são de {referencia} "
                 "— a ingestão das entidades pára a meio (as dimensões knn são imutáveis)")
        else:
            ok(f"{indice}: {dim} dimensões")


def contagens(slug: str, museu: str = "") -> None:
    """Depois de indexar: quantos documentos ficaram no índice."""
    print(f"\n=== contagens no índice para «{slug}»")
    try:
        cliente, s = ligar_backend()
        for indice in (s.OPENSEARCH_INDEX_ARTIFACT, s.OPENSEARCH_INDEX_IMAGE):
            corpo = {"query": {"bool": {"filter": [{"term": {"museum_id": slug}}]}}}
            total = cliente.count(index=indice, body=corpo)["count"]
            corpo["query"]["bool"]["filter"].append({"term": {"in_tour": True}})
            no_tour = cliente.count(index=indice, body=corpo)["count"]
            print(f"  {indice}: {total} documentos | {no_tour} na visita")
        # as entidades são globais e não têm museum_id: o museu está na lista `museums`
        if museu:
            for indice in (s.OPENSEARCH_INDEX_AUTOR, s.OPENSEARCH_INDEX_CONJUNTO, s.OPENSEARCH_INDEX_EXPOSICAO):
                corpo = {"query": {"bool": {"filter": [{"term": {"museums": museu}}]}}}
                print(f"  {indice}: {cliente.count(index=indice, body=corpo)['count']} com peças neste museu")
    except Exception as exc:
        erro(f"não consegui contar no índice: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--museu", required=True, help='nome exato, ex.: "Museu Nacional do Azulejo"')
    parser.add_argument("--fonte", choices=("raiz", "excel"), default="raiz", help="origem dos dados")
    parser.add_argument("--contagens", metavar="SLUG", default=None, help="contar documentos já indexados (mnt, mnsr, mnaz, mj)")
    args = parser.parse_args()

    verificar(args.museu, args.fonte)
    verificar_indices()
    if args.contagens:
        contagens(args.contagens, args.museu)

    print()
    if problemas:
        print(f"{len(problemas)} impedimento(s):")
        for problema in problemas:
            print(f"  - {problema}")
    if avisos:
        print(f"{len(avisos)} aviso(s) (não impedem, mas convém ler).")
    if not problemas:
        print("pronto a ingerir.")
    sys.exit(1 if problemas else 0)
