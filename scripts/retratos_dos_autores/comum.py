"""Peças comuns: Qwen local (:8005), Wikidata/Commons, ficheiros."""
import base64
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

PASTA = os.path.expanduser(os.environ.get("RETRATOS_PASTA") or "~/.cache/patrimonio360-retratos")
os.makedirs(PASTA, exist_ok=True)
QWEN = "http://127.0.0.1:8005/v1/chat/completions"
MODELO = "Qwen/Qwen3-VL-8B-Instruct-FP8"
AGENTE = "museum-catalogue-research/1.0 (python-urllib)"
_trava = threading.Lock()


def ler(nome, omissao=None):
    caminho = os.path.join(PASTA, nome)
    if not os.path.exists(caminho):
        return omissao
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


def gravar(nome, dados):
    caminho = os.path.join(PASTA, nome)
    with _trava:
        with open(caminho + ".tmp", "w", encoding="utf-8") as f:
            json.dump(dados, f, ensure_ascii=False, indent=1)
        os.replace(caminho + ".tmp", caminho)


def qwen(prompt, *, imagem: bytes | None = None, mime="image/jpeg", max_tokens=120):
    """Devolve o JSON da resposta (dict) ou None."""
    conteudo = prompt
    if imagem is not None:
        conteudo = [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(imagem).decode()}"}},
            {"type": "text", "text": prompt},
        ]
    corpo = {"model": MODELO, "messages": [{"role": "user", "content": conteudo}], "temperature": 0, "max_tokens": max_tokens,
             "response_format": {"type": "json_object"}}
    pedido = urllib.request.Request(QWEN, data=json.dumps(corpo).encode(), headers={"Content-Type": "application/json"})
    for tentativa in range(3):
        try:
            bruto = json.loads(urllib.request.urlopen(pedido, timeout=120).read())["choices"][0]["message"]["content"]
            dados = json.loads(bruto)
            return dados if isinstance(dados, dict) else None
        except json.JSONDecodeError:
            return None
        except Exception:  # noqa: BLE001
            time.sleep(2 + 3 * tentativa)
    return None


def api(base, **params):
    """GET a uma API MediaWiki, com repetição quando o servidor pede para esperar."""
    params.setdefault("format", "json")
    params.setdefault("maxlag", "5")
    url = base + "?" + urllib.parse.urlencode(params)
    for tentativa in range(6):
        try:
            pedido = urllib.request.Request(url, headers={"User-Agent": AGENTE})
            dados = json.loads(urllib.request.urlopen(pedido, timeout=40).read())
            if isinstance(dados, dict) and dados.get("error", {}).get("code") == "maxlag":
                time.sleep(5)
                continue
            return dados
        except urllib.error.HTTPError as erro:
            time.sleep(int(erro.headers.get("Retry-After") or 0) or 5 * (tentativa + 1))
        except Exception:  # noqa: BLE001
            time.sleep(3 * (tentativa + 1))
    return {}


WD = "https://www.wikidata.org/w/api.php"
COMMONS = "https://commons.wikimedia.org/w/api.php"
