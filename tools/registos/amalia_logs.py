#!/usr/bin/env python3
"""Mostra, pedido a pedido, os passos do RAG que o backend regista.

Lê o log do uvicorn (~/.amalia_backend.log, o que o deploy/amalia_online.sh
escreve), junta as linhas de cada pedido pelo `rid` e apresenta-as por passos:
leitura do pedido, encaminhamento, pesquisa, cartões, resposta. Só usa a
biblioteca padrão, para correr com qualquer python3 (não precisa do conda).

    deploy/amalia_logs.sh                 os últimos 5 pedidos de conversa
    deploy/amalia_logs.sh -f              em direto: cada passo aparece quando acontece
    deploy/amalia_logs.sh -l              índice: uma linha por pedido
    deploy/amalia_logs.sh --rid 2d0636    um pedido (basta o início do rid)
    deploy/amalia_logs.sh -c -n 1         o último, com prompts e respostas por extenso
    deploy/amalia_logs.sh --arranque      o que o backend carregou ao arrancar

O texto dos prompts e das respostas do modelo, e a resposta final, só existem
no log com BACKEND_RAG_DEBUG_LLM_IO=true no .env; sem isso veem-se os passos,
as decisões, os documentos e os tempos, mas dos textos só o tamanho.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
import unicodedata
from collections import OrderedDict
from datetime import datetime
from typing import Any, Iterable, Iterator

DEFAULT_LOG = os.environ.get("AMALIA_BACKEND_LOG") or os.path.expanduser("~/.amalia_backend.log")

# 2026-11-22 17:07:42[.123] INFO     [app.services.x] [rid=2d0636adee42] evento k=v ...
# `search` e não `match`: a barra de progresso dos embeddings (stderr, sem fim
# de linha) fica por vezes colada ao início da linha seguinte.
HEADER = re.compile(
    r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(?:[.,](\d{1,6}))? "
    r"(DEBUG|INFO|WARNING|ERROR|CRITICAL)\s+\[([^\]]+)\] \[rid=([^\]]*)\] (.*)$"
)
KV = re.compile(r"""([A-Za-z_][\w.]*)=('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|\S*)""")
TRUNCATION_MARK = "... truncated terminal JSON"
NOISE = re.compile(r"^(INFO|WARNING|ERROR|DEBUG|CRITICAL):\s|^Generating embeddings|^\s*\d+%\|")

# pedidos que contam como «de conversa»: os que têm pelo menos um destes eventos
CHAT_PREFIXES = ("chat.message.", "chat.image.", "chat.model.", "chat.point.", "rag.", "route.", "llm.generate")
END_EVENTS = {"chat.response", "chat.stream.error"}
# logs de antes do chat.response: o fim só se via nestes (e nem todos os caminhos os têm)
OLD_END_EVENTS = {"chat.message.finish", "chat.image.finish", "chat.model.finish"}
# pares início/fim que se juntam numa só linha, com a duração
SPAN_PREFIXES = ("llm.generate", "opensearch.", "embedding.", "warmup.")
SPAN_ENDINGS = (".finish", ".error", ".parse_error", ".empty_response")

P_APONTAR, P_LEITURA, P_ROTA, P_PESQUISA, P_CARTOES, P_RESPOSTA, P_AUTOR, P_FECHO = (
    "APONTAR", "LEITURA DO PEDIDO", "ENCAMINHAMENTO", "PESQUISA",
    "CARTÕES", "RESPOSTA", "AUTOR", "FECHO",
)
# a ordem conta: ganha o primeiro prefixo que servir
PHASE_BY_EVENT = (
    ("chat.message.start", P_LEITURA),
    ("chat.point.", P_APONTAR), ("point.", P_APONTAR),
    ("chat.focused_author.", P_AUTOR), ("chat.author_", P_AUTOR),
    ("chat.implicit_artifact.", P_LEITURA), ("chat.selected_artifact.", P_LEITURA),
    ("chat.request_reading", P_LEITURA),
    ("chat.message.route", P_ROTA), ("rag.router_decision", P_ROTA),
    ("chat.message.finish", P_FECHO), ("chat.image.finish", P_FECHO), ("chat.model.finish", P_FECHO),
    ("rag.answer", P_FECHO), ("chat.response", P_FECHO), ("chat.message.thumbnail_fetch", P_FECHO),
    ("chat.card_", P_CARTOES), ("chat.visual_pick", P_CARTOES), ("chat.visual_card_facts", P_CARTOES),
    ("rag.visible_context", P_CARTOES), ("chat.national_treasures", P_CARTOES), ("chat.set_object_counts", P_CARTOES),
    ("chat.draft", P_RESPOSTA), ("rag.final_prompt", P_RESPOSTA), ("chat.message.llm.", P_RESPOSTA),
    ("chat.image.llm.", P_RESPOSTA), ("chat.model.llm.", P_RESPOSTA), ("chat.grounding.", P_RESPOSTA),
    ("chat.stream.", P_RESPOSTA), ("chat.tour_context.llm", P_RESPOSTA),
    ("rag.", P_PESQUISA), ("chat.retrieval.", P_PESQUISA), ("multimodal.", P_PESQUISA),
    ("chat.message.retrieval.", P_PESQUISA), ("chat.image.", P_PESQUISA), ("chat.model.", P_PESQUISA),
    ("chat.catalogue_", P_PESQUISA), ("route.", P_PESQUISA), ("chat.tour_", P_PESQUISA),
    ("chat.describe_artifact.", P_PESQUISA),
)
# o nome que o backend dá a cada chamada ao modelo (campo `step`)
PHASE_BY_STEP = (
    ("leitura_do_pedido", P_LEITURA), ("peca_implicita", P_LEITURA), ("resolucao_do_seguimento", P_LEITURA),
    ("autor_da_peca_em_foco", P_AUTOR), ("ficha_do_autor", P_AUTOR), ("so_pecas_do_autor", P_AUTOR),
    ("resposta_so_do_autor", P_AUTOR),
    ("router", P_ROTA), ("leitura_de_listagem", P_ROTA),
    ("reescrita_da_pesquisa", P_PESQUISA), ("interpretacao_temporal", P_PESQUISA),
    ("intencao_visual", P_PESQUISA), ("tema_do_percurso", P_PESQUISA),
    ("escolha_visual", P_CARTOES), ("pre_selecao_dos_cartoes", P_CARTOES),
    ("rascunho", P_RESPOSTA), ("resposta", P_RESPOSTA),
)
STEP_LABELS = {
    "leitura_do_pedido": "leitura do pedido", "peca_implicita": "peça implícita",
    "resolucao_do_seguimento": "resolução do seguimento", "autor_da_peca_em_foco": "autor da peça em foco",
    "ficha_do_autor": "ficha do autor", "so_pecas_do_autor": "só peças do autor",
    "resposta_so_do_autor": "resposta só do autor", "router": "router",
    "leitura_de_listagem": "leitura de listagem", "reescrita_da_pesquisa": "reescrita da pesquisa",
    "interpretacao_temporal": "interpretação temporal", "intencao_visual": "intenção visual",
    "tema_do_percurso": "tema do percurso", "escolha_visual": "escolha visual",
    "pre_selecao_dos_cartoes": "pré-seleção dos cartões", "resposta_final": "resposta final",
    "resposta_da_pagina_de_resultados": "resposta da página de resultados",
}
# consultas ao OpenSearch que só vão buscar dados de peças já escolhidas
MINOR_OPENSEARCH = ("opensearch.fetch_",)
SEARCH_OPENSEARCH = ("opensearch.search_", "opensearch.aggregate_", "opensearch.count_")
HIDDEN_FIELDS = {"conversation_id", "query_id"}


# --------------------------------------------------------------------------- #
# Leitura do log
# --------------------------------------------------------------------------- #

def parse_kv(text: str) -> "OrderedDict[str, str]":
    fields: "OrderedDict[str, str]" = OrderedDict()
    for key, raw in KV.findall(text or ""):
        value = raw
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
            try:
                value = str(ast.literal_eval(raw))
            except (ValueError, SyntaxError):
                value = raw[1:-1]
        fields[key] = value
    return fields


def _closers(text: str) -> str | None:
    """O que falta fechar num JSON cortado; None se o corte ficou dentro de um texto."""
    stack: list[str] = []
    in_string = escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]" and stack:
            stack.pop()
    return None if in_string else "".join(reversed(stack))


def parse_json_lenient(text: str) -> tuple[Any, bool]:
    """(objeto, cortado). O backend corta os blocos grandes (BACKEND_RAG_DEBUG_MAX_CHARS)
    e o que sobra já não é JSON: lê-se até à última linha inteira e fecha-se o resto."""
    try:
        return json.loads(text), False
    except ValueError:
        pass
    lines = text.split("\n")
    while lines and (not lines[-1].strip() or lines[-1].startswith(TRUNCATION_MARK)):
        lines.pop()
    for _ in range(200):
        if not lines:
            break
        lines.pop()  # a última linha pode ter ficado a meio
        candidate = "\n".join(lines).rstrip().rstrip(",")
        closers = _closers(candidate)
        if closers is None:
            continue
        try:
            return json.loads(candidate + closers), True
        except ValueError:
            continue
    return None, True


class Event:
    __slots__ = ("ts", "has_ms", "level", "logger", "rid", "name", "fields", "raw_json", "extra", "_payload", "_cut",
                 "arrived")

    def __init__(self, ts: datetime, has_ms: bool, level: str, logger: str, rid: str, message: str) -> None:
        name, _, rest = message.partition(" ")
        self.ts, self.has_ms, self.level, self.logger, self.rid = ts, has_ms, level, logger, rid
        self.name = name.rstrip(":")
        self.fields = parse_kv(rest)
        self.raw_json: str | None = None
        self.extra: list[str] = []
        self._payload: Any = None
        self._cut: bool | None = None
        self.arrived = time.monotonic()

    @property
    def payload(self) -> Any:
        if self._cut is None:
            self._payload, self._cut = parse_json_lenient(self.raw_json) if self.raw_json else (None, False)
        return self._payload

    @property
    def cut(self) -> bool:
        _ = self.payload
        return bool(self._cut)


class Parser:
    """Recebe linhas, devolve eventos completos (um evento pode trazer um bloco JSON
    ou um traceback nas linhas seguintes, por isso só fecha quando chega o próximo)."""

    def __init__(self) -> None:
        self.current: Event | None = None
        self.block: list[str] = []
        self.open_block = False
        self.may_open = False

    def _close(self) -> Event | None:
        event, self.current = self.current, None
        if event is not None and self.block:
            event.raw_json = "\n".join(self.block)
        self.block, self.open_block, self.may_open = [], False, False
        return event

    def flush(self) -> Event | None:
        return self._close()

    def feed(self, line: str) -> Event | None:
        line = line.rstrip("\r\n")
        match = HEADER.search(line)
        if match and not self.open_block:
            done = self._close()
            stamp, frac, level, logger, rid, message = match.groups()
            ts = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")
            if frac:
                ts = ts.replace(microsecond=int(frac.ljust(6, "0")[:6]))
            self.current = Event(ts, bool(frac), level, logger, rid, message)
            self.may_open = " " not in message.strip()
            return done
        if self.current is None:
            return None
        if self.open_block:
            self.block.append(line)
            if line in ("}", "]") or line.startswith(TRUNCATION_MARK):
                self.open_block = False
            return None
        if self.may_open and not self.block and line[:1] in ("{", "["):
            self.block.append(line)
            self.open_block = line.strip() not in ("{}", "[]")
            self.may_open = False
            return None
        if line.strip() and not NOISE.search(line) and len(self.current.extra) < 80:
            if self.current.level in ("WARNING", "ERROR", "CRITICAL") or line.startswith(("Traceback", " ")):
                self.current.extra.append(line)
        return None


class Request:
    def __init__(self, rid: str) -> None:
        self.rid = rid
        self.events: list[Event] = []
        self.touched = time.monotonic()
        self.printed = False
        self.announced = False
        # modo em direto: o que já saiu para o ecrã
        self.live_seen: set[int] = set()
        self.live_waiting: set[int] = set()
        self.live_phase: str | None = None
        self.live_number = 0

    def add(self, event: Event) -> None:
        self.events.append(event)
        self.touched = time.monotonic()

    # ---- o que identifica o pedido ---- #
    def first(self, *names: str) -> Event | None:
        for event in self.events:
            if event.name in names:
                return event
        return None

    def last(self, *names: str) -> Event | None:
        for event in reversed(self.events):
            if event.name in names:
                return event
        return None

    @property
    def is_chat(self) -> bool:
        return any(event.name.startswith(CHAT_PREFIXES) for event in self.events)

    @property
    def ended(self) -> bool:
        if any(event.name in END_EVENTS for event in self.events):
            return True
        return not self.expects_response and any(event.name in OLD_END_EVENTS for event in self.events)

    @property
    def expects_response(self) -> bool:
        """O backend que escreve milésimos na hora é o que fecha todos os pedidos
        com chat.response; num log mais antigo a falta dele não quer dizer nada."""
        return any(event.has_ms for event in self.events)

    @property
    def start(self) -> datetime:
        return self.events[0].ts

    def field(self, key: str) -> str | None:
        for event in self.events:
            if key in event.fields:
                return event.fields[key]
        for event in self.events:
            if event.raw_json and isinstance(event.payload, dict) and event.payload.get(key):
                return str(event.payload[key])
        return None

    @property
    def question(self) -> str | None:
        decision = self.first("rag.router_decision")
        if decision is not None and isinstance(decision.payload, dict):
            message = (decision.payload.get("input") or {}).get("message")
            if message:
                return text_of(message)
        answer = self.last("rag.answer")
        if answer is not None and isinstance(answer.payload, dict) and answer.payload.get("message"):
            return text_of(answer.payload["message"])
        response = self.last("chat.response")
        if response is not None and response.fields.get("message"):
            return response.fields["message"]
        reading = self.first("chat.request_reading")
        if reading is not None and reading.fields.get("request"):
            return reading.fields["request"]
        return None

    @property
    def subject(self) -> str:
        """O que pôr no lugar da pergunta quando o pedido não a traz no log."""
        route = self.last("chat.message.route")
        piece = (route.fields.get("selected_artifact_id") if route is not None else None) or ""
        if self.first("chat.point.start"):
            decided = self.last("chat.point.decided")
            chosen = decided.fields.get("chosen") if decided is not None else None
            return f"(apontar → {chosen or piece or 'sem peça escolhida'})"
        if piece:
            return f"(peça {piece})"
        return f"({self.kind})"

    @property
    def kind(self) -> str:
        if self.first("chat.point.start"):
            return "apontar"
        if self.first("chat.image.start"):
            return "imagem"
        if self.first("chat.model.start"):
            return "modelo 3D"
        return "texto"

    @property
    def answer(self) -> str | None:
        event = self.last("rag.answer")
        if event is not None and isinstance(event.payload, dict):
            return text_of(event.payload.get("reply"))
        return None

    @property
    def finish(self) -> Event | None:
        return self.last("chat.message.finish", "chat.image.finish", "chat.model.finish")

    @property
    def problems(self) -> list[Event]:
        return [event for event in self.events if event.level in ("WARNING", "ERROR", "CRITICAL")]

    @property
    def status(self) -> str:
        finish = self.finish
        if finish is not None and finish.fields.get("status", "ok") != "ok":
            return finish.fields["status"]
        if finish is not None or self.last("chat.response") is not None:
            return "ok"
        if any(event.level in ("ERROR", "CRITICAL") for event in self.events):
            return "erro"
        return "sem fim" if self.expects_response else "—"

    @property
    def duration_ms(self) -> float | None:
        for event in (self.last("chat.response"), self.finish):
            if event is not None and event.fields.get("duration_ms"):
                return to_float(event.fields["duration_ms"])
        return None

    @property
    def route(self) -> str:
        route = self.last("chat.message.route")
        if route is None:
            return self.kind
        parts = [route.fields.get("mode"), route.fields.get("intent")]
        label = "/".join(part for part in parts if part)
        fast_path = route.fields.get("fast_path")
        if fast_path and fast_path != route.fields.get("intent"):
            label += f" ({fast_path})"
        return label or self.kind


def read_events(lines: Iterable[str]) -> Iterator[Event]:
    parser = Parser()
    for line in lines:
        event = parser.feed(line)
        if event is not None:
            yield event
    event = parser.flush()
    if event is not None:
        yield event


# --------------------------------------------------------------------------- #
# Formatação
# --------------------------------------------------------------------------- #

class Style:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled and text else text

    def bold(self, text: str) -> str: return self._wrap("1", text)
    def dim(self, text: str) -> str: return self._wrap("2", text)
    def red(self, text: str) -> str: return self._wrap("91", text)
    def green(self, text: str) -> str: return self._wrap("92", text)
    def yellow(self, text: str) -> str: return self._wrap("93", text)
    def blue(self, text: str) -> str: return self._wrap("94", text)
    def magenta(self, text: str) -> str: return self._wrap("95", text)
    def cyan(self, text: str) -> str: return self._wrap("96", text)


def to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_ms(value: Any) -> str:
    ms = to_float(value)
    if ms is None:
        return "?"
    if ms < 1000:
        return f"{ms:.0f} ms"
    return f"{ms / 1000:.2f} s".replace(".", ",")


def text_of(value: Any) -> str:
    """Um texto do log; os compridos vêm como {"_type": "truncated_string", ...}."""
    if value is None:
        return ""
    if isinstance(value, dict) and value.get("_type") == "truncated_string":
        return f"{value.get('preview', '')} […cortado no log: {value.get('length')} caracteres ao todo]"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


ANSI = re.compile(r"\033\[[0-9;]*m")


def visible_len(text: str) -> int:
    return len(ANSI.sub("", text))


def wrap_ansi(line: str, width: int, indent: str) -> list[str]:
    """Parte uma linha já colorida pelos espaços; as continuações levam `indent`."""
    if visible_len(line) <= width:
        return [line]
    lead = len(line) - len(line.lstrip(" "))
    rows: list[str] = []
    current = line[:lead]
    used = lead
    for word in line[lead:].split(" "):
        size = visible_len(word)
        if used > len(indent) and used + 1 + size > width and current.strip():
            rows.append(current)
            current, used = indent + word, len(indent) + size
        else:
            joiner = "" if not current.strip() else " "
            current += joiner + word
            used += len(joiner) + size
    rows.append(current)
    return rows


def short_model(model: str | None) -> str:
    return (model or "?").rsplit("/", 1)[-1]


def short(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def yes_no(value: Any) -> str:
    return "sim" if str(value).lower() in ("true", "1", "yes") else "não"


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def fold(text: str) -> str:
    plain = unicodedata.normalize("NFKD", text or "")
    return "".join(char for char in plain if not unicodedata.combining(char)).lower()


class Item:
    """Uma linha do pedido: um evento, ou um par início/fim já junto."""

    def __init__(self, event: Event, start: Event | None = None, open_span: bool = False) -> None:
        self.event = event
        self.start = start
        self.open_span = open_span
        self.io: Event | None = None  # rag.llm_call desta chamada
        self.fallback = False
        self.fields: "OrderedDict[str, str]" = OrderedDict()
        if start is not None:
            self.fields.update(start.fields)
        self.fields.update(event.fields)

    @property
    def name(self) -> str:
        name = self.event.name
        if self.start is not None or self.open_span:
            for ending in (".start",) + SPAN_ENDINGS:
                if name.endswith(ending):
                    return name[: -len(ending)]
        return name

    @property
    def is_span(self) -> bool:
        return self.start is not None or self.open_span

    @property
    def failed(self) -> bool:
        return self.open_span or (self.start is not None and not self.event.name.endswith(".finish"))

    @property
    def ts(self) -> datetime:
        return (self.start or self.event).ts

    @property
    def duration_ms(self) -> float | None:
        return to_float(self.fields.get("duration_ms"))


def build_items(events: list[Event]) -> list[Item]:
    items: list[Item] = []
    waiting: dict[str, list[Item]] = {}  # base → inícios à espera do fim (por ordem)
    for event in events:
        name = event.name
        if name == "rag.llm_call":
            payload = event.payload if isinstance(event.payload, dict) else {}
            for item in reversed(items):
                if item.name == "llm.generate" and item.io is None and not item.open_span:
                    if item.fields.get("model") in (None, payload.get("model")):
                        item.io = event
                        break
            continue
        if name == "llm.generate.fallback":
            for item in reversed(waiting.get("llm.generate", [])):
                item.fallback = True
                break
            continue
        if name.startswith(SPAN_PREFIXES):
            if name.endswith(".start"):
                item = Item(event, open_span=True)
                waiting.setdefault(name[: -len(".start")], []).append(item)
                items.append(item)
                continue
            ending = next((e for e in SPAN_ENDINGS if name.endswith(e)), None)
            if ending:
                queue = waiting.get(name[: -len(ending)]) or []
                opened = _pick_open(queue, event)
                if opened is not None:
                    queue.remove(opened)
                    items.remove(opened)
                    merged = Item(event, start=opened.event)
                    merged.fallback = opened.fallback
                    items.append(merged)
                    continue
        items.append(Item(event))
    return items


def _pick_open(queue: list[Item], finish: Event) -> Item | None:
    """Chamadas em paralelo acabam fora de ordem: escolhe o início do mesmo
    passo/modelo quando o fim o diz, senão o mais antigo."""
    if not queue:
        return None
    for key in ("step", "model"):
        wanted = finish.fields.get(key)
        if wanted:
            same = [item for item in queue if item.event.fields.get(key) == wanted]
            if same:
                return same[0]
    return queue[0]


def phase_of(item: Item) -> str | None:
    if item.name == "llm.generate":
        step = item.fields.get("step") or ""
        for prefix, phase in PHASE_BY_STEP:
            if step.startswith(prefix):
                return phase
        # logs sem o nome do passo: a que escreve texto corrido é a resposta
        if not step and (item.fields.get("stream") == "true" or item.fields.get("response_format") == "text"):
            return P_RESPOSTA
        return None
    if item.is_span:
        return None
    for prefix, phase in PHASE_BY_EVENT:
        if item.name.startswith(prefix):
            return phase
    return None


def assign_phases(items: list[Item]) -> list[str]:
    """As chamadas sem fase própria (OpenSearch, embeddings, modelo sem nome) ficam
    no passo do evento que vem a seguir: é esse que relata o que elas fizeram."""
    phases: list[str | None] = [phase_of(item) for item in items]
    following: str | None = None
    for index in range(len(items) - 1, -1, -1):
        if phases[index] is None:
            # uma pesquisa cujo relato não ficou no log não é parte da resposta
            searching = items[index].name.startswith(SEARCH_OPENSEARCH)
            phases[index] = P_PESQUISA if searching and following in (P_RESPOSTA, P_FECHO, None) else following
        else:
            following = phases[index]
    previous = P_LEITURA
    for index, phase in enumerate(phases):
        if phase is None:
            phases[index] = previous
        else:
            previous = phase
    return [str(phase) for phase in phases]


class Renderer:
    def __init__(self, style: Style, *, full: bool, width: int) -> None:
        self.s = style
        self.full = full
        self.width = max(60, min(width, 160))

    def status_color(self, status: str) -> Any:
        return {"ok": self.s.green, "sem fim": self.s.yellow, "—": self.s.dim}.get(status, self.s.red)

    # ---- blocos de texto ---- #
    def wrap(self, text: str, indent: str, limit_lines: int | None = None) -> list[str]:
        out: list[str] = []
        for paragraph in (text or "").splitlines() or [""]:
            if not paragraph.strip():
                out.append(indent.rstrip())
                continue
            out.extend(
                textwrap.wrap(paragraph, width=self.width - 12, initial_indent=indent, subsequent_indent=indent,
                              break_long_words=False, break_on_hyphens=False) or [indent]
            )
        if limit_lines is not None and len(out) > limit_lines:
            hidden = len(out) - limit_lines
            out = out[:limit_lines] + [indent + self.s.dim(f"… mais {hidden} linhas (-c mostra tudo)")]
        return out

    def kv(self, fields: "dict[str, str]", skip: Iterable[str] = ()) -> str:
        skipped = set(skip) | HIDDEN_FIELDS
        return "  ".join(
            f"{self.s.dim(key + '=')}{value}" for key, value in fields.items() if key not in skipped and value != ""
        )

    def offset(self, request_start: datetime, item: Item, precise: bool) -> str:
        delta = (item.ts - request_start).total_seconds()
        text = f"+{delta:6.2f}s".replace(".", ",") if precise else f"+{delta:3.0f}s"
        return self.s.dim(text)

    # ---- um pedido inteiro ---- #
    def request(self, request: Request) -> list[str]:
        items = build_items(request.events)
        phases = assign_phases(items)
        precise = any(event.has_ms for event in request.events)
        out = self.header(request)

        current: str | None = None
        number = 0
        index = 0
        while index < len(items):
            item, phase = items[index], phases[index]
            if item.name.startswith("http.request."):
                index += 1
                continue
            if phase != current:
                current = phase
                number += 1
                out.extend(self.heading(number, phase))
            # várias idas seguidas ao OpenSearch só para ir buscar dados: uma linha
            if not self.full and self._minor(item):
                group = [item]
                while index + len(group) < len(items) and self._minor(items[index + len(group)]) \
                        and phases[index + len(group)] == phase:
                    group.append(items[index + len(group)])
                if len(group) > 1:
                    out.extend(self._lines(request, item, precise, self._minor_group(group)))
                    index += len(group)
                    continue
            out.extend(self._lines(request, item, precise, self.item(item)))
            index += 1

        out.append("")
        out.extend(self.totals(request, items))
        return out

    def heading(self, number: int, phase: str) -> list[str]:
        text = f" {number} · {phase} "
        return ["", self.s.bold(self.s.blue(text)) + self.s.blue("─" * max(0, self.width - len(text)))]

    def header(self, request: Request) -> list[str]:
        s = self.s
        out: list[str] = []
        rule = "═" * self.width
        status = request.status
        color = self.status_color(status)
        question = request.question
        title = f"«{question}»" if question else request.subject
        out.append(s.cyan(rule))
        out.extend(s.bold(line) for line in self.wrap(f"{request.start:%d/%m %H:%M:%S}  {title}", " "))
        meta = [part for part in (request.field("museum_slug"), request.field("language"), request.route) if part]
        meta.append(color(status))
        if request.duration_ms is not None:
            meta.append(fmt_ms(request.duration_ms))
        finish = request.finish
        if finish is not None and finish.fields.get("shown_artifacts"):
            shown = finish.fields["shown_artifacts"]
            meta.append(f"{shown} peça mostrada" if shown == "1" else f"{shown} peças mostradas")
        out.append(" " + " · ".join(meta) + s.dim(f"   rid {request.rid}"))
        http = request.first("http.request.start")
        ids = [f"conversa {request.field('conversation_id') or '?'}", f"query {request.field('query_id') or '?'}"]
        if http is not None:
            ids.append(f"{http.fields.get('method', '')} {http.fields.get('path', '')}")
            if http.fields.get("client"):
                ids.append(http.fields["client"])
        out.extend(s.dim(row) for row in wrap_ansi(" " + " · ".join(ids), self.width, " "))
        out.append(s.cyan(rule))
        return out

    # ---- em direto: só o que ainda não saiu ---- #
    def live(self, request: Request, now: float, final: bool) -> list[str]:
        """Os passos novos de um pedido que ainda decorre. Uma chamada (modelo,
        OpenSearch) sai quando acaba, com a duração; se demorar, sai antes um aviso
        de que está em curso. As chamadas sem passo próprio esperam pelo evento
        que as relata (ou 1 s), que é o que diz a que passo pertencem."""
        s = self.s
        items = build_items(request.events)
        phases = assign_phases(items)
        precise = any(event.has_ms for event in request.events)
        quiet = now - request.touched
        out: list[str] = []
        if not request.announced:
            request.announced = True
            http = request.first("http.request.start")
            path = f" · {http.fields.get('path', '')}" if http is not None else ""
            text = f"═══ {request.start:%H:%M:%S} pedido novo ({request.kind}) · rid {request.rid}{path} "
            out.extend(["", s.cyan(s.bold(text)) + s.cyan("═" * max(0, self.width - len(text)))])
        for index, item in enumerate(items):
            key = id(item.event)
            if key in request.live_seen or item.name.startswith("http.request."):
                continue
            if item.open_span and not final:
                if key not in request.live_waiting and now - item.event.arrived > 0.8:
                    request.live_waiting.add(key)
                    what = item.name
                    if item.name == "llm.generate":
                        step = item.fields.get("step") or ""
                        label = STEP_LABELS.get(step, step.replace("_", " "))
                        what = f"modelo{f' «{label}»' if label else ''} {short_model(item.fields.get('model'))}"
                    out.extend(self._lines(request, item, precise, [s.dim(f"⏳ {what} em curso…")]))
                continue
            if phase_of(item) is None and not final and quiet < 1.0 \
                    and not any(phase_of(later) is not None for later in items[index + 1:]):
                continue
            request.live_seen.add(key)
            if phases[index] != request.live_phase:
                request.live_phase = phases[index]
                request.live_number += 1
                out.extend(self.heading(request.live_number, phases[index]))
            out.extend(self._lines(request, item, precise, self.item(item)))
        if final:
            status = request.status
            question = request.question
            title = f"«{short(question, 80)}»" if question else request.subject
            closing = [title, self.status_color(status)(status)]
            if request.duration_ms is not None:
                closing.append(fmt_ms(request.duration_ms))
            out.append("")
            out.append(s.bold(" FIM ") + " · ".join(closing) + s.dim(f"   rid {request.rid}"))
            out.extend(self.totals(request, items)[1:])
            out.append(s.cyan("═" * self.width))
        return out

    def _minor(self, item: Item) -> bool:
        return item.is_span and item.name.startswith(MINOR_OPENSEARCH) and not item.failed

    def _minor_group(self, group: list[Item]) -> list[str]:
        counts: "OrderedDict[str, int]" = OrderedDict()
        for item in group:
            label = item.name[len("opensearch.fetch_"):]
            counts[label] = counts.get(label, 0) + 1
        names = ", ".join(f"{label}{f' ×{n}' if n > 1 else ''}" for label, n in counts.items())
        total = sum(item.duration_ms or 0 for item in group)
        return [f"{self.s.dim('OpenSearch')} {len(group)} consultas de dados ({names}) · {fmt_ms(total)}"]

    def _lines(self, request: Request, item: Item, precise: bool, lines: list[str]) -> list[str]:
        if not lines:
            return []
        prefix = f"  {self.offset(request.start, item, precise)}  "
        pad = " " * (len(f"  +{0:6.2f}s  ") if precise else len("  +  0s  "))
        out: list[str] = []
        for position, line in enumerate(lines):
            lead = " " * (len(line) - len(line.lstrip(" ")))
            rows = wrap_ansi(line, self.width - len(pad), lead + "    ")
            out.extend(((prefix if position == 0 and row is rows[0] else pad) + row) for row in rows)
        return out

    # ---- uma linha ---- #
    def item(self, item: Item) -> list[str]:
        s = self.s
        name, f = item.name, item.fields
        handler = getattr(self, "_ev_" + re.sub(r"\W", "_", name), None)
        if name == "llm.generate":
            lines = self._llm(item)
        elif name.startswith("opensearch."):
            lines = self._opensearch(item)
        elif name.startswith("embedding."):
            lines = self._embedding(item)
        elif handler is not None:
            lines = handler(item)
        else:
            label = s.dim(name) if item.event.level == "INFO" else name
            lines = [f"{label}  {self.kv(f)}".rstrip()]
            if item.event.raw_json and self.full:
                lines.extend(self._json_block(item.event))
        if item.event.level in ("WARNING", "ERROR", "CRITICAL") or item.failed:
            paint = s.yellow if item.event.level == "WARNING" and not item.failed else s.red
            mark = "⚠ " if paint == s.yellow else "✖ "
            lines = [paint(mark) + lines[0]] + lines[1:]
            if f.get("error") and "error" not in lines[0]:
                lines.extend(paint(line) for line in self.wrap(f"erro: {f['error']}", "    "))
            extra = item.event.extra if self.full else item.event.extra[-4:]
            lines.extend(s.dim("    " + line) for line in extra)
        if item.open_span:
            lines[0] += s.red("  (começou e não há registo do fim)")
        return lines

    def _json_block(self, event: Event, value: Any = None) -> list[str]:
        payload = event.payload if value is None else value
        if payload is None:
            return [self.s.dim("    " + line) for line in (event.raw_json or "").split("\n")]
        rendered = json.dumps(payload, ensure_ascii=False, indent=2)
        lines = [self.s.dim("    " + line) for line in rendered.split("\n")]
        if event.cut:
            lines.append(self.s.yellow("    (bloco cortado no log: BACKEND_RAG_DEBUG_MAX_CHARS)"))
        return lines

    def _llm(self, item: Item) -> list[str]:
        s, f = self.s, item.fields
        step = f.get("step")
        label = STEP_LABELS.get(step or "", (step or "").replace("_", " "))
        head = [s.magenta("modelo") + (f" «{label}»" if label else ""), short_model(f.get("model"))]
        if f.get("response_format") == "json_object":
            head.append("JSON")
        if f.get("prompt_tokens") or f.get("completion_tokens"):
            head.append(f"{f.get('prompt_tokens', '?')}→{f.get('completion_tokens', '?')} tokens")
        else:
            head.append(f"prompt {f.get('prompt_chars', '?')} car.")
        if item.duration_ms is not None:
            head.append(s.bold(fmt_ms(item.duration_ms)))
        if f.get("first_delta_ms"):
            head.append(f"1.ª palavra aos {fmt_ms(f['first_delta_ms'])}")
        if f.get("finish_reason") and f["finish_reason"] != "stop":
            head.append(s.yellow(f"parou por {f['finish_reason']}"))
        if item.fallback or f.get("endpoint") == "fallback":
            head.append(s.yellow("pela API remota (fallback)"))
        lines = [" · ".join(head)]
        io = item.io.payload if item.io is not None and isinstance(item.io.payload, dict) else None
        if io is None:
            return lines
        if self.full:
            if io.get("system_prompt"):
                lines.append(s.dim("    ┌ instruções de sistema"))
                lines.extend(self.wrap(text_of(io["system_prompt"]), "    │ "))
            lines.append(s.dim("    ┌ prompt"))
            lines.extend(self.wrap(text_of(io.get("prompt")), "    │ "))
            lines.append(s.dim("    └ resposta"))
            lines.extend(s.green(line) for line in self.wrap(text_of(io.get("response")), "      "))
        elif io.get("response") is not None:
            lines.extend(s.green(line) for line in self.wrap("↳ " + text_of(io.get("response")), "    ", limit_lines=6))
        return lines

    def _opensearch(self, item: Item) -> list[str]:
        s, f = self.s, item.fields
        head = [s.dim("OpenSearch") + " " + item.name[len("opensearch."):]]
        if item.duration_ms is not None:
            head.append(s.bold(fmt_ms(item.duration_ms)))
        if f.get("result_count") is not None:
            total = f" de {f['total']}" if f.get("total") else ""
            head.append(f"{f['result_count']} resultados{total}")
        shown = {"duration_ms", "result_count", "total", "museum_slug", "embedding_dim"}
        rest = self.kv(f, skip=shown)
        return [" · ".join(head) + (f"  {rest}" if rest else "")]

    def _embedding(self, item: Item) -> list[str]:
        s, f = self.s, item.fields
        head = [s.dim("embedding") + " " + item.name[len("embedding."):]]
        if f.get("model"):
            head.append(short_model(f["model"]))
        if item.duration_ms is not None:
            head.append(s.bold(fmt_ms(item.duration_ms)))
        rest = self.kv(f, skip={"duration_ms", "model"})
        return [" · ".join(head) + (f"  {rest}" if rest else "")]

    def _ev_chat_message_start(self, item: Item) -> list[str]:
        f = item.fields
        return [f"pedido recebido · museu {f.get('museum_slug', '?')} · língua {f.get('language', '?')} · "
                f"{f.get('message_chars', '?')} caracteres · formato {f.get('response_format', '?')}"]

    def _ev_chat_request_reading(self, item: Item) -> list[str]:
        f = item.fields
        line = (f"pedido lido: «{f.get('request', '')}» · seguimento: {yes_no(f.get('follow_up'))} · "
                f"usado na pesquisa: {yes_no(f.get('used_request'))}")
        rest = self.kv(f, skip={"request", "follow_up", "used_request", "language"})
        return [line] + ([f"  {rest}"] if rest else [])

    def _ev_chat_message_route(self, item: Item) -> list[str]:
        f = item.fields
        head = f"rota: modo {self.s.bold(f.get('mode', '?'))} · intenção {self.s.bold(f.get('intent', '?'))}"
        if f.get("fast_path"):
            head += f" · atalho {self.s.bold(f['fast_path'])}"
        rest = self.kv(f, skip={"mode", "intent", "fast_path", "search_museum_id"})
        return [head + (f"  {rest}" if rest else "")]

    def _ev_rag_router_decision(self, item: Item) -> list[str]:
        payload = item.event.payload
        if not isinstance(payload, dict):
            return [self.s.dim("rag.router_decision") + " (bloco ilegível)"]
        decision = payload.get("router_decision") or {}
        policy = payload.get("context_policy") or {}
        lines = [f"mensagem: «{text_of((payload.get('input') or {}).get('message'))}»"]
        lines.append(
            f"decisão: intenção {decision.get('intent')} · modo {decision.get('mode')} · "
            f"pesquisa: {yes_no(decision.get('needs_retrieval'))} · motivo: {decision.get('reason') or '—'}"
        )
        if decision.get("rewritten_query"):
            lines.append(f"pergunta para a pesquisa: «{text_of(decision['rewritten_query'])}»")
        lines.append(
            f"contexto: seguimento {yes_no(policy.get('is_follow_up'))} · histórico na pesquisa "
            f"{yes_no(policy.get('use_history_for_query'))} · histórico na resposta "
            f"{yes_no(policy.get('use_history_for_answer'))} · herda filtros {yes_no(policy.get('carry_filters'))}"
            + (f" · {policy.get('reason')}" if policy.get("reason") else "")
        )
        for key, label in (("filters_delta", "filtros novos"), ("sort_delta", "ordenação nova")):
            if decision.get(key):
                lines.append(f"{label}: {compact(decision[key])}")
        selected = payload.get("selected_artifact")
        if isinstance(selected, dict) and selected:
            lines.append(
                f"peça selecionada: {selected.get('inventory_number') or ''} {selected.get('title') or ''}"
                f" ({payload.get('selected_context_mode')})"
            )
        scope = payload.get("search_scope") or {}
        if scope.get("is_cross_museum"):
            lines.append("âmbito: todos os museus")
        if self.full:
            lines.extend(self._json_block(item.event))
        return lines

    def _ev_rag_text_query_plan(self, item: Item) -> list[str]:
        plan = item.event.payload
        if not isinstance(plan, dict):
            return [self.s.dim("rag.text_query_plan") + " (bloco ilegível)"]
        lines = [f"plano · texto (BM25): «{text_of(plan.get('lexical_query'))}»"]
        if plan.get("embedding_query") != plan.get("lexical_query"):
            lines.append(f"       vetor: «{text_of(plan.get('embedding_query'))}»")
        if plan.get("raw_query") not in (plan.get("lexical_query"), None):
            lines.append(f"       pedido original: «{text_of(plan.get('raw_query'))}»")
        details = [f"filtros {compact(plan.get('filters') or {})}", f"ordenação {compact(plan.get('sort') or {})}"]
        if plan.get("temporal_query"):
            details.append(f"tempo {compact(plan['temporal_query'])}")
        if plan.get("retrieval_boosts"):
            details.append(f"reforços {compact(plan['retrieval_boosts'])}")
        if plan.get("tour_scope"):
            details.append(f"visita {compact(plan['tour_scope'])}")
        lines.append("       " + " · ".join(details))
        lines.append(
            f"       janela {plan.get('retrieval_window_size')} · página {plan.get('retrieval_page_size')} · "
            f"reescrita por {plan.get('query_rewrite_source')} · vetor "
            f"{'sim' if plan.get('query_embedding') else 'não'}"
        )
        if self.full:
            lines.extend(self._json_block(item.event))
        return lines

    def _docs(self, docs: Any, limit: int) -> list[str]:
        lines: list[str] = []
        if not isinstance(docs, list):
            return lines
        for position, doc in enumerate(docs[:limit], start=1):
            if not isinstance(doc, dict):
                continue
            scores = []
            if doc.get("score") is not None:
                scores.append(f"score {to_float(doc['score']) or 0:.3f}")
            if doc.get("rerank_score") is not None:
                scores.append(f"rerank {to_float(doc['rerank_score']) or 0:.3f}")
            line = (f"  {doc.get('rank') or position:>2}. {str(doc.get('inventory_number') or '—'):<14} "
                    f"{short(text_of(doc.get('title')) or '(sem título)', 62)}")
            tail = []
            if doc.get("creator"):
                tail.append(text_of(doc["creator"]))
            if doc.get("date_or_period"):
                tail.append(text_of(doc["date_or_period"]))
            if doc.get("category"):
                tail.append(text_of(doc["category"]))
            tail.extend(scores)
            if doc.get("in_tour"):
                tail.append("na visita")
            lines.append(line + (self.s.dim("  " + " · ".join(tail)) if tail else ""))
            if self.full and doc.get("snippet"):
                lines.extend(self.s.dim(row) for row in self.wrap(text_of(doc["snippet"]), "        ", limit_lines=3))
        if len(docs) > limit:
            lines.append(self.s.dim(f"      … mais {len(docs) - limit}"))
        return lines

    def _ev_chat_message_retrieval_finish(self, item: Item) -> list[str]:
        f = item.fields
        return [f"recuperação {f.get('retrieval_kind', '')} · {f.get('retrieved_docs_count', '?')} documentos · "
                f"{f.get('image_matches', '?')} imagens · {self.s.bold(fmt_ms(f.get('duration_ms')))}"]

    def _ev_rag_retrieval_result(self, item: Item) -> list[str]:
        payload = item.event.payload
        if not isinstance(payload, dict):
            return [self.s.dim("rag.retrieval_result") + " (bloco ilegível)"]
        request = payload.get("retrieval_request") or {}
        docs = payload.get("retrieved_docs") or []
        lines = [
            f"resultado: {payload.get('retrieved_docs_count')} documentos ({request.get('kind') or '?'}) · "
            f"contexto {payload.get('context_chars')} car. · primeiros {len(docs)}:"
        ]
        lines.extend(self._docs(docs, 8 if not self.full else 50))
        for key, label in (("effective_filters", "filtros aplicados"), ("effective_sort", "ordenação aplicada")):
            if payload.get(key):
                lines.append(f"{label}: {compact(payload[key])}")
        if self.full and request.get("opensearch_query"):
            lines.append(self.s.dim("    corpo enviado ao OpenSearch:"))
            lines.extend(self._json_block(item.event, request["opensearch_query"]))
        elif item.event.cut:
            lines.append(self.s.yellow("    (bloco cortado no log)"))
        return lines

    def _ev_multimodal_router_decision(self, item: Item) -> list[str]:
        f = item.fields
        return [f"ramo visual: {self.s.bold(f.get('decision', '?'))} (usa imagens: {yes_no(f.get('use_visual'))}) · "
                f"regra {f.get('rule', '?')} · confiança {f.get('confidence', '?')} · {f.get('reason', '')}"]

    def _ev_multimodal_router_model_decision(self, item: Item) -> list[str]:
        f = item.fields
        return [f"ramo visual, resposta do modelo: {self.s.bold(f.get('answer', '?'))} · {fmt_ms(f.get('elapsed_ms'))}"]

    def _ev_chat_card_preselection(self, item: Item) -> list[str]:
        f = item.fields
        head = (f"pré-seleção dos cartões: {f.get('shown_before', '?')} → {self.s.bold(f.get('shown_after', '?'))} · "
                f"regra {f.get('rule', '?')} · {fmt_ms(f.get('elapsed_ms'))}")
        rest = self.kv(f, skip={"shown_before", "shown_after", "rule", "elapsed_ms"})
        return [head + (f"  {rest}" if rest else "")]

    def _ev_rag_visible_context(self, item: Item) -> list[str]:
        payload = item.event.payload
        if not isinstance(payload, dict):
            return [self.s.dim("rag.visible_context") + " (bloco ilegível)"]
        cards = payload.get("artifact_results") or []
        results = payload.get("results") or {}
        head = f"cartões que o visitante vê: {results.get('shown_artifacts', len(cards))}"
        if results:
            head += (f" · imagens {results.get('shown_images')} · total {results.get('total')} · "
                     f"página {results.get('page')} · há mais: {yes_no(results.get('has_more'))}")
        if payload.get("visible_context_chars") is not None:
            head += f" · contexto para o modelo {payload['visible_context_chars']} car."
        lines = [head]
        lines.extend(self._docs(cards, 12 if not self.full else 60))
        if item.event.cut:
            lines.append(self.s.dim("    (bloco cortado no log: a lista pode estar incompleta)"))
        return lines

    def _ev_chat_draft(self, item: Item) -> list[str]:
        f = item.fields
        return [f"rascunho: {f.get('cards', '?')} cartões · {f.get('sentences_from_model', '?')} frases do modelo · "
                f"abertura {f.get('opening', '?')} · {fmt_ms(f.get('elapsed_ms'))}"]

    def _ev_rag_final_prompt(self, item: Item) -> list[str]:
        payload = item.event.payload
        if not isinstance(payload, dict):
            return [self.s.dim("rag.final_prompt") + " (bloco ilegível)"]
        parts = [f"prompt final: {payload.get('final_prompt_chars')} car."]
        if payload.get("retrieval_context_chars") is not None:
            parts.append(f"contexto {payload['retrieval_context_chars']} car.")
        if payload.get("retry_context_result_count") is not None:
            parts.append(f"{payload['retry_context_result_count']} peças no contexto")
        if "use_history_for_answer" in payload:
            parts.append(f"com histórico: {yes_no(payload['use_history_for_answer'])}")
        if payload.get("intent"):
            parts.append(f"intenção {payload['intent']}")
        return [" · ".join(parts)]

    def _ev_chat_message_llm_finish(self, item: Item) -> list[str]:
        return [f"resposta do modelo recebida · {fmt_ms(item.fields.get('duration_ms'))}"]

    def _ev_chat_draft_rewritten(self, item: Item) -> list[str]:
        f = item.fields
        return [f"reescrita do rascunho: {f.get('rewritten', '?')} de {f.get('lines', '?')} linhas reescritas"]

    def _ev_chat_message_finish(self, item: Item) -> list[str]:
        f = item.fields
        head = f"fim · estado {f.get('status', '?')} · {self.s.bold(fmt_ms(f.get('duration_ms')))}"
        rest = self.kv(f, skip={"status", "duration_ms"})
        return [head + (f"  {rest}" if rest else "")]

    def _ev_chat_response(self, item: Item) -> list[str]:
        f = item.fields
        return [f"resposta entregue · {f.get('reply_chars', '?')} caracteres · {f.get('cards', '?')} cartões · "
                f"{f.get('results_total', '?')} resultados ao todo · {self.s.bold(fmt_ms(f.get('duration_ms')))}"]

    _ev_chat_image_finish = _ev_chat_message_finish
    _ev_chat_model_finish = _ev_chat_message_finish

    def _ev_rag_answer(self, item: Item) -> list[str]:
        payload = item.event.payload
        if not isinstance(payload, dict):
            return [self.s.dim("rag.answer") + " (bloco ilegível)"]
        lines = [self.s.bold("resposta enviada ao visitante:")]
        lines.extend(self.s.green(line) for line in self.wrap(text_of(payload.get("reply")), "  ",
                                                             limit_lines=None if self.full else 30))
        cards = payload.get("cards") or []
        extras = [f"{len(cards)} cartões"]
        if payload.get("results_total"):
            extras.append(f"{payload['results_total']} resultados ao todo")
        if payload.get("author"):
            extras.append(f"ficha do autor: {payload['author']}")
        if payload.get("route"):
            extras.append(f"percurso «{payload['route'].get('title')}» com {payload['route'].get('stops')} paragens")
        if payload.get("navigation_targets"):
            extras.append(f"{payload['navigation_targets']} destinos na visita")
        if payload.get("selected_artifact_status"):
            extras.append(f"peça selecionada: {payload['selected_artifact_status']}")
        lines.append(" · ".join(extras))
        lines.extend(self._docs(cards, 12 if not self.full else 60))
        return lines

    # ---- contas ---- #
    def totals(self, request: Request, items: list[Item]) -> list[str]:
        s = self.s
        buckets: "OrderedDict[str, list[float]]" = OrderedDict()
        for item in items:
            if not item.is_span or item.duration_ms is None:
                continue
            if item.name == "llm.generate":
                key = f"modelo {short_model(item.fields.get('model'))}"
            elif item.name.startswith("opensearch."):
                key = "OpenSearch"
            elif item.name.startswith("embedding."):
                key = "embeddings"
            else:
                continue
            buckets.setdefault(key, []).append(item.duration_ms)
        parts = [f"{key}: {len(values)} × = {fmt_ms(sum(values))}" for key, values in buckets.items()]
        lines = [s.bold(" TEMPOS ") + s.dim("(somas; as chamadas em paralelo contam todas)")]
        if parts:
            lines.append("  " + " · ".join(parts))
        if request.duration_ms is not None:
            lines.append(f"  total do pedido: {s.bold(fmt_ms(request.duration_ms))}")
        problems = request.problems
        if problems:
            lines.append(s.yellow(f"  {len(problems)} aviso(s)/erro(s): ") + ", ".join(
                OrderedDict.fromkeys(event.name for event in problems)))
        if not any(item.io is not None for item in items) and any(item.name == "llm.generate" for item in items):
            lines.append(s.dim("  sem o texto dos prompts e das respostas: BACKEND_RAG_DEBUG_LLM_IO=true no .env"))
        return lines

    # ---- índice ---- #
    def index_line(self, request: Request) -> str:
        s = self.s
        status = request.status
        color = self.status_color(status)
        duration = fmt_ms(request.duration_ms) if request.duration_ms is not None else "—"
        question = request.question
        title = f"«{short(question, 70)}»" if question else request.subject
        warn = s.yellow(f" ⚠{len(request.problems)}") if request.problems else ""
        return (f"{request.start:%d/%m %H:%M:%S}  {s.dim(request.rid[:12]):<12}  {duration:>8}  "
                f"{color(f'{status:<7}')}  {short(request.route, 34):<34}  {title}{warn}")


def render_startup(events: list[Event], renderer: Renderer) -> list[str]:
    """Os eventos sem pedido (rid «-») desde o último arranque."""
    last_start = 0
    for index, event in enumerate(events):
        if event.name == "app.configure":
            last_start = index
    events = events[last_start:]
    s = renderer.s
    if not events:
        return ["Sem eventos de arranque neste log."]
    out = [s.bold(f" ARRANQUE do backend · {events[0].ts:%d/%m/%Y %H:%M:%S}"), ""]
    for item in build_items(events):
        lines = renderer.item(item)
        if item.is_span and not item.name.startswith(("opensearch.", "embedding.", "llm.")):
            lines = [f"{s.dim(item.name)} · {s.bold(fmt_ms(item.duration_ms))}  "
                     f"{renderer.kv(item.fields, skip={'duration_ms'})}"]
        out.append(f"  {item.ts:%H:%M:%S}  {lines[0]}")
        out.extend("            " + line for line in lines[1:])
    ready = next((event for event in reversed(events) if event.name == "warmup.chat_stack.finish"), None)
    if ready is not None:
        out.append("")
        flags = [f"{key.replace('_ready', '')}: {s.green('pronto') if value == 'true' else s.yellow('não carregado')}"
                 for key, value in ready.fields.items() if key.endswith("_ready")]
        out.append("  " + " · ".join(flags))
    return out


# --------------------------------------------------------------------------- #
# Seleção e programa
# --------------------------------------------------------------------------- #

def matches(request: Request, args: argparse.Namespace) -> bool:
    if args.rid and not request.rid.startswith(args.rid):
        return False
    if args.conversa and not (request.field("conversation_id") or "").startswith(args.conversa):
        return False
    if args.query and not any(
        (event.fields.get("query_id") or "").startswith(args.query) for event in request.events
    ):
        return False
    if args.erros and not request.problems and request.status in ("ok", "—"):
        return False
    if args.desde and request.start < args.desde:
        return False
    if args.procura:
        haystack = fold(" ".join(filter(None, [request.question, request.answer, request.route])))
        if fold(args.procura) not in haystack:
            return False
    return True


def parse_since(text: str) -> datetime:
    for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, pattern)
        except ValueError:
            pass
    for pattern in ("%H:%M:%S", "%H:%M"):
        try:
            clock = datetime.strptime(text, pattern)
            return datetime.now().replace(hour=clock.hour, minute=clock.minute, second=clock.second, microsecond=0)
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(f"não percebo a data «{text}» (HH:MM ou AAAA-MM-DD HH:MM)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="amalia_logs",
        description="Os passos do RAG de cada pedido, lidos do log do backend.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "exemplos:\n"
            "  amalia_logs.sh                  os últimos 5 pedidos\n"
            "  amalia_logs.sh -f               em direto, passo a passo (Ctrl+C sai)\n"
            "  amalia_logs.sh -f -l            em direto, uma linha por pedido quando acaba\n"
            "  amalia_logs.sh -l -n 30         índice dos últimos 30\n"
            "  amalia_logs.sh --rid 2d0636 -c  um pedido com prompts e respostas por extenso\n"
            "  amalia_logs.sh -p bosch -n 0    todos os pedidos que falam de «bosch»\n"
            "  amalia_logs.sh -e               só pedidos com avisos ou erros\n"
            "  amalia_logs.sh --arranque       o que carregou quando o backend arrancou\n"
        ),
    )
    parser.add_argument("-n", "--ultimos", type=int, default=None, metavar="N",
                        help="quantos pedidos mostrar, a contar do fim (0 = todos; por omissão 5, ou 30 no índice)")
    parser.add_argument("-f", "--seguir", action="store_true", help="em direto: mostra cada passo quando acontece (com -l ou um filtro, cada pedido quando acaba)")
    parser.add_argument("-l", "--lista", action="store_true", help="índice: uma linha por pedido")
    parser.add_argument("-c", "--completo", action="store_true",
                        help="tudo por extenso: prompts, respostas do modelo, corpo do OpenSearch, blocos JSON")
    parser.add_argument("--rid", help="o pedido com este rid (basta o início)")
    parser.add_argument("--conversa", help="os pedidos desta conversa (início do conversation_id)")
    parser.add_argument("--query", help="o pedido com este query_id (basta o início)")
    parser.add_argument("-p", "--procura", metavar="TEXTO", help="pedidos cuja pergunta ou resposta contém o texto")
    parser.add_argument("-e", "--erros", action="store_true", help="só pedidos com avisos, erros ou sem fim")
    parser.add_argument("--desde", type=parse_since, metavar="HH:MM", help="só a partir desta hora (ou AAAA-MM-DD HH:MM)")
    parser.add_argument("--arranque", action="store_true", help="mostra o arranque do backend em vez dos pedidos")
    parser.add_argument("--ficheiro", default=DEFAULT_LOG, metavar="LOG",
                        help=f"log a ler (por omissão {DEFAULT_LOG}; serve para os de logs/archive/)")
    parser.add_argument("--cor", action="store_true", help="força as cores (ex.: para `| less -R`)")
    parser.add_argument("--sem-cor", action="store_true", help="sem cores")
    parser.add_argument("--sem-pager", action="store_true", help="não passa a saída pelo less")
    return parser


def load(path: str) -> tuple["OrderedDict[str, Request]", list[Event]]:
    requests: "OrderedDict[str, Request]" = OrderedDict()
    system: list[Event] = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for event in read_events(handle):
            file_event(event, requests, system)
    return requests, system


def file_event(event: Event, requests: "OrderedDict[str, Request]", system: list[Event]) -> Request | None:
    if event.rid in ("-", ""):
        system.append(event)
        return None
    request = requests.get(event.rid)
    if request is None:
        request = requests[event.rid] = Request(event.rid)
    request.add(event)
    return request


def follow(path: str, args: argparse.Namespace, renderer: Renderer, requests: "OrderedDict[str, Request]",
           system: list[Event], out: Any) -> None:
    """Depois do que já lá estava, lê o que for chegando. Em direto: cada passo
    sai quando acontece. Com -l ou com um filtro (que só se decide com o pedido
    inteiro), cada pedido sai de uma vez, 2 s depois do fim (a ficha do autor
    ainda escreve linhas a seguir)."""
    for request in requests.values():
        request.printed = True
    s = renderer.s
    whole = bool(args.lista or args.erros or args.procura or args.rid or args.conversa or args.query)
    mode = "cada pedido aparece quando acaba" if whole else "em direto, passo a passo"
    out.write(s.dim(f"a seguir {path} — {mode} (Ctrl+C sai)\n"))
    out.flush()
    parser = Parser()
    handle = open(path, encoding="utf-8", errors="replace")
    handle.seek(0, os.SEEK_END)
    inode = os.fstat(handle.fileno()).st_ino
    pending = ""
    last_data = time.monotonic()
    last_show = 0.0
    last_rid: str | None = None

    def emit(lines: list[str]) -> None:
        if lines:
            out.write("\n".join(lines) + "\n")
            out.flush()

    def take(event: Event | None) -> None:
        if event is None:
            return
        request = file_event(event, requests, system)
        if whole and request is not None and not request.announced and request.is_chat:
            request.announced = True
            emit([s.dim(f"⏳ {event.ts:%H:%M:%S} pedido {request.rid} em curso…")])

    def show(now: float) -> None:
        nonlocal last_rid
        for request in list(requests.values()):
            if request.printed or not request.is_chat:
                continue
            quiet = now - request.touched
            final = (request.ended and quiet >= 2.0) or quiet >= 120.0
            if whole:
                if final:
                    request.printed = True
                    if matches(request, args):
                        emit([renderer.index_line(request)] if args.lista else renderer.request(request) + [""])
                continue
            announced = request.announced
            lines = renderer.live(request, now, final)
            if lines and announced and last_rid not in (None, request.rid):
                # dois pedidos ao mesmo tempo: diz de qual são as linhas que se seguem
                lines.insert(0, s.dim(f"┄┄ continua o pedido {request.rid} ┄┄"))
            if lines:
                last_rid = request.rid
            emit(lines)
            if final:
                request.printed = True

    while True:
        chunk = handle.readline()
        now = time.monotonic()
        if chunk:
            pending += chunk
            if pending.endswith("\n"):
                take(parser.feed(pending))
                pending = ""
            last_data = now
            if now - last_show > 0.2:  # um log sempre a escrever não atrasa o ecrã
                show(now)
                last_show = now
            continue
        if now - last_data > 0.3:  # a última linha só fecha quando chega a seguinte
            take(parser.flush())
        show(now)
        last_show = now
        try:  # o deploy/amalia_online.sh recomeça o ficheiro quando o backend arranca
            stat = os.stat(path)
            if stat.st_ino != inode or stat.st_size < handle.tell():
                handle.close()
                handle = open(path, encoding="utf-8", errors="replace")
                inode = os.fstat(handle.fileno()).st_ino
                parser, pending = Parser(), ""
                emit([s.yellow("— o log recomeçou (o backend foi reiniciado) —")])
        except FileNotFoundError:
            pass
        time.sleep(0.15)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not os.path.isfile(args.ficheiro):
        print(f"Não encontro o log {args.ficheiro} — o backend já arrancou? (deploy/amalia_online.sh --status)",
              file=sys.stderr)
        return 2
    use_color = args.cor or (sys.stdout.isatty() and not args.sem_cor and not os.environ.get("NO_COLOR"))
    style = Style(use_color and not args.sem_cor)
    renderer = Renderer(style, full=args.completo, width=shutil.get_terminal_size((110, 30)).columns - 1)

    requests, system = load(args.ficheiro)
    chats = [request for request in requests.values() if request.is_chat]
    selected = [request for request in chats if matches(request, args)]
    limit = args.ultimos
    if limit is None:
        limit = 0 if (args.rid or args.query) else (30 if args.lista else 5)
        if args.seguir:
            limit = -1  # a seguir, por omissão não repete o que já passou
    shown = [] if limit < 0 else (selected if limit == 0 else selected[-limit:])

    lines: list[str] = []
    if args.arranque:
        lines = render_startup(system, renderer)
    else:
        size_mb = os.path.getsize(args.ficheiro) / 1e6
        span = f"{chats[0].start:%d/%m %H:%M} → {chats[-1].start:%d/%m %H:%M}" if chats else "sem pedidos de conversa"
        lines.append(style.dim(
            f"{args.ficheiro} · {size_mb:.1f} MB · pedidos de conversa: {len(chats)} ({span})"
            + (f" · a mostrar {len(shown)} de {len(selected)}" if not args.seguir else "")
        ))
        if args.lista:
            lines.extend(renderer.index_line(request) for request in shown)
        else:
            for request in shown:
                lines.append("")
                lines.extend(renderer.request(request))
        if not shown and not args.seguir:
            lines.append("Nenhum pedido corresponde ao que foi pedido.")

    text = "\n".join(lines) + "\n"
    pager = None
    if not args.seguir and not args.sem_pager and sys.stdout.isatty() and shutil.which("less"):
        pager = subprocess.Popen(["less", "-RFX"], stdin=subprocess.PIPE, text=True, encoding="utf-8")
    try:
        if pager is not None and pager.stdin is not None:
            pager.stdin.write(text)
            pager.stdin.close()
            pager.wait()
        else:
            sys.stdout.write(text)
            sys.stdout.flush()
        if args.seguir and not args.arranque:
            follow(args.ficheiro, args, renderer, requests, system, sys.stdout)
    except (BrokenPipeError, KeyboardInterrupt):
        try:
            sys.stdout.close()
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
