"""Ditado por voz no servidor: Whisper (faster-whisper) em fluxo para o
WebSocket /api/v1/speech/stream.

O áudio chega em pedaços de 100 ms (PCM Int16, 16 kHz, mono). O Silero VAD (o
mesmo ONNX do faster-whisper) corre em fluxo e decide os troços:

- o troço aberto é transcrito a cada ~400 ms de áudio novo e o texto vai como
  «partial» quando muda;
- 200 ms depois de a fala parar transcreve-se logo o troço inteiro, para que o
  «final» (≥ 600 ms de silêncio) saia sem nova transcrição;
- um troço longo (> 12 s) confirma as palavras iguais em duas hipóteses
  seguidas (LocalAgreement, como no whisper_streaming), corta o áudio no fim
  da última e continua com o resto.

O modelo carrega uma vez no arranque e é partilhado; as transcrições correm
numa pool de threads (o CTranslate2 larga o GIL). O áudio nunca é guardado.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import json
import logging
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Protocol

import numpy as np

from app.core.config import Settings, get_settings
from app.core.logging import log_event

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
_VAD_WINDOW = 512  # 32 ms: a janela do Silero a 16 kHz
_VAD_CONTEXT = 64
_SPEECH_THRESHOLD = 0.5
_SILENCE_THRESHOLD = 0.35  # entre os dois, conta como fala se já se estava a falar
_PREROLL = SAMPLE_RATE // 2  # áudio guardado antes do início da fala
_MIN_VOICED_PARTIAL = int(0.3 * SAMPLE_RATE)  # primeira hipótese com 300 ms de fala
_MIN_VOICED_FINAL = int(0.1 * SAMPLE_RATE)  # menos do que isto é um estalido
_SETTLE = int(0.2 * SAMPLE_RATE)  # fala parada há 200 ms: transcreve já o troço inteiro
_UNSTABLE_TAIL = 0.5  # s: palavras tão perto do fim do áudio ainda podem mudar
_HARD_SEGMENT_EXTRA = 8.0  # s além do troço longo sem acordo: força o «final» (a janela do Whisper é 30 s)

PARTIAL, SETTLE, FINAL, LONG = "partial", "settle", "final", "long"


# --- alucinações típicas do Whisper em silêncio ou ruído --------------------

_HALLUCINATION_EXACT = frozenset({
    "musica", "obrigado", "obrigada", "muito obrigado", "muito obrigada",
    "obrigado por assistir", "obrigada por assistir", "obrigado por assistirem", "obrigado por ver",
    "thank you", "thanks", "thank you very much", "thank you for watching", "thanks for watching", "you",
})
_HALLUCINATION_PATTERN = re.compile(
    r"\blegendas? (?:por|pela|pelo|de)\b|\blegendad[oa] por\b|\bamara org\b"
    r"|\bsubscrev\w* (?:o|ao|a) (?:nosso )?canal\b|\binscrev\w*(?: se)? (?:no|ao|o) (?:nosso )?canal\b"
    r"|\bsubtitles? by\b|\bsubscribe to\b|\blike and subscribe\b|\btranscri\w* por\b"
)


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", (text or "").lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def is_hallucination(text: str) -> bool:
    folded = _fold(text)
    return not folded or folded in _HALLUCINATION_EXACT or bool(_HALLUCINATION_PATTERN.search(folded))


@dataclass(frozen=True)
class Word:
    text: str  # com o espaço à frente, como vem do Whisper
    start: float
    end: float


@dataclass(frozen=True)
class Hypothesis:
    text: str
    words: tuple[Word, ...] = ()


def hypothesis_from_segments(segments: Iterable[Any], *, no_speech_threshold: float) -> Hypothesis:
    """Junta os segmentos do Whisper, sem os de no_speech_prob alto nem as frases
    que ele inventa em silêncio («Obrigado.», «Legendas por…»)."""
    texts: list[str] = []
    words: list[Word] = []
    for segment in segments:
        text = (getattr(segment, "text", "") or "").strip()
        if float(getattr(segment, "no_speech_prob", 0.0) or 0.0) >= no_speech_threshold:
            continue
        if is_hallucination(text):
            continue
        texts.append(text)
        words.extend(Word(w.word, float(w.start), float(w.end)) for w in (getattr(segment, "words", None) or ()))
    text = " ".join(texts).strip()
    if is_hallucination(text):
        return Hypothesis("")
    return Hypothesis(text, tuple(words))


# --- VAD em fluxo ------------------------------------------------------------

class StreamingVad:
    """Silero VAD em fluxo com o ONNX do faster-whisper: guarda o estado LSTM e
    os 64 samples de contexto entre pedaços, e devolve uma probabilidade de fala
    por janela de 32 ms (igual a correr o modelo no áudio todo de uma vez)."""

    def __init__(self, model: Any | None = None) -> None:
        if model is None:
            from faster_whisper.vad import get_vad_model

            model = get_vad_model()
        self._session = model.session
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(_VAD_CONTEXT, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)

    def feed(self, audio: np.ndarray) -> np.ndarray:
        buffer = np.concatenate([self._pending, audio.astype(np.float32, copy=False)])
        count = len(buffer) // _VAD_WINDOW
        self._pending = buffer[count * _VAD_WINDOW:]
        if not count:
            return np.zeros(0, dtype=np.float32)
        windows = buffer[: count * _VAD_WINDOW].reshape(count, _VAD_WINDOW)
        contexts = np.concatenate([self._context[None, :], windows[:-1, -_VAD_CONTEXT:]], axis=0)
        batch = np.ascontiguousarray(np.concatenate([contexts, windows], axis=1), dtype=np.float32)
        output, self._h, self._c = self._session.run(None, {"input": batch, "h": self._h, "c": self._c})
        self._context = windows[-1, -_VAD_CONTEXT:].copy()
        return np.asarray(output, dtype=np.float32).reshape(-1)


# --- o troço em fluxo --------------------------------------------------------

class Transcriber(Protocol):
    async def transcribe(
        self, audio: np.ndarray, language: str, *, word_timestamps: bool = False, urgent: bool = False
    ) -> Hypothesis: ...


@dataclass(frozen=True)
class StreamOptions:
    partial_interval: int  # samples
    endpoint_silence: int
    max_segment: int
    max_session: int

    @classmethod
    def from_settings(cls, settings: Settings) -> "StreamOptions":
        return cls(
            partial_interval=int(settings.SPEECH_PARTIAL_INTERVAL_MS * SAMPLE_RATE / 1000),
            endpoint_silence=int(settings.SPEECH_ENDPOINT_SILENCE_MS * SAMPLE_RATE / 1000),
            max_segment=int(settings.SPEECH_MAX_SEGMENT_SECONDS * SAMPLE_RATE),
            max_session=int(settings.SPEECH_MAX_SESSION_SECONDS * SAMPLE_RATE),
        )


def _join(words: Iterable[Word]) -> str:
    return "".join(word.text for word in words).strip()


class DictationStream:
    """Estado de uma sessão de ditado. As decisões contam samples, não relógio:
    com o áudio ao ritmo real dá o mesmo, e os testes são deterministas."""

    def __init__(self, engine: Transcriber, language: str, options: StreamOptions, vad: Any) -> None:
        self.engine = engine
        self.language = language
        self.options = options
        self.vad = vad
        self.finals: list[str] = []
        self._audio = np.zeros(0, dtype=np.float32)  # o troço aberto
        self._total = 0  # samples recebidos na sessão
        self._vad_pos = 0  # samples já vistos pelo VAD
        self._speaking = False
        self._voiced = 0  # samples com fala no troço aberto
        self._silence = 0  # silêncio seguido desde a última fala do troço
        self._hyp_text: str | None = None  # última hipótese do troço aberto
        self._hyp_until = 0  # até onde (em _total) essa hipótese ouviu
        self._hyp_words: list[str] = []  # as suas palavras, normalizadas, para o acordo
        self._shown = ""  # último «partial» enviado

    @property
    def duration_seconds(self) -> float:
        return self._total / SAMPLE_RATE

    @property
    def too_long(self) -> bool:
        return self._total >= self.options.max_session

    def feed(self, pcm: bytes) -> None:
        usable = len(pcm) - len(pcm) % 2
        if usable <= 0:
            return
        audio = np.frombuffer(pcm[:usable], dtype="<i2").astype(np.float32) / 32768.0
        self._total += len(audio)
        self._audio = np.concatenate([self._audio, audio])
        for prob in self.vad.feed(audio):
            self._vad_pos += _VAD_WINDOW
            if prob >= _SPEECH_THRESHOLD or (self._speaking and prob >= _SILENCE_THRESHOLD):
                self._speaking = True
                self._voiced += _VAD_WINDOW
                self._silence = 0
            else:
                self._speaking = False
                if self._voiced:
                    self._silence += _VAD_WINDOW
        if 0 < self._voiced < _MIN_VOICED_FINAL and self._silence >= self.options.endpoint_silence:
            self._reset_segment()  # um estalido: não chega a ser fala
        if not self._voiced and len(self._audio) > _PREROLL:
            self._audio = self._audio[-_PREROLL:]

    def _covers_speech(self) -> bool:
        return self._hyp_text is not None and self._hyp_until >= self._vad_pos - self._silence

    def pending_action(self) -> str | None:
        if self._voiced < _MIN_VOICED_FINAL:
            return None
        if self._silence >= self.options.endpoint_silence:
            return FINAL
        if self._covers_speech():
            return None
        if self._silence >= _SETTLE:
            return SETTLE
        # enquanto não há texto à vista (o Whisper inventa com 300 ms de fala), tenta-se mais cedo
        interval = self.options.partial_interval if self._shown else self.options.partial_interval // 2
        if self._voiced >= _MIN_VOICED_PARTIAL and self._total - self._hyp_until >= interval:
            return LONG if len(self._audio) >= self.options.max_segment else PARTIAL
        return None

    async def step(self, action: str) -> list[dict[str, str]]:
        if action == FINAL:
            return await self._final()
        if action == LONG:
            return await self._long()
        # a hipótese de quando a fala para vai ser o «final»: passa à frente dos «partial»
        hypothesis, _ = await self._transcribe(urgent=action == SETTLE)
        return self._show(hypothesis.text)

    async def finish(self) -> list[dict[str, str]]:
        """«stop»: confirma o que falta do troço aberto."""
        if self._voiced < _MIN_VOICED_FINAL:
            messages = self._confirm("")
        elif self._hyp_text is not None and self._hyp_until >= self._total:
            messages = self._confirm(self._hyp_text)
        else:
            hypothesis, _ = await self._transcribe(urgent=True)
            messages = self._confirm(hypothesis.text)
        self._reset_segment()
        return messages

    # -- ações

    async def _transcribe(self, *, word_timestamps: bool = False, urgent: bool = False) -> tuple[Hypothesis, int]:
        audio, until = self._audio, self._total
        hypothesis = await self.engine.transcribe(audio, self.language, word_timestamps=word_timestamps, urgent=urgent)
        self._hyp_text = hypothesis.text
        self._hyp_until = until
        self._hyp_words = (
            [_fold(word.text) for word in hypothesis.words] if hypothesis.words else _fold(hypothesis.text).split()
        )
        return hypothesis, len(audio)

    async def _final(self) -> list[dict[str, str]]:
        if self._covers_speech():
            text = self._hyp_text or ""
        else:
            hypothesis, _ = await self._transcribe(urgent=True)
            text = hypothesis.text
        messages = self._confirm(text)
        self._reset_segment()
        return messages

    async def _long(self) -> list[dict[str, str]]:
        previous = self._hyp_words
        hypothesis, heard = await self._transcribe(word_timestamps=True)
        words = list(hypothesis.words)
        current = [_fold(word.text) for word in words]
        stable_until = heard / SAMPLE_RATE - _UNSTABLE_TAIL
        agreed = 0
        while (
            agreed < min(len(words), len(previous))
            and current[agreed] == previous[agreed]
            and words[agreed].end <= stable_until
        ):
            agreed += 1
        if agreed:
            messages = self._confirm(_join(words[:agreed]))
            # corta a meio da pausa: os tempos das palavras falham por 100–200 ms e
            # cortar no fim da última apanhava o início da seguinte
            cut = words[agreed - 1].end
            if agreed < len(words):
                cut = (cut + max(cut, words[agreed].start)) / 2
            self._cut(cut, heard)
            self._hyp_text = _join(words[agreed:])
            self._hyp_words = current[agreed:]
            return messages + self._show(self._hyp_text)
        if heard >= self.options.max_segment + _HARD_SEGMENT_EXTRA * SAMPLE_RATE:
            messages = self._confirm(hypothesis.text)
            self._cut(words[-1].end if words else heard / SAMPLE_RATE, heard)
            self._hyp_text, self._hyp_words = "", []
            return messages
        return self._show(hypothesis.text)

    # -- estado

    def _show(self, text: str) -> list[dict[str, str]]:
        if text == self._shown:
            return []
        self._shown = text
        return [{"type": "partial", "text": text}]

    def _confirm(self, text: str) -> list[dict[str, str]]:
        text = (text or "").strip()
        if text:
            self.finals.append(text)
            messages = [{"type": "final", "text": text}]
        elif self._shown:
            messages = [{"type": "partial", "text": ""}]  # apaga o provisório que não se confirmou
        else:
            messages = []
        self._shown = ""
        return messages

    def _cut(self, seconds: float, heard: int) -> None:
        cut = max(0, min(int(round(seconds * SAMPLE_RATE)), heard))
        self._audio = self._audio[cut:]

    def _reset_segment(self) -> None:
        self._audio = self._audio[-_PREROLL:]
        self._voiced = 0
        self._silence = 0
        self._hyp_text = None
        self._hyp_until = self._total
        self._hyp_words = []


# --- o modelo partilhado -----------------------------------------------------

class _PriorityGate:
    """Lugares no modelo: com fila, as transcrições urgentes (as que fecham um
    troço) passam à frente dos «partial» das outras sessões."""

    def __init__(self, slots: int) -> None:
        self._free = slots
        self._waiting: list[tuple[int, int, asyncio.Future]] = []
        self._order = itertools.count()

    async def acquire(self, priority: int) -> None:
        if self._free > 0 and not self._waiting:
            self._free -= 1
            return
        future = asyncio.get_running_loop().create_future()
        heapq.heappush(self._waiting, (priority, next(self._order), future))
        try:
            await future
        except asyncio.CancelledError:
            if future.done() and not future.cancelled():
                self.release()  # o lugar chegou ao mesmo tempo que o cancelamento
            raise

    def release(self) -> None:
        while self._waiting:
            _, _, future = heapq.heappop(self._waiting)
            if not future.done():
                future.set_result(None)
                return
        self._free += 1


class SpeechEngine:
    """O Whisper carregado uma vez (no arranque) e partilhado pelas sessões."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: Any | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._gate: _PriorityGate | None = None
        self._slots = threading.Lock()
        self._active = 0
        self.error: str | None = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def active_sessions(self) -> int:
        return self._active

    def load(self) -> bool:
        if self.ready:
            return True
        settings = self.settings
        workers = max(1, int(settings.SPEECH_NUM_WORKERS))
        try:
            started = time.perf_counter()
            if settings.SPEECH_DEVICE.startswith("cuda"):
                # o CTranslate2 abre o libcublas/libcudnn por nome: vêm das rodas
                # nvidia-* que o torch já carrega
                import torch  # noqa: F401
            from faster_whisper import WhisperModel
            from faster_whisper.vad import get_vad_model

            logging.getLogger("faster_whisper").setLevel(logging.WARNING)  # uma linha INFO por transcrição

            options = dict(device=settings.SPEECH_DEVICE, compute_type=settings.SPEECH_COMPUTE_TYPE, num_workers=workers)
            try:
                model = WhisperModel(settings.SPEECH_MODEL, local_files_only=True, **options)
            except Exception:
                model = WhisperModel(settings.SPEECH_MODEL, **options)  # ainda não está na cache
            loaded = time.perf_counter()
            # aquecimento com 1 s de silêncio (sem VAD, para correr mesmo o modelo)
            segments, _ = model.transcribe(
                np.zeros(SAMPLE_RATE, dtype=np.float32), language="pt", beam_size=1,
                vad_filter=False, without_timestamps=True,
            )
            list(segments)
            get_vad_model()(np.zeros(32 * _VAD_WINDOW, dtype=np.float32))
            self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="speech-stt")
            self._model = model
            log_event(
                logger, logging.INFO, "speech.model.ready",
                model=settings.SPEECH_MODEL, device=settings.SPEECH_DEVICE, compute_type=settings.SPEECH_COMPUTE_TYPE,
                workers=workers, load_ms=round((loaded - started) * 1000), warmup_ms=round((time.perf_counter() - loaded) * 1000),
            )
            return True
        except Exception as exc:  # o resto do backend arranca na mesma; o frontend usa o ditado do browser
            self.error = f"{type(exc).__name__}: {exc}"
            logger.exception("speech.model.failed model=%s", settings.SPEECH_MODEL)
            return False

    def new_vad(self) -> StreamingVad:
        return StreamingVad()

    def try_acquire(self) -> bool:
        with self._slots:
            if self._active >= max(0, int(self.settings.SPEECH_MAX_SESSIONS)):
                return False
            self._active += 1
            return True

    def release(self) -> None:
        with self._slots:
            self._active = max(0, self._active - 1)

    async def transcribe(
        self, audio: np.ndarray, language: str, *, word_timestamps: bool = False, urgent: bool = False
    ) -> Hypothesis:
        if self._model is None or self._executor is None:
            raise RuntimeError("speech model not loaded")
        if self._gate is None:
            self._gate = _PriorityGate(max(1, int(self.settings.SPEECH_NUM_WORKERS)))
        await self._gate.acquire(0 if urgent else 1)
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._executor, self._transcribe_sync, audio, language, word_timestamps)
        finally:
            self._gate.release()

    def _transcribe_sync(self, audio: np.ndarray, language: str, word_timestamps: bool) -> Hypothesis:
        segments, _ = self._model.transcribe(
            audio,
            language=language,
            beam_size=1,
            temperature=0.0,  # sem as repetições a temperaturas mais altas: a latência manda
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=True,
            word_timestamps=word_timestamps,
        )
        return hypothesis_from_segments(segments, no_speech_threshold=float(self.settings.SPEECH_NO_SPEECH_THRESHOLD))


@lru_cache(maxsize=1)
def get_speech_engine() -> SpeechEngine:
    return SpeechEngine(get_settings())


# --- a sessão no WebSocket ---------------------------------------------------

_START_TIMEOUT = 10.0
_WALL_GRACE = 30.0  # além da duração máxima de áudio, para clientes que param de enviar


def normalize_language(value: Any) -> str:
    return "en" if str(value or "").strip().lower().startswith("en") else "pt"


def origin_allowed(origin: str | None, allowed: list[str]) -> bool:
    """As mesmas origens do CORS do chat. Sem Origin (cliente que não é browser) passa."""
    if not origin or "*" in allowed:
        return True
    wanted = origin.strip().rstrip("/").lower()
    return any(wanted == item.strip().rstrip("/").lower() for item in allowed)


class _Closed(Exception):
    pass


async def run_dictation_session(websocket: Any, engine: Any, settings: Settings, query_logger: Any | None = None) -> str:
    """Protocolo do serverDictation.ts do frontend: start → ready → áudio →
    partial/final → stop → final(s) → done. Fechar sem stop descarta tudo."""

    async def send(message: dict[str, str]) -> None:
        try:
            await websocket.send_json(message)
        except Exception as exc:  # o cliente já fechou
            raise _Closed from exc

    async def fail(code: str, message: str) -> None:
        try:
            await send({"type": "error", "code": code, "message": message})
            await websocket.close()
        except Exception:
            pass

    try:
        start = await asyncio.wait_for(_receive_start(websocket), _START_TIMEOUT)
    except asyncio.TimeoutError:
        await fail("bad_request", "Faltou a mensagem start.")
        return "bad_request"
    except _Closed:
        return "closed"
    if start is None:
        await fail("bad_request", "A primeira mensagem tem de ser {\"type\": \"start\"}.")
        return "bad_request"
    if int(start.get("sample_rate") or 0) != SAMPLE_RATE or str(start.get("encoding") or "") != "pcm_s16le":
        await fail("unsupported_format", "Só PCM Int16 little-endian, mono, 16 kHz.")
        return "unsupported_format"
    if not engine.try_acquire():
        await fail("busy", "O ditado está ocupado; tenta daqui a pouco.")
        return "busy"

    language = normalize_language(start.get("language"))
    stream = DictationStream(engine, language, StreamOptions.from_settings(settings), engine.new_vad())
    end = "closed"
    try:
        end = await _drive(websocket, stream, settings, send)
        if end == "too_long":
            await fail("too_long", "O ditado chegou ao limite de duração.")
        elif end == "failed":
            await fail("failed", "O ditado falhou.")
        elif end == "stop":
            try:
                await websocket.close()
            except Exception:
                pass
        return end
    finally:
        engine.release()
        await _log_session(start, stream, end, query_logger)


async def _receive_start(websocket: Any) -> dict[str, Any] | None:
    while True:
        message = await websocket.receive()
        if message.get("type") == "websocket.disconnect":
            raise _Closed
        text = message.get("text")
        if text is None:
            continue  # áudio antes do start: ignora
        try:
            payload = json.loads(text)
        except ValueError:
            return None
        return payload if isinstance(payload, dict) and payload.get("type") == "start" else None


async def _drive(websocket: Any, stream: DictationStream, settings: Settings, send: Any) -> str:
    await send({"type": "ready"})
    wake = asyncio.Event()
    stopping = False

    async def process() -> None:
        while True:
            await wake.wait()
            wake.clear()
            while not stopping and (action := stream.pending_action()) is not None:
                for message in await stream.step(action):
                    await send(message)
            if stopping:
                return

    processor = asyncio.create_task(process())
    loop = asyncio.get_running_loop()
    deadline = loop.time() + float(settings.SPEECH_MAX_SESSION_SECONDS) + _WALL_GRACE
    end = "closed"
    try:
        while True:
            if processor.done():
                end = "closed" if isinstance(processor.exception(), _Closed) else "failed"
                break
            try:
                message = await asyncio.wait_for(websocket.receive(), max(0.0, deadline - loop.time()))
            except asyncio.TimeoutError:
                end = "too_long"
                break
            if message.get("type") == "websocket.disconnect":
                end = "closed"
                break
            data = message.get("bytes")
            if data:
                stream.feed(data)
                if stream.too_long:
                    end = "too_long"
                    break
                wake.set()
                continue
            text = message.get("text")
            if text:
                try:
                    payload = json.loads(text)
                except ValueError:
                    continue
                if isinstance(payload, dict) and payload.get("type") == "stop":
                    end = "stop"
                    break
    except Exception:
        end = "closed"

    if end == "closed":
        processor.cancel()
        try:
            await processor
        except BaseException:
            pass
        return end

    # stop, limite ou falha: acaba o passo em curso e confirma o resto
    stopping = True
    wake.set()
    try:
        await processor
    except _Closed:
        return "closed"
    except Exception:
        logger.exception("speech.session.failed")
        end = "failed"
    if end == "failed":
        return end
    try:
        for message in await stream.finish():
            await send(message)
        if end == "stop":
            await send({"type": "done"})
    except _Closed:
        return "closed"
    except Exception:
        logger.exception("speech.session.failed")
        return "failed"
    return end


async def _log_session(start: dict[str, Any], stream: DictationStream, end: str, query_logger: Any | None) -> None:
    """Só metadados e o texto confirmado; o áudio nunca é guardado."""
    fields = {
        "session_id": str(start.get("session_id") or "") or None,
        "museum_slug": str(start.get("museum_slug") or "") or None,
        "language": stream.language,
        "duration_s": round(stream.duration_seconds, 2),
        "end": end,
    }
    log_event(logger, logging.INFO, "speech.session.end", finals=len(stream.finals), **fields)
    if query_logger is None:
        return
    from app.services.query_logger import utc_timestamp

    try:
        await query_logger.log_speech_session(
            {"event_type": "speech_session", "timestamp": utc_timestamp(), **fields, "text": " ".join(stream.finals)}
        )
    except Exception:
        logger.warning("speech.session.log_failed", exc_info=True)
