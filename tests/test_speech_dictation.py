"""Ditado por voz no servidor: o troço em fluxo (partial/final, acordo nos troços
longos), o filtro de alucinações, o VAD em fluxo e o protocolo do WebSocket.

O motor e o VAD são falsos e deterministas: cada «palavra» é um tom cuja
amplitude codifica o número da palavra, e o motor devolve « pN» por tom.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.core.config import Settings
from app.services.speech_dictation import (
    DictationStream,
    Hypothesis,
    StreamOptions,
    Word,
    hypothesis_from_segments,
    is_hallucination,
    origin_allowed,
)

SR = 16000
OPTIONS = StreamOptions(partial_interval=int(0.4 * SR), endpoint_silence=int(0.6 * SR), max_segment=12 * SR, max_session=120 * SR)


def tone(word_id: int, seconds: float) -> np.ndarray:
    amp = 0.1 + 0.015 * word_id
    n = int(round(seconds * SR))
    return (amp * np.where((np.arange(n) // 40) % 2 == 0, 1.0, -1.0)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(round(seconds * SR)), dtype=np.float32)


def speech(ids, word: float = 0.3, gap: float = 0.1) -> np.ndarray:
    parts = []
    for word_id in ids:
        parts += [tone(word_id, word), silence(gap)]
    return np.concatenate(parts[:-1])


def pcm(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


class FakeVad:
    def __init__(self) -> None:
        self._pending = np.zeros(0, dtype=np.float32)

    def feed(self, audio: np.ndarray) -> np.ndarray:
        buffer = np.concatenate([self._pending, audio])
        count = len(buffer) // 512
        self._pending = buffer[count * 512:]
        windows = buffer[: count * 512].reshape(count, 512)
        return (np.abs(windows).max(axis=1) > 0.05).astype(np.float32) if count else np.zeros(0, dtype=np.float32)


class FakeEngine:
    def __init__(self, *, ready: bool = True, max_sessions: int = 6) -> None:
        self.ready = ready
        self.max_sessions = max_sessions
        self.active = 0
        self.calls: list[tuple[float, bool, bool]] = []

    def new_vad(self) -> FakeVad:
        return FakeVad()

    def try_acquire(self) -> bool:
        if self.active >= self.max_sessions:
            return False
        self.active += 1
        return True

    def release(self) -> None:
        self.active -= 1

    async def transcribe(
        self, audio: np.ndarray, language: str, *, word_timestamps: bool = False, urgent: bool = False
    ) -> Hypothesis:
        self.calls.append((len(audio) / SR, word_timestamps, urgent))
        loud = np.abs(audio) > 0.05
        edges = np.flatnonzero(np.diff(np.concatenate([[0], loud.astype(np.int8), [0]])))
        words = []
        for start, end in zip(edges[::2], edges[1::2]):
            word_id = int(round((np.abs(audio[start:end]).max() - 0.1) / 0.015))
            words.append(Word(f" p{word_id}", start / SR, end / SR))
        text = "".join(w.text for w in words).strip()
        return Hypothesis(text, tuple(words) if word_timestamps else ())


async def run_stream(audio: np.ndarray, *, options: StreamOptions = OPTIONS, stop: bool = True):
    """Envia o áudio em pedaços de 100 ms e corre as ações pendentes depois de cada um.
    Devolve (stream, engine, [(t_audio, mensagem, transcrições feitas nesse passo)])."""
    engine = FakeEngine()
    stream = DictationStream(engine, "pt", options, FakeVad())
    data = pcm(audio)
    frame = 1600 * 2
    events = []
    for offset in range(0, len(data), frame):
        stream.feed(data[offset: offset + frame])
        t = min(len(data), offset + frame) / 2 / SR
        while (action := stream.pending_action()) is not None:
            before = len(engine.calls)
            for message in await stream.step(action):
                events.append((t, message, len(engine.calls) - before))
    if stop:
        before = len(engine.calls)
        for message in await stream.finish():
            events.append((len(audio) / SR, message, len(engine.calls) - before))
    return stream, engine, events


def of_type(events, kind):
    return [(t, m["text"], calls) for t, m, calls in events if m["type"] == kind]


class TestStream(unittest.IsolatedAsyncioTestCase):
    async def test_phrase_partial_soon_and_final_soon(self) -> None:
        # fala de 1,0 s a 2,1 s
        stream, engine, events = await run_stream(np.concatenate([silence(1.0), speech([1, 2, 3]), silence(1.5)]))
        partials, finals = of_type(events, "partial"), of_type(events, "final")
        self.assertLess(partials[0][0] - 1.0, 1.0)
        self.assertEqual([text for _, text, _ in finals], ["p1 p2 p3"])
        self.assertLess(finals[0][0] - 2.1, 1.0)
        # o «final» reaproveita a hipótese feita logo depois de a fala parar
        self.assertEqual(finals[0][2], 0)
        self.assertEqual(stream.finals, ["p1 p2 p3"])
        # «partial» só quando o texto muda
        texts = [text for _, text, _ in partials]
        self.assertTrue(all(a != b for a, b in zip(texts, texts[1:])))

    async def test_settle_and_final_are_urgent_partials_are_not(self) -> None:
        engine = FakeEngine()
        stream = DictationStream(engine, "pt", OPTIONS, FakeVad())
        stream.feed(pcm(np.concatenate([silence(0.3), speech([1, 2])])))
        self.assertEqual(stream.pending_action(), "partial")
        await stream.step("partial")
        stream.feed(pcm(np.concatenate([speech([3]), silence(0.3)])))
        self.assertEqual(stream.pending_action(), "settle")
        await stream.step("settle")
        stream.feed(pcm(speech([4])))
        await stream.finish()
        self.assertEqual([urgent for _, _, urgent in engine.calls], [False, True, True])

    async def test_two_phrases_two_finals_without_repeating(self) -> None:
        audio = np.concatenate([silence(1.0), speech([1, 2]), silence(1.2), speech([3, 4]), silence(1.5)])
        stream, _, events = await run_stream(audio)
        self.assertEqual([text for _, text, _ in of_type(events, "final")], ["p1 p2", "p3 p4"])

    async def test_stop_mid_speech_confirms_the_rest(self) -> None:
        stream, _, events = await run_stream(np.concatenate([silence(0.5), speech([1, 2, 3, 4])]))
        self.assertEqual(events[-1][1], {"type": "final", "text": "p1 p2 p3 p4"})
        self.assertEqual(stream.finals, ["p1 p2 p3 p4"])

    async def test_without_stop_nothing_is_confirmed_mid_phrase(self) -> None:
        stream, _, events = await run_stream(np.concatenate([silence(0.5), speech([1, 2, 3])]), stop=False)
        self.assertEqual(of_type(events, "final"), [])
        self.assertTrue(of_type(events, "partial"))

    async def test_silence_and_clicks_are_not_transcribed(self) -> None:
        stream, engine, events = await run_stream(np.concatenate([silence(1.0), tone(5, 0.05), silence(2.0)]))
        self.assertEqual(engine.calls, [])
        self.assertEqual(events, [])

    async def test_long_segment_confirms_stable_words_and_continues(self) -> None:
        ids = list(range(1, 46))  # 18 s sem pausas de 600 ms
        audio = np.concatenate([silence(0.5), speech(ids), silence(1.5)])
        stream, engine, events = await run_stream(audio)
        finals = of_type(events, "final")
        self.assertGreaterEqual(len(finals), 2)
        self.assertLess(finals[0][0], 0.5 + 18.0)  # confirmado ainda durante a fala
        self.assertEqual(" ".join(text for _, text, _ in finals), " ".join(f"p{i}" for i in ids))
        self.assertTrue(any(word_timestamps for _, word_timestamps, _ in engine.calls))
        # o Whisper nunca recebe mais do que a janela de 30 s
        self.assertLess(max(seconds for seconds, _, _ in engine.calls), 30.0)

    async def test_session_limit(self) -> None:
        options = StreamOptions(OPTIONS.partial_interval, OPTIONS.endpoint_silence, OPTIONS.max_segment, 2 * SR)
        stream = DictationStream(FakeEngine(), "pt", options, FakeVad())
        stream.feed(pcm(silence(1.9)))
        self.assertFalse(stream.too_long)
        stream.feed(pcm(silence(0.2)))
        self.assertTrue(stream.too_long)

    async def test_unconfirmed_partial_is_cleared(self) -> None:
        # a hipótese mostrada passa a vazia (alucinação filtrada): o provisório apaga-se
        stream = DictationStream(FakeEngine(), "pt", OPTIONS, FakeVad())
        stream._shown = "Obrigado."
        self.assertEqual(stream._confirm(""), [{"type": "partial", "text": ""}])
        self.assertEqual(stream._confirm(""), [])


class TestPriorityGate(unittest.IsolatedAsyncioTestCase):
    async def test_urgent_requests_go_first(self) -> None:
        from app.services.speech_dictation import _PriorityGate

        gate = _PriorityGate(1)
        await gate.acquire(1)  # ocupado
        order = []

        async def wait(name: str, priority: int) -> None:
            await gate.acquire(priority)
            order.append(name)
            gate.release()

        tasks = [asyncio.create_task(wait("partial-a", 1)), asyncio.create_task(wait("partial-b", 1))]
        await asyncio.sleep(0)
        tasks.append(asyncio.create_task(wait("final", 0)))
        cancelled = asyncio.create_task(wait("cancelado", 0))
        await asyncio.sleep(0)
        cancelled.cancel()
        await asyncio.sleep(0)
        gate.release()
        await asyncio.gather(*tasks)
        self.assertEqual(order, ["final", "partial-a", "partial-b"])
        self.assertEqual(gate._free, 1)


class TestFilters(unittest.TestCase):
    def test_hallucinations(self) -> None:
        for text in ("Obrigado.", "Obrigada por assistir!", "Música", "Legendas pela comunidade Amara.org", "Subscreve o canal!",
                     "Inscreva-se no canal", "Thank you.", "Thanks for watching!", "", " . "):
            self.assertTrue(is_hallucination(text), text)
        for text in ("Obrigado, onde fica a saída?", "Mostra-me leques do século XIX.", "Thank you for the tour, where is the exit?"):
            self.assertFalse(is_hallucination(text), text)

    def test_segments_with_high_no_speech_or_hallucinations_are_dropped(self) -> None:
        w = SimpleNamespace(word=" leques", start=0.5, end=0.9)
        segments = [
            SimpleNamespace(text=" Mostra-me leques", no_speech_prob=0.01, words=[w]),
            SimpleNamespace(text=" ruído", no_speech_prob=0.8, words=[]),
            SimpleNamespace(text=" Legendas por Fulano", no_speech_prob=0.1, words=[]),
        ]
        hypothesis = hypothesis_from_segments(segments, no_speech_threshold=0.6)
        self.assertEqual(hypothesis.text, "Mostra-me leques")
        self.assertEqual(hypothesis.words, (Word(" leques", 0.5, 0.9),))
        only = hypothesis_from_segments([SimpleNamespace(text=" Obrigado.", no_speech_prob=0.0, words=[])], no_speech_threshold=0.6)
        self.assertEqual(only.text, "")

    def test_origin(self) -> None:
        self.assertTrue(origin_allowed(None, ["https://a.pt"]))
        self.assertTrue(origin_allowed("https://x.pt", ["*"]))
        self.assertTrue(origin_allowed("https://A.pt/", ["https://a.pt"]))
        self.assertFalse(origin_allowed("https://b.pt", ["https://a.pt"]))


class TestStreamingVad(unittest.TestCase):
    def test_matches_the_batch_silero(self) -> None:
        try:
            from faster_whisper.vad import get_vad_model
        except ImportError:
            self.skipTest("faster-whisper não instalado")
        from app.services.speech_dictation import StreamingVad

        rng = np.random.default_rng(1)
        audio = np.concatenate([rng.normal(0, 0.001, SR), np.sin(np.arange(SR) * 0.07).astype(np.float32) * 0.3,
                                rng.normal(0, 0.001, SR)]).astype(np.float32)
        audio = audio[: len(audio) // 512 * 512]
        batch = np.asarray(get_vad_model()(audio)).reshape(-1)
        vad = StreamingVad()
        streamed = np.concatenate([vad.feed(audio[i: i + 1600]) for i in range(0, len(audio), 1600)])
        self.assertEqual(len(streamed), len(batch))
        np.testing.assert_allclose(streamed, batch, atol=1e-4)


class FakeQueryLogger:
    def __init__(self) -> None:
        self.events = []

    async def log_speech_session(self, event) -> None:
        self.events.append(event)


class TestWebSocket(unittest.TestCase):
    URL = "/api/v1/speech/stream"
    START = {"type": "start", "language": "pt", "sample_rate": 16000, "encoding": "pcm_s16le",
             "session_id": "s-1", "conversation_id": None, "museum_slug": "mnt"}

    def setUp(self) -> None:
        from app.api.routes import speech

        self.engine = FakeEngine()
        self.query_logger = FakeQueryLogger()
        settings = Settings(_env_file=None, CORS_ALLOW_ORIGINS="https://museu.pt", SPEECH_STT_ENABLED=True)
        patches = [
            mock.patch.object(speech, "get_settings", return_value=settings),
            mock.patch.object(speech, "get_speech_engine", return_value=self.engine),
            mock.patch.object(speech, "get_query_logger", return_value=self.query_logger),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        app = FastAPI()
        app.include_router(speech.router, prefix="/api/v1/speech")
        self.client = TestClient(app)

    def _connect(self, origin: str = "https://museu.pt"):
        return self.client.websocket_connect(self.URL, headers={"origin": origin})

    def test_refused_without_model_so_the_frontend_uses_the_browser(self) -> None:
        self.engine.ready = False
        with self.assertRaises(WebSocketDisconnect) as ctx:
            with self._connect():
                pass
        self.assertEqual(ctx.exception.code, 1013)

    def test_refused_from_other_origins(self) -> None:
        with self.assertRaises(WebSocketDisconnect) as ctx:
            with self._connect("https://outro.example"):
                pass
        self.assertEqual(ctx.exception.code, 1008)

    def test_dictation_until_done(self) -> None:
        audio = np.concatenate([silence(1.0), speech([1, 2, 3]), silence(1.5)])
        data = pcm(audio)
        with self._connect() as ws:
            ws.send_json(self.START)
            self.assertEqual(ws.receive_json(), {"type": "ready"})
            for offset in range(0, len(data), 3200):
                ws.send_bytes(data[offset: offset + 3200])
            ws.send_json({"type": "stop"})
            messages = []
            while not messages or messages[-1]["type"] != "done":
                messages.append(ws.receive_json())
        finals = [m["text"] for m in messages if m["type"] == "final"]
        self.assertEqual(finals, ["p1 p2 p3"])
        self.assertTrue(all(m["type"] in {"partial", "final", "done"} for m in messages))
        self.assertEqual(self.engine.active, 0)
        (event,) = self.query_logger.events
        self.assertEqual({k: event[k] for k in ("session_id", "museum_slug", "language", "text", "end")},
                         {"session_id": "s-1", "museum_slug": "mnt", "language": "pt", "text": "p1 p2 p3", "end": "stop"})
        self.assertAlmostEqual(event["duration_s"], len(audio) / SR, places=1)
        self.assertFalse(any(isinstance(v, (bytes, list)) for v in event.values()))  # nunca áudio

    def test_close_without_stop_discards(self) -> None:
        with self._connect() as ws:
            ws.send_json(self.START)
            ws.receive_json()
            ws.send_bytes(pcm(np.concatenate([silence(0.3), speech([1, 2])])))
        self.assertEqual(self.engine.active, 0)
        self.assertEqual(self.query_logger.events[-1]["end"], "closed")

    def test_busy(self) -> None:
        self.engine.max_sessions = 0
        with self._connect() as ws:
            ws.send_json(self.START)
            message = ws.receive_json()
        self.assertEqual((message["type"], message["code"]), ("error", "busy"))

    def test_unsupported_format(self) -> None:
        with self._connect() as ws:
            ws.send_json({**self.START, "sample_rate": 44100})
            message = ws.receive_json()
        self.assertEqual((message["type"], message["code"]), ("error", "unsupported_format"))

    def test_too_long(self) -> None:
        settings = Settings(_env_file=None, SPEECH_STT_ENABLED=True, SPEECH_MAX_SESSION_SECONDS=1.0)
        from app.api.routes import speech as speech_route

        with mock.patch.object(speech_route, "get_settings", return_value=settings), self._connect() as ws:
            ws.send_json(self.START)
            ws.receive_json()
            ws.send_bytes(pcm(np.concatenate([silence(0.2), speech([1, 2, 3])])))
            messages = []
            while not messages or messages[-1]["type"] != "error":
                messages.append(ws.receive_json())
        self.assertEqual(messages[-1]["code"], "too_long")
        self.assertEqual(self.query_logger.events[-1]["end"], "too_long")


if __name__ == "__main__":
    unittest.main()
