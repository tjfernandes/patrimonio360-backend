"""Testa o ditado por voz (/api/v1/speech/stream) com um WAV de 16 kHz mono.

Envia o áudio em pedaços de 100 ms ao ritmo real (cada pedaço sai quando acaba,
como o microfone do browser) e imprime cada mensagem com o tempo desde o início
do áudio. No fim compara com os troços de fala do WAV (por energia): o primeiro
«partial» deve chegar < 1 s depois de a fala começar e cada «final» < 1 s
depois de ela acabar.

    python scripts/testar_ditado_ws.py frase.wav
    python scripts/testar_ditado_ws.py frase.wav --url wss://patrimonio360.webredirect.org:8081/api/v1/speech/stream
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import wave

import numpy as np
from websockets.sync.client import connect

SR = 16000
FRAME = 1600  # 100 ms


def read_wav(path: str) -> np.ndarray:
    with wave.open(path) as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SR, 1, 2):
            sys.exit(f"{path}: precisa de PCM 16 bits, mono, 16 kHz")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


def speech_intervals(samples: np.ndarray, threshold_db: float = -40.0, max_gap: float = 0.4) -> list[tuple[float, float]]:
    """Troços de fala por energia (janelas de 20 ms), com pausas < 400 ms juntas."""
    hop = SR // 50
    x = samples[: len(samples) // hop * hop].astype(np.float32).reshape(-1, hop) / 32768.0
    loud = 20 * np.log10(np.sqrt((x ** 2).mean(axis=1)) + 1e-9) > threshold_db
    intervals: list[list[float]] = []
    for i in np.flatnonzero(loud):
        start, end = i * hop / SR, (i + 1) * hop / SR
        if intervals and start - intervals[-1][1] < max_gap:
            intervals[-1][1] = end
        else:
            intervals.append([start, end])
    return [(a, b) for a, b in intervals if b - a >= 0.1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wav", nargs="+")
    parser.add_argument("--url", default="ws://127.0.0.1:8001/api/v1/speech/stream")
    parser.add_argument("--language", default="pt")
    parser.add_argument("--museum", default="mnt")
    parser.add_argument("--session-id", default="teste-ditado")
    parser.add_argument("--origin", default=None, help="cabeçalho Origin (como o browser)")
    args = parser.parse_args()
    ok = True
    for path in args.wav:
        ok = run(path, args) and ok
    return 0 if ok else 1


def run(path: str, args: argparse.Namespace) -> bool:
    samples = read_wav(path)
    speech = speech_intervals(samples)
    print(f"\n=== {path} ({len(samples) / SR:.2f} s) · fala: " + ", ".join(f"{a:.2f}–{b:.2f} s" for a, b in speech))
    opened = time.monotonic()
    with connect(args.url, origin=args.origin, open_timeout=10, max_size=None) as ws:
        ws.send(json.dumps({
            "type": "start", "language": args.language, "sample_rate": SR, "encoding": "pcm_s16le",
            "session_id": args.session_id, "conversation_id": None, "museum_slug": args.museum,
        }))
        first = json.loads(ws.recv(timeout=4))
        print(f"  ligação + ready: {(time.monotonic() - opened) * 1000:.0f} ms → {first}")
        if first.get("type") != "ready":
            return False

        received: list[tuple[float, dict]] = []
        t0 = time.monotonic()

        def listen() -> None:
            try:
                for raw in ws:
                    message = json.loads(raw)
                    t = time.monotonic() - t0
                    received.append((t, message))
                    text = message.get("text", message.get("code", ""))
                    print(f"  {t:6.2f} s  {message['type']:8} {text}")
                    if message["type"] in {"done", "error"}:
                        return
            except Exception:
                return

        listener = threading.Thread(target=listen, daemon=True)
        listener.start()
        for index in range(0, len(samples), FRAME):
            wait = t0 + (index + FRAME) / SR - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            ws.send(samples[index: index + FRAME].tobytes())
        audio_end = time.monotonic() - t0
        ws.send(json.dumps({"type": "stop"}))
        print(f"  {audio_end:6.2f} s  (stop)")
        listener.join(timeout=5)

    partials = [(t, m) for t, m in received if m["type"] == "partial" and m.get("text")]
    finals = [(t, m) for t, m in received if m["type"] == "final"]
    good = bool(speech and finals and received and received[-1][1]["type"] == "done")
    if speech and partials:
        lag = partials[0][0] - speech[0][0]
        good &= lag < 1.0
        print(f"  primeiro partial: {lag:.2f} s depois de a fala começar {'✓' if lag < 1.0 else '✗'}")
    # fim de fala = seguido de ≥ 600 ms de silêncio (ou do fim do WAV); os «final»
    # antes disso são troços longos confirmados a meio da fala
    total = len(samples) / SR
    endpoints = [end for (_, end), nxt in zip(speech, speech[1:] + [(total + 1.0, 0)]) if nxt[0] - end >= 0.6]
    used: set[int] = set()
    for end in endpoints:
        after = [(i, t, m) for i, (t, m) in enumerate(finals) if t >= end and i not in used]
        if not after:
            good = False
            print(f"  fala acabou aos {end:.2f} s e não veio «final» ✗")
            continue
        i, t, message = after[0]
        used.add(i)
        lag = t - end
        after_stop = t > audio_end
        good &= lag < 1.0 or after_stop
        mark = "(depois do stop)" if after_stop else ("✓" if lag < 1.0 else "✗")
        print(f"  final «{message['text']}»: {lag:.2f} s depois de a fala acabar {mark}")
    for i, (t, message) in enumerate(finals):
        if i not in used:
            print(f"  final a meio da fala ({t:.2f} s): «{message['text']}»")
    print("  texto: " + " ".join(m["text"] for _, m in finals))
    return good


if __name__ == "__main__":
    sys.exit(main())
