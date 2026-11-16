#!/usr/bin/env python3
"""Real-time EMG workout assistant (EMG envelope only, no video at inference).

    python assistant.py --replay data/sim_sets/set3_fatigued.csv --speed 5
    python assistant.py --live                      # Arduino sensor on serial

Pipeline: EMG envelope -> RepTracker -> fatigue metrics -> RAG retrieval over
knowledge/*.md -> DeepSeek chat model -> short coaching message.

The DeepSeek key is read from the DEEPSEEK_API_KEY environment variable or from a
.env file next to this script (line: DEEPSEEK_API_KEY=...). Without a key the
assistant still works and prints a rule-based message from the retrieved text.

The data source is always printed in the banner: replayed files are SIMULATED.
"""
import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from emg_features import RepTracker, fatigue_metrics
from fatigue_model import predict_stop_rep
from rag import Retriever

HERE = Path(__file__).parent
API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
SYSTEM = (
    "You are a concise workout assistant for push-ups. You only see an EMG "
    "(muscle activation) envelope from one sensor, summarised as the metrics given; "
    "you have no camera. Use the metrics and the retrieved guidance. Reply in at most "
    "two short sentences, plain language, give one clear action (continue, slow down, "
    "stop the set, or rest). Prefer stopping with reps in reserve over training to "
    "failure. 'predicted_stop_rep' (if present) is a rough learned estimate of the rep "
    "where the stop zone starts; mention it only as approximate. Do not invent numbers. "
    "This is training guidance, not medical advice.")


def load_key():
    k = os.environ.get("DEEPSEEK_API_KEY")
    if k:
        return k.strip()
    env = HERE / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("DEEPSEEK_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"\'')
    return None


def ask_llm(key, metrics, ctx, phase):
    user = (f"Phase: {phase}\nMetrics: {json.dumps(metrics)}\n\nRetrieved guidance:\n"
            + "\n".join(f"- ({t}) {c}" for t, c, _ in ctx))
    body = json.dumps({"model": MODEL, "temperature": 0.3, "max_tokens": 120,
                       "messages": [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(API_URL, body, {
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()


def rule_message(metrics, ctx, phase):
    s = metrics["status"]
    if phase == "set finished":
        rest = "3 minutes" if s in ("at failure", "stop zone") else "2 minutes"
        return (f"Set done: {metrics['n']} reps, tempo loss {metrics['vl_pct']}%. "
                f"Rest about {rest} before the next set.")
    return {"warming up": "Settle into a steady tempo.",
            "fresh": "Tempo is steady, keep going.",
            "building fatigue": "Fatigue is building; hold your tempo and form.",
            "stop zone": "Tempo has dropped about 20%: finish this rep and end the set with reps in reserve.",
            "at failure": "You are at failure: stop the set and rest."}[s]


class Coach:
    def __init__(self, key, on_message=None):
        """on_message(text, source, t, phase) is called instead of printing."""
        self.key, self.rag = key, Retriever()
        self.last_status, self.last_spoken_n = None, 0
        self.busy = False
        self.on_message = on_message

    def maybe_speak(self, reps, t, phase="during set", force=False):
        m = fatigue_metrics(reps)
        n = m["n"]
        # fatigue accumulates within a set, so the status never steps back down
        order = ["warming up", "fresh", "building fatigue", "stop zone", "at failure"]
        if n == 0 or n < getattr(self, "_n_prev", 0):   # new set
            self._peak_rank = 0
        self._n_prev = n
        self._peak_rank = max(getattr(self, "_peak_rank", 0), order.index(m["status"]))
        m["status"] = order[self._peak_rank]
        p = predict_stop_rep(reps)     # learned estimate, available after 8 reps
        if p is not None:
            m["predicted_stop_rep"] = round(p)
        changed = m["status"] != self.last_status
        due = n - self.last_spoken_n >= 5 or (m["status"] in ("stop zone", "at failure")
                                              and n - self.last_spoken_n >= 2)
        if not (force or (n >= 4 and (changed or due))) or self.busy:
            return m
        self.last_status, self.last_spoken_n = m["status"], n
        q = (f"{m['status']} tempo loss {m['vl_pct']}% amplitude rise {m['amp_rise_pct']}% "
             f"stop set recovery rest" + (" rest between sets" if phase == "set finished" else ""))
        ctx = self.rag.search(q, k=3)
        self.busy = True
        threading.Thread(target=self._say, args=(m, ctx, phase, t), daemon=True).start()
        return m

    def _say(self, m, ctx, phase, t):
        try:
            if self.key:
                try:
                    msg, src = ask_llm(self.key, m, ctx, phase), "DeepSeek+RAG"
                except (urllib.error.URLError, KeyError, ValueError) as e:
                    msg, src = rule_message(m, ctx, phase), f"rules (LLM error: {e})"
            else:
                msg, src = rule_message(m, ctx, phase), "rules (no DEEPSEEK_API_KEY)"
            if self.on_message:
                self.on_message(msg, src, t, phase)
            else:
                print(f"\n  COACH [{src}] t={t:5.1f}s: {msg}\n", flush=True)
        finally:
            self.busy = False


def stream_replay(path, speed):
    d = np.genfromtxt(path, delimiter=",", skip_header=1)
    meta = Path(path).with_suffix(".json")
    sim = json.loads(meta.read_text()).get("simulated") if meta.exists() else None
    banner = ("SOURCE: SIMULATED EMG replay of " + Path(path).name if sim
              else "SOURCE: recorded EMG file " + Path(path).name)

    def gen():
        t0 = time.perf_counter()
        for i, (t, v) in enumerate(d):
            if i % 25 == 0 and speed > 0:
                ahead = t / speed - (time.perf_counter() - t0)
                if ahead > 0:
                    time.sleep(ahead)
            yield float(t), float(v)
    return banner, gen()


def stream_live():
    import serial
    from serial.tools import list_ports
    ports = [p.device for p in list_ports.comports() if "ACM" in p.device or "USB" in p.device]
    if not ports:
        raise SystemExit("No Arduino found.")
    ser = serial.Serial(ports[0], 115200, timeout=0.1)
    time.sleep(2)
    ser.reset_input_buffer()

    def gen():
        t0 = time.perf_counter()
        while True:
            try:
                v = int(ser.readline().decode().strip().split(",")[1])
            except (ValueError, IndexError):
                continue
            yield time.perf_counter() - t0, float(v)
    return "SOURCE: live Arduino EMG sensor on " + ports[0], gen()


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--replay", help="CSV with t,adc")
    g.add_argument("--live", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed (0 = as fast as possible)")
    args = ap.parse_args()

    key = load_key()
    banner, stream = stream_live() if args.live else stream_replay(args.replay, args.speed)
    print("=" * 70)
    print("EMG workout assistant  |  input: EMG envelope only (no video)")
    print(banner)
    print("LLM:", "DeepSeek (RAG)" if key else "none (set DEEPSEEK_API_KEY) - rule-based fallback")
    print("=" * 70, flush=True)

    tracker, coach = RepTracker(), Coach(key)
    last_t = 0.0
    try:
        for t, v in stream:
            last_t = t
            rep = tracker.update(t, v)
            if rep:
                m = coach.maybe_speak(tracker.reps, t)
                iv = "  -  " if rep.interval_ms != rep.interval_ms else f"{rep.interval_ms:5.0f}"
                print(f"[t={t:5.1f}s] rep {rep.n:2d} | interval {iv} ms | peak {rep.peak:5.0f} | "
                      f"tempo loss {m['vl_pct']:5.1f}% | amp {m['amp_rise_pct']:+5.0f}% | {m['status']}"
                      + (f" | model: stop zone ~rep {m['predicted_stop_rep']}"
                         if "predicted_stop_rep" in m else ""),
                      flush=True)
            elif tracker.reps and t - tracker.reps[-1].t_peak > 6 and coach.last_spoken_n != -1:
                coach.last_spoken_n = -1  # fire once per set
                time.sleep(0.05)
                coach.busy = False
                coach.maybe_speak(tracker.reps, t, phase="set finished", force=True)
    except KeyboardInterrupt:
        pass
    if tracker.reps and coach.last_spoken_n != -1:   # stream ended: summarise the set
        time.sleep(0.3)
        coach.busy = False
        coach.maybe_speak(tracker.reps, last_t, phase="set finished", force=True)
    time.sleep(2.5)  # let the last coach message print
    print(f"Finished: {len(tracker.reps)} reps over {last_t:.0f} s")


if __name__ == "__main__":
    main()
