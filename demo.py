#!/usr/bin/env python3
"""Demo window: original video + EMG panel + metrics + assistant messages.

    python demo.py recordings/<take> --emg data/sim_sets/set1_real_timing.csv
    python demo.py recordings/<take> --emg ... --save demo.mp4   # also writes a video

The VIDEO IS FOR VIEWING ONLY. The assistant sees nothing but the EMG samples.
The EMG panel is labelled with its provenance (SIMULATED when the .json beside
the csv says so). Keys: space = pause, q = quit.
"""
import argparse
import json
import textwrap
import time
from pathlib import Path

import cv2
import numpy as np

from assistant import Coach, load_key
from emg_features import RepTracker, fatigue_metrics

W, H = 1280, 720
BG, PANEL, TXT, DIM = (24, 24, 28), (38, 38, 46), (235, 235, 235), (150, 150, 160)
GREEN, YELLOW, ORANGE, RED, BLUE = (110, 200, 90), (60, 210, 240), (60, 150, 255), (70, 70, 230), (240, 170, 70)
STATUS_COL = {"warming up": DIM, "fresh": GREEN, "building fatigue": YELLOW,
              "stop zone": ORANGE, "at failure": RED}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def text(img, s, org, scale=0.55, col=TXT, th=1):
    cv2.putText(img, s, org, FONT, scale, col, th, cv2.LINE_AA)


def panel(img, x, y, w, h, title=None):
    cv2.rectangle(img, (x, y), (x + w, y + h), PANEL, -1)
    if title:
        text(img, title, (x + 8, y + 18), 0.5, DIM)


def draw_emg(img, x, y, w, h, t, emg_t, emg_v, reps, win=20.0, vmax=800.0, title=""):
    panel(img, x, y, w, h, title)
    px0, py0, pw, ph = x + 40, y + 28, w - 52, h - 40
    for val in (0, 200, 400, 600, 800):
        yy = int(py0 + ph - val / vmax * ph)
        cv2.line(img, (px0, yy), (px0 + pw, yy), (60, 60, 70), 1)
        text(img, str(val), (x + 6, yy + 4), 0.38, DIM)
    t0 = t - win
    i0, i1 = np.searchsorted(emg_t, [t0, t]) if len(emg_t) else (0, 0)
    seg_t, seg_v = emg_t[i0:i1:4], emg_v[i0:i1:4]
    if len(seg_t) > 1:
        xs = px0 + ((seg_t - t0) / win * pw).astype(int)
        ys = py0 + ph - (np.clip(seg_v, 0, vmax) / vmax * ph).astype(int)
        cv2.polylines(img, [np.stack([xs, ys], 1).reshape(-1, 1, 2)], False, GREEN, 1, cv2.LINE_AA)
    for r in reps:
        if t0 <= r.t_peak <= t:
            xx = px0 + int((r.t_peak - t0) / win * pw)
            cv2.line(img, (xx, py0), (xx, py0 + ph), BLUE, 1)
            text(img, str(r.n), (xx - 6, py0 + 12), 0.4, BLUE)


def draw_metrics(img, x, y, w, h, m, reps):
    panel(img, x, y, w, h, "fatigue metrics (from EMG only)")
    col = STATUS_COL[m["status"]]
    text(img, m["status"].upper(), (x + 10, y + 52), 0.95, col, 2)
    text(img, f"reps {m['n']}", (x + 330, y + 52), 0.8, TXT, 2)
    last = reps[-1].interval_ms if reps and reps[-1].interval_ms == reps[-1].interval_ms else 0
    text(img, f"last rep interval {last:5.0f} ms", (x + 10, y + 82), 0.5)
    text(img, f"EMG amplitude vs first reps {m['amp_rise_pct']:+.0f}%", (x + 10, y + 104), 0.5)
    if m.get("predicted_stop_rep"):
        text(img, f"model estimate: stop zone ~ rep {m['predicted_stop_rep']}", (x + 330, y + 82),
             0.5, BLUE)
    # tempo-loss bar 0..40 %, marks at 20 % (stop zone) and 35 %
    bx, by, bw = x + 10, y + 124, w - 20
    text(img, f"tempo loss {m['vl_pct']:.0f}%", (bx, by + 14), 0.5)
    cv2.rectangle(img, (bx + 130, by), (bx + bw, by + 18), (60, 60, 70), -1)
    v = int(np.clip(m["vl_pct"], 0, 40) / 40 * (bw - 130))
    cv2.rectangle(img, (bx + 130, by), (bx + 130 + v, by + 18), col, -1)
    for mark in (20, 35):
        mx = bx + 130 + int(mark / 40 * (bw - 130))
        cv2.line(img, (mx, by - 3), (mx, by + 21), TXT, 1)
        text(img, f"{mark}%", (mx - 12, by + 36), 0.38, DIM)
    # interval bars per rep, with the 20 % tempo-loss line
    cy, ch = y + 172, h - 180
    if len(reps) > 1:
        iv = np.array([r.interval_ms for r in reps[1:]])
        top = max(iv.max(), 1500.0) * 1.05
        n = len(iv)
        bwid = max(3, min(18, (w - 30) // max(n, 1) - 2))
        for k, val in enumerate(iv):
            hh = int(val / top * (ch - 4))
            xx = x + 12 + k * (bwid + 2)
            cv2.rectangle(img, (xx, cy + ch - hh), (xx + bwid, cy + ch), col, -1)
        if m["n"] >= 4:
            ly = cy + ch - int(m["ref_interval_ms"] / 0.8 / top * (ch - 4))
            cv2.line(img, (x + 8, ly), (x + w - 8, ly), ORANGE, 1)
            text(img, "20% tempo-loss line", (x + w - 150, ly - 4), 0.38, ORANGE)


def draw_messages(img, x, y, w, h, msgs):
    panel(img, x, y, w, h, "assistant (EMG metrics + RAG + LLM)")
    yy = y + 44
    for t, src, msg in msgs[-2:]:
        tag = f"[{t:4.0f}s {src}] "
        lines = textwrap.wrap(msg, 120 - len(tag))
        for i, ln in enumerate(lines):
            text(img, (tag if i == 0 else " " * len(tag)) + ln, (x + 10, yy), 0.5,
                 YELLOW if i == 0 else TXT)
            yy += 22
        yy += 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("take", type=Path)
    ap.add_argument("--emg", required=True, help="EMG csv (t,adc) shown and fed to the assistant")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--save", type=Path, help="also write the demo to this mp4")
    ap.add_argument("--no-window", action="store_true", help="render only (needs --save)")
    ap.add_argument("--show-measured", action="store_true",
                    help="add a strip with the real sensor reading from the take")
    args = ap.parse_args()

    emg = np.genfromtxt(args.emg, delimiter=",", skip_header=1)
    emg_t, emg_v = emg[:, 0], emg[:, 1]
    meta_p = Path(args.emg).with_suffix(".json")
    simulated = bool(json.loads(meta_p.read_text()).get("simulated")) if meta_p.exists() else False
    emg_label = ("SIMULATED EMG envelope" if simulated else "EMG envelope (sensor)")
    source = ("SOURCE: SIMULATED EMG (replay of " + Path(args.emg).name + ")" if simulated
              else "SOURCE: " + Path(args.emg).name)

    measured = None
    if args.show_measured and (args.take / "emg.csv").exists():
        mm = np.genfromtxt(args.take / "emg.csv", delimiter=",", skip_header=1)
        measured = (mm[:, 0], mm[:, 1])

    cap = cv2.VideoCapture(str(args.take / "video.mp4"))
    ft = np.genfromtxt(args.take / "frames.csv", delimiter=",", skip_header=1)[:, 1]
    frame, fi = None, -1
    t_end = float(max(ft[-1], emg_t[-1]))

    msgs = []
    key = load_key()
    coach = Coach(key, on_message=lambda msg, src, t, ph: msgs.append((t, src, msg)))
    tracker = RepTracker()
    ei, metrics, finished = 0, fatigue_metrics([]), False

    writer = None
    out_fps = 20.0
    if args.save:
        writer = cv2.VideoWriter(str(args.save), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W, H))

    t, t_wall0, paused, t_wall_pause = 0.0, time.perf_counter(), False, 0.0
    while t < t_end + 4:
        if not paused:
            if writer:
                while coach.busy:               # keep messages in step in saved video
                    time.sleep(0.02)
                t += args.speed / out_fps
            else:
                t = (time.perf_counter() - t_wall0 - t_wall_pause) * args.speed
        # feed EMG samples up to t (assistant input: EMG only)
        while ei < len(emg_t) and emg_t[ei] <= t:
            rep = tracker.update(float(emg_t[ei]), float(emg_v[ei]))
            if rep:
                metrics = coach.maybe_speak(tracker.reps, float(emg_t[ei]))
            ei += 1
        if tracker.reps and not finished and t - tracker.reps[-1].t_peak > 6:
            finished = True
            coach.busy = False
            coach.maybe_speak(tracker.reps, t, phase="set finished", force=True)
        # video frame for time t
        while fi + 1 < len(ft) and ft[fi + 1] <= t:
            ok, im = cap.read()
            if not ok:
                break
            frame, fi = im, fi + 1

        img = np.full((H, W, 3), BG, np.uint8)
        text(img, "EMG push-up assistant", (12, 26), 0.75, TXT, 2)
        text(img, source, (330, 26), 0.55, ORANGE if simulated else GREEN, 1)
        text(img, "assistant input: EMG only", (W - 250, 26), 0.5, DIM)
        # video (display only)
        panel(img, 10, 40, 640, 480)
        if frame is not None:
            img[40:520, 10:650] = cv2.resize(frame, (640, 480))
        text(img, "video: display only, not used by the assistant", (16, 534), 0.45, DIM)
        text(img, f"t = {t:5.1f} s", (560, 534), 0.5, TXT)
        # EMG + metrics
        draw_emg(img, 660, 40, 610, 200 if measured else 240, t, emg_t, emg_v, tracker.reps,
                 title=emg_label + " (ADC counts)")
        if measured:
            draw_emg(img, 660, 245, 610, 90, t, measured[0], measured[1], [], vmax=800.0,
                     title="MEASURED sensor reading from this take (ADC counts)")
            draw_metrics(img, 660, 340, 610, 180, metrics, tracker.reps)
        else:
            draw_metrics(img, 660, 285, 610, 235, metrics, tracker.reps)
        draw_messages(img, 10, 545, 1260, 165, msgs)
        if paused:
            text(img, "PAUSED", (W // 2 - 50, 20), 0.7, RED, 2)

        if not args.no_window:
            cv2.imshow("EMG assistant demo", img)
        if writer:
            writer.write(img)
        if args.no_window:
            time.sleep(0.002)
            k = 255
        else:
            k = cv2.waitKey(1 if writer else 15) & 0xFF
        if k == ord("q"):
            break
        if k == ord(" "):
            paused = not paused
            if paused:
                t_pause_start = time.perf_counter()
            else:
                t_wall_pause += time.perf_counter() - t_pause_start
    if writer:
        writer.release()
        print("saved", args.save)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
