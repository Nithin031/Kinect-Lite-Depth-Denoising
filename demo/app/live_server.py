r"""
LIVE WEB UI: pick a video, press Start, and the metrics fill in while ICP is computing.

    python demo/app/live_server.py            -> opens http://127.0.0.1:8765/

Videos (sequences) come from sequences.py: the ICL-NUIM clip, every TUM folder in "RGB TUM", and anything you drop into
demo/sequences/<name> (TUM format: rgb.txt, depth.txt, groundtruth.txt, rgb/, depth/). Ground truth is mapped automatically.

Modes (buttons on the page):
  Live    Runs ICP live for no cleaning, median 9x9 and Fu et al. (depth frames cleaned in advance on Kaggle).
  Replay  Plays back the saved Fu et al. poses for ICL.

Metrics per frame and method: position error and its RMS after a rigid fit to ground truth (cm),
and step error (RPE, mm).
"""
import argparse
import os
import sys
import threading
import time
import webbrowser
from multiprocessing import Pipe, Process

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.dirname(HERE)
ROOT = os.path.dirname(DEMO)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "4")

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory

import icl_compare as C
import sequences as SQ
import tracker as TR

app = Flask(__name__)
LOCK = threading.Lock()
IMG = {}                                                   # (name, k) -> jpeg bytes
S = dict(mode="idle", running=False, stop=False, done=False, error=None, start=0, n=0, k=-1, rows=[], est=[],
         truth=[], note="", frames=[], seq="icl", seq_name="", methods=[], colors=[])
VIDEOS = os.path.join(ROOT, "videos")
SAVED_RUN = os.path.join(DEMO, "data", "results", "run_icl_303.npz")
REPLAY_START = 303


def jpeg(img, q=80):
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()


def depth_jpeg(d, lo, hi):
    n = np.clip((np.nan_to_num(d, nan=lo) - lo) / (hi - lo), 0, 1)
    img = cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    img[np.isnan(d) | (d <= 0)] = 0
    return jpeg(cv2.resize(img, (640, 480), interpolation=cv2.INTER_NEAREST))


def run(seq, mode, frames, speed, fast):
    """Background thread: one clip, frame by frame."""
    workers = []
    try:
        methods = seq.methods()
        n, M = len(frames), len(methods)
        truth = seq.truth(frames)
        rel = [np.linalg.inv(truth[0]) @ t for t in truth]
        truth_positions = np.array(rel)[:, :3, 3]
        with LOCK:
            S["truth"] = [[float(r[0, 3]), float(r[2, 3])] for r in rel]
        saved = saved_aligned = None
        if mode == "replay":
            with np.load(SAVED_RUN, allow_pickle=True) as run_data:
                saved = run_data["poses"]
                saved_aligned = run_data["aligned"]
        indices = [SQ.ALL_METHODS.index(method) for method in methods]
        if mode == "live":
            for _, slug, _ in methods:
                parent, child = Pipe(duplex=False)
                process = Process(target=TR.worker, args=(seq.id, slug, frames, child, fast), daemon=True)
                process.start()
                child.close()
                workers.append((process, parent))
        rpe2 = [0.0] * M
        positions = np.empty((M, n, 3))
        prev, lo_hi = None, None
        for k, i in enumerate(frames):
            if S["stop"]:
                break
            t0 = time.time()
            if mode == "live":
                poses = []
                for _, conn in workers:
                    while not S["stop"] and not conn.poll(0.1):
                        pass
                    if S["stop"]:
                        return
                    poses.append(conn.recv())
                bad = [p for p in poses if isinstance(p, Exception)]
                if bad:
                    raise bad[0]
            else:
                poses = [saved[m, k].astype(np.float64) for m in indices]
                time.sleep(speed)
            raw = seq.raw(i)
            if lo_hi is None:
                valid = raw[np.isfinite(raw) & (raw > 0)]
                lo, hi = np.percentile(valid, [2, 98]) if valid.size else (0.0, 1.0)
                lo_hi = lo, max(hi, lo + 1e-6)
            img = {"rgb": jpeg(seq.rgb(i)), "raw": depth_jpeg(raw, *lo_hi)}
            row = dict(frame=i, err=[], ate=[], rpe=[], rpe_rms=[])
            for m in range(M):
                positions[m, k] = poses[m][:3, 3]
                if k > 0:
                    est = np.linalg.inv(prev[m]) @ poses[m]
                    tru = np.linalg.inv(rel[k - 1]) @ rel[k]
                    r = float(np.linalg.norm(est[:3, 3] - tru[:3, 3]) * 1000)
                    rpe2[m] += r * r
                    row["rpe"].append(r)
                    row["rpe_rms"].append((rpe2[m] / k) ** 0.5)
                else:
                    row["rpe"].append(None)
                    row["rpe_rms"].append(None)
            prev = poses
            prefix_truth = truth_positions[:k + 1]
            aligned = (saved_aligned[indices, :k + 1] if mode == "replay" else
                       np.array([C.align_positions(path[:k + 1], prefix_truth) for path in positions]))
            errors = np.linalg.norm(aligned - prefix_truth, axis=2) * 100
            row["ate"] = np.sqrt(np.mean(errors ** 2, axis=1)).tolist()
            row["sec"] = time.time() - t0
            with LOCK:
                for name, b in img.items():
                    IMG[(name, k)] = b
                S["rows"].append(row)
                for sample, error in zip(S["rows"], errors.T):
                    sample["err"] = error.tolist()
                S["est"] = aligned[:, :, [0, 2]].tolist()
                S["k"] = k
    except Exception as e:                                # show the problem on the page instead of dying silently
        S["error"] = f"{type(e).__name__}: {e}"
    finally:
        for process, conn in workers:
            conn.close()
            if process.is_alive():
                process.terminate()
            process.join()
        S["running"], S["done"] = False, True


@app.route("/")
def index():
    return send_from_directory(HERE, "live_ui.html")


@app.route("/videos/<path:name>")
def videos(name):
    return send_from_directory(VIDEOS, name)                # D:\MIni Project\videos


@app.route("/<path:name>")
def files(name):
    if name.lower().endswith((".html", ".mp4", ".png", ".jpg", ".pdf")):
        return send_from_directory(DEMO, name)
    return ("not found", 404)


@app.route("/api/sequences")
def sequences():
    return jsonify([s.info() for s in SQ.refresh()])


@app.route("/api/status")
def status():
    with LOCK:
        return jsonify({k: v for k, v in S.items()})


@app.route("/api/img/<name>/<int:k>")
def image(name, k):
    b = IMG.get((name, k))
    return Response(b, mimetype="image/jpeg", headers={"Cache-Control": "no-store"}) if b else ("", 404)


@app.route("/api/start", methods=["POST"])
def start():
    if S["running"]:
        return jsonify(ok=False, error="a run is already going")
    body = request.get_json(force=True)
    mode = body.get("mode", "live")
    sid = body.get("seq", "icl")
    try:
        seq = SQ.get(sid)
    except KeyError:
        SQ.refresh()
        seq = SQ.get(sid)
    if mode == "replay" and seq.id != "icl":
        return jsonify(ok=False, error="Choose the full ICL living room sequence for instant replay.")
    step = seq.step
    st = int(body.get("start", seq.first))
    n = max(2, min(int(body.get("frames", 30)), (seq.last - seq.first) // step + 1))
    note = seq.note
    if mode == "replay":
        if st != REPLAY_START:
            note = f"Replay always starts at frame {REPLAY_START}."
        st, n = REPLAY_START, min(n, 295)
    frames = [st + k * step for k in range(n)]
    if st < seq.first or frames[-1] > seq.last:
        every = f" (every {step} frames)" if step > 1 else ""
        return jsonify(ok=False, error=f"for {n} frames{every} start between {seq.first} and {seq.last - (n - 1) * step}; "
                                       f"this video ends at frame {seq.last}")
    missing = seq.missing_fu_frame(frames)
    if missing is not None:
        return jsonify(ok=False, error=f"Frame {missing} needs Fu et al. depth preparation before it can run.")
    IMG.clear()
    info = seq.info()
    with LOCK:
        S.update(mode=mode, running=True, stop=False, done=False, error=None, start=st, n=n, k=-1, rows=[],
                 est=[[] for _ in info["methods"]], truth=[], note=note, frames=frames, seq=seq.id, seq_name=seq.name,
                 methods=info["methods"], colors=info["colors"])
    threading.Thread(target=run, args=(seq, mode, frames, float(body.get("speed", 0.1)), bool(body.get("fast", False))),
                     daemon=True).start()
    return jsonify(ok=True)


@app.route("/api/stop", methods=["POST"])
def stop():
    S["stop"] = True
    return jsonify(ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Live UI on {url}   (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    app.run(host="127.0.0.1", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
