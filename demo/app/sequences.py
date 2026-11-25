r"""
The videos (sequences) the live page can run, each with its ground truth mapped automatically.

Found automatically:
  ICL-NUIM living room   built in (frames 303-597 are pre-cleaned for all three methods in demo/data/cache/icl)
  RGB TUM/<folder>       TUM-format folders (rgb.txt, depth.txt, groundtruth.txt, rgb/, depth/)
  demo/sequences/<name>  Custom RGB-D clips in the same TUM format.

The live picker lists sequences with Fu et al. frames prepared in
demo/data/cache/<name>/paper__center_paper/<frame>.png.

Optional demo/sequences/<name>/sequence.json:  {"name": "My video", "fx": 535.4, "fy": 539.2, "cx": 320.1, "cy": 247.6, "step": 2}
Without it the camera is chosen from the folder name (freiburg1 / freiburg2 / freiburg3), else the fr3 camera.
"""
import bisect
import json
import os
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.dirname(HERE)
ROOT = os.path.dirname(DEMO)
sys.path.insert(0, os.path.join(ROOT, "src"))

import clean as CL
import data as D
import icl_compare as C

CACHE = os.path.join(DEMO, "data", "cache")
ALL_METHODS = [("No cleaning", "gate_only", "#8E98A3"),
               ("Median 9x9", "median_9x9", "#2C64A8"),
               ("Fu et al.", "paper__center_paper", "#D9692A")]
CAMERAS = {"freiburg1": (517.3, 516.5, 318.6, 255.3), "freiburg2": (520.9, 521.0, 325.1, 249.7),
           "freiburg3": (535.4, 539.2, 320.1, 247.6)}


def _rows(path):
    return [line.split() for line in open(path) if line.strip() and not line.startswith("#")]


class Seq:
    def __init__(self, sid, name, kind, root=None, cam=(481.20, 480.00, 319.50, 239.50), step=1, first=0, last=0, note=""):
        self.id, self.name, self.kind, self.root, self.cam, self.step = sid, name, kind, root, cam, step
        self.first, self.last, self.note = first, last, note
        self.cache = os.path.join(CACHE, sid)
        if kind == "tum":
            self.depth_rows = [(float(r[0]), r[1]) for r in _rows(os.path.join(root, "depth.txt"))]
            self.rgb_rows = [(float(r[0]), r[1]) for r in _rows(os.path.join(root, "rgb.txt"))]
            gt = [(float(r[0]), list(map(float, r[1:8]))) for r in _rows(os.path.join(root, "groundtruth.txt"))]
            self.gt_t, self.gt_v = [g[0] for g in gt], [g[1] for g in gt]
            self.rgb_t = [r[0] for r in self.rgb_rows]
            self.last = len(self.depth_rows) - 1

    # -- methods that exist for this sequence ----------------------------------------------------------------------------
    def methods(self):
        fu = os.path.join(self.cache, ALL_METHODS[2][1])
        if os.path.isdir(fu) and any(f.endswith(".png") for f in os.listdir(fu)):
            return list(ALL_METHODS)      # raw, median and Fu et al. all tracked live (frames are prepared in advance)
        return []

    def missing_fu_frame(self, frames):
        folder = os.path.join(self.cache, ALL_METHODS[2][1])
        return next((i for i in frames if not os.path.isfile(os.path.join(folder, f"{i}.png"))), None)

    def prepared_bounds(self):
        folder = os.path.join(self.cache, ALL_METHODS[2][1])
        if not os.path.isdir(folder):
            return None
        cached = sorted(
            int(name[:-4]) for name in os.listdir(folder)
            if name.endswith(".png") and name[:-4].isdecimal()
            and self.first <= int(name[:-4]) <= self.last
        )
        present = set(cached)
        longest = None
        for start in cached:
            if start - self.step in present:
                continue
            end = start
            while end + self.step in present:
                end += self.step
            if end > start and (longest is None or end - start > longest[1] - longest[0]):
                longest = start, end
        return longest

    # -- data ------------------------------------------------------------------------------------------------------------
    def raw(self, i):
        if self.kind == "icl":
            return D.icl_noisy(i)
        return D.png_to_m(D._read(os.path.join(self.root, self.depth_rows[i][1])))

    def gt_depth(self, i):
        return D.icl_gt(i) if self.kind == "icl" else None

    def rgb(self, i):
        if self.kind == "icl":
            return cv2.imread(os.path.join(ROOT, "ICL NUIM", "lr_kt0_noisy", "rgb", f"{i}.png"))
        t = self.depth_rows[i][0]
        j = bisect.bisect_left(self.rgb_t, t)
        j = min(range(max(0, j - 1), min(len(self.rgb_t), j + 1)), key=lambda q: abs(self.rgb_t[q] - t))
        return cv2.imread(os.path.join(self.root, self.rgb_rows[j][1]))

    def cleaned(self, slug, i):
        """Depth (metres, NaN = hole) of frame i for one method; filled from the cache when it is there."""
        p = os.path.join(self.cache, slug, f"{i}.png")
        if os.path.exists(p):
            return D.png_to_m(cv2.imread(p, cv2.IMREAD_UNCHANGED))
        if slug == "paper__center_paper":
            raise FileNotFoundError(f"Fu et al. frame {i} is not cleaned yet ({p})")
        gated = CL.gate(self.raw(i), 0.4)
        if slug == "gate_only":
            return gated
        out = CL.median_filter(gated, 9)                      # median 9x9, then cached so the next run is instant
        os.makedirs(os.path.dirname(p), exist_ok=True)
        cv2.imwrite(p, np.round(np.nan_to_num(out) * D.DEPTH_SCALE).clip(0, 65535).astype(np.uint16))
        return out

    def truth(self, frames):
        """4x4 camera poses of the frames from the ground truth (TUM: the pose recorded closest in time)."""
        if self.kind == "icl":
            return C.true_poses(frames)
        poses = []
        for i in frames:
            t = self.depth_rows[i][0]
            j = bisect.bisect_left(self.gt_t, t)
            j = min(range(max(0, j - 1), min(len(self.gt_t), j + 1)), key=lambda q: abs(self.gt_t[q] - t))
            x, y, z, qx, qy, qz, qw = self.gt_v[j]
            p = np.eye(4)
            p[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            p[:3, 3] = [x, y, z]
            poses.append(p)
        return poses

    def camera(self):
        import open3d as o3d
        fx, fy, cx, cy = self.cam
        return o3d.camera.PinholeCameraIntrinsic(640, 480, fx, fy, cx, cy)

    def info(self):
        first, last = self.prepared_bounds() or (self.first, self.last)
        return dict(id=self.id, name=self.name, kind=self.kind, first=first, last=last, step=self.step,
                    methods=[m[0] for m in self.methods()], colors=[m[2] for m in self.methods()],
                    has_depth_truth=self.kind == "icl", note=self.note)


def discover():
    seqs = [Seq("icl", "ICL living room · Full sequence", "icl", first=303, last=597)]
    for first, last, title in ((450, 469, "Tracking demo"), (303, 342, "Early pass"), (550, 589, "Late pass")):
        clip = Seq(f"icl_{first}", f"ICL living room · {title}", "icl", first=first, last=last)
        clip.cache = seqs[0].cache
        seqs.append(clip)
    for base, tag in ((os.path.join(ROOT, "RGB TUM"), "TUM"), (os.path.join(DEMO, "sequences"), "Mine")):
        if not os.path.isdir(base):
            continue
        for d in sorted(os.listdir(base)):
            root = os.path.join(base, d)
            if not all(os.path.exists(os.path.join(root, f)) for f in ("rgb.txt", "depth.txt", "groundtruth.txt")):
                continue
            meta = {}
            if os.path.exists(os.path.join(root, "sequence.json")):
                meta = json.load(open(os.path.join(root, "sequence.json")))
            cam = next((v for k, v in CAMERAS.items() if k in d), CAMERAS["freiburg3"])
            cam = (meta.get("fx", cam[0]), meta.get("fy", cam[1]), meta.get("cx", cam[2]), meta.get("cy", cam[3]))
            name = meta.get("name") or f"{tag}: " + d.replace("rgbd_dataset_", "")
            s = Seq(d, name, "tum", root=root, cam=cam, step=int(meta.get("step", 2)),
                    note="Ground truth is mapped from groundtruth.txt by time.")
            if "freiburg3_long_office" in d:
                s.first = 302                                  # frames 0-299 were used in the report
            seqs.append(s)
    return seqs


_CACHE = {}


def get(sid):
    if not _CACHE:
        _CACHE.update({s.id: s for s in discover()})
    return _CACHE[sid]


def refresh():
    seqs = discover()
    _CACHE.update({s.id: s for s in seqs})
    return [s for s in seqs if s.prepared_bounds() is not None]
