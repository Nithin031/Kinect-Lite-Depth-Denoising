import os
import sys
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import data as D
import paper_pipeline as P
import icl_compare as C
from clean import gate, median_filter

FRAMES = list(range(101, 119, 2))   # 9 tracked frames, 8 consecutive pairs (step 2, as in the main experiment)
VOXEL = 0.02


def paper(i):
    lo, hi = P.window(D.ICL_N_FRAMES, i)
    return P.restore([D.icl_noisy(j) for j in range(lo, hi)], i - lo, center="paper")["restored"]


METHODS = {
    "true depth": lambda i: D.icl_gt(i),
    "raw (gate only)": lambda i: gate(D.icl_noisy(i)),
    "median 9x9": lambda i: median_filter(gate(D.icl_noisy(i))),
    "paper": paper,
}


def cloud(depth):
    pc = C._cloud(depth).voxel_down_sample(VOXEL)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * VOXEL, max_nn=30))
    pc.orient_normals_towards_camera_location(np.zeros(3))
    return pc


depth = {m: {i: f(i) for i in FRAMES} for m, f in METHODS.items()}
clouds = {m: {i: cloud(depth[m][i]) for i in FRAMES} for m in METHODS}
poses = C.true_poses(FRAMES)

print("A. Surface-normal error against normals of the true depth (degrees, same ICP normal settings)")
print(f"{'method':18s}{'median':>8s}{'mean':>8s}{'p90':>8s}   matched points")
for m in METHODS:
    if m == "true depth":
        continue
    ang, kept, tot = [], 0, 0
    for i in FRAMES:
        t, c = clouds["true depth"][i], clouds[m][i]
        tree = cKDTree(np.asarray(t.points))
        d, idx = tree.query(np.asarray(c.points))
        ok = d < 0.02
        n1 = np.asarray(c.normals)[ok]
        n2 = np.asarray(t.normals)[idx[ok]]
        ang.append(np.degrees(np.arccos(np.clip((n1 * n2).sum(1), -1, 1))))
        kept += ok.sum()
        tot += len(ok)
    a = np.concatenate(ang)
    print(f"{m:18s}{np.median(a):8.1f}{a.mean():8.1f}{np.percentile(a, 90):8.1f}   {100 * kept / tot:.0f}%")

print()
print("B. Do consecutive frames agree under the TRUE camera motion? Point-to-plane residual of frame j")
print("   (moved into frame i with the true pose) against frame i's surface. mm, matches within 5 cm.")
print(f"{'method':18s}{'median|r|':>10s}{'mean r':>9s}{'rms r':>8s}   (mean r = systematic bias)")
for m in METHODS:
    res = []
    for k in range(len(FRAMES) - 1):
        i, j = FRAMES[k], FRAMES[k + 1]
        rel = np.linalg.inv(poses[k]) @ poses[k + 1]
        ci, cj = clouds[m][i], clouds[m][j]
        pj = np.asarray(cj.points) @ rel[:3, :3].T + rel[:3, 3]
        d, idx = cKDTree(np.asarray(ci.points)).query(pj)
        ok = d < 0.05
        r = ((pj[ok] - np.asarray(ci.points)[idx[ok]]) * np.asarray(ci.normals)[idx[ok]]).sum(1) * 1000
        res.append(r)
    r = np.concatenate(res)
    print(f"{m:18s}{np.median(np.abs(r)):10.1f}{r.mean():9.2f}{np.sqrt((r ** 2).mean()):8.1f}")
