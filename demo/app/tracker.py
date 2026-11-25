r"""One ICP tracker process (used by live_server.py): coarse-to-fine point-to-plane ICP, frame to frame, pose chaining."""
import os
import queue
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "src"))
sys.path.insert(0, HERE)

import numpy as np
import open3d as o3d

import icl_compare as C
import sequences as SQ

SCALES = [(0.04, 0.10), (0.02, 0.05), (0.01, 0.02)]      # same coarse-to-fine ICP as the rest of the project
FAST_SCALES = SCALES[:2]                                  # fast mode: skip the finest (1 cm) level


def levels(depth, camera, scales):
    pc = C._cloud(depth, camera)
    out = []
    for voxel, _ in scales:
        down = pc.voxel_down_sample(voxel)
        down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * voxel, max_nn=30))
        out.append(down)
    return out


def worker(seq_id, slug, frames, conn, fast=False):
    """All frame numbers are sent up front; a feeder thread prepares the point clouds (normals are the slow part) for the
    coming frames while the main loop runs ICP, then sends back the camera pose of every frame in order."""
    seq = SQ.get(seq_id)
    camera = seq.camera()
    scales, iters = (FAST_SCALES, 15) if fast else (SCALES, 30)
    q = queue.Queue(maxsize=6)

    def feeder():
        for i in frames:
            try:
                q.put(levels(seq.cleaned(slug, i), camera, scales))
            except Exception as e:                        # tell the page what went wrong instead of hanging
                q.put(e)
                return
        q.put(None)

    threading.Thread(target=feeder, daemon=True).start()
    prev, pose, motion = None, np.eye(4), np.eye(4)
    while True:
        cur = q.get()
        if cur is None:
            break
        if isinstance(cur, Exception):
            conn.send(cur)
            break
        if prev is not None:
            guess = motion
            for (voxel, dist), src, dst in zip(scales, cur, prev):
                guess = o3d.pipelines.registration.registration_icp(
                    src, dst, dist, guess, C._estimator("plane", dist),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters)).transformation
            motion = guess
            pose = pose @ motion
        prev = cur
        conn.send(pose)
