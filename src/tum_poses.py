r"""True camera poses of the TUM fr3 sequence (groundtruth.txt), one per depth frame."""
import os

import numpy as np
from scipy.spatial.transform import Rotation

import data as D


def read_list(name):
    """(timestamp, value) rows of a TUM list file such as depth.txt or groundtruth.txt."""
    rows = [line.split() for line in open(os.path.join(D.TUM_DIR, name)) if line[0] != "#"]
    return [(float(r[0]), r[1:]) for r in rows]


def closest(rows, time):
    """The row whose timestamp is closest to `time`."""
    return min(rows, key=lambda r: abs(r[0] - time))


def true_poses(indices):
    """4x4 pose of the camera for each depth frame, from the ground-truth pose recorded closest in time."""
    truth = [(t, v) for t, v in read_list("groundtruth.txt")]
    depth_times = [t for t, _ in read_list("depth.txt")]

    poses = []
    for i in indices:
        x, y, z, qx, qy, qz, qw = map(float, closest(truth, depth_times[i])[1])
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        pose[:3, 3] = [x, y, z]
        poses.append(pose)
    return poses
