# Kaggle: ICP variants (point-to-point, point-to-plane, robust point-to-plane, generalized ICP) on cleaned depth,
# ICL and TUM (src/icp_variants.py). Depth only.
import glob
import os
import shutil
import subprocess
import sys
import zipfile

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "open3d"], check=True)

PROJ, DATA = "/kaggle/working/proj", "/kaggle/working/data"

src = os.path.dirname(glob.glob("/kaggle/input/**/icp_variants.py", recursive=True)[0])
os.makedirs(f"{PROJ}/src", exist_ok=True)
for f in glob.glob(os.path.join(src, "*.py")):
    shutil.copy(f, f"{PROJ}/src/")


def unpack(folder, target):
    """Copy a Kaggle input folder to target; folders uploaded zipped (<name>.zip) are extracted to <name>/."""
    shutil.copytree(folder, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.zip"))
    for z in glob.glob(os.path.join(folder, "*.zip")):
        name = os.path.splitext(os.path.basename(z))[0]
        with zipfile.ZipFile(z) as f:
            inside = f.namelist()[0].startswith(name + "/")
            f.extractall(target if inside else os.path.join(target, name))


icl = os.path.dirname(glob.glob("/kaggle/input/**/lr_kt0_noisy*", recursive=True)[0])
unpack(icl, f"{DATA}/ICL NUIM")
if not os.path.exists(f"{DATA}/ICL NUIM/gt/livingRoom0.gt.freiburg"):
    os.makedirs(f"{DATA}/ICL NUIM/gt", exist_ok=True)
    shutil.copy(glob.glob(f"{DATA}/ICL NUIM/**/livingRoom0.gt.freiburg", recursive=True)[0], f"{DATA}/ICL NUIM/gt/")
tum = os.path.dirname(glob.glob("/kaggle/input/**/groundtruth.txt", recursive=True)[0])
unpack(tum, f"{DATA}/RGB TUM/rgbd_dataset_freiburg3_long_office_household")
for path in ("ICL NUIM/lr_kt0_noisy/depth", "ICL NUIM/lr_kt0_noisefree/depth",
             "RGB TUM/rgbd_dataset_freiburg3_long_office_household/depth"):
    print(path, len(os.listdir(f"{DATA}/{path}")), "files", flush=True)

subprocess.run([sys.executable, "src/icp_variants.py", "--dataset", "both", "--frames", "300", "--step", "2",
                "--jobs", str(os.cpu_count())], cwd=PROJ, env=dict(os.environ, KINECT_DATA=DATA), check=True)

shutil.copytree(f"{PROJ}/src/results", "/kaggle/working/results", dirs_exist_ok=True)
shutil.rmtree(DATA)
shutil.rmtree(f"{PROJ}/cache", ignore_errors=True)    # keep the notebook output small
print(open("/kaggle/working/results/icp_variants.txt").read())
