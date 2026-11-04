# Kaggle: hole-filling methods after the 9x9 median (experiments/hole_filling/compare_fills.py) on ICL and TUM.
import glob
import os
import shutil
import subprocess
import sys
import zipfile

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "open3d"], check=True)

PROJ, DATA = "/kaggle/working/proj", "/kaggle/working/data"

# Code: the pipeline into src/, the experiment into experiments/hole_filling/ (the layout the script expects)
src = os.path.dirname(glob.glob("/kaggle/input/**/compare_fills.py", recursive=True)[0])
os.makedirs(f"{PROJ}/src", exist_ok=True)
os.makedirs(f"{PROJ}/experiments/hole_filling", exist_ok=True)
for f in glob.glob(os.path.join(src, "*.py")):
    dst = "experiments/hole_filling" if os.path.basename(f) == "compare_fills.py" else "src"
    shutil.copy(f, f"{PROJ}/{dst}/")


def unpack(folder, target):
    """Copy a Kaggle input folder to target; folders uploaded zipped (<name>.zip) are extracted to <name>/."""
    shutil.copytree(folder, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.zip"))
    for z in glob.glob(os.path.join(folder, "*.zip")):
        name = os.path.splitext(os.path.basename(z))[0]
        with zipfile.ZipFile(z) as f:
            inside = f.namelist()[0].startswith(name + "/")
            f.extractall(target if inside else os.path.join(target, name))


# Data, in the layout data.py expects under KINECT_DATA
icl = os.path.dirname(glob.glob("/kaggle/input/**/lr_kt0_noisy*", recursive=True)[0])   # folder or .zip
unpack(icl, f"{DATA}/ICL NUIM")
if not os.path.exists(f"{DATA}/ICL NUIM/gt/livingRoom0.gt.freiburg"):          # gt uploaded loose, not in gt/
    os.makedirs(f"{DATA}/ICL NUIM/gt", exist_ok=True)
    shutil.copy(glob.glob(f"{DATA}/ICL NUIM/**/livingRoom0.gt.freiburg", recursive=True)[0], f"{DATA}/ICL NUIM/gt/")
tum = os.path.dirname(glob.glob("/kaggle/input/**/groundtruth.txt", recursive=True)[0])
unpack(tum, f"{DATA}/RGB TUM/rgbd_dataset_freiburg3_long_office_household")
for path in ("ICL NUIM/lr_kt0_noisy/depth", "ICL NUIM/lr_kt0_noisy/rgb", "ICL NUIM/lr_kt0_noisefree/depth",
             "RGB TUM/rgbd_dataset_freiburg3_long_office_household/depth"):
    print(path, len(os.listdir(f"{DATA}/{path}")), "files", flush=True)

subprocess.run([sys.executable, "experiments/hole_filling/compare_fills.py", "--frames", "300", "--step", "2",
                "--jobs", str(os.cpu_count())], cwd=PROJ, env=dict(os.environ, KINECT_DATA=DATA), check=True)

shutil.copytree(f"{PROJ}/experiments/hole_filling/results", "/kaggle/working/results", dirs_exist_ok=True)
shutil.rmtree(DATA)                                   # keep the notebook output small
print(open("/kaggle/working/results/fills.txt").read())
