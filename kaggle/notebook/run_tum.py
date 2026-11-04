# Kaggle: TUM fr3 long_office_household, paper pipeline vs median filter, ICP against groundtruth.txt.
import glob
import os
import shutil
import subprocess
import sys
import zipfile

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "open3d"], check=True)

# Code: copy into a writable folder (the cache and results are written next to it)
src = os.path.dirname(glob.glob("/kaggle/input/**/tum_compare.py", recursive=True)[0])
shutil.copytree(src, "/kaggle/working/code/src", dirs_exist_ok=True)

# Data: the folder holding groundtruth.txt. depth/ and rgb/ may arrive as depth.zip / rgb.zip: unpack them if so.
tum = os.path.dirname(glob.glob("/kaggle/input/**/groundtruth.txt", recursive=True)[0])
if not os.path.isdir(os.path.join(tum, "depth")):
    work = "/kaggle/working/tum"
    shutil.copytree(tum, work, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.zip"))
    for z in glob.glob(os.path.join(tum, "*.zip")):
        with zipfile.ZipFile(z) as f:
            name = os.path.splitext(os.path.basename(z))[0]
            inside = f.namelist()[0].startswith(name + "/")
            f.extractall(work if inside else os.path.join(work, name))
    tum = work
print("TUM_DIR =", tum, "| depth frames:", len(os.listdir(os.path.join(tum, "depth"))), flush=True)

subprocess.run([sys.executable, "src/tum_compare.py", "--start", "0", "--frames", "300", "--step", "2",
                "--jobs", str(os.cpu_count())],
               cwd="/kaggle/working/code", env=dict(os.environ, TUM_DIR=tum), check=True)

shutil.copytree("/kaggle/working/code/src/results", "/kaggle/working/results", dirs_exist_ok=True)
print(open("/kaggle/working/results/tum_compare.txt").read())
