KINECT LITE - LIVE DEMO

To run:
1. Double-click run_live_ui.bat
2. The browser opens the demo page (http://127.0.0.1:8765/)
3. Choose ICL living room, start frame 450, 60 frames, Accurate
4. Click "Start live run"

On Mac: activate the Python environment, then run  ./demo/run_live_ui.sh  (first time: chmod +x demo/run_live_ui.sh)
or just  python3 demo/app/live_server.py   (full setup steps are in the main README.md)

To stop, close the black window (Mac: Ctrl+C).

What you see:
Camera tracking (ICP) running live on three kinds of depth: no cleaning, median filter, and Fu et al.
It shows the camera path, the error per frame, ATE (cm) and RPE (mm).
It runs at about 2 frames per second.

The depth cleaning is NOT done live. Fu et al. takes about a minute per frame,
so the frames were cleaned earlier on Kaggle and saved in data/cache/icl (frames 303 to 597).

If the live run fails:
- Click "Instant replay" on the page (plays a saved run), or
- Double-click open_dashboard.bat to play the video.

Files:
- run_live_ui.bat       starts the live demo
- open_dashboard.bat    opens the video page
- app/                  the code (live_server.py is the main one)
- data/cache/icl/       cleaned frames
- data/results/         saved run used by Instant replay
- sequences/            put extra clips here (TUM format)
- app/live_stream.py and app/dashboard.py are an old version, not used

Needs these folders next to this one:
- datasets (ICL NUIM and RGB TUM inside it), src (code), videos (video_demo_full.mp4)

Python: Anaconda environment "kinectlite" in D:\Anaconda.
If Anaconda is somewhere else, change that line in run_live_ui.bat.
