@echo off
REM LIVE DEMO: starts the local server and opens http://127.0.0.1:8765/  (press "Start live run").
REM Tracks no cleaning, median and Fu et al. with ICP, about 0.5 s per frame. "Instant replay" is the saved fallback.
REM Close this window to stop. If Anaconda is not in D:\Anaconda, change the next line.
cd /d "%~dp0.."
call D:\Anaconda\Scripts\activate.bat kinectlite
python demo\app\live_server.py
pause
