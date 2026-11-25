#!/bin/bash
# LIVE DEMO on Mac/Linux: starts the local server and opens http://127.0.0.1:8765/
# First time only:  chmod +x demo/run_live_ui.sh
# Uses the python of the active environment (venv or conda). Close the terminal or press Ctrl+C to stop.
cd "$(dirname "$0")/.." || exit 1
python3 demo/app/live_server.py
