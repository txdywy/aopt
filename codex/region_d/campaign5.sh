#!/bin/bash
cd "$(dirname "$0")/results" && export PYTHONPATH="$(cd .. && pwd)"
run() { echo "=== $(date +%T) $* ==="; "$@" 2>&1 | grep --line-buffered -E "NOTE|^total|^pos|ALL DONE|RESID|^residuals|Traceback|Error"; }
run python3 ../drive_h2.py 1 ch_p1.jsonl 1.5 60 3
run python3 ../drive_h2.py 2 ch_p2.jsonl 1.5 60 3
run python3 ../drive.py cs_m2_s1_31.jsonl 1 31 8 30 3 2 -
run python3 ../drive.py cs_m3_s1_31.jsonl 1 31 8 30 3 3 -
echo "CAMPAIGN5 DONE $(date +%T)"
