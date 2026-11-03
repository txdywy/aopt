#!/bin/bash
cd "$(dirname "$0")/results" && export PYTHONPATH="$(cd .. && pwd)"
run() { echo "=== $(date +%T) $* ==="; "$@" 2>&1 | grep --line-buffered -E "NOTE|SKIP|^todo|variable-shift|mode|ALL DONE|RESIDV|Traceback|Error"; }
run nice -n 10 python3 ../drive_v2.py cv_c8.jsonl 8 30 2 1.5
run nice -n 10 python3 ../resid_v.py cvf_res.jsonl 120 2
run nice -n 10 python3 ../drive_g2.py kescape cg_kescape.jsonl 8 60 2 1.5
echo "VCHAIN5 STAGE1 DONE $(date +%T)"
