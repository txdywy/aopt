#!/bin/bash
cd "$(dirname "$0")/results" && export PYTHONPATH="$(cd .. && pwd)"
run() { echo "=== $(date +%T) $* ==="; "$@" 2>&1 | grep --line-buffered -E "NOTE|mode|ALL DONE|progress|Traceback|Error"; }
run nice -n 5 python3 ../drive_g2.py multi cg_multi.jsonl 8 10 1 1.5
echo "MCHAIN DONE $(date +%T)"
