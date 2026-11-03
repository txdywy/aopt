#!/bin/bash
# Final residual stage, run after the main chains: every residual gets the strongest
# exact treatment (case splits / fixed shift literals / escalation), then coverage.
cd "$(dirname "$0")/results" && export PYTHONPATH="$(cd .. && pwd)"
run() { echo "=== $(date +%T) $* ==="; "$@" 2>&1 | grep --line-buffered -vE "^\s*$"; }
run python3 ../resid_c.py cc_res.jsonl 300
run python3 ../resid_m.py cs_m2_s1_31.jsonl cm2_res.jsonl 60
run python3 ../resid_m.py cs_m3_s1_31.jsonl cm3_res.jsonl 60
run python3 ../resid_m.py cg_kescape.jsonl ck_res.jsonl 60
run python3 ../resid_m.py cg_multi.jsonl cm_res.jsonl 120
run python3 ../resid_v.py cvf_res.jsonl 300 2
run python3 ../coverage2.py
echo "FINAL DONE $(date +%T)"
