# Codex optimized kernel

This directory is an independent implementation of the scored
`(height=10, nodes=2047, batch=256, rounds=16)` workload. It leaves the root
solution and `tests/` untouched.

Measured with the frozen simulator:

- **1203 cycles**
- **122.80x** faster than the 147734-cycle baseline
- 1371 / 1536 scratch words
- all public speed thresholds pass, including `<1363`

For the emitted operation DAG, the tightest simple resource bound is 1067
cycles (load engine), leaving a 136-cycle scheduling/critical-path gap.

Key optimizations are full unrolling, a mirrored local-path representation,
an internal `value ^ 0xB55A4F09` hash representation, cached shallow tree
levels, in-place deep gathers, one-time tree-node transformation, scalar/SIMD
engine balancing, and dependency-aware cohort scheduling.

Run the unchanged public suite against this implementation:

```bash
python3 codex/run_submission_tests.py
```

Run the deterministic 8-seed verifier and print resource-slot counts:

```bash
python3 codex/verify.py
```

`tune_schedule.py` is the offline launch-phase tuner used to search scheduling
offsets. Kernel construction itself selects the tuned schedule deterministically.
