# Codex optimized kernel

This directory is an independent implementation of the scored
`(height=10, nodes=2047, batch=256, rounds=16)` workload. It leaves the root
solution and `tests/` untouched.

Measured with the frozen simulator:

- **997 cycles** (identical across 64 deterministic seeds)
- **148.18x** faster than the 147734-cycle baseline
- 1535 / 1536 scratch words
- one cycle behind the latest public top-10 snapshot cutoff of 996

For the emitted operation DAG, the simple resource lower bounds are 965 ALU,
986 VALU, 965 load, 31 store, and 895 flow cycles.  The tightest bound is
therefore 986 cycles, leaving an 11-cycle scheduling/critical-path gap.

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
`repair_schedule.py`, `solve_tail.py`, and `solve_groups.py` provide progressively
larger CP-SAT neighborhoods for exact schedule compression.  `solve_tail.py`
also accepts comma-separated `SCALAR_FINAL_C5`, `SCALAR_FINAL_JOIN`,
`SCALAR_FINAL_SHIFT`, and `SCALAR_FINAL_HASH23_JOIN` environment variables for
testing resource-balanced epilogue rewrites.
