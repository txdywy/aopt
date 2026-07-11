# opus — independent optimized kernel

An independent, from-scratch re-implementation of the `KernelBuilder` for the
performance take-home, focused on minimizing simulated clock cycles.

## Result

```
h=10, rounds=16, batch=256   ->   1285 cycles   (114.97x over the 147734 baseline)
```

Validated for correctness over 100+ random seeds and across many other
`(forest_height, rounds, batch_size)` configurations. The `tests/` folder is
never touched; run `python3 opus/run_tests.py` to reproduce.

Benchmarks from the repo README (2-hour variant):

| solver | cycles | opus vs it |
|---|---|---|
| Claude Opus 4 (many hours) | 2164 | **beats** |
| Opus 4.5 (casual) | 1790 | **beats** |
| Opus 4.5 (2 hr harness) | 1579 | **beats** |
| Sonnet 4.5 (many hours) | 1548 | **beats** |
| Opus 4.5 (11.5 hr harness) | 1487 | **beats** |
| Opus 4.5 (improved harness) | 1363 | **beats** |
| GPT-5.6 | 1158 | not beaten (see analysis) |

## Key ideas

1. **Periodic depth / offset tracking.** Traversal depth is periodic:
   `depth = round % (forest_height + 1)`, because at the deepest level every
   node wraps to the root regardless of data. We track only the *offset within
   the level* (path bits): `offset_{d+1} = 2*offset_d + (val & 1)`. This makes
   the index update cheap (2 ops; 1 at depth 0; 0 at the wrap round), makes the
   wrap free, and lets shallow levels be selected from registers with no gather.

2. **Depth preloading (P=4).** Depths 0/1/2/3 (1/2/4/8 nodes) are broadcast into
   registers once and selected on the otherwise-idle FLOW engine (`vselect`),
   with `off` kept intact so it also feeds the offset update. This cuts the
   gather rounds from 10 to 8, moving the bottleneck off the load engine.

3. **11-op hash via multiply_add fusion.** Stages 0/2/4 are `a*k + C` (one fused
   op). Stage 3's `<<9` is `*512`, so stages 2+3 fuse into two multiply_adds
   plus one xor — 3 ops for both (vs 4). Stages 1/5 use right shifts (3 ops).

4. **Dual-engine execution.** The valu (vector) engine is the bottleneck while
   the scalar ALU engine (12 slots) is otherwise idle. So ~24 of the 256
   elements are peeled off and run *entirely on the ALU engine* in parallel with
   the vector groups, using the same hash/offset/gather logic in scalar ops.

5. **Whole-program list scheduling** packs setup + both engine streams into VLIW
   bundles, group-major with band racing for the vector stream.

## Why beating 1158 is hard on this ISA

Cost floor per engine (elements=256, rounds=16, hash=11 vector ops):

- **Hashing alone** = `256*16*11/8 = 5632` valu ops → **939 cycles** on the
  6-slot valu engine.
- **Loads**: each gather round issues one load/element. With depth-4 preload the
  vector path gathers 8 rounds; total gathers plus the scalar path's put the
  **load engine (2 slots) floor around 1098 cycles**.

A pure-vector kernel is valu-bound at ~1265. Splitting work onto the scalar ALU
lowers the valu pressure, but the shared **2-slot load engine** then becomes the
wall (~1098 floor). At the balanced split the greedy list scheduler cannot keep
valu + alu + load all saturated (scalar hash chains stall waiting for load
slots), so the achieved cycle count (1285) sits above the ~1098 floor. Reaching
GPT-5.6's 1158 would require a scheduler that keeps all three engines saturated
simultaneously at the balanced operating point (essentially software pipelining
/ modulo scheduling), which the current greedy scheduler does not achieve.

## Running

```
python3 opus/run_tests.py                 # default (NSCALAR auto)
NSCALAR=0 python3 opus/run_tests.py       # pure-vector variant (~1391)
```
