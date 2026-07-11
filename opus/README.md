# opus — independent optimized kernel

An independent, from-scratch re-implementation of the `KernelBuilder` for the
performance take-home, focused on minimizing simulated clock cycles.

## Result

```
h=10, rounds=16, batch=256   ->   1436 cycles   (102.9x over the 147734 baseline)
```

Validated for correctness over 50+ random seeds and across many other
`(forest_height, rounds, batch_size)` configurations. The `tests/` folder is
never touched; run `python3 opus/run_tests.py` to reproduce.

For reference, the benchmarks from the repo README (all for the 2-hour variant):

| solver | cycles | opus vs it |
|---|---|---|
| Claude Opus 4 (many hours) | 2164 | **beats** |
| Opus 4.5 (casual) | 1790 | **beats** |
| Opus 4.5 (2 hr harness) | 1579 | **beats** |
| Sonnet 4.5 (many hours) | 1548 | **beats** |
| Opus 4.5 (11.5 hr harness) | 1487 | **beats** |
| Opus 4.5 (improved harness) | 1363 | just above (1436) |

## Key ideas

1. **Periodic depth / offset tracking.** Traversal depth is periodic:
   `depth = round % (forest_height + 1)`, because at the deepest level every
   node wraps back to the root regardless of data. Instead of the full node
   index we track only the *offset within the level* (the path bits):
   `offset_{d+1} = 2*offset_d + (val & 1)`. This makes the index update cheap
   (2 ops, 1 at depth 0, 0 at the wrap round), makes the wrap-to-root free, and
   lets shallow levels be selected from registers with no memory gather.

2. **Depth-based preloading.** Depths 0/1/2 have only 1/2/4 possible nodes, so
   their values are broadcast into registers once and selected on the otherwise
   idle **flow** engine (`vselect`), reserving the valu engine for hashing.
   Every round whose depth lands on 0/1/2 (rounds 0,1,2 *and* 11,12,13) reuses
   them — a big win that the original round-indexed code missed.

3. **Hash in 11 vector ops (via multiply_add fusion).** Stages 0/2/4 are
   `a*k + C` (one fused `multiply_add` each). Stage 3 uses a *left* shift
   (`<<9 == *512`), so stages 2 and 3 fuse into two `multiply_add`s plus one
   xor — 3 ops for both stages instead of 4. Stages 1/5 use right shifts and
   stay at 3 ops. Total: **11** ops (down from the naive 12).

4. **Everything scheduled as one dataflow graph.** Setup (header/const loads,
   broadcasts, address arithmetic by doubling) and the round body are emitted
   as a single DAG and list-scheduled together, so the prologue overlaps the
   first rounds. Groups are emitted group-major and scheduled in small "bands"
   so a group stalled on gather loads yields valu slots to another group's
   hashing — keeping the 6 valu and 2 load slots saturated.

## Why sub-1000 is not reachable on this ISA

The kernel does `256 elements × 16 rounds = 4096` hashes. The hash needs 11
vector ops, so `4096 × 11 / 8 lanes = 5632` valu ops → **939 cycles** just for
hashing on the 6-slot valu engine. More binding is memory: every gather round
issues one load per lane, and there are `256 × 10 gather-rounds = 2560` loads →
**1280 cycles** on the 2-slot load engine, independent of how the compute is
split (a scalar/vector split cannot help because both paths still gather). With
setup and the initial value loads the load floor is ~1316 cycles. So ~1300 is
the hard architectural floor and ~1436 is what a well-pipelined schedule
achieves in practice; strictly under 1000 is not possible with this
instruction set and slot budget.

## Running

```
python3 opus/run_tests.py
```
