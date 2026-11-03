# Anthropic Performance Take-Home — 864 cycles

**English** | [中文](README.zh-CN.md)

This repository records a long-running optimization of Anthropic's original performance take-home. The task is a parallel tree-traversal and hash kernel that runs on a VLIW SIMD simulator.

| Metric | Value |
|---|---|
| **Best verified result** | **864 cycles** |
| Speedup over the repo baseline (147,734 cycles) | **170.99×** |
| Speedup over the 2-hour starting point (18,532 cycles) | 21.4× |
| Official tests | `tests/submission_tests.py`: 9/9 pass, all at 864 |
| Extended validation | 270/270 cases: 256 full-32-bit random, 8 ordinary, 6 boundary. Each case runs after a JSON round trip of the instructions. |
| `tests/`, `problem.py` | Byte-identical to the original repo |

The current submission is [`perf_takehome.py`](perf_takehome.py) at the root. [`codex/perf_takehome_864.py`](codex/perf_takehome_864.py) is a byte-identical standalone copy. About two weeks of sub-864 research followed, across dozens of passes. **None of it produced an 863-cycle program.** That work gives a resource lower-bound argument for 864 within the current algorithm family ([below](#why-it-stops-at-864)). It is not a proof that 864 is optimal for every possible algorithm.

## Quick verification

Requires Python 3.10+. The records used 3.12; 3.13 has also been checked.

```bash
# tests/ and problem.py must match the original (expect no output)
git diff 4363da3 -- tests/ problem.py
# Official suite: expect "CYCLES:  864" nine times
python3 tests/submission_tests.py
# Independent check: instruction digest, architectural writes, memory bounds, final outputs
python3 -m codex.verify_compact
```

`verify_compact` should report the compact instruction-JSON SHA256 `3f4cc919d7c5b05d105b5e41ba2ea3502020a03ec1318388a1138718abf2d4f1`.

The submission embeds a compressed instruction stream that does not depend on the runtime tree or input values. Every tree transformation, load and hash runs on the original frozen simulator. The kernel is specialized to `(height=10, nodes=2047, batch=256, rounds=16)` with all initial indices zero. It follows the official test contract, which checks only the final **values**; indices are not written out.

## Score history

| Cycles | Key change | Archive |
|---:|---|---|
| 147,734 | Original repo baseline | — |
| 1,939 → 1,600 | Vectorization, setup optimization, group-staggered scheduling | git history |
| 1,436 → 1,285 | Independent `opus/` rewrite: periodic depth tracking, shallow-level preload + `vselect`, vector/scalar dual engine | [`opus/`](opus/README.md) |
| 1,023 | Extreme-kernel experiment branch | [`codex/perf_takehome_1023.py`](codex/perf_takehome_1023.py) |
| 995 → 979 | Branch-dispatched tails, compact setup, constant staging | `codex/perf_takehome_98x.py`, [`_981_portable`](codex/perf_takehome_981_portable.py) |
| 880 | New instruction compiler, word-SSA register allocator, mixed scalar/vector scheduler | [`codex/perf_takehome_880.py`](codex/perf_takehome_880.py) |
| 868 → 866 | Reversed, XOR-adjusted four-word records; bilinear addressing; anchor banks | [`_868`](codex/perf_takehome_868.py), [`_866`](codex/perf_takehome_866.py) |
| 865 | Three-word FLOW broadcasts, STORE/reload broadcasts | [`codex/perf_takehome_865.py`](codex/perf_takehome_865.py) |
| **864** | Constant-4 staging, scalar deepest-leaf loads, raw-hash reassociation, joint compute and memory scheduling (CP-SAT) | [`codex/perf_takehome_864.py`](codex/perf_takehome_864.py) |

See the [864 result note](codex/SUB864_864_RESULT_20261023.md) for reproduction steps and [`codex/README.md`](codex/README.md) for earlier notes.

## Main techniques in the 864 kernel

- **Hash in 10 arithmetic ops per round.** Each `+`/`<<` stage pair becomes one `multiply_add`. Stages 2 and 3 are reassociated into two multiply-adds from the same source, followed by an XOR. The final-stage constant C5 is deferred to the next round: it is pre-XORed into the d0–d8 node values (511 lanes) and restored per element at d9/d10 and at the output. With the node XOR, each round costs 11 ops.
- **Reversed, XOR-adjusted four-word records.** Each record holds a node and its two children, so one gather covers two tree levels. A locally searched 224-word layout lets the d3 and d5 records share bilinear address arithmetic. The d5 records also carry a tree-block pointer (the anchor), which removes later address work.
- **Overlapping scratch windows and `vselect`** reuse already-loaded children. The shallow d0–d2 nodes stay resident in scratch and are selected on the FLOW engine.
- **Reuse of consumed memory.** Once read, the input, header, tree and index regions serve as temporary record and selection banks.
- **Scheduling.** A custom word-SSA allocator and a mixed scalar/vector scheduler build the program. CP-SAT then solves the compute and memory calendar jointly, and an exact suffix solve finishes it.

The readable compiler and scheduler are in `codex/`: `stride4_family.py`, `research_sub864.py`, `research_stage864.py`, `word_ssa.py`, `packet_modes.py` and `packet_mode_search.cpp`. The verified 864 calendar is saved in `codex/stage_astra_constant4_864_schedule.txt`.

## Why it stops at 864

Per-packet resource ledger of the 864 program:

```
Capacity 864 × 60 lanes = 51,840     Arithmetic work C = 51,665 lanes
Idle = head 52 + middle 7 + tail 116 = 175
LOAD 1,688 (saturated before its deadline)  FLOW 851 (about 2 spare)  STORE 530 (plenty spare)
```

- **Head waste ≥ 52.** Scratch is all zero at cycle 0, so the only useful work possible there is one all-ones vector.
- **Tail waste ≥ 116.** The last cycle can only STORE, and the two cycles before it can only feed those stores.
- **Required cut.** 863 cycles needs `C ≤ 863×60 − 168 = 51,612`, so at least **53 more lanes** must go (about 7 VALU ops).
- **Per-element work.** Hash 16×10, node XOR 16, parity 14, gather addresses 6, C5 restores 3. Each item has been shown locally optimal per depth or per opcode.
- **No profitable trade.** Every known trade costs at least 4 lanes per FLOW or LOAD slot freed, and buys back at most 1 lane. STORE is the only resource with spare capacity, but data written by STORE must be read back by LOAD, which is already saturated.

The full argument is in [`SUB863_LOWER_BOUND_ARGUMENT_20261027.md`](codex/SUB863_LOWER_BOUND_ARGUMENT_20261027.md) and [`SUB863_SHADOW_PRICES_20261027.md`](codex/SUB863_SHADOW_PRICES_20261027.md) (in Chinese).

### Remaining open routes

1. **Cut the per-round hash from 10 ops to 9.** One saved op is worth 4,096 lanes. This would need a new rewrite that uses right shifts, division or values carried across rounds, and lies outside the 3–4-instruction windows already searched exhaustively. The latest full-32-bit z3 re-run closed 46 more windows with 0 SAT; 26 still time out ([`SUB863_FULL32_RERUN_20261103.md`](codex/SUB863_FULL32_RERUN_20261103.md)).
2. **A new memory primitive** that turns spare STORE capacity into saved arithmetic without spending LOAD. None has been found in this ISA.

## Research index (`codex/SUB863_*`, mostly in Chinese)

| Topic | Main notes |
|---|---|
| Overall argument and resource prices | [lower bound](codex/SUB863_LOWER_BOUND_ARGUMENT_20261027.md), [shadow prices](codex/SUB863_SHADOW_PRICES_20261027.md), [structural audit](codex/SUB863_STRUCTURAL_AUDIT_20261023.md), [resource review](codex/SUB863_FRESH_PASS_20261027.md) |
| Fresh algorithm-family review | [first principles](codex/SUB863_FRESH_ALGO_20261103.md), [extra negative results](codex/SUB863_NEGATIVE_NOTES_20261103.md) |
| Hash head / C1 absorption | [head grammar](codex/SUB863_HEAD_GRAMMAR_20261028.md), [encoding rigidity](codex/SUB863_REENCODING_RIGIDITY_20261027.md), [affine module](codex/SUB863_AFFINE_MODULE_20261027.md), [FLOW sink](codex/SUB863_FLOW_SINK_20261027.md) |
| Hash middle / tail / C5 | [cross-stage middle](codex/SUB863_MIDDLE_CROSS_STAGE_20261023.md), [shift middle](codex/SUB863_SHIFT_COMPLETE_20261024.md), [C5 into middle](codex/SUB863_MIDDLE_C5_20261029.md), [C5 absorption](codex/SUB863_C5_ABSORPTION_20261028.md), [joint tail](codex/SUB863_JOINT_TAIL_20261024.md) |
| Division, comparison, runtime shifts | [division/comparison](codex/SUB863_DIVISION_COMPARISON_20261028.md), [single-division core](codex/SUB863_SINGLE_DIVISION_CORE_20261028.md), [multi-division](codex/SUB863_TWO_DIVISION_20261029.md), [runtime shift counts](codex/SUB863_RUNTIME_COUNT_20261029.md) |
| Round 0 and the root | [round-0 merge](codex/SUB863_ROUND0_MERGE_20261028.md), [pinned shifts](codex/SUB863_ROUND0_PINNED_20261028.md), [root boundary](codex/SUB863_ROOT_BOUNDARY_20261028.md) |
| Records, selection, liveness | [select-lane lemma](codex/SUB863_SELECT_LEMMA_20261027.md), [lane liveness](codex/SUB863_LIVENESS_20261027.md), [tail deferral](codex/SUB863_TAIL_DEFERRAL_20261027.md) |
| Graph construction and constants | [374-config sweep](codex/SUB863_CONFIG_SWEEP_20261027.md), [uniform chains](codex/SUB863_UNIFORM_CHAINS_20261027.md), [scalar uniforms](codex/SUB863_SCALAR_UNIFORMS_20261025.md) |
| Simulator semantics | [frozen simulator reread](codex/SUB863_SIMULATOR_REREAD_20261027.md) |

## Layout

| Path | Contents |
|---|---|
| `perf_takehome.py` | Current 864 submission; self-contained, needs no solver or data files |
| `problem.py`, `tests/` | Original simulator and frozen tests (unmodified) |
| `codex/` | 864 compiler, schedulers, solver scripts, every archived submission, and the research notes |
| `opus/` | Independent 1,285-cycle rewrite and its write-up |
| `anti/` | Early experimental variants |
| Root `temp_perf*.py`, `test_hang*.py`, etc. | Historical debugging scripts, unrelated to the current submission |

## Acknowledgements

Signed-parity bilinear addressing and stride-four records were informed by [Josu San Martin's public research log](https://github.com/josusanmartin/josusanmartin.github.io/blob/master/vliw/chat.log). [Sakauma's 880-cycle kernel](https://github.com/Sakauma/my_original_performance_takehome/blob/cb2e0554735ef7d0dfde1a04bc2c12c4bb2ed38d/perf_takehome.py) supplied the initial hash scheduling priorities. The final 864 program is a new, locally generated instruction stream; the original source notice is retained.

---

## Original take-home README

This repo contains a version of Anthropic's original performance take-home, before Claude Opus 4.5 started doing better than humans given only 2 hours.

The original take-home was a 4-hour one that starts close to the contents of this repo, after Claude Opus 4 beat most humans at that, it was updated to a 2-hour one which started with code which achieved 18532 cycles (7.97x faster than this repo starts you). This repo is based on the newer take-home which has a few more instructions and comes with better debugging tools, but has the starter code reverted to the slowest baseline. After Claude Opus 4.5 we started using a different base for our time-limited take-homes.

Now you can try to beat Claude Opus 4.5 given unlimited time!

## Performance benchmarks 

Measured in clock cycles from the simulated machine. All of these numbers are for models doing the 2 hour version which started at 18532 cycles:

- **2164 cycles**: Claude Opus 4 after many hours in the test-time compute harness
- **1790 cycles**: Claude Opus 4.5 in a casual Claude Code session, approximately matching the best human performance in 2 hours
- **1579 cycles**: Claude Opus 4.5 after 2 hours in our test-time compute harness
- **1548 cycles**: Claude Sonnet 4.5 after many more than 2 hours of test-time compute
- **1487 cycles**: Claude Opus 4.5 after 11.5 hours in the harness
- **1363 cycles**: Claude Opus 4.5 in an improved test time compute harness
- **??? cycles**: Best human performance ever is substantially better than the above, but we won't say how much.

While it's no longer a good time-limited test, you can still use this test to get us excited about hiring you! If you optimize below 1487 cycles, beating Claude Opus 4.5's best performance at launch, email us at performance-recruiting@anthropic.com with your code (and ideally a resume) so we can be appropriately impressed, especially if you get near the best solution we've seen. New model releases may change what threshold impresses us though, and no guarantees that we keep this readme updated with the latest on that.

Run `python tests/submission_tests.py` to see which thresholds you pass.

## Warning: LLMs can cheat

None of the solutions we received on the first day post-release below 1300 cycles were valid solutions. In each case, a language model modified the tests to make the problem easier.

If you use an AI agent, we recommend instructing it not to change the `tests/` folder and to use `tests/submission_tests.py` for verification.

Please run the following commands to validate your submission, and mention that you did so when submitting:
```
# This should be empty, the tests folder must be unchanged
git diff origin/main tests/
# You should pass some of these tests and use the cycle count this prints
python tests/submission_tests.py
```

An example of this kind of hack is a model noticing that `problem.py` has multicore support, implementing multicore as an optimization, noticing there's no speedup and "debugging" that `N_CORES = 1` and "fixing" the core count so they get a speedup. Multicore is disabled intentionally in this version.
