# Optimized Python submission

The current standalone submission is [`../perf_takehome.py`](../perf_takehome.py).
An identical, separately preserved 865-cycle copy is
[`perf_takehome_865.py`](perf_takehome_865.py).
It executes **865 cycles**, a **170.79x** speedup over the 147734-cycle baseline.
This improves the previous 866 result by one cycle. The current target of
**fewer than 864 cycles remains open**.
It specializes `(height=10, nodes=2047, batch=256, rounds=16)`, initial indices
all zero, and the **Without Indices / final values only** contract.

The new instruction compiler, word-SSA register allocator and mixed scalar/vector
scheduler are implemented locally. Signed-parity bilinear addressing and
stride-four records were informed by [Josu San Martin's public research
log](https://github.com/josusanmartin/josusanmartin.github.io/blob/master/vliw/chat.log).
[Sakauma's preserved 880-cycle kernel](https://github.com/Sakauma/my_original_performance_takehome/blob/cb2e0554735ef7d0dfde1a04bc2c12c4bb2ed38d/perf_takehome.py)
supplied initial hash scheduling priorities. The final 865-cycle program is a
new, locally generated instruction stream. The original source notice and
attribution are retained; no claim of global optimality is made.

Validation on the unchanged frozen simulator:

- All 9 unchanged official tests pass at 865 cycles.
- 256 full-width random seeds, 8 ordinary seeds, and 6 boundary patterns pass.
- Those 270 cases execute instructions after a JSON serialization round trip.
- Every packet satisfies engine slot and full-vector scratch-address limits;
  no scratch location has two writes in the same packet.
- Runtime memory reads and writes stay within the original allocation.
- 1294 scratch words; 865 instruction packets; 18494 operations.
- Python source: 153007 bytes. Default instruction-list JSON: 489061 bytes.
- `tests/` and `problem.py` are unchanged from `origin/main`.

Consumed header, tree, index and already-loaded input-value memory is used for
temporary records and selection banks. Final indices are outside this contract. Browser
submission acceptance has not been independently tested; JSON transport and
frozen-machine execution have been checked locally.

## What changed

Reversed, XOR-adjusted four-word records each hold a node and its children.
A locally searched 224-word layout allows depth-three and depth-five records
to share bilinear address arithmetic. Depth-five records also hold an existing
tree-block pointer, reducing subsequent address calculations. Overlapping
scratch windows and vector selects reuse loaded children. Mixed scalar/vector
issue, early constant and pointer construction, and final-hash reassociation
bring the frozen-machine execution to 868 cycles. The 866 revision moves the
bilinear image into the consumed input-value region, reuses input block
pointers, uses one anchor bank, and builds pairs of uniform vectors with one
known-mask FLOW select plus scalar copies. The 865 revision uses three-word
FLOW broadcasts, selected STORE/reload broadcasts, an early native constant
broadcast, and scheduler priority for native vector instructions.

See [the 865 result and exact reproduction note](SUB864_865_RESULT_20261021.md).
Further three-word FLOW broadcast experiments reduced arithmetic work from
51744 to 51669 lanes and passed 20 full-width cases, but still took 866
cycles. Those alternatives are preserved in the [866 research note](SUB864_RESEARCH_20261021.md).
The active 865 submission has completed its own official and 270-case validation.
The [868-cycle note](SUB870_BREAKTHROUGH_20261021.md) preserves the earlier result. The
[880-cycle reproduction note](SUB900_BREAKTHROUGH_20261021.md) remains historical.

## Reproduce

Use Python 3.10 or newer; Python 3.12.15 was used for these checks.

```bash
python3.12 tests/submission_tests.py
python3.12 -m codex.verify_compact
git diff origin/main -- tests/ problem.py
```

`verify_compact.py` checks the current submission, including its instruction
digest, architectural writes, memory bounds and final output values. It uses
the unchanged frozen simulator and reference. No optimizer or extra data file
is needed to build the standalone submission.

Program SHA256 (compact instruction JSON):
`91f09dee8f416e1eb40358e30e6c3cfa368407e9a98e8801d62a07ba25bd0c51`.

The payload is compressed ISA, independent of runtime tree/input values.
All tree transformations, loads and hashes execute on the original machine.
The readable compiler is `stride4_family.py` with `research_sub864.py`
and `research_bootstrap864.py`;
`word_ssa.py`, `packet_modes.py`
and `packet_mode_search.cpp` handle scheduling and register allocation.
`stride4_865_schedule.txt` preserves the verified calendar so the exact result
can be regenerated without running the heuristic search again.

## Preserved experiments

The 865-cycle standalone submission is saved byte-for-byte as
[`perf_takehome_865.py`](perf_takehome_865.py). The previous 866-cycle submission
is saved as [`perf_takehome_866.py`](perf_takehome_866.py).
The previous 868-cycle submission is saved as
[`perf_takehome_868.py`](perf_takehome_868.py), with its original result metadata
and calendar. The previous 880-cycle submission is saved as
[`perf_takehome_880.py`](perf_takehome_880.py). The earlier portable submission is
[`perf_takehome_981_portable.py`](perf_takehome_981_portable.py), alongside the
older `perf_takehome_983.py`. The separate 979-cycle research artifact expands
to roughly 2.20 GB of JSON and previously failed browser submission.

`reoptimize_compact.py`, `compact_setup_schedule.json`, `test_compact_setup.py`
and `resource_profile_981.json` describe the historical 981-cycle startup
optimization. To reproduce that version, write to a separate file:

```bash
python3.12 -m codex.reoptimize_compact --emit-python /tmp/aopt-981-regenerated.py
python3.12 -m unittest codex.test_compact_setup
```

The [earlier sub-900 review](SUB900_REVIEW_20261021.md) records unsuccessful
experiments before this public implementation was found. Its conclusions
about the then-current 981-cycle program are historical.

Other files include historical optimization experiments. Their defaults
and recorded scores refer to their own configurations, not necessarily this
standalone submission.
