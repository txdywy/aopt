# Optimized Python submission

The current standalone submission is [`../perf_takehome.py`](../perf_takehome.py).
An identical, separately preserved 864-cycle copy is
[`perf_takehome_864.py`](perf_takehome_864.py).
It executes **864 cycles**, a **170.99x** speedup over the 147734-cycle baseline.
This improves the previous 865 result by one cycle and reaches the 864 target.
The previous [`perf_takehome_865.py`](perf_takehome_865.py) is unchanged.
The latest [runtime-count study](SUB863_RUNTIME_COUNT_20261029.md) establishes
a global 16..23 low-half count band and a 65535 quotient bound for a sole SHR.
It excludes 5,520 first-count prefixes, proves all 552 one-T count shapes
must become prepared constants, and closes 1,656 count-aware final-MAD
connections. The third tail uses 4,718,592 parameter cases checked by two
independent implementations. These are scoped local exclusions; the
executable remains 864 and broader grammars remain open.
The preceding [quotient-range and keyed-live study](SUB863_QUOTIENT_RANGE_20261029.md)
closes the remaining 127 prepared-division connections, completing the
528-connection sample. It also excludes 1,899 distinct four-op C5-absorption
connections with earlier live values. A four-point masked-prior certificate
closes an arbitrary-parameter family. All 138 research tests pass; the
executable remains 864, and complete four-op grammars remain open.
The preceding [prepared-shift and division study](SUB863_PREPARED_OPERANDS_20261029.md)
excluded 1,384 sampled prepared-shift connections and established a
478-state information bound, arbitrary one-high-word tail certificates,
complete permutation-sum divisor scans and a wide-product division encoding.
The preceding [multiple-division audit](SUB863_TWO_DIVISION_20261029.md)
closes all 343 three-instruction orders over MAD/XOR/AND/OR/DIV/REM/CDIV
on the stated nine live words, including arbitrary root-only constants and
prepared-count shifts. It also covers two initial-data downward operations
with variable right-shift counts, and DIV/bitop-or-MAD/final-SHR.
Prepared-data variable
shifts, further variable-shift combinations and broader algorithms remain open.
The [expanded four-instruction search](SUB863_JOINT_FOCUSED_20261029.md)
adds 16,020 distinct connections beyond the earlier two four-op manifests.
Every sampled connection is excluded after refinement; the full four-op grammar
remains open. That search also exposed and fixed a concrete-value typing error
in the research solver, with targeted regression and positive controls.
The preceding [single-division core audit](SUB863_SINGLE_DIVISION_CORE_20261028.md)
records the earlier single-division and prepared-count SHR-then-division proofs.
The preceding [computed-denominator audit](SUB863_DENOMINATOR_GRAMMAR_20261028.md)
records the complete computed-denominator family. The earlier
[computed-prefix division audit](SUB863_COMPUTED_DIVISION_20261028.md) records
the complete inverse-parameter enumeration and earlier collision certificates.
The preceding [division/comparison audit](SUB863_DIVISION_COMPARISON_20261028.md)
records exact full-divisor screens and its explicitly scoped closures.
The [sub-863 structural audit](SUB863_STRUCTURAL_AUDIT_20261023.md) proves a
864 lower bound for the current graph and records 32 unsuccessful structural
variants. It does not claim a universal lower bound or a new executable score.
The [cross-stage hash audit](SUB863_MIDDLE_CROSS_STAGE_20261023.md) additionally
excludes every ordering of two MADs and one XOR for the four-operation middle
hash with one input and arbitrary constants; the executable remains 864 cycles.
The [remaining-routes audit](SUB863_REMAINING_ROUTES_20261023.md) extends that
result to all 64 three-instruction MAD/XOR/AND/OR orders and records nine
inconclusive searches with additional live intermediates.
The [direct-input and free-encoding continuation](SUB863_ENCODING_FRONTIER_20261024.md)
adds 336 joint queries, automatic recovery of the full32 six-op control,
and two affine-family exclusions with arbitrary endpoint XOR keys.
General three-op encoded programs remain unresolved; the verified score is still 864.
It specializes `(height=10, nodes=2047, batch=256, rounds=16)`, initial indices
all zero, and the **Without Indices / final values only** contract.

The new instruction compiler, word-SSA register allocator and mixed scalar/vector
scheduler are implemented locally. Signed-parity bilinear addressing and
stride-four records were informed by [Josu San Martin's public research
log](https://github.com/josusanmartin/josusanmartin.github.io/blob/master/vliw/chat.log).
[Sakauma's preserved 880-cycle kernel](https://github.com/Sakauma/my_original_performance_takehome/blob/cb2e0554735ef7d0dfde1a04bc2c12c4bb2ed38d/perf_takehome.py)
supplied initial hash scheduling priorities. The final 864-cycle program is a
new, locally generated instruction stream. The original source notice and
attribution are retained; no claim of global optimality is made.

Validation on the unchanged frozen simulator:

- All 9 unchanged official tests pass at 864 cycles.
- 256 full-width random seeds, 8 ordinary seeds, and 6 boundary patterns pass.
- Those 270 cases execute instructions after a JSON serialization round trip.
- Every packet satisfies engine slot and full-vector scratch-address limits;
  no scratch location has two writes in the same packet.
- Runtime memory reads and writes stay within the original allocation.
- 1301 scratch words; 864 instruction packets; 18558 operations.
- Python source: 153921 bytes. Default instruction-list JSON: 490657 bytes.
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

The 864 revision adds constant staging, scalar deepest-leaf loads and raw-hash
reassociation, then jointly repairs compute and memory scheduling. It uses
51,665 arithmetic lanes; after startup and drain, only seven middle lanes
are idle. See [the verified 864 result and reproduction recipe](SUB864_864_RESULT_20261023.md).
The [structural audit](SUB863_STRUCTURAL_AUDIT_20261023.md) excludes 863 for
the current graph and records 32 alternative construction checks.
The [hash rewrite study](SUB863_HASH_RESEARCH_20261023.md) checks 406 templates
across three hash slices: 405 are excluded by SMT and one by an independently
checked polynomial certificate. It does not improve the verified 864 score.
The [public-source refresh and coverage extension](PUBLIC_RESEARCH_20261023.md)
adds 336 checks and proves first-instruction coverage for these isolated
two-arithmetic-op searches. This closes that restricted search space, not
whole-hash or whole-kernel optimality; the verified score remains 864.

Earlier results remain in [the 865 reproduction note](SUB864_865_RESULT_20261021.md).
The [next research pass](SUB864_CONTINUATION_20261021.md) preserves 20 further
configurations and a graph-specific 865-cycle lower bound. It did not improve
the active score or reach fewer than 864 cycles.
The [2026-10-22 frontier study](SUB864_FRONTIER_20261022.md) adds 122 startup
configurations, joint constant and pointer packing, and a terminal address-table
experiment. Its candidate took 867 cycles. Productive-work and fixed-stream
bounds rejected those tested graphs.
The [next 864 study](STAGE864_RESEARCH_20261022.md) builds different constant
packs, scalar deepest-leaf loads and one-arm drain transposes. Its best new
graph is verified at 866, while another verified graph reaches 51,665
arithmetic lanes. That study passed the necessary 864 budget checks but did
not yet produce a complete 864 program.
The [Astra review](ASTRA864_RESEARCH_20261022.md) adds a verified 867-cycle
constant-4 staging variant with six fewer arithmetic lanes. The rawfold
branch produced a dense 670-cycle prefix and a separate 864 LF relaxation.
The Astra graph subsequently became the verified 864 submission through the
joint scheduling and exact suffix solve described in the new result note.
Further three-word FLOW broadcast experiments reduced arithmetic work from
51744 to 51669 lanes and passed 20 full-width cases, but still took 866
cycles. Those alternatives are preserved in the [866 research note](SUB864_RESEARCH_20261021.md).
The active 864 submission has completed its own official and 270-case validation.
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
`3f4cc919d7c5b05d105b5e41ba2ea3502020a03ec1318388a1138718abf2d4f1`.

The payload is compressed ISA, independent of runtime tree/input values.
All tree transformations, loads and hashes execute on the original machine.
The readable compiler is `stride4_family.py` with `research_sub864.py`
and `research_stage864.py` (built on `research_bootstrap864.py`);
`word_ssa.py`, `packet_modes.py`
and `packet_mode_search.cpp` handle scheduling and register allocation.
`stage_astra_constant4_864_schedule.txt` preserves the verified calendar so the exact result
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
