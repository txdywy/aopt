# Continuation toward 863 cycles

The active and separately preserved submissions remain **865 cycles**.
This pass did not reach 864 or the requested 863-or-better target. Both
`../perf_takehome.py` and `perf_takehome_865.py` retain source SHA256
`58ce2efca049a18ef343ba82ceaeb72fd74f638bd7c8bbc23a14e6e796d72f44`.
The unchanged official nine tests and 270 extended cases pass again.

## Experiments and measured results

`research_sub863.py` preserves 20 new configurations, including alternating
hash-branch constants, two selector replication layouts, partial native-vector
overwrites, early 4097 construction, vectorized cache encoding, native cache
broadcasts, and additional STORE/reload broadcasts. Every saved configuration
passed three full-width frozen-machine controls. The complete flags and
available measurements are in `SUB864_CONTINUATION_20261021.json`.

Ten heuristic searches ran 8800 iterations / 35200 forward or reverse
scheduling passes. The best new searched calendars took 866 cycles. A fully
lowered 866-cycle alternating-constant calendar passed 20 full-width cases
and is saved separately as `stride4_hashpack_866_schedule.txt`, with digests
and measurements in `stride4_hashpack_866_result.json`. It uses 1324 scratch
words and 51666 arithmetic lanes, versus the active submission's 51707.

The lowest-work graph uses **51648** lanes, 1688 LOADs and 844 FLOWs. Its
searched calendar took **871**, not an improvement. Removing initialization
operations also removed useful early vector work; the remaining STORE/reload
and constant dependencies then delayed the hash pipeline.

The optional native constant layouts reuse only observed lanes of full vector
writes. Later scalar definitions overwrite unused positions, while allocation
continues to track all eight architectural destinations. Serial controls now
follow the IR's dependency order, including these overwrite dependencies.

The bootstrap compiler also fixes the experimental `--small-stage=9` / `19`
path: a requested reload no longer shares its SSA destination with a retained
native broadcast. The default 865 model reproduces byte for byte after this
change; its scheduling-model SHA256 remains
`29e9b35d471646f6d7810ad6b0d0d92d44d258b7241ae747900db28913a521dc`.

## A boundary lower bound for the existing 865 graph

`packet_budget_bound.py` adds an independent CP-SAT boundary relaxation. It
permits individual scalar lanes to straddle a boundary, or any exported
eight-lane group to issue as one native vector operation. Dependencies,
architectural overwrite edges, and each engine's capacity are retained.
The prefix and suffix are optimized independently, so the result is a lower
bound for this specific graph and its exported mode choices.

For the active graph, the solver reports OPTIMAL for both boundaries:

| Boundary | Available arithmetic lanes | Maximum issued work | Unavoidable unused lanes |
|---|---:|---:|---:|
| First six cycles | 360 | 298 | 62 |
| Last three cycles | 180 | 64 | 116 |

The suffix also has a structural bound: at most two output stores can issue
in a cycle. Within three reverse cycles, each first-cycle store exposes at
most 24 arithmetic lanes, and each second-cycle store at most eight. Thus
the maximum is `2*24 + 2*8 = 64`. The tool checks the unique-store ancestry
before applying this cut; it does not assume this property for other graphs.

For any schedule represented by this graph:

```text
cycles >= ceil((51707 + 62 + 116) / 60) = 865
```

The measured 865 calendar therefore meets this graph's cycle lower bound.
This excludes reaching 864 by further retiming or the exported scalar/vector
choices alone. It does **not** prove 865 is optimal across other algorithms,
instruction eliminations, or new instruction groupings.

The certificate and model digest are saved in `stride4_865_budget_bound.json`.
The 51648-lane alternative also has a graph-specific 865 lower bound: its
six-cycle prefix loses at least 116 lanes, canceling the smaller work count.
The native alternating-constant variants passed execution controls but their
boundary bounds also remained 865. Further random search of these graphs
cannot reach the requested score.

## Reproduce

The standalone submission still uses only Python's standard library. The
boundary tool additionally needs the optional OR-Tools research dependency.

Rebuild the exact active graph and its lower bound:

```bash
python3.12 -m codex.research_sub863 --prefix /tmp/aopt-bound-base --layout tight --small-stage= --literal-flow 2
python3.12 -m codex.packet_modes export --ir /tmp/aopt-bound-base.ir.pkl --prefix /tmp/aopt-bound-base-modes
python3.12 -m codex.packet_budget_bound /tmp/aopt-bound-base-modes.pkl --prefix 6 --suffix 3 --seconds 10 --output /tmp/aopt-bound-base.json
```

If OR-Tools is installed in a separate research directory, pass
`--ortools-path /path/to/that/directory` to the last command.

Rebuild the preserved 51666-lane experiment without repeating its search:

```bash
python3.12 -m codex.research_sub863 --prefix /tmp/aopt-hashpack --hash-pack --coefficients --literals 2 --literal-flow 7
python3.12 -m codex.packet_modes export --ir /tmp/aopt-hashpack.ir.pkl --prefix /tmp/aopt-hashpack-modes
cp codex/stride4_hashpack_866_schedule.txt /tmp/aopt-hashpack-modes.sol
python3.12 -m codex.packet_modes check --prefix /tmp/aopt-hashpack-modes
```

## Remaining direction

Reaching 863 requires changing the operation graph. A useful change must
reduce the combined arithmetic work and boundary losses while fitting the
dense LOAD/FLOW calendar. Lower arithmetic counts alone have repeatedly
failed this test. The next investigation should change a node-selection,
addressing, or hash mechanism before starting another scheduling sweep.

The [public research directory](https://github.com/josusanmartin/josusanmartin.github.io/tree/master/vliw)
was checked again in this pass. It contains the public chat log and its viewer
assets, rather than a newly published executable kernel. No leaderboard
change or new browser-submission result is asserted here.
