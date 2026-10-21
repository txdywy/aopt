# Verified 868-cycle submission

Historical result: the current standalone submission is now 866 cycles. See
`SUB864_RESEARCH_20261021.md`; `perf_takehome_868.py` preserves this exact source.

This milestone executed **868 cycles** on the unchanged frozen
simulator, exceeding the user's target of fewer than 870 cycles. It is a
standalone Python submission; no optimizer, C++ compiler, predecessor file or
external data is needed to build its instruction list.

## Verified result

|Metric|Value|
|---|---:|
|Cycles / instruction packets|868|
|Baseline speedup, 147734 cycles|170.20x|
|Cycle reduction from 880|1.36%|
|Scratch words|1335 / 1536|
|ALU operations|10338|
|VALU operations|5190|
|LOAD operations|1672|
|STORE operations|374|
|FLOW operations|854|
|Total operations|18428|
|Standalone Python bytes|151945|
|Default instruction-list JSON bytes|486761|

Compact instruction JSON SHA256:
`3e89c7172e69bd17504d63abf1bc9102162558b268aee74511818a9d3a833367`.

Validation used Python 3.12.15:

- All 9 original official tests passed, each reporting 868 cycles.
- 256 full-width random seeds, 8 ordinary seeds and 6 boundary patterns passed.
- These 270 cases executed the instruction list after a JSON round trip.
- Every architectural vector lane stays within the 1536-word scratch space;
  no packet contains conflicting scratch writes, and slot limits hold.
- All runtime memory accesses stay within the original 2566-word image.
- `git diff origin/main -- tests/ problem.py` is empty.

The supported contract remains height 10, 2047 nodes, 256 initially zero indices,
16 rounds, and final values only (**Without Indices**). Consumed header, tree and
index memory holds temporary records. Browser acceptance has not been tested.

## Address and record changes

The compiler keeps the deferred final hash XOR, so a saved parity bit `b` is
the complement of the true hash parity. Selected multipliers use `b - 2`,
represented directly by OR with `0xfffffffe`. This makes two traversal levels
share one multiply-add address calculation.

The locally searched constants, in word addresses, are:

```text
X = (-72, -52)
Y = (2042, 2154)
V = (-4, 4)
A3 = Y[b1] + X[b0] * (b2 - 2)
A5 = A3 + V[b3] * (b4 - 2)
```

The 8 depth-three and 32 depth-five records fit without overlap in the
224-word interval `[2086, 2310)`. Each four-word record holds the encoded right
child, encoded left child, encoded node, and padding or an anchor. The
depth-five anchor is an existing tree-block pointer `6 + 16*q5`, so it adds
no per-record address arithmetic. Two small store/reload banks at `[2054,2070)`
transport those anchors into vectors. Depth-seven records occupy `[6,518)`.

Gather windows reuse children through overlapping vector views; their full
architectural writes remain present even when only some loaded lanes matter.
All 511 shallow tree nodes are XOR-adjusted once. Deeper raw loads restore
the lazy hash state before consuming the node. The final XOR is associated
with the last hash's preceding stage, shortening the output path.

Early constants use available FLOW slots; other constants and pointers are
derived using initially idle arithmetic slots. A word-SSA allocator permits
the new records and overlapping vector windows to be scheduled safely. The
scheduler chooses native VALU issue or eight scalar ALU operations per group.
For this fixed arithmetic graph the combined ALU/VALU lane-work bound is
`ceil((10338 + 8*5190) / (12 + 8*6)) = 865` cycles. That is a bound for this
graph, not proof of global optimality.

The new family first reached 872 cycles with 20 full-width seeds passing.
Startup and tail changes then reached 868, followed by the full validation
above. The preserved 880 version is `codex/perf_takehome_880.py`.

## Reproduce the verified calendar

From the repository root, with Python 3.10 or newer:

```bash
python3.12 tests/submission_tests.py
python3.12 -m codex.verify_compact
```

The following reconstructs the program from the readable compiler and saved
calendar. It does not run heuristic search. Pickle files here are locally
generated compiler intermediates; no downloaded pickle is used.

```bash
python3.12 -m codex.stride4_family --head --tail --prefix /tmp/aopt-868
python3.12 -m codex.packet_modes export --ir /tmp/aopt-868.ir.pkl --prefix /tmp/aopt-868-modes
cp codex/stride4_868_schedule.txt /tmp/aopt-868-modes.sol
python3.12 -m codex.packet_modes check --prefix /tmp/aopt-868-modes
python3.12 -m codex.emit_packet_submission /tmp/aopt-868-modes.json /tmp/aopt-868.py --cycles 868
```

The check validates dependency edges, allocates physical scratch, and verifies
20 full-width seeds before emitting JSON. Compare its compact JSON digest
against the value above. `stride4_868_result.json` records artifact digests.

For further scheduling research, compile `packet_mode_search.cpp` with C++17
and run it on the exported `.txt` model, writing a separate `.sol` file. The
verified calendar was found using seed 743 and 5000 as the iteration limit;
the search was interrupted after reaching 868. Every candidate needs fresh
dependency, scratch and frozen-machine checks before promotion.

```bash
c++ -std=c++17 -O3 codex/packet_mode_search.cpp -o /tmp/aopt-packet-mode-search
/tmp/aopt-packet-mode-search /tmp/aopt-868-modes.txt /tmp/aopt-new.sol 5000 743
```

`stride4_layout_search.cpp` preserves the small sumset search that found the
224-word layout. `research_packet_870.py` also retains older experimental
transformations; its physical-register boundary-folding prototype is not the
compiler path used for this submission.

## Provenance

[Josu San Martin's public research log](https://github.com/josusanmartin/josusanmartin.github.io/blob/master/vliw/chat.log)
reports 864 cycles and describes signed-parity bilinear addresses and
stride-four records. It did not supply a complete executable 864-cycle
kernel. These ideas informed the locally written frontend, layout search,
word-SSA model, register allocator and scalar/vector scheduler.

[Sakauma's public 880-cycle kernel](https://github.com/Sakauma/my_original_performance_takehome/blob/cb2e0554735ef7d0dfde1a04bc2c12c4bb2ed38d/perf_takehome.py)
was previously reproduced and is preserved locally. Its first hash
multiply-add times supplied initial scheduling priorities; the final 868
instruction stream is newly generated and differs from that kernel.

The standalone source preserves attribution and the original Anthropic notice.
This result is a local verified improvement, not a claim of a global optimum
or an independently reproduced 864-cycle implementation.
