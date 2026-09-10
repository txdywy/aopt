# Node reuse algorithm branch

`python -m codex.profile_node_reuse --seeds 20` profiles20 seeds each at
30-bit and full32 values. All40 complete traversals match the unchanged
frozen reference in values AND indices. No generated kernel is modified.

For rounds4..10 (the seven256-lane gather rounds), ideal duplicate-load
elimination within fixed contiguous groups gives:

| Group size | Min saved loads | Mean | Max |
|---|---:|---:|---:|
| 2 | 2 | 15.3 | 29 |
| 8 | 73 | 101.25 | 132 |
| 16 | 152 | 196.15 | 222 |
| 32 | 309 | 343.475 | 394 |
| 256 | 942 | 979.2 | 998 |

These are ideal access counts, NOT cycle results or guarantees for unseen
inputs. Current983 needs91 fewer loads even before startup/dependencies.
Adjacent-pair deduplication is insufficient in every sampled case; group8
deduplication is insufficient in some, even with all routing costs ignored.
Naive exhaustive comparisons inside group8 require7*32*28=6272 scalar
comparisons before selections/copies/control, worsening arithmetic pressure.

Do not implement naive pairwise deduplication as a promising sub900 change.
The potentially interesting branch is global or larger-group routing with
amortized cost across consecutive levels. It would need an explicit design
for stable values/output mapping, dynamic population sizes, scratch capacity,
and comparison/scatter cost, and must be judged against the same combined
ALU/VALU/LOAD/FLOW budget. No grouping implementation or speedup is claimed.

## Homogeneous SIMD bucket cost

The same40 verified traversals now count `sum(ceil(population/8))` vectors
when each vector is restricted to a single node. Average unused lanes per
round4..10 are55.2,107.0,260.4,638.0,1038.2,1356.6,1558.2.
With the current ten-operation hash, processing these padded vectors adds
48640..53040 arithmetic lane-work units (mean50136), before any grouping
cost. This defeats the naive node-homogeneous all-vector implementation.

This does not reject all global grouping. A viable design would have to
mix node populations within vectors or handle bucket remainders differently,
then explicitly account for the extra node-value routing or scalar hash work.
Do not treat the earlier979.2 average load savings as a standalone speedup.

## Per-operation scalar/vector relaxation

`python -m codex.bound_bucket_hash` allows every operation independently to
use ALU or VALU, including mixed execution within a node remainder. For the
existing ten-op hash, six basic operations cost r ALU slots versus one VALU
slot; four MADs cost2r ALU slots versus one VALU slot. For each possible VALU
count, choosing the largest scalar costs maximizes saved ALU work. Four tiny
instances are checked against exhaustive allocation enumeration.

Across the same40 reference-verified traversals, hash-only capacity floors
are836..852 (mean842.25). Including one node XOR per hash gives909..928
(mean917.1). This still ignores all routing, loads, stores, control, path
updates, dependencies and register constraints. Other rounds get ideal fully
packed vectors, not homogeneous-node padding.

Thus the sampled inputs already rule out sub900 for this particular existing
hash-plus-XOR lowering with node-homogeneous chunks in rounds4..10, even with
free per-op scalar/vector allocation. It does not exclude new hash formulas,
node-XOR fusion, or vectors containing walkers from different nodes. The next
viable grouping design must address mixed-node vector assembly explicitly.

## One-time grouping retains cross-level locality

A concrete memory-scatter regrouping design with one scalar value store per
walker at every round4..10 consumes1792 STORE slots. Even optimistic final
output via32 vector stores brings this to1824 slots,912 cycles on STORE alone,
before IDs or any other stores. This rejects that seven-pass scalar-scatter
design, not arbitrary permutation algorithms or fewer-pass alternatives.

New profile: sort lane IDs by their round4 node ONCE, then preserve that
permutation through round10. Keep full8-lane vectors, permitting mixed nodes
at bucket boundaries and in later rounds. Across the40 verified traversals,
ideal within-vector duplicate-load savings total599..700 (mean655.775), versus
73..132 (mean101.25) without that initial grouping. Reference simulation and
final values/indices checks remain unchanged; the permutation is only used
for read-only counting here.

This changes the next design action: examine a single shallow partition plus
mixed-node value distribution, not per-level sorting or homogeneous SIMD.
The savings are not an executable performance claim. Sorting, permutation
recovery, node dispatch and comparison/copy costs remain unimplemented and
must be included before accepting the design. One initial sort could in
principle amortize locality across multiple levels; measurements do not prove
its implementation will be cheaper than the current kernel.

## Executable algorithm reference prototype

`python -m codex.prototype_once_grouped` actually permutes (index,value,id)
records at round4, evaluates mixed-node vectors while keeping that ordering,
then restores both final values and indices by original ID. It matches the
unchanged frozen reference on40 random inputs plus all-zero,all-ones,and
alternating full32 boundary patterns. This is Python algorithm code, NOT
a generated VLIW implementation or a cycles benchmark.

Among1280 mixed vectors from the40 random cases,743 have one round4 parent,
536 have two,and ONE has three. Therefore a two-parent-only lowering would
be incorrect; general fallback is required. Descendant prefetch demand using
all children (not only observed children), with eight scalar lane loads as
fallback when cheaper, averages45.45,90.9,181.7 loads at depths4,5,6. These
figures exclude discovering parents,selecting node values,sorting,ID transport,
scratch allocation and output recovery. The independent prototype reproduces
655.775 ideal saved loads across rounds4..10.

## Actual workspace audit

`python -m codex.check_grouping_workspace` verifies the canonical frozen
memory image has2566 words and zero tail space after the256 input values.
Although build_mem_image initially allocates extra_room, its final assignment
`mem[inp_values_p:] = inp.values` shrinks the tail. Do not modify the frozen
builder or allocate beyond the actual memory bounds in a submission.

The existing index/value regions supply512 words only if their contents are
preserved or no longer needed; they are not additional scratch. A barrier
sort with256 input keys,256 input values,256 output keys,256 output values,
16 counters,and16 bases needs1056 scratch words, leaving480 for all other
live state. This is a capacity sketch, not an allocation proof. The ISA has
no general indirect scratch write, so an actual lowering still needs static
permutation networks, control-dispatched writes, or safe memory scatter.
The next implementation must choose and measure that routing mechanism;
ordinary Python array indexing is not an available machine instruction.

## Frozen-ISA rank-scatter transport microbenchmark

`python -m codex.benchmark_rank_scatter` implements a real VLIW transport
stage: ranks supplied in the input-index region and values in the input-value
region are loaded into scratch, destination addresses are formed, values are
scalar-scattered into the existing index region,and read back with vloads.
Forty permutations (identity,reverse,and38 shuffled) with full32 values pass
both memory and scratch-result checks. Tree/input-value memory is preserved;
memory remains2566 words and scratch high-water is1112.

Measured216 cycles including66 setup packets; the implemented transport
after inputs are ready occupies150 packets (6 vector-address,128 scalar-store,
16 vector-load). This is an implementation measurement, not an optimal lower
bound. Rankings are supplied by the microbenchmark, NOT computed by the code.
Original-ID transport,node distribution,hashing,and final output recovery are
excluded. The code must not be submitted as a complete challenge solution.

### Transporting original IDs too

`python -m codex.benchmark_rank_scatter --with-ids` adds real scalar-scattered
IDs into the original value buffer after all input values are resident in
scratch. ID constants are pipelined with stores using frozen read-before-write
semantics. Both sorted arrays are read back. All40 permutations pass ID/value
alignment,Python inverse recovery,and unchanged-tree/memory-size checks.

Measured367 cycles with the same66 setup packets,301 transport packets after
inputs ready,scratch high-water1128. Default value-only mode still measures216
total/150 transport. ISA final recovery is not included in either result.
This separate value/ID scatter path incurs512 scalar stores and is expensive.
Next investigate retaining the original-order rank map and inverse-gathering
final values instead of scattering an ID array; this alternative is not yet
implemented or measured here. Cycle counts are microbenchmarks,not challenge
scores,and their differences cannot simply be subtracted from whole-kernel
cycles without scheduling/bottleneck analysis.

### Original-order rank-map round trip

`python -m codex.benchmark_rank_roundtrip` implements the alternative in the
frozen ISA. Save256 absolute sorted-value addresses contiguously in the
original value region (16 vstore packets), rather than scattering IDs.
A mock vector XOR changes all sorted values and stores them in the index
region; the original rank scratch is explicitly erased. Reload the saved
map,scalar-gather final values,and vstore them in original order.

Forty identity/reverse/shuffled permutations with full32 values pass final
memory/scratch recovery,unchanged-tree,and2566-word memory checks. Measured
phases:66 setup,166 forward transport,30 mock-compute/clobber,160 inverse
recovery;422 total and326 transport+recovery. Scratch high-water1128.

These are real ISA measurements but ranks are still supplied,not computed.
The mock XOR is not the problem hash. The326 round-trip figure must not be
compared directly to the old301 ID-forward figure,which omits ISA recovery.
This design avoids scattered IDs but still has substantial transport cost;
inserting the measured stages as barriers is not evidence of a faster kernel.
Any viable integration needs less routing or overlap with useful computation,
plus the still-missing rank generation and mixed-node distribution costs.

## General chained-dispatch bucket ranks measured

`python -m codex.benchmark_bucket_rank` now computes stable ranks from actual
runtime4-bit keys in the frozen ISA. First pass copies the old bucket count
and increments it,then computes16 prefix bases; second pass adds the proper
base to each within-bucket ordinal. Both dynamic lookups use fixed instruction
variants and chained indirect jumps,not host-language sorting at runtime.

Forty random distributions plus all-zero,all-fifteen,and alternating extreme
keys pass stable-rank checks against Python sorting. Memory remains unchanged,
scratch high-water1160,static program8299 packets. Measured619 executed cycles:
85 setup,256 histogram traversal,22 prefix/retarget,256 rank traversal.
No value transport,hash,or output recovery is included.

This closes the straightforward per-element two-pass dispatch implementation
as a useful sub900 integration candidate: its512 FLOW traversal cycles dominate
before transport. It is a valid ranking microbenchmark,not an optimized complete
submission. Further grouping work must eliminate most per-element dispatch
(e.g. genuinely batched rank generation),not merely plug this routine into983.
