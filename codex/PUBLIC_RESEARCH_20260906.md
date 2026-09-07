# Public-source refresh, 2026-09-06

Sources read directly:

- https://github.com/fiigii/ai-comp
- https://raw.githubusercontent.com/fiigii/ai-comp/main/compiler/pass_config.json

The compiler config enables partial ALU filling by devectorization, load-unblock
priority, automatic stream staggering, static devectorization, and constants
via FLOW. Its pipeline places MAD synthesis after SLP vectorization and CSE.
These are configured features, not independent evidence of speed or a sub-900
solution. No better executable kernel was obtained in this search.

Implication for our experiments (independent inference): static hash-restoration
counts optimize aggregate capacity but ignore when each engine has capacity.
The 43-lookup experiment has lower work counts yet takes 1054 cycles. A useful
next scheduling experiment is late, readiness-aware scalar/vector selection,
with instruction choices charged against actual per-cycle capacity. Whole-vector
MAD must not be treated as an ordinary one-to-eight conversion: scalar ISA lacks
MAD and expansion costs two scalar operations per lane. Partial vector splitting
also needs explicit per-lane dependencies and register-lifetime accounting.

This does not by itself solve the sub-900 target: the current experimental
arithmetic work still has a relaxed bound above 900. Work reduction and dynamic
lowering must be evaluated separately; no amount of scheduling can defeat a
valid resource bound for unchanged work.

A search also surfaced 'Superoptimization for the EMCA DSP', but the PDF fetch
failed. Its contents were not read and must not be cited as supporting an
optimization. Search results for older challenge writeups mostly describe
scores slower than the current 983 artifact; these are not breakthroughs.

## First dynamic-lowering implementation

`DYNAMIC_VECTOR_TO_ALU` is default-off. After ordinary packet filling, if all
six VALU slots are occupied and eight ALU slots remain, select one ready basic
vector operation and emit its eight scalar lanes together. MAD and broadcasts
are excluded. The original dependency node completes in that cycle; no partial
lane completion is introduced. This conservative first version does not reduce
arithmetic work and does not split vectors across cycles.

The overlapping group-3, rounds 4 and 15, arity-4 experiment with teacher
weight 1024, compact returns, recycled stride, relaxed jumps, no packet repair,
and vector indexing improves from 990 to **989 cycles** with the new flag.
All 50 full32 seeds pass, 263458 code packets, scratch 1536. The global best
remains 983. Unit tests check scalar capacity and consumer latency, and reject
MAD scalarization; baseline fingerprint and multi-round tests also pass.

Note: DAG engine counts remain the pre-lowering counts. Use executed-slot
profiling for resource totals, not the DAG histogram, when this flag is on.

Full 31-cache four-lane configuration with dynamic lowering: 20 full32 seeds
pass at **1016 cycles**, unchanged from the same configuration without dynamic
lowering. Executed totals ALU 11643, VALU 5563, LOAD 1887, FLOW 847, STORE 62.
Six vector operations were expanded, but no critical-path reduction resulted.
The relaxed arithmetic bound is ceil((11643 + 8*5563)/60) = 936; unchanged
ALU allocation requires 971 cycles and LOAD requires 944. This experiment
therefore needs both work reduction and schedule changes, not further claims
that opportunistic scalarization alone can reach 899.

The driver now emits executed-resource bounds for target 899 whenever slot
profiling is enabled. This uses actual main plus table execution, avoiding
misleading conclusions from pre-lowering DAG counts.
