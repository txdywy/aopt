# Three-lane table prototype: corrected, slower than983

The experimental driver accepts arity3 using records(0,1,2) and(3,4,5).
Each table has4096 variants instead of65536. Lanes6,7 remain ordinary gathers.
Only scalar index packing is accepted; the existing pair-vector packing is
not valid for odd-width records. Default arity2/4 lane maps are unchanged.

The first scalar-cache/feedback test generated1013 main packets but failed
execution with an out-of-range memory access. This is NOT a cycle score.
After restricting fallback gathers to exactly the selected group/round pairs,
an independent no-feedback/no-scalar-cache run also failed: PC335 (VM PC at
load entry), load_offset943,943,6 read scratch949 as4294966497 with2566 memory
words. Thus the issue is not exclusive to feedback or scalar cache. The
mixed partial-table/gather representation needs further tracing; a register
layout conflict is a hypothesis, not yet a proven root cause.

Do not export this configuration or claim browser compatibility/correctness.
No simulator or frozen tests were changed. The next diagnostic should trace
the reaching definition of scratch949 and its mirror/address source rather
than increase solver time or tune scheduler priorities.

## Correction and verification, September9

Reaching-definition replay found scratch949 =37-scratch693, with scratch693
holding836, not a four-bit index. Saved-path groups intentionally stop keeping
a full packed mirror. Their reconstructed direct_index_base must also feed
the two remaining ordinary-gather lanes. Fixing this removed the fault.
The cached/GATHER_LEVEL4 path then XORed table lanes a second time; it now
skips fused direct lanes exactly as the ordinary gather path already does.

No-feedback/no-scalar-cache:3 full32 seeds all1011 cycles. With scalar cache
and4 feedback rounds at10 percent:20 full32 seeds all1009 cycles,255286
packets,135715470 JSON bytes. Executed slots (not pre-assembly DAG counts):
ALU10973, VALU5641, LOAD1949, FLOW847, STORE62. At899 the LOAD excess is151
and arithmetic-lane excess2161, despite52 spare FLOW slots. This configuration
is not faster than983 and is not a browser-verified submission. The new
end-to-end regression exercises both original failure paths on3 full32 seeds.

## Full coverage using a padded final pair

`--arity 3 --cover-remainder` uses(0,1,2),(3,4,5),(6,7). All blocks retain
4096 entries for the existing uniform-stride dispatcher; the last pair uses
the low two index digits, with the leading digit implicitly zero. Table
decoding now uses the actual record length. No extra index multiply is needed
for padding. Default arity2/4 maps and decoding are unchanged.

Scalar cache plus feedback4/10:20 full32 seeds all987 cycles,382240 packets,
205319224 default JSON bytes. Executed slots:ALU11035,VALU5641,LOAD1887,FLOW909,
STORE62. Compared with partial coverage, this removes62 executed LOAD slots
at the expense of62 FLOW and62 ALU slots, improving1009→987. It does not beat
the incumbent983 and its205MB JSON has not been browser-tested. At899 it still
exceeds LOAD by89,FLOW by10 and arithmetic lane work by2223. Two end-to-end
regression cases now cover both partial and padded full coverage.

## Expanded second-traversal coverage

Keeping first-pass groups fixed, final groups
`0,1,3,4,5,6,7,9,10,11,12,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30`
with padded arity3/scalar cache/feedback4-at10 yielded1035 cycles on3 full32
seeds,529744 packets. Executed work:ALU11131,VALU5641,LOAD1791,FLOW981,STORE62.
Thus LOAD fits899 but FLOW exceeds it by82, and arithmetic lane work by2319.
Compared with the987 triple candidate,96 LOAD slots were removed at the cost
of72 FLOW and96 ALU slots. This is a bottleneck transfer, not progress in score.

The existing post-schedule return fuser found97 candidate sites and fused0:
83 capacity,12 register-dependency,2 memory-order rejections. Three full32
seeds remained1035. First-conflicting-variant capacity diagnostics reported
ALU76,VALU79,LOAD76 sites (overlapping categories). They are not additive
counts of independent failures. Merely toggling return fusion does not solve
the control bottleneck; a future design must account for fusion before
packing unrelated work into table packets, or reduce dispatches structurally.

## Vector triple index packing

Depth4 first-pass `--vector-index` now supports arity3. The overlapping-source
MAD creates adjacent pairs `16*m[i]+m[i+1]`. Three-lane records append the
original third mirror lane using shift4 plus addition; the padded final pair
already has its complete index. Odd starting lane3 is legal in this mode.
Depth3 triple vector packing remains rejected because its radix needs a
separate implementation. Existing arity2/4 paths retain their old formulas.

Padded baseline with vector packing/scalar cache/feedback4-at10 gives986
cycles on20 full32 seeds and3 boundary-pattern seeds. Executed ALU10939 is96
less than the scalar-packed version; VALU5641,LOAD1887,FLOW909,STORE62 are
unchanged. Packet count382239. This beats this branch's987, not the compact
incumbent983. A third end-to-end regression covers the vector triple case.

## Scheduling-aware return-fusion probe

Vector packing on expanded low-load coverage gives1029 (3 full32 seeds),
six fewer cycles than its scalar-packed1035. FLOW remains981, so this is not
enough to make899 feasible.

New opt-in `--isolate-jumps` changes scheduling to FLOW-first and gives paired
indirect jumps exclusive packets, making later return fusion possible. On
the ordinary padded triple coverage, scalar cache, no feedback, with
`--fuse-returns`,3 full32 seeds pass at1066 cycles. Of64 fusion sites,10 fuse,
51 fail register dependencies and3 fail capacity. Thus pre-reservation does
enable real fusion but globally reserving empty jump packets costs more than
it saves. Do not enable this globally in a submission. The next possible
refinement is selective reservation based on proven fusion eligibility, not
rerunning the blanket-isolation switch. Default generator fingerprint remains
`62fd0cfa456526bcf9816eae8e8becc053f4130173a5cf4960ed902a6af34a04`.

## Early target preparation removes fusion hazards

`--jump-address-lead 2` strengthens the address-producer→jump dependency from
one to two cycles (default remains one). Together with isolated jumps this
keeps target-address writes out of the preceding table-return packet.
Ordinary padded triple coverage/no feedback now fuses62 of66 sites and runs
at1014 cycles on3 full32 seeds, versus1066 with lead1. An end-to-end regression
covers this setting; it is still slower than the normal986 triple schedule.

Expanded low-load coverage with lead2/isolation/fusion passes3 full32 seeds
at1031 cycles. Of91 sites,86 fuse;4 register-dependency and1 capacity failure
remain. Executed LOAD1791 and FLOW895 now both fit899. ALU11035 and VALU5641
still imply a relaxed arithmetic floor937, with2223 excess lane-work units.
Four feedback rounds at10 percent reproduce1031 and identical resource counts,
not an improvement. This is a concrete control-work reduction, but not a new
best executable score or proof that the remaining arithmetic can reach899.

## Arithmetic DCE screen of the low-load/fused configuration

The pre-assembly DAG has18 unconsumed definitions:one vbroadcast47,3,
sixteen scalar cache copies303..423 from166..181, and the last table-base
advance(+6,6,9), group21 round15. The sixteen copies are implicit table inputs,
not deletable dead definitions. The other two are only candidates pending
assembled-instruction audit. Even optimistically removing their9 lane-work
units cannot close the2223 arithmetic deficit. This DAG-only screen does not
prove whole-program DCE optimality; it excludes treating this visible set as
a substantial arithmetic-work solution. No deletion or score claim follows
from this screen, and the default submission remains unchanged.

## Jump-packet fill screen (2026-09-09)

Ordinary padded triple coverage, vector index, scalar cache, no feedback,
lead2/isolation/return fusion, three full32 seeds per configuration:

| Allowed fill engine | Cycles (all three seeds) | Fused sites |
|---|---:|---:|
| none (prior baseline) | 1014 | 62/66 |
| alu | 1074 | 1/70 |
| load | 1045 | 7/65 |
| valu | 1074 | 0/62 |

Allowing one engine to fill every isolated jump destroys most all-variant
return-fusion opportunities: first conflicting variants exceed the selected
engine capacity at56 ALU sites,50 LOAD sites,47 VALU sites respectively.
This rejects blanket single-engine filling for this configuration, not
selective variant-aware filling. No submission artifact is replaced.
Since even the low-load/low-flow branch has arithmetic floor937, further
scheduling-only parameter screens cannot establish sub900; prioritize
arithmetic-work reductions or a different algorithm before revisiting fill.

## Depth-five padded triples (2026-09-09)

All use round5,arity3,cover-remainder,compact-return,recycle-stride,
relax-jump,no-packet-repair,keep-final,teacher-weight1024,policy4,
three full32 seeds and executed slot profiling. No vector-index or feedback.

| Groups | Extra options | Cycles | ALU | VALU | LOAD | FLOW | Packets |
|---|---|---:|---:|---:|---:|---:|---:|
| none (`--groups 0`) | none | 994 | 11486 | 5764 | 1891 | 940 | 1319 |
| 3 | none | 1000 | 11494 | 5765 | 1885 | 946 | 99629 |
| 3,4,5,7 | isolate-jumps,jump-address-lead2,fuse-returns | 1010 | 11518 | 5768 | 1861 | 964 | 394551 |

All three seeds agree in each row. Scratch1530 throughout. Single-group
actual default JSON is55739531 bytes, so this is not a small submission
alternative. Four-group fusion sees3 sites and rejects all by capacity.

Relative to the zero-group experimental baseline, arithmetic lane-work grows
by16 per group and FLOW by6 per group. LOAD falls by8 per group minus2 fixed
setup loads (one group saves6, four save30). Thus these depth-five lookups
trade load work for more arithmetic/control; they do not address the necessary
sub900 arithmetic reduction. The relaxed arithmetic floors are960,961,962.
Do not scale this exact lowering to more groups expecting to close that gap.
This is not a proof against alternative depth-five representations/lowerings.
No best-score submission artifact was replaced.
