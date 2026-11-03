# Replan against public959 reference

Read on2026-09-07:
- https://github.com/RubySunset/original_performance_takehome
- https://raw.githubusercontent.com/RubySunset/original_performance_takehome/main/perf_takehome.py
- https://pengchengneo.github.io/posts/anthropic-vliw-simd-kernel-optimization/

Public author reports959 cycles, with5719 VALU,11392 ALU,946 FLOW and1891 LOAD
slots. This is an external report, not a locally reproduced score. The blog
explicitly attributes its checkpoint to RubySunset; these are not two
independent implementations. The source exposes virtual-register IR and
late binding, flexible engine placement and a rescheduling search. Do not
copy its implementation or substitute its claimed score for our verified980.

Independent comparison: our recorded981 operation mix has5657 VALU and10891
ALU, hence56147 arithmetic lane-slots, versus57144 in the reported reference.
Our smaller arithmetic workload does not translate into a shorter runtime.
The reference's arithmetic relaxation is ceil(57144/60)=953; ours is936.
This counts MAD as one VALU operation and is only a relaxed resource bound.
It is not proof that either schedule can attain that bound, and it is not a
global lower bound across all possible algorithms.

Decision: keep the correct early-tail prototype but stop treating more tail
readiness as the principal route to899. Split follow-up into two gates:

1. Global scheduling: regenerate flexible placement priorities from realized
   schedules, preserving instruction semantics and explicit table-return
   constraints. Test whether the existing reduced operation mix can approach
   its own bound before adding more lookup tables.
2. Work reduction: a candidate for899 needs at least2207 arithmetic lane-slots
   removed from the recorded981 mix (56147-899*60), plus LOAD reduction.
   Reject candidates whose combined resource demands already exceed899 even
   before doing long schedule searches. Engine reassignment alone cannot
   remove that lane-work deficit.

The current static scoreboard fetch contains headings but no ranking rows;
no new live rank has been verified. Chrome CDP check failed; only public
read-only web extraction was used. No login or upload occurred.

## Executed backward/forward feedback screen

Added opt-in `--feedback-rounds` to our existing paired-table driver. It
preserves the best forward schedule while using later forward issue times
as backward priorities, then later backward issue times as forward priorities.
This is independently implemented on our own DAG; no external code imported.

The recorded981 configuration with `--feedback-rounds 4` produced pre-compaction
lengths985,999,1000,999,1001. The original985 was retained, compacting to981.
Full program assembly and3 full32 frozen-reference executions passed at981;
all5 `test_multiround_tables` tests passed. The four worse candidates were
screened, not individually executed or claimed correct. Delivered980 intact.

Interpretation: simple unconditioned reverse/forward feedback does not improve
this graph. The next scheduler experiment must preserve table adjacency and
the original phase priorities rather than replacing them wholesale. Repeating
the same four feedback passes has no evidence of value.

## Anchored feedback produces an independently verified980

Added `--feedback-percent` (default100 preserves the earlier experiment).
Each forward pass blends normalized original external priorities with the
backward completion priorities. Screening four rounds per ratio:

| Feedback percent | Best pre-compaction | Compacted preflight |
| --- | ---: | ---: |
|5|985|981|
|10|984|980|
|15|984|981|
|25|984|980|
|50|985|981|

The10-percent configuration was fully assembled and executed on20 full32
seeds: all980,4064537 packets, scratch1536. Reproduce the recorded981 driver
command with `--feedback-rounds 4 --feedback-percent 10 --seeds 20 --pattern full32`.
This result does not include the standalone980 tail rewrite: it gains one
cycle through scheduling alone. It ties, not beats, the existing best980.
No standalone export or official full-suite verification is claimed for this
new variant; delivered files remain unchanged.

Important evidence:15 percent also shrinks the raw schedule but loses the
gain during compaction. Thus selecting only by raw length is insufficient;
future search must rank fully compacted candidates and reject invalid table
adjacency before claiming improvement. Combining this scheduling change with
the earlier tail rewrite is a hypothesis, not an additive-cycle guarantee.

## Combined execution reaches979

`/tmp/aopt-smt/bin/python -m codex.inspect_feedback_tail --combine` captures the
assembled10-percent feedback program, locates the shifted final special table
at968 and all64 variants returning969, and applies the speculative node-XOR
rewrite. It reschedules the straight-line tail at969 with a10-cycle horizon,
preserving all absolute out-of-line table addresses.

SAT:131 operations,288 edges. All20 full32 seeds and3 boundary patterns match
the frozen reference at979 executed cycles. Result recorded in
`feedback_tail_979_result.json`. This is a new experimental best, not yet a
standalone upload artifact or full official-suite result. Existing980 and981
submission files remain intact. The below900 objective remains unmet.

Follow-up validation expanded to100 full32 seeds plus3 boundary patterns,
then ran all9 unchanged official tests using the already generated ISA list
(avoids a second4-million-packet allocation). All outputs match and every
execution is979 cycles. This validates the generated program, not a future
standalone export or browser memory compatibility. Result JSON updated.

##2026-09-08: nearby-weight screen closes this local sweep

Four feedback rounds at weights7,8,9,11,12,20 produced compacted preflight
lengths981,981,980,980,980,980 respectively. All six subprocesses exited0;
none was executed as a new kernel because none improved the already verified
10-percent scheduling result. These are scheduling screens, not new scores.
Together with the earlier5/10/15/25/50 tests, this is evidence of a local
plateau, not a reason to continue fine-grained weight sweeps. Retain10 percent
and focus subsequent work on changing the operation mix or a substantially
different scheduling representation. Best executed result remains979.

## Scalar-cache work reduction now retains979

The old scalar level4-cache rewrite formerly cost a cycle with the original
calendar. Combining it with10-percent feedback produces a980 pre-tail program.
`inspect_feedback_tail --combine --scalar-cache` then passes100 full32 seeds,
3 boundary patterns and all9 official tests at979. The pre-tail DAG changes
from10895 ALU/5657 VALU to10911 ALU/5641 VALU:16 broadcasts become16 scalar
copies, reducing arithmetic lane-work by112 without increasing executed cycles.
This is reduced work at the same best score, not a sub979 claim. Separate
result recorded in `feedback_scalar_tail_979_result.json`; neither standalone
submission was overwritten. Keep this lower-work variant for further work.

## Preprocessing on the lower-work feedback graph

Screened depths5,6,8 with scalar cache,10-percent feedback and4 rounds.
Best raw lengths986,987,1010 compact to984,985,1006, respectively. All three
screens exited0; no execution score is claimed for them. Depth7 remains the
980 pre-tail/979 combined baseline. The lower-load alternatives still lose
to additional hash-node transformations and schedule effects in this graph.
This closes the uniform-depth sweep for this configuration, not every
possible selective or partial-node preprocessing design.

## Local engine-placement screen

On the scalar-cache/10-percent-feedback configuration, restore-hash-count
80,84,92,96 each produced raw lengths986,984,984,984,984 and compacted preflight
980, matching restore88. All four screens exited0, but no new executed score
is claimed. This closes this nearby placement sweep; retain88 rather than
changing parameters without an end-to-end gain. Standalone979 remains best.
