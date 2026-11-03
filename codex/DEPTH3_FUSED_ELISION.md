# Depth-3 fused lookup audit, 2026-09-07

Public sources read this turn:

- https://wouterkool.github.io/ lists combinatorial optimization and vehicle-routing research. It does not describe the author's VLIW kernel. No kernel technique or cycle result can be inferred from these papers alone.
- https://vliw-challenge.fly.dev/ returned only the scoreboard shell through public text retrieval, not current score rows. The current ranking was not verified.
- Search rediscovered https://github.com/fiigii/ai-comp but no new executable sub-900 solution. Browser CDP was unavailable; public read-only retrieval was sufficient for the pages above.

## Independent experiment

All 32 groups use depth-3 four-lane tables in round 3. This replaces
the depth-3 selection stage, not depth-4 tables. Its initial DAG contains
32 unconsumed `dynamic_node_transform_vector` operations: fused table copies
already XOR the transformed cached node into `value`; the subsequent temporary
node transform has no consumer.

Default-off `OMIT_FUSED_DIRECT_NODE_TRANSFORM` omits this operation only when
`FUSED_DIRECT_BRANCH_XOR` is enabled and the union of individual and paired
direct lanes covers all eight lanes. Partial coverage and non-fused copies
retain the transform. This is explicit semantic elimination, not blanket DAG
dead-code deletion: assembled table sources have implicit consumers.

Reproducer:

```sh
/opt/homebrew/bin/python3.12 codex/experiment_paired_base.py \
  --round 3 --group-ids 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 \
  --arity 4 --keep-final --compact-return --recycle-stride --relax-jump \
  --no-packet-repair --prioritize-copies --parallel --restore-hash-count 88 \
  --dispatch-lead 32 --teacher-weight 1024 --last-group-lead 8 --policy 3 \
  --seeds 20 --pattern full32 --omit-fused-node-transform
```

With elimination: 20 full32 inputs correct at **1018 cycles**, 263487 packets,
scratch 1536. Without elimination: 3 full32 inputs correct at **1019 cycles**,
263488 packets. DAG VALU 5916 -> 5884; ALU 11295, LOAD 1878, FLOW 843, STORE 62
unchanged. These are DAG counts, not an executed-resource bound.

This does not improve the best 981-cycle experiment. It saves 256 arithmetic
lane-work units in this depth-3 family, but the family still carries the normal
depth-4 selection cost. A mixed-depth design must quantify both lookup stages
and their index preparation; combining individually promising tables is not
automatically beneficial. Current direct-table machinery has a global depth,
so mixed-depth execution is not implemented or validated here.
