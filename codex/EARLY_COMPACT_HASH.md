# Compact983 final-group load/hash advance

Reproduce with `python -m codex.audit_compact_liveness --verify --early-loads --early-hash`.

After conservative liveness cleanup, move address calculations for lanes1,2,5
to966; load lanes1,5 at967 and lanes2,6 at970. Remove corresponding old copies
from the main stream and every relevant jump-table return variant. All eight
node XORs for value599 move to972, and its first hash MAD moves from975 to974.
All packet engine limits are checked. Twenty full32 seeds and three boundary
patterns pass complete output comparison, each at983 cycles. JSON is4512152
bytes. This is a verified prototype, not a faster exported submission.

Adding this MAD to every return974 table packet instead would overflow VALU
at packet5408 (seven slots); this rejected candidate never reached execution.

With `--reschedule-tail`, complete window974..983 is tested at nine cycles.
It remains UNSAT by a ten-cycle dependency floor:121 operations,364 edges,
digest`f77a0290a9f674929e5fadf58881891464bfdcb3392261fc5c046af97c8180b2`.
The advance removed one late input obstacle but did not eliminate every
critical path. Do not report a new cycle score from the earlier hash start.

## Remaining critical chain

`--show-chain` reports the surviving ten-cycle chain for value679:
vselect → first MAD → XOR constant → XOR shifted value → middle MAD →
middle XOR → final MAD → final XOR constant → final XOR shift → output store.
These are true value dependencies, not register-name-only hazards. Register
renaming cannot remove this chain. Advancing value599 alone is therefore
insufficient. The first MAD for value679 also cannot simply move from964
to963 in the preceding hash: its input is produced by vselect at963, and
the frozen machine reads old scratch values within a packet.

The useful next boundary is selection/hash fusion or changing when selection
inputs become available, not another retry of the same terminal scheduler.
This explains why the verified earlier599 prototype retains983 cycles.
