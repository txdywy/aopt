# Bounded select / first-hash fusion probe

Target `(x XOR node[x&1])*4097 + 0x7ed55d16` modulo 2^32.
Nodes for the negative test: 0x12345678, 0x9abcdef0. Four arbitrary
preprocessed constants, inputs x and its parity, two instructions.

`probe_select_hash_fusion.py` finite constraints returned UNSAT for both
orders of MAD with XOR, AND, OR, and for MAD followed by ADD or SUB.
ADD→MAD and SUB→MAD timed out (UNKNOWN), not solver proofs.

Those two remaining templates have an independent simple exclusion: with
parity fixed, ADD/SUB followed by MAD is a polynomial in x of degree at most
two, even if either instruction is bypassed or operands repeated. Its third
forward difference with stride two must vanish modulo 2^32. The target has
nonzero third differences on that same fixed-parity domain (direct integer
evaluation over x<128). Thus these templates cannot represent this node pair.

Positive controls: identical nodes with XOR→MAD synthesize and verify across
all 32-bit inputs. Correlated nodes 0 and 1 also synthesize XOR→MAD and verify
under parity=x&1. No full kernel changes, no faster score. This excludes only
the stated two-instruction families; shifts, additional runtime intermediates,
cross-round state transformations, and alternative table layouts remain outside
scope. Do not continue increasing SMT time for the polynomial templates.

## Shift extension and stopping rule

With the same correlated target and arbitrary four constants, SHL→MAD,
MAD→SHL, and MAD→logical-SHR are sample UNSAT. Logical-SHR→MAD returns
UNKNOWN at 8 seconds. Shift amounts are arbitrary selected operands, not
restricted literals. UNKNOWN is an open case, not an exclusion.

Stop expanding this two-op branch for now: no implementable candidate emerged,
and its maximum saving applies to only 249 selections unless a substantially
different global representation is introduced. Reopening should be driven by
a concrete identity or available intermediate, not another timeout increase.

Cross-round state candidate to assess next: T(x)=x XOR (x>>16) is an
involution and commutes with XOR. Storing pre-final-T state lets node values be
pretransformed by T, moving T across the node XOR. However, this alone merely
moves a shift/XOR pair from one round's end to the next round's beginning;
it does not delete it. A useful design must merge that moved T with another
stage, and account separately for parity extraction and initial/final conversion.

`prototype_prefinal_state.py` implements that conjugated recurrence at the
algorithm level and matches all final values AND indices for 40 full32 inputs.
State is T(value XOR C5), transformed nodes are T(node XOR C5), and each
round computes prefinal(T(state XOR transformed_node)). The correct next
branch is `(state XOR (state>>16) XOR C5)&1` after the update.

Consequently the round-input T still costs a shift/XOR pair; parity now needs
the unmaterialized output T as well (or another equivalent parity circuit).
Current representation already has the output T for free when extracting its
low bit. This simple conjugation is therefore not a work reduction, even before
node preprocessing and boundary conversions. Do not lower it into ISA unless
a second identity eliminates one of those transformations. The prototype is
not a new performance result.

## Runtime special-pair coverage

`profile_node_pair_structure.py` screens 40 deterministic full32 random trees:
40920 sibling pairs, including 600 pairs at cached depths 1..4. None are equal,
none differ in at most two bits, and none differ only in the low/sign bits.
This is empirical coverage, not a restriction on legal inputs or a proof that
such pairs never occur. For uniformly random nodes the at-most-two-bit event
has probability (1+32+496)/2^32, so special-case dispatch for these sparse
differences is not a promising typical-workload path. Any such optimization
would still require a correct general fallback and charge runtime detection.
Do not specialize generated code to these observed node values.
