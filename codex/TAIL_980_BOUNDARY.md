# Boundary of the current 980 tail

The standalone980 recipe was checked against standalone981. All62 large table
recipes are unchanged, as are main packets0..968. Outside the main stream,
exactly64 special table entries change; all preserve their return to970.
No absolute table addresses move. Two new regression tests pass.

The fixed instruction graph of packets970..979 contains131 operations and288
dependency edges. A9-cycle exact scheduling query is UNSAT, with a10-cycle
dependency floor. The chain runs through final vselect, hash MAD, XOR/shift
join, middle MAD/join, final MAD/join and output store. Graph digest:
`dd640e4a9341fe4770a289d9c1e1ec654967209868b9274ebe25bc9cd00d0dc9`.

This proves only that these fixed tail operations cannot fit9 cycles. It does
not rule out earlier speculation, another hash representation, another table
layout, or a different global algorithm. It is not a global980-cycle bound.

At packet969 the predicate and both speculative candidate vectors are produced.
They are unavailable at the start of that packet, so merely moving vselect
from970 to969 would read stale data. Further latency reduction requires moving
their producers earlier or computing a different equivalent representation.
Moving MAD onto both candidates at970 followed by selection971 only exchanges
the order of two serial stages and does not itself shorten the chain.

Preserve the delivered980 file. Deprioritize retiming this identical terminal
window; investigate earlier producers or whole-kernel work reductions instead.

## Producer trace

`trace_980_producers.py` executes the frozen machine and records per-lane RAW
producers (all reads precede writes within a packet). Seed0 still executes980
cycles. Predicate1191 and candidate words431/437/447/453 share the group31
hash chain: selected node at959, XOR960, first MAD961, then consecutive hash
stages through968, followed by predicate/candidate generation969.

Earlier node selection uses three FLOW vselects at957,958,959. The predicate
for the final one is ready after957. The first two selectors do not use that
predicate, but both overwrite1191, also used by the preceding hash through956.
This identifies a concrete next experiment: separate the selector temporary
from the live hash temporary and look for earlier FLOW capacity. Merely moving
the existing writes earlier would corrupt the preceding hash.

Trace results intentionally exclude control, memory, WAR and WAW dependencies;
they identify candidate producers, not a proof that moving them is legal.
Unaligned vector operations are traced lane-by-lane to avoid attributing a
live output lane to unrelated dead leading lanes from another SIMD group.
