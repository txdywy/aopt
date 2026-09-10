# Arithmetic work decomposition of low-load991

Current pre-assembly DAG (scalar cache, expanded final tables):11007 ALU,
5641 VALU. Executed profile differs slightly due to table assembly; use
`final_gather12_feedback_result.json` for the measured899 deficit of2191.

Largest non-hash tag costs in scalar-equivalent lane units:

|Tag|Lane work|
|---|---:|
|node_xor|3744|
|mirror_bit plus mirror_bit_vector|3584|
|mirror_update|2064|
|gather_address|1696|
|dynamic_node_transform_vector plus scalar|768|

The ten hash stages each cost4096 units when scalar/vector variants are
combined. A genuine one-operation-per-hash elimination would save4096,
but existing synthesis failures do not constitute a general impossibility
proof. Address/parity representation is the other large target: removing
all gather_address work alone is insufficient for2191.

Graph probe of existing OVERLAP_DEEP_ADDRESS=True on the same configuration:
1773 LOAD,11359 ALU,5642 VALU,895 FLOW,62 STORE. This saves8 DAG loads but
adds360 arithmetic lane-work. It also exposes123 DAG-unconsumed
gather_address_seed scalar definitions, largely on round15 where table
copies replaced gathers. Even removing all123 would leave237 added units.
No executable score or correctness claim is made for this graph probe.

Conclusion: simply enabling the existing overlap-address mode is not a
work-reducing solution. Any new address representation should explicitly
target parity extraction plus mirror/address updates together, rather than
only changing which cycle performs the same arithmetic. The mixed-table
interaction also needs auditing before treating those123 definitions as dead.

## Affine-coordinate proof

`prove_affine_address_encoding.py` proves two uint32 identities for binary p
and original recurrence a_next=2*a-5-p:

-b=a-6 gives b_next=2*b+(1-p).
-b=5-a gives b_next=2*b+p.

Both could look like one MAD after parity is available, but neither is a
free address optimization. The positive encoding needs complemented parity
and relocation from physical address a to a-6 (or an extra conversion per
gather). The negative encoding wraps outside valid tree memory. A whole-tree
vector relocation needs256 loads and256 stores before accounting for fusion
with existing preprocessing; the low-load candidate has only7 spare LOAD
slots at899. Complemented parity production must also be charged explicitly.
This excludes treating a mere coordinate rename as elimination of address
work. It does not exclude selective relocation or a different hash encoding.

Whole-level selective relocation screen (`screen_selective_relocation.py`):
the measured gather_address costs by depth are160 at depth4 (rounds4+15),
and256 each at depths5..10. Optimistically reusing every existing preprocessing
load makes depths4..7 free to relocate in LOAD terms, saving at most928 address
lane operations. Depth8 alone needs32 new vector loads against only7 spare
slots at899; depths9/10 cost still more. Exhaustive128 subset enumeration
therefore leaves1263 arithmetic units even under zero conversion/parity/store
overhead assumptions. This narrow family cannot be the entire sub900 design;
it would need an independent saving in parity/hash/path work as well.

The single-operation address-offset screen now also covers unsigned division,
modulo and ceiling division. For q op C, all three are UNSAT for
-5-(q&1), with nonzero divisor enforced. C op q is undefined at q=0 and
therefore cannot be a total uint32 replacement without extra instructions.
Ceiling division uses a33-bit addition intermediate to match the frozen
Python simulator, not a wrapping32-bit add. This extends only the narrow
constant-operand family exclusion; it is not a global address lower bound.
