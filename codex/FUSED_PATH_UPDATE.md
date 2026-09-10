# Single-operation path-update screen

Run `python -m codex.prove_fused_path_update` with z3-solver installed.

Targets are packed path `2*m+(q&1)` and original physical address
`2*a-5-(q&1)`. Each candidate is one basic binary ALU operation or one
VALU multiply_add. Every operand independently selects either input or one
of three arbitrary uint32 constants, including repeated input selections.

All eighteen opcode/target cases returned UNSAT on28 necessary input pairs.
The packed-path examples use m in0,1,2,7; physical examples use a in14,15,16,21,
all valid depth-3 addresses. Thus a general rewrite valid for arbitrary hash
words and these legal paths cannot consist of one of these single operations.
The positive control proves the existing parity-plus-update expression equals
`m+m+(q%2)` over all uint32 inputs.

This does NOT exclude fused multi-operation transformations, additional
available hash intermediates, changed path encodings, division/select opcodes,
or exploiting a separately proved correlation between hash and path. No
kernel was changed and no new cycle score was obtained. Stop searching this
narrow one-op replacement family; pursue cross-round representation changes
or work elimination involving hash intermediates instead.
