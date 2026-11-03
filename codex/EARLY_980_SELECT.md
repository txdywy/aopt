# Earlier node selection experiment

`python3.12 -m codex.experiment_early_980_select` changes an in-memory copy of
the980 recipe, preserving the delivered file. The first FLOW selection at951
is replaced by scalar arithmetic:

-947: one node difference into dead scalar431.
-950: eight condition-times-difference multiplications into711..718.
-953: eight base additions complete the selected vector.

FLOW951 now produces the previously957 candidate in dead vector439..446;
FLOW957 combines it with711; FLOW958 performs the final selection previously
at959. FLOW959 is freed. Main-stream live-range checks and checks on table
recipes returning into that live interval protect the reused scratch words.
Earlier table accesses are irrelevant after their old output values are dead.

Twenty full32 seeds passed at980 cycles. This moves node availability earlier
by one packet but does not advance its consumer: the XOR at960 still waits for
ALU capacity. Packet959 has four table-index multiplications and eight unrelated
hash operations, filling all12 ALU slots. Thus earlier selection alone is not
a cycle improvement. It adds17 ALU operations and removes one FLOW operation.
Keep this prototype separate from the best submission; any follow-up must
justify its added work with an actual cycle reduction.

The follow-up run also passed zero, all-ones and alternating boundary inputs
(23 total cases), all980. The delivered980 file retains SHA256
`7628ae65327ed3d3489cbf3d07d66a53e4add2c47c23b3c62498b0190323b529`.

## Advancing the first hash stages

`--advance-hash` moves the group31 input XOR to VALU959, moving the value591
XOR to eight ALU lanes at960. The group31 first MAD moves961->960 while the
value591 MAD moves960->961. This exchange adds no arithmetic lane-work beyond
the17 ALU operations already introduced above. Both the main placeholder and
the large-table recipe returning to962 are patched consistently.

`--advance-hash --advance-front` additionally moves value639's join961->962
and value583's MAD961->963. It uses the freed slots to advance group31 shift
and constant-XOR962->961 and its join963->962. The latter packet has64 special
table realizations, all patched together with the main-stream placeholder.

Both configurations separately passed20 full32 seeds plus3 boundary patterns
at980 cycles. The earlier readiness has not propagated through the remainder
of the hash, so this is not a new score or a replacement upload artifact.

## Middle and candidate advance

`--advance-hash --advance-front --advance-middle` passed all23 cases at980.
Group31's penultimate hash join is now967 instead968. Adding
`--advance-candidates` also passes all23 cases at980: the parity predicate and
first six lanes of both final candidates are now ready after968 instead969.
The value591 XOR is scalarized into eight ALU slots to respect VALU capacity.

An initial attempt moving the original unaligned candidate instructions failed
seed5: writing429/430 at968 destroys cached nodes read by table969. The corrected
version uses aligned destinations431 and447, sources687/943 and679. The last
two provisional lanes are overwritten by the table's fused XOR at969. This
preserves those cached nodes and has passed the full23-case experiment.

No end-to-end cycle reduction is claimed: the last two candidate lanes still
arrive at969, whose FLOW slot is a table return. The final vselect remains970.
The next necessary change is therefore in that table/control boundary, not
another advance of the already-ready first six lanes. Submission980 is intact.
