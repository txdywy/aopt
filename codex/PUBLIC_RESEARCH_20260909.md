# Public research, 2026-09-09

Searches for892-cycle source and sub900 hash implementations did not locate
a new usable implementation. Weng Jialin's article still describes a July12
snapshot, not a verified current leaderboard:
https://wengjialin.com/blog/agent-harness-performance-takehome/

New relevant primary-source excerpt: *Superoptimization for the EMCA DSP*,
section5.4,page60, retrieved through the search index of the university PDF:
https://liu.diva-portal.org/smash/get/diva2%3A2072079/FULLTEXT01.pdf
Full PDF fetching failed twice; do not claim to have read the complete thesis.
The excerpt calls for a target-specific cost model including code size and
parallelism, and proposes backend evaluation to include global constraints
such as register pressure. These are future-work proposals, not a demonstrated
solution for this challenge. Our application: retain actual execution and
serialization checks as gates; expression instruction count is only a screen.

The accompanying new experiment broadens C1-absorption from hand-selected
affine formulas to operand-selecting SSA programs. See HASH_C1_ABSORPTION.md.
This experiment is independently designed, not an implementation copied from
the thesis. It has not produced an improved cycle score.

## Public implementation recheck after the hash screens

Reopened https://github.com/wouterkool?tab=repositories and
https://wouterkool.github.io/ . The retrieved29-repository listing and homepage
show routing/ML/optimization projects, not an identified VLIW kernel solution.
This is not proof no public solution exists. The VLIW site's static response
does not expose readable leaderboard rows; current standings were not verified.
Search still finds Weng's historical July12 snapshot, not new892-cycle source.

New primary research: KernelPro, arXiv2606.26453v2, revised2026-08-08:
https://arxiv.org/html/2606.26453v2
Read the abstract/introduction and section index, not the entire paper.
It separates bottleneck interpretation from code generation and describes
search branching/pruning to escape local optima. CUDA-specific performance
claims do not establish a speedup on this VLIW ISA. The paper states code
will be released upon publication; no implementation was obtained here.

Decision for this workspace: stop serially expanding nearby SMT templates
as the sole optimization strategy. These exclusions are useful, but have
not improved the executable score. Re-open an algorithm-level branch while
retaining unresolved synthesis cases as secondary work. Any candidate must
reduce the measured arithmetic/load/FLOW deficits, pass unchanged correctness
tests, and remain serializable. No new best-score claim follows from research.
