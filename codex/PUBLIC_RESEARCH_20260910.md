# Public-source refresh, 2026-09-10

Goal remains a verified complete kernel below 900 executed cycles, with a
browser-serializable submission. No new complete-kernel score this refresh.

Read first-party sources:

- https://github.com/fiigii/ai-comp/blob/main/Readme.md
  Generic compiler, configurable HIR/LIR/MIR/VLIW pipeline; README inspected
  does not establish a sub-900 result.
- https://raw.githubusercontent.com/fiigii/ai-comp/main/compiler/pass_config.json
  Concrete next inspection targets: `mad-synthesis`, `slsr`, and SROA table
  promotion. Configuration enables MAD synthesis after SLP and CSE. This is
  not evidence of a novel algebraic identity; inspect implementation before
  claiming novelty. Scheduling already includes scalarization and staggering,
  which are not sufficient reasons to replace our existing backend.
- https://github.com/developertogo/beat-opus-4.5-challenge
  Author reports 1107 cycles; source is explicitly not published. Described
  unrolling/renaming/SIMD are already represented locally. Not a sub-900 lead.

Searches for the 892-cycle implementation again surfaced Weng's historical
July 12 snapshot, not an independently verified current leaderboard or source.
Chrome CDP preflight failed; public web reads only, no account operations.

Next bounded action: inspect ai-comp's MAD synthesis and straight-line strength
reduction code, compare actual rewrites against our 10-op hash. Do not import
or execute an entire remote compiler merely because its pipeline looks useful.

## Follow-up: implementation inspected

Read the public raw `compiler/passes/mad_synthesis.py` and `slsr.py`.
MAD synthesis matches a single-use vector multiplication feeding addition;
this is already present in our hash lowering, not a new hash identity.

SLSR changes state to x+k and propagates compensation through affine chains.
A uniform offset requires (A-1)k=C modulo 2^32. Independently checked with
Python integer gcd arithmetic for our four MAD stages:

| A | C | gcd(A-1,2^32) | Uniform offset exists |
|---|---|---|---|
|4097|7ed55d16|4096|no|
|33|e9f8cc1d|32|no|
|16896|accf6200|1|yes|
|9|fd7046c5|8|no|

The one solvable branch feeds XOR, not an affine recurrence. The inspected
SLSR implementation restricts outside consumers to compensable additions or
subtractions (or dead uses), so it does not directly absorb that offset through
the XOR join. Also, deleting a constant addend from an already fused MAD does
not by itself save an instruction. These observations reject direct adoption,
not all possible state representations or nonuniform compensation schemes.

Public API directory lookup failed (403); raw web reads succeeded. No remote
code executed or copied into the submission. Next algebraic candidates need
to cross a nonlinear XOR boundary, or improve scalar lowering of existing MADs,
rather than merely normalize affine constants.
