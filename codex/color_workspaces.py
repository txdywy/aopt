"""Color shared path workspaces for a full-round wavefront."""

from __future__ import annotations

from z3 import Int, Or, Solver, sat


OFFSETS = (
    0, 0, 0, 0, 6, 6, 6, 8, 8, 8, 8, 8, 11, 11, 13, 13,
    13, 17, 17, 15, 15, 15, 19, 19, 20, 20, 20, 12, 12, 22, 22, 22,
)


def solve(n_workspaces: int, stride: int, first_cache=frozenset(), final_cache=frozenset(range(18))):
    colors = [Int(f"c{group}") for group in range(32)]
    solver = Solver()
    for color in colors:
        solver.add(color >= 0, color < n_workspaces)

    # A group's first four rounds retain three path bits in its base color.
    for g in range(32):
        end = OFFSETS[g] + (4 if g in first_cache else 3)
        for h in range(g + 1, 32):
            other_end = OFFSETS[h] + (4 if h in first_cache else 3)
            if OFFSETS[g] <= other_end and OFFSETS[h] <= end:
                solver.add(colors[g] != colors[h])

    # Later shallow selects write ephemeral conditions.  They may not land in
    # a workspace while another group still owns persistent first-path bits.
    writes = [(13, 2), (14, 3)]
    writes += [(15, 4)]
    for owner in range(32):
        end = OFFSETS[owner] + (4 if owner in first_cache else 3)
        for writer in range(32):
            for rnd, multiple in writes:
                if rnd == 15 and writer not in final_cache:
                    continue
                when = OFFSETS[writer] + rnd
                if OFFSETS[owner] <= when <= end:
                    solver.add(
                        colors[owner]
                        != (colors[writer] + multiple * stride) % n_workspaces
                    )

    if solver.check() != sat:
        return None
    model = solver.model()
    return tuple(model[color].as_long() for color in colors)


if __name__ == "__main__":
    for n in range(8, 13):
        for stride in range(n):
            assignment = solve(n, stride)
            if assignment is not None:
                print(n, stride, assignment)
