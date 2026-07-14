"""Verify an environment-configured structural variant and exact schedule."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import random

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target


def main() -> None:
    configure_target()
    schedule = json.loads(Path(os.environ["SCHEDULE"]).read_text())
    kernel.SCHEDULE_EXACT_CYCLES = schedule["cycles"]

    # Import only after mutating the shared kernel module so the cached builder
    # in verify.py is constructed with this variant rather than the default.
    from codex.verify import optimized_builder, run_once

    if bool(int(os.environ.get("DIAGNOSE", "0"))):
        from frozen_problem import (
            Input,
            Machine,
            N_CORES,
            Tree,
            build_mem_image,
        )

        random.seed(0)
        forest = Tree.generate(10)
        inp = Input.generate(forest, 256, 16)
        mem = build_mem_image(forest, inp)
        builder = optimized_builder()
        machine = Machine(
            mem, builder.instrs, builder.debug_info(), n_cores=N_CORES
        )
        machine.enable_pause = False
        machine.enable_debug = False
        try:
            machine.run()
        except Exception:
            core = machine.cores[0]
            pc = core.pc - 1
            bundle = machine.program[pc]
            print(
                f"failure_cycle={machine.cycle} pc={pc} "
                f"bundle={bundle} rewritten={machine.rewrite_instr(bundle)}"
            )
            for slot in bundle.get("load", ()):
                if slot[0] == "load_offset":
                    _, _, address, offset = slot
                    print(
                        f"load_address_reg={address + offset} "
                        f"value={core.scratch[address + offset]} "
                        f"mem_size={len(machine.mem)}"
                    )
            failing_bases = {
                slot[2]
                for slot in bundle.get("load", ())
                if slot[0] == "load_offset"
            }
            print(
                "virtual_assignments_on_failing_bases="
                + repr(
                    {
                        name: assignment
                        for name, assignment in getattr(
                            builder, "virtual_workspace_assignment", {}
                        ).items()
                        if assignment[0] in failing_bases
                    }
                )
            )
            raise
        print(f"diagnostic run completed in {machine.cycle} cycles")

    seed_count = int(os.environ.get("SEEDS", "8"))
    cycles = [run_once(seed) for seed in range(seed_count)]
    builder = optimized_builder()
    slots = Counter(
        engine
        for bundle in builder.instrs
        for engine, engine_slots in bundle.items()
        for _ in engine_slots
    )
    dag_slots = Counter(op.engine for op in builder.dag_ops)
    print(
        f"cycles={cycles[0] if cycles else schedule['makespan']} "
        f"seeds={len(cycles)} "
        f"scratch={builder.scratch_ptr} ops={len(schedule['cycles'])}"
    )
    print("slots=" + " ".join(f"{key}:{slots[key]}" for key in sorted(slots)))
    print(
        "dag_slots="
        + " ".join(f"{key}:{dag_slots[key]}" for key in sorted(dag_slots))
    )


if __name__ == "__main__":
    main()
