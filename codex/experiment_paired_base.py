"""Compare serial and pipelined paired-table preparation on the frozen ISA."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from codex import perf_takehome_under1000 as kernel
from frozen_problem import Input, Machine, N_CORES, Tree, build_mem_image, reference_kernel2


def source_priorities(source_ops, target_ops, source_cycles, weight, dispatch_lead):
    """Reuse source issue cycles by operation identity without solver imports."""
    def key(op):
        if op.identity:
            return op.engine, "identity", op.group, op.round, op.identity
        return op.engine, op.slot[0], op.tag, op.group, op.round

    exact = defaultdict(deque)
    group_round = defaultdict(list)
    tags = defaultdict(list)
    lookup = defaultdict(list)
    for op, cycle in zip(source_ops, source_cycles):
        exact[key(op)].append(cycle)
        group_round[op.group, op.round].append(cycle)
        tags[op.tag].append(cycle)
        if op.tag == "node_xor":
            lookup[op.group, op.round].append(cycle)
    scores = []
    for op in target_ops:
        matches = exact[key(op)]
        if matches:
            cycle = matches.popleft()
        else:
            candidates = (
                lookup.get((op.group, op.round))
                if "direct_branch" in op.tag or "direct_table" in op.tag
                else None
            )
            candidates = candidates or group_round.get((op.group, op.round)) or tags.get(op.tag) or [0]
            cycle = int(statistics.median(candidates))
        if (
            ("direct_branch" in op.tag or "direct_table" in op.tag)
            and not op.tag.endswith(("jump", "copy", "target"))
        ):
            cycle -= dispatch_lead
        scores.append(-cycle * weight)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, default=15)
    parser.add_argument("--group-ids", default="")
    parser.add_argument("--also-final-group-ids", default="")
    parser.add_argument("--arity", type=int, choices=(2, 4), default=2)
    parser.add_argument("--round", type=int, choices=(3, 4, 5, 14, 15), default=15)
    parser.add_argument("--graph-only", action="store_true")
    parser.add_argument("--schedule-only", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--no-packet-repair", action="store_true")
    parser.add_argument("--keep-final", action="store_true")
    parser.add_argument("--compact-return", action="store_true")
    parser.add_argument("--recycle-stride", action="store_true")
    parser.add_argument("--alu-stride", action="store_true")
    parser.add_argument("--relax-jump", action="store_true")
    parser.add_argument("--vector-index", action="store_true")
    parser.add_argument("--horner-index", action="store_true")
    parser.add_argument("--omit-level4-diffs", action="store_true")
    parser.add_argument("--scalar-level4-cache", action="store_true")
    parser.add_argument("--omit-fused-node-transform", action="store_true")
    parser.add_argument("--vector-depth3-index", action="store_true")
    parser.add_argument("--prioritize-copies", action="store_true")
    parser.add_argument("--dynamic-vector-to-alu", action="store_true")
    parser.add_argument("--lower-tail", action="store_true")
    parser.add_argument("--fuse-returns", action="store_true")
    parser.add_argument("--restore-hash-count", type=int, default=0)
    parser.add_argument("--preprocess-depth", type=int, choices=(0, 4, 5, 6, 7, 8))
    parser.add_argument("--profile-slots", action="store_true")
    parser.add_argument("--policy", type=int, default=4)
    parser.add_argument("--teacher-weight", type=int, default=0)
    parser.add_argument("--feedback-rounds", type=int, default=0,
                        help="Refine forward priorities through backward/forward schedules")
    parser.add_argument("--feedback-percent", type=int, default=100,
                        help="Blend backward feedback with original priorities (0..100)")
    parser.add_argument("--dispatch-lead", type=int, default=0)
    parser.add_argument("--last-group-lead", type=int, default=0,
                        help="Advance group31 priorities relative to the unchanged teacher")
    parser.add_argument("--height-weight", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--pattern", choices=("normal", "full32", "zero", "ones", "alternating", "boundaries"), default="normal")
    args = parser.parse_args()
    if not 0 <= args.feedback_percent <= 100:
        parser.error("feedback-percent must be in0..100")
    if args.feedback_rounds:
        original_schedule = kernel.KernelBuilder._schedule
        def feedback_schedule(ops, policy, return_cycles=False, **kwargs):
            best, cycles = original_schedule(ops, policy, return_cycles=True, **kwargs)
            best_cycles = cycles
            anchor = kwargs.get('external_scores')
            if anchor is None:
                anchor = [-1024*c for c in cycles]
            anchor_peak = max(anchor)
            anchor = [s-anchor_peak for s in anchor]
            reverse = kernel.KernelBuilder._reverse_ops(ops)
            lengths = [len(best)]
            for iteration in range(args.feedback_rounds):
                # Reversed node i corresponds to original node n-1-i.
                # A later forward issue is a higher backward priority.
                backward, reverse_cycles = original_schedule(
                    reverse, policy, return_cycles=True,
                    external_scores=[c*1024 for c in reversed(cycles)],
                    height_weight=args.height_weight)
                # A later backward issue is an earlier forward priority.
                feedback = [c*1024 for c in reversed(reverse_cycles)]
                peak = max(feedback)
                weight = args.feedback_percent
                blended = [(weight*(s-peak)+(100-weight)*a)//100
                           for s,a in zip(feedback,anchor)]
                candidate, cycles = original_schedule(
                    ops, policy, return_cycles=True,
                    external_scores=blended,
                    height_weight=args.height_weight)
                lengths.append(len(candidate))
                if len(candidate) < len(best):
                    best, best_cycles = candidate, cycles
            print(json.dumps({'feedback_schedule_lengths':lengths,
                              'selected_packets':len(best)}), flush=True)
            return (best,best_cycles) if return_cycles else best
        kernel.KernelBuilder._schedule = staticmethod(feedback_schedule)
    if args.pattern == "boundaries" and args.seeds < 3:
        parser.error("boundaries requires at least three seeds to cover every pattern")
    source = None
    source_cycles = None
    if args.teacher_weight:
        source_cycles = list(kernel.SCHEDULE_EXACT_CYCLES)
        kernel.SCHEDULE_EXACT_CYCLES = []
        source = kernel.KernelBuilder()
        try:
            source.build_kernel(10, 2047, 256, 16)
        except ValueError as error:
            if str(error) != "exact cycle schedule does not match DAG":
                raise
    groups = sorted(kernel.FINAL_CACHE_SET if args.round >= 11 else kernel.FIRST_CACHE_SET)
    groups = (
        [int(group) for group in args.group_ids.split(",")]
        if args.group_ids else groups[: args.groups]
    )
    if any(group not in range(32) for group in groups):
        raise ValueError("groups must be in 0..31")
    kernel.PAIRED_DIRECT_BRANCH_LOOKUPS = {
        group: tuple(tuple(range(lane, lane + args.arity)) for lane in range(0, 8, args.arity))
        for group in groups
    }
    if args.also_final_group_ids:
        final_groups = [int(g) for g in args.also_final_group_ids.split(",")]
        if args.round != 4 or any(g not in range(32) for g in final_groups):
            raise ValueError("mixed-round tables require round-4 and valid round-15 groups")
        kernel.DIRECT_BRANCH_ROUND_OVERRIDES = {g: 15 for g in final_groups if g not in groups}
        kernel.PAIRED_DIRECT_EXTRA_ROUNDS = {g: (15,) for g in final_groups if g in groups}
        kernel.PAIRED_DIRECT_BRANCH_LOOKUPS.update({
            g: tuple(tuple(range(lane, lane + args.arity)) for lane in range(0, 8, args.arity))
            for g in final_groups
        })
    kernel.DIRECT_BRANCH_ARITY = args.arity
    kernel.ELIDE_DIRECT_BRANCH_TARGET = not args.compact_return
    kernel.COMPACT_PAIRED_DIRECT_RETURN = args.compact_return
    kernel.RECYCLE_PAIRED_DIRECT_STRIDE = args.recycle_stride
    kernel.ALU_PAIRED_DIRECT_STRIDE = args.alu_stride
    kernel.RELAX_PAIRED_DIRECT_JUMP = args.relax_jump
    kernel.VECTOR_PAIRED_DIRECT_INDEX = args.vector_index
    kernel.HORNER_PAIRED_DIRECT_INDEX = args.horner_index
    kernel.OMIT_LEVEL4_DIFF_SETUP = args.omit_level4_diffs
    kernel.SCALAR_LEVEL4_TABLE_CACHE = args.scalar_level4_cache
    kernel.OMIT_FUSED_DIRECT_NODE_TRANSFORM = args.omit_fused_node_transform
    kernel.VECTOR_DEPTH3_DIRECT_INDEX = args.vector_depth3_index
    kernel.PRIORITIZE_PAIRED_COPIES = args.prioritize_copies
    kernel.DYNAMIC_VECTOR_TO_ALU = args.dynamic_vector_to_alu
    kernel.LOWER_TAIL_BEFORE_SCHEDULE = args.lower_tail
    kernel.CHAINED_PAIRED_DIRECT_BASE = True
    kernel.PARALLEL_PAIRED_DIRECT_PREP = args.parallel
    kernel.PAIRED_DIRECT_PACKET_REPAIR = not args.no_packet_repair
    kernel.PAIRED_DIRECT_BASE_ON_FLOW = False
    if not args.keep_final:
        kernel.PAIRED_BRANCH_FINAL = False
        kernel.BRANCH_FINAL_LANES = ()
    kernel.DIRECT_BRANCH_DEPTH = args.round if args.round < 11 else args.round - 11
    kernel.DIRECT_BRANCH_ROUND = args.round
    if args.restore_hash_count:
        scalar_pairs = sorted(
            ((g, r) for g in range(32) for r in range(16)
             if ((g + r) % kernel.HASH_SCALAR_MOD == 0 or (g, r) in kernel.HASH_SCALAR_EXTRA)
             and (g, r) not in kernel.HASH_VECTOR_FORCE_SET),
            key=lambda pair: (pair[1], kernel.FULL_ROUND_OFFSETS[pair[0]]), reverse=True,
        )
        if not 0 <= args.restore_hash_count <= len(scalar_pairs):
            raise ValueError("restore count exceeds scalar hash candidates")
        kernel.HASH_VECTOR_FORCE_SET |= frozenset(scalar_pairs[:args.restore_hash_count])
    if args.round == 5:
        # Keep the independent second-traversal workspace. Reusing a first-
        # pass bit workspace here corrupts still-live mirrors. Mux spills
        # are shorter lived and their existing modulo allocator can share.
        kernel.N_WORKSPACES = 9
        kernel.N_SPILL_WORKSPACES = 4
    kernel.SCHEDULE_POLICIES = (args.policy,)
    kernel.SCHEDULE_EXACT_CYCLES = None
    if args.reverse:
        kernel.REVERSED_RELOCATED_TREE = True
        kernel.PREPROCESS_MAX_DEPTH = 8
    if args.preprocess_depth is not None:
        kernel.PREPROCESS_MAX_DEPTH = args.preprocess_depth
    started = time.monotonic()
    builder = kernel.KernelBuilder()
    if args.graph_only or args.teacher_weight:
        kernel.SCHEDULE_EXACT_CYCLES = []
        try:
            builder.build_kernel(10, 2047, 256, 16)
        except ValueError as error:
            if str(error) != "exact cycle schedule does not match DAG":
                raise
        if args.graph_only:
            from codex.audit_dead_values import report
            print(json.dumps({**vars(args), "operations": dict(Counter(op.engine for op in builder.dag_ops)),
                              "unconsumed_definitions": report(builder.dag_ops)}))
            return
        kernel.SCHEDULE_EXTERNAL_SCORES = source_priorities(
            source.dag_ops, builder.dag_ops, source_cycles, args.teacher_weight, args.dispatch_lead
        )
        kernel.SCHEDULE_EXTERNAL_SCORES = [score + (args.last_group_lead * args.teacher_weight
                                                   if op.group == 31 else 0)
                                           for score, op in zip(kernel.SCHEDULE_EXTERNAL_SCORES, builder.dag_ops)]
        kernel.SCHEDULE_EXTERNAL_HEIGHT_WEIGHT = args.height_weight
        kernel.SCHEDULE_EXACT_CYCLES = None
        builder = kernel.KernelBuilder()
    kernel.SCHEDULE_PREFLIGHT_ONLY = args.schedule_only
    try:
        builder.build_kernel(10, 2047, 256, 16)
    except ValueError as error:
        if not args.schedule_only or str(error) != "schedule preflight only; tables not assembled":
            raise
        print(json.dumps({**vars(args), "preflight_packets": builder.preflight_packet_count,
                          "verified_execution": False,
                          "warning": "Scheduling screen only; table layout and execution not verified"}))
        return
    fusion = None
    if args.fuse_returns:
        from codex.fuse_table_returns import fuse_returns
        fusion = fuse_returns(builder.instrs, builder.branch_main_cycles)
    scores = []
    slot_profiles = []
    class SlotProfileMachine(Machine):
        def step(self, instr, core):
            phase = "table" if builder.branch_main_cycles and core.pc - 1 >= builder.branch_main_cycles else "main"
            self.slot_profile[phase + "_cycles"] += int(any(e != "debug" and slots for e, slots in instr.items()))
            if phase == "table" and not instr.get("valu"):
                self.slot_profile["table_zero_valu_cycles"] += 1
            for engine, slots in instr.items():
                if engine != "debug":
                    self.slot_profile[phase + "_" + engine] += len(slots)
            return super().step(instr, core)
    for seed in range(args.seeds):
        pattern = ("zero", "ones", "alternating")[seed % 3] if args.pattern == "boundaries" else args.pattern
        random.seed(seed)
        forest = Tree.generate(10)
        inp = Input.generate(forest, 256, 16)
        if pattern == "full32":
            forest.values = [random.getrandbits(32) for _ in forest.values]
            inp.values = [random.getrandbits(32) for _ in inp.values]
        elif pattern != "normal":
            def value_at(index):
                if pattern == "zero":
                    return 0
                if pattern == "ones":
                    return 0xFFFFFFFF
                return 0xFFFFFFFF if index % 2 else 0
            forest.values = [value_at(index) for index in range(len(forest.values))]
            inp.values = [value_at(index) for index in range(len(inp.values))]
        mem = build_mem_image(forest, inp)
        machine_class = SlotProfileMachine if args.profile_slots else Machine
        machine = machine_class(mem, builder.instrs, builder.debug_info(), n_cores=N_CORES)
        if args.profile_slots:
            machine.slot_profile = Counter()
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()
        for ref_mem in reference_kernel2(mem):
            pass
        start = ref_mem[6]
        actual = machine.mem[start : start + 256]
        expected = ref_mem[start : start + 256]
        assert actual == expected, {
            "seed": seed,
            "wrong_groups": sorted({i // 8 for i, (a, b) in enumerate(zip(actual, expected)) if a != b}),
        }
        scores.append(machine.cycle)
        if args.profile_slots:
            assert machine.slot_profile["main_cycles"] + machine.slot_profile["table_cycles"] == machine.cycle
            slot_profiles.append(dict(machine.slot_profile))
    executed_resource_bounds = None
    if slot_profiles:
        from codex.profile_resource_gap import gap, LIMITS
        executed_resource_bounds = []
        for profile in slot_profiles:
            counts = {engine: profile.get("main_" + engine, 0) + profile.get("table_" + engine, 0)
                      for engine in LIMITS}
            executed_resource_bounds.append({"slots": counts, **gap(counts, 899)})
    print(json.dumps({
        **vars(args), "cycles": scores, "main_cycles": builder.branch_main_cycles,
        "instructions": len(builder.instrs), "scratch": builder.scratch_ptr,
        "operations": dict(Counter(op.engine for op in builder.dag_ops)),
        "return_fusion": fusion,
        "slot_profiles": slot_profiles,
        "executed_resource_bounds_target_899": executed_resource_bounds,
        "seconds": round(time.monotonic() - started, 3),
    }), flush=True)


if __name__ == "__main__":
    main()
