"""Map a known exact schedule onto a structurally modified kernel DAG.

Unchanged operations keep their proven cycle.  A tiny CP-SAT model places only
new operations, which makes compiler graph rewrites cheap to evaluate without
asking the global scheduler to rediscover a 20k-operation incumbent.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import replace
import json
import os
from pathlib import Path

from ortools.sat.python import cp_model

import codex.perf_takehome_under1000 as kernel
from problem import SLOT_LIMITS


def group_set(name: str) -> frozenset[int]:
    return frozenset(
        int(value)
        for value in os.environ.get(name, "").split(",")
        if value
    )


def pair_set(name: str) -> frozenset[tuple[int, int]]:
    return frozenset(
        tuple(int(component) for component in item.split(":"))
        for item in os.environ.get(name, "").split(",")
        if item
    )


def direct_lookup_map(name: str) -> dict[int, tuple[int, ...]]:
    return {
        int(group): tuple(range(int(count)))
        for group, count in (
            item.split(":")
            for item in os.environ.get(name, "").split(",")
            if item
        )
    }


def paired_direct_lookup_map(
    name: str,
) -> dict[int, tuple[tuple[int, int], ...]]:
    return {
        int(group): tuple(
            (2 * pair, 2 * pair + 1)
            for pair in range(int(count))
        )
        for group, count in (
            item.split(":")
            for item in os.environ.get(name, "").split(",")
            if item
        )
    }


def hybrid_override_map(name: str) -> dict[tuple[int, int], int]:
    return {
        (int(group), int(rnd)): int(count)
        for group, rnd, count in (
            item.split(":")
            for item in os.environ.get(name, "").split(",")
            if item
        )
    }


def configure_target() -> None:
    if "FULL_ROUND_OFFSETS" in os.environ:
        offsets = tuple(
            int(value)
            for value in os.environ["FULL_ROUND_OFFSETS"].split(",")
            if value
        )
        if len(offsets) != kernel.N_GROUPS or min(offsets) < 0:
            raise ValueError(
                "FULL_ROUND_OFFSETS must contain 32 non-negative integers"
            )
        kernel.FULL_ROUND_OFFSETS = offsets
        # The coarse hash-engine balancing knob selects a prefix ordered by
        # launch phase.  Recompute that order after retiming the software
        # pipeline; otherwise an offset search silently evaluates the scalar
        # choices of the original schedule.
        kernel._SCALAR_CANDIDATES.sort(
            key=lambda pair: (
                pair[1],
                kernel.FULL_ROUND_OFFSETS[pair[0]],
            ),
            reverse=True,
        )
    if "WORKSPACE_ASSIGNMENT" in os.environ:
        assignment = tuple(
            int(value)
            for value in os.environ["WORKSPACE_ASSIGNMENT"].split(",")
            if value
        )
        if (
            len(assignment) != kernel.N_GROUPS
            or min(assignment) < 0
            or max(assignment) >= kernel.N_WORKSPACES
        ):
            raise ValueError(
                "WORKSPACE_ASSIGNMENT must contain 32 workspace indices"
            )
        kernel.WORKSPACE_ASSIGNMENT = assignment
    if "SECOND_WORKSPACE_ASSIGNMENT" in os.environ:
        raw_assignment = os.environ["SECOND_WORKSPACE_ASSIGNMENT"]
        if raw_assignment:
            assignment = tuple(
                int(value)
                for value in raw_assignment.split(",")
                if value
            )
            if (
                len(assignment) != kernel.N_GROUPS
                or min(assignment) < 0
                or max(assignment) >= kernel.N_WORKSPACES
            ):
                raise ValueError(
                    "SECOND_WORKSPACE_ASSIGNMENT must contain 32 "
                    "workspace indices"
                )
            kernel.SECOND_WORKSPACE_ASSIGNMENT = assignment
        else:
            kernel.SECOND_WORKSPACE_ASSIGNMENT = None
    if "ALU_DERIVED_MISC_SET" in os.environ:
        kernel.ALU_DERIVED_MISC_SET = frozenset(
            value
            for value in os.environ["ALU_DERIVED_MISC_SET"].split(",")
            if value
        )
    if "ALU_DERIVED_POWER_COUNT" in os.environ:
        kernel.ALU_DERIVED_POWER_COUNT = int(
            os.environ["ALU_DERIVED_POWER_COUNT"]
        )
    if "ALU_DERIVED_DEPTH_START" in os.environ:
        kernel.ALU_DERIVED_DEPTH_START = int(
            os.environ["ALU_DERIVED_DEPTH_START"]
        )
    if "FLOW_SCALAR_CONSTANT_COUNT" in os.environ:
        kernel.FLOW_SCALAR_CONSTANT_COUNT = int(
            os.environ["FLOW_SCALAR_CONSTANT_COUNT"]
        )
    if "FLOW_SCALAR_CONSTANT_SET" in os.environ:
        kernel.FLOW_SCALAR_CONSTANT_SET = frozenset(
            int(value)
            for value in os.environ["FLOW_SCALAR_CONSTANT_SET"].split(",")
            if value
        )
    if "FLOW_ZERO_BASE_CONSTANT_SET" in os.environ:
        kernel.FLOW_ZERO_BASE_CONSTANT_SET = frozenset(
            int(value)
            for value in os.environ["FLOW_ZERO_BASE_CONSTANT_SET"].split(",")
            if value
        )
    if "FLOW_ONE_CONSTANT" in os.environ:
        kernel.FLOW_ONE_CONSTANT = bool(int(os.environ["FLOW_ONE_CONSTANT"]))
    if "ELIDE_DEAD_SCALAR_CONSTANTS" in os.environ:
        kernel.ELIDE_DEAD_SCALAR_CONSTANTS = bool(
            int(os.environ["ELIDE_DEAD_SCALAR_CONSTANTS"])
        )
    if "LOAD_IMMEDIATE_TAGS" in os.environ:
        kernel.LOAD_IMMEDIATE_TAGS = frozenset(
            value
            for value in os.environ["LOAD_IMMEDIATE_TAGS"].split(",")
            if value
        )
    if "MEMORY_VECTOR_CONSTANT_SET" in os.environ:
        kernel.MEMORY_VECTOR_CONSTANT_SET = frozenset(
            value
            for value in os.environ["MEMORY_VECTOR_CONSTANT_SET"].split(",")
            if value
        )
    if "MEMORY_VECTOR_CONSTANT_ORDER" in os.environ:
        kernel.MEMORY_VECTOR_CONSTANT_ORDER = tuple(
            value
            for value in os.environ["MEMORY_VECTOR_CONSTANT_ORDER"].split(",")
            if value
        )
    if "MEMORY_STAGE_SETUP_C5" in os.environ:
        kernel.MEMORY_STAGE_SETUP_C5 = bool(
            int(os.environ["MEMORY_STAGE_SETUP_C5"])
        )
    if "MEMORY_STAGE_NEGATIVE_FIVE" in os.environ:
        kernel.MEMORY_STAGE_NEGATIVE_FIVE = bool(
            int(os.environ["MEMORY_STAGE_NEGATIVE_FIVE"])
        )
    if "DIRECT_LOAD_VECTOR_CONSTANT_SET" in os.environ:
        kernel.DIRECT_LOAD_VECTOR_CONSTANT_SET = frozenset(
            value
            for value in os.environ[
                "DIRECT_LOAD_VECTOR_CONSTANT_SET"
            ].split(",")
            if value
        )
    if "DIRECT_LOAD_SETUP_C5" in os.environ:
        kernel.DIRECT_LOAD_SETUP_C5 = bool(
            int(os.environ["DIRECT_LOAD_SETUP_C5"])
        )
    if "DIRECT_LOAD_NEGATIVE_FIVE" in os.environ:
        kernel.DIRECT_LOAD_NEGATIVE_FIVE = bool(
            int(os.environ["DIRECT_LOAD_NEGATIVE_FIVE"])
        )
    if "MEMORY_CACHED_NODE_SET" in os.environ:
        kernel.MEMORY_CACHED_NODE_SET = group_set(
            "MEMORY_CACHED_NODE_SET"
        )
    if "ALU_CACHED_NODE_SET" in os.environ:
        kernel.ALU_CACHED_NODE_SET = group_set("ALU_CACHED_NODE_SET")
    if "MEMORY_CACHED_NODE_ORDER" in os.environ:
        kernel.MEMORY_CACHED_NODE_ORDER = tuple(
            int(value)
            for value in os.environ["MEMORY_CACHED_NODE_ORDER"].split(",")
            if value
        )
    if "ALU_CHAIN_MEMORY_CONSTANT_ADDRESSES" in os.environ:
        kernel.ALU_CHAIN_MEMORY_CONSTANT_ADDRESSES = bool(
            int(os.environ["ALU_CHAIN_MEMORY_CONSTANT_ADDRESSES"])
        )
    if "FLOW_MEMORY_CONSTANT_BASES" in os.environ:
        kernel.FLOW_MEMORY_CONSTANT_BASES = bool(
            int(os.environ["FLOW_MEMORY_CONSTANT_BASES"])
        )
    if "COPY_MEMORY_CONSTANT_BASES_FROM_TOP_POINTERS" in os.environ:
        kernel.COPY_MEMORY_CONSTANT_BASES_FROM_TOP_POINTERS = bool(
            int(os.environ["COPY_MEMORY_CONSTANT_BASES_FROM_TOP_POINTERS"])
        )
    if "MEMORY_VECTOR_CONSTANT_BASE" in os.environ:
        kernel.MEMORY_VECTOR_CONSTANT_BASE = int(
            os.environ["MEMORY_VECTOR_CONSTANT_BASE"]
        )
    if "MEMORY_VECTOR_CONSTANT_BUFFER_COUNT" in os.environ:
        kernel.MEMORY_VECTOR_CONSTANT_BUFFER_COUNT = int(
            os.environ["MEMORY_VECTOR_CONSTANT_BUFFER_COUNT"]
        )
    if "MEMORY_VECTOR_CONSTANT_SWITCH_AFTER" in os.environ:
        kernel.MEMORY_VECTOR_CONSTANT_SWITCH_AFTER = int(
            os.environ["MEMORY_VECTOR_CONSTANT_SWITCH_AFTER"]
        )
    if "MEMORY_VECTOR_CONSTANT_ADDRESS_GROUP" in os.environ:
        kernel.MEMORY_VECTOR_CONSTANT_ADDRESS_GROUP = int(
            os.environ["MEMORY_VECTOR_CONSTANT_ADDRESS_GROUP"]
        )
    if "SSA_MEMORY_VECTOR_CONSTANT_ADDRESS" in os.environ:
        kernel.SSA_MEMORY_VECTOR_CONSTANT_ADDRESS = bool(
            int(os.environ["SSA_MEMORY_VECTOR_CONSTANT_ADDRESS"])
        )
    if "ALU_RAW_ROOT_COPY" in os.environ:
        kernel.ALU_RAW_ROOT_COPY = bool(int(os.environ["ALU_RAW_ROOT_COPY"]))
    if "REUSE_TOP_RELOCATION_LEVEL4" in os.environ:
        kernel.REUSE_TOP_RELOCATION_LEVEL4 = bool(
            int(os.environ["REUSE_TOP_RELOCATION_LEVEL4"])
        )
    if "INDEPENDENT_TOP_P0" in os.environ:
        kernel.INDEPENDENT_TOP_P0 = bool(int(os.environ["INDEPENDENT_TOP_P0"]))
    if "INDEPENDENT_TOP_P1" in os.environ:
        kernel.INDEPENDENT_TOP_P1 = bool(int(os.environ["INDEPENDENT_TOP_P1"]))
    if "INDEPENDENT_RELOCATION_LOAD_POINTERS" in os.environ:
        kernel.INDEPENDENT_RELOCATION_LOAD_POINTERS = bool(
            int(os.environ["INDEPENDENT_RELOCATION_LOAD_POINTERS"])
        )
    if "INDEPENDENT_INPUT_POINTERS" in os.environ:
        kernel.INDEPENDENT_INPUT_POINTERS = bool(
            int(os.environ["INDEPENDENT_INPUT_POINTERS"])
        )
    if "INPUT_POINTER_STREAMS" in os.environ:
        kernel.INPUT_POINTER_STREAMS = int(
            os.environ["INPUT_POINTER_STREAMS"]
        )
    if "DERIVE_TOP_P1_FROM_P0" in os.environ:
        kernel.DERIVE_TOP_P1_FROM_P0 = bool(
            int(os.environ["DERIVE_TOP_P1_FROM_P0"])
        )
    if "DERIVE_SETUP_SECOND_POINTERS" in os.environ:
        kernel.DERIVE_SETUP_SECOND_POINTERS = bool(
            int(os.environ["DERIVE_SETUP_SECOND_POINTERS"])
        )
    if "DERIVE_OUTPUT_SECOND_POINTER" in os.environ:
        kernel.DERIVE_OUTPUT_SECOND_POINTER = bool(
            int(os.environ["DERIVE_OUTPUT_SECOND_POINTER"])
        )
    if "PRESERVE_OUTPUT_BASE" in os.environ:
        kernel.PRESERVE_OUTPUT_BASE = bool(
            int(os.environ["PRESERVE_OUTPUT_BASE"])
        )
    if "COPY_PRESERVED_OUTPUT_BASE_FROM_INPUT" in os.environ:
        kernel.COPY_PRESERVED_OUTPUT_BASE_FROM_INPUT = bool(
            int(os.environ["COPY_PRESERVED_OUTPUT_BASE_FROM_INPUT"])
        )
    scalar_sets = {
        "SCALAR_FINAL_C5": "SCALAR_FINAL_C5_SET",
        "SCALAR_FINAL_JOIN": "SCALAR_FINAL_JOIN_SET",
        "SCALAR_FINAL_SHIFT": "SCALAR_FINAL_SHIFT_SET",
        "SCALAR_FINAL_HASH23_JOIN": "SCALAR_FINAL_HASH23_JOIN_SET",
        "SCALAR_FINAL_HASH4": "SCALAR_FINAL_HASH4_SET",
        "VALU_FINAL_C5": "VALU_FINAL_C5_SET",
    }
    for env_name, attribute in scalar_sets.items():
        if env_name in os.environ:
            setattr(kernel, attribute, group_set(env_name))
    for env_name, attribute in (
        ("SCALAR_HASH1_JOIN", "SCALAR_HASH1_JOIN_SET"),
        ("SCALAR_HASH23_JOIN", "SCALAR_HASH23_JOIN_SET"),
        ("SCALAR_HASH5_JOIN", "SCALAR_HASH5_JOIN_SET"),
    ):
        if env_name in os.environ:
            setattr(kernel, attribute, pair_set(env_name))
    if "MADD_FIRST_DEPTH1_SET" in os.environ:
        kernel.MADD_FIRST_DEPTH1_SET = group_set("MADD_FIRST_DEPTH1_SET")
    if "PAIRED_EARLY_XOR" in os.environ:
        kernel.PAIRED_EARLY_XOR = bool(int(os.environ["PAIRED_EARLY_XOR"]))
    if "PAIRED_FLOW_SELECT" in os.environ:
        kernel.PAIRED_FLOW_SELECT = bool(int(os.environ["PAIRED_FLOW_SELECT"]))
    if "PAIRED_BRANCH_FINAL" in os.environ:
        kernel.PAIRED_BRANCH_FINAL = bool(int(os.environ["PAIRED_BRANCH_FINAL"]))
    if "BRANCH_FINAL_GROUP" in os.environ:
        kernel.BRANCH_FINAL_GROUP = int(os.environ["BRANCH_FINAL_GROUP"])
    if "BRANCH_DISPATCH_PADDING" in os.environ:
        kernel.BRANCH_DISPATCH_PADDING = bool(
            int(os.environ["BRANCH_DISPATCH_PADDING"])
        )
    if "BRANCH_DEDICATED_DEAD_REGS" in os.environ:
        kernel.BRANCH_DEDICATED_DEAD_REGS = bool(
            int(os.environ["BRANCH_DEDICATED_DEAD_REGS"])
        )
    if "BRANCH_DEAD_CANDIDATE_GROUP" in os.environ:
        kernel.BRANCH_DEAD_CANDIDATE_GROUP = int(
            os.environ["BRANCH_DEAD_CANDIDATE_GROUP"]
        )
    if "BRANCH_DEAD_CONTROL_GROUP" in os.environ:
        kernel.BRANCH_DEAD_CONTROL_GROUP = int(
            os.environ["BRANCH_DEAD_CONTROL_GROUP"]
        )
    if "BRANCH_DIRECT_FULL_TABLE" in os.environ:
        kernel.BRANCH_DIRECT_FULL_TABLE = bool(
            int(os.environ["BRANCH_DIRECT_FULL_TABLE"])
        )
    if "BRANCH_FINAL_LANES" in os.environ:
        kernel.BRANCH_FINAL_LANES = tuple(
            int(value)
            for value in os.environ["BRANCH_FINAL_LANES"].split(",")
            if value
        )
    if "DELAYED_PAIR_BRANCH_GROUPS" in os.environ:
        kernel.DELAYED_PAIR_BRANCH_GROUPS = group_set(
            "DELAYED_PAIR_BRANCH_GROUPS"
        )
    if "SAVED_SECOND_PATH_EXTRA_GROUPS" in os.environ:
        kernel.SAVED_SECOND_PATH_EXTRA_GROUPS = group_set(
            "SAVED_SECOND_PATH_EXTRA_GROUPS"
        )
    if "DIRECT_BRANCH_LOOKUPS" in os.environ:
        kernel.DIRECT_BRANCH_LOOKUPS = direct_lookup_map("DIRECT_BRANCH_LOOKUPS")
    if "PAIRED_DIRECT_BRANCH_LOOKUPS" in os.environ:
        kernel.PAIRED_DIRECT_BRANCH_LOOKUPS = paired_direct_lookup_map(
            "PAIRED_DIRECT_BRANCH_LOOKUPS"
        )
    if "CHAINED_DIRECT_BRANCH_BASE" in os.environ:
        kernel.CHAINED_DIRECT_BRANCH_BASE = bool(
            int(os.environ["CHAINED_DIRECT_BRANCH_BASE"])
        )
    if "FLOW_OUTPUT_ADVANCE_POSITIONS" in os.environ:
        kernel.FLOW_OUTPUT_ADVANCE_POSITIONS = group_set(
            "FLOW_OUTPUT_ADVANCE_POSITIONS"
        )
    if "FIRST_CACHE_SET" in os.environ:
        kernel.FIRST_CACHE_SET = group_set("FIRST_CACHE_SET")
    if "FINAL_CACHE_SET" in os.environ:
        kernel.FINAL_CACHE_SET = group_set("FINAL_CACHE_SET")
    if "VALU_FINAL_CACHE_SET" in os.environ:
        kernel.VALU_FINAL_CACHE_SET = group_set("VALU_FINAL_CACHE_SET")
    if "VALU_FINAL_CACHE_COUNTS" in os.environ:
        kernel.VALU_FINAL_CACHE_COUNTS = {
            int(group): int(count)
            for group, count in (
                item.split(":")
                for item in os.environ["VALU_FINAL_CACHE_COUNTS"].split(",")
                if item
            )
        }
    if "SCALAR_VALU_FINAL_DIFF_SET" in os.environ:
        kernel.SCALAR_VALU_FINAL_DIFF_SET = pair_set(
            "SCALAR_VALU_FINAL_DIFF_SET"
        )
    if "EARLY_FINAL_CACHE_SET" in os.environ:
        kernel.EARLY_FINAL_CACHE_SET = group_set("EARLY_FINAL_CACHE_SET")
    if "REVERSED_RELOCATED_TREE" in os.environ:
        kernel.REVERSED_RELOCATED_TREE = bool(
            int(os.environ["REVERSED_RELOCATED_TREE"])
        )
    if "RELOCATION_STAGE_ORDER" in os.environ:
        kernel.RELOCATION_STAGE_ORDER = os.environ["RELOCATION_STAGE_ORDER"]
    if "RELOCATION_STORE_STREAMS" in os.environ:
        kernel.RELOCATION_STORE_STREAMS = int(
            os.environ["RELOCATION_STORE_STREAMS"]
        )
    if "COPY_RELOCATION_STORE_POINTER_FROM_SIXTEEN" in os.environ:
        kernel.COPY_RELOCATION_STORE_POINTER_FROM_SIXTEEN = bool(
            int(os.environ[
                "COPY_RELOCATION_STORE_POINTER_FROM_SIXTEEN"
            ])
        )
    if "DERIVE_RELOCATION_LOAD_POINTER_FROM_TOP" in os.environ:
        kernel.DERIVE_RELOCATION_LOAD_POINTER_FROM_TOP = bool(
            int(os.environ[
                "DERIVE_RELOCATION_LOAD_POINTER_FROM_TOP"
            ])
        )
    if "ALU_DERIVED_NEGATIVE_FIVE" in os.environ:
        kernel.ALU_DERIVED_NEGATIVE_FIVE = bool(
            int(os.environ["ALU_DERIVED_NEGATIVE_FIVE"])
        )
    if "ALU_DERIVED_TREE_REFLECT" in os.environ:
        kernel.ALU_DERIVED_TREE_REFLECT = bool(
            int(os.environ["ALU_DERIVED_TREE_REFLECT"])
        )
    if "RELOCATION_LOAD_STREAMS" in os.environ:
        kernel.RELOCATION_LOAD_STREAMS = int(
            os.environ["RELOCATION_LOAD_STREAMS"]
        )
    if "VECTOR_TOP_C5_BLOCKS" in os.environ:
        kernel.VECTOR_TOP_C5_BLOCKS = int(os.environ["VECTOR_TOP_C5_BLOCKS"])
    if "SSA_LEVEL4_WORKSPACES" in os.environ:
        kernel.SSA_LEVEL4_WORKSPACES = bool(
            int(os.environ["SSA_LEVEL4_WORKSPACES"])
        )
    if "SSA_WORKSPACES" in os.environ:
        kernel.SSA_WORKSPACES = bool(int(os.environ["SSA_WORKSPACES"]))
    if "SSA_ALL_WORKSPACES" in os.environ:
        kernel.SSA_ALL_WORKSPACES = bool(
            int(os.environ["SSA_ALL_WORKSPACES"])
        )
    if "SSA_SECOND_WORKSPACES" in os.environ:
        kernel.SSA_SECOND_WORKSPACES = bool(
            int(os.environ["SSA_SECOND_WORKSPACES"])
        )
    if "SSA_FIRST_WORKSPACE_GROUPS" in os.environ:
        kernel.SSA_FIRST_WORKSPACE_GROUPS = group_set(
            "SSA_FIRST_WORKSPACE_GROUPS"
        )
    if "PREPROCESS_DEPTH" in os.environ:
        kernel.PREPROCESS_MAX_DEPTH = int(os.environ["PREPROCESS_DEPTH"])
    if "VECTOR_PARITY_SET" in os.environ:
        kernel.VECTOR_PARITY_SET = pair_set("VECTOR_PARITY_SET")
    if "VECTOR_NODE_XOR_SET" in os.environ:
        kernel.VECTOR_NODE_XOR_SET = pair_set("VECTOR_NODE_XOR_SET")
    if "HYBRID_MADD_OVERRIDES" in os.environ:
        kernel.HYBRID_MADD_OVERRIDES = hybrid_override_map(
            "HYBRID_MADD_OVERRIDES"
        )
    if "HASH_SCALAR_EXTRA_ADD" in os.environ:
        kernel.HASH_SCALAR_EXTRA = frozenset(
            set(kernel.HASH_SCALAR_EXTRA) | set(pair_set("HASH_SCALAR_EXTRA_ADD"))
        )
    if "HASH_SCALAR_EXTRA_REMOVE" in os.environ:
        kernel.HASH_SCALAR_EXTRA = frozenset(
            set(kernel.HASH_SCALAR_EXTRA)
            - set(pair_set("HASH_SCALAR_EXTRA_REMOVE"))
        )
    if "HASH_SCALAR_EXTRA_COUNT" in os.environ:
        count = int(os.environ["HASH_SCALAR_EXTRA_COUNT"])
        kernel.HASH_SCALAR_EXTRA = frozenset(
            kernel._BASE_SCALAR | set(kernel._SCALAR_CANDIDATES[:count])
        )
    if "HASH_SCALAR_EXTRA_ADD_AFTER" in os.environ:
        kernel.HASH_SCALAR_EXTRA = frozenset(
            set(kernel.HASH_SCALAR_EXTRA)
            | set(pair_set("HASH_SCALAR_EXTRA_ADD_AFTER"))
        )
    if "HASH_SCALAR_EXTRA_REMOVE_AFTER" in os.environ:
        kernel.HASH_SCALAR_EXTRA = frozenset(
            set(kernel.HASH_SCALAR_EXTRA)
            - set(pair_set("HASH_SCALAR_EXTRA_REMOVE_AFTER"))
        )
    if "HASH_VECTOR_FORCE_SET" in os.environ:
        kernel.HASH_VECTOR_FORCE_SET = pair_set("HASH_VECTOR_FORCE_SET")


def real_tail_ops(ops: list[kernel._Op]) -> list[kernel._Op]:
    first_tail_group = kernel.N_GROUPS - kernel.INDEPENDENT_TAIL_GROUP_COUNT
    last_prefix_store = max(
        i
        for i, op in enumerate(ops)
        if op.tag == "output_store"
        and op.group is not None
        and op.group < first_tail_group
    )
    adjusted = []
    for i, op in enumerate(ops):
        if op.tag == "pointer_advance" and i > last_prefix_store:
            adjusted.append(
                replace(op, engine="debug", parents={}, reads=(), writes=())
            )
        elif (
            op.tag == "output_store"
            and op.group is not None
            and op.group >= first_tail_group
        ):
            adjusted.append(
                replace(
                    op,
                    parents={
                        parent: lag
                        for parent, lag in op.parents.items()
                        if ops[parent].tag not in {"output_pointer", "pointer_advance"}
                    },
                )
            )
        else:
            adjusted.append(op)
    return adjusted


def op_key(op: kernel._Op) -> tuple[object, ...]:
    """Identify an operation by semantics rather than register allocation.

    Structural experiments deliberately change virtual/physical register
    assignments.  Including the complete instruction tuple in this key made
    every such renaming look like a new operation, so the "local" repair
    model unnecessarily rescheduled most of the kernel.  Construction order
    disambiguates repeated lane operations through the per-key deque below.
    """
    if op.identity:
        return (op.engine, "identity", op.group, op.round, op.identity)
    opcode = op.slot[0] if op.slot else None
    return (op.engine, opcode, op.tag, op.group, op.round)


def validate(ops: list[kernel._Op], cycles: list[int]) -> None:
    usage = Counter((cycles[i], op.engine) for i, op in enumerate(ops))
    for (cycle, engine), count in usage.items():
        if engine != "debug" and count > SLOT_LIMITS[engine]:
            raise AssertionError((cycle, engine, count))
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            if cycles[child] < cycles[parent] + lag:
                raise AssertionError(
                    (parent, child, lag, cycles[parent], cycles[child])
                )


def main() -> None:
    source_path = Path(os.environ["SOURCE"])
    source_cycles = json.loads(source_path.read_text())["cycles"]

    if "SOURCE_SAVED_SECOND_PATH_EXTRA_GROUPS" in os.environ:
        kernel.SAVED_SECOND_PATH_EXTRA_GROUPS = group_set(
            "SOURCE_SAVED_SECOND_PATH_EXTRA_GROUPS"
        )
    if "SOURCE_HASH_SCALAR_EXTRA_REMOVE" in os.environ:
        kernel.HASH_SCALAR_EXTRA = frozenset(
            set(kernel.HASH_SCALAR_EXTRA)
            - set(pair_set("SOURCE_HASH_SCALAR_EXTRA_REMOVE"))
        )
    if "SOURCE_HASH_SCALAR_EXTRA_ADD" in os.environ:
        kernel.HASH_SCALAR_EXTRA = frozenset(
            set(kernel.HASH_SCALAR_EXTRA)
            | set(pair_set("SOURCE_HASH_SCALAR_EXTRA_ADD"))
        )
    if "SOURCE_DIRECT_BRANCH_LOOKUPS" in os.environ:
        kernel.DIRECT_BRANCH_LOOKUPS = direct_lookup_map(
            "SOURCE_DIRECT_BRANCH_LOOKUPS"
        )
    if "SOURCE_PAIRED_DIRECT_BRANCH_LOOKUPS" in os.environ:
        kernel.PAIRED_DIRECT_BRANCH_LOOKUPS = paired_direct_lookup_map(
            "SOURCE_PAIRED_DIRECT_BRANCH_LOOKUPS"
        )
    if "SOURCE_CHAINED_DIRECT_BRANCH_BASE" in os.environ:
        kernel.CHAINED_DIRECT_BRANCH_BASE = bool(
            int(os.environ["SOURCE_CHAINED_DIRECT_BRANCH_BASE"])
        )
    if "SOURCE_FLOW_OUTPUT_ADVANCE_POSITIONS" in os.environ:
        kernel.FLOW_OUTPUT_ADVANCE_POSITIONS = group_set(
            "SOURCE_FLOW_OUTPUT_ADVANCE_POSITIONS"
        )
    if "SOURCE_FINAL_CACHE_SET" in os.environ:
        kernel.FINAL_CACHE_SET = group_set("SOURCE_FINAL_CACHE_SET")
    if "SOURCE_EARLY_FINAL_CACHE_SET" in os.environ:
        kernel.EARLY_FINAL_CACHE_SET = group_set("SOURCE_EARLY_FINAL_CACHE_SET")
    if "SOURCE_VECTOR_PARITY_SET" in os.environ:
        kernel.VECTOR_PARITY_SET = pair_set("SOURCE_VECTOR_PARITY_SET")
    if "SOURCE_VECTOR_NODE_XOR_SET" in os.environ:
        kernel.VECTOR_NODE_XOR_SET = pair_set("SOURCE_VECTOR_NODE_XOR_SET")
    if "SOURCE_HYBRID_MADD_OVERRIDES" in os.environ:
        kernel.HYBRID_MADD_OVERRIDES = hybrid_override_map(
            "SOURCE_HYBRID_MADD_OVERRIDES"
        )
    for env_name, attribute in (
        ("SOURCE_SCALAR_FINAL_C5", "SCALAR_FINAL_C5_SET"),
        ("SOURCE_SCALAR_FINAL_JOIN", "SCALAR_FINAL_JOIN_SET"),
        ("SOURCE_SCALAR_FINAL_SHIFT", "SCALAR_FINAL_SHIFT_SET"),
        (
            "SOURCE_SCALAR_FINAL_HASH23_JOIN",
            "SCALAR_FINAL_HASH23_JOIN_SET",
        ),
        ("SOURCE_SCALAR_FINAL_HASH4", "SCALAR_FINAL_HASH4_SET"),
    ):
        if env_name in os.environ:
            setattr(kernel, attribute, group_set(env_name))
    for env_name, attribute in (
        ("SOURCE_SCALAR_HASH1_JOIN", "SCALAR_HASH1_JOIN_SET"),
        ("SOURCE_SCALAR_HASH23_JOIN", "SCALAR_HASH23_JOIN_SET"),
        ("SOURCE_SCALAR_HASH5_JOIN", "SCALAR_HASH5_JOIN_SET"),
    ):
        if env_name in os.environ:
            setattr(kernel, attribute, pair_set(env_name))
    # Rebuilding the source graph with its own exact schedule serves two
    # purposes: it accepts non-default structural incumbents and computes the
    # physical coloring of their virtual saved-path registers from the proven
    # cycle assignment rather than from a heuristic schedule.
    kernel.SCHEDULE_EXACT_CYCLES = source_cycles
    source_builder = kernel.KernelBuilder()
    source_builder.build_kernel(10, 2047, 256, 16)
    source_ops = real_tail_ops(source_builder.dag_ops)
    if len(source_cycles) != len(source_ops):
        raise ValueError(
            f"source length {len(source_cycles)} != default DAG {len(source_ops)}"
        )
    validate(source_ops, source_cycles)

    # The production module carries an embedded exact schedule for the
    # default graph.  Structural target variants have a different operation
    # count, so let the builder expose their DAG without trying to replay the
    # default schedule while mapping.
    kernel.SCHEDULE_EXACT_CYCLES = None
    configure_target()
    target_builder = kernel.KernelBuilder()
    try:
        target_builder.build_kernel(10, 2047, 256, 16)
    except (AssertionError, StopIteration):
        if not hasattr(target_builder, "dag_ops"):
            raise
    target_ops = real_tail_ops(target_builder.dag_ops)
    if bool(int(os.environ.get("PRINT_TAG_DELTA", "0"))):
        source_tag_counts = Counter((op.engine, op.tag) for op in source_ops)
        target_tag_counts = Counter((op.engine, op.tag) for op in target_ops)
        print(
            "tag_delta="
            + repr(
                sorted(
                    (key, target_tag_counts[key] - source_tag_counts[key])
                    for key in source_tag_counts.keys() | target_tag_counts.keys()
                    if target_tag_counts[key] != source_tag_counts[key]
                )
            ),
            flush=True,
        )

    source_by_key: dict[tuple, deque[int]] = defaultdict(deque)
    for index, op in enumerate(source_ops):
        source_by_key[op_key(op)].append(index)

    result = [-1] * len(target_ops)
    for index, op in enumerate(target_ops):
        matches = source_by_key[op_key(op)]
        if matches:
            result[index] = source_cycles[matches.popleft()]

    unmatched = [i for i, cycle in enumerate(result) if cycle < 0]
    unmatched_set = set(unmatched)
    horizon = int(os.environ.get("TARGET", str(max(source_cycles) + 1)))
    cutoff = int(os.environ.get("CUTOFF", str(horizon)))
    unmatched_set.update(
        i
        for i, cycle in enumerate(result)
        if cycle >= cutoff or cycle >= horizon
    )
    prefix_cutoff = int(os.environ.get("PREFIX_CUTOFF", "-1"))
    if prefix_cutoff >= 0:
        unmatched_set.update(
            i
            for i, cycle in enumerate(result)
            if 0 <= cycle <= prefix_cutoff
        )
    local_groups = group_set("LOCAL_GROUPS")
    if local_groups:
        unmatched_set.update(
            i for i, op in enumerate(target_ops) if op.group in local_groups
        )
    local_engines = frozenset(
        engine
        for engine in os.environ.get("LOCAL_ENGINES", "").split(",")
        if engine
    )
    if local_engines:
        unmatched_set.update(
            i for i, op in enumerate(target_ops) if op.engine in local_engines
        )
    local_indices = {
        int(value)
        for value in os.environ.get("LOCAL_INDICES", "").split(",")
        if value
    }
    if local_indices:
        invalid_indices = sorted(
            i for i in local_indices if not 0 <= i < len(target_ops)
        )
        if invalid_indices:
            raise ValueError(f"invalid LOCAL_INDICES: {invalid_indices}")
        unmatched_set.update(local_indices)
    print(
        f"source_ops={len(source_ops)} target_ops={len(target_ops)} "
        f"unmatched={len(unmatched)} horizon={horizon}",
        flush=True,
    )

    # A rewrite can alter parents of otherwise identical instructions.  Keep
    # common operations fixed when legal; if a fixed/fixed edge is no longer
    # satisfied, pull both endpoints into the small repair model.
    changed = True
    while changed:
        changed = False
        for child, op in enumerate(target_ops):
            for parent, lag in op.parents.items():
                if parent in unmatched_set or child in unmatched_set:
                    continue
                if result[child] < result[parent] + lag:
                    unmatched_set.update((parent, child))
                    changed = True
    # Rewrites often insert an operation whose consumer still looks identical
    # and therefore remains fixed.  Grow the repair set through the concrete
    # dependency blockers until every local operation has a non-empty
    # earliest/latest window.  This is substantially smaller than pulling in
    # every transitive neighbour of each changed group.
    closure_round = 0
    boundary_conflicts = []
    while True:
        unmatched = sorted(unmatched_set)
        boundary_earliest = {i: 0 for i in unmatched}
        earliest_reason: dict[int, int] = {}
        for child in unmatched:
            for parent, lag in target_ops[child].parents.items():
                parent_cycle = (
                    boundary_earliest[parent]
                    if parent in unmatched_set
                    else result[parent]
                )
                candidate = parent_cycle + lag
                if candidate > boundary_earliest[child]:
                    boundary_earliest[child] = candidate
                    earliest_reason[child] = parent
        boundary_latest = {i: horizon - 1 for i in unmatched}
        latest_reason: dict[int, int] = {}
        for child in reversed(range(len(target_ops))):
            for parent, lag in target_ops[child].parents.items():
                if parent not in unmatched_set:
                    continue
                child_cycle = (
                    boundary_latest[child]
                    if child in unmatched_set
                    else result[child]
                )
                candidate = child_cycle - lag
                if candidate < boundary_latest[parent]:
                    boundary_latest[parent] = candidate
                    latest_reason[parent] = child
        boundary_conflicts = [
            i for i in unmatched if boundary_earliest[i] > boundary_latest[i]
        ]
        if not boundary_conflicts:
            break
        blockers: set[int] = set()
        for conflict in boundary_conflicts:
            node = conflict
            while node in earliest_reason:
                parent = earliest_reason[node]
                if parent not in unmatched_set:
                    blockers.add(parent)
                    break
                node = parent
            node = conflict
            while node in latest_reason:
                child = latest_reason[node]
                if child not in unmatched_set:
                    blockers.add(child)
                    break
                node = child
        blockers.difference_update(unmatched_set)
        if not blockers:
            break
        unmatched_set.update(blockers)
        closure_round += 1
        if closure_round >= int(os.environ.get("BOUNDARY_CLOSURE_LIMIT", "200")):
            break

    unmatched = sorted(unmatched_set)
    print(
        f"repair_ops={len(unmatched)} cutoff={cutoff} "
        f"prefix_cutoff={prefix_cutoff} "
        f"boundary_closure_rounds={closure_round}",
        flush=True,
    )
    if bool(int(os.environ.get("PRINT_REPAIR_OPS", "0"))):
        for i in unmatched:
            op = target_ops[i]
            print(
                f"repair i={i:5d} old={result[i]:4d} "
                f"window=[{boundary_earliest[i]:3d},"
                f"{boundary_latest[i]:3d}] "
                f"{op.engine:5s} g={op.group:2d} r={op.round:2d} "
                f"{op.tag} {op.slot[0] if op.slot else None}",
                flush=True,
            )
    if boundary_conflicts:
        examples = [
            (
                i,
                target_ops[i].tag,
                target_ops[i].group,
                target_ops[i].round,
                boundary_earliest[i],
                boundary_latest[i],
            )
            for i in boundary_conflicts[:12]
        ]
        print(
            f"boundary_conflicts={len(boundary_conflicts)} examples={examples}",
            flush=True,
        )

    model = cp_model.CpModel()
    assignments: dict[tuple[int, int], cp_model.IntVar] = {}
    starts = {}
    time_indexed = bool(int(os.environ.get("TIME_INDEXED", "1")))
    hint_radius = int(os.environ.get("HINT_RADIUS", "0"))
    domains = {}
    for i in unmatched:
        old_cycle = result[i]
        if old_cycle >= 0 and hint_radius:
            lower = max(0, old_cycle - hint_radius)
            upper = min(horizon - 1, old_cycle + hint_radius)
        else:
            lower, upper = 0, horizon - 1
        lower = max(lower, boundary_earliest[i])
        upper = min(upper, boundary_latest[i])
        if lower > upper:
            raise ValueError(
                f"empty repair domain for op {i}: [{lower},{upper}]"
            )
        domains[i] = (lower, upper)
        starts[i] = model.new_int_var(lower, upper, f"s{i}")
        if time_indexed:
            choices = []
            for cycle in range(lower, upper + 1):
                choice = model.new_bool_var(f"x{i}_{cycle}")
                assignments[i, cycle] = choice
                choices.append(choice)
                if old_cycle >= 0:
                    model.add_hint(
                        choice,
                        int(cycle == min(upper, max(lower, old_cycle))),
                    )
            model.add_exactly_one(choices)
            model.add(
                starts[i]
                == sum(
                    cycle * assignments[i, cycle]
                    for cycle in range(lower, upper + 1)
                )
            )
        if old_cycle >= 0:
            model.add_hint(starts[i], min(upper, max(lower, old_cycle)))

    for child, op in enumerate(target_ops):
        for parent, lag in op.parents.items():
            if child in unmatched_set and parent in unmatched_set:
                model.add(starts[child] >= starts[parent] + lag)
            elif child in unmatched_set:
                model.add(starts[child] >= result[parent] + lag)
            elif parent in unmatched_set:
                model.add(result[child] >= starts[parent] + lag)

    if bool(int(os.environ.get("PACK_BRANCH_TRACES", "1"))):
        copy_tags = {
            "direct_branch_copy",
            "paired_direct_branch_copy",
            "paired_branch_copy",
            "paired_branch_delayed_copy",
        }
        target_tags = {
            "direct_branch_target",
            "paired_direct_branch_target",
            "paired_branch_target",
            "paired_branch_delayed_target",
        }

        def exact_distance(left: int, right: int, distance: int) -> None:
            left_local = left in unmatched_set
            right_local = right in unmatched_set
            if left_local and right_local:
                model.add(starts[left] == starts[right] + distance)
            elif left_local:
                model.add(starts[left] == result[right] + distance)
            elif right_local:
                model.add(result[left] == starts[right] + distance)
            elif result[left] != result[right] + distance:
                raise ValueError("fixed branch trace is not packetized")

        for child, op in enumerate(target_ops):
            if op.tag in copy_tags:
                jump_parent = next(
                    (
                        parent
                        for parent in op.parents
                        if target_ops[parent].tag.endswith("branch_jump")
                    ),
                    None,
                )
                if jump_parent is not None:
                    exact_distance(child, jump_parent, 1)
            elif op.tag in target_tags:
                for parent in op.parents:
                    if target_ops[parent].tag in copy_tags:
                        exact_distance(child, parent, 0)

    fixed_use: dict[str, Counter[int]] = defaultdict(Counter)
    for i, op in enumerate(target_ops):
        if i not in unmatched_set:
            fixed_use[op.engine][result[i]] += 1
    fixed_overloads = {
        engine: sorted(
            (
                (cycle, demand - SLOT_LIMITS[engine])
                for cycle, demand in usage.items()
                if engine != "debug" and demand > SLOT_LIMITS[engine]
            ),
            key=lambda item: (-item[1], item[0]),
        )
        for engine, usage in fixed_use.items()
    }
    fixed_overloads = {
        engine: overloads
        for engine, overloads in fixed_overloads.items()
        if overloads
    }
    if fixed_overloads:
        print(
            "fixed_overloads="
            + repr(
                {
                    engine: overloads[:12]
                    for engine, overloads in fixed_overloads.items()
                }
            ),
            flush=True,
        )
    repair_hall: dict[str, tuple[int, int, int, int, int]] = {}
    for engine, capacity in SLOT_LIMITS.items():
        if engine == "debug":
            continue
        local = [i for i in unmatched if target_ops[i].engine == engine]
        if not local:
            continue
        residual = [
            capacity - fixed_use[engine][cycle]
            for cycle in range(horizon)
        ]
        residual_prefix = [0]
        for available in residual:
            residual_prefix.append(residual_prefix[-1] + available)
        best = (-10**9, 0, 0, 0, 0)
        for left in range(horizon):
            histogram = [0] * horizon
            for i in local:
                if boundary_earliest[i] >= left:
                    histogram[boundary_latest[i]] += 1
            contained = 0
            for right in range(left, horizon):
                contained += histogram[right]
                available = residual_prefix[right + 1] - residual_prefix[left]
                overload = contained - available
                if overload > best[0]:
                    best = (overload, left, right, contained, available)
        repair_hall[engine] = best
    if repair_hall:
        print(f"repair_hall={repair_hall}", flush=True)
    if bool(int(os.environ.get("PRINT_HALL_BLOCKERS", "0"))):
        natural_earliest = [0] * len(target_ops)
        children: list[list[tuple[int, int]]] = [
            [] for _ in target_ops
        ]
        for child, op in enumerate(target_ops):
            for parent, lag in op.parents.items():
                natural_earliest[child] = max(
                    natural_earliest[child],
                    natural_earliest[parent] + lag,
                )
                children[parent].append((child, lag))
        natural_tail = [0] * len(target_ops)
        for parent in reversed(range(len(target_ops))):
            natural_tail[parent] = max(
                (
                    lag + natural_tail[child]
                    for child, lag in children[parent]
                ),
                default=0,
            )
        natural_latest = [
            horizon - 1 - tail for tail in natural_tail
        ]
        for engine, (overload, left, right, _, _) in repair_hall.items():
            if overload <= 0:
                continue
            blockers = [
                i
                for i, op in enumerate(target_ops)
                if i not in unmatched_set
                and op.engine == engine
                and left <= result[i] <= right
                and (
                    natural_earliest[i] < left
                    or natural_latest[i] > right
                )
            ]
            blockers.sort(
                key=lambda i: (
                    -max(
                        left - natural_earliest[i],
                        natural_latest[i] - right,
                    ),
                    result[i],
                    i,
                )
            )
            print(
                f"hall_blockers engine={engine} "
                f"window=[{left},{right}] overload={overload}",
                flush=True,
            )
            for i in blockers[:40]:
                op = target_ops[i]
                print(
                    f"  i={i:5d} old={result[i]:3d} "
                    f"natural=[{natural_earliest[i]:3d},"
                    f"{natural_latest[i]:3d}] "
                    f"g={op.group:2d} r={op.round:2d} "
                    f"{op.tag} {op.slot[0] if op.slot else None}",
                    flush=True,
                )
    for engine, capacity in SLOT_LIMITS.items():
        if engine == "debug":
            continue
        if time_indexed:
            for cycle in range(horizon):
                choices = [
                    assignments[i, cycle]
                    for i in unmatched
                    if target_ops[i].engine == engine
                    and domains[i][0] <= cycle <= domains[i][1]
                ]
                if choices:
                    model.add(sum(choices) <= capacity - fixed_use[engine][cycle])
                elif fixed_use[engine][cycle] > capacity:
                    model.add_bool_or([])
        else:
            intervals = []
            demands = []
            for i in unmatched:
                if target_ops[i].engine == engine:
                    intervals.append(
                        model.new_fixed_size_interval_var(starts[i], 1, f"i{i}")
                    )
                    demands.append(1)
            for cycle, demand in fixed_use[engine].items():
                intervals.append(
                    model.new_fixed_size_interval_var(
                        cycle, 1, f"f_{engine}_{cycle}"
                    )
                )
                demands.append(demand)
            model.add_cumulative(intervals, demands, capacity)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(os.environ.get("TIME_LIMIT", "60"))
    solver.parameters.num_workers = int(os.environ.get("WORKERS", "8"))
    status = solver.solve(model)
    print("status", solver.status_name(status), flush=True)
    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return
    for i in unmatched:
        result[i] = solver.value(starts[i])
    validate(target_ops, result)
    output = Path(os.environ.get("OUT", "/tmp/aopt-mapped-variant.json"))
    output.write_text(json.dumps({"makespan": max(result) + 1, "cycles": result}))
    print(f"makespan={max(result) + 1} output={output}", flush=True)


if __name__ == "__main__":
    main()
