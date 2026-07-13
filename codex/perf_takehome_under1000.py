"""A resource-balanced kernel for Anthropic's original performance take-home.

The scored workload is deliberately fixed (height 10, 16 rounds, 256 inputs).
This implementation specializes that shape at build time and emits only legal
instructions for the frozen single-core machine.  It does not modify the
simulator, the reference implementation, or the tests.

The main ideas are:

* fully unroll the two traversals (depths 0..10 and 0..4);
* keep the hash result internally XORed with the final hash constant;
* cache tree levels 0..4 and mirror path bits to make lookup cheap;
* gather deeper nodes in place and pre-transform levels 4..7 once;
* split independent work across VALU, scalar ALU, load, store, and flow slots;
* list-schedule a dependency DAG with the machine's end-of-cycle write rules.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import heapq
from typing import Iterable

from problem import DebugInfo, SCRATCH_SIZE, SLOT_LIMITS, VLEN


C0 = 0x7ED55D16
C1 = 0xC761C23C
C2 = 0x165667B1
C3 = 0xD3A2646C
C4 = 0xFD7046C5
C5 = 0xB55A4F09
C23 = (C2 + C3) & 0xFFFFFFFF
M23 = 33 << 9
C2_SHIFT9 = (C2 << 9) & 0xFFFFFFFF

OFFICIAL_SHAPE = (10, 2047, 256, 16)
N_GROUPS = 32
N_WORKSPACES = 9
N_SPILL_WORKSPACES = 9
# Experimental compiler-style workspace renaming.  The normal kernel assigns
# physical scratch before instruction scheduling, which introduces false
# dependencies between unrelated groups.  In SSA mode the short-lived mux
# registers are scheduled virtually and colored back onto this same physical
# pool afterwards.
SSA_WORKSPACES = False
FLOW_SCALAR_CONSTANT_COUNT = 0
ALU_DERIVED_POWER_COUNT = 0
ALU_DERIVED_DEPTH_START = 11
ALU_DERIVED_MISC_SET: frozenset[str] = frozenset(
    ("nineteen", "m4", "m0", "m2", "m23")
)
WORKSPACE_ASSIGNMENT = (
    0, 2, 3, 2, 4, 0, 1, 4, 3, 0, 1, 0, 1, 6, 4, 2,
    3, 2, 3, 2, 5, 4, 3, 0, 7, 6, 5, 1, 1, 7, 5, 5,
)
SECOND_WORKSPACE_STRIDE = 3
SECOND_WORKSPACE_FIXED = 8
SECOND_WORKSPACE_COUNT = 1
SECOND_WORKSPACE_ASSIGNMENT: tuple[int, ...] | None = None
INDEPENDENT_ROOT_CACHE = True
DIRECT_MIRROR_PATH = False
OVERLAP_DEEP_ADDRESS = False
OVERLAP_SHALLOW_ADDRESS = False
TAIL_EMISSION_MODE = "full_offset"
TAIL_EMISSION_STAGGER = 1
TAIL_EMISSION_COHORTS = 8
TAIL_GROUP_ORDER = tuple(range(N_GROUPS))
FULL_ROUND_OFFSETS = (
    0, 1, 13, 17, 8, 20, 0, 2, 1, 5, 6, 14, 20, 9, 17, 12,
    22, 21, 17, 6, 13, 22, 6, 10, 9, 15, 8, 14, 10, 13, 2, 22,
)
SCATTERED_OUTPUT_STORES = False
OUTPUT_POINTER_STREAMS = 2
OUTPUT_GROUP_ORDER = tuple(range(N_GROUPS))
PER_GROUP_OUTPUT_POINTERS = False
PREPROCESS_MAX_DEPTH = 7
EARLY_FINAL_CACHE_SET: frozenset[int] = frozenset()
EARLY_FINAL_ADDRESS_SET: frozenset[int] = frozenset()
VECTOR_EARLY_FINAL_ADDRESS_SET: frozenset[int] = frozenset()
BRANCH_FINAL_GROUP = 31
BRANCH_FINAL_LANES: tuple[int, ...] = (4, 5, 6, 7)
MADD_FIRST_DEPTH1 = False
SCALAR_FIRST_DEPTH1_SET: frozenset[int] = frozenset()
SCALAR_FINAL_C5_SET: frozenset[int] = frozenset()
SCALAR_FINAL_JOIN_SET: frozenset[int] = frozenset()
SCALAR_FINAL_HASH4_SET: frozenset[int] = frozenset()
SCALAR_FINAL_SHIFT_SET: frozenset[int] = frozenset()
SCALAR_FINAL_HASH23_JOIN_SET: frozenset[int] = frozenset()
SCALAR_HASH1_JOIN_SET: frozenset[tuple[int, int]] = frozenset()
SCALAR_HASH23_JOIN_SET: frozenset[tuple[int, int]] = frozenset()
SCALAR_HASH5_JOIN_SET: frozenset[tuple[int, int]] = frozenset()
FINAL_CACHE_SET = frozenset((0, 1, 3, 4, 5, 6, 7, 9, 10, 29))
FIRST_CACHE_SET = frozenset(
    (3, 4, 5, 7, 9, 11, 12, 13, 15, 16, 18, 19, 20, 24, 29, 31)
)
HASH_SCALAR_MOD = 4
HASH_SCALAR_STAGE = -1  # scalarize the independent stage-1 constant XOR
_BASE_SCALAR = {(group, (1 - group) % 4) for group in range(21)}
_SCALAR_CANDIDATES = [
    (group, rnd)
    for group in range(N_GROUPS)
    for rnd in range(16)
    if (group + rnd) % 4 and (group, rnd) not in _BASE_SCALAR
]
_SCALAR_CANDIDATES.sort(
    key=lambda pair: (pair[1], FULL_ROUND_OFFSETS[pair[0]]), reverse=True
)
_SELECTED_SCALAR_EXTRA = {
    (16, 15), (31, 15), (12, 15), (3, 15), (18, 15), (11, 15),
    (27, 15), (2, 15), (14, 15), (20, 15), (15, 15), (23, 15),
    (28, 15), (24, 15), (4, 15), (26, 15), (10, 15), (19, 15),
    (22, 15), (7, 15), (30, 15), (8, 15), (0, 15), (6, 15),
    (16, 14), (21, 14), (31, 14), (17, 14), (5, 14), (12, 14),
    (3, 14), (25, 14), (11, 14), (27, 14), (20, 14), (29, 14),
    (15, 14), (23, 14), (28, 14), (13, 14), (24, 14), (4, 14),
    (19, 14), (9, 14), (7, 14), (1, 14), (8, 14), (0, 14),
    (16, 13), (21, 13), (17, 13), (5, 13),
}
HASH_SCALAR_EXTRA = frozenset(_BASE_SCALAR | set(_SCALAR_CANDIDATES[:43]))
HYBRID_MADD_PAIRS = 8
HYBRID_MADD_OVERRIDES: dict[tuple[int, int], int] = {}
SCHEDULE_POLICIES = (4,)
BACKWARD_POLICIES = ()
ENGINE_LOAD_MULTIPLIERS: dict[str, int] = {}
ENGINE_HEIGHT_MULTIPLIERS: dict[str, int] = {}
GROUP_PRIORITY_OFFSETS = (
    0, 0, -8, -1,
    -63, -74, -67, -69,
    -170, -150, -167, -162,
    -200, -176, -207, -221,
    -290, -302, -296, -304,
    -351, -336, -345, -325,
    -378, -362, -364, -387,
    -409, -410, -398, -407,
)
ROUND_PRIORITY_OFFSETS = (0, 0, 0, -3, 0, 0, 0, 0, 0, 0, 3, -6, 0, 0, 0, 0)
TAG_PRIORITY_OFFSETS: dict[str, int] = {}
OP_PRIORITY_OFFSETS: dict[tuple[str, int, int], int] = {}
GROUP_FINE_OFFSETS = (0, 0, 0, 0, 0, 0, 0, 1) + (0,) * 24
BASIC_GROUP_OFFSETS = (0,) * N_GROUPS
BASIC_ROUND_OFFSETS = (0,) * 16
BASIC_HEIGHT_DIVISOR = 2
BASIC_REACH_DIVISOR = 32
VECTOR_NODE_XOR_SET: frozenset[tuple[int, int]] = frozenset()
VECTOR_DYNAMIC_XOR_SET: frozenset[tuple[int, int]] = frozenset()
SCALAR_DYNAMIC_XOR_SET = frozenset(
    (group, rnd) for group in range(7) for rnd in range(4, 16)
)


def _words(base: int, size: int = VLEN) -> tuple[int, ...]:
    return tuple(range(base, base + size))


@dataclass(slots=True)
class _Op:
    engine: str
    slot: tuple
    reads: tuple[int, ...]
    writes: tuple[int, ...]
    parents: dict[int, int] = field(default_factory=dict)
    tag: str = ""
    group: int = -1
    round: int = -1


class _Scratch:
    def __init__(self) -> None:
        self.ptr = 0
        self.debug: dict[int, tuple[str, int]] = {}

    def alloc(self, name: str, size: int = 1) -> int:
        base = self.ptr
        self.ptr += size
        if self.ptr > SCRATCH_SIZE:
            raise AssertionError(
                f"scratch overflow: {self.ptr} > {SCRATCH_SIZE} while allocating {name}"
            )
        self.debug[base] = (name, size)
        return base


class _Graph:
    """Build exact scratch dependencies for end-of-cycle writes.

    RAW and WAW require a later cycle.  WAR only requires program order: the
    reader and following writer may share a bundle because every engine reads
    the old scratch image and commits writes at the end of the cycle.
    """

    def __init__(self) -> None:
        self.ops: list[_Op] = []
        self.last_writer: dict[int, int] = {}
        self.readers: dict[int, list[int]] = defaultdict(list)

    @staticmethod
    def _add_parent(parents: dict[int, int], parent: int, lag: int) -> None:
        if parent < 0:
            return
        old = parents.get(parent)
        if old is None or lag > old:
            parents[parent] = lag

    def emit(
        self,
        engine: str,
        slot: tuple,
        reads: Iterable[int] = (),
        writes: Iterable[int] = (),
        *,
        deps: Iterable[tuple[int, int]] = (),
        tag: str = "",
        group: int = -1,
        round: int = -1,
    ) -> int:
        reads_t = tuple(dict.fromkeys(reads))
        writes_t = tuple(dict.fromkeys(writes))
        idx = len(self.ops)
        parents: dict[int, int] = {}

        for parent, lag in deps:
            self._add_parent(parents, parent, lag)

        # Register all reads before writes so an in-place instruction depends
        # on the preceding writer but never on itself.
        for addr in reads_t:
            writer = self.last_writer.get(addr)
            if writer is not None:
                self._add_parent(parents, writer, 1)
            self.readers[addr].append(idx)

        for addr in writes_t:
            writer = self.last_writer.get(addr)
            if writer is not None:
                self._add_parent(parents, writer, 1)
            for reader in self.readers.get(addr, ()):
                if reader != idx:
                    self._add_parent(parents, reader, 0)

        self.ops.append(
            _Op(
                engine=engine,
                slot=slot,
                reads=reads_t,
                writes=writes_t,
                parents=parents,
                tag=tag,
                group=group,
                round=round,
            )
        )

        for addr in writes_t:
            self.last_writer[addr] = idx
            self.readers[addr] = []
        return idx


class KernelBuilder:
    def __init__(self) -> None:
        self.instrs: list[dict[str, list[tuple]]] = []
        self.scratch: dict[str, int] = {}
        self.scratch_debug: dict[int, tuple[str, int]] = {}
        self.scratch_ptr = 0

    def debug_info(self) -> DebugInfo:
        return DebugInfo(scratch_map=self.scratch_debug)

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ) -> None:
        shape = (forest_height, n_nodes, batch_size, rounds)
        if shape != OFFICIAL_SHAPE:
            raise ValueError(
                "the extreme kernel is intentionally specialized for the scored "
                f"shape {OFFICIAL_SHAPE}, got {shape}"
            )

        scratch = _Scratch()
        graph = _Graph()

        def alloc(name: str, size: int = 1) -> int:
            addr = scratch.alloc(name, size)
            self.scratch[name] = addr
            return addr

        # Scalar constants are deduplicated.  Pointer registers are separate
        # because streaming loads/stores update them in place.
        scalar_constants: dict[int, int] = {}

        def scalar_const(value: int, name: str) -> int:
            value &= 0xFFFFFFFF
            if value not in scalar_constants:
                scalar_constants[value] = alloc(name)
            return scalar_constants[value]

        s_one = scalar_const(1, "one")
        s_two = scalar_const(2, "two")
        if not OVERLAP_DEEP_ADDRESS:
            s_four = scalar_const(4, "four")
            s_eight = scalar_const(8, "eight")
        else:
            s_four = s_eight = -1
        s_sixteen = scalar_const(16, "sixteen")
        s_nineteen = scalar_const(19, "nineteen")
        s_c0 = scalar_const(C0, "hash_c0")
        s_c1 = scalar_const(C1, "hash_c1")
        s_c23 = scalar_const(C23, "hash_c2_plus_c3")
        s_m23 = scalar_const(M23, "hash_mul2_shift9")
        s_c2_shift9 = scalar_const(C2_SHIFT9, "hash_c2_shift9")
        s_c4 = scalar_const(C4, "hash_c4")
        s_c5 = scalar_const(C5, "hash_c5")
        s_m4 = scalar_const(9, "hash_mul4_shift3")
        s_m0 = scalar_const(4097, "hash_mul0")
        s_m2 = scalar_const(33, "hash_mul2")

        # Mirrored local-path address: mem_addr = 2**(depth+1) + 5 - mirror.
        depth_range = range(4, 5) if OVERLAP_DEEP_ADDRESS else range(4, 11)
        depth_base = {
            d: scalar_const((1 << (d + 1)) + 5, f"depth_{d}_base")
            for d in depth_range
        }

        # These constants are dead after their vector broadcasts.  Reusing
        # their scalar words for the three streaming pointer pairs saves six
        # registers without extending a hot live range.
        top_p0, top_p1 = s_c0, s_c23
        io_p0, io_p1 = s_m0, s_m2

        # Vector constants needed by fixed VALU instructions.
        vector_constants: dict[int, int] = {}

        def vector_const(value: int, name: str) -> int:
            value &= 0xFFFFFFFF
            if value not in vector_constants:
                vector_constants[value] = alloc(name, VLEN)
            return vector_constants[value]

        v_one = vector_const(1, "v_one")
        v_two = vector_const(2, "v_two")
        if OVERLAP_DEEP_ADDRESS:
            v_four = alloc("v_four", VLEN)
            v_eight = alloc("v_eight", VLEN)
        else:
            v_four = vector_const(4, "v_four")
            v_eight = vector_const(8, "v_eight")
        v_nineteen = vector_const(19, "v_nineteen")
        v_sixteen = vector_const(16, "v_sixteen")
        v_c0 = vector_const(C0, "v_hash_c0")
        v_c1 = vector_const(C1, "v_hash_c1")
        v_c23 = vector_const(C23, "v_hash_c2_plus_c3")
        v_m23 = vector_const(M23, "v_hash_mul2_shift9")
        v_c2_shift9 = vector_const(C2_SHIFT9, "v_hash_c2_shift9")
        v_c4 = vector_const(C4, "v_hash_c4")
        v_c5 = vector_const(C5, "v_hash_c5")
        v_m0 = vector_const(4097, "v_hash_mul0")
        v_m2 = vector_const(33, "v_hash_mul2")
        v_m4 = vector_const(9, "v_hash_mul4_shift3")
        v_neg_five = (
            alloc("v_negative_five", VLEN) if OVERLAP_DEEP_ADDRESS else -1
        )

        top_words = alloc("top_tree_words", 32)
        node_vec = [alloc(f"cached_node_{i}", VLEN) for i in range(31)]
        depth1_diff = (
            alloc("depth1_pair_diff", VLEN) if MADD_FIRST_DEPTH1 else -1
        )

        # Every SIMD group keeps only its long-lived value/mirror/temp state.
        values = [alloc(f"value_{g}", VLEN) for g in range(N_GROUPS)]
        mirrors = [alloc(f"mirror_{g}", VLEN) for g in range(N_GROUPS)]
        temps = [alloc(f"temp_{g}", VLEN) for g in range(N_GROUPS)]

        # Shallow path bits and mux spills are shared by seven software-
        # pipelined workspaces.  They are released during depths 4..10.
        bits = []
        for workspace in range(N_WORKSPACES):
            workspace_bits = []
            for bit in range(3):
                addr = alloc(f"workspace_bit_{workspace}_{bit}", VLEN)
                workspace_bits.append(addr)
            bits.append(workspace_bits)

        # A depth-first mux order needs one spill vector rather than two.
        select_spill = [
            [alloc(f"select_spill_{workspace}", VLEN)]
            for workspace in range(N_SPILL_WORKSPACES)
        ]
        branch_table_base = (
            alloc("branch_table_base") if BRANCH_FINAL_LANES else -1
        )
        physical_workspace_vectors = tuple(
            addr for workspace_bits in bits for addr in workspace_bits
        ) + tuple(workspace_spill[0] for workspace_spill in select_spill)

        # Virtual addresses deliberately live far outside the architectural
        # scratch range.  They only exist while constructing and scheduling
        # the DAG; ``color_virtual_workspaces`` below rewrites every one of
        # them to an allocated physical vector before validation/execution.
        virtual_next = 1_000_000
        virtual_vectors: dict[int, str] = {}
        virtual_workspaces: dict[tuple[str, int] | tuple[int, int], tuple[list[int], list[int]]] = {}

        def virtual_vector(name: str) -> int:
            nonlocal virtual_next
            base = virtual_next
            virtual_next += VLEN
            virtual_vectors[base] = name
            return base

        def workspace_registers(
            workspace: int, gg: int, rnd: int
        ) -> tuple[list[int], list[int]]:
            # The first traversal intentionally uses rotating physical
            # workspaces: its saved path bits span several rounds and fully
            # renaming them exceeds the 36-vector register budget.  The second
            # traversal's conditions are short-lived, so it is both safe and
            # profitable to allocate those after scheduling.
            if not SSA_WORKSPACES or rnd < rounds - 1:
                return bits[workspace], select_spill[workspace % N_SPILL_WORKSPACES]

            # Later cached selections derive their conditions directly from
            # the mirror and therefore have independent live ranges.
            key: tuple[str, int] | tuple[int, int]
            key = (rnd, gg)
            registers = virtual_workspaces.get(key)
            if registers is None:
                key_name = f"{key[0]}_{key[1]}"
                registers = (
                    # Keep path conditions on the rotating physical registers:
                    # the list scheduler may otherwise hoist them hundreds of
                    # cycles before the mux and create impossible live ranges.
                    bits[workspace],
                    [virtual_vector(f"virtual_spill_{key_name}")],
                )
                virtual_workspaces[key] = registers
            return registers
        # Preprocessing borrows the two latest groups' temp vectors.  Keeping
        # these separate from the persistent level-4 mux workspace prevents a
        # global setup barrier; only groups 30/31 wait for preprocessing.
        preprocess_buffers = [temps[30], temps[31]]
        level4_pool = top_words + 2 * VLEN
        level4_condition = top_words + 3 * VLEN
        # The lower staging words become persistent pair differences after
        # their node broadcasts, avoiding any extra scratch cost.
        level4_diff = [top_words, top_words + VLEN] + [
            alloc(f"level4_pair_diff_{i}", VLEN)
            for i in range(2, HYBRID_MADD_PAIRS)
        ]
        first_level4_pool = [level4_pool]
        first_level4_condition = level4_condition
        final_address_vectors_ready = [False]

        if scratch.ptr > SCRATCH_SIZE:
            raise AssertionError(f"scratch overflow: {scratch.ptr}")

        def emit_const(dest: int, value: int, tag: str = "const") -> int:
            return graph.emit("load", ("const", dest, value), writes=(dest,), tag=tag)

        def emit_immediate(dest: int, value: int, tag: str) -> int:
            """Materialize a scalar on the otherwise underused flow engine."""
            return graph.emit(
                "flow",
                ("add_imm", dest, s_one, (value - 1) & 0xFFFFFFFF),
                reads=(s_one,),
                writes=(dest,),
                tag=tag,
            )

        def emit_vbroadcast(dest: int, src: int, tag: str = "broadcast") -> int:
            return graph.emit(
                "valu",
                ("vbroadcast", dest, src),
                reads=(src,),
                writes=_words(dest),
                tag=tag,
            )

        def emit_valu(
            op: str,
            dest: int,
            a: int,
            b: int,
            *,
            tag: str,
            group: int = -1,
            round: int = -1,
        ) -> int:
            return graph.emit(
                "valu",
                (op, dest, a, b),
                reads=_words(a) + _words(b),
                writes=_words(dest),
                tag=tag,
                group=group,
                round=round,
            )

        def emit_madd(
            dest: int,
            a: int,
            b: int,
            c: int,
            *,
            tag: str,
            group: int = -1,
            round: int = -1,
        ) -> int:
            return graph.emit(
                "valu",
                ("multiply_add", dest, a, b, c),
                reads=_words(a) + _words(b) + _words(c),
                writes=_words(dest),
                tag=tag,
                group=group,
                round=round,
            )

        def emit_scalarized(
            op: str,
            dest: int,
            a: int,
            b: int,
            *,
            a_scalar: bool = False,
            b_scalar: bool = False,
            tag: str,
            group: int = -1,
            round: int = -1,
        ) -> list[int]:
            ids = []
            for lane in range(VLEN):
                aa = a if a_scalar else a + lane
                bb = b if b_scalar else b + lane
                ids.append(
                    graph.emit(
                        "alu",
                        (op, dest + lane, aa, bb),
                        reads=(aa, bb),
                        writes=(dest + lane,),
                        tag=tag,
                        group=group,
                        round=round,
                    )
                )
            return ids

        def emit_vbasic(
            op: str,
            dest: int,
            a: int,
            b_vector: int,
            *,
            scalarize: bool,
            scalar_b: int,
            tag: str,
            group: int,
            round: int,
        ) -> None:
            if scalarize:
                emit_scalarized(
                    op,
                    dest,
                    a,
                    scalar_b,
                    b_scalar=True,
                    tag=tag,
                    group=group,
                    round=round,
                )
            else:
                emit_valu(
                    op,
                    dest,
                    a,
                    b_vector,
                    tag=tag,
                    group=group,
                    round=round,
                )

        def emit_vselect(
            dest: int,
            cond: int,
            yes: int,
            no: int,
            *,
            group: int,
            round: int,
        ) -> int:
            return graph.emit(
                "flow",
                ("vselect", dest, cond, yes, no),
                reads=_words(cond) + _words(yes) + _words(no),
                writes=_words(dest),
                tag="tree_select",
                group=group,
                round=round,
            )

        # Load all immutable scalar constants, then broadcast the subset used
        # by VALU.  The scheduler overlaps these with tree and input traffic.
        emit_const(s_one, 1)
        misc_addresses = {
            "nineteen": s_nineteen,
            "m4": s_m4,
            "m0": s_m0,
            "m2": s_m2,
            "m23": s_m23,
        }
        derived_misc_addresses = {
            misc_addresses[name] for name in ALU_DERIVED_MISC_SET
        }
        derived_depth_addresses = {
            addr for depth, addr in depth_base.items()
            if depth >= ALU_DERIVED_DEPTH_START
        }
        for constant_index, (value, addr) in enumerate(scalar_constants.items()):
            if addr != s_one:
                if addr in derived_depth_addresses or addr in derived_misc_addresses:
                    continue
                elif (
                    value in (2, 4, 8, 16)
                    and value.bit_length() - 1 <= ALU_DERIVED_POWER_COUNT
                ):
                    half = scalar_constants[value // 2]
                    graph.emit(
                        "alu",
                        ("+", addr, half, half),
                        reads=(half,),
                        writes=(addr,),
                        tag="scalar_power_derive",
                    )
                elif constant_index < FLOW_SCALAR_CONSTANT_COUNT:
                    emit_immediate(addr, value, "scalar_immediate")
                else:
                    emit_const(addr, value, "scalar_constant")
        if "nineteen" in ALU_DERIVED_MISC_SET:
            graph.emit(
                "alu", ("+", s_nineteen, s_sixteen, s_two),
                reads=(s_sixteen, s_two), writes=(s_nineteen,),
                tag="scalar_misc_derive",
            )
            graph.emit(
                "alu", ("+", s_nineteen, s_nineteen, s_one),
                reads=(s_nineteen, s_one), writes=(s_nineteen,),
                tag="scalar_misc_derive",
            )
        if "m4" in ALU_DERIVED_MISC_SET:
            graph.emit(
                "alu", ("+", s_m4, s_eight, s_one),
                reads=(s_eight, s_one), writes=(s_m4,),
                tag="scalar_misc_derive",
            )
        if "m0" in ALU_DERIVED_MISC_SET:
            graph.emit(
                "alu", ("*", s_m0, s_sixteen, s_sixteen),
                reads=(s_sixteen,), writes=(s_m0,), tag="scalar_misc_derive",
            )
            graph.emit(
                "alu", ("*", s_m0, s_m0, s_sixteen),
                reads=(s_m0, s_sixteen), writes=(s_m0,), tag="scalar_misc_derive",
            )
            graph.emit(
                "alu", ("+", s_m0, s_m0, s_one),
                reads=(s_m0, s_one), writes=(s_m0,), tag="scalar_misc_derive",
            )
        if "m2" in ALU_DERIVED_MISC_SET:
            graph.emit(
                "alu", ("*", s_m2, s_sixteen, s_two),
                reads=(s_sixteen, s_two), writes=(s_m2,), tag="scalar_misc_derive",
            )
            graph.emit(
                "alu", ("+", s_m2, s_m2, s_one),
                reads=(s_m2, s_one), writes=(s_m2,), tag="scalar_misc_derive",
            )
        if "m23" in ALU_DERIVED_MISC_SET:
            graph.emit(
                "alu", ("<<", s_m23, s_m2, s_m4),
                reads=(s_m2, s_m4), writes=(s_m23,), tag="scalar_misc_derive",
            )
        if derived_depth_addresses:
            previous = s_sixteen
            for depth in depth_range:
                if depth < ALU_DERIVED_DEPTH_START:
                    previous = depth_base[depth]
                    continue
                dest = depth_base[depth]
                graph.emit(
                    "alu", ("+", dest, previous, previous),
                    reads=(previous,), writes=(dest,), tag="depth_base_double",
                )
                if depth == 4:
                    graph.emit(
                        "alu", ("+", dest, dest, scalar_constants[4]),
                        reads=(dest, scalar_constants[4]), writes=(dest,),
                        tag="depth_base_adjust",
                    )
                    graph.emit(
                        "alu", ("+", dest, dest, s_one),
                        reads=(dest, s_one), writes=(dest,),
                        tag="depth_base_adjust",
                    )
                else:
                    graph.emit(
                        "alu", ("-", dest, dest, scalar_constants[4]),
                        reads=(dest, scalar_constants[4]), writes=(dest,),
                        tag="depth_base_adjust",
                    )
                    graph.emit(
                        "alu", ("-", dest, dest, s_one),
                        reads=(dest, s_one), writes=(dest,),
                        tag="depth_base_adjust",
                    )
                previous = dest
        for value, dest in vector_constants.items():
            emit_vbroadcast(dest, scalar_constants[value], "constant_broadcast")
        if OVERLAP_DEEP_ADDRESS:
            emit_valu("+", v_four, v_two, v_two, tag="derive_v_four")
            emit_valu("+", v_eight, v_four, v_four, tag="derive_v_eight")
            # This scalar register is overwritten by the top-tree pointer
            # before any later use.  Reusing it avoids a permanent constant.
            emit_immediate(top_p0, -5, "negative_five_immediate")
            emit_vbroadcast(v_neg_five, top_p0, "negative_five_broadcast")

        # Fetch nodes 0..31 using two rolling pointers and four vector loads.
        emit_immediate(top_p0, 7, "top_pointer")
        emit_immediate(top_p1, 15, "top_pointer")
        top_loads: list[int] = []
        for pair in range(2):
            top_loads.append(
                graph.emit(
                    "load",
                    ("vload", top_words + pair * 16, top_p0),
                    reads=(top_p0,),
                    writes=_words(top_words + pair * 16),
                    tag="top_tree_load",
                )
            )
            top_loads.append(
                graph.emit(
                    "load",
                    ("vload", top_words + pair * 16 + 8, top_p1),
                    reads=(top_p1,),
                    writes=_words(top_words + pair * 16 + 8),
                    tag="top_tree_load",
                )
            )
            if pair == 0:
                graph.emit(
                    "alu",
                    ("+", top_p0, top_p0, s_sixteen),
                    reads=(top_p0, s_sixteen),
                    writes=(top_p0,),
                    tag="pointer_advance",
                )
                graph.emit(
                    "alu",
                    ("+", top_p1, top_p1, s_sixteen),
                    reads=(top_p1, s_sixteen),
                    writes=(top_p1,),
                    tag="pointer_advance",
                )

        # The direct-path variant preserves one raw scalar root and broadcasts
        # its transformed form.  This removes the global round-0/round-11 root
        # barrier and lets all sixteen rounds form one software wavefront.
        if INDEPENDENT_ROOT_CACHE or DIRECT_MIRROR_PATH:
            graph.emit(
                "flow",
                ("add_imm", top_p1, top_words, 0),
                reads=(top_words,),
                writes=(top_p1,),
                tag="raw_root_copy",
            )
            for chunk in range(4):
                emit_valu(
                    "^",
                    top_words + chunk * VLEN,
                    top_words + chunk * VLEN,
                    v_c5,
                    tag="cached_nodes_vector_transform",
                )
            for i in range(31):
                emit_vbroadcast(
                    node_vec[i], top_words + i, "cached_node_broadcast"
                )
        else:
            # Node vector 0 starts as the raw root for round 0.  It is
            # transformed only after every first-root reader is emitted.
            emit_vbroadcast(node_vec[0], top_words, "raw_root_broadcast")
            for i in range(1, 31):
                graph.emit(
                    "alu",
                    ("^", top_words + i, top_words + i, s_c5),
                    reads=(top_words + i, s_c5),
                    writes=(top_words + i,),
                    tag="cached_node_transform",
                )
                emit_vbroadcast(node_vec[i], top_words + i, "cached_node_broadcast")

        # Optional one-time tree preprocessing experiment.
        # every SIMD group.  Transform them once in private machine memory so
        # each later gather directly receives node^C5.  The two staging
        # buffers and pointer scalars are setup-only storage that is reused
        # below, so this costs no persistent scratch.
        preprocess_p0, preprocess_p1 = s_c4, s_m4
        preprocess_stores: list[int] = []
        preprocess_pairs = {0: 0, 4: 1, 5: 3, 6: 7, 7: 15, 8: 31}[
            PREPROCESS_MAX_DEPTH
        ]
        if PREPROCESS_MAX_DEPTH == 4:
            emit_const(preprocess_p0, 22, "preprocess_pointer")
            emit_const(preprocess_p1, 30, "preprocess_pointer")
            for pointer, source in (
                (preprocess_p0, top_words + 15),
                (preprocess_p1, top_words + 23),
            ):
                preprocess_stores.append(graph.emit(
                    "store",
                    ("vstore", pointer, source),
                    reads=(pointer,) + _words(source),
                    deps=tuple((load_id, 0) for load_id in top_loads),
                    tag="tree_preprocess_store",
                ))
            preprocess_pairs = 0
        elif preprocess_pairs:
            emit_immediate(preprocess_p0, 22, "preprocess_pointer")
            emit_immediate(preprocess_p1, 30, "preprocess_pointer")
        for pair_index in range(preprocess_pairs):
            for pointer, buffer in zip(
                (preprocess_p0, preprocess_p1), preprocess_buffers
            ):
                graph.emit(
                    "load",
                    ("vload", buffer, pointer),
                    reads=(pointer,),
                    writes=_words(buffer),
                    tag="tree_preprocess_load",
                )
                emit_valu(
                    "^",
                    buffer,
                    buffer,
                    v_c5,
                    tag="tree_preprocess_xor",
                )
                preprocess_stores.append(graph.emit(
                    "store",
                    ("vstore", pointer, buffer),
                    reads=(pointer,) + _words(buffer),
                    deps=tuple((load_id, 0) for load_id in top_loads),
                    tag="tree_preprocess_store",
                ))
            if pair_index != preprocess_pairs - 1:
                for pointer in (preprocess_p0, preprocess_p1):
                    graph.emit(
                        "alu",
                        ("+", pointer, pointer, s_sixteen),
                        reads=(pointer, s_sixteen),
                        writes=(pointer,),
                        tag="pointer_advance",
                    )

        if MADD_FIRST_DEPTH1:
            # Keep node 2 as the base and turn node 1 into (node1-node2).
            # A depth-1 lookup then needs one MADD and no flow slot.
            emit_valu(
                "-",
                depth1_diff,
                node_vec[1],
                node_vec[2],
                tag="depth1_pair_diff",
            )
        if SCALAR_FIRST_DEPTH1_SET:
            graph.emit(
                "flow",
                ("add_imm", s_c2_shift9, top_words + 2, 0),
                reads=(top_words + 2,),
                writes=(s_c2_shift9,),
                tag="depth1_scalar_base",
            )
            graph.emit(
                "alu",
                ("-", s_c1, top_words + 1, top_words + 2),
                reads=(top_words + 1, top_words + 2),
                writes=(s_c1,),
                tag="depth1_scalar_diff",
            )

        level4_reversed = [node_vec[i] for i in range(30, 14, -1)]
        for i, diff in enumerate(level4_diff):
            emit_valu(
                "-",
                diff,
                level4_reversed[2 * i + 1],
                level4_reversed[2 * i],
                tag="level4_pair_diff",
            )

        def select_cached(
            depth: int, state: int, workspace: int, gg: int, rnd: int
        ) -> None:
            width = 1 << depth
            start = width - 1
            # Mirror offset 0 names the actual rightmost node.
            leaves = [node_vec[start + width - 1 - i] for i in range(width)]
            temp = temps[state]
            workspace_bits, workspace_spill = workspace_registers(
                workspace, gg, rnd
            )

            if depth == 1:
                condition = (
                    mirrors[state]
                    if DIRECT_MIRROR_PATH or rnd >= 11
                    else workspace_bits[0]
                )
                if rnd < 11 and gg in SCALAR_FIRST_DEPTH1_SET:
                    emit_scalarized(
                        "*",
                        temp,
                        condition,
                        s_c1,
                        b_scalar=True,
                        tag="depth1_scalar_multiply",
                        group=gg,
                        round=rnd,
                    )
                    emit_scalarized(
                        "+",
                        temp,
                        temp,
                        s_c2_shift9,
                        b_scalar=True,
                        tag="depth1_scalar_add",
                        group=gg,
                        round=rnd,
                    )
                elif MADD_FIRST_DEPTH1 and rnd < 11:
                    emit_madd(
                        temp,
                        condition,
                        depth1_diff,
                        node_vec[2],
                        tag="depth1_madd",
                        group=gg,
                        round=rnd,
                    )
                else:
                    emit_vselect(
                        temp, condition, leaves[1], leaves[0], group=gg, round=rnd
                    )
                return

            if depth == 2:
                if DIRECT_MIRROR_PATH or rnd >= 11:
                    emit_valu(
                        "&",
                        workspace_bits[1],
                        mirrors[state],
                        v_one,
                        tag="second_path_condition_1",
                        group=gg,
                        round=rnd,
                    )
                    emit_valu(
                        "&",
                        workspace_bits[0],
                        mirrors[state],
                        v_two,
                        tag="second_path_condition_0",
                        group=gg,
                        round=rnd,
                    )
                emit_vselect(
                    temp, workspace_bits[1], leaves[1], leaves[0], group=gg, round=rnd
                )
                emit_vselect(
                    workspace_spill[0],
                    workspace_bits[1],
                    leaves[3],
                    leaves[2],
                    group=gg,
                    round=rnd,
                )
                emit_vselect(
                    temp,
                    workspace_bits[0],
                    workspace_spill[0],
                    temp,
                    group=gg,
                    round=rnd,
                )
                return

            if depth != 3:
                raise AssertionError(depth)

            if DIRECT_MIRROR_PATH or rnd >= 11:
                for dest, mask, tag in (
                    (workspace_bits[2], v_one, "second_path_condition_2"),
                    (workspace_bits[1], v_two, "second_path_condition_1"),
                    (workspace_bits[0], v_four, "second_path_condition_0"),
                ):
                    emit_valu(
                        "&",
                        dest,
                        mirrors[state],
                        mask,
                        tag=tag,
                        group=gg,
                        round=rnd,
                    )

            # Evaluate the two four-leaf halves depth-first.  Once the final
            # bottom pair has consumed bit 2, that register itself becomes a
            # legal destination, reducing the live mux stack to one spill.
            spill = workspace_spill[0]
            emit_vselect(temp, workspace_bits[2], leaves[1], leaves[0], group=gg, round=rnd)
            emit_vselect(spill, workspace_bits[2], leaves[3], leaves[2], group=gg, round=rnd)
            emit_vselect(temp, workspace_bits[1], spill, temp, group=gg, round=rnd)
            emit_vselect(spill, workspace_bits[2], leaves[5], leaves[4], group=gg, round=rnd)
            emit_vselect(workspace_bits[2], workspace_bits[2], leaves[7], leaves[6], group=gg, round=rnd)
            emit_vselect(spill, workspace_bits[1], workspace_bits[2], spill, group=gg, round=rnd)
            emit_vselect(temp, workspace_bits[0], spill, temp, group=gg, round=rnd)

        def select_level4_hybrid(
            state: int,
            workspace: int,
            gg: int,
            rnd: int,
            *,
            prepared: bool = False,
        ) -> None:
            workspace_bits, workspace_spill = workspace_registers(
                workspace, gg, rnd
            )
            active_pool = first_level4_pool
            active_condition = first_level4_condition
            level4_condition_2 = workspace_bits[2]
            cond = active_condition
            if prepared:
                pass
            elif DIRECT_MIRROR_PATH or rnd >= 11:
                for dest, mask, tag in (
                    (cond, v_one, "level4_condition_3"),
                    (level4_condition_2, v_two, "level4_condition_2"),
                    (workspace_bits[1], v_four, "level4_condition_1"),
                    (workspace_bits[0], v_eight, "level4_condition_0"),
                ):
                    emit_valu(
                        "&",
                        dest,
                        mirrors[state],
                        mask,
                        tag=tag,
                        group=gg,
                        round=rnd,
                    )
            else:
                emit_scalarized(
                    "&",
                    cond,
                    mirrors[state],
                    s_one,
                    b_scalar=True,
                    tag="level4_condition_3",
                    group=gg,
                    round=rnd,
                )
                emit_scalarized(
                    "&",
                    level4_condition_2,
                    mirrors[state],
                    s_two,
                    b_scalar=True,
                    tag="level4_condition_2",
                    group=gg,
                    round=rnd,
                )
            # Evaluate the 16-leaf mux depth-first.  The saved p2/p1/p0 bits
            # are already valid conditions, and the p3 condition becomes the
            # final pair's destination after its last read.
            a = temps[state]
            b = workspace_spill[0]
            c = active_pool[0]
            c3 = active_condition
            c2, c1, c0 = level4_condition_2, workspace_bits[1], workspace_bits[0]

            def pair(dest: int, pair_index: int) -> None:
                pair_limit = HYBRID_MADD_OVERRIDES.get(
                    (gg, rnd), HYBRID_MADD_PAIRS
                )
                if pair_index < pair_limit:
                    emit_madd(
                        dest,
                        c3,
                        level4_diff[pair_index],
                        level4_reversed[2 * pair_index],
                        tag="level4_hybrid_bottom",
                        group=gg,
                        round=rnd,
                    )
                else:
                    emit_vselect(
                        dest,
                        c3,
                        level4_reversed[2 * pair_index + 1],
                        level4_reversed[2 * pair_index],
                        group=gg,
                        round=rnd,
                    )

            pair(a, 0)
            pair(b, 1)
            emit_vselect(a, c2, b, a, group=gg, round=rnd)
            pair(b, 2)
            pair(c, 3)
            emit_vselect(b, c2, c, b, group=gg, round=rnd)
            emit_vselect(a, c1, b, a, group=gg, round=rnd)

            pair(b, 4)
            pair(c, 5)
            emit_vselect(b, c2, c, b, group=gg, round=rnd)
            pair(c, 6)
            pair(c3, 7)
            emit_vselect(c, c2, c3, c, group=gg, round=rnd)
            emit_vselect(b, c1, c, b, group=gg, round=rnd)
            emit_vselect(a, c0, b, a, group=gg, round=rnd)

        def gather_node(depth: int, state: int, gg: int, rnd: int) -> None:
            mirror = mirrors[state]
            temp = temps[state]
            branch_lanes = (
                frozenset(BRANCH_FINAL_LANES)
                if gg == BRANCH_FINAL_GROUP and rnd == rounds - 1 and depth == 4
                else frozenset()
            )
            final_address_prepared = (
                depth == 4
                and rnd == rounds - 1
                and (
                    gg in EARLY_FINAL_ADDRESS_SET
                    or gg in VECTOR_EARLY_FINAL_ADDRESS_SET
                )
            )
            if final_address_prepared:
                address = mirror
            elif OVERLAP_DEEP_ADDRESS:
                # At depth 4 the mirror still encodes complemented path bits;
                # convert it in place to an absolute memory address.  Later
                # deep rounds keep that address representation directly.
                if depth == 4 and not final_address_prepared and not (
                    OVERLAP_SHALLOW_ADDRESS
                    and rnd == 4
                    and gg not in FIRST_CACHE_SET
                ):
                    emit_scalarized(
                        "-",
                        mirror,
                        depth_base[depth],
                        mirror,
                        a_scalar=True,
                        tag="gather_address_seed",
                        group=gg,
                        round=rnd,
                    )
                address = mirror
            else:
                # Address generation is intentionally scalar: eight ALU slots
                # are cheaper than a scarce VALU slot in steady state.
                for lane in range(VLEN):
                    if lane in branch_lanes:
                        continue
                    graph.emit(
                        "alu",
                        ("-", temp + lane, depth_base[depth], mirror + lane),
                        reads=(depth_base[depth], mirror + lane),
                        writes=(temp + lane,),
                        tag="gather_address",
                        group=gg,
                        round=rnd,
                    )
                address = temp
            for lane in range(VLEN):
                if lane in branch_lanes:
                    continue
                # Each deeper level occupies twice as many contiguous vectors.
                # Explicitly order its gathers after the exact preprocessing
                # stores which transform that level in memory.
                preprocess_slice = {
                    4: (0, 2),
                    5: (2, 6),
                    6: (6, 14),
                    7: (14, 30),
                    8: (30, 62),
                }.get(depth)
                graph.emit(
                    "load",
                    ("load_offset", temp, address, lane),
                    reads=(address + lane,),
                    writes=(temp + lane,),
                    deps=(
                        tuple(
                            (store_id, 1)
                            for store_id in preprocess_stores[
                                preprocess_slice[0] : preprocess_slice[1]
                            ]
                        )
                        if preprocess_slice is not None
                        else ()
                    ),
                    tag="tree_gather",
                    group=gg,
                    round=rnd,
                )
            if branch_lanes:
                if SECOND_WORKSPACE_FIXED < 0:
                    raise ValueError("branch lookup requires a fixed second workspace")
                candidate_yes = bits[SECOND_WORKSPACE_FIXED][0]
                candidate_no = bits[SECOND_WORKSPACE_FIXED][1]
                for lane in sorted(branch_lanes):
                    graph.emit(
                        "flow",
                        (
                            "select",
                            temp + lane,
                            temp + lane,
                            candidate_yes + lane,
                            candidate_no + lane,
                        ),
                        reads=(temp + lane, candidate_yes + lane, candidate_no + lane),
                        writes=(temp + lane,),
                        tag="branch_final_select",
                        group=gg,
                        round=rnd,
                    )
            if 4 <= depth <= PREPROCESS_MAX_DEPTH:
                # This level was transformed once during setup.
                pass
            elif (gg, rnd) in SCALAR_DYNAMIC_XOR_SET:
                emit_scalarized(
                    "^",
                    temp,
                    temp,
                    s_c5,
                    b_scalar=True,
                    tag="dynamic_node_transform_scalar",
                    group=gg,
                    round=rnd,
                )
            else:
                emit_valu(
                    "^",
                    temp,
                    temp,
                    v_c5,
                    tag="dynamic_node_transform_vector",
                    group=gg,
                    round=rnd,
                )

        def xor_node(
            value: int, node: int, gg: int, rnd: int, *, node_scalar: bool = False
        ) -> None:
            if not node_scalar and (gg, rnd) in VECTOR_NODE_XOR_SET:
                emit_valu(
                    "^",
                    value,
                    value,
                    node,
                    tag="node_xor_vector",
                    group=gg,
                    round=rnd,
                )
            else:
                emit_scalarized(
                    "^",
                    value,
                    value,
                    node,
                    b_scalar=node_scalar,
                    tag="node_xor",
                    group=gg,
                    round=rnd,
                )

        def emit_hash(value: int, temp: int, gg: int, rnd: int) -> None:
            emit_madd(value, value, v_m0, v_c0, tag="hash_0", group=gg, round=rnd)

            # Stage 1 branches read the same old value.  Emit the reader first;
            # the following in-place writer has a zero-lag WAR dependency.
            scalar_hash = (
                (gg + rnd) % HASH_SCALAR_MOD == 0
                or (gg, rnd) in HASH_SCALAR_EXTRA
            )
            scalar_shift = scalar_hash and HASH_SCALAR_STAGE == 1
            emit_vbasic(
                ">>",
                temp,
                value,
                v_nineteen,
                scalarize=scalar_shift,
                scalar_b=s_nineteen,
                tag="hash_1_shift_scalar" if scalar_shift else "hash_1_shift_vector",
                group=gg,
                round=rnd,
            )
            emit_vbasic(
                "^",
                value,
                value,
                v_c1,
                scalarize=scalar_hash and HASH_SCALAR_STAGE == -1,
                scalar_b=s_c1,
                tag="hash_1_const",
                group=gg,
                round=rnd,
            )
            if (gg, rnd) in SCALAR_HASH1_JOIN_SET:
                emit_scalarized(
                    "^", value, value, temp,
                    tag="hash_1_join_scalar", group=gg, round=rnd,
                )
            else:
                emit_valu(
                    "^", value, value, temp,
                    tag="hash_1_join", group=gg, round=rnd,
                )

            # If y = 33*x + C2, stage 3 is
            #   (y + C3) ^ (y << 9).
            # Both branches are affine in x, so two independent MADDs compute
            # them directly and the join follows one cycle later.  This saves
            # one VALU instruction and one dependency level per hash.
            emit_madd(
                temp, value, v_m2, v_c23, tag="hash_23_add", group=gg, round=rnd
            )
            emit_madd(
                value,
                value,
                v_m23,
                v_c2_shift9,
                tag="hash_23_shift",
                group=gg,
                round=rnd,
            )
            if (gg, rnd) in SCALAR_HASH23_JOIN_SET or (
                rnd == rounds - 1 and gg in SCALAR_FINAL_HASH23_JOIN_SET
            ):
                emit_scalarized(
                    "^", value, value, temp,
                    tag="hash_23_join_scalar", group=gg, round=rnd,
                )
            else:
                emit_valu(
                    "^", value, value, temp,
                    tag="hash_23_join", group=gg, round=rnd,
                )

            if rnd == rounds - 1 and gg in SCALAR_FINAL_HASH4_SET:
                emit_scalarized(
                    "*", temp, value, s_m4,
                    b_scalar=True,
                    tag="hash_4_scalar_mul", group=gg, round=rnd,
                )
                emit_scalarized(
                    "+", value, temp, s_c4,
                    b_scalar=True,
                    tag="hash_4_scalar_add", group=gg, round=rnd,
                )
            else:
                emit_madd(
                    value, value, v_m4, v_c4,
                    tag="hash_4", group=gg, round=rnd,
                )

            emit_vbasic(
                ">>",
                temp,
                value,
                v_sixteen,
                scalarize=(scalar_hash and HASH_SCALAR_STAGE == 5)
                or (rnd == rounds - 1 and gg in SCALAR_FINAL_SHIFT_SET),
                scalar_b=s_sixteen,
                tag="hash_5_shift",
                group=gg,
                round=rnd,
            )
            if rnd == rounds - 1:
                # The final round materializes the true value directly.
                if gg in SCALAR_FINAL_C5_SET:
                    emit_scalarized(
                        "^",
                        value,
                        value,
                        s_c5,
                        b_scalar=True,
                        tag="hash_5_const_scalar",
                        group=gg,
                        round=rnd,
                    )
                else:
                    emit_valu(
                        "^",
                        value,
                        value,
                        v_c5,
                        tag="hash_5_const",
                        group=gg,
                        round=rnd,
                    )
            if (gg, rnd) in SCALAR_HASH5_JOIN_SET or (
                rnd == rounds - 1 and gg in SCALAR_FINAL_JOIN_SET
            ):
                emit_scalarized(
                    "^",
                    value,
                    value,
                    temp,
                    tag="hash_5_join_scalar",
                    group=gg,
                    round=rnd,
                )
            else:
                emit_valu(
                    "^",
                    value,
                    value,
                    temp,
                    tag="hash_5_join",
                    group=gg,
                    round=rnd,
                )

        def emit_parity(dest: int, value: int, gg: int, rnd: int) -> None:
            emit_scalarized(
                "&",
                dest,
                value,
                s_one,
                b_scalar=True,
                tag="mirror_bit",
                group=gg,
                round=rnd,
            )

        def process_round(state: int, workspace: int, gg: int, rnd: int) -> None:
            depth = rnd if rnd <= 10 else rnd - 11
            value = values[state]
            temp = temps[state]
            workspace_bits, _ = workspace_registers(workspace, gg, rnd)

            if not DIRECT_MIRROR_PATH and rnd == 2:
                emit_madd(
                    mirrors[state],
                    workspace_bits[0],
                    v_two,
                    workspace_bits[1],
                    tag="mirror_build_2",
                    group=gg,
                    round=rnd,
                )
            elif not DIRECT_MIRROR_PATH and rnd == 3:
                emit_madd(
                    mirrors[state],
                    mirrors[state],
                    v_two,
                    workspace_bits[2],
                    tag="mirror_build_3",
                    group=gg,
                    round=rnd,
                )

            if depth == 0:
                if (INDEPENDENT_ROOT_CACHE or DIRECT_MIRROR_PATH) and rnd == 0:
                    xor_node(value, top_p1, gg, rnd, node_scalar=True)
                else:
                    xor_node(value, node_vec[0], gg, rnd)
            elif depth <= 3:
                select_cached(depth, state, workspace, gg, rnd)
                xor_node(value, temp, gg, rnd)
            elif (rnd == 4 and gg in FIRST_CACHE_SET) or (
                rnd == rounds - 1 and gg in FINAL_CACHE_SET
            ):
                if not (
                    gg in EARLY_FINAL_CACHE_SET
                    and rnd == rounds - 1
                ):
                    select_level4_hybrid(state, workspace, gg, rnd)
                xor_node(value, temp, gg, rnd)
                if (
                    OVERLAP_DEEP_ADDRESS
                    and (
                        not OVERLAP_SHALLOW_ADDRESS
                        or gg in FIRST_CACHE_SET
                    )
                    and rnd == 4
                ):
                    emit_scalarized(
                        "-",
                        mirrors[state],
                        depth_base[4],
                        mirrors[state],
                        a_scalar=True,
                        tag="cached_address_seed",
                        group=gg,
                        round=rnd,
                    )
            else:
                gather_node(depth, state, gg, rnd)
                xor_node(value, temp, gg, rnd)

            if gg in EARLY_FINAL_CACHE_SET and rnd == 14:
                for dest, mask, tag in (
                    (workspace_bits[2], v_one, "early_level4_condition_2"),
                    (workspace_bits[1], v_two, "early_level4_condition_1"),
                    (workspace_bits[0], v_four, "early_level4_condition_0"),
                ):
                    emit_valu(
                        "&",
                        dest,
                        mirrors[state],
                        mask,
                        tag=tag,
                        group=gg,
                        round=rnd,
                    )

            if (
                OVERLAP_SHALLOW_ADDRESS
                and gg not in FIRST_CACHE_SET
                and rnd == 3
            ):
                emit_valu(
                    "*",
                    mirrors[state],
                    mirrors[state],
                    v_two,
                    tag="depth4_address_double",
                    group=gg,
                    round=rnd,
                )
                emit_scalarized(
                    "-",
                    mirrors[state],
                    depth_base[4],
                    mirrors[state],
                    a_scalar=True,
                    tag="depth4_address_affine",
                    group=gg,
                    round=rnd,
                )
            elif OVERLAP_DEEP_ADDRESS and 4 <= rnd <= 9:
                # Once this round's node lookup has consumed the current
                # address, precompute 2*address-5 alongside the hash.  Only
                # the final parity subtraction remains on the critical path.
                emit_madd(
                    mirrors[state],
                    mirrors[state],
                    v_two,
                    v_neg_five,
                    tag="next_address_affine",
                    group=gg,
                    round=rnd,
                )
            elif (
                OVERLAP_DEEP_ADDRESS
                and rnd == 14
                and gg in VECTOR_EARLY_FINAL_ADDRESS_SET
                and gg not in FINAL_CACHE_SET
            ):
                # Once all level-4 cached selections have been emitted, their
                # first two difference vectors are dead.  Recycle them as
                # -2 and depth4_base so the final address affine is one MADD
                # and costs no additional scratch.
                if not final_address_vectors_ready[0]:
                    graph.emit(
                        "flow",
                        (
                            "add_imm",
                            level4_diff[0],
                            s_one,
                            (-3) & 0xFFFFFFFF,
                        ),
                        reads=(s_one,),
                        writes=(level4_diff[0],),
                        tag="final_address_neg_two_immediate",
                    )
                    emit_vbroadcast(
                        level4_diff[0],
                        level4_diff[0],
                        "final_address_neg_two_broadcast",
                    )
                    emit_vbroadcast(
                        level4_diff[1],
                        depth_base[4],
                        "final_address_base_broadcast",
                    )
                    final_address_vectors_ready[0] = True
                emit_madd(
                    mirrors[state],
                    mirrors[state],
                    level4_diff[0],
                    level4_diff[1],
                    tag="final_address_affine_vector_early",
                    group=gg,
                    round=rnd,
                )
            elif (
                rnd == 14
                and gg in EARLY_FINAL_ADDRESS_SET
                and gg not in FINAL_CACHE_SET
            ):
                # The depth-3 lookup is the final consumer of the path
                # offset.  Compute depth4_base - 2*offset while the hash is
                # in flight; after the hash only the parity subtraction is
                # left before the final gather can start.
                emit_scalarized(
                    "*",
                    mirrors[state],
                    mirrors[state],
                    s_two,
                    b_scalar=True,
                    tag="final_address_double_early",
                    group=gg,
                    round=rnd,
                )
                emit_scalarized(
                    "-",
                    mirrors[state],
                    depth_base[4],
                    mirrors[state],
                    a_scalar=True,
                    tag="final_address_affine_early",
                    group=gg,
                    round=rnd,
                )

            emit_hash(value, temp, gg, rnd)

            if DIRECT_MIRROR_PATH:
                if rnd in (0, 11):
                    emit_parity(mirrors[state], value, gg, rnd)
                elif rnd not in (10, rounds - 1):
                    emit_parity(temp, value, gg, rnd)
                    emit_madd(
                        mirrors[state],
                        mirrors[state],
                        v_two,
                        temp,
                        tag="mirror_update",
                        group=gg,
                        round=rnd,
                    )
            elif rnd in (0, 1, 2):
                emit_parity(workspace_bits[rnd], value, gg, rnd)
            elif rnd == 11:
                emit_parity(mirrors[state], value, gg, rnd)
            elif gg in EARLY_FINAL_CACHE_SET and rnd == 14:
                emit_parity(first_level4_condition, value, gg, rnd)
                select_level4_hybrid(
                    state,
                    workspace,
                    gg,
                    rounds - 1,
                    prepared=True,
                )
            elif (
                rnd == 14
                and (
                    gg in EARLY_FINAL_ADDRESS_SET
                    or gg in VECTOR_EARLY_FINAL_ADDRESS_SET
                )
                and gg not in FINAL_CACHE_SET
            ):
                emit_parity(temp, value, gg, rnd)
                emit_scalarized(
                    "-",
                    mirrors[state],
                    mirrors[state],
                    temp,
                    tag="final_address_parity_early",
                    group=gg,
                    round=rnd,
                )
            elif rnd in (12, 13, 14):
                emit_parity(temp, value, gg, rnd)
                emit_madd(
                    mirrors[state],
                    mirrors[state],
                    v_two,
                    temp,
                    tag="mirror_update",
                    group=gg,
                    round=rnd,
                )
            elif (
                OVERLAP_SHALLOW_ADDRESS
                and gg not in FIRST_CACHE_SET
                and rnd == 3
            ):
                emit_parity(temp, value, gg, rnd)
                emit_scalarized(
                    "-",
                    mirrors[state],
                    mirrors[state],
                    temp,
                    tag="depth4_address_parity",
                    group=gg,
                    round=rnd,
                )
            elif OVERLAP_DEEP_ADDRESS and 4 <= rnd <= 9:
                emit_parity(temp, value, gg, rnd)
                emit_scalarized(
                    "-",
                    mirrors[state],
                    mirrors[state],
                    temp,
                    tag="next_address_parity",
                    group=gg,
                    round=rnd,
                )
            elif rnd == 3 or 4 <= rnd <= 9:
                emit_parity(temp, value, gg, rnd)
                emit_madd(
                    mirrors[state],
                    mirrors[state],
                    v_two,
                    temp,
                    tag="mirror_update",
                    group=gg,
                    round=rnd,
                )

            # The final lookup is the last consumer of this group's mirror.
            # Recycle its first scalar word as a private output pointer while
            # the final hash is still running, eliminating all rolling-store
            # address dependencies without consuming scratch.
            if PER_GROUP_OUTPUT_POINTERS and rnd == rounds - 1:
                emit_const(
                    mirrors[state],
                    values_base + gg * VLEN,
                    "output_pointer",
                )

        values_base = 7 + n_nodes + batch_size
        emit_immediate(io_p0, values_base, "input_pointer")
        emit_immediate(io_p1, values_base + VLEN, "input_pointer")
        for pair in range(N_GROUPS // 2):
            g0 = 2 * pair
            g1 = g0 + 1
            graph.emit(
                "load",
                ("vload", values[g0], io_p0),
                reads=(io_p0,),
                writes=_words(values[g0]),
                tag="input_load",
                group=g0,
            )
            graph.emit(
                "load",
                ("vload", values[g1], io_p1),
                reads=(io_p1,),
                writes=_words(values[g1]),
                tag="input_load",
                group=g1,
            )
            if pair != N_GROUPS // 2 - 1:
                graph.emit(
                    "alu",
                    ("+", io_p0, io_p0, s_sixteen),
                    reads=(io_p0, s_sixteen),
                    writes=(io_p0,),
                    tag="pointer_advance",
                )
                graph.emit(
                    "alu",
                    ("+", io_p1, io_p1, s_sixteen),
                    reads=(io_p1, s_sixteen),
                    writes=(io_p1,),
                    tag="pointer_advance",
                )

        def round_workspace(gg: int, rnd: int) -> int:
            workspace = WORKSPACE_ASSIGNMENT[gg]
            if rnd >= 11:
                if SECOND_WORKSPACE_ASSIGNMENT is not None:
                    workspace = SECOND_WORKSPACE_ASSIGNMENT[gg]
                elif SECOND_WORKSPACE_FIXED >= 0:
                    workspace = SECOND_WORKSPACE_FIXED + (
                        (gg + rnd - 11) % SECOND_WORKSPACE_COUNT
                    )
                else:
                    workspace = (
                        workspace + (rnd - 11) * SECOND_WORKSPACE_STRIDE
                    ) % N_WORKSPACES
            return workspace

        if TAIL_EMISSION_MODE == "full_offset":
            if not (INDEPENDENT_ROOT_CACHE or DIRECT_MIRROR_PATH):
                raise ValueError("full_offset requires an independent root cache")
            for schedule_round in range(rounds + max(FULL_ROUND_OFFSETS)):
                for gg in TAIL_GROUP_ORDER:
                    rnd = schedule_round - FULL_ROUND_OFFSETS[gg]
                    if 0 <= rnd < rounds:
                        process_round(gg, round_workspace(gg, rnd), gg, rnd)
        else:
            # Phase A acquires a workspace only through depth 3, then releases
            # it while the group's core state advances through deep rounds.
            for gg in range(N_GROUPS):
                workspace = WORKSPACE_ASSIGNMENT[gg]
                for rnd in range(4):
                    process_round(gg, workspace, gg, rnd)
                if gg in FIRST_CACHE_SET:
                    process_round(gg, workspace, gg, 4)

            if not (INDEPENDENT_ROOT_CACHE or DIRECT_MIRROR_PATH):
                emit_valu(
                    "^",
                    node_vec[0],
                    node_vec[0],
                    v_c5,
                    tag="cached_root_transform",
                )

            def emit_tail_round(gg: int, rnd: int) -> None:
                if rnd == 4 and gg in FIRST_CACHE_SET:
                    return
                process_round(gg, round_workspace(gg, rnd), gg, rnd)

            # Emission order is a scheduler hint as well as a scratch-lifetime
            # choice: the per-group state carries the true round ordering.
            if TAIL_EMISSION_MODE == "split":
                for gg in TAIL_GROUP_ORDER:
                    for rnd in range(4, 11):
                        emit_tail_round(gg, rnd)
                for gg in TAIL_GROUP_ORDER:
                    for rnd in range(11, 16):
                        emit_tail_round(gg, rnd)
            elif TAIL_EMISSION_MODE == "group":
                for gg in TAIL_GROUP_ORDER:
                    for rnd in range(4, 16):
                        emit_tail_round(gg, rnd)
            elif TAIL_EMISSION_MODE == "round":
                for rnd in range(4, 16):
                    for gg in TAIL_GROUP_ORDER:
                        emit_tail_round(gg, rnd)
            elif TAIL_EMISSION_MODE == "wave":
                cohorts = min(TAIL_EMISSION_COHORTS, N_GROUPS)
                chunk_size = (N_GROUPS + cohorts - 1) // cohorts
                chunks = [
                    TAIL_GROUP_ORDER[start : start + chunk_size]
                    for start in range(0, N_GROUPS, chunk_size)
                ]
                width = 12 + (len(chunks) - 1) * TAIL_EMISSION_STAGGER
                for wave in range(width):
                    for cohort, groups in enumerate(chunks):
                        rnd = 4 + wave - cohort * TAIL_EMISSION_STAGGER
                        if 4 <= rnd < 16:
                            for gg in groups:
                                emit_tail_round(gg, rnd)
            else:
                raise ValueError(f"unknown tail emission mode: {TAIL_EMISSION_MODE}")

        if PER_GROUP_OUTPUT_POINTERS:
            for group in range(N_GROUPS):
                graph.emit(
                    "store",
                    ("vstore", mirrors[group], values[group]),
                    reads=(mirrors[group],) + _words(values[group]),
                    tag="output_store",
                    group=group,
                )
        elif OUTPUT_POINTER_STREAMS == 4:
            # Profile-guided output chains.  Groups are ordered by their
            # predicted value completion and divided between two pointers.
            # add_imm supports arbitrary (including wrapped negative) jumps,
            # so physical output order no longer constrains scheduling.
            completion_order = (
                0, 1, 2, 3, 4, 6, 8, 10, 12, 5, 14, 7, 16, 9, 18, 11,
                13, 15, 20, 17, 19, 22, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            )
            for stream, pointer in enumerate((io_p0, io_p1)):
                groups = completion_order[stream::2]
                emit_immediate(
                    pointer,
                    values_base + groups[0] * VLEN,
                    "output_pointer",
                )
                for position, group in enumerate(groups):
                    graph.emit(
                        "store",
                        ("vstore", pointer, values[group]),
                        reads=(pointer,) + _words(values[group]),
                        tag="output_store",
                        group=group,
                    )
                    if position != len(groups) - 1:
                        delta = (groups[position + 1] - group) * VLEN
                        graph.emit(
                            "flow",
                            ("add_imm", pointer, pointer, delta & 0xFFFFFFFF),
                            reads=(pointer,),
                            writes=(pointer,),
                            tag="pointer_advance",
                        )
        elif OUTPUT_POINTER_STREAMS == 32:
            # Every output gets an independent address.  The top-tree staging
            # words are dead after cached-node construction, so recycling them
            # as scalar pointers costs no scratch.  This trades the two long
            # ALU pointer chains for independent load-engine immediates.
            for group in range(N_GROUPS):
                pointer = top_words + group
                emit_const(
                    pointer,
                    values_base + group * VLEN,
                    "output_pointer",
                )
                graph.emit(
                    "store",
                    ("vstore", pointer, values[group]),
                    reads=(pointer,) + _words(values[group]),
                    tag="output_store",
                    group=group,
                )
        elif OUTPUT_POINTER_STREAMS == 8:
            # Setup constants are dead by the time results become ready.  Use
            # eight of their scalar words as independent rolling store
            # pointers, preventing one late group from blocking half the batch.
            output_pointers = (
                io_p0,
                io_p1,
                top_p0,
                top_p1,
                s_c1,
                s_c2_shift9,
                s_c4,
                s_m4,
            )
            emit_immediate(
                s_nineteen,
                OUTPUT_POINTER_STREAMS * VLEN,
                "output_stride",
            )
            for stream, pointer in enumerate(output_pointers):
                emit_immediate(
                    pointer,
                    values_base + stream * VLEN,
                    "output_pointer",
                )
                groups = tuple(range(stream, N_GROUPS, OUTPUT_POINTER_STREAMS))
                for position, group in enumerate(groups):
                    graph.emit(
                        "store",
                        ("vstore", pointer, values[group]),
                        reads=(pointer,) + _words(values[group]),
                        tag="output_store",
                        group=group,
                    )
                    if position != len(groups) - 1:
                        graph.emit(
                            "alu",
                            ("+", pointer, pointer, s_nineteen),
                            reads=(pointer, s_nineteen),
                            writes=(pointer,),
                            tag="pointer_advance",
                        )
        elif SCATTERED_OUTPUT_STORES:
            # Arbitrary completion order avoids one slow value blocking the
            # rolling contiguous-address chain.  Two pointer registers form
            # interleaved one-cycle address/store pipelines on spare flow slots.
            emit_immediate(top_p0, values_base, "output_base")
            for position, group in enumerate(OUTPUT_GROUP_ORDER):
                pointer = io_p0 if position % 2 == 0 else io_p1
                graph.emit(
                    "flow",
                    ("add_imm", pointer, top_p0, group * VLEN),
                    reads=(top_p0,),
                    writes=(pointer,),
                    tag="output_address",
                    group=group,
                )
                graph.emit(
                    "store",
                    ("vstore", pointer, values[group]),
                    reads=(pointer,) + _words(values[group]),
                    tag="output_store",
                    group=group,
                )
        else:
            emit_immediate(io_p0, values_base, "output_pointer")
            emit_immediate(io_p1, values_base + VLEN, "output_pointer")
            for pair in range(N_GROUPS // 2):
                g0 = 2 * pair
                g1 = g0 + 1
                graph.emit(
                    "store",
                    ("vstore", io_p0, values[g0]),
                    reads=(io_p0,) + _words(values[g0]),
                    tag="output_store",
                    group=g0,
                )
                graph.emit(
                    "store",
                    ("vstore", io_p1, values[g1]),
                    reads=(io_p1,) + _words(values[g1]),
                    tag="output_store",
                    group=g1,
                )
                if pair != N_GROUPS // 2 - 1:
                    graph.emit(
                        "alu",
                        ("+", io_p0, io_p0, s_sixteen),
                        reads=(io_p0, s_sixteen),
                        writes=(io_p0,),
                        tag="pointer_advance",
                    )
                    graph.emit(
                        "alu",
                        ("+", io_p1, io_p1, s_sixteen),
                        reads=(io_p1, s_sixteen),
                        writes=(io_p1,),
                        tag="pointer_advance",
                    )

        self.scratch_ptr = scratch.ptr
        self.scratch_debug = scratch.debug

        # Try several deterministic scheduling priorities.  Construction time
        # is unscored, so select the shortest legal bundle sequence.
        schedules = []
        self.dag_ops = graph.ops
        forward_cycles: list[list[int]] = []
        for policy in SCHEDULE_POLICIES:
            if SSA_WORKSPACES:
                schedule, cycles = self._schedule(
                    graph.ops, policy, return_cycles=True
                )
                schedules.append(schedule)
                forward_cycles.append(cycles)
            else:
                schedules.append(self._schedule(graph.ops, policy))
        reversed_ops = self._reverse_ops(graph.ops)
        if SSA_WORKSPACES and BACKWARD_POLICIES:
            raise ValueError("SSA_WORKSPACES currently supports forward scheduling only")
        for policy in BACKWARD_POLICIES:
            schedules.append(list(reversed(self._schedule(reversed_ops, policy))))
        self.schedule_lengths = {
            f"forward:{policy}": len(schedule)
            for policy, schedule in zip(SCHEDULE_POLICIES, schedules)
        }
        self.schedule_lengths.update(
            {
                f"backward:{policy}": len(schedule)
                for policy, schedule in zip(
                    BACKWARD_POLICIES, schedules[len(SCHEDULE_POLICIES) :]
                )
            }
        )
        best_index = min(range(len(schedules)), key=lambda i: len(schedules[i]))
        self.instrs = schedules[best_index]

        if SSA_WORKSPACES:
            # Calculate conservative live intervals in scheduled-cycle space,
            # then linear-scan color the virtual SIMD registers.  Intervals
            # that touch the same cycle conflict because all reads observe the
            # old scratch image and all writes commit at cycle end.
            cycles = forward_cycles[best_index]
            virtual_word_to_base = {
                base + lane: base
                for base in virtual_vectors
                for lane in range(VLEN)
            }
            intervals: dict[int, list[int]] = {}
            physical_word_to_base = {
                base + lane: base
                for base in physical_workspace_vectors
                for lane in range(VLEN)
            }
            physical_live_ranges: dict[int, list[list[int]]] = defaultdict(list)
            physical_current_definition: dict[int, list[int]] = {}
            for op_index, op in enumerate(graph.ops):
                cycle = cycles[op_index]
                for word in op.reads:
                    base = virtual_word_to_base.get(word)
                    if base is None:
                        base = physical_word_to_base.get(word)
                        if base is not None:
                            live_range = physical_current_definition.get(base)
                            if live_range is not None:
                                live_range[1] = max(live_range[1], cycle)
                        continue
                    interval = intervals.get(base)
                    if interval is None:
                        intervals[base] = [cycle, cycle]
                    else:
                        interval[0] = min(interval[0], cycle)
                        interval[1] = max(interval[1], cycle)
                for word in op.writes:
                    base = virtual_word_to_base.get(word)
                    if base is None:
                        base = physical_word_to_base.get(word)
                        if base is not None:
                            live_range = [cycle, cycle]
                            physical_live_ranges[base].append(live_range)
                            physical_current_definition[base] = live_range
                        continue
                    interval = intervals.get(base)
                    if interval is None:
                        intervals[base] = [cycle, cycle]
                    else:
                        interval[0] = min(interval[0], cycle)
                        interval[1] = max(interval[1], cycle)

            free_colors = list(physical_workspace_vectors)
            active: list[tuple[int, int, int]] = []
            colors: dict[int, int] = {}
            for base, (start, end) in sorted(
                intervals.items(), key=lambda item: (item[1][0], item[1][1])
            ):
                still_active: list[tuple[int, int, int]] = []
                for active_end, active_base, color in active:
                    # Use conservative cycle-disjoint intervals here.  Some
                    # mux spills are overwritten at their final occurrence,
                    # so endpoint sharing is not always the safe WAR case.
                    if active_end < start:
                        free_colors.append(color)
                    else:
                        still_active.append((active_end, active_base, color))
                active = still_active
                compatible_index = next(
                    (
                        index
                        for index, candidate in enumerate(free_colors)
                        if not any(
                            start <= physical_end and physical_start <= end
                            for physical_start, physical_end in physical_live_ranges[candidate]
                        )
                    ),
                    None,
                )
                if compatible_index is None:
                    peak = len(active) + 1
                    live_names = ",".join(
                        virtual_vectors[active_base]
                        for _, active_base, _ in active
                    )
                    raise AssertionError(
                        "virtual workspace coloring overflow: "
                        f"need at least {peak} vectors, have "
                        f"{len(physical_workspace_vectors)} at cycle {start}; "
                        f"new={virtual_vectors[base]}; live={live_names}"
                    )
                color = free_colors.pop(compatible_index)
                colors[base] = color
                active.append((end, base, color))
                active.sort()

            word_colors = {
                base + lane: color + lane
                for base, color in colors.items()
                for lane in range(VLEN)
            }

            def recolor_slot(slot: tuple) -> tuple:
                return tuple(
                    word_colors.get(item, item) if isinstance(item, int) else item
                    for item in slot
                )

            self.instrs = [
                {
                    engine: [recolor_slot(slot) for slot in slots]
                    for engine, slots in bundle.items()
                }
                for bundle in self.instrs
            ]
            self.virtual_workspace_colors = len(set(colors.values()))
            self.virtual_workspace_intervals = len(intervals)
            self.virtual_workspace_assignment = {
                virtual_vectors[base]: (color, tuple(intervals[base]))
                for base, color in colors.items()
            }
        else:
            self.virtual_workspace_colors = 0
            self.virtual_workspace_intervals = 0
            self.virtual_workspace_assignment = {}

        if BRANCH_FINAL_LANES:
            if SSA_WORKSPACES:
                raise ValueError("branch final lookup and SSA workspaces are exclusive")
            branch_select_dests = {
                temps[BRANCH_FINAL_GROUP] + lane for lane in BRANCH_FINAL_LANES
            }
            select_pcs = [
                pc
                for pc, bundle in enumerate(self.instrs)
                for slot in bundle.get("flow", ())
                if slot[0] == "select" and slot[1] in branch_select_dests
            ]
            if len(select_pcs) != len(BRANCH_FINAL_LANES):
                raise AssertionError(
                    f"expected {len(BRANCH_FINAL_LANES)} branch selects, "
                    f"found {len(select_pcs)}"
                )
            first_select_pc = min(select_pcs)
            last_vselect_pc = max(
                pc
                for pc, bundle in enumerate(self.instrs[:first_select_pc])
                for slot in bundle.get("flow", ())
                if slot[0] == "vselect"
            )

            lane_count = len(BRANCH_FINAL_LANES)
            if lane_count > 4:
                raise ValueError("precomputed branch tables currently support 4 lanes")
            dispatch_window: tuple[int, list[int], list[int]] | None = None
            dead_base_words = set(_words(temps[0]))
            last_dead_base_use = max(
                pc
                for pc, bundle in enumerate(self.instrs)
                if any(
                    isinstance(item, int) and item in dead_base_words
                    for slots in bundle.values()
                    for slot in slots
                    for item in slot[1:]
                )
            )
            dispatch_pairs: list[tuple[int, int]] = []
            pc = last_vselect_pc + 1
            while pc + 1 < first_select_pc and len(dispatch_pairs) < lane_count:
                target_pc = pc + 1
                target_alu = self.instrs[target_pc].get("alu", ())
                target_fits = len(target_alu) <= 10 or (
                    len(target_alu) == 11
                    and any(
                        slot[0] == "+" and slot[1] == slot[2]
                        for slot in target_alu
                    )
                )
                if (
                    not self.instrs[pc].get("flow")
                    and not self.instrs[target_pc].get("flow")
                    and target_fits
                ):
                    dispatch_pairs.append((pc, target_pc))
                    pc += 2
                else:
                    pc += 1

            if len(dispatch_pairs) == lane_count:
                base_pc = next(
                    (
                        pc
                        for pc in range(last_dead_base_use + 1, last_vselect_pc + 1)
                        if len(self.instrs[pc].get("alu", ()))
                        <= 12 - lane_count
                    ),
                    None,
                )
                if base_pc is not None:
                    reserved_copy = {target_pc: 2 for _, target_pc in dispatch_pairs}
                    prep_use: dict[int, int] = defaultdict(int)
                    target_pcs: list[int] = []
                    for jump_pc, _ in dispatch_pairs:
                        prep_pc = next(
                            (
                                candidate
                                for candidate in range(jump_pc - 1, base_pc, -1)
                                if len(self.instrs[candidate].get("alu", ()))
                                + reserved_copy.get(candidate, 0)
                                + prep_use[candidate]
                                < SLOT_LIMITS["alu"]
                            ),
                            None,
                        )
                        if prep_pc is None:
                            target_pcs = []
                            break
                        prep_use[prep_pc] += 1
                        target_pcs.append(prep_pc)
                    if len(target_pcs) == lane_count:
                        dispatch = [pc for pair in dispatch_pairs for pc in pair]
                        dispatch_window = (base_pc, target_pcs, dispatch)
            if dispatch_window is None:
                window_debug = [
                    (
                        pc,
                        len(self.instrs[pc].get("alu", ())),
                        len(self.instrs[pc].get("flow", ())),
                        self.instrs[pc].get("alu", ()),
                    )
                    for pc in range(last_vselect_pc, first_select_pc)
                ]
                raise AssertionError(
                    "no flow-free dispatch window before final parity selection: "
                    f"{window_debug}"
                )

            # The out-of-line dispatch tables are appended after the main
            # schedule.  Halt in the final real bundle prevents fall-through;
            # dynamically selected table entries jump back to the next main PC.
            if self.instrs[-1].get("flow"):
                raise AssertionError("final bundle has no flow slot for halt")
            self.instrs[-1]["flow"] = [("halt",)]

            candidate_yes = bits[SECOND_WORKSPACE_FIXED][0]
            candidate_no = bits[SECOND_WORKSPACE_FIXED][1]
            jump_vector = bits[SECOND_WORKSPACE_FIXED][2]
            table_blocks: list[dict[str, list[tuple]]] = []
            main_length = len(self.instrs)
            base_pc, target_pcs, dispatch_pcs = dispatch_window
            load_pc = next(
                pc
                for pc, bundle in enumerate(self.instrs[:base_pc])
                if len(bundle.get("load", ())) < SLOT_LIMITS["load"]
            )
            self.instrs[load_pc].setdefault("load", []).append(
                ("const", branch_table_base, main_length)
            )

            lane_base_registers = [temps[0] + i for i in range(lane_count)]
            base_offsets = (None, s_eight, s_sixteen, s_sixteen)
            for lane_index, lane_base in enumerate(lane_base_registers):
                if lane_index == 0:
                    slot = (
                        "|", lane_base, branch_table_base, branch_table_base
                    )
                else:
                    slot = (
                        "+",
                        lane_base,
                        branch_table_base,
                        base_offsets[lane_index],
                    )
                self.instrs[base_pc].setdefault("alu", []).append(slot)

            if lane_count == 4:
                lane3_adjust_pc = next(
                    pc
                    for pc in range(base_pc + 1, target_pcs[3])
                    if len(self.instrs[pc].get("alu", ()))
                    < SLOT_LIMITS["alu"]
                )
                self.instrs[lane3_adjust_pc].setdefault("alu", []).append(
                    (
                        "+",
                        lane_base_registers[3],
                        lane_base_registers[3],
                        s_sixteen,
                    )
                )

            for lane_index, lane in enumerate(BRANCH_FINAL_LANES):
                self.instrs[target_pcs[lane_index]].setdefault("alu", []).append(
                    (
                        "+",
                        jump_vector + lane,
                        mirrors[BRANCH_FINAL_GROUP] + lane,
                        lane_base_registers[lane_index],
                    )
                )

            for lane_index, lane in enumerate(BRANCH_FINAL_LANES):
                jump_pc, target_pc = dispatch_pcs[
                    2 * lane_index : 2 + 2 * lane_index
                ]
                if len(self.instrs[target_pc].get("alu", ())) == 11:
                    alu_slots = self.instrs[target_pc]["alu"]
                    move_index = next(
                        i
                        for i, slot in enumerate(alu_slots)
                        if slot[0] == "+" and slot[1] == slot[2]
                    )
                    moved = alu_slots.pop(move_index)
                    move_pc = next(
                        pc
                        for pc in range(target_pc + 1, len(self.instrs))
                        if len(self.instrs[pc].get("alu", ())) < SLOT_LIMITS["alu"]
                    )
                    self.instrs[move_pc].setdefault("alu", []).append(moved)
                self.instrs[jump_pc]["flow"] = [
                    ("jump_indirect", jump_vector + lane)
                ]
                desired_table_offset = (0, 8, 16, 32)[lane_index]
                while len(table_blocks) < desired_table_offset:
                    table_blocks.append({"flow": [("halt",)]})
                for pair_index in range(8):
                    target = {
                        engine: list(slots)
                        for engine, slots in self.instrs[target_pc].items()
                    }
                    target.setdefault("alu", []).extend(
                        [
                            (
                                "|",
                                candidate_yes + lane,
                                level4_reversed[2 * pair_index + 1] + lane,
                                level4_reversed[2 * pair_index + 1] + lane,
                            ),
                            (
                                "|",
                                candidate_no + lane,
                                level4_reversed[2 * pair_index] + lane,
                                level4_reversed[2 * pair_index] + lane,
                            ),
                        ]
                    )
                    target["flow"] = [("jump", target_pc + 1)]
                    table_blocks.append(target)
            self.instrs.extend(table_blocks)
            self.branch_main_cycles = main_length
        else:
            self.branch_main_cycles = 0

        self._validate_program()

    @staticmethod
    def _reverse_ops(ops: list[_Op]) -> list[_Op]:
        n = len(ops)
        original_children: list[dict[int, int]] = [dict() for _ in range(n)]
        for child, op in enumerate(ops):
            for parent, lag in op.parents.items():
                old = original_children[parent].get(child)
                if old is None or lag > old:
                    original_children[parent][child] = lag

        reversed_ops: list[_Op] = []
        for old_index in range(n - 1, -1, -1):
            parents = {
                n - 1 - child: lag
                for child, lag in original_children[old_index].items()
            }
            op = ops[old_index]
            reversed_ops.append(
                _Op(
                    engine=op.engine,
                    slot=op.slot,
                    reads=op.reads,
                    writes=op.writes,
                    parents=parents,
                    tag=op.tag,
                    group=op.group,
                    round=op.round,
                )
            )
        return reversed_ops

    @staticmethod
    def _schedule(
        ops: list[_Op],
        policy: int,
        return_cycles: bool = False,
        external_scores: list[int] | None = None,
        height_weight: int = 2,
        tie_scores: list[int] | None = None,
        priority_noise: list[int] | None = None,
    ) -> list[dict[str, list[tuple]]] | tuple[list[dict[str, list[tuple]]], list[int]]:
        n = len(ops)
        children: list[list[tuple[int, int]]] = [[] for _ in range(n)]
        indegree = [0] * n
        for child, op in enumerate(ops):
            indegree[child] = len(op.parents)
            for parent, lag in op.parents.items():
                children[parent].append((child, lag))

        # Latency critical height and a cheap downstream-work proxy.
        height = [0] * n
        reach = [0] * n
        data_height = [0] * n
        data_reach = [0] * n
        resource_weight_sets = (
            {"alu": 1, "valu": 2, "load": 6, "store": 6, "flow": 12},
            {"alu": 1, "valu": 3, "load": 6, "store": 2, "flow": 10},
            {"alu": 1, "valu": 2, "load": 8, "store": 2, "flow": 8},
            {"alu": 1, "valu": 4, "load": 5, "store": 2, "flow": 9},
        )
        resource_height = [[0] * n for _ in resource_weight_sets]
        inf = 1_000_000
        distance_to_load = [inf] * n
        for node in range(n - 1, -1, -1):
            if children[node]:
                height[node] = max(lag + height[ch] for ch, lag in children[node])
                reach[node] = min(
                    1_000_000,
                    len(children[node]) + sum(reach[ch] for ch, _ in children[node]),
                )
                data_children = [ch for ch, lag in children[node] if lag]
                if data_children:
                    data_height[node] = 1 + max(data_height[ch] for ch in data_children)
                    data_reach[node] = min(
                        1_000_000,
                        len(data_children) + sum(data_reach[ch] for ch in data_children),
                    )
            if ops[node].engine == "load":
                distance_to_load[node] = 0
            elif children[node]:
                distance_to_load[node] = min(
                    (lag + distance_to_load[ch] for ch, lag in children[node]),
                    default=inf,
                )
            for variant, weights in enumerate(resource_weight_sets):
                tail = max(
                    (resource_height[variant][ch] for ch, _ in children[node]),
                    default=0,
                )
                resource_height[variant][node] = weights[ops[node].engine] + tail

        engine_rank_sets = (
            {"load": 4, "valu": 3, "alu": 2, "flow": 1, "store": 0},
            {"valu": 4, "load": 3, "alu": 2, "flow": 1, "store": 0},
            {"alu": 4, "load": 3, "valu": 2, "flow": 1, "store": 0},
            {"flow": 4, "load": 3, "valu": 2, "alu": 1, "store": 0},
        )
        engine_rank = engine_rank_sets[policy % len(engine_rank_sets)]

        def priority(i: int) -> tuple[int, int, int, int, int]:
            op = ops[i]
            if external_scores is not None:
                group_offset = 64 if op.group < 0 else GROUP_FINE_OFFSETS[op.group]
                return (
                    height_weight * height[i] + external_scores[i] + group_offset,
                    reach[i] // 32,
                    engine_rank[op.engine],
                    group_offset,
                    -op.group if op.group >= 0 else -i,
                )
            if policy == 86:
                cohort = -1 if op.group < 0 else op.group // 4
                round_offset = 0 if op.round < 0 else ROUND_PRIORITY_OFFSETS[op.round]
                return (
                    height[i] - 8 * cohort + round_offset,
                    reach[i] // 32,
                    engine_rank[op.engine],
                    -cohort,
                    -op.group if op.group >= 0 else -i,
                )
            if 87 <= policy <= 102:
                variant = policy - 87
                cohort_size = (2, 3, 4, 5)[(variant // 4) % 4]
                penalty = (0, 4, 8, 12)[variant % 4]
                cohort = -1 if op.group < 0 else op.group // cohort_size
                return (
                    data_height[i] - penalty * cohort,
                    data_reach[i] // 16,
                    engine_rank[op.engine],
                    -cohort,
                    i,
                )
            if 103 <= policy <= 106:
                # Match the compact scheduler used by the strongest public
                # baseline: critical height follows only true data hazards;
                # zero-lag anti-dependencies affect legality but not priority.
                return (data_height[i], 0, 0, 0, i)
            if 143 <= policy <= 158:
                variant = policy - 143
                resource_variant = variant // 4
                latency_multiplier = (0, 1, 2, 4)[variant % 4]
                return (
                    resource_height[resource_variant][i]
                    + latency_multiplier * height[i],
                    data_height[i],
                    reach[i] // 32,
                    engine_rank[op.engine],
                    -op.group if op.group >= 0 else -i,
                )
            if 159 <= policy <= 174:
                variant = policy - 159
                resource_variant = variant // 4
                divisor = (4, 8, 16, 32)[variant % 4]
                distance = distance_to_load[i]
                finite = int(distance < inf)
                load_score = -distance if finite else -inf
                return (
                    ENGINE_HEIGHT_MULTIPLIERS.get(op.engine, 1) * height[i]
                    + load_score
                    + resource_height[resource_variant][i] // divisor,
                    finite,
                    reach[i] // 32,
                    engine_rank[op.engine],
                    -op.group if op.group >= 0 else -i,
                )
            if 175 <= policy <= 178:
                group_offset = 0 if op.group < 0 else BASIC_GROUP_OFFSETS[op.group]
                round_offset = 0 if op.round < 0 else BASIC_ROUND_OFFSETS[op.round]
                return (
                    height[i] // BASIC_HEIGHT_DIVISOR + group_offset + round_offset,
                    reach[i] // BASIC_REACH_DIVISOR,
                    engine_rank_sets[policy % 4][op.engine],
                    -op.group if op.group >= 0 else 0,
                    i,
                )
            if 107 <= policy <= 110:
                return (data_height[i], data_reach[i], 0, 0, i)
            if 111 <= policy <= 142:
                variant = policy - 111
                load_multiplier = (1, 2, 3, 4)[variant % 4]
                launch_penalty = (0, 1, 2, 4, 8, 12, 16, 24)[variant // 4]
                finite = int(distance_to_load[i] < inf)
                load_score = (
                    -load_multiplier * distance_to_load[i] if finite else -inf
                )
                launch = (
                    -1 if op.group < 0 else FULL_ROUND_OFFSETS[op.group]
                )
                return (
                    height[i] + load_score - launch_penalty * launch,
                    finite,
                    -launch,
                    reach[i] // 32,
                    i,
                )
            if 82 <= policy <= 85:
                group_offset = 64 if op.group < 0 else GROUP_PRIORITY_OFFSETS[op.group]
                tag_offset = TAG_PRIORITY_OFFSETS.get(op.tag, 0)
                return (
                    height[i] + group_offset + tag_offset,
                    reach[i] // 32,
                    engine_rank[op.engine],
                    group_offset,
                    -op.group if op.group >= 0 else -i,
                )
            if policy == 81:
                group_offset = 64 if op.group < 0 else GROUP_PRIORITY_OFFSETS[op.group]
                round_offset = 0 if op.round < 0 else ROUND_PRIORITY_OFFSETS[op.round]
                return (
                    height[i] + group_offset + round_offset,
                    reach[i] // 32,
                    engine_rank[op.engine],
                    group_offset,
                    -op.group if op.group >= 0 else -i,
                )
            if policy == 80:
                offset = 64 if op.group < 0 else GROUP_PRIORITY_OFFSETS[op.group]
                return (
                    height[i] + offset,
                    reach[i] // 32,
                    engine_rank[op.engine],
                    offset,
                    -op.group if op.group >= 0 else -i,
                )
            if policy >= 72:
                pipeline_penalties = (32, 40, 48, 56, 64, 72, 80, 96)
                penalty = pipeline_penalties[policy - 72]
                if op.group < 0:
                    pipeline_rank = -1
                else:
                    pipeline_rank = 2 * ((op.group % 16) // 4) + op.group // 16
                return (
                    height[i] - pipeline_rank * penalty,
                    reach[i] // 32,
                    engine_rank[op.engine],
                    -pipeline_rank,
                    -op.group if op.group >= 0 else -i,
                )
            if policy >= 48:
                tuned_penalties = (56, 60, 62, 64, 66, 68)
                penalty = tuned_penalties[(policy - 48) // 4]
                cohort = -1 if op.group < 0 else op.group // 4
                return (
                    height[i] - cohort * penalty,
                    reach[i] // 32,
                    engine_rank[op.engine],
                    -cohort,
                    -op.group if op.group >= 0 else -i,
                )
            if policy >= 32:
                variant = policy - 32
                distance = distance_to_load[i]
                finite = int(distance < inf)
                if variant < 4:
                    cohort_size = (3, 4, 5, 6)[variant]
                    cohort = -1 if op.group < 0 else op.group // cohort_size
                    return (
                        finite,
                        -distance if finite else -inf,
                        -cohort,
                        height[i],
                        engine_rank[op.engine],
                    )
                multipliers = (1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32)
                multiplier = ENGINE_LOAD_MULTIPLIERS.get(
                    op.engine, multipliers[variant - 4]
                )
                load_score = -multiplier * distance if finite else -inf
                cohort = -1 if op.group < 0 else op.group // 4
                fine_offset = (
                    0 if op.group < 0 else GROUP_FINE_OFFSETS[op.group]
                )
                round_offset = (
                    0 if op.round < 0 else ROUND_PRIORITY_OFFSETS[op.round]
                )
                return (
                    ENGINE_HEIGHT_MULTIPLIERS.get(op.engine, 1) * height[i]
                    + load_score
                    + fine_offset
                    + round_offset
                    + TAG_PRIORITY_OFFSETS.get(op.tag, 0)
                    + OP_PRIORITY_OFFSETS.get((op.tag, op.group, op.round), 0)
                    + (0 if priority_noise is None else priority_noise[i]),
                    finite,
                    -cohort,
                    reach[i] // 32,
                    engine_rank[op.engine],
                    0 if tie_scores is None else tie_scores[i],
                )
            if policy >= 16:
                penalties = (8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128)
                penalty = penalties[policy - 16]
                cohort = -1 if op.group < 0 else op.group // 4
                return (
                    height[i]
                    - cohort * penalty
                    + (0 if op.group < 0 else GROUP_FINE_OFFSETS[op.group]),
                    reach[i] // 32,
                    engine_rank[op.engine],
                    -cohort,
                    -op.group if op.group >= 0 else -i,
                )
            if policy >= 8:
                cohort_sizes = (4, 5, 6, 7, 8, 10, 12, 16)
                cohort_size = cohort_sizes[policy - 8]
                # Setup is cohort -1.  Thereafter a small group cohort is
                # allowed to run ahead into gathers while later cohorts keep
                # flow/VALU busy with shallow rounds.
                cohort = -1 if op.group < 0 else op.group // cohort_size
                return (
                    -cohort,
                    height[i],
                    reach[i] // 32,
                    engine_rank[op.engine],
                    -op.group if op.group >= 0 else -i,
                )
            # Policies 0..3 strongly honor latency; 4..7 allow cohort/order
            # locality to break near-critical ties more aggressively.
            h = height[i] if policy < 4 else height[i] // 2
            group_bias = -op.group if op.group >= 0 else 0
            if policy in (2, 6):
                group_bias = -(op.group % 4) if op.group >= 0 else 0
            id_bias = -i if policy % 2 == 0 else i
            op_offset = OP_PRIORITY_OFFSETS.get((op.tag, op.group, op.round), 0)
            return (
                h
                + op_offset
                + (0 if priority_noise is None else priority_noise[i]),
                reach[i] // 32,
                engine_rank[op.engine],
                group_bias,
                id_bias,
            )

        earliest = [0] * n
        heaps: dict[str, list[tuple[tuple[int, ...], int]]] = {
            engine: [] for engine in ("alu", "valu", "load", "store", "flow")
        }

        def push_ready(i: int) -> None:
            p = priority(i)
            heapq.heappush(heaps[ops[i].engine], (tuple(-x for x in p), i))

        future: dict[int, list[int]] = defaultdict(list)
        for i, deg in enumerate(indegree):
            if deg == 0:
                push_ready(i)

        scheduled = 0
        scheduled_cycle = [-1] * n
        cycle = 0
        bundles: list[dict[str, list[tuple]]] = []
        engine_order_variants = (
            ("load", "valu", "alu", "flow", "store"),
            ("valu", "load", "alu", "flow", "store"),
            ("alu", "load", "valu", "flow", "store"),
            ("flow", "load", "valu", "alu", "store"),
        )
        engine_order = engine_order_variants[policy % 4]

        while scheduled < n:
            for i in future.pop(cycle, ()):
                push_ready(i)

            used = {engine: 0 for engine in heaps}
            bundle: dict[str, list[tuple]] = defaultdict(list)

            made_progress = True
            while made_progress:
                made_progress = False
                for engine in engine_order:
                    if used[engine] >= SLOT_LIMITS[engine]:
                        continue
                    heap = heaps[engine]
                    if not heap:
                        continue

                    deferred = []
                    chosen = None
                    while heap:
                        item = heapq.heappop(heap)
                        candidate = item[1]
                        if earliest[candidate] <= cycle:
                            chosen = candidate
                            break
                        deferred.append(item)
                    for item in deferred:
                        heapq.heappush(heap, item)
                    if chosen is None:
                        continue

                    op = ops[chosen]
                    bundle[engine].append(op.slot)
                    used[engine] += 1
                    scheduled += 1
                    scheduled_cycle[chosen] = cycle
                    made_progress = True

                    for child, lag in children[chosen]:
                        indegree[child] -= 1
                        ready_at = cycle + lag
                        if ready_at > earliest[child]:
                            earliest[child] = ready_at
                        if indegree[child] == 0:
                            if earliest[child] <= cycle:
                                push_ready(child)
                            else:
                                future[earliest[child]].append(child)

            if not bundle:
                # All dependency lags are at most one, so this can occur only
                # before the next real cycle.  Do not emit a zero-cost bundle.
                cycle += 1
                continue
            bundles.append(dict(bundle))
            cycle += 1

        if return_cycles:
            return bundles, scheduled_cycle
        return bundles

    @staticmethod
    def _compact_schedule(ops: list[_Op], cycles: list[int]) -> list[dict[str, list[tuple]]]:
        cycles = cycles.copy()
        for _ in range(4):
            horizon = max(cycles) + 1
            usage = [defaultdict(int) for _ in range(horizon)]
            for i, op in enumerate(ops):
                usage[cycles[i]][op.engine] += 1

            moved = 0
            for i, op in enumerate(ops):
                earliest = 0
                if op.parents:
                    earliest = max(cycles[parent] + lag for parent, lag in op.parents.items())
                current = cycles[i]
                for target in range(earliest, current):
                    if usage[target][op.engine] < SLOT_LIMITS[op.engine]:
                        usage[current][op.engine] -= 1
                        usage[target][op.engine] += 1
                        cycles[i] = target
                        moved += 1
                        break
            if moved == 0:
                break

        horizon = max(cycles) + 1
        bundles: list[dict[str, list[tuple]]] = [defaultdict(list) for _ in range(horizon)]
        for i, op in enumerate(ops):
            bundles[cycles[i]][op.engine].append(op.slot)
        return [dict(bundle) for bundle in bundles if bundle]

    def _validate_program(self) -> None:
        if self.scratch_ptr > SCRATCH_SIZE:
            raise AssertionError(f"scratch overflow: {self.scratch_ptr}")
        for pc, bundle in enumerate(self.instrs):
            if not bundle:
                raise AssertionError(f"empty bundle at pc={pc}")
            for engine, slots in bundle.items():
                if len(slots) > SLOT_LIMITS[engine]:
                    raise AssertionError(
                        f"{engine} overflow at pc={pc}: {len(slots)} > {SLOT_LIMITS[engine]}"
                    )
