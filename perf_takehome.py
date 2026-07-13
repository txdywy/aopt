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
FLOW_OUTPUT_POINTER_ADVANCE = False
FLOW_OUTPUT_ADVANCE_POSITIONS: frozenset[int] = frozenset()
OUTPUT_GROUP_ORDER = tuple(range(N_GROUPS))
PER_GROUP_OUTPUT_POINTERS = False
INDEPENDENT_TAIL_OUTPUTS = True
PRESERVED_TAIL_OUTPUT_POINTERS = False
PREPROCESS_MAX_DEPTH = 7
EARLY_FINAL_CACHE_SET: frozenset[int] = frozenset()
EARLY_FINAL_ADDRESS_SET: frozenset[int] = frozenset()
VECTOR_EARLY_FINAL_ADDRESS_SET: frozenset[int] = frozenset()
BRANCH_FINAL_GROUP = 31
BRANCH_FINAL_LANES: tuple[int, ...] = tuple(range(VLEN))
PAIRED_BRANCH_FINAL = True
PAIRED_EARLY_XOR = False
SAVED_SECOND_PATH_EXTRA_GROUPS: frozenset[int] = frozenset()
PIPELINED_DEPTH3_GROUPS: frozenset[int] = frozenset()
PIPELINED_DEPTH3_WORKSPACE_OVERRIDES: dict[int, int] = {}
PAIRED_TARGET_SENTINEL = 0xB2A00003
DIRECT_BRANCH_LOOKUPS: dict[int, tuple[int, ...]] = {28: (0, 1)}
PAIRED_DIRECT_BRANCH_LOOKUPS: dict[int, tuple[tuple[int, int], ...]] = {}
DIRECT_PREP_SENTINEL = 0xD1A00001
DIRECT_JUMP_SENTINEL = 0xD1A00002
DIRECT_TARGET_SENTINEL = 0xD1A00003
DIRECT_BRANCH_PRIORITY = 10_000
MADD_FIRST_DEPTH1 = False
SCALAR_FIRST_DEPTH1_SET: frozenset[int] = frozenset()
SCALAR_SECOND_PATH_GROUPS: frozenset[int] = frozenset()
SCALAR_SECOND_PATH_DEPTH2_GROUPS: frozenset[int] = frozenset()
SCALAR_SECOND_PATH_DEPTH3_GROUPS: frozenset[int] = frozenset()
SCALAR_LEVEL4_CONDITION_GROUPS: frozenset[int] = frozenset((0,))
SCALAR_FINAL_C5_SET: frozenset[int] = frozenset((18, 20))
SCALAR_FINAL_JOIN_SET: frozenset[int] = frozenset((21,))
SCALAR_FINAL_HASH4_SET: frozenset[int] = frozenset()
SCALAR_FINAL_SHIFT_SET: frozenset[int] = frozenset((17, 20, 23, 26))
SCALAR_FINAL_HASH23_JOIN_SET: frozenset[int] = frozenset((17, 26))
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
HASH_SCALAR_EXTRA = frozenset(
    (_BASE_SCALAR | set(_SCALAR_CANDIDATES[:43]))
    - {(27, 15), (28, 15), (29, 15), (30, 15), (31, 15)}
)
HYBRID_MADD_PAIRS = 8
HYBRID_MADD_OVERRIDES: dict[tuple[int, int], int] = {}
SCHEDULE_POLICIES = (4,)
BACKWARD_POLICIES = ()
SCHEDULE_NOISE_SEED: int | None = None
SCHEDULE_NOISE_AMPLITUDE = 1
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
_FINAL_HASH_TAIL_TAGS = (
    "hash_23_add",
    "hash_23_shift",
    "hash_23_join",
    "hash_4",
    "hash_5_shift",
    "hash_5_const",
    "hash_5_join",
)
_FINAL_HASH_FULL_TAGS = (
    "paired_branch_select",
    "paired_branch_node_xor",
    "hash_0",
    "hash_1_shift_vector",
    "hash_1_const",
    "hash_1_join",
) + _FINAL_HASH_TAIL_TAGS
OP_PRIORITY_OFFSETS: dict[tuple[str, int, int], int] = {
    **{
        (tag, group, 15): 20_000
        for tag in _FINAL_HASH_TAIL_TAGS
        for group in (27, 28, 29, 30)
    },
    **{(tag, 31, 15): 1_000 for tag in _FINAL_HASH_FULL_TAGS},
    **{
        (tag, group, 15): 50_000
        for tag in ("tree_gather", "node_xor")
        for group in (24, 25, 26, 27, 28, 30)
    },
}
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
        paired_candidate_yes = mirrors[0]
        paired_candidate_no = temps[0]
        paired_base_registers = [mirrors[1] + pair for pair in range(4)]
        paired_jump_registers = [temps[1] + pair for pair in range(4)]
        # The last group is the critical tail.  During its second traversal,
        # keep the three hash parity bits in registers that are already dead
        # at the tail instead of rebuilding them from the packed mirror with
        # five VALU ``&`` instructions.  The first two vectors subsequently
        # become the paired-dispatch candidates; the third is dead group-2
        # mirror storage (group 1 is reserved for dispatch bases).
        saved_second_path_bits = (
            {
                BRANCH_FINAL_GROUP: (
                    paired_candidate_yes,
                    paired_candidate_no,
                    mirrors[2],
                )
            }
            if PAIRED_BRANCH_FINAL
            else {}
        )
        saved_second_path_mux = (
            {
                BRANCH_FINAL_GROUP: (
                    temps[2],
                    mirrors[3],
                    temps[3],
                    mirrors[4],
                )
            }
            if PAIRED_BRANCH_FINAL
            else {}
        )

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
        for group in SAVED_SECOND_PATH_EXTRA_GROUPS:
            saved_second_path_bits[group] = tuple(bits[8])
            saved_second_path_mux[group] = (
                select_spill[8][0],
                bits[7][0],
                bits[7][1],
                bits[7][2],
            )
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
        direct_branch_records = tuple(
            (group, lane)
            for group in sorted(DIRECT_BRANCH_LOOKUPS)
            for lane in DIRECT_BRANCH_LOOKUPS[group]
        )
        if len(direct_branch_records) > 8:
            raise ValueError("direct branch lookup supports at most eight lanes")
        direct_branch_index = {
            record: index for index, record in enumerate(direct_branch_records)
        }
        direct_table_offsets = (0, 16, 33, 69, 133, 261, 517, 1029)
        direct_offset_registers = (
            -1,
            s_sixteen,
            s_m2,
            depth_base.get(5, -1),
            depth_base.get(6, -1),
            depth_base.get(7, -1),
            depth_base.get(8, -1),
            depth_base.get(9, -1),
        )
        paired_direct_records = tuple(
            (group, lanes)
            for group in sorted(PAIRED_DIRECT_BRANCH_LOOKUPS)
            for lanes in PAIRED_DIRECT_BRANCH_LOOKUPS[group]
        )
        if len(paired_direct_records) > 4:
            raise ValueError("paired direct lookup supports at most four pairs")
        paired_direct_index = {
            record: index for index, record in enumerate(paired_direct_records)
        }
        paired_direct_offsets = (0, 1029, 2053, 4097)
        paired_direct_offset_registers = (
            -1,
            depth_base.get(9, -1),
            depth_base.get(10, -1),
            s_m0,
        )
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
            saved_path = saved_second_path_bits.get(gg) if rnd >= 11 else None

            if depth == 1:
                condition = (
                    saved_path[0]
                    if saved_path is not None
                    else (
                        mirrors[state]
                        if DIRECT_MIRROR_PATH or rnd >= 11
                        else workspace_bits[0]
                    )
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
                if saved_path is not None:
                    condition_1 = saved_path[1]
                    condition_0 = saved_path[0]
                    mux = saved_second_path_mux[gg]
                    # Schedule the mux from the oldest (most-significant)
                    # path bit toward the newest one.  p0 is available a
                    # whole hash round before p1, so both half selections can
                    # execute speculatively while the next parity is still in
                    # flight; only the final select waits for p1.
                    emit_vselect(
                        temp,
                        condition_0,
                        leaves[2],
                        leaves[0],
                        group=gg,
                        round=rnd,
                    )
                    emit_vselect(
                        mux[0],
                        condition_0,
                        leaves[3],
                        leaves[1],
                        group=gg,
                        round=rnd,
                    )
                    emit_vselect(
                        temp,
                        condition_1,
                        mux[0],
                        temp,
                        group=gg,
                        round=rnd,
                    )
                    return
                else:
                    condition_1 = workspace_bits[1]
                    condition_0 = workspace_bits[0]
                if saved_path is None and (DIRECT_MIRROR_PATH or rnd >= 11):
                    for dest, vector_mask, scalar_mask, tag in (
                        (
                            workspace_bits[1],
                            v_one,
                            s_one,
                            "second_path_condition_1",
                        ),
                        (
                            workspace_bits[0],
                            v_two,
                            s_two,
                            "second_path_condition_0",
                        ),
                    ):
                        if (
                            gg in SCALAR_SECOND_PATH_GROUPS
                            or gg in SCALAR_SECOND_PATH_DEPTH2_GROUPS
                        ):
                            emit_scalarized(
                                "&",
                                dest,
                                mirrors[state],
                                scalar_mask,
                                b_scalar=True,
                                tag=tag,
                                group=gg,
                                round=rnd,
                            )
                        else:
                            emit_valu(
                                "&",
                                dest,
                                mirrors[state],
                                vector_mask,
                                tag=tag,
                                group=gg,
                                round=rnd,
                            )
                emit_vselect(
                    temp, condition_1, leaves[1], leaves[0], group=gg, round=rnd
                )
                emit_vselect(
                    workspace_spill[0],
                    condition_1,
                    leaves[3],
                    leaves[2],
                    group=gg,
                    round=rnd,
                )
                emit_vselect(
                    temp,
                    condition_0,
                    workspace_spill[0],
                    temp,
                    group=gg,
                    round=rnd,
                )
                return

            if depth != 3:
                raise AssertionError(depth)

            if gg in PIPELINED_DEPTH3_GROUPS and rnd == 14:
                pipeline_workspace = PIPELINED_DEPTH3_WORKSPACE_OVERRIDES.get(
                    gg, WORKSPACE_ASSIGNMENT[gg]
                )
                pipeline_bits = bits[pipeline_workspace]
                pipeline_spill = select_spill[
                    pipeline_workspace % N_SPILL_WORKSPACES
                ][0]
                emit_vselect(
                    temp,
                    pipeline_spill,
                    pipeline_bits[2],
                    pipeline_bits[1],
                    group=gg,
                    round=rnd,
                )
                return

            if saved_path is not None:
                condition_2, condition_1, condition_0 = (
                    saved_path[2],
                    saved_path[1],
                    saved_path[0],
                )
                mux = saved_second_path_mux[gg]
                # A conventional depth-first mux waits for the newest p2 bit
                # before issuing any of its seven flow operations.  Reverse
                # the tree: issue four p0 selections as soon as round 11
                # finishes, collapse them with p1 after round 12, and leave
                # only one select dependent on round-13 parity p2.
                emit_vselect(
                    temp, condition_0, leaves[4], leaves[0], group=gg, round=rnd
                )
                emit_vselect(
                    mux[0],
                    condition_0,
                    leaves[5],
                    leaves[1],
                    group=gg,
                    round=rnd,
                )
                emit_vselect(
                    mux[1],
                    condition_0,
                    leaves[6],
                    leaves[2],
                    group=gg,
                    round=rnd,
                )
                emit_vselect(
                    mux[2],
                    condition_0,
                    leaves[7],
                    leaves[3],
                    group=gg,
                    round=rnd,
                )
                emit_vselect(
                    temp,
                    condition_1,
                    mux[1],
                    temp,
                    group=gg,
                    round=rnd,
                )
                emit_vselect(
                    mux[3],
                    condition_1,
                    mux[2],
                    mux[0],
                    group=gg,
                    round=rnd,
                )
                emit_vselect(
                    temp,
                    condition_2,
                    mux[3],
                    temp,
                    group=gg,
                    round=rnd,
                )
                return
            else:
                condition_2, condition_1, condition_0 = (
                    workspace_bits[2],
                    workspace_bits[1],
                    workspace_bits[0],
                )
            if saved_path is None and (DIRECT_MIRROR_PATH or rnd >= 11):
                for dest, vector_mask, scalar_mask, tag in (
                    (
                        workspace_bits[2],
                        v_one,
                        s_one,
                        "second_path_condition_2",
                    ),
                    (
                        workspace_bits[1],
                        v_two,
                        s_two,
                        "second_path_condition_1",
                    ),
                    (
                        workspace_bits[0],
                        v_four,
                        s_four,
                        "second_path_condition_0",
                    ),
                ):
                    if (
                        gg in SCALAR_SECOND_PATH_GROUPS
                        or gg in SCALAR_SECOND_PATH_DEPTH3_GROUPS
                    ):
                        emit_scalarized(
                            "&",
                            dest,
                            mirrors[state],
                            scalar_mask,
                            b_scalar=True,
                            tag=tag,
                            group=gg,
                            round=rnd,
                        )
                    else:
                        emit_valu(
                            "&",
                            dest,
                            mirrors[state],
                            vector_mask,
                            tag=tag,
                            group=gg,
                            round=rnd,
                        )

            # Evaluate the two four-leaf halves depth-first.  Once the final
            # bottom pair has consumed bit 2, that register itself becomes a
            # legal destination, reducing the live mux stack to one spill.
            spill = workspace_spill[0]
            emit_vselect(temp, condition_2, leaves[1], leaves[0], group=gg, round=rnd)
            emit_vselect(spill, condition_2, leaves[3], leaves[2], group=gg, round=rnd)
            emit_vselect(temp, condition_1, spill, temp, group=gg, round=rnd)
            emit_vselect(spill, condition_2, leaves[5], leaves[4], group=gg, round=rnd)
            emit_vselect(condition_2, condition_2, leaves[7], leaves[6], group=gg, round=rnd)
            emit_vselect(spill, condition_1, condition_2, spill, group=gg, round=rnd)
            emit_vselect(temp, condition_0, spill, temp, group=gg, round=rnd)

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
                for dest, vector_mask, scalar_mask, tag in (
                    (cond, v_one, s_one, "level4_condition_3"),
                    (level4_condition_2, v_two, s_two, "level4_condition_2"),
                    (workspace_bits[1], v_four, s_four, "level4_condition_1"),
                    (workspace_bits[0], v_eight, s_eight, "level4_condition_0"),
                ):
                    if gg in SCALAR_LEVEL4_CONDITION_GROUPS:
                        emit_scalarized(
                            "&",
                            dest,
                            mirrors[state],
                            scalar_mask,
                            b_scalar=True,
                            tag=tag,
                            group=gg,
                            round=rnd,
                        )
                    else:
                        emit_valu(
                            "&",
                            dest,
                            mirrors[state],
                            vector_mask,
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

        direct_global_previous_target = [-1]

        def gather_node(depth: int, state: int, gg: int, rnd: int) -> None:
            mirror = mirrors[state]
            temp = temps[state]
            branch_lanes = (
                frozenset(range(VLEN))
                if PAIRED_BRANCH_FINAL
                and gg == BRANCH_FINAL_GROUP
                and rnd == rounds - 1
                and depth == 4
                else frozenset(BRANCH_FINAL_LANES)
                if gg == BRANCH_FINAL_GROUP and rnd == rounds - 1 and depth == 4
                else frozenset()
            )
            direct_branch_lanes = (
                frozenset(DIRECT_BRANCH_LOOKUPS.get(gg, ()))
                if rnd == rounds - 1 and depth == 4
                else frozenset()
            )
            paired_direct_lanes = (
                frozenset(
                    lane
                    for pair in PAIRED_DIRECT_BRANCH_LOOKUPS.get(gg, ())
                    for lane in pair
                )
                if rnd == rounds - 1 and depth == 4
                else frozenset()
            )
            skipped_lanes = branch_lanes | direct_branch_lanes | paired_direct_lanes
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
                    if lane in skipped_lanes:
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
                if lane in skipped_lanes:
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
            if branch_lanes and not PAIRED_BRANCH_FINAL:
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
            elif branch_lanes:
                value = values[state]
                if PAIRED_EARLY_XOR:
                    emit_valu(
                        "-",
                        paired_candidate_yes,
                        paired_candidate_yes,
                        paired_candidate_no,
                        tag="paired_branch_difference",
                        group=gg,
                        round=rnd,
                    )
                    emit_madd(
                        value,
                        temp,
                        paired_candidate_yes,
                        paired_candidate_no,
                        tag="paired_branch_select",
                        group=gg,
                        round=rnd,
                    )
                else:
                    emit_madd(
                        temp,
                        temp,
                        paired_candidate_yes,
                        paired_candidate_no,
                        tag="paired_branch_select",
                        group=gg,
                        round=rnd,
                    )
                    emit_valu(
                        "^",
                        value,
                        value,
                        temp,
                        tag="paired_branch_node_xor",
                        group=gg,
                        round=rnd,
                    )
            if direct_branch_lanes:
                previous_target = direct_global_previous_target[0]
                for lane in sorted(direct_branch_lanes):
                    record_index = direct_branch_index[(gg, lane)]
                    prep = graph.emit(
                        "alu",
                        (
                            "+",
                            mirror + lane,
                            mirror + lane,
                            s_m23,
                        ),
                        reads=(mirror + lane, s_m23),
                        writes=(mirror + lane,),
                        deps=((previous_target, 1),) if previous_target >= 0 else (),
                        tag="direct_branch_base",
                        group=gg,
                        round=rnd,
                    )
                    if record_index:
                        offset_register = direct_offset_registers[record_index]
                        if offset_register < 0:
                            raise ValueError("missing direct branch offset register")
                        prep = graph.emit(
                            "alu",
                            (
                                "+",
                                mirror + lane,
                                mirror + lane,
                                offset_register,
                            ),
                            reads=(mirror + lane, offset_register),
                            writes=(mirror + lane,),
                            deps=((prep, 1),),
                            tag="direct_branch_offset",
                            group=gg,
                            round=rnd,
                        )
                    jump = graph.emit(
                        "flow",
                        ("jump_indirect", mirror + lane),
                        reads=(mirror + lane,),
                        deps=((prep, 1),),
                        tag="direct_branch_jump",
                        group=gg,
                        round=rnd,
                    )
                    copy = graph.emit(
                        "alu",
                        (
                            "|",
                            temp + lane,
                            node_vec[0] + lane,
                            node_vec[0] + lane,
                        ),
                        reads=(node_vec[0] + lane,),
                        writes=(temp + lane,),
                        deps=((jump, 1),),
                        tag="direct_branch_copy",
                        group=gg,
                        round=rnd,
                    )
                    previous_target = graph.emit(
                        "flow",
                        (
                            "add_imm",
                            mirror + lane,
                            mirror + lane,
                            DIRECT_TARGET_SENTINEL,
                        ),
                        deps=((copy, 0),),
                        tag="direct_branch_target",
                        group=gg,
                        round=rnd,
                    )
                direct_global_previous_target[0] = previous_target
            if paired_direct_lanes:
                previous_target = -1
                for lanes in PAIRED_DIRECT_BRANCH_LOOKUPS[gg]:
                    lane0, lane1 = lanes
                    record_index = paired_direct_index[(gg, lanes)]
                    prep = graph.emit(
                        "alu",
                        ("*", mirror + lane0, mirror + lane0, s_sixteen),
                        reads=(mirror + lane0, s_sixteen),
                        writes=(mirror + lane0,),
                        deps=((previous_target, 1),) if previous_target >= 0 else (),
                        tag="paired_direct_branch_index_high",
                        group=gg,
                        round=rnd,
                    )
                    prep = graph.emit(
                        "alu",
                        ("+", mirror + lane0, mirror + lane0, mirror + lane1),
                        reads=(mirror + lane0, mirror + lane1),
                        writes=(mirror + lane0,),
                        deps=((prep, 1),),
                        tag="paired_direct_branch_index_low",
                        group=gg,
                        round=rnd,
                    )
                    prep = graph.emit(
                        "alu",
                        ("*", mirror + lane0, mirror + lane0, s_two),
                        reads=(mirror + lane0, s_two),
                        writes=(mirror + lane0,),
                        deps=((prep, 1),),
                        tag="paired_direct_branch_stride",
                        group=gg,
                        round=rnd,
                    )
                    prep = graph.emit(
                        "alu",
                        ("*", mirror + lane0, mirror + lane0, s_two),
                        reads=(mirror + lane0, s_two),
                        writes=(mirror + lane0,),
                        deps=((prep, 1),),
                        tag="paired_direct_branch_stride",
                        group=gg,
                        round=rnd,
                    )
                    prep = graph.emit(
                        "alu",
                        ("+", mirror + lane0, mirror + lane0, s_m23),
                        reads=(mirror + lane0, s_m23),
                        writes=(mirror + lane0,),
                        deps=((prep, 1),),
                        tag="paired_direct_branch_base",
                        group=gg,
                        round=rnd,
                    )
                    if record_index:
                        offset_register = paired_direct_offset_registers[
                            record_index
                        ]
                        prep = graph.emit(
                            "alu",
                            (
                                "+",
                                mirror + lane0,
                                mirror + lane0,
                                offset_register,
                            ),
                            reads=(mirror + lane0, offset_register),
                            writes=(mirror + lane0,),
                            deps=((prep, 1),),
                            tag="paired_direct_branch_offset",
                            group=gg,
                            round=rnd,
                        )
                    jump = graph.emit(
                        "flow",
                        ("jump_indirect", mirror + lane0),
                        reads=(mirror + lane0,),
                        deps=((prep, 1),),
                        tag="paired_direct_branch_jump",
                        group=gg,
                        round=rnd,
                    )
                    copies = []
                    for lane in lanes:
                        copies.append(
                            graph.emit(
                                "alu",
                                (
                                    "|",
                                    temp + lane,
                                    node_vec[0] + lane,
                                    node_vec[0] + lane,
                                ),
                                reads=(node_vec[0] + lane,),
                                writes=(temp + lane,),
                                deps=((jump, 1),),
                                tag="paired_direct_branch_copy",
                                group=gg,
                                round=rnd,
                            )
                        )
                    previous_target = graph.emit(
                        "flow",
                        (
                            "add_imm",
                            mirror + lane0,
                            mirror + lane0,
                            DIRECT_TARGET_SENTINEL,
                        ),
                        deps=tuple((copy, 0) for copy in copies),
                        tag="paired_direct_branch_target",
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
            if (
                PAIRED_BRANCH_FINAL
                and PAIRED_EARLY_XOR
                and gg == BRANCH_FINAL_GROUP
                and rnd == 14
            ):
                for candidate, suffix in (
                    (paired_candidate_yes, "yes"),
                    (paired_candidate_no, "no"),
                ):
                    emit_valu(
                        "^",
                        candidate,
                        candidate,
                        value,
                        tag=f"paired_branch_{suffix}_hash_value",
                        group=gg,
                        round=rnd,
                    )
                    emit_valu(
                        "^",
                        candidate,
                        candidate,
                        temp,
                        tag=f"paired_branch_{suffix}_hash_shift",
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
                if not (
                    PAIRED_BRANCH_FINAL
                    and gg == BRANCH_FINAL_GROUP
                    and rnd == rounds - 1
                ):
                    xor_node(value, temp, gg, rnd)

            if PAIRED_BRANCH_FINAL and gg == BRANCH_FINAL_GROUP and rnd == 14:
                previous_target = -1
                paired_start_dep = max(
                    i
                    for i, op in enumerate(graph.ops)
                    if op.tag == "tree_select"
                    and op.group == gg
                    and op.round == rnd
                )
                for pair in range(4):
                    lane0 = 2 * pair
                    lane1 = lane0 + 1
                    jump_register = paired_jump_registers[pair]
                    prep = graph.emit(
                        "alu",
                        ("*", jump_register, mirrors[state] + lane0, s_eight),
                        reads=(mirrors[state] + lane0, s_eight),
                        writes=(jump_register,),
                        deps=(),
                        tag="paired_branch_index_high",
                        group=gg,
                        round=rnd,
                    )
                    prep = graph.emit(
                        "alu",
                        (
                            "+",
                            jump_register,
                            jump_register,
                            mirrors[state] + lane1,
                        ),
                        reads=(jump_register, mirrors[state] + lane1),
                        writes=(jump_register,),
                        deps=((prep, 1),),
                        tag="paired_branch_index_low",
                        group=gg,
                        round=rnd,
                    )
                    prep = graph.emit(
                        "alu",
                        (
                            "+",
                            jump_register,
                            jump_register,
                            paired_base_registers[pair],
                        ),
                        reads=(jump_register, paired_base_registers[pair]),
                        writes=(jump_register,),
                        deps=((prep, 1),),
                        tag="paired_branch_table_base",
                        group=gg,
                        round=rnd,
                    )
                    jump = graph.emit(
                        "flow",
                        ("jump_indirect", jump_register),
                        reads=(jump_register,),
                        deps=(
                            ((prep, 1), (previous_target, 1))
                            if previous_target >= 0
                            else ((prep, 1), (paired_start_dep, 0))
                        ),
                        tag="paired_branch_jump",
                        group=gg,
                        round=rnd,
                    )
                    copies = []
                    for dest in (
                        paired_candidate_yes + lane0,
                        paired_candidate_no + lane0,
                        paired_candidate_yes + lane1,
                        paired_candidate_no + lane1,
                    ):
                        copies.append(
                            graph.emit(
                                "alu",
                                ("|", dest, node_vec[0], node_vec[0]),
                                reads=(node_vec[0],),
                                writes=(dest,),
                                deps=((jump, 1),),
                                tag="paired_branch_copy",
                                group=gg,
                                round=rnd,
                            )
                        )
                    previous_target = graph.emit(
                        "flow",
                        (
                            "add_imm",
                            jump_register,
                            jump_register,
                            PAIRED_TARGET_SENTINEL,
                        ),
                        deps=tuple((copy, 0) for copy in copies),
                        tag="paired_branch_target",
                        group=gg,
                        round=rnd,
                    )

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

            if gg in PIPELINED_DEPTH3_GROUPS and rnd == 13:
                pipeline_workspace = PIPELINED_DEPTH3_WORKSPACE_OVERRIDES.get(
                    gg, WORKSPACE_ASSIGNMENT[gg]
                )
                pipeline_bits = bits[pipeline_workspace]
                pipeline_spill = select_spill[
                    pipeline_workspace % N_SPILL_WORKSPACES
                ][0]
                leaves = [node_vec[14 - index] for index in range(8)]
                emit_valu(
                    "&",
                    pipeline_bits[0],
                    mirrors[state],
                    v_two,
                    tag="pipelined_depth3_condition_0",
                    group=gg,
                    round=rnd,
                )
                emit_vselect(
                    pipeline_bits[1],
                    pipeline_bits[0],
                    leaves[4],
                    leaves[0],
                    group=gg,
                    round=rnd,
                )
                emit_vselect(
                    pipeline_bits[2],
                    pipeline_bits[0],
                    leaves[5],
                    leaves[1],
                    group=gg,
                    round=rnd,
                )
                emit_vselect(
                    pipeline_spill,
                    pipeline_bits[0],
                    leaves[6],
                    leaves[2],
                    group=gg,
                    round=rnd,
                )
                emit_vselect(
                    pipeline_bits[0],
                    pipeline_bits[0],
                    leaves[7],
                    leaves[3],
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
            elif gg in PIPELINED_DEPTH3_GROUPS and rnd == 13:
                pipeline_workspace = PIPELINED_DEPTH3_WORKSPACE_OVERRIDES.get(
                    gg, WORKSPACE_ASSIGNMENT[gg]
                )
                pipeline_bits = bits[pipeline_workspace]
                pipeline_spill = select_spill[
                    pipeline_workspace % N_SPILL_WORKSPACES
                ][0]
                emit_valu(
                    "&",
                    temp,
                    mirrors[state],
                    v_one,
                    tag="pipelined_depth3_condition_1",
                    group=gg,
                    round=rnd,
                )
                emit_vselect(
                    pipeline_bits[1],
                    temp,
                    pipeline_spill,
                    pipeline_bits[1],
                    group=gg,
                    round=rnd,
                )
                emit_vselect(
                    pipeline_bits[2],
                    temp,
                    pipeline_bits[0],
                    pipeline_bits[2],
                    group=gg,
                    round=rnd,
                )
                emit_parity(pipeline_spill, value, gg, rnd)
                emit_madd(
                    mirrors[state],
                    mirrors[state],
                    v_two,
                    pipeline_spill,
                    tag="mirror_update_pipelined_depth3",
                    group=gg,
                    round=rnd,
                )
            elif gg in saved_second_path_bits and rnd == 11:
                emit_parity(saved_second_path_bits[gg][0], value, gg, rnd)
            elif gg in saved_second_path_bits and rnd in (12, 13):
                saved_path = saved_second_path_bits[gg]
                parity = saved_path[rnd - 11]
                emit_parity(parity, value, gg, rnd)
                if rnd == 12:
                    emit_madd(
                        mirrors[state],
                        saved_path[0],
                        v_two,
                        saved_path[1],
                        tag="mirror_update_saved_path",
                        group=gg,
                        round=rnd,
                    )
                else:
                    emit_madd(
                        mirrors[state],
                        mirrors[state],
                        v_two,
                        saved_path[2],
                        tag="mirror_update_saved_path",
                        group=gg,
                        round=rnd,
                    )
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
            elif PAIRED_BRANCH_FINAL and gg == BRANCH_FINAL_GROUP and rnd == 14:
                # The paired final lookup consumes only the new parity bit;
                # the packed four-bit mirror has no remaining user.  Avoid a
                # dead MADD and release mirror[31] for its output pointer.
                emit_parity(temp, value, gg, rnd)
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
        elif PRESERVED_TAIL_OUTPUT_POINTERS:
            # The two input streams naturally finish at groups 30 and 31.
            # Preserve those endpoint addresses for the last two stores.
            # Constants whose vector broadcasts are already complete become
            # independent pointers for groups 26..29, while c0/c23 roll only
            # through groups 0..25.  This removes all late address-chain
            # dependencies without any additional scratch allocation.
            emit_immediate(top_p0, values_base, "output_pointer")
            emit_immediate(top_p1, values_base + VLEN, "output_pointer")
            for pair in range(13):
                g0 = 2 * pair
                g1 = g0 + 1
                graph.emit(
                    "store",
                    ("vstore", top_p0, values[g0]),
                    reads=(top_p0,) + _words(values[g0]),
                    tag="output_store",
                    group=g0,
                )
                graph.emit(
                    "store",
                    ("vstore", top_p1, values[g1]),
                    reads=(top_p1,) + _words(values[g1]),
                    tag="output_store",
                    group=g1,
                )
                if pair != 12:
                    for pointer in (top_p0, top_p1):
                        graph.emit(
                            "alu",
                            ("+", pointer, pointer, s_sixteen),
                            reads=(pointer, s_sixteen),
                            writes=(pointer,),
                            tag="pointer_advance",
                        )

            for group, pointer in zip(
                range(26, 30),
                (s_nineteen, s_c2_shift9, s_c4, s_four),
            ):
                emit_const(
                    pointer,
                    values_base + group * VLEN,
                    "tail_output_pointer",
                )
                graph.emit(
                    "store",
                    ("vstore", pointer, values[group]),
                    reads=(pointer,) + _words(values[group]),
                    tag="output_store",
                    group=group,
                )

            for group, pointer in ((30, io_p0), (31, io_p1)):
                graph.emit(
                    "store",
                    ("vstore", pointer, values[group]),
                    reads=(pointer,) + _words(values[group]),
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
        elif OUTPUT_POINTER_STREAMS == 16:
            # Four independent modulo-4 store chains.  Five scalar lanes from
            # group 5's mirror are dead after its final lookup,
            # so they provide pointer/stride registers without extending any
            # hash-constant lifetime.  The fifth word holds the 32-word stride, keeping
            # the total scalar-ALU work below the original two-chain design
            # while avoiding contention with the critical paired flow trace.
            output_pointer_base = mirrors[5]
            output_pointers = tuple(
                output_pointer_base + stream for stream in range(4)
            )
            output_stride = output_pointer_base + 4
            emit_const(output_stride, 4 * VLEN, "output_stride")
            for stream, pointer in enumerate(output_pointers):
                emit_const(
                    pointer,
                    values_base + stream * VLEN,
                    "output_pointer",
                )
                groups = tuple(range(stream, N_GROUPS, 4))
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
                            ("+", pointer, pointer, output_stride),
                            reads=(pointer, output_stride),
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
            rolling_pairs = (
                N_GROUPS // 2
                if INDEPENDENT_TAIL_OUTPUTS
                else N_GROUPS // 2
            )
            for pair in range(rolling_pairs):
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
                if pair != rolling_pairs - 1:
                    for pointer in (io_p0, io_p1):
                        if FLOW_OUTPUT_POINTER_ADVANCE or pair in FLOW_OUTPUT_ADVANCE_POSITIONS:
                            graph.emit(
                                "flow",
                                ("add_imm", pointer, pointer, 16),
                                reads=(pointer,),
                                writes=(pointer,),
                                tag="pointer_advance",
                            )
                        else:
                            graph.emit(
                                "alu",
                                ("+", pointer, pointer, s_sixteen),
                                reads=(pointer, s_sixteen),
                                writes=(pointer,),
                                tag="pointer_advance",
                            )
            if INDEPENDENT_TAIL_OUTPUTS:
                # Tail stores are filled into already-scheduled empty bundles
                # below.  Keeping them out of the DAG prevents their short
                # address loads from distorting the global load-distance
                # heuristic and delaying the actual computation.
                pass

        self.scratch_ptr = scratch.ptr
        self.scratch_debug = scratch.debug

        # Try several deterministic scheduling priorities.  Construction time
        # is unscored, so select the shortest legal bundle sequence.
        schedules = []
        self.dag_ops = graph.ops
        forward_cycles: list[list[int]] = []
        for policy in SCHEDULE_POLICIES:
            priority_noise = None
            if SCHEDULE_NOISE_SEED is not None:
                modulus = 2 * SCHEDULE_NOISE_AMPLITUDE + 1
                priority_noise = []
                for index in range(len(graph.ops)):
                    op = graph.ops[index]
                    if op.round != 15 and op.tag not in {
                        "output_store",
                        "pointer_advance",
                    }:
                        priority_noise.append(0)
                        continue
                    word = (
                        SCHEDULE_NOISE_SEED
                        + 0x9E3779B9 * (index + 1)
                    ) & 0xFFFFFFFF
                    word ^= word >> 16
                    word = (word * 0x7FEB352D) & 0xFFFFFFFF
                    word ^= word >> 15
                    priority_noise.append(
                        word % modulus - SCHEDULE_NOISE_AMPLITUDE
                    )
            schedule, cycles = self._schedule(
                graph.ops,
                policy,
                return_cycles=True,
                priority_noise=priority_noise,
            )
            schedules.append(schedule)
            forward_cycles.append(cycles)
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

        if INDEPENDENT_TAIL_OUTPUTS:
            if best_index >= len(SCHEDULE_POLICIES):
                raise ValueError("late tail stores require a forward schedule")
            op_cycles = forward_cycles[best_index]

            def first_bundle_with_room(engine: str, start: int) -> int:
                cycle = start
                while True:
                    if cycle == len(self.instrs):
                        self.instrs.append({})
                    if len(self.instrs[cycle].get(engine, ())) < SLOT_LIMITS[engine]:
                        return cycle
                    cycle += 1

            # Retain all rolling stores in the scheduling DAG because their
            # downstream height is a useful completion heuristic.  Replace
            # only groups 24..31 after scheduling: setup scalars provide
            # die during setup provide safe independent pointers, avoiding
            # both head-of-line blocking and hidden jump-table liveness.
            tail_groups = tuple(range(N_GROUPS - 8, N_GROUPS))
            output_indices = [
                index
                for index, op in enumerate(graph.ops)
                if op.tag == "output_store" and op.group in tail_groups
            ]
            group23_store = max(
                index
                for index, op in enumerate(graph.ops)
                if op.tag == "output_store" and op.group == N_GROUPS - 9
            )
            removable = output_indices + [
                index
                for index, op in enumerate(graph.ops)
                if index > group23_store and op.tag == "pointer_advance"
            ]
            for index in removable:
                op = graph.ops[index]
                slots = self.instrs[op_cycles[index]].get(op.engine, [])
                slots.remove(op.slot)
                if not slots:
                    self.instrs[op_cycles[index]].pop(op.engine, None)

            value_ready_by_group = {}
            for group in tail_groups:
                value_words = set(_words(values[group]))
                value_ready_by_group[group] = max(
                    op_cycles[index]
                    for index, op in enumerate(graph.ops)
                    if value_words.intersection(op.writes)
                )
            pointer_for_group = dict(
                zip(
                    tail_groups,
                    (
                        s_nineteen,
                        s_c2_shift9,
                        s_c0,
                        s_one,
                        s_c4,
                        s_four,
                        s_m4,
                        s_c23,
                    ),
                )
            )
            late_store_jobs: list[tuple[int, int, int]] = []
            for group in tail_groups:
                pointer = pointer_for_group[group]
                pointer_last_use = max(
                    (
                        op_cycles[index]
                        for index, op in enumerate(graph.ops)
                        if pointer in op.reads or pointer in op.writes
                    ),
                    default=-1,
                )
                pointer_cycle = first_bundle_with_room(
                    "load", pointer_last_use + 1
                )
                self.instrs[pointer_cycle].setdefault("load", []).append(
                    ("const", pointer, values_base + group * VLEN)
                )

                value_ready = value_ready_by_group[group]
                late_store_jobs.append(
                    (max(pointer_cycle, value_ready) + 1, group, pointer)
                )

            # Unit-time, capacity-two jobs with release dates: earliest-release
            # order is optimal for makespan and avoids group-number ordering
            # leaving a usable store slot idle before a late critical value.
            for release, group, pointer in sorted(late_store_jobs):
                store_cycle = first_bundle_with_room("store", release)
                self.instrs[store_cycle].setdefault("store", []).append(
                    ("vstore", pointer, values[group])
                )

            while self.instrs and not any(self.instrs[-1].values()):
                self.instrs.pop()

        if BRANCH_FINAL_LANES and not PAIRED_BRANCH_FINAL:
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

        if PAIRED_BRANCH_FINAL:
            main_length = len(self.instrs)
            if self.instrs[-1].get("flow"):
                raise AssertionError("final bundle has no flow slot for paired halt")
            self.instrs[-1]["flow"] = [("halt",)]

            paired_jump_pcs = []
            paired_target_pcs = []
            for pair, jump_register in enumerate(paired_jump_registers):
                jump_pc = next(
                    pc
                    for pc, bundle in enumerate(self.instrs[:main_length])
                    if ("jump_indirect", jump_register) in bundle.get("flow", ())
                )
                target_pc = next(
                    pc
                    for pc, bundle in enumerate(self.instrs[:main_length])
                    if (
                        "add_imm",
                        jump_register,
                        jump_register,
                        PAIRED_TARGET_SENTINEL,
                    )
                    in bundle.get("flow", ())
                )
                if target_pc != jump_pc + 1:
                    raise AssertionError(
                        f"paired trace {pair} is not contiguous: "
                        f"{jump_pc},{target_pc}"
                    )
                paired_jump_pcs.append(jump_pc)
                paired_target_pcs.append(target_pc)

            first_base_use_pc = min(
                pc
                for pc, bundle in enumerate(self.instrs[:main_length])
                for slot in bundle.get("alu", ())
                if slot[0] == "+" and slot[3] in paired_base_registers
            )
            base_words = set(paired_base_registers)
            last_old_base_use = max(
                pc
                for pc, bundle in enumerate(self.instrs[:first_base_use_pc])
                if any(
                    isinstance(item, int) and item in base_words
                    for slots in bundle.values()
                    for slot in slots
                    for item in slot[1:]
                )
            )
            base_prep_pc = next(
                pc
                for pc in range(last_old_base_use + 1, first_base_use_pc)
                if len(self.instrs[pc].get("alu", ())) <= 8
            )
            paired_offsets = (0, 69, 133, 261)
            paired_offset_registers = (
                -1,
                depth_base[5],
                depth_base[6],
                depth_base[7],
            )
            for pair, base_register in enumerate(paired_base_registers):
                if pair == 0:
                    slot = (
                        "|",
                        base_register,
                        branch_table_base,
                        branch_table_base,
                    )
                else:
                    slot = (
                        "+",
                        base_register,
                        branch_table_base,
                        paired_offset_registers[pair],
                    )
                self.instrs[base_prep_pc].setdefault("alu", []).append(slot)

            load_base_pc = next(
                pc
                for pc in range(base_prep_pc)
                if len(self.instrs[pc].get("load", ())) < SLOT_LIMITS["load"]
            )
            self.instrs[load_base_pc].setdefault("load", []).append(
                ("const", branch_table_base, main_length)
            )

            paired_table_blocks: list[dict[str, list[tuple]]] = []
            for pair, target_pc in enumerate(paired_target_pcs):
                while len(paired_table_blocks) < paired_offsets[pair]:
                    paired_table_blocks.append({"flow": [("halt",)]})
                lane0 = 2 * pair
                lane1 = lane0 + 1
                placeholders = (
                    (
                        "|",
                        paired_candidate_yes + lane0,
                        node_vec[0],
                        node_vec[0],
                    ),
                    (
                        "|",
                        paired_candidate_no + lane0,
                        node_vec[0],
                        node_vec[0],
                    ),
                    (
                        "|",
                        paired_candidate_yes + lane1,
                        node_vec[0],
                        node_vec[0],
                    ),
                    (
                        "|",
                        paired_candidate_no + lane1,
                        node_vec[0],
                        node_vec[0],
                    ),
                )
                for placeholder in placeholders:
                    if placeholder not in self.instrs[target_pc].get("alu", ()):
                        raise AssertionError(
                            f"paired copy is not packetized at pc={target_pc}"
                        )
                for combined_index in range(64):
                    mirror0, mirror1 = divmod(combined_index, 8)
                    replacements = (
                        (
                            level4_reversed[2 * mirror0 + 1] + lane0
                            if PAIRED_EARLY_XOR
                            else level4_diff[mirror0] + lane0
                        ),
                        level4_reversed[2 * mirror0] + lane0,
                        (
                            level4_reversed[2 * mirror1 + 1] + lane1
                            if PAIRED_EARLY_XOR
                            else level4_diff[mirror1] + lane1
                        ),
                        level4_reversed[2 * mirror1] + lane1,
                    )
                    target = {
                        engine: list(slots)
                        for engine, slots in self.instrs[target_pc].items()
                    }
                    for placeholder, source in zip(placeholders, replacements):
                        copy_index = target["alu"].index(placeholder)
                        target["alu"][copy_index] = (
                            "|",
                            placeholder[1],
                            source,
                            source,
                        )
                    target["flow"] = [("jump", target_pc + 1)]
                    paired_table_blocks.append(target)
            self.instrs.extend(paired_table_blocks)
            self.branch_main_cycles = main_length

        if DIRECT_BRANCH_LOOKUPS:
            if not BRANCH_FINAL_LANES:
                main_length = len(self.instrs)
                if self.instrs[-1].get("flow"):
                    raise AssertionError("final bundle has no flow slot for halt")
                self.instrs[-1]["flow"] = [("halt",)]
                self.branch_main_cycles = main_length
            direct_table_blocks: list[dict[str, list[tuple]]] = []
            direct_base = len(self.instrs)
            broadcast_pc = next(
                pc
                for pc, bundle in enumerate(
                    self.instrs[: self.branch_main_cycles]
                )
                if ("vbroadcast", v_m23, s_m23) in bundle.get("valu", ())
            )
            first_jump_pc = min(
                pc
                for pc, bundle in enumerate(
                    self.instrs[: self.branch_main_cycles]
                )
                for slot in bundle.get("flow", ())
                if slot[0] == "jump_indirect"
                and any(
                    slot[1] == mirrors[group] + lane
                    for group in DIRECT_BRANCH_LOOKUPS
                    for lane in DIRECT_BRANCH_LOOKUPS[group]
                )
            )
            base_load_pc = next(
                pc
                for pc in range(broadcast_pc + 1, first_jump_pc)
                if len(self.instrs[pc].get("load", ())) < SLOT_LIMITS["load"]
            )
            self.instrs[base_load_pc].setdefault("load", []).append(
                ("const", s_m23, direct_base)
            )
            for group in sorted(DIRECT_BRANCH_LOOKUPS):
                for lane in DIRECT_BRANCH_LOOKUPS[group]:
                    mirror_word = mirrors[group] + lane
                    temp_word = temps[group] + lane
                    jump_pcs = [
                        pc
                        for pc, bundle in enumerate(
                            self.instrs[: self.branch_main_cycles]
                        )
                        for slot in bundle.get("flow", ())
                        if slot == ("jump_indirect", mirror_word)
                    ]
                    target_pcs = [
                        pc
                        for pc, bundle in enumerate(
                            self.instrs[: self.branch_main_cycles]
                        )
                        for slot in bundle.get("flow", ())
                        if slot
                        == (
                            "add_imm",
                            mirror_word,
                            mirror_word,
                            DIRECT_TARGET_SENTINEL,
                        )
                    ]
                    if not (len(jump_pcs) == len(target_pcs) == 1):
                        raise AssertionError(
                            f"missing direct branch trace for group={group} lane={lane}"
                        )
                    jump_pc, target_pc = jump_pcs[0], target_pcs[0]
                    if target_pc != jump_pc + 1:
                        raise AssertionError(
                            "direct branch trace is not contiguous: "
                            f"group={group} lane={lane} pcs="
                            f"{jump_pc},{target_pc}"
                        )
                    record_index = direct_branch_index[(group, lane)]
                    desired_offset = direct_table_offsets[record_index]
                    while len(direct_table_blocks) < desired_offset:
                        direct_table_blocks.append({"flow": [("halt",)]})
                    placeholder_copy = (
                        "|",
                        temp_word,
                        node_vec[0] + lane,
                        node_vec[0] + lane,
                    )
                    if placeholder_copy not in self.instrs[target_pc].get(
                        "alu", ()
                    ):
                        raise AssertionError(
                            "direct branch copy is not packetized with target: "
                            f"group={group} lane={lane} pc={target_pc}"
                        )
                    for mirror_value in range(16):
                        target = {
                            engine: list(slots)
                            for engine, slots in self.instrs[target_pc].items()
                        }
                        copy_index = target["alu"].index(placeholder_copy)
                        target["alu"][copy_index] = (
                            "|",
                            temp_word,
                            level4_reversed[mirror_value] + lane,
                            level4_reversed[mirror_value] + lane,
                        )
                        target["flow"] = [("jump", target_pc + 1)]
                        direct_table_blocks.append(target)
            self.instrs.extend(direct_table_blocks)
            self.direct_branch_table_bundles = len(direct_table_blocks)
        else:
            self.direct_branch_table_bundles = 0

        if PAIRED_DIRECT_BRANCH_LOOKUPS:
            if DIRECT_BRANCH_LOOKUPS:
                raise ValueError("individual and paired direct lookups are exclusive")
            paired_direct_base = len(self.instrs)
            first_jump_pc = min(
                pc
                for pc, bundle in enumerate(
                    self.instrs[: self.branch_main_cycles]
                )
                for group, lanes in paired_direct_records
                if ("jump_indirect", mirrors[group] + lanes[0])
                in bundle.get("flow", ())
            )
            broadcast_pc = next(
                pc
                for pc, bundle in enumerate(
                    self.instrs[: self.branch_main_cycles]
                )
                if ("vbroadcast", v_m23, s_m23) in bundle.get("valu", ())
            )
            base_load_pc = next(
                pc
                for pc in range(broadcast_pc + 1, first_jump_pc)
                if len(self.instrs[pc].get("load", ())) < SLOT_LIMITS["load"]
            )
            self.instrs[base_load_pc].setdefault("load", []).append(
                ("const", s_m23, paired_direct_base)
            )

            paired_direct_blocks: list[dict[str, list[tuple]]] = []
            for group, lanes in paired_direct_records:
                lane0, lane1 = lanes
                mirror_word = mirrors[group] + lane0
                jump_pc = next(
                    pc
                    for pc, bundle in enumerate(
                        self.instrs[: self.branch_main_cycles]
                    )
                    if ("jump_indirect", mirror_word) in bundle.get("flow", ())
                )
                target_pc = next(
                    pc
                    for pc, bundle in enumerate(
                        self.instrs[: self.branch_main_cycles]
                    )
                    if (
                        "add_imm",
                        mirror_word,
                        mirror_word,
                        DIRECT_TARGET_SENTINEL,
                    )
                    in bundle.get("flow", ())
                )
                trace_length = target_pc - jump_pc
                if not 1 <= trace_length <= 4:
                    raise AssertionError(
                        f"paired direct trace is too long: "
                        f"group={group} lanes={lanes} pcs={jump_pc},{target_pc}"
                    )
                record_index = paired_direct_index[(group, lanes)]
                while len(paired_direct_blocks) < paired_direct_offsets[record_index]:
                    paired_direct_blocks.append({"flow": [("halt",)]})
                placeholders = tuple(
                    (
                        "|",
                        temps[group] + lane,
                        node_vec[0] + lane,
                        node_vec[0] + lane,
                    )
                    for lane in lanes
                )
                copy_pcs = {
                    pc
                    for pc in range(jump_pc + 1, target_pc + 1)
                    if all(
                        placeholder in self.instrs[pc].get("alu", ())
                        for placeholder in placeholders
                    )
                }
                if len(copy_pcs) != 1:
                    raise AssertionError(
                        f"paired direct copies are split: pcs={sorted(copy_pcs)}"
                    )
                copy_pc = next(iter(copy_pcs))
                for combined_index in range(256):
                    mirror0, mirror1 = divmod(combined_index, 16)
                    entry_blocks = []
                    for trace_pc in range(jump_pc + 1, target_pc + 1):
                        target = {
                            engine: list(slots)
                            for engine, slots in self.instrs[trace_pc].items()
                        }
                        if trace_pc == copy_pc:
                            for placeholder, mirror_value, lane in zip(
                                placeholders, (mirror0, mirror1), lanes
                            ):
                                source = level4_reversed[mirror_value] + lane
                                copy_index = target["alu"].index(placeholder)
                                target["alu"][copy_index] = (
                                    "|",
                                    placeholder[1],
                                    source,
                                    source,
                                )
                        if trace_pc == target_pc:
                            target["flow"] = [("jump", target_pc + 1)]
                        entry_blocks.append(target)
                    while len(entry_blocks) < 4:
                        entry_blocks.append({"flow": [("halt",)]})
                    paired_direct_blocks.extend(entry_blocks)
            self.instrs.extend(paired_direct_blocks)
            self.direct_branch_table_bundles = len(paired_direct_blocks)

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
            direct_branch_bias = (
                DIRECT_BRANCH_PRIORITY
                if op.tag.startswith("direct_branch_")
                or op.tag.startswith("paired_branch_")
                or op.tag.startswith("paired_direct_branch_")
                else 0
            )
            return (
                h
                + op_offset
                + direct_branch_bias
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
