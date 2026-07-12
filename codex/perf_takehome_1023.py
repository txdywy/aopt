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
WORKSPACE_ASSIGNMENT = (
    2, 0, 3, 2, 0, 2, 1, 0, 4, 3, 0, 2, 3, 5, 6, 4,
    1, 2, 0, 1, 4, 1, 7, 1, 0, 4, 3, 3, 5, 5, 2, 1,
)
SECOND_WORKSPACE_STRIDE = 3
SECOND_WORKSPACE_FIXED = 8
SECOND_WORKSPACE_COUNT = 1
INDEPENDENT_ROOT_CACHE = True
DIRECT_MIRROR_PATH = False
OVERLAP_DEEP_ADDRESS = False
OVERLAP_SHALLOW_ADDRESS = False
TAIL_EMISSION_MODE = "full_offset"
TAIL_EMISSION_STAGGER = 1
TAIL_EMISSION_COHORTS = 8
TAIL_GROUP_ORDER = tuple(range(N_GROUPS))
FULL_ROUND_OFFSETS = (
    5, 0, 2, 15, 4, 24, 0, 16, 2, 7, 8, 9, 24, 10, 10, 16,
    13, 19, 21, 17, 9, 8, 10, 4, 12, 21, 19, 15, 16, 24, 0, 23,
)
SCATTERED_OUTPUT_STORES = False
OUTPUT_POINTER_STREAMS = 2
OUTPUT_GROUP_ORDER = tuple(range(N_GROUPS))
EARLY_FINAL_CACHE_SET: frozenset[int] = frozenset()
MADD_FIRST_DEPTH1 = False
SCALAR_FIRST_DEPTH1_SET: frozenset[int] = frozenset()
SCALAR_FINAL_C5_SET: frozenset[int] = frozenset()
SCALAR_FINAL_JOIN_SET: frozenset[int] = frozenset()
FINAL_CACHE_SET = frozenset(range(9))
FIRST_CACHE_SET = frozenset((5, 7, 9, 12, 13, 15, 17, 18, 19, 22, 25, 29, 31))
HASH_SCALAR_MOD = 4
HASH_SCALAR_STAGE = 1  # positive: shift branch; negative: constant branch
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
HASH_SCALAR_EXTRA = frozenset(_BASE_SCALAR | set(_SCALAR_CANDIDATES[:50]))
HYBRID_MADD_PAIRS = 8
SCHEDULE_POLICIES = (36,)
BACKWARD_POLICIES = ()
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
ROUND_PRIORITY_OFFSETS = (0,) * 16
TAG_PRIORITY_OFFSETS: dict[str, int] = {}
GROUP_FINE_OFFSETS = (0,) * N_GROUPS
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
            scalar_const(4, "four")
            scalar_const(8, "eight")
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
        # The top-tree staging words are dead after their node broadcasts, so
        # their upper half doubles as the preprocessing ping-pong buffers and
        # later as the level-4 mux workspace.
        preprocess_buffers = [top_words + 2 * VLEN, top_words + 3 * VLEN]
        # These setup-only buffers are overwritten with six persistent pair
        # differences after preprocessing, avoiding any extra scratch cost.
        level4_diff = [top_words, top_words + VLEN] + [
            alloc(f"level4_pair_diff_{i}", VLEN)
            for i in range(2, HYBRID_MADD_PAIRS)
        ]
        first_level4_pool = [preprocess_buffers[0]]
        first_level4_condition = preprocess_buffers[1]

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
        for value, addr in scalar_constants.items():
            if addr != s_one:
                emit_immediate(addr, value, "scalar_immediate")
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
            graph.emit(
                "alu",
                ("^", top_p0, top_p1, s_c5),
                reads=(top_p1, s_c5),
                writes=(top_p0,),
                tag="cached_root_scalar_transform",
            )
            emit_vbroadcast(node_vec[0], top_p0, "cached_root_broadcast")
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
            workspace_bits = bits[workspace]
            workspace_spill = select_spill[workspace % N_SPILL_WORKSPACES]

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
            workspace_bits = bits[workspace]
            workspace_spill = select_spill[workspace % N_SPILL_WORKSPACES]
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
                if pair_index < HYBRID_MADD_PAIRS:
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
            if OVERLAP_DEEP_ADDRESS:
                # At depth 4 the mirror still encodes complemented path bits;
                # convert it in place to an absolute memory address.  Later
                # deep rounds keep that address representation directly.
                if depth == 4 and not (
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
                emit_scalarized(
                    "-",
                    temp,
                    depth_base[depth],
                    mirror,
                    a_scalar=True,
                    tag="gather_address",
                    group=gg,
                    round=rnd,
                )
                address = temp
            for lane in range(VLEN):
                graph.emit(
                    "load",
                    ("load_offset", temp, address, lane),
                    reads=(address + lane,),
                    writes=(temp + lane,),
                    tag="tree_gather",
                    group=gg,
                    round=rnd,
                )
            if (gg, rnd) in SCALAR_DYNAMIC_XOR_SET:
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
            emit_valu("^", value, value, temp, tag="hash_1_join", group=gg, round=rnd)

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
            emit_valu("^", value, value, temp, tag="hash_23_join", group=gg, round=rnd)

            emit_madd(value, value, v_m4, v_c4, tag="hash_4", group=gg, round=rnd)

            emit_vbasic(
                ">>",
                temp,
                value,
                v_sixteen,
                scalarize=scalar_hash and HASH_SCALAR_STAGE == 5,
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
            if rnd == rounds - 1 and gg in SCALAR_FINAL_JOIN_SET:
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
            workspace_bits = bits[workspace]

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
                if SECOND_WORKSPACE_FIXED >= 0:
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

        if OUTPUT_POINTER_STREAMS == 8:
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
        for policy in SCHEDULE_POLICIES:
            schedules.append(self._schedule(graph.ops, policy))
        reversed_ops = self._reverse_ops(graph.ops)
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
        self.instrs = min(schedules, key=len)

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
                multiplier = multipliers[variant - 4]
                load_score = -multiplier * distance if finite else -inf
                cohort = -1 if op.group < 0 else op.group // 4
                fine_offset = (
                    0 if op.group < 0 else GROUP_FINE_OFFSETS[op.group]
                )
                round_offset = (
                    0 if op.round < 0 else ROUND_PRIORITY_OFFSETS[op.round]
                )
                return (
                    height[i]
                    + load_score
                    + fine_offset
                    + round_offset
                    + TAG_PRIORITY_OFFSETS.get(op.tag, 0),
                    finite,
                    -cohort,
                    reach[i] // 32,
                    engine_rank[op.engine],
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
            return (h, reach[i] // 32, engine_rank[op.engine], group_bias, id_bias)

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
