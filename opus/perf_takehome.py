"""
Independent optimized KernelBuilder for the performance take-home.

See opus/README.md for the high-level strategy. The public surface matches what
the submission harness expects:

    kb = KernelBuilder()
    kb.build_kernel(forest_height, n_nodes, batch_size, rounds)
    kb.instrs        # list[Instruction]
    kb.debug_info()  # DebugInfo
"""

from collections import defaultdict, deque

from problem import (
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    SCRATCH_SIZE,
    HASH_STAGES,
)


class Assembler:
    """Accumulates slots into VLIW bundles."""

    def __init__(self):
        self.instrs = []
        self.curr = defaultdict(list)

    def add(self, engine, slot):
        self.curr[engine].append(slot)
        assert len(self.curr[engine]) <= SLOT_LIMITS[engine], (
            f"Too many slots for {engine}"
        )

    def emit(self):
        if self.curr:
            self.instrs.append(dict(self.curr))
            self.curr = defaultdict(list)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0

    def alloc_scratch(self, name=None, size=1):
        addr = self.scratch_ptr
        self.scratch_ptr += size
        assert self.scratch_ptr <= SCRATCH_SIZE, (
            f"Scratch overflow: {self.scratch_ptr} > {SCRATCH_SIZE}"
        )
        if name is not None:
            self.scratch[name] = addr
            for i in range(size):
                self.scratch_debug[addr + i] = f"{name}_{i}" if size > 1 else name
        return addr

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    # ------------------------------------------------------------------
    def build_kernel(self, forest_height, n_nodes, batch_size, rounds):
        VG = batch_size // VLEN  # number of vector groups
        assert VG * VLEN == batch_size, "batch_size must be a multiple of VLEN"
        period = forest_height + 1
        # Preload depth: shallow levels selected from registers (no gather).
        PRELOAD = min(3, period)

        asm = Assembler()

        # ---------------- scalar scratch ----------------
        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                     "forest_values_p", "inp_indices_p", "inp_values_p"]
        s_vars = {v: self.alloc_scratch(v) for v in init_vars}

        s_zero = self.alloc_scratch("s_zero")

        s_addr_val = [self.alloc_scratch(f"s_addr_val_{g}") for g in range(VG)]

        # ---------------- vector constant regs ----------------
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)

        # Hash constants.
        #
        # Stages 0,2,4 are `a*k + C` (op1=op2='+', op3='<<'), one fused
        # multiply_add each. Stages 1,3,5 are `(a op1 C1) ^ (a shift s)`.
        #
        # Stage 3 uses a LEFT shift (<<9 == *512), so stages 2 and 3 fuse:
        #   a2 = 33*a1 + C2
        #   a3 = (a2 + C3) ^ (a2 << 9)
        #      = (33*a1 + (C2+C3)) ^ ((33*512)*a1 + 512*C2)
        # i.e. two multiply_adds from a1 plus one xor -> 3 ops for both stages
        # (vs 4), cutting the hash to 11 vector ops. Stages 1 and 5 use RIGHT
        # shifts and can't be folded this way.
        M = 1 << 32
        C0 = HASH_STAGES[0][1]
        C1 = HASH_STAGES[1][1]; S1 = HASH_STAGES[1][4]
        C2 = HASH_STAGES[2][1]
        C3 = HASH_STAGES[3][1]; S3 = HASH_STAGES[3][4]
        C4 = HASH_STAGES[4][1]
        C5 = HASH_STAGES[5][1]; S5 = HASH_STAGES[5][4]

        v_hc0 = self.alloc_scratch("v_hc0", VLEN)      # C0, stage 0 add
        v_m4097 = self.alloc_scratch("v_m4097", VLEN)  # stage 0 mult
        v_hc1 = self.alloc_scratch("v_hc1", VLEN)      # C1
        v_s1 = self.alloc_scratch("v_s1", VLEN)        # shift 19
        v_m33 = self.alloc_scratch("v_m33", VLEN)      # stage 2 mult
        v_fadd = self.alloc_scratch("v_fadd", VLEN)    # (C2+C3) mod 2^32
        v_m2s = self.alloc_scratch("v_m2s", VLEN)      # 33*2^S3 mod 2^32
        v_sadd = self.alloc_scratch("v_sadd", VLEN)    # (2^S3 * C2) mod 2^32
        v_m9 = self.alloc_scratch("v_m9", VLEN)        # stage 4 mult
        v_hc4 = self.alloc_scratch("v_hc4", VLEN)      # C4
        v_hc5 = self.alloc_scratch("v_hc5", VLEN)      # C5
        v_s5 = self.alloc_scratch("v_s5", VLEN)        # shift 16

        hash_consts = [
            (v_hc0, C0), (v_m4097, (1 << HASH_STAGES[0][4]) + 1),
            (v_hc1, C1), (v_s1, S1),
            (v_m33, (1 << HASH_STAGES[2][4]) + 1),
            (v_fadd, (C2 + C3) % M),
            (v_m2s, (((1 << HASH_STAGES[2][4]) + 1) << S3) % M),
            (v_sadd, ((1 << S3) * C2) % M),
            (v_m9, (1 << HASH_STAGES[4][4]) + 1),
            (v_hc4, C4), (v_hc5, C5), (v_s5, S5),
        ]

        # Preloaded leaf values (broadcast). Depth-1/2 node selection is done on
        # the otherwise-idle FLOW engine via vselect, so we keep the raw leaves
        # (not diffs) as vector constants.
        v_leaf = [self.alloc_scratch(f"v_leaf{i}", VLEN) for i in range(7)]

        # base_d = forest_values_p + (2^d - 1) for gather depths
        gather_depths = [d for d in range(PRELOAD, period)]
        v_base = {d: self.alloc_scratch(f"v_base_{d}", VLEN) for d in gather_depths}

        # ---------------- per-group working regs ----------------
        v_val = [self.alloc_scratch(f"v_val_{g}", VLEN) for g in range(VG)]
        v_off = [self.alloc_scratch(f"v_off_{g}", VLEN) for g in range(VG)]
        # 2 private temps per group; a small shared pool supplies the rare 3rd
        # temp needed only by the depth-2 select.
        v_t = [[self.alloc_scratch(f"v_t{j}_{g}", VLEN) for j in range(2)]
               for g in range(VG)]
        NPOOL = 16
        v_pool = [self.alloc_scratch(f"v_pool_{i}", VLEN) for i in range(NPOOL)]

        assert self.scratch_ptr <= SCRATCH_SIZE, (
            f"scratch overflow {self.scratch_ptr}"
        )

        # ---------------- setup scratch temps ----------------
        n_leaf_scalar = min((1 << PRELOAD) - 1, n_nodes)
        s_leaf = [self.alloc_scratch(f"s_leaf_{i}") for i in range(7)]
        s_hdr = [self.alloc_scratch(f"s_hdr_{i}") for i in range(7)]
        s_leafaddr = [self.alloc_scratch(f"s_la_{i}") for i in range(7)]
        hc_tmp = [self.alloc_scratch(f"s_hc_{i}") for i in range(len(hash_consts))]
        s_one = self.alloc_scratch("s_one")
        s_two = self.alloc_scratch("s_two")
        s_base = {d: self.alloc_scratch(f"s_base_{d}") for d in gather_depths}
        pow_regs = {}
        k = 0
        while (1 << k) < VG:
            pow_regs[k] = self.alloc_scratch(f"s_pow_{k}")
            k += 1

        assert self.scratch_ptr <= SCRATCH_SIZE, f"scratch overflow {self.scratch_ptr}"

        # =============================================================
        # EVERYTHING is emitted as a dataflow graph and scheduled together, so
        # one-time setup (header/const loads, broadcasts, address arithmetic)
        # overlaps the first rounds' work instead of running as a serial prologue.
        # =============================================================
        ops = []

        def emit_op(engine, slot, reads, writes, tag=0):
            ops.append({"engine": engine, "slot": slot, "reads": reads,
                        "writes": writes, "tag": tag})

        def vr(reg):
            return list(range(reg, reg + VLEN))

        SETUP = -1  # setup tag: scheduled first (deps also enforce this)

        # ----- header words mem[0..6] -----
        for a in range(7):
            emit_op("load", ("const", s_hdr[a], a), reads=[], writes=[s_hdr[a]], tag=SETUP)
            emit_op("load", ("load", s_vars[init_vars[a]], s_hdr[a]),
                    reads=[s_hdr[a]], writes=[s_vars[init_vars[a]]], tag=SETUP)

        # ----- small scalar constants -----
        emit_op("load", ("const", s_zero, 0), reads=[], writes=[s_zero], tag=SETUP)
        emit_op("load", ("const", s_one, 1), reads=[], writes=[s_one], tag=SETUP)
        emit_op("load", ("const", s_two, 2), reads=[], writes=[s_two], tag=SETUP)
        emit_op("valu", ("vbroadcast", v_one, s_one), reads=[s_one], writes=vr(v_one), tag=SETUP)
        emit_op("valu", ("vbroadcast", v_two, s_two), reads=[s_two], writes=vr(v_two), tag=SETUP)

        fvp = s_vars["forest_values_p"]
        # ----- leaves leaf0..leaf6 (scalar) then broadcast -----
        emit_op("load", ("load", s_leaf[0], fvp), reads=[fvp], writes=[s_leaf[0]], tag=SETUP)
        for i in range(1, n_leaf_scalar):
            emit_op("flow", ("add_imm", s_leafaddr[i], fvp, i),
                    reads=[fvp], writes=[s_leafaddr[i]], tag=SETUP)
            emit_op("load", ("load", s_leaf[i], s_leafaddr[i]),
                    reads=[s_leafaddr[i]], writes=[s_leaf[i]], tag=SETUP)
        for i in range(7):
            emit_op("valu", ("vbroadcast", v_leaf[i], s_leaf[i]),
                    reads=[s_leaf[i]], writes=vr(v_leaf[i]), tag=SETUP)

        # ----- hash constants -----
        for i, (vreg, val) in enumerate(hash_consts):
            emit_op("load", ("const", hc_tmp[i], val), reads=[], writes=[hc_tmp[i]], tag=SETUP)
            emit_op("valu", ("vbroadcast", vreg, hc_tmp[i]),
                    reads=[hc_tmp[i]], writes=vr(vreg), tag=SETUP)

        # ----- base_d = forest_values_p + (2^d - 1) then broadcast -----
        for d in gather_depths:
            emit_op("flow", ("add_imm", s_base[d], fvp, (1 << d) - 1),
                    reads=[fvp], writes=[s_base[d]], tag=SETUP)
            emit_op("valu", ("vbroadcast", v_base[d], s_base[d]),
                    reads=[s_base[d]], writes=vr(v_base[d]), tag=SETUP)

        # ----- value addresses inp_values_p + g*VLEN, built by doubling -----
        ivp = s_vars["inp_values_p"]
        pk = sorted(pow_regs)
        for kk in pk:
            emit_op("load", ("const", pow_regs[kk], VLEN * (1 << kk)),
                    reads=[], writes=[pow_regs[kk]], tag=SETUP)
        emit_op("alu", ("+", s_addr_val[0], ivp, s_zero),
                reads=[ivp, s_zero], writes=[s_addr_val[0]], tag=SETUP)
        k = 0
        while (1 << k) < VG:
            half = 1 << k
            for g in range(half, min(2 * half, VG)):
                emit_op("alu", ("+", s_addr_val[g], s_addr_val[g - half], pow_regs[k]),
                        reads=[s_addr_val[g - half], pow_regs[k]], writes=[s_addr_val[g]],
                        tag=SETUP)
            k += 1

        def hash_ops(g, tag):
            val = v_val[g]
            t0, t1 = v_t[g][0], v_t[g][1]
            # stage 0: val = 4097*val + C0
            emit_op("valu", ("multiply_add", val, val, v_m4097, v_hc0),
                    reads=vr(val) + vr(v_m4097) + vr(v_hc0), writes=vr(val), tag=tag)
            # stage 1: val = (val ^ C1) ^ (val >> 19)
            emit_op("valu", (">>", t0, val, v_s1),
                    reads=vr(val) + vr(v_s1), writes=vr(t0), tag=tag)
            emit_op("valu", ("^", t1, val, v_hc1),
                    reads=vr(val) + vr(v_hc1), writes=vr(t1), tag=tag)
            emit_op("valu", ("^", val, t1, t0),
                    reads=vr(t1) + vr(t0), writes=vr(val), tag=tag)
            # stages 2+3 fused: val = (33*val + (C2+C3)) ^ (33*512*val + 512*C2)
            emit_op("valu", ("multiply_add", t0, val, v_m33, v_fadd),
                    reads=vr(val) + vr(v_m33) + vr(v_fadd), writes=vr(t0), tag=tag)
            emit_op("valu", ("multiply_add", t1, val, v_m2s, v_sadd),
                    reads=vr(val) + vr(v_m2s) + vr(v_sadd), writes=vr(t1), tag=tag)
            emit_op("valu", ("^", val, t0, t1),
                    reads=vr(t0) + vr(t1), writes=vr(val), tag=tag)
            # stage 4: val = 9*val + C4
            emit_op("valu", ("multiply_add", val, val, v_m9, v_hc4),
                    reads=vr(val) + vr(v_m9) + vr(v_hc4), writes=vr(val), tag=tag)
            # stage 5: val = (val ^ C5) ^ (val >> 16)
            emit_op("valu", (">>", t0, val, v_s5),
                    reads=vr(val) + vr(v_s5), writes=vr(t0), tag=tag)
            emit_op("valu", ("^", t1, val, v_hc5),
                    reads=vr(val) + vr(v_hc5), writes=vr(t1), tag=tag)
            emit_op("valu", ("^", val, t1, t0),
                    reads=vr(t1) + vr(t0), writes=vr(val), tag=tag)

        def emit_round(g, rnd, tag):
            depth = rnd % period
            val = v_val[g]
            off = v_off[g]
            t0, t1 = v_t[g]
            t2 = v_pool[g % NPOOL]

            # ----- combine node value into val -----
            # Node selection for shallow (preloaded) depths uses the FLOW engine
            # (vselect) so the valu engine is reserved for hashing.
            if depth == 0:
                emit_op("valu", ("^", val, val, v_leaf[0]),
                        reads=vr(val) + vr(v_leaf[0]), writes=vr(val), tag=tag)
            elif depth == 1:
                # off in {0,1}: node = off ? leaf2 : leaf1
                emit_op("flow", ("vselect", t0, off, v_leaf[2], v_leaf[1]),
                        reads=vr(off) + vr(v_leaf[2]) + vr(v_leaf[1]), writes=vr(t0), tag=tag)
                emit_op("valu", ("^", val, val, t0),
                        reads=vr(val) + vr(t0), writes=vr(val), tag=tag)
            elif depth == 2:
                # off in {0,1,2,3}: 4-way vselect tree on bits b0, b1
                emit_op("valu", ("&", t0, off, v_one),
                        reads=vr(off) + vr(v_one), writes=vr(t0), tag=tag)          # b0
                emit_op("valu", (">>", t1, off, v_one),
                        reads=vr(off) + vr(v_one), writes=vr(t1), tag=tag)          # b1
                emit_op("flow", ("vselect", t2, t0, v_leaf[4], v_leaf[3]),
                        reads=vr(t0) + vr(v_leaf[4]) + vr(v_leaf[3]), writes=vr(t2), tag=tag)  # lo
                emit_op("flow", ("vselect", t0, t0, v_leaf[6], v_leaf[5]),
                        reads=vr(t0) + vr(v_leaf[6]) + vr(v_leaf[5]), writes=vr(t0), tag=tag)  # hi
                emit_op("flow", ("vselect", t2, t1, t0, t2),
                        reads=vr(t1) + vr(t0) + vr(t2), writes=vr(t2), tag=tag)     # node
                emit_op("valu", ("^", val, val, t2),
                        reads=vr(val) + vr(t2), writes=vr(val), tag=tag)
            else:
                # gather: addr = base_d + off ; then 8 scalar-lane loads
                node = t1
                emit_op("valu", ("+", t0, v_base[depth], off),
                        reads=vr(v_base[depth]) + vr(off), writes=vr(t0), tag=tag)
                for lane in range(VLEN):
                    emit_op("load", ("load_offset", node, t0, lane),
                            reads=[t0 + lane], writes=[node + lane], tag=tag)
                emit_op("valu", ("^", val, val, node),
                        reads=vr(val) + vr(node), writes=vr(val), tag=tag)

            # ----- hash -----
            hash_ops(g, tag)

            # ----- offset (path) update for the next round -----
            if rnd < rounds - 1 and depth != period - 1:
                if depth == 0:
                    # root offset is 0 -> off_next = val & 1
                    emit_op("valu", ("&", off, val, v_one),
                            reads=vr(val) + vr(v_one), writes=vr(off), tag=tag)
                else:
                    emit_op("valu", ("&", t0, val, v_one),
                            reads=vr(val) + vr(v_one), writes=vr(t0), tag=tag)
                    emit_op("valu", ("multiply_add", off, off, v_two, t0),
                            reads=vr(off) + vr(v_two) + vr(t0), writes=vr(off), tag=tag)

        # Emit in group-major order so the scheduler can pipeline groups at
        # staggered rounds: a group stalled on gather loads yields valu slots to
        # another group doing preload/hash work, keeping both engines busy.
        for g in range(VG):
            # Load this group's initial values (offset starts at 0 -> derived).
            emit_op("load", ("vload", v_val[g], s_addr_val[g]),
                    reads=[s_addr_val[g]], writes=vr(v_val[g]), tag=g)
            for rnd in range(rounds):
                emit_round(g, rnd, tag=g)

        # ----- store final values back to memory -----
        # The VG groups cover all `batch_size` elements in a single pass, so
        # there is no outer batch loop (no loop control / address bumps needed).
        for g in range(VG):
            emit_op("store", ("vstore", s_addr_val[g], v_val[g]),
                    reads=[s_addr_val[g]] + vr(v_val[g]), writes=[])

        scheduled = self._schedule(ops, band_size=max(1, VG // 8))
        for b in scheduled:
            for eng, slots in b.items():
                for slot in slots:
                    asm.add(eng, slot)
            asm.emit()

        self.instrs.extend(asm.instrs)

    # ------------------------------------------------------------------
    @staticmethod
    def _schedule(ops, band_size=4):
        n = len(ops)
        last_writer = {}
        readers = defaultdict(list)
        parents = [set() for _ in range(n)]

        for i, op in enumerate(ops):
            for r in op["reads"]:
                if r in last_writer:
                    parents[i].add(last_writer[r])
            for w in op["writes"]:
                if w in last_writer:
                    parents[i].add(last_writer[w])
                for ri in readers[w]:
                    parents[i].add(ri)
            for r in op["reads"]:
                readers[r].append(i)
            for w in op["writes"]:
                last_writer[w] = i
                readers[w] = []

        children = [set() for _ in range(n)]
        for i in range(n):
            for p in parents[i]:
                children[p].add(i)

        indeg = [len(parents[i]) for i in range(n)]
        q = deque(i for i in range(n) if indeg[i] == 0)
        topo = []
        while q:
            u = q.popleft()
            topo.append(u)
            for c in children[u]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    q.append(c)
        heights = [0] * n
        for u in reversed(topo):
            for c in children[u]:
                heights[u] = max(heights[u], heights[c] + 1)

        # ASAP depth (longest path from a source) for slack computation.
        asap = [0] * n
        for u in topo:
            for c in children[u]:
                asap[c] = max(asap[c], asap[u] + 1)
        cp = max(heights) if n else 0
        slack = [cp - (asap[i] + heights[i]) for i in range(n)]

        # Group ops into "bands" so a handful of groups are raced ahead into
        # gather (load) rounds while later groups still do preload/hash (valu)
        # work, keeping both engines busy. A band of ~VG/8 finishes groups in
        # small batches, which overlaps their pipeline drains. Within a band we
        # order by critical-path slack (most urgent first).
        band = max(1, band_size)

        indeg = [len(parents[i]) for i in range(n)]
        ready = [i for i in range(n) if indeg[i] == 0]
        sched_cycle = [-1] * n
        bundles = []
        cyc = 0
        while ready:
            avail = [i for i in ready if all(sched_cycle[p] < cyc for p in parents[i])]

            def prio(i):
                eng = ops[i]["engine"]
                ep = 2 if eng == "load" else (1 if eng == "store" else 0)
                return (-(ops[i]["tag"] // band), -slack[i], ep)

            avail.sort(key=prio, reverse=True)
            bundle = defaultdict(list)
            chosen = []
            for i in avail:
                eng = ops[i]["engine"]
                if len(bundle[eng]) < SLOT_LIMITS.get(eng, 64):
                    bundle[eng].append(ops[i]["slot"])
                    chosen.append(i)
            for i in chosen:
                ready.remove(i)
                sched_cycle[i] = cyc
                for c in children[i]:
                    indeg[c] -= 1
                    if indeg[c] == 0:
                        ready.append(c)
            bundles.append(dict(bundle) if bundle else {})
            cyc += 1
        return bundles
