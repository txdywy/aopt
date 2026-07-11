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
        import os as _os
        VG = batch_size // VLEN  # total vector groups if all-vector
        assert VG * VLEN == batch_size, "batch_size must be a multiple of VLEN"

        # ---- dual-engine split -------------------------------------------
        # The valu (vector) engine is the bottleneck; the scalar ALU engine
        # (12 slots) is otherwise idle. So we peel off NSCALAR elements and
        # process them entirely on the ALU engine, in parallel with the vector
        # groups, cutting the makespan below the pure-vector floor.
        # Offloading to the scalar engine only pays off once the vector engine
        # is the bottleneck (large batches). For smaller batches the scalar
        # path's per-element overhead dominates, so stay pure-vector.
        default_ns = 24 if batch_size >= 256 else 0
        NSCALAR = int(_os.environ.get("NSCALAR", str(default_ns)))
        NSCALAR = max(0, min(NSCALAR, batch_size - VLEN))
        while NSCALAR and (batch_size - NSCALAR) % VLEN != 0:
            NSCALAR -= 1
        VG_v = (batch_size - NSCALAR) // VLEN
        scalar_base = VG_v * VLEN  # first element index handled by scalar path
        period = forest_height + 1
        # Preload depth: shallow levels (0..PRELOAD-1) are selected from
        # registers on the FLOW engine (vselect) with no memory gather. Deeper
        # levels are gathered. Preloading depth 3 cuts the gather rounds from 10
        # to 8, moving the bottleneck off the load engine onto valu.
        PRELOAD = min(4, period)

        asm = Assembler()

        # ---------------- scalar scratch ----------------
        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                     "forest_values_p", "inp_indices_p", "inp_values_p"]
        s_vars = {v: self.alloc_scratch(v) for v in init_vars}

        s_zero = self.alloc_scratch("s_zero")

        s_addr_val = [self.alloc_scratch(f"s_addr_val_{g}") for g in range(VG_v)]

        # ---------------- vector constant regs ----------------
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_four = self.alloc_scratch("v_four", VLEN)  # bit-2 mask for depth-3

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

        # Preloaded leaf values (broadcast). Node selection for preloaded depths
        # is done on the otherwise-idle FLOW engine via vselect, so we keep the
        # raw leaves as vector constants. Leaves 0..(2^PRELOAD - 2) are needed.
        n_leaves_v = (1 << PRELOAD) - 1
        v_leaf = [self.alloc_scratch(f"v_leaf{i}", VLEN) for i in range(n_leaves_v)]

        # base_d = forest_values_p + (2^d - 1) for gather depths
        gather_depths = [d for d in range(PRELOAD, period)]
        v_base = {d: self.alloc_scratch(f"v_base_{d}", VLEN) for d in gather_depths}

        # ---------------- per-group working regs ----------------
        v_val = [self.alloc_scratch(f"v_val_{g}", VLEN) for g in range(VG_v)]
        v_off = [self.alloc_scratch(f"v_off_{g}", VLEN) for g in range(VG_v)]
        # 2 private temps per group; a shared pool supplies the extra temps
        # needed by the depth-2 (1 extra) and depth-3 (2 extra) selects. The
        # pool is used as pairs indexed by g % NPAIR so groups that share a pair
        # are kept temporally apart by the banded scheduler.
        v_t = [[self.alloc_scratch(f"v_t{j}_{g}", VLEN) for j in range(2)]
               for g in range(VG_v)]
        NPAIR = 7
        v_pool = [self.alloc_scratch(f"v_pool_{i}", VLEN) for i in range(2 * NPAIR)]

        # ---------------- scalar-engine working regs ----------------
        # NSCALAR elements processed on the ALU engine. Each keeps a value and
        # an offset; a shared temp pool (2 per concurrent element) supports the
        # hash chain and selects.
        s_sval = [self.alloc_scratch(f"s_sval_{j}") for j in range(NSCALAR)]
        s_soff = [self.alloc_scratch(f"s_soff_{j}") for j in range(NSCALAR)]
        SPAIR = NSCALAR  # one private temp pair per scalar element (no WAR sharing)
        s_tpool = [self.alloc_scratch(f"s_tp_{i}") for i in range(2 * SPAIR)]
        s_saddr = [self.alloc_scratch(f"s_sa_{i}") for i in range(6 if NSCALAR else 0)]
        if NSCALAR:
            s_diff1 = self.alloc_scratch("s_diff1")
            s_d34 = self.alloc_scratch("s_d34")
            s_d56 = self.alloc_scratch("s_d56")

        assert self.scratch_ptr <= SCRATCH_SIZE, (
            f"scratch overflow {self.scratch_ptr}"
        )

        # ---------------- setup scratch temps ----------------
        n_leaf_scalar = min((1 << PRELOAD) - 1, n_nodes)
        s_leaf = [self.alloc_scratch(f"s_leaf_{i}") for i in range(n_leaves_v)]
        s_hdr = [self.alloc_scratch(f"s_hdr_{i}") for i in range(7)]
        NLA = 4  # round-robin leaf-address temps (setup only)
        s_leafaddr = [self.alloc_scratch(f"s_la_{i}") for i in range(NLA)]
        hc_tmp = [self.alloc_scratch(f"s_hc_{i}") for i in range(len(hash_consts))]
        # The scalar path reuses the hash-constant scalars (hc_tmp holds the
        # constant values, written once and read-only thereafter). Indices match
        # the hash_consts list order.
        (s_hc0, s_hm4097, s_hc1, s_hs1, s_hm33, s_hfadd,
         s_hm2s, s_hsadd, s_hm9, s_hc4, s_hc5, s_hs5) = hc_tmp
        s_one = self.alloc_scratch("s_one")
        s_two = self.alloc_scratch("s_two")
        s_four = self.alloc_scratch("s_four")
        # scalar path gathers from depth PS..period-1 (PS<=PRELOAD), so it may
        # need base registers for depths the vector path preloads.
        PS = min(3, period)
        base_depths = sorted(set(gather_depths) | (set(range(PS, period)) if NSCALAR else set()))
        s_base = {d: self.alloc_scratch(f"s_base_{d}") for d in base_depths}
        pow_regs = {}
        k = 0
        while (1 << k) < VG_v:
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
        emit_op("load", ("const", s_four, 4), reads=[], writes=[s_four], tag=SETUP)
        emit_op("valu", ("vbroadcast", v_one, s_one), reads=[s_one], writes=vr(v_one), tag=SETUP)
        emit_op("valu", ("vbroadcast", v_two, s_two), reads=[s_two], writes=vr(v_two), tag=SETUP)
        emit_op("valu", ("vbroadcast", v_four, s_four), reads=[s_four], writes=vr(v_four), tag=SETUP)

        fvp = s_vars["forest_values_p"]
        # ----- preloaded leaves (scalar loads) then broadcast -----
        emit_op("load", ("load", s_leaf[0], fvp), reads=[fvp], writes=[s_leaf[0]], tag=SETUP)
        for i in range(1, n_leaves_v):
            la = s_leafaddr[i % NLA]
            emit_op("flow", ("add_imm", la, fvp, i),
                    reads=[fvp], writes=[la], tag=SETUP)
            emit_op("load", ("load", s_leaf[i], la),
                    reads=[la], writes=[s_leaf[i]], tag=SETUP)
        for i in range(n_leaves_v):
            emit_op("valu", ("vbroadcast", v_leaf[i], s_leaf[i]),
                    reads=[s_leaf[i]], writes=vr(v_leaf[i]), tag=SETUP)
        # scalar leaf diffs for arithmetic (branchless) selects on the ALU engine
        if NSCALAR:
            emit_op("alu", ("-", s_diff1, s_leaf[2], s_leaf[1]),
                    reads=[s_leaf[2], s_leaf[1]], writes=[s_diff1], tag=SETUP)
            emit_op("alu", ("-", s_d34, s_leaf[4], s_leaf[3]),
                    reads=[s_leaf[4], s_leaf[3]], writes=[s_d34], tag=SETUP)
            emit_op("alu", ("-", s_d56, s_leaf[6], s_leaf[5]),
                    reads=[s_leaf[6], s_leaf[5]], writes=[s_d56], tag=SETUP)

        # ----- hash constants -----
        for i, (vreg, val) in enumerate(hash_consts):
            emit_op("load", ("const", hc_tmp[i], val), reads=[], writes=[hc_tmp[i]], tag=SETUP)
            emit_op("valu", ("vbroadcast", vreg, hc_tmp[i]),
                    reads=[hc_tmp[i]], writes=vr(vreg), tag=SETUP)

        # ----- base_d = forest_values_p + (2^d - 1) (scalar), broadcast for vector -----
        for d in base_depths:
            emit_op("flow", ("add_imm", s_base[d], fvp, (1 << d) - 1),
                    reads=[fvp], writes=[s_base[d]], tag=SETUP)
        for d in gather_depths:
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
        while (1 << k) < VG_v:
            half = 1 << k
            for g in range(half, min(2 * half, VG_v)):
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
            t2 = v_pool[(g % NPAIR) * 2]
            t3 = v_pool[(g % NPAIR) * 2 + 1]

            # ----- combine node value into val -----
            # Node selection for shallow (preloaded) depths uses the FLOW engine
            # (vselect) so the valu engine is reserved for hashing. `off` is only
            # read here (never clobbered), so it survives for the offset update.
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
                # off in {0..3}: 4-way vselect tree on bits b0, b1
                emit_op("valu", ("&", t0, off, v_one),
                        reads=vr(off) + vr(v_one), writes=vr(t0), tag=tag)          # b0
                emit_op("valu", ("&", t1, off, v_two),
                        reads=vr(off) + vr(v_two), writes=vr(t1), tag=tag)          # b1 (!=0)
                emit_op("flow", ("vselect", t2, t0, v_leaf[4], v_leaf[3]),
                        reads=vr(t0) + vr(v_leaf[4]) + vr(v_leaf[3]), writes=vr(t2), tag=tag)  # lo
                emit_op("flow", ("vselect", t0, t0, v_leaf[6], v_leaf[5]),
                        reads=vr(t0) + vr(v_leaf[6]) + vr(v_leaf[5]), writes=vr(t0), tag=tag)  # hi
                emit_op("flow", ("vselect", t2, t1, t0, t2),
                        reads=vr(t1) + vr(t0) + vr(t2), writes=vr(t2), tag=tag)     # node
                emit_op("valu", ("^", val, val, t2),
                        reads=vr(val) + vr(t2), writes=vr(val), tag=tag)
            elif depth == 3:
                # off in {0..7}: 8-way vselect tree on bits b0,b1,b2 (masks 1/2/4)
                emit_op("valu", ("&", t0, off, v_one),
                        reads=vr(off) + vr(v_one), writes=vr(t0), tag=tag)          # b0
                emit_op("valu", ("&", t1, off, v_two),
                        reads=vr(off) + vr(v_two), writes=vr(t1), tag=tag)          # b1 (!=0)
                emit_op("flow", ("vselect", t2, t0, v_leaf[8], v_leaf[7]),
                        reads=vr(t0) + vr(v_leaf[8]) + vr(v_leaf[7]), writes=vr(t2), tag=tag)   # lo0
                emit_op("flow", ("vselect", t3, t0, v_leaf[10], v_leaf[9]),
                        reads=vr(t0) + vr(v_leaf[10]) + vr(v_leaf[9]), writes=vr(t3), tag=tag)  # lo1
                emit_op("flow", ("vselect", t2, t1, t3, t2),
                        reads=vr(t1) + vr(t3) + vr(t2), writes=vr(t2), tag=tag)     # m0
                emit_op("flow", ("vselect", t3, t0, v_leaf[12], v_leaf[11]),
                        reads=vr(t0) + vr(v_leaf[12]) + vr(v_leaf[11]), writes=vr(t3), tag=tag)  # lo2
                emit_op("flow", ("vselect", t0, t0, v_leaf[14], v_leaf[13]),
                        reads=vr(t0) + vr(v_leaf[14]) + vr(v_leaf[13]), writes=vr(t0), tag=tag)  # lo3 (b0 consumed)
                emit_op("flow", ("vselect", t3, t1, t0, t3),
                        reads=vr(t1) + vr(t0) + vr(t3), writes=vr(t3), tag=tag)     # m1
                emit_op("valu", ("&", t0, off, v_four),
                        reads=vr(off) + vr(v_four), writes=vr(t0), tag=tag)         # b2 (!=0)
                emit_op("flow", ("vselect", t2, t0, t3, t2),
                        reads=vr(t0) + vr(t3) + vr(t2), writes=vr(t2), tag=tag)     # node
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
        for g in range(VG_v):
            # Load this group's initial values (offset starts at 0 -> derived).
            emit_op("load", ("vload", v_val[g], s_addr_val[g]),
                    reads=[s_addr_val[g]], writes=vr(v_val[g]), tag=g)
            for rnd in range(rounds):
                emit_round(g, rnd, tag=g)

        # ----- store vector results back to memory -----
        for g in range(VG_v):
            emit_op("store", ("vstore", s_addr_val[g], v_val[g]),
                    reads=[s_addr_val[g]] + vr(v_val[g]), writes=[])

        # ================= SCALAR ENGINE PATH =================
        # Elements scalar_base..batch_size-1 run entirely on the ALU engine.
        # Scalar preloads depths 0/1/2 (selects on the flow engine) and gathers
        # depths >=3 (one scalar load each), mirroring the reference exactly.
        ivp = s_vars["inp_values_p"]
        for j in range(NSCALAR):
            e = scalar_base + j
            sv = s_sval[j]; so = s_soff[j]
            st0 = s_tpool[(j % SPAIR) * 2]
            st1 = s_tpool[(j % SPAIR) * 2 + 1]
            sa = s_saddr[j % len(s_saddr)]
            # Stagger scalar elements across the vector band range so their
            # gather-load demand is spread over time (avoids load-engine storms)
            # while keeping enough chains concurrently active to feed the ALU.
            stag = (j * VG_v) // max(1, NSCALAR)
            # load initial value
            emit_op("flow", ("add_imm", sa, ivp, e),
                    reads=[ivp], writes=[sa], tag=stag)
            emit_op("load", ("load", sv, sa), reads=[sa], writes=[sv], tag=stag)
            for rnd in range(rounds):
                d = rnd % period
                # ---- node combine ----
                if d == 0:
                    emit_op("alu", ("^", sv, sv, s_leaf[0]),
                            reads=[sv, s_leaf[0]], writes=[sv], tag=stag)
                elif d == 1:
                    # node = leaf1 + off*(leaf2-leaf1)  (off in {0,1}), all on ALU
                    emit_op("alu", ("*", st0, so, s_diff1), reads=[so, s_diff1], writes=[st0], tag=stag)
                    emit_op("alu", ("+", st0, st0, s_leaf[1]), reads=[st0, s_leaf[1]], writes=[st0], tag=stag)
                    emit_op("alu", ("^", sv, sv, st0), reads=[sv, st0], writes=[sv], tag=stag)
                elif d == 2:
                    # 4-way arithmetic select on ALU: b0=off&1, b1=off>>1
                    emit_op("alu", ("&", st0, so, s_one), reads=[so, s_one], writes=[st0], tag=stag)  # b0
                    emit_op("alu", ("*", sa, st0, s_d34), reads=[st0, s_d34], writes=[sa], tag=stag)
                    emit_op("alu", ("+", sa, sa, s_leaf[3]), reads=[sa, s_leaf[3]], writes=[sa], tag=stag)  # lo
                    emit_op("alu", ("*", st1, st0, s_d56), reads=[st0, s_d56], writes=[st1], tag=stag)
                    emit_op("alu", ("+", st1, st1, s_leaf[5]), reads=[st1, s_leaf[5]], writes=[st1], tag=stag)  # hi
                    emit_op("alu", (">>", st0, so, s_one), reads=[so, s_one], writes=[st0], tag=stag)  # b1
                    emit_op("alu", ("-", st1, st1, sa), reads=[st1, sa], writes=[st1], tag=stag)  # hi-lo
                    emit_op("alu", ("*", st0, st0, st1), reads=[st0, st1], writes=[st0], tag=stag)
                    emit_op("alu", ("+", st0, st0, sa), reads=[st0, sa], writes=[st0], tag=stag)  # node
                    emit_op("alu", ("^", sv, sv, st0), reads=[sv, st0], writes=[sv], tag=stag)
                else:
                    emit_op("alu", ("+", sa, s_base[d], so), reads=[s_base[d], so], writes=[sa], tag=stag)
                    emit_op("load", ("load", st0, sa), reads=[sa], writes=[st0], tag=stag)
                    emit_op("alu", ("^", sv, sv, st0), reads=[sv, st0], writes=[sv], tag=stag)
                # ---- hash (scalar, no multiply_add) ----
                # stage 0: sv = 4097*sv + C0
                emit_op("alu", ("*", sv, sv, s_hm4097), reads=[sv, s_hm4097], writes=[sv], tag=stag)
                emit_op("alu", ("+", sv, sv, s_hc0), reads=[sv, s_hc0], writes=[sv], tag=stag)
                # stage 1: sv = (sv ^ C1) ^ (sv >> 19)
                emit_op("alu", (">>", st0, sv, s_hs1), reads=[sv, s_hs1], writes=[st0], tag=stag)
                emit_op("alu", ("^", st1, sv, s_hc1), reads=[sv, s_hc1], writes=[st1], tag=stag)
                emit_op("alu", ("^", sv, st1, st0), reads=[st1, st0], writes=[sv], tag=stag)
                # stages 2+3 fused: sv = (33*sv + (C2+C3)) ^ (33*512*sv + 512*C2)
                emit_op("alu", ("*", st0, sv, s_hm33), reads=[sv, s_hm33], writes=[st0], tag=stag)
                emit_op("alu", ("+", st0, st0, s_hfadd), reads=[st0, s_hfadd], writes=[st0], tag=stag)
                emit_op("alu", ("*", st1, sv, s_hm2s), reads=[sv, s_hm2s], writes=[st1], tag=stag)
                emit_op("alu", ("+", st1, st1, s_hsadd), reads=[st1, s_hsadd], writes=[st1], tag=stag)
                emit_op("alu", ("^", sv, st0, st1), reads=[st0, st1], writes=[sv], tag=stag)
                # stage 4: sv = 9*sv + C4
                emit_op("alu", ("*", sv, sv, s_hm9), reads=[sv, s_hm9], writes=[sv], tag=stag)
                emit_op("alu", ("+", sv, sv, s_hc4), reads=[sv, s_hc4], writes=[sv], tag=stag)
                # stage 5: sv = (sv ^ C5) ^ (sv >> 16)
                emit_op("alu", (">>", st0, sv, s_hs5), reads=[sv, s_hs5], writes=[st0], tag=stag)
                emit_op("alu", ("^", st1, sv, s_hc5), reads=[sv, s_hc5], writes=[st1], tag=stag)
                emit_op("alu", ("^", sv, st1, st0), reads=[st1, st0], writes=[sv], tag=stag)
                # ---- offset update ----
                if rnd < rounds - 1 and d != period - 1:
                    if d == 0:
                        emit_op("alu", ("&", so, sv, s_one), reads=[sv, s_one], writes=[so], tag=stag)
                    else:
                        emit_op("alu", ("&", st0, sv, s_one), reads=[sv, s_one], writes=[st0], tag=stag)
                        emit_op("alu", ("<<", so, so, s_one), reads=[so, s_one], writes=[so], tag=stag)
                        emit_op("alu", ("+", so, so, st0), reads=[so, st0], writes=[so], tag=stag)
            # store scalar result
            emit_op("flow", ("add_imm", sa, ivp, e), reads=[ivp], writes=[sa], tag=stag)
            emit_op("store", ("store", sa, sv), reads=[sa, sv], writes=[], tag=stag)

        scheduled = self._schedule(ops, band_size=1)
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
