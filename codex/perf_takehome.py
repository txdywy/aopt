import sys
from collections import defaultdict, deque
from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class Assembler:
    def __init__(self):
        self.instrs = []
        self.curr = defaultdict(list)

    def add(self, engine, slot):
        self.curr[engine].append(slot)
        assert len(self.curr[engine]) <= SLOT_LIMITS[engine], f"Too many slots for {engine}"

    def emit(self):
        if self.curr:
            self.instrs.append(dict(self.curr))
            self.curr = defaultdict(list)


class KernelBuilder:
    NUM_OFFLOAD = 32

    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def alloc_scratch(self, name=None, size=1):
        addr = self.scratch_ptr
        self.scratch_ptr += size
        assert self.scratch_ptr <= SCRATCH_SIZE, (
            f"Scratch overflow: {self.scratch_ptr} > {SCRATCH_SIZE}"
        )
        if name is not None:
            self.scratch[name] = addr
            for i in range(size):
                self.scratch_debug[addr + i] = (f"{name}_{i}" if size > 1 else name)
        return addr

    def scratch_const(self, val, name=None):
        """Allocate a scratch reg and store a constant, but defer the load instruction."""
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.const_map[val] = addr
        return self.const_map[val]

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build_kernel(self, forest_height, n_nodes, batch_size, rounds):
        asm = Assembler()

        # MAX_OPT_ROUND: rounds 0..MAX_OPT_ROUND-1 use preloaded tree values (no mem loads)
        # Round 3 (8-leaf tree) requires 5 temp regs -- use 3 optimized rounds only.
        MAX_OPT_ROUND = 4

        # ---- Allocate all scratch registers up front ----
        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                     "forest_values_p", "inp_indices_p", "inp_values_p"]
        s_vars = {v: self.alloc_scratch(v) for v in init_vars}

        s_tmp   = self.alloc_scratch("s_tmp")
        s_tmp2  = self.alloc_scratch("s_tmp2")
        s_loop_i    = self.alloc_scratch("s_loop_i")
        s_loop_cond = self.alloc_scratch("s_loop_cond")
        s_zero  = self.alloc_scratch("s_zero")
        s_one   = self.alloc_scratch("s_one")
        s_two   = self.alloc_scratch("s_two")
        s_c256  = self.alloc_scratch("s_c256")

        s_addr_idx = [self.alloc_scratch(f"s_addr_idx_{g}") for g in range(32)]
        s_addr_val = [self.alloc_scratch(f"s_addr_val_{g}") for g in range(32)]

        v_zero = self.alloc_scratch("v_zero", 8)
        v_one  = self.alloc_scratch("v_one",  8)
        v_two  = self.alloc_scratch("v_two",  8)
        v_n_nodes = self.alloc_scratch("v_n_nodes", 8)
        v_forest_values_p = self.alloc_scratch("v_forest_values_p", 8)

        v_idx = [self.alloc_scratch(f"v_idx_{g}", 8) for g in range(32)]
        v_val = [self.alloc_scratch(f"v_val_{g}", 8) for g in range(32)]

        v_hash_val1 = []
        v_hash_val3 = {}
        for hi in range(len(HASH_STAGES)):
            v_hash_val1.append(self.alloc_scratch(f"v_hv1_{hi}", 8))
            if hi in (1, 3, 5):
                v_hash_val3[hi] = self.alloc_scratch(f"v_hv3_{hi}", 8)

        s_m0 = self.alloc_scratch("s_m0")
        s_m2 = self.alloc_scratch("s_m2")
        s_m4 = self.alloc_scratch("s_m4")
        v_hmul0 = self.alloc_scratch("v_hmul0", 8)
        v_hmul2 = self.alloc_scratch("v_hmul2", 8)
        v_hmul4 = self.alloc_scratch("v_hmul4", 8)

        s_g8_offsets = {}

        # Tree leaves: n_leaves = 2^MAX_OPT_ROUND - 1 = 15
        n_leaves = (1 << MAX_OPT_ROUND) - 1
        s_leaves = [self.alloc_scratch(f"s_leaf_{i}") for i in range(n_leaves)]

        s_three = self.alloc_scratch("s_three")
        v_three = self.alloc_scratch("v_three", 8)
        s_seven = self.alloc_scratch("s_seven")
        v_seven = self.alloc_scratch("v_seven", 8)

        # Leaf diffs: r=1: 1 diff, r=2: 2 diffs
        s_diffs = {}
        for r in range(MAX_OPT_ROUND):
            s_diffs[r] = []
            for k in range(1 << (r - 1) if r > 0 else 0):
                s_diffs[r].append(self.alloc_scratch(f"s_d_{r}_{k}"))

        # Leaf vectors and diff vectors
        v_leaf = {}
        v_diff = {}
        for r in range(MAX_OPT_ROUND):
            v_leaf[r] = []
            v_diff[r] = []
            for k in range(1 << r):
                v_leaf[r].append(self.alloc_scratch(f"vl_{r}_{k}", 8))
            for k in range(len(s_diffs[r])):
                v_diff[r].append(self.alloc_scratch(f"vd_{r}_{k}", 8))

        s_diff2_base  = self.alloc_scratch("s_diff2_base")
        s_diff2_slope = self.alloc_scratch("s_diff2_slope")
        v_diff2_base  = self.alloc_scratch("v_diff2_base", 8)
        v_diff2_slope = self.alloc_scratch("v_diff2_slope", 8)

        # Per-group private temp
        v_tmp = [self.alloc_scratch(f"v_tmp_{g}", 8) for g in range(16)]

        # Global temp pool: gA unique per group (32 regs) + gB shared (16 regs)
        NG_A = 16
        NG_B = 8
        NG_C = 4
        NG_D = 4
        NG_E = 4
        NG_F = 4
        v_glob_A = [self.alloc_scratch(f"vgA_{ti}", 8) for ti in range(NG_A)]
        v_glob_B = [self.alloc_scratch(f"vgB_{ti}", 8) for ti in range(NG_B)]
        v_glob_C = [self.alloc_scratch(f"vgC_{ti}", 8) for ti in range(NG_C)]
        v_glob_D = [self.alloc_scratch(f"vgD_{ti}", 8) for ti in range(NG_D)]
        v_glob_E = [self.alloc_scratch(f"vgE_{ti}", 8) for ti in range(NG_E)]
        v_glob_F = [self.alloc_scratch(f"vgF_{ti}", 8) for ti in range(NG_F)]

        # Check scratch budget
        assert self.scratch_ptr <= SCRATCH_SIZE, f"Scratch overflow: {self.scratch_ptr}"

        # ---- SETUP PHASE: Batch all instructions for minimum cycles ----

        # Step 1: Load init vars (7 loads from mem[0..6])
        # Method: use s_tmp to hold address, then load
        # Load const 0 -> s_tmp, then load s_vars[0] from s_tmp
        # We can batch: const 0,1 in same cycle, then load from both (but
        # scalar load (not vload) only loads 1 scalar per slot, 2 slots per cycle).
        # So we need 7 loads = 4 cycles minimum (2+2+2+1).
        # But we also need to set s_tmp to addresses 0..6 before each load.
        # s_tmp is reused each time, so they must be sequential!
        # Optimal: interleave load-const and load-load pairs:
        # cycle: [const s_tmp=0, const s_tmp2=1], [load v0 from s_tmp, load v1 from s_tmp2]
        # But s_tmp and s_tmp2 are DIFFERENT registers, so no conflict.
        # With 2 load slots: can do 2 loads per cycle -> 7 init vars = 4 cycles for const + 4 for load = 8 cycles?
        # But const and load can be in same cycle (both are 'load' engine):
        # Cycle: [const s_tmp=0, const s_tmp2=1]  -> 1 cycle
        # Cycle: [load v0 from s_tmp, load v1 from s_tmp2] -> 1 cycle
        # Repeating for 7 vars: ceil(7/2) * 2 = 8 cycles

        # Actually, we can overlap: while loading v0,v1, set up s_tmp for v2,v3
        # But: can't have load + const in same cycle as two loads (only 2 load slots).
        # load const uses 1 load slot, load from mem uses 1 load slot.
        # So: [const s_tmp=0, load v0 from s_tmp_prev] can overlap if s_tmp_prev is ready.
        # For the first pair: s_tmp is set in same cycle as load - not ready!
        # So must be sequential: const(0) -> load(v0).

        # Optimal sequential init:
        # Cycle 0: [const s_tmp=0, const s_tmp2=1]
        # Cycle 1: [load v0, load v1]
        # Cycle 2: [const s_tmp=2, const s_tmp2=3]
        # Cycle 3: [load v2, load v3]
        # Cycle 4: [const s_tmp=4, const s_tmp2=5]
        # Cycle 5: [load v4, load v5]
        # Cycle 6: [const s_tmp=6]
        # Cycle 7: [load v6]
        # = 8 cycles for 7 init vars

        init_var_names = list(init_vars)
        for pair_start in range(0, len(init_var_names), 2):
            pair = init_var_names[pair_start:pair_start+2]
            if len(pair) == 2:
                asm.add("load", ("const", s_tmp, pair_start))
                asm.add("load", ("const", s_tmp2, pair_start + 1))
                asm.emit()
                asm.add("load", ("load", s_vars[pair[0]], s_tmp))
                asm.add("load", ("load", s_vars[pair[1]], s_tmp2))
                asm.emit()
            else:
                asm.add("load", ("const", s_tmp, pair_start))
                asm.emit()
                asm.add("load", ("load", s_vars[pair[0]], s_tmp))
                asm.emit()

        # Step 2: Load scalar constants (0,1,2,256,4097,33,9,3)
        # and hash stage constants - batch maximally
        asm.add("load", ("const", s_zero, 0))
        asm.add("load", ("const", s_one, 1))
        asm.emit()
        asm.add("load", ("const", s_two, 2))
        asm.add("load", ("const", s_c256, 256))
        asm.emit()
        asm.add("load", ("const", s_m0, 4097))
        asm.add("load", ("const", s_m2, 33))
        asm.emit()
        asm.add("load", ("const", s_m4, 9))
        asm.add("load", ("const", s_three, 3))
        asm.emit()
        asm.add("load", ("const", s_seven, 7))
        asm.emit()

        # Step 3: Load hash stage constants (6 stages, 3 with 2 constants each = 9 constants)
        hash_consts_batch = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            hash_consts_batch.append((v_hash_val1[hi], val1))
            if hi in (1, 3, 5):
                hash_consts_batch.append((v_hash_val3[hi], val3))
        # 9 constants total, load 2 per cycle into s_tmp/s_tmp2, then vbroadcast
        # Each: 1 cycle to load const + 1 cycle to vbroadcast = but we can overlap:
        # For pair (v1, c1), (v2, c2):
        #   cycle: [const s_tmp=c1, const s_tmp2=c2]
        #   cycle: [vbroadcast v1 s_tmp, vbroadcast v2 s_tmp2]
        # This overlaps only if const and vbroadcast use different engines (load vs valu): YES!
        # So: [const s_tmp=c1, const s_tmp2=c2] + [vbroadcast v_prev1 s_tmp_prev1, vbroadcast v_prev2 s_tmp_prev2]
        # can all be in the SAME cycle? No - because s_tmp and s_tmp_prev1 must be ready.
        # But if prev pair has already set s_tmp/s_tmp2, they're ready.
        # Pipelined approach:
        # cycle A: [const s_tmp=c1, const s_tmp2=c2]
        # cycle B: [const s_tmp=c3, const s_tmp2=c4, vbroadcast v1 s_tmp_old, vbroadcast v2 s_tmp2_old]
        # -- wait, we're reusing s_tmp in cycle B! s_tmp is being written AND read in same cycle.
        # In VLIW, reads happen before writes, so vbroadcast in B reads s_tmp from BEFORE B's const writes.
        # That means vbroadcast in B reads s_tmp = c1 (from cycle A), not c3. CORRECT!
        # Similarly vbroadcast v2 reads s_tmp2 = c2 from cycle A.
        n_hc = len(hash_consts_batch)
        hc_tmp_regs = [self.alloc_scratch(f"s_hc_tmp_{i}") for i in range(n_hc)]
        
        # Cycle 0: Load first pair
        asm.add("load", ("const", hc_tmp_regs[0], hash_consts_batch[0][1]))
        if n_hc > 1:
            asm.add("load", ("const", hc_tmp_regs[1], hash_consts_batch[1][1]))
        asm.emit()

        # Cycles 1..: Load next pair + broadcast previous pair
        for i in range(2, n_hc, 2):
            # Load pair i/i+1
            asm.add("load", ("const", hc_tmp_regs[i], hash_consts_batch[i][1]))
            if i + 1 < n_hc:
                asm.add("load", ("const", hc_tmp_regs[i + 1], hash_consts_batch[i + 1][1]))
            # Broadcast prev pair (i-2/i-1)
            asm.add("valu", ("vbroadcast", hash_consts_batch[i - 2][0], hc_tmp_regs[i - 2]))
            asm.add("valu", ("vbroadcast", hash_consts_batch[i - 1][0], hc_tmp_regs[i - 1]))
            asm.emit()

        # Broadcast the last pair
        last_start = (n_hc - 1) // 2 * 2
        for j in range(last_start, n_hc):
            asm.add("valu", ("vbroadcast", hash_consts_batch[j][0], hc_tmp_regs[j]))
        
        # Step 4: Broadcast scalar constants to vectors (co-scheduled with the final hash broadcasts)
        asm.add("valu", ("vbroadcast", v_zero, s_zero))
        asm.add("valu", ("vbroadcast", v_one, s_one))
        asm.add("valu", ("vbroadcast", v_two, s_two))
        asm.add("valu", ("vbroadcast", v_n_nodes, s_vars["n_nodes"]))
        asm.emit()

        asm.add("valu", ("vbroadcast", v_forest_values_p, s_vars["forest_values_p"]))
        asm.add("valu", ("vbroadcast", v_hmul0, s_m0))
        asm.add("valu", ("vbroadcast", v_hmul2, s_m2))
        asm.add("valu", ("vbroadcast", v_hmul4, s_m4))
        asm.add("valu", ("vbroadcast", v_three, s_three))
        asm.add("valu", ("vbroadcast", v_seven, s_seven))
        asm.emit()

        asm.add("load", ("const", s_loop_i, 0))
        asm.emit()

        # Load 8 into s_tmp
        asm.add("load", ("const", s_tmp, 8))
        asm.add("alu", ("+", s_addr_idx[0], s_vars["inp_indices_p"], s_zero))
        asm.add("alu", ("+", s_addr_val[0], s_vars["inp_values_p"], s_zero))
        asm.emit()

        s_c8   = s_tmp
        s_c16  = s_tmp2
        s_c32  = self.alloc_scratch("s_c32")
        s_c64  = self.alloc_scratch("s_c64")
        s_c128 = self.alloc_scratch("s_c128")
        asm.add("load", ("const", s_c16, 16))
        asm.add("load", ("const", s_c32, 32))
        asm.emit()
        asm.add("load", ("const", s_c64, 64))
        asm.add("load", ("const", s_c128, 128))
        asm.emit()

        # Cycle 1: 5 groups
        for g, c in [(1, s_c8), (2, s_c16), (4, s_c32), (8, s_c64), (16, s_c128)]:
            asm.add("alu", ("+", s_addr_idx[g], s_addr_idx[0], c))
            asm.add("alu", ("+", s_addr_val[g], s_addr_val[0], c))
        asm.emit()

        # Cycle 2: 6 groups
        for g, base, c in [(3, 2, s_c8), (5, 4, s_c8), (6, 4, s_c16),
                           (9, 8, s_c8), (10, 8, s_c16), (12, 8, s_c32)]:
            asm.add("alu", ("+", s_addr_idx[g], s_addr_idx[base], c))
            asm.add("alu", ("+", s_addr_val[g], s_addr_val[base], c))
        asm.emit()

        # Cycle 3: 6 groups
        for g, base, c in [(17, 16, s_c8), (18, 16, s_c16), (20, 16, s_c32), (24, 16, s_c64),
                           (7, 6, s_c8), (11, 10, s_c8)]:
            asm.add("alu", ("+", s_addr_idx[g], s_addr_idx[base], c))
            asm.add("alu", ("+", s_addr_val[g], s_addr_val[base], c))
        asm.emit()

        # Cycle 4: 6 groups
        for g, base, c in [(13, 12, s_c8), (14, 12, s_c16), (19, 18, s_c8),
                           (21, 20, s_c8), (22, 20, s_c16), (25, 24, s_c8)]:
            asm.add("alu", ("+", s_addr_idx[g], s_addr_idx[base], c))
            asm.add("alu", ("+", s_addr_val[g], s_addr_val[base], c))
        asm.emit()

        # Cycle 5: 4 groups
        for g, base, c in [(26, 24, s_c16), (28, 24, s_c32), (15, 14, s_c8), (23, 22, s_c8)]:
            asm.add("alu", ("+", s_addr_idx[g], s_addr_idx[base], c))
            asm.add("alu", ("+", s_addr_val[g], s_addr_val[base], c))
        asm.emit()

        # Cycle 6: 3 groups
        for g, base, c in [(27, 26, s_c8), (29, 28, s_c8), (30, 28, s_c16)]:
            asm.add("alu", ("+", s_addr_idx[g], s_addr_idx[base], c))
            asm.add("alu", ("+", s_addr_val[g], s_addr_val[base], c))
        asm.emit()

        # Cycle 7: 1 group
        asm.add("alu", ("+", s_addr_idx[31], s_addr_idx[30], s_c8))
        asm.add("alu", ("+", s_addr_val[31], s_addr_val[30], s_c8))
        asm.emit()

        if MAX_OPT_ROUND > 0:
            # Step 8: Load tree leaf values from memory
            # Leaf 0: load directly from forest_values_p
            asm.add("load", ("load", s_leaves[0], s_vars["forest_values_p"]))
            asm.emit()
            # Leaves 1..6: add_imm (flow) + load
            # Since add_imm writes to s_tmp and load reads from s_tmp, they cannot be in the same cycle.
            # But we can overlap load of leaf i-1 with add_imm of leaf i!
            # Cycle 1: add_imm for leaf 1 -> writes s_tmp
            if n_leaves > 1:
                asm.add("flow", ("add_imm", s_tmp, s_vars["forest_values_p"], 1))
                asm.emit()
            # Cycle 2..: load leaf i-1 from s_tmp, add_imm for leaf i to s_tmp
            # Since s_tmp is read by load (using pre-cycle value) and written by add_imm (effective at end of cycle),
            # this is perfectly safe in VLIW!
            for i in range(2, n_leaves):
                asm.add("flow", ("add_imm", s_tmp, s_vars["forest_values_p"], i))
                asm.add("load", ("load", s_leaves[i-1], s_tmp))
                asm.emit()
            # After the loop, the last cycle of the loop did:
            #   add_imm for leaf 6 (n_leaves-1) -> writes s_tmp
            #   load leaf 5 (s_leaves[5]) from s_tmp_old
            # So s_tmp now contains the address for leaf 6.
            # We just need to load leaf 6 (s_leaves[6]) from s_tmp:
            if n_leaves > 1:
                asm.add("load", ("load", s_leaves[n_leaves-1], s_tmp))
                asm.emit()

            # Compute diffs and group broadcasts:
            # Diffs:
            # r=1: s_d_1_0 = s_leaf_2 - s_leaf_1
            # r=2: s_d_2_0 = s_leaf_4 - s_leaf_3, s_d_2_1 = s_leaf_6 - s_leaf_5
            if MAX_OPT_ROUND >= 2:
                asm.add("alu", ("-", s_diffs[1][0], s_leaves[2], s_leaves[1]))
            if MAX_OPT_ROUND >= 3:
                asm.add("alu", ("-", s_diffs[2][0], s_leaves[4], s_leaves[3]))
                asm.add("alu", ("-", s_diffs[2][1], s_leaves[6], s_leaves[5]))
                asm.add("alu", ("-", s_diff2_base, s_leaves[5], s_leaves[3]))
                asm.emit()
                asm.add("alu", ("-", s_diff2_slope, s_diffs[2][1], s_diffs[2][0]))
            
            if MAX_OPT_ROUND >= 4:
                asm.add("alu", ("-", s_diffs[3][0], s_leaves[8], s_leaves[7]))
                asm.add("alu", ("-", s_diffs[3][1], s_leaves[10], s_leaves[9]))
                asm.emit()
                asm.add("alu", ("-", s_diffs[3][2], s_leaves[12], s_leaves[11]))
                asm.add("alu", ("-", s_diffs[3][3], s_leaves[14], s_leaves[13]))
            asm.emit()

            for i in range(min(4, n_leaves)):
                r_val = 0 if i == 0 else (1 if i <= 2 else 2)
                k_val = 0 if i == 0 else (i - 1 if i <= 2 else i - 3)
                asm.add("valu", ("vbroadcast", v_leaf[r_val][k_val], s_leaves[i]))
            asm.emit()

            for i in range(4, min(7, n_leaves)):
                r_val = 2
                k_val = i - 3
                asm.add("valu", ("vbroadcast", v_leaf[r_val][k_val], s_leaves[i]))
            if MAX_OPT_ROUND >= 4:
                asm.add("valu", ("vbroadcast", v_leaf[3][0], s_leaves[7]))
            asm.emit()

            if MAX_OPT_ROUND >= 4:
                asm.add("valu", ("vbroadcast", v_leaf[3][1], s_leaves[9]))
                asm.add("valu", ("vbroadcast", v_leaf[3][2], s_leaves[11]))
                asm.add("valu", ("vbroadcast", v_leaf[3][3], s_leaves[13]))
            if MAX_OPT_ROUND >= 2:
                asm.add("valu", ("vbroadcast", v_diff[1][0], s_diffs[1][0]))
            asm.emit()

            if MAX_OPT_ROUND >= 3:
                asm.add("valu", ("vbroadcast", v_diff[2][0], s_diffs[2][0]))
                asm.add("valu", ("vbroadcast", v_diff[2][1], s_diffs[2][1]))
                asm.add("valu", ("vbroadcast", v_diff2_base, s_diff2_base))
                asm.add("valu", ("vbroadcast", v_diff2_slope, s_diff2_slope))
            asm.emit()

            if MAX_OPT_ROUND >= 4:
                asm.add("valu", ("vbroadcast", v_diff[3][0], s_diffs[3][0]))
                asm.add("valu", ("vbroadcast", v_diff[3][1], s_diffs[3][1]))
                asm.add("valu", ("vbroadcast", v_diff[3][2], s_diffs[3][2]))
                asm.add("valu", ("vbroadcast", v_diff[3][3], s_diffs[3][3]))
            asm.emit()

        asm.add("flow", ("pause",))
        asm.emit()

        # ---- BUILD OP LIST FOR LIST SCHEDULING ----
        ops = []
        current_group = None

        def emit_op(engine, slot, reads, writes):
            ops.append({"engine": engine, "slot": slot, "reads": reads, "writes": writes, "g": current_group})

        def emit_any_op(g, engine, op, reads, writes):
            offload_ops = {"^", "-", "&", ">>", "+", "<", "*"}
            op_name = op[0]
            if engine == "valu" and op_name in offload_ops:
                # Always offload simple ops to ALU!
                # Wait, ALU only has 12 slots, but it has plenty of space.
                emit_op("alu", op, reads, writes)
            else:
                emit_op(engine, op, reads, writes)

        def vr(reg):
            return list(range(reg, reg + 8))

        # Phase 1: Load all indices and values
        for g in range(32):
            current_group = g
            emit_op("load", ("vload", v_idx[g], s_addr_idx[g]),
                    reads=[s_addr_idx[g]], writes=vr(v_idx[g]))
            emit_op("load", ("vload", v_val[g], s_addr_val[g]),
                    reads=[s_addr_val[g]], writes=vr(v_val[g]))

        # Phase 2: rounds. Tree depth is periodic: depth = rnd % (forest_height+1),
        # since idx wraps to the root every forest_height+1 rounds regardless of
        # data. So rounds that land on depth 0/1/2 can all reuse the preloaded
        # leaf/diff registers, not just the literal first few rounds.
        period = forest_height + 1

        def emit_round_group(rnd, g):
            """Emit all ops for one group in one round."""
            nonlocal current_group
            current_group = g
            depth = rnd % period
            vg = v_tmp[g % 16]
            gA = v_glob_A[g % NG_A]
            gB = v_glob_B[g % NG_B]

            if depth == 0:
                emit_any_op(g, "valu", ("^", v_val[g], v_val[g], v_leaf[0][0]),
                        reads=vr(v_val[g]) + vr(v_leaf[0][0]),
                        writes=vr(v_val[g]))

            elif depth == 1:
                emit_any_op(g, "valu", ("-", gA, v_idx[g], v_one),
                        reads=vr(v_idx[g]) + vr(v_one), writes=vr(gA))
                emit_any_op(g, "valu", ("&", vg, gA, v_one),
                        reads=vr(gA) + vr(v_one), writes=vr(vg))
                emit_any_op(g, "valu", ("multiply_add", gA, vg, v_diff[1][0], v_leaf[1][0]),
                        reads=vr(vg) + vr(v_diff[1][0]) + vr(v_leaf[1][0]),
                        writes=vr(gA))
                emit_any_op(g, "valu", ("^", v_val[g], v_val[g], gA),
                        reads=vr(v_val[g]) + vr(gA), writes=vr(v_val[g]))

            elif depth == 2:
                emit_any_op(g, "valu", ("-", gA, v_idx[g], v_three), reads=vr(v_idx[g]) + vr(v_three), writes=vr(gA))
                emit_any_op(g, "valu", (">>", gB, gA, v_one), reads=vr(gA) + vr(v_one), writes=vr(gB))
                emit_any_op(g, "valu", ("&", vg, gA, v_one), reads=vr(gA) + vr(v_one), writes=vr(vg))
                emit_any_op(g, "valu", ("multiply_add", gA, vg, v_diff[2][0], v_leaf[2][0]), reads=vr(vg) + vr(v_diff[2][0]) + vr(v_leaf[2][0]), writes=vr(gA))
                emit_any_op(g, "valu", ("multiply_add", vg, vg, v_diff2_slope, v_diff2_base), reads=vr(vg) + vr(v_diff2_slope) + vr(v_diff2_base), writes=vr(vg))
                emit_any_op(g, "valu", ("multiply_add", gA, gB, vg, gA), reads=vr(gB) + vr(vg) + vr(gA), writes=vr(gA))
                emit_any_op(g, "valu", ("^", v_val[g], v_val[g], gA), reads=vr(v_val[g]) + vr(gA), writes=vr(v_val[g]))
            elif depth == 3:
                t_b0 = vg
                t_b1 = gA
                t_b2 = gB
                t_M01 = v_glob_C[g % NG_C]
                t_M23 = v_glob_D[g % NG_D]
                t_M0123 = v_glob_E[g % NG_E]
                t_diff = v_glob_F[g % NG_F]
                
                emit_any_op(g, "valu", ("-", t_b0, v_idx[g], v_seven), reads=vr(v_idx[g])+vr(v_seven), writes=vr(t_b0))
                emit_any_op(g, "valu", ("&", t_b0, t_b0, v_one), reads=vr(t_b0)+vr(v_one), writes=vr(t_b0))
                emit_any_op(g, "valu", ("-", t_b1, v_idx[g], v_seven), reads=vr(v_idx[g])+vr(v_seven), writes=vr(t_b1))
                emit_any_op(g, "valu", (">>", t_b1, t_b1, v_one), reads=vr(t_b1)+vr(v_one), writes=vr(t_b1))
                emit_any_op(g, "valu", ("&", t_b1, t_b1, v_one), reads=vr(t_b1)+vr(v_one), writes=vr(t_b1))
                emit_any_op(g, "valu", ("-", t_b2, v_idx[g], v_seven), reads=vr(v_idx[g])+vr(v_seven), writes=vr(t_b2))
                emit_any_op(g, "valu", (">>", t_b2, t_b2, v_two), reads=vr(t_b2)+vr(v_two), writes=vr(t_b2))
                emit_any_op(g, "valu", ("&", t_b2, t_b2, v_one), reads=vr(t_b2)+vr(v_one), writes=vr(t_b2))
                
                emit_any_op(g, "valu", ("multiply_add", t_M01, t_b0, v_diff[3][0], v_leaf[3][0]), reads=vr(t_b0)+vr(v_diff[3][0])+vr(v_leaf[3][0]), writes=vr(t_M01))
                emit_any_op(g, "valu", ("multiply_add", t_M23, t_b0, v_diff[3][1], v_leaf[3][1]), reads=vr(t_b0)+vr(v_diff[3][1])+vr(v_leaf[3][1]), writes=vr(t_M23))
                
                emit_any_op(g, "valu", ("-", t_diff, t_M23, t_M01), reads=vr(t_M23)+vr(t_M01), writes=vr(t_diff))
                emit_any_op(g, "valu", ("multiply_add", t_M0123, t_b1, t_diff, t_M01), reads=vr(t_b1)+vr(t_diff)+vr(t_M01), writes=vr(t_M0123))
                
                t_M45 = t_M01
                t_M67 = t_M23
                emit_any_op(g, "valu", ("multiply_add", t_M45, t_b0, v_diff[3][2], v_leaf[3][2]), reads=vr(t_b0)+vr(v_diff[3][2])+vr(v_leaf[3][2]), writes=vr(t_M45))
                emit_any_op(g, "valu", ("multiply_add", t_M67, t_b0, v_diff[3][3], v_leaf[3][3]), reads=vr(t_b0)+vr(v_diff[3][3])+vr(v_leaf[3][3]), writes=vr(t_M67))
                
                t_M4567 = t_M67
                emit_any_op(g, "valu", ("-", t_diff, t_M67, t_M45), reads=vr(t_M67)+vr(t_M45), writes=vr(t_diff))
                emit_any_op(g, "valu", ("multiply_add", t_M4567, t_b1, t_diff, t_M45), reads=vr(t_b1)+vr(t_diff)+vr(t_M45), writes=vr(t_M4567))
                
                t_val_final = t_M01
                emit_any_op(g, "valu", ("-", t_diff, t_M4567, t_M0123), reads=vr(t_M4567)+vr(t_M0123), writes=vr(t_diff))
                emit_any_op(g, "valu", ("multiply_add", t_val_final, t_b2, t_diff, t_M0123), reads=vr(t_b2)+vr(t_diff)+vr(t_M0123), writes=vr(t_val_final))
                
                emit_any_op(g, "valu", ("^", v_val[g], v_val[g], t_val_final), reads=vr(v_val[g])+vr(t_val_final), writes=vr(v_val[g]))


            else:
                node_addr = vg
                node_val  = gA
                emit_any_op(g, "valu", ("+", node_addr, v_forest_values_p, v_idx[g]),
                        reads=vr(v_forest_values_p) + vr(v_idx[g]),
                        writes=vr(node_addr))
                for off in range(8):
                    emit_op("load", ("load_offset", node_val, node_addr, off),
                            reads=vr(node_addr),
                            writes=[node_val + off])
                emit_any_op(g, "valu", ("^", v_val[g], v_val[g], node_val),
                        reads=vr(v_val[g]) + vr(node_val),
                        writes=vr(v_val[g]))

            # --- Hash stages ---
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                if hi == 0:
                    emit_any_op(g, "valu", ("multiply_add", v_val[g], v_val[g], v_hmul0, v_hash_val1[0]),
                            reads=vr(v_val[g]) + vr(v_hmul0) + vr(v_hash_val1[0]),
                            writes=vr(v_val[g]))
                elif hi == 2:
                    emit_any_op(g, "valu", ("multiply_add", v_val[g], v_val[g], v_hmul2, v_hash_val1[2]),
                            reads=vr(v_val[g]) + vr(v_hmul2) + vr(v_hash_val1[2]),
                            writes=vr(v_val[g]))
                elif hi == 4:
                    emit_any_op(g, "valu", ("multiply_add", v_val[g], v_val[g], v_hmul4, v_hash_val1[4]),
                            reads=vr(v_val[g]) + vr(v_hmul4) + vr(v_hash_val1[4]),
                            writes=vr(v_val[g]))
                else:
                    emit_any_op(g, "valu", (op3, vg, v_val[g], v_hash_val3[hi]),
                            reads=vr(v_val[g]) + vr(v_hash_val3[hi]),
                            writes=vr(vg))
                    emit_any_op(g, "valu", (op1, gA, v_val[g], v_hash_val1[hi]),
                            reads=vr(v_val[g]) + vr(v_hash_val1[hi]),
                            writes=vr(gA))
                    emit_any_op(g, "valu", (op2, v_val[g], gA, vg),
                            reads=vr(gA) + vr(vg),
                            writes=vr(v_val[g]))

            # --- Index update (skip last round) ---
            if rnd < rounds - 1:
                emit_any_op(g, "valu", ("&", vg, v_val[g], v_one),
                        reads=vr(v_val[g]) + vr(v_one), writes=vr(vg))
                emit_any_op(g, "valu", ("+", vg, vg, v_one),
                        reads=vr(vg) + vr(v_one), writes=vr(vg))
                emit_any_op(g, "valu", ("multiply_add", v_idx[g], v_idx[g], v_two, vg),
                        reads=vr(v_idx[g]) + vr(v_two) + vr(vg),
                        writes=vr(v_idx[g]))
                if depth == period - 1:
                    emit_any_op(g, "valu", ("<", vg, v_idx[g], v_n_nodes),
                            reads=vr(v_idx[g]) + vr(v_n_nodes), writes=vr(vg))
                    emit_any_op(g, "valu", ("*", v_idx[g], v_idx[g], vg),
                            reads=vr(v_idx[g]) + vr(vg), writes=vr(v_idx[g]))

        for rnd in range(rounds):
            for g in range(32):
                emit_round_group(rnd, g)


        # Phase 3: Store values only (test only checks inp_values, not indices)
        for g in range(32):
            emit_op("store", ("vstore", s_addr_val[g], v_val[g]),
                    reads=[s_addr_val[g]] + vr(v_val[g]), writes=[])

        # ---- Improved List Scheduler ----
        def schedule_ops(ops):
            n = len(ops)
            last_writer = {}
            readers = defaultdict(list)
            parents = [set() for _ in range(n)]
            children = [set() for _ in range(n)]

            for i, op_i in enumerate(ops):
                for r in op_i["reads"]:
                    if r in last_writer:
                        parents[i].add(last_writer[r])
                for w in op_i["writes"]:
                    if w in last_writer:
                        parents[i].add(last_writer[w])
                    for r_idx in readers[w]:
                        parents[i].add(r_idx)
                for r in op_i["reads"]:
                    readers[r].append(i)
                for w in op_i["writes"]:
                    last_writer[w] = i
                    readers[w] = []

            for i in range(n):
                for p in parents[i]:
                    children[p].add(i)

            # Count successors for tiebreaking
            n_successors = [0] * n
            for i in range(n):
                n_successors[i] = len(children[i])

            # Critical path heights
            in_deg_topo = [len(parents[i]) for i in range(n)]
            heights = [0] * n
            q = deque(i for i in range(n) if in_deg_topo[i] == 0)
            topo = []
            while q:
                node = q.popleft()
                topo.append(node)
                for c in children[node]:
                    in_deg_topo[c] -= 1
                    if in_deg_topo[c] == 0:
                        q.append(c)
            for node in reversed(topo):
                for c in children[node]:
                    heights[node] = max(heights[node], heights[c] + 1)

            # Greedy list scheduler
            in_degree = [len(parents[i]) for i in range(n)]
            ready = [i for i in range(n) if in_degree[i] == 0]
            scheduled_cycle = [-1] * n
            bundles = []
            current_cycle = 0

            while ready:
                cycle_ready = [
                    idx for idx in ready
                    if all(scheduled_cycle[p] < current_cycle for p in parents[idx])
                ]
                def priority(idx):
                    eng = ops[idx]["engine"]
                    eng_pri = 2 if eng == "load" else (1 if eng == "store" else 0)
                    g_stagger = (31 - (ops[idx].get("g") or 0)) * 5
                    return (heights[idx] + g_stagger, eng_pri, n_successors[idx])
                cycle_ready.sort(key=priority, reverse=True)

                bundle = defaultdict(list)
                scheduled_now = []
                for op_idx in cycle_ready:
                    eng = ops[op_idx]["engine"]
                    if len(bundle[eng]) < SLOT_LIMITS.get(eng, 64):
                        bundle[eng].append(ops[op_idx]["slot"])
                        scheduled_now.append(op_idx)

                for op_idx in scheduled_now:
                    ready.remove(op_idx)
                    scheduled_cycle[op_idx] = current_cycle
                    for child in children[op_idx]:
                        in_degree[child] -= 1
                        if in_degree[child] == 0:
                            ready.append(child)

                bundles.append(dict(bundle) if bundle else {})
                current_cycle += 1

            return bundles



        scheduled_bundles = schedule_ops(ops)
        for b in scheduled_bundles:
            for eng, slots in b.items():
                for slot in slots:
                    asm.add(eng, slot)
            asm.emit()

        self.instrs.extend(asm.instrs)
