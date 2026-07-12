import copy
from collections import defaultdict
from problem import SLOT_LIMITS, HASH_STAGES

def kernel_builder(forest_height, n_nodes, batch_size, rounds):
    return KernelBuilder()

class KernelBuilder:
    def __init__(self):
        self.scratch_ptr = 0
        self.instrs = []
        self.curr = defaultdict(list)
    
    def alloc_scratch(self, name=None, size=1):
        ptr = self.scratch_ptr
        self.scratch_ptr += size
        assert self.scratch_ptr <= 1536, f"Scratch overflow: {self.scratch_ptr} > 1536"
        return ptr

    def add(self, engine, args):
        assert len(self.curr[engine]) <= SLOT_LIMITS[engine], f"Too many slots for {engine}"
        self.curr[engine].append(args)

    def emit(self):
        if self.curr:
            self.instrs.append(dict(self.curr))
            self.curr.clear()

    def build_kernel(self, forest_height, n_nodes, batch_size, rounds):
        asm = self
        
        MAX_OPT_ROUND = 4
        n_leaves = (1 << MAX_OPT_ROUND) - 1

        init_vars = [
            "s_addr_in_idx", "s_addr_in_val", "s_addr_out_val", "s_addr_forest",
            "s_batch_size", "s_n_nodes", "s_c1", "s_c2"
        ]
        s_vars = {v: self.alloc_scratch(v) for v in init_vars}
        
        s_tmp   = self.alloc_scratch("s_tmp")
        s_tmp2  = self.alloc_scratch("s_tmp2")
        s_loop_i    = self.alloc_scratch("s_loop_i")
        s_loop_cond = self.alloc_scratch("s_loop_cond")
        s_zero  = self.alloc_scratch("s_zero")
        s_one   = self.alloc_scratch("s_one")
        s_two   = self.alloc_scratch("s_two")
        
        s_addr_idx = [self.alloc_scratch(f"s_addr_idx_{g}") for g in range(32)]
        s_addr_val = [self.alloc_scratch(f"s_addr_val_{g}") for g in range(32)]
        
        v_one  = self.alloc_scratch("v_one",  8)
        v_two  = self.alloc_scratch("v_two",  8)
        v_three = self.alloc_scratch("v_three", 8)
        v_seven = self.alloc_scratch("v_seven", 8)
        v_n_nodes = self.alloc_scratch("v_n_nodes", 8)
        v_forest_values_p = self.alloc_scratch("v_forest_values_p", 8)
        
        v_idx = [self.alloc_scratch(f"v_idx_{g}", 8) for g in range(32)]
        v_val = [self.alloc_scratch(f"v_val_{g}", 8) for g in range(32)]
        
        # We share 5 sets of 7 temps
        NG_TMP = 5
        v_tmp = [[self.alloc_scratch(f"v_tmp_{g}_{i}", 8) for i in range(7)] for g in range(NG_TMP)]
        
        v_hash_val1 = []
        v_hash_val3 = {}
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_hash_val1.append(self.alloc_scratch(f"v_hv1_{hi}", 8))
            if op3 is not None:
                v_hash_val3[hi] = self.alloc_scratch(f"v_hv3_{hi}", 8)
                
        v_hmul0 = self.alloc_scratch("v_hmul0", 8)
        v_hmul2 = self.alloc_scratch("v_hmul2", 8)
        v_hmul4 = self.alloc_scratch("v_hmul4", 8)
        
        s_leaves = [self.alloc_scratch(f"s_leaf_{i}") for i in range(n_leaves)]
        v_leaf = [self.alloc_scratch(f"vl_{i}", 8) for i in range(n_leaves)]
        
        s_diffs = []
        v_diff = []
        for r in range(MAX_OPT_ROUND):
            s_diffs.append([])
            v_diff.append([])
            if r > 0:
                for k in range(1 << (r - 1)):
                    s_diffs[r].append(self.alloc_scratch(f"s_d_{r}_{k}"))
                    v_diff[r].append(self.alloc_scratch(f"vd_{r}_{k}", 8))
                    
        s_diff2_base  = self.alloc_scratch("s_diff2_base")
        s_diff2_slope = self.alloc_scratch("s_diff2_slope")
        v_diff2_base  = self.alloc_scratch("v_diff2_base", 8)
        v_diff2_slope = self.alloc_scratch("v_diff2_slope", 8)
        
        NUM_OFFLOAD = 4
        tmp_madd = [[self.alloc_scratch(f"tmp_madd_{g}_{i}") for i in range(8)] for g in range(NUM_OFFLOAD)]

        asm.add("flow", ("add_imm", s_vars["s_addr_in_idx"], 0, 0))
        asm.emit()
        asm.add("flow", ("add_imm", s_vars["s_addr_in_val"], 0, 1))
        asm.emit()
        asm.add("flow", ("add_imm", s_vars["s_addr_out_val"], 0, 2))
        asm.emit()
        asm.add("flow", ("add_imm", s_vars["s_addr_forest"], 0, 3))
        asm.emit()
        asm.add("flow", ("add_imm", s_vars["s_batch_size"], 0, 4))
        asm.emit()
        asm.add("flow", ("add_imm", s_vars["s_n_nodes"], 0, 5))
        asm.emit()
        asm.add("flow", ("add_imm", s_vars["s_c1"], 0, 6))
        asm.emit()
        asm.add("flow", ("add_imm", s_vars["s_c2"], 0, 7))
        asm.emit()

        for v in init_vars:
            asm.add("load", ("load", s_vars[v], s_vars[v]))
            asm.emit()
        asm.add("flow", ("pause",))
        asm.emit()

        asm.add("load", ("const", s_zero, 0))
        asm.emit()
        asm.add("load", ("const", s_one, 1))
        asm.emit()
        asm.add("load", ("const", s_two, 2))
        asm.emit()
        
        asm.add("load", ("const", s_tmp, 3))
        asm.emit()
        asm.add("valu", ("vbroadcast", v_three, s_tmp))
        
        asm.add("load", ("const", s_tmp, 7))
        asm.emit()
        asm.add("valu", ("vbroadcast", v_seven, s_tmp))
        asm.emit()

        asm.add("valu", ("vbroadcast", v_one, s_one))
        asm.add("valu", ("vbroadcast", v_two, s_two))
        asm.add("valu", ("vbroadcast", v_n_nodes, s_vars["s_n_nodes"]))
        asm.add("valu", ("vbroadcast", v_forest_values_p, s_vars["s_addr_forest"]))
        for g in range(32):
            asm.add("flow", ("add_imm", s_tmp, s_zero, g * 8))
            asm.emit()
            asm.add("flow", ("add_imm", s_addr_idx[g], s_vars["s_addr_in_idx"], g * 8))
            asm.emit()
            asm.add("flow", ("add_imm", s_addr_val[g], s_vars["s_addr_in_val"], g * 8))
            asm.emit()

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            asm.add("load", ("const", s_tmp, val1))
            asm.emit()
            asm.add("valu", ("vbroadcast", v_hash_val1[hi], s_tmp))
            if op3 is not None:
                asm.add("load", ("const", s_tmp, val3))
                asm.emit()
                asm.add("valu", ("vbroadcast", v_hash_val3[hi], s_tmp))
            asm.emit()
        
        asm.add("load", ("const", s_tmp, HASH_STAGES[0][1]))
        asm.emit()
        asm.add("valu", ("vbroadcast", v_hmul0, s_tmp))
        asm.add("load", ("const", s_tmp, HASH_STAGES[2][1]))
        asm.emit()
        asm.add("valu", ("vbroadcast", v_hmul2, s_tmp))
        asm.add("load", ("const", s_tmp, HASH_STAGES[4][1]))
        asm.emit()
        asm.add("valu", ("vbroadcast", v_hmul4, s_tmp))
        asm.emit()

        for i in range(n_leaves):
            asm.add("flow", ("add_imm", s_tmp, s_vars["s_addr_forest"], i))
            asm.emit()
            asm.add("load", ("load", s_leaves[i], s_tmp))
            asm.emit()
            asm.add("valu", ("vbroadcast", v_leaf[i], s_leaves[i]))
            asm.emit()
            
        for r in range(1, MAX_OPT_ROUND):
            start = (1 << r) - 1
            for k in range(1 << (r - 1)):
                L = s_leaves[start + 2 * k]
                R = s_leaves[start + 2 * k + 1]
                asm.add("alu", ("+", s_tmp, R, s_zero))  
                asm.add("alu", ("*", s_tmp2, L, s_vars["s_c1"])) 
                asm.emit()
                asm.add("alu", ("+", s_diffs[r][k], s_tmp, s_tmp2))
                asm.emit()
                asm.add("valu", ("vbroadcast", v_diff[r][k], s_diffs[r][k]))
                asm.emit()
                
        asm.add("alu", ("+", s_tmp, s_diffs[2][1], s_zero))
        asm.add("alu", ("*", s_tmp2, s_diffs[2][0], s_vars["s_c1"]))
        asm.emit()
        asm.add("alu", ("+", s_diff2_slope, s_tmp, s_tmp2))
        asm.emit()
        asm.add("alu", ("+", s_tmp, s_leaves[5], s_zero))
        asm.add("alu", ("*", s_tmp2, s_leaves[3], s_vars["s_c1"]))
        asm.emit()
        asm.add("alu", ("+", s_diff2_base, s_tmp, s_tmp2))
        asm.emit()
        
        asm.add("valu", ("vbroadcast", v_diff2_slope, s_diff2_slope))
        asm.add("valu", ("vbroadcast", v_diff2_base, s_diff2_base))
        asm.emit()

        asm.add("flow", ("pause",))
        asm.emit()

        ops = []
        def emit_op(engine, slot, reads=None, writes=None, g=None):
            if reads is None: reads = []
            if writes is None: writes = []
            ops.append({"engine": engine, "slot": slot, "reads": reads, "writes": writes, "g": g})

        def vr(reg):
            return list(range(reg, reg + 8))
            
        def emit_any_op(g, default_eng, slot, reads=None, writes=None):
            if reads is None: reads = []
            if writes is None: writes = []
            if g < NUM_OFFLOAD: 
                if default_eng == "valu":
                    if slot[0] == "multiply_add":
                        _, dest, a, b, c_op = slot
                        for i in range(8):
                            t = tmp_madd[g][i]
                            emit_op("alu", ("*", t, a+i, b+i), reads=[a+i, b+i], writes=[t], g=g)
                            emit_op("alu", ("+", dest+i, t, c_op+i), reads=[t, c_op+i], writes=[dest+i], g=g)
                    elif slot[0] == "vbroadcast":
                        _, dest, src = slot
                        for i in range(8):
                            emit_op("alu", ("+", dest+i, src, s_zero), reads=[src, s_zero], writes=[dest+i], g=g)
                    else:
                        op, dest, a, b = slot
                        for i in range(8):
                            emit_op("alu", (op, dest+i, a+i, b+i), reads=[a+i, b+i], writes=[dest+i], g=g)
                elif default_eng == "load" and slot[0] == "load_offset":
                    _, dest, addr, off = slot
                    emit_op("load", ("load", dest+off, addr+off), reads=[addr+off], writes=[dest+off], g=g)
                else:
                    emit_op(default_eng, slot, reads=reads, writes=writes, g=g)
            else:
                emit_op(default_eng, slot, reads=reads, writes=writes, g=g)

        for g in range(32):
            emit_any_op(g, "load", ("vload", v_idx[g], s_addr_idx[g]), reads=[s_addr_idx[g]], writes=vr(v_idx[g]))
            emit_any_op(g, "load", ("vload", v_val[g], s_addr_val[g]), reads=[s_addr_val[g]], writes=vr(v_val[g]))

        period = forest_height + 1
        
        for rnd in range(rounds):
            for g in range(32):
                depth = rnd % period
                
                t_b0 = v_tmp[g % NG_TMP][0]
                t_b1 = v_tmp[g % NG_TMP][1]
                t_b2 = v_tmp[g % NG_TMP][2]
                t_M01 = v_tmp[g % NG_TMP][3]
                t_M23 = v_tmp[g % NG_TMP][4]
                t_diff = v_tmp[g % NG_TMP][5]
                t_M0123 = v_tmp[g % NG_TMP][6]
                
                vg = t_b0
                gA = t_b1
                gB = t_b2
                
                # TREE LOOKUP
                if depth == 0:
                    emit_any_op(g, "valu", ("^", v_val[g], v_val[g], v_leaf[0]), reads=vr(v_val[g])+vr(v_leaf[0]), writes=vr(v_val[g]))
                elif depth == 1:
                    emit_any_op(g, "valu", ("-", gA, v_idx[g], v_one), reads=vr(v_idx[g])+vr(v_one), writes=vr(gA))
                    emit_any_op(g, "valu", ("&", vg, gA, v_one), reads=vr(gA)+vr(v_one), writes=vr(vg))
                    emit_any_op(g, "valu", ("multiply_add", gA, vg, v_diff[1][0], v_leaf[1]), reads=vr(vg)+vr(v_diff[1][0])+vr(v_leaf[1]), writes=vr(gA))
                    emit_any_op(g, "valu", ("^", v_val[g], v_val[g], gA), reads=vr(v_val[g])+vr(gA), writes=vr(v_val[g]))
                elif depth == 2:
                    emit_any_op(g, "valu", ("-", gA, v_idx[g], v_three), reads=vr(v_idx[g])+vr(v_three), writes=vr(gA))
                    emit_any_op(g, "valu", (">>", gB, gA, v_one), reads=vr(gA)+vr(v_one), writes=vr(gB))
                    emit_any_op(g, "valu", ("&", vg, gA, v_one), reads=vr(gA)+vr(v_one), writes=vr(vg))
                    emit_any_op(g, "valu", ("multiply_add", gA, vg, v_diff[2][0], v_leaf[3]), reads=vr(vg)+vr(v_diff[2][0])+vr(v_leaf[3]), writes=vr(gA))
                    emit_any_op(g, "valu", ("multiply_add", vg, vg, v_diff2_slope, v_diff2_base), reads=vr(vg)+vr(v_diff2_slope)+vr(v_diff2_base), writes=vr(vg))
                    emit_any_op(g, "valu", ("multiply_add", gA, gB, vg, gA), reads=vr(gB)+vr(vg)+vr(gA), writes=vr(gA))
                    emit_any_op(g, "valu", ("^", v_val[g], v_val[g], gA), reads=vr(v_val[g])+vr(gA), writes=vr(v_val[g]))
                elif depth == 3:
                    # 3D Trilinear Interpolation
                    # Extract bits from index
                    emit_any_op(g, "valu", ("-", gA, v_idx[g], v_seven), reads=vr(v_idx[g])+vr(v_seven), writes=vr(gA))
                    emit_any_op(g, "valu", ("&", t_b0, gA, v_one), reads=vr(gA)+vr(v_one), writes=vr(t_b0))
                    emit_any_op(g, "valu", (">>", t_b1, gA, v_one), reads=vr(gA)+vr(v_one), writes=vr(t_b1))
                    emit_any_op(g, "valu", ("&", t_b1, t_b1, v_one), reads=vr(t_b1)+vr(v_one), writes=vr(t_b1))
                    emit_any_op(g, "valu", (">>", t_b2, gA, v_two), reads=vr(gA)+vr(v_two), writes=vr(t_b2))
                    emit_any_op(g, "valu", ("&", t_b2, t_b2, v_one), reads=vr(t_b2)+vr(v_one), writes=vr(t_b2))
                    
                    # Level 1 (pairs 01 and 23)
                    emit_any_op(g, "valu", ("multiply_add", t_M01, t_b0, v_diff[3][0], v_leaf[7]), reads=vr(t_b0)+vr(v_diff[3][0])+vr(v_leaf[7]), writes=vr(t_M01))
                    emit_any_op(g, "valu", ("multiply_add", t_M23, t_b0, v_diff[3][1], v_leaf[9]), reads=vr(t_b0)+vr(v_diff[3][1])+vr(v_leaf[9]), writes=vr(t_M23))
                    # Level 2 (pair 0123)
                    emit_any_op(g, "valu", ("-", t_diff, t_M23, t_M01), reads=vr(t_M23)+vr(t_M01), writes=vr(t_diff))
                    emit_any_op(g, "valu", ("multiply_add", t_M0123, t_b1, t_diff, t_M01), reads=vr(t_b1)+vr(t_diff)+vr(t_M01), writes=vr(t_M0123))
                    
                    # Level 1 (pairs 45 and 67)
                    t_M45 = t_M01
                    t_M67 = t_M23
                    emit_any_op(g, "valu", ("multiply_add", t_M45, t_b0, v_diff[3][2], v_leaf[11]), reads=vr(t_b0)+vr(v_diff[3][2])+vr(v_leaf[11]), writes=vr(t_M45))
                    emit_any_op(g, "valu", ("multiply_add", t_M67, t_b0, v_diff[3][3], v_leaf[13]), reads=vr(t_b0)+vr(v_diff[3][3])+vr(v_leaf[13]), writes=vr(t_M67))
                    # Level 2 (pair 4567)
                    t_M4567 = t_M67
                    emit_any_op(g, "valu", ("-", t_diff, t_M67, t_M45), reads=vr(t_M67)+vr(t_M45), writes=vr(t_diff))
                    emit_any_op(g, "valu", ("multiply_add", t_M4567, t_b1, t_diff, t_M45), reads=vr(t_b1)+vr(t_diff)+vr(t_M45), writes=vr(t_M4567))
                    
                    # Level 3 (final)
                    t_final = t_M45
                    emit_any_op(g, "valu", ("-", t_diff, t_M4567, t_M0123), reads=vr(t_M4567)+vr(t_M0123), writes=vr(t_diff))
                    emit_any_op(g, "valu", ("multiply_add", t_final, t_b2, t_diff, t_M0123), reads=vr(t_b2)+vr(t_diff)+vr(t_M0123), writes=vr(t_final))
                    
                    emit_any_op(g, "valu", ("^", v_val[g], v_val[g], t_final), reads=vr(v_val[g])+vr(t_final), writes=vr(v_val[g]))
                else:
                    node_addr = t_b0 # Reuse t_b0
                    node_val  = t_b1 # Reuse t_b1
                    emit_any_op(g, "valu", ("+", node_addr, v_forest_values_p, v_idx[g]), reads=vr(v_forest_values_p)+vr(v_idx[g]), writes=vr(node_addr))
                    for off in range(8):
                        emit_any_op(g, "load", ("load_offset", node_val, node_addr, off), reads=vr(node_addr), writes=[node_val + off])
                    emit_any_op(g, "valu", ("^", v_val[g], v_val[g], node_val), reads=vr(v_val[g])+vr(node_val), writes=vr(v_val[g]))
                
                # HASH STAGES
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    if hi == 0:
                        emit_any_op(g, "valu", ("multiply_add", v_val[g], v_val[g], v_hmul0, v_hash_val1[0]), reads=vr(v_val[g])+vr(v_hmul0)+vr(v_hash_val1[0]), writes=vr(v_val[g]))
                    elif hi == 2:
                        emit_any_op(g, "valu", ("multiply_add", v_val[g], v_val[g], v_hmul2, v_hash_val1[2]), reads=vr(v_val[g])+vr(v_hmul2)+vr(v_hash_val1[2]), writes=vr(v_val[g]))
                    elif hi == 4:
                        emit_any_op(g, "valu", ("multiply_add", v_val[g], v_val[g], v_hmul4, v_hash_val1[4]), reads=vr(v_val[g])+vr(v_hmul4)+vr(v_hash_val1[4]), writes=vr(v_val[g]))
                    else:
                        emit_any_op(g, "valu", (op3, vg, v_val[g], v_hash_val3[hi]), reads=vr(v_val[g])+vr(v_hash_val3[hi]), writes=vr(vg))
                        emit_any_op(g, "valu", (op1, gA, v_val[g], v_hash_val1[hi]), reads=vr(v_val[g])+vr(v_hash_val1[hi]), writes=vr(gA))
                        emit_any_op(g, "valu", (op2, v_val[g], gA, vg), reads=vr(gA)+vr(vg), writes=vr(v_val[g]))
                
                # INDEX UPDATE
                if rnd < rounds - 1:
                    emit_any_op(g, "valu", ("&", vg, v_val[g], v_one), reads=vr(v_val[g])+vr(v_one), writes=vr(vg))
                    emit_any_op(g, "valu", ("+", vg, vg, v_one), reads=vr(vg)+vr(v_one), writes=vr(vg))
                    emit_any_op(g, "valu", ("multiply_add", v_idx[g], v_idx[g], v_two, vg), reads=vr(v_idx[g])+vr(v_two)+vr(vg), writes=vr(v_idx[g]))
                    if depth == period - 1:
                        emit_any_op(g, "valu", ("<", vg, v_idx[g], v_n_nodes), reads=vr(v_idx[g])+vr(v_n_nodes), writes=vr(vg))
                        emit_any_op(g, "valu", ("*", v_idx[g], v_idx[g], vg), reads=vr(v_idx[g])+vr(vg), writes=vr(v_idx[g]))

        for g in range(32):
            emit_any_op(g, "store", ("vstore", s_addr_val[g], v_val[g]), reads=[s_addr_val[g]]+vr(v_val[g]), writes=[])

        # Schedule
        n = len(ops)
        last_writer = {}
        parents = [set() for _ in range(n)]
        children = [set() for _ in range(n)]
        for i, op in enumerate(ops):
            deps = set()
            for r in op["reads"]:
                if r in last_writer:
                    deps.add(last_writer[r])
            for w in op["writes"]:
                if w in last_writer:
                    deps.add(last_writer[w])
            for d in deps:
                parents[i].add(d)
                children[d].add(i)
            for w in op["writes"]:
                last_writer[w] = i

        heights = [0] * n
        for i in reversed(range(n)):
            if children[i]:
                heights[i] = max(heights[c] for c in children[i]) + 1
        
        in_degree = [len(p) for p in parents]
        ready = [i for i in range(n) if in_degree[i] == 0]
        scheduled_cycle = {}
        current_cycle = 0
        bundles = []

        while ready:
            cycle_ready = [idx for idx in ready if all(scheduled_cycle[p] < current_cycle for p in parents[idx])]
            def priority(idx):
                eng = ops[idx]["engine"]
                eng_pri = 2 if eng == "load" else (1 if eng == "store" else 0)
                g_stagger = (31 - (ops[idx].get("g") or 0)) * 5
                return (heights[idx] + g_stagger, eng_pri)
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

        for b in bundles:
            for eng, slots in b.items():
                for slot in slots:
                    asm.add(eng, slot)
            asm.emit()

        self.instrs.extend(asm.instrs)
