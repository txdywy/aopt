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

        # Allocations
        init_vars = [
            "s_addr_in_idx", "s_addr_in_val", "s_addr_out_val", "s_addr_forest",
            "s_batch_size", "s_n_nodes", "s_c1", "s_c2"
        ]
        s_vars = {v: self.alloc_scratch(v) for v in init_vars}
        
        s_tmp   = self.alloc_scratch("s_tmp")
        s_tmp2  = self.alloc_scratch("s_tmp2")
        s_zero  = self.alloc_scratch("s_zero")
        s_one   = self.alloc_scratch("s_one")
        s_two   = self.alloc_scratch("s_two")
        
        s_addr_idx = [self.alloc_scratch(f"s_addr_idx_{g}") for g in range(32)]
        s_addr_val = [self.alloc_scratch(f"s_addr_val_{g}") for g in range(32)]
        
        v_one  = self.alloc_scratch("v_one",  8)
        v_two  = self.alloc_scratch("v_two",  8)
        v_n_nodes = self.alloc_scratch("v_n_nodes", 8)
        v_forest_values_p = self.alloc_scratch("v_forest_values_p", 8)
        
        v_idx = [self.alloc_scratch(f"v_idx_{g}", 8) for g in range(32)]
        v_val = [self.alloc_scratch(f"v_val_{g}", 8) for g in range(32)]
        
        # Shared tmp
        v_tmp = [self.alloc_scratch(f"v_tmp_{g}", 8) for g in range(8)]
        # Temps for vselect
        v_sel = [self.alloc_scratch(f"v_sel_{g}", 8) for g in range(8)]
        
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
        
        v_shifts = [self.alloc_scratch(f"v_shift_{i}", 8) for i in range(MAX_OPT_ROUND)]
        v_offsets = [self.alloc_scratch(f"v_offset_{i}", 8) for i in range(MAX_OPT_ROUND)]

        # INITIALIZATION
        asm.add("flow", ("add_imm", s_vars["s_addr_in_idx"], 0, 0))
        asm.add("flow", ("add_imm", s_vars["s_addr_in_val"], 0, 1))
        asm.add("flow", ("add_imm", s_vars["s_addr_out_val"], 0, 2))
        asm.add("flow", ("add_imm", s_vars["s_addr_forest"], 0, 3))
        asm.add("flow", ("add_imm", s_vars["s_batch_size"], 0, 4))
        asm.add("flow", ("add_imm", s_vars["s_n_nodes"], 0, 5))
        asm.add("flow", ("add_imm", s_vars["s_c1"], 0, 6))
        asm.add("flow", ("add_imm", s_vars["s_c2"], 0, 7))
        asm.emit()

        for v in init_vars:
            asm.add("load", ("load", s_vars[v], s_vars[v]))
        asm.emit()
        asm.add("flow", ("pause",))
        asm.emit()

        asm.add("load", ("const", s_zero, 0))
        asm.add("load", ("const", s_one, 1))
        asm.add("load", ("const", s_two, 2))
        asm.emit()

        asm.add("valu", ("vbroadcast", v_one, s_one))
        asm.add("valu", ("vbroadcast", v_two, s_two))
        asm.add("valu", ("vbroadcast", v_n_nodes, s_vars["s_n_nodes"]))
        asm.add("valu", ("vbroadcast", v_forest_values_p, s_vars["s_addr_forest"]))
        for g in range(32):
            asm.add("flow", ("add_imm", s_tmp, s_zero, g * 8))
            asm.add("flow", ("add_imm", s_addr_idx[g], s_vars["s_addr_in_idx"], g * 8))
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
        
        # hmul constants
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

        # PRELOAD LEAVES
        for i in range(n_leaves):
            asm.add("flow", ("add_imm", s_tmp, s_vars["s_addr_forest"], i))
            asm.emit()
            asm.add("load", ("load", s_leaves[i], s_tmp))
            asm.emit()
            asm.add("valu", ("vbroadcast", v_leaf[i], s_leaves[i]))
            asm.emit()
            
        for i in range(MAX_OPT_ROUND):
            asm.add("load", ("const", s_tmp, i))
            asm.add("load", ("const", s_tmp2, (1<<i)-1))
            asm.emit()
            asm.add("valu", ("vbroadcast", v_shifts[i], s_tmp))
            asm.add("valu", ("vbroadcast", v_offsets[i], s_tmp2))
            asm.emit()

        asm.add("flow", ("pause",))
        asm.emit()

        ops = []
        def emit_op(engine, slot, reads=None, writes=None):
            if reads is None: reads = []
            if writes is None: writes = []
            ops.append({"engine": engine, "slot": slot, "reads": reads, "writes": writes})

        def vr(reg):
            return list(range(reg, reg + 8))

        # Phase 1: Load all indices and values
        for g in range(32):
            emit_op("load", ("vload", v_idx[g], s_addr_idx[g]), reads=[s_addr_idx[g]], writes=vr(v_idx[g]))
            emit_op("load", ("vload", v_val[g], s_addr_val[g]), reads=[s_addr_val[g]], writes=vr(v_val[g]))

        period = forest_height + 1
        
        for rnd in range(rounds):
            for g in range(32):
                depth = rnd % period
                vg = v_tmp[g % 8]
                gA = v_tmp[(g + 1) % 8]
                gB = v_tmp[(g + 2) % 8]
                
                # TREE LOOKUP
                if depth < MAX_OPT_ROUND:
                    if depth == 0:
                        emit_op("valu", ("^", v_val[g], v_val[g], v_leaf[0]), reads=vr(v_val[g])+vr(v_leaf[0]), writes=vr(v_val[g]))
                    else:
                        num_leaves = 1 << depth
                        start_idx = num_leaves - 1
                        
                        # rel_idx = v_idx - offset
                        emit_op("valu", ("-", vg, v_idx[g], v_offsets[depth]), reads=vr(v_idx[g])+vr(v_offsets[depth]), writes=vr(vg))
                        
                        # Extract bits and select
                        current_level_vars = [v_leaf[start_idx + i] for i in range(num_leaves)]
                        
                        for level in range(depth):
                            # extract bit
                            if level == 0:
                                emit_op("valu", ("&", gA, vg, v_one), reads=vr(vg)+vr(v_one), writes=vr(gA))
                            else:
                                emit_op("valu", (">>", gB, vg, v_shifts[level]), reads=vr(vg)+vr(v_shifts[level]), writes=vr(gB))
                                emit_op("valu", ("&", gA, gB, v_one), reads=vr(gB)+vr(v_one), writes=vr(gA))
                            
                            next_level_vars = []
                            for i in range(0, len(current_level_vars), 2):
                                # If vselect uses condition gA, when gA != 0 (i.e. bit is 1), it picks right child (idx+1).
                                # When gA == 0 (bit is 0), it picks left child (idx).
                                # So vselect(dest, gA, right, left)
                                dest_var = v_sel[i//2] if level < depth - 1 else gA # write to gA in last level
                                left = current_level_vars[i]
                                right = current_level_vars[i+1]
                                emit_op("flow", ("vselect", dest_var, gA, right, left), reads=vr(gA)+vr(right)+vr(left), writes=vr(dest_var))
                                next_level_vars.append(dest_var)
                            current_level_vars = next_level_vars
                        
                        emit_op("valu", ("^", v_val[g], v_val[g], gA), reads=vr(v_val[g])+vr(gA), writes=vr(v_val[g]))
                else:
                    node_addr = vg
                    node_val  = gA
                    emit_op("valu", ("+", node_addr, v_forest_values_p, v_idx[g]), reads=vr(v_forest_values_p)+vr(v_idx[g]), writes=vr(node_addr))
                    for off in range(8):
                        emit_op("load", ("load_offset", node_val, node_addr, off), reads=vr(node_addr), writes=[node_val + off])
                    emit_op("valu", ("^", v_val[g], v_val[g], node_val), reads=vr(v_val[g])+vr(node_val), writes=vr(v_val[g]))
                
                # HASH STAGES
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    if hi == 0:
                        emit_op("valu", ("multiply_add", v_val[g], v_val[g], v_hmul0, v_hash_val1[0]), reads=vr(v_val[g])+vr(v_hmul0)+vr(v_hash_val1[0]), writes=vr(v_val[g]))
                    elif hi == 2:
                        emit_op("valu", ("multiply_add", v_val[g], v_val[g], v_hmul2, v_hash_val1[2]), reads=vr(v_val[g])+vr(v_hmul2)+vr(v_hash_val1[2]), writes=vr(v_val[g]))
                    elif hi == 4:
                        emit_op("valu", ("multiply_add", v_val[g], v_val[g], v_hmul4, v_hash_val1[4]), reads=vr(v_val[g])+vr(v_hmul4)+vr(v_hash_val1[4]), writes=vr(v_val[g]))
                    else:
                        emit_op("valu", (op3, vg, v_val[g], v_hash_val3[hi]), reads=vr(v_val[g])+vr(v_hash_val3[hi]), writes=vr(vg))
                        emit_op("valu", (op1, gA, v_val[g], v_hash_val1[hi]), reads=vr(v_val[g])+vr(v_hash_val1[hi]), writes=vr(gA))
                        emit_op("valu", (op2, v_val[g], gA, vg), reads=vr(gA)+vr(vg), writes=vr(v_val[g]))
                
                # INDEX UPDATE
                if rnd < rounds - 1:
                    emit_op("valu", ("&", vg, v_val[g], v_one), reads=vr(v_val[g])+vr(v_one), writes=vr(vg))
                    emit_op("valu", ("+", vg, vg, v_one), reads=vr(vg)+vr(v_one), writes=vr(vg))
                    emit_op("valu", ("multiply_add", v_idx[g], v_idx[g], v_two, vg), reads=vr(v_idx[g])+vr(v_two)+vr(vg), writes=vr(v_idx[g]))
                    if depth == period - 1:
                        emit_op("valu", ("<", vg, v_idx[g], v_n_nodes), reads=vr(v_idx[g])+vr(v_n_nodes), writes=vr(vg))
                        emit_op("valu", ("*", v_idx[g], v_idx[g], vg), reads=vr(v_idx[g])+vr(vg), writes=vr(v_idx[g]))

        for g in range(32):
            emit_op("store", ("vstore", s_addr_val[g], v_val[g]), reads=[s_addr_val[g]]+vr(v_val[g]), writes=[])

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
                return (heights[idx], eng_pri)
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
