import os, sys
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("tests"))

with open("perf_takehome.py", "r") as f:
    code = f.read()

code = code.replace(
    'def emit_op(engine, slot, reads=None, writes=None, g=None):',
    'def emit_op(engine, slot, reads=None, writes=None, g=None):\n            if reads is None: reads = []\n            if writes is None: writes = []\n            ops.append({"engine": engine, "slot": slot, "reads": reads, "writes": writes, "g": g, "rnd": current_rnd, "depth": current_depth})\n            return'
)
code = code.replace(
    '            ops.append({"engine": engine, "slot": slot, "reads": reads, "writes": writes, "g": g})',
    '            pass'
)
code = code.replace(
    'def emit_round_group(rnd, g):',
    'def emit_round_group(rnd, g):\n            nonlocal current_rnd, current_depth\n            current_rnd = rnd\n            current_depth = rnd % period'
)
code = code.replace(
    'ops = []\n        current_group = None',
    'ops = []\n        current_group = None\n        current_rnd = 0\n        current_depth = 0'
)

# And add the print in schedule_ops
new_lines = []
for line in code.splitlines():
    if "scheduled_cycle[op_idx] = current_cycle" in line:
        new_lines.append(line)
        new_lines.append("                    if ops[op_idx].get('g') == 0 and ops[op_idx].get('rnd') == 4:\n")
        new_lines.append("                        print(f\"Group 0 Rnd 4: Op={ops[op_idx].get('slot')}, parents={[ops[p].get('slot') for p in parents[op_idx]]}, cycle={current_cycle}\")\n")
    else:
        new_lines.append(line)

with open("temp_perf6.py", "w") as f:
    f.write("\n".join(new_lines))

import temp_perf6
kb = temp_perf6.KernelBuilder()
kb.build_kernel(10, 16, 256, 16)
