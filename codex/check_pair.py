import collections
import json

import codex.perf_takehome_under1000 as k
from codex.map_variant_schedule import configure_target, real_tail_ops

configure_target()
k.SCHEDULE_EXACT_CYCLES = None
b = k.KernelBuilder()
try:
    b.build_kernel(10, 2047, 256, 16)
except Exception:
    pass
ops = real_tail_ops(b.dag_ops)
cycles = {}
for path in (
    "/private/tmp/valu-conditioned-on-flow.json",
    "/private/tmp/constload-flow959.json",
):
    payload = json.loads(open(path).read())
    cycles.update({int(index): int(cycle) for index, cycle in payload["cycles"].items()})
print("counts", len(cycles), collections.Counter(ops[index].engine for index in cycles))
violations = []
for child, op in enumerate(ops):
    if child not in cycles:
        continue
    for parent, lag in op.parents.items():
        if parent in cycles and cycles[child] < cycles[parent] + lag:
            violations.append((cycles[parent] + lag - cycles[child], parent, child, lag))
print("direct_pair_edge_violations", len(violations), sorted(violations, reverse=True)[:10])
for engine, capacity in (("valu", 6), ("flow", 1)):
    usage = collections.Counter(
        cycles[index]
        for index, op in enumerate(ops)
        if index in cycles and op.engine == engine
    )
    print(
        engine,
        "minmax",
        min(usage),
        max(usage),
        "overflow",
        [(cycle, count) for cycle, count in usage.items() if count > capacity][:8],
    )
