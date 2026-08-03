"""Split a full schedule JSON into engine hint payloads (research utility)."""
import json, os
from pathlib import Path
import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops

def main():
    configure_target(); kernel.SCHEDULE_EXACT_CYCLES=None
    b=kernel.KernelBuilder()
    try:b.build_kernel(10,2047,256,16)
    except (AssertionError,StopIteration):
        if not hasattr(b,'dag_ops'): raise
    ops=real_tail_ops(b.dag_ops)
    p=Path(os.environ['SOURCE']); data=json.loads(p.read_text()); raw=data['cycles']
    cycles = raw if isinstance(raw, list) else {int(i): int(c) for i,c in raw.items()}
    prefix=Path(os.environ.get('OUT_PREFIX',str(p.with_suffix(''))))
    for eng in ('alu','valu','load','store','flow','debug'):
        vals={str(i):int(cycles[i]) for i,o in enumerate(ops) if o.engine==eng}
        if not vals: continue
        horizon = max(cycles if isinstance(cycles, list) else cycles.values()) + 1
        out=Path(f'{prefix}-{eng}.json'); out.write_text(json.dumps({'engine':eng,'horizon':horizon,'cycles':vals})); print(out)
if __name__=='__main__': main()
