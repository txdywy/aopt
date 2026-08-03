"""Clamp normalized engine hints to exact full-DAG release/deadline windows."""
import json, os
from pathlib import Path
import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops
from codex.solve_full_dag import topological_data

def main():
    configure_target(); kernel.SCHEDULE_EXACT_CYCLES=None; b=kernel.KernelBuilder()
    try:b.build_kernel(10,2047,256,16)
    except (AssertionError,StopIteration):
        if not hasattr(b,'dag_ops'): raise
    ops=real_tail_ops(b.dag_ops); _,_,earliest,tail=topological_data(ops)
    horizon=int(os.environ.get('TARGET','959')); p=Path(os.environ['SOURCE']); d=json.loads(p.read_text())
    raw_cycles = d['cycles']
    if isinstance(raw_cycles, list):
        out = [
            min(horizon - 1 - tail[i], max(earliest[i], int(cycle)))
            for i, cycle in enumerate(raw_cycles)
        ]
    else:
        out={}
        for raw,cycle in raw_cycles.items():
            i=int(raw); out[str(i)]=min(horizon-1-tail[i],max(earliest[i],int(cycle)))
    q=dict(d); q['cycles']=out; q['horizon']=horizon
    Path(os.environ['OUT']).write_text(json.dumps(q)); print('clamped',p,'->',os.environ['OUT'])
if __name__=='__main__':main()
