"""Test scalar output streaming against the 864 record-kernel boundaries.

Each scalar store has its actual address producer and unchanged ISA semantics.
No submitted kernel or saved schedule is modified.
"""
from copy import deepcopy
import argparse
import json
from pathlib import Path

from codex.research_flow_stage863 import original,measure,serial_control
from codex import research_stage864 as stage
from codex import research_frontier863 as frontier
from codex import research_sub864 as research


def build(count):
    if count not in range(33):raise ValueError('zero to 32 output groups required')
    builder=deepcopy(original())
    added=[]
    for op in builder.ops:
        if op.tag[0]!='output' or op.tag[1][0]<32-count:
            added.append(op);continue
        address,values=op.args
        for lane,value in enumerate(values):
            if lane:
                pointer=builder.new(1)
                builder.put('alu','+',[pointer,address,builder.scalar[1]],op.time-1,
                    ('scalar_output_pointer',(op.tag[1][0],lane)))
                added.append(builder.ops.pop())
                address=pointer
            builder.put('store','store',[address,(value,)],op.time,
                ('output',(op.tag[1][0],lane)))
            added.append(builder.ops.pop())
    builder.ops=added
    frontier.logical_order(builder)
    builder.sink=builder.literals=builder.literal_flow=0
    for _ in range(3):
        stage.prune_select_inputs(builder)
        ir=research.clean(builder)
    return ir,builder


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--controls',action='store_true')
    args=ap.parse_args()
    report=[]
    for count in (0,1,2,8,32):
        ir,builder=build(count)
        # measure expands the IR in place; serial replay needs the original.
        row=dict(scalar_groups=count,**measure(deepcopy(ir)))
        if args.controls and count in (1,32):row['serial_control']=serial_control(ir,1)
        report.append(row)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(groups=count,work=row['arithmetic_lanes'],
            suffix=row['suffix']['waste_lower_bound'],bound=row['cycle_lower_bound'])),flush=True)


if __name__=='__main__':main()
