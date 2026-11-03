"""Build direct d9 payload loads with the actual d10 address compensation.

The d10 loads in the baseline are already scalar. Advancing d9 pointers by
another three words allows scalar d9 loads, but d10 then needs bit-6. This
implementation charges the real AND, OR and -6 mask rather than treating
state OR -6 as a parity extractor (that expression is incorrect).
"""
from collections import Counter
from copy import deepcopy
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from codex import research_flow_stage863 as flow
from codex import research_frontier863 as frontier
from codex import research_stage864 as stage
from codex import research_sub864 as research
from codex.word_ssa import kinds


def build():
    b=deepcopy(flow.original())
    writers={v:op for op in b.ops if op.engine!='store'for v in op.args[0]}
    # These four roots only feed the d9 selector tree and their staged packs.
    scalars={v:b.scalar[v]for v in range(1029,1033)}
    vectors={v:b.vectors[v]for v in range(1029,1033)}
    for source in scalars.values():
        op=writers[source[0]]
        assert op.code in('const','add_imm')
        op.args[-1]+=3
    for mapping,old in((b.scalar,scalars),(b.vectors,vectors)):
        for value in old:mapping.pop(value)
        mapping.update({value+3:source for value,source in old.items()})
    mask=b.new(8)
    b.put('valu','-',[mask,b.vectors[0xfffffffe],b.vectors[4]],10,('d9_direct_mask',()))
    b.vectors[0xfffffffa]=mask
    replacements={}
    for g in range(32):
        gathers=sorted((op for op in b.ops if op.tag[0]=='raw_gather'and op.tag[1][:2]==(9,g)),
                       key=lambda op:op.tag[1][2])
        assert len(gathers)==8 and all(op.code=='vload'for op in gathers)
        for op in gathers:
            op.code,op.args,op.kinds='load',[(op.args[0][3],),op.args[1]],kinds('load','load')
        next_load=next(op for op in b.ops if op.tag==('raw_gather',(10,g,0)))
        ptr=writers[next_load.args[1][0]]
        assert ptr.code=='multiply_add'
        parity=ptr.args[3]
        assert writers[parity[0]].code=='&'
        biased=b.new(8)
        b.put('valu','|',[biased,parity,mask],ptr.time-1,('d9_direct_compensation',(g,)))
        ptr.args[3]=biased
        rows=[op for op in b.ops if op.tag==('node_xor',(9,g))]
        assert len(rows)==8 and all(op.engine=='alu'for op in rows)
        args=[tuple(op.args[i][0]for op in rows)for i in range(3)]
        vector=SimpleNamespace(engine='valu',code='^',args=args,kinds=kinds('valu','^'),
            time=min(op.time for op in rows),tag=('node_xor',(9,g)))
        replacements[id(rows[0])]=vector
        replacements.update({id(op):None for op in rows[1:]})
    b.ops=[replacements.get(id(op),op)for op in b.ops
           if id(op)not in replacements or replacements[id(op)]is not None]
    frontier.logical_order(b)
    b.sink=b.literals=b.literal_flow=0
    for _ in range(3):
        stage.prune_select_inputs(b)
        ir=research.clean(b)
    return ir,b


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--seeds',type=int,default=2)
    args=ap.parse_args()
    if args.seeds<1:ap.error('positive replay seeds required')
    ir,_=build()
    counts=Counter(op.engine for op in ir.ops)
    lanes=counts['alu']+8*counts['valu']
    report=dict(counts=dict(counts),arithmetic_lanes=lanes,
        arithmetic_only_cycle_lower_bound=(lanes+59)//60,
        scalar_d9_loads=sum(op.tag[0]=='raw_gather'and op.tag[1][0]==9 and op.code=='load'for op in ir.ops),
        d9_vector_node_xors=sum(op.tag[0]=='node_xor'and op.tag[1][0]==9 and op.engine=='valu'for op in ir.ops),
        bias_mask=0xfffffffa,wrong_shortcut_counterexample=dict(state=4,
            shortcut=4|0xfffffffa,correct=((4&1)-6)&0xffffffff),
        seeds=args.seeds,serial_control=flow.serial_control(ir,args.seeds),new_executable=False,
        scope='This direct-load graph with an explicit extra bit-6 instruction per group; '
              'the arithmetic bound ignores startup/drain and is not a scheduled score.')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
