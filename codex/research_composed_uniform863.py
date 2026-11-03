"""Combine actual uniform packing with d7 FLOW trades and stage reloads."""
from collections import Counter
from copy import deepcopy
import argparse
import json
from pathlib import Path

from codex import research_flow_stage863 as flow
from codex import research_frontier863 as frontier
from codex import research_stage864 as stage
from codex import research_sub864 as research
from codex.bound_boundary_no_solver863 import prefix,suffix
from codex.packet_budget_bound import stream_bounds
from codex.packet_modes import expand
from codex.research_joint_ones863 import NAMES, pack as pack_ones
from codex.research_triple_uniform863 import triple_pack
from codex.word_ssa import rebuild


def build(kind, names, staged=(), thirds=(), fused=0, trades=0, load4097=False):
    if not 0<=fused<=32:raise ValueError('fused must be between 0 and 32')
    b=deepcopy(flow.original())
    if load4097:flow.early_4097(b)
    flow.fuse_digits(b,list(range(fused)))
    layout=(pack_ones(b, names) if kind=='ones' else triple_pack(b,names))
    frontier.logical_order(b)
    methods={value:'stage' for value in staged}
    if set(methods)&set(thirds):raise ValueError('a uniform cannot use both methods')
    methods.update({value:'third' for value in thirds})
    if 1 in methods:raise ValueError('the ones vector is already part of the mask')
    if methods:
        # The joint pack places its final nonzero constant next to the zeros;
        # using the original ones view as a mixed mask would impose an
        # impossible second physical layout. FLOW accepts any nonzero value.
        original_ones=b.vectors[1]
        if kind=='ones':b.vectors[1]=b.vectors[NAMES[names[-1]]]
        try:flow.materialize_uniforms(b,methods)
        finally:b.vectors[1]=original_ones
    flow.trade_loads(b,trades,include_input=trades>=4)
    frontier.logical_order(b)
    b.sink=b.literals=b.literal_flow=0
    for _ in range(3):
        stage.prune_select_inputs(b)
        ir=research.clean(b)
    return ir,b,layout


def parse_values(s):
    return tuple('root' if x=='root' else int(x,0) for x in s.split(',') if x)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kind',choices=('ones','triple'),default='ones')
    ap.add_argument('--names',default='c0')
    ap.add_argument('--staged',default='')
    ap.add_argument('--thirds',default='')
    ap.add_argument('--fused',type=int,default=0)
    ap.add_argument('--trades',type=int,default=0)
    ap.add_argument('--load4097',action='store_true')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--controls',action='store_true')
    args=ap.parse_args()
    cfg=dict(kind=args.kind,names=tuple(args.names.split(',')),
             staged=parse_values(args.staged),thirds=parse_values(args.thirds),
             fused=args.fused,trades=args.trades,load4097=args.load4097)
    ir,b,layout=build(**cfg)
    counts=Counter(op.engine for op in ir.ops)
    work=counts['alu']+8*counts['valu'];streams=stream_bounds(ir)
    expanded=expand(deepcopy(ir))
    expanded.views=[a for _,operands,_,_ in expanded.groups for a in operands]
    rebuild(expanded)
    head,tail=prefix(expanded),suffix(expanded)
    lower=max((work+head['waste_lower_bound']+tail['waste_lower_bound']+59)//60,
              *(v['cycle_lower_bound'] for v in streams.values()))
    result=dict(config=cfg,layout=layout,counts=dict(counts),arithmetic_lanes=work,
                prefix=head,suffix=tail,streams=streams,cycle_lower_bound=lower,
                new_executable=False,scope='Only this explicit combined graph; bounds and serial replay are not scores.')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('arithmetic_lanes','cycle_lower_bound','prefix','streams')}),flush=True)
    if args.controls:
        control=flow.serial_control(deepcopy(ir),2)
        result['serial_control']=dict(scratch=control['scratch'],
            serialized_cycles=control['verified'],seeds=2)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'serial_control':result['serial_control']}),flush=True)


if __name__=='__main__':main()
