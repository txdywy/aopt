"""Measure real three-vector pack orders with solver-free safe bounds."""
import argparse
from collections import Counter
from copy import deepcopy
from itertools import permutations
import json
from pathlib import Path

from codex.bound_boundary_no_solver863 import prefix,suffix
from codex.packet_budget_bound import stream_bounds
from codex.packet_modes import expand
from codex.research_triple_uniform863 import build
from codex.word_ssa import rebuild


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--groups',default='root,c0,c1',
                        help='three comma-separated names chosen from root,c0,c1,3,9,19')
    parser.add_argument('--direct-scalars',action='store_true')
    args=parser.parse_args()
    groups=tuple(args.groups.split(','))
    if len(groups)!=3 or len(set(groups))!=3:parser.error('three distinct group names required')
    report=dict(new_executable=False,results=[],groups=list(groups),
        scope='Six concrete triple-pack orders; source roots may require real '
              'extra ALUs. Bounds relax instruction packing and are not scores.')
    for order in permutations(groups):
        try:
            ir,_,layout=build(order,args.direct_scalars)
            counts=Counter(op.engine for op in ir.ops)
            work=counts['alu']+8*counts['valu']
            streams=stream_bounds(ir)
            expanded=expand(deepcopy(ir))
            expanded.views=[arg for _,args,_,_ in expanded.groups for arg in args]
            rebuild(expanded)
            head,tail=prefix(expanded),suffix(expanded)
            bound=max((work+head['waste_lower_bound']+tail['waste_lower_bound']+59)//60,
                      *(v['cycle_lower_bound'] for v in streams.values()))
            row=dict(order=list(order),layout=layout,counts=dict(counts),arithmetic_lanes=work,
                prefix=head,suffix=tail,streams=streams,cycle_lower_bound=bound,
                status='necessary_bound_only',new_executable=False)
        except (AssertionError,ValueError) as error:
            row=dict(order=list(order),status='invalid_graph',reason=str(error),new_executable=False)
        report['results'].append(row)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:row.get(k) for k in ('order','status','arithmetic_lanes','cycle_lower_bound')}),flush=True)


if __name__=='__main__':main()
