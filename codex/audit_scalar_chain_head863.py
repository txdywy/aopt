"""A two-cycle mode-aware upper bound for an expanded candidate graph.

Only modes already present in packet_modes.expand are counted. Newly imposed
operand layouts or a different instruction graph are outside this certificate.
"""
from collections import Counter
from copy import deepcopy
import argparse
import json
from pathlib import Path
import pickle

from codex.bound_boundary_no_solver863 import order_and_distance,suffix
from codex.packet_budget_bound import lane_weight,stream_bounds
from codex.packet_modes import expand
from codex.research_scalar_chain863 import build
from codex.word_ssa import rebuild


def first_two(ir):
    _,dist=order_and_distance(ir.deps,ir.succ)
    first={j for j,op in enumerate(ir.ops)
           if op.engine in ('alu','valu') and dist[j]==0}
    weight=sum(lane_weight(ir,j) for j in first)
    if weight!=8:
        return dict(applicable=False,first_cycle_work_upper=weight)
    # An 863 calendar with the known 116-lane drain bound and C=51612
    # cannot afford to leave any of these eight lanes unexecuted at t=0.
    native=[j for j,op in enumerate(ir.ops)
            if op.engine=='valu' and dist[j]<=1 and j not in first]
    scalar=[j for j,op in enumerate(ir.ops)
            if op.engine=='alu' and dist[j]<=1 and j not in first]
    groups=[g for g,(_,_,ids,_) in enumerate(ir.groups)
            if not first.intersection(ids) and max(dist[j] for j in ids)<=1]
    # Double-counting grouped scalar lanes and overlapping groups is an
    # intentional relaxation. Non-arithmetic resource limits are ignored.
    upper=min(6,len(native)+len(groups))*8+min(12,len(scalar))
    return dict(applicable=True,first_cycle_work_upper=8,
        second_native_valu_count=len(native),second_vectorizable_group_count=len(groups),
        second_scalar_lane_count=len(scalar),second_cycle_work_upper=upper,
        first_two_waste_lower_bound=120-8-upper,
        first_tags=[list(ir.ops[j].tag) for j in sorted(first)],
        native_tags=[list(ir.ops[j].tag) for j in native],
        group_tags=[list(ir.ops[ir.groups[g][2][0]].tag) for g in groups],
        scope='Only existing exported scalar/vector modes; overlapping candidate '
              'groups and scalar/vector slots are freely double-counted.')


def audit(ir):
    counts=Counter(op.engine for op in ir.ops)
    work=counts['alu']+8*counts['valu']
    streams=stream_bounds(ir)
    expanded=expand(deepcopy(ir))
    expanded.views=[a for _,operands,_,_ in expanded.groups for a in operands]
    rebuild(expanded)
    head=first_two(expanded);tail=suffix(expanded)
    if not head['applicable'] or max(v['cycle_lower_bound'] for v in streams.values())<6:
        raise ValueError('The fixed first-cycle and non-overlap assumptions do not hold')
    # A first cycle below eight alone gives at least 53 wasted slots. If all
    # eight execute, the more restrictive second-cycle capacity applies.
    first_case=(work+53+tail['waste_lower_bound']+59)//60
    full_case=(work+head['first_two_waste_lower_bound']+tail['waste_lower_bound']+59)//60
    lower=max(min(first_case,full_case),*(v['cycle_lower_bound'] for v in streams.values()))
    return dict(arithmetic_lanes=work,head=head,suffix=tail,streams=streams,
        cycle_lower_bound=lower,new_executable=False,
        scope='This candidate and these explicit packet_modes groups only; '
              'new group layouts, graph rewrites or different programs are not excluded.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    source=ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--ir',type=Path)
    source.add_argument('--candidate',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.ir:
        with args.ir.open('rb') as f:ir=pickle.load(f)
    else:
        config=json.loads(args.candidate.read_text())['config']
        ir,_,_=build(**config)
    row=audit(ir);args.output.write_text(json.dumps(row,indent=2)+'\n')
    print(json.dumps({k:row[k] for k in ('arithmetic_lanes','head','cycle_lower_bound')}))


if __name__=='__main__':main()
