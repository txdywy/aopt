"""Bound raw d8 records under arbitrary child swaps and arithmetic copies.

The existing 64 vloads and vstores are retained. All four raw children in a
two-record vstore come from consecutive lanes of one original vload, while
the four destination slots have a two-word gap. Optimistically permit any
permutation and any offset between their two physical scratch views.
"""
import argparse
from collections import Counter
from itertools import permutations
import json
from pathlib import Path
from types import SimpleNamespace

from codex import research_stage864 as stage
from codex.scan_stage863 import baseline


def bound():
    ir,builder=stage.build(SimpleNamespace(**baseline()))
    writers={v:op for op in builder.ops if op.engine!='store' for v in op.args[0]}
    stores=[op for op in builder.ops if op.tag[0]=='d7_image']
    assert len(stores)==64
    cases=[]
    for op in stores:
        payload=list(op.args[1]);slots=(0,1,4,5)
        encoded=[writers[payload[i]] for i in slots]
        assert all(row.engine=='alu' and row.code=='^' and
                   row.args[2]==builder.scalar[0xb55a4f09] for row in encoded)
        raw=[row.args[1][0] for row in encoded]
        source=writers[raw[0]]
        assert source.code=='vload' and all(writers[value] is source for value in raw)
        offsets=tuple(source.args[0].index(value) for value in raw)
        assert len(set(offsets))==4 and max(offsets)-min(offsets)==3
        # Extra freedom: even swap children across records, which the actual
        # gather semantics may not allow. This can only increase the bound.
        maximum=max(sum(int(source_lane==destination+shift)
                        for source_lane,destination in zip(order,slots))
                    for order in permutations(offsets) for shift in range(-8,9))
        assert maximum==2
        cases.append(dict(tag=list(op.tag[1]),source_tag=list(source.tag[1]),
                          source_offsets=list(offsets),destination_slots=list(slots),
                          optimistic_direct_aliases=maximum))
    counts=Counter(op.engine for op in ir.ops)
    base=counts['alu']+8*counts['valu']
    removed_xors=4*len(stores)
    added_state_restore=32*8
    minimum_copies=removed_xors-sum(row['optimistic_direct_aliases'] for row in cases)
    assert base==51665 and removed_xors==added_state_restore==256 and minimum_copies==128
    work=base-removed_xors+added_state_restore+minimum_copies
    return dict(executable_best_cycles=864,new_executable=False,records=len(stores),
        baseline_arithmetic_lanes=base,removed_child_encoding_xors=removed_xors,
        added_state_restore_lanes=added_state_restore,optimistic_minimum_copy_lanes=minimum_copies,
        arithmetic_lanes_lower_bound=work,arithmetic_only_cycle_lower_bound=(work+59)//60,
        cases=cases,result='863_excluded_for_arithmetic_copy_family',
        scope='Retain the 864 graph, its 64 eight-lane raw vloads, two stride-four '
              'records per eight-word vstore and fixed node-XOR semantics; children may '
              'be permuted arbitrarily within a store. Copies use ALU/VALU, and all '
              'other graph costs are held fixed. FLOW copies, scalar loads, changed stride, '
              'other address/hash/layout changes and kernel-wide optimum are not covered.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=bound();args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('records','optimistic_minimum_copy_lanes',
        'arithmetic_lanes_lower_bound','arithmetic_only_cycle_lower_bound')}))


if __name__=='__main__':main()
