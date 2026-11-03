"""Partition the remaining live-input MAD,bitwise,MAD operand roles.

The final MAD must use u and at least one of b,a,prior,t: sole-u cases have
separate certificates. If the middle operation bypasses t it can be moved
to the already-excluded bitwise-prefix order. This leaves 58 final roles
per opcode. Every bounded result is saved, including UNKNOWN and low-bit SAT.
"""
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path

from codex.search_live_middle863 import search


def roles():
    # Map seven conceptual operands to the existing SSA pool including zero.
    mapping=(0,1,2,4,5,6,7)  # b,a,prior,t,u,k0,k1
    return [tuple(mapping[i] for i in (a,b,c))
            for a,b in itertools.combinations_with_replacement(range(7),2)
            for c in range(7) if 4 in (a,b,c) and any(i < 4 for i in (a,b,c))]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout-ms',type=int,default=1000)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    rows=json.loads(args.output.read_text())['rows'] if args.resume and args.output.exists() else []
    for op,role in itertools.product(('^','&','|'),roles()):
        if any(r['sequence'][1] == op and r['last_mad_role'] == list(role) for r in rows):
            continue
        row=search(['mad',op,'mad'],11,args.timeout_ms,
                   last_mad_role=role,middle_uses_first=True)
        rows.append(row)
        report={'width':11,'timeout_ms':args.timeout_ms,'roles_per_opcode':len(roles()),
                'counts':dict(Counter(r['result'] for r in rows)),
                'scope':'Only final MAD using u and another live input; first MAD arbitrary, '
                        'middle u=t OP b/a/prior/K. UNKNOWN is inconclusive.', 'rows':rows}
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'op':op,'role':role,'result':row['result'],
                          'verification':row.get('verification'),'seconds':row['seconds']}),flush=True)


if __name__ == '__main__':
    main()
