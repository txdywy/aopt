"""Refine UNKNOWN final roles by exact middle source and constant-name symmetry."""
import argparse
from collections import Counter
import json
from pathlib import Path

from codex.search_live_middle863 import search


def canonical(role):
    # Only these residual roles contain one final constant. Rename it k0.
    if 6 in role and 7 in role:
        raise ValueError('two distinct constants cannot be collapsed')
    a,b,c=(6 if i == 7 else i for i in role)
    return (*sorted((a,b)),c)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--timeout-ms',type=int,default=3000)
    args=parser.parse_args()
    source=json.loads(args.input.read_text())
    cases=sorted({(r['sequence'][1],canonical(r['last_mad_role'])) for r in source['rows']
                  if r['result']=='unknown'})
    rows=[]
    for op,role in cases:
        for other in (0,1,2,5):
            row=search(['mad',op,'mad'],11,args.timeout_ms,last_mad_role=role,
                       middle_uses_first=True,middle_other=other)
            rows.append(row)
            result={'width':11,'source':str(args.input),'canonical_unknown_roles':len(cases),
                    'counts':dict(Counter(r['result'] for r in rows)), 'rows':rows}
            args.output.write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps({'op':op,'role':role,'other':other,'result':row['result'],
                              'verification':row.get('verification'),'seconds':row['seconds']}),flush=True)


if __name__ == '__main__':
    main()
