"""Transfer priorities from a verified packet calendar into a changed model."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import pickle


def keys(ir):
    counts=defaultdict(int)
    for op in ir.ops:
        tag,coords=op.tag
        imm=op.args[-1] if op.code in ('const','add_imm') else None
        values=tuple(ir.known.get(v) for v in op.args[0]) if op.engine!='store' else ()
        constant=values if values and all(v is not None for v in values) else None
        key=(op.engine,op.code,tag,tuple(coords),op.time,imm,constant)
        occurrence=counts[key];counts[key]+=1
        yield key,occurrence


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source',help='Verified prefix, with .pkl and .sol')
    ap.add_argument('target',help='Changed prefix, with .pkl and .txt')
    args=ap.parse_args()
    with open(args.source+'.pkl','rb') as stream:old=pickle.load(stream)
    with open(args.target+'.pkl','rb') as stream:new=pickle.load(stream)
    solution=list(map(int,Path(args.source+'.sol').read_text().split()))
    n,ng=solution[:2];assert (n,ng)==(len(old.ops),len(old.groups))
    times=solution[2:2+n]
    mapping=dict(zip(keys(old),times))
    lines=Path(args.target+'.txt').read_text().splitlines()
    assert int(lines[0].split()[0])==len(new.ops)
    matches=0
    for j,key in enumerate(keys(new)):
        if key in mapping:
            engine,_=lines[j+1].split()
            lines[j+1]=f'{engine} {mapping[key]}'
            matches+=1
    output=Path(args.target+'.warm.txt')
    output.write_text('\n'.join(lines)+'\n')
    print({'matched':matches,'total':len(new.ops),'output':str(output)})


if __name__=='__main__':main()
