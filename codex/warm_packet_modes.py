"""Transfer priorities from a verified packet calendar into a changed model."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import pickle


def keys(ir, scalar_leaves=False, ssa_destinations=False, swaps=None):
    counts=defaultdict(int)
    for op in ir.ops:
        tag,coords=op.tag
        if ssa_destinations:
            dest=tuple((swaps or {}).get(v,v) for v in op.args[0])
            key=(op.engine,op.code,tag,tuple(coords),dest)
            occurrence=counts[key];counts[key]+=1
            assert occurrence==0, ('Duplicate SSA destination key',key)
            yield key,occurrence
            continue
        imm=op.args[-1] if op.code in ('const','add_imm') else None
        values=tuple(ir.known.get(v) for v in op.args[0]) if op.engine!='store' else ()
        constant=values if values and all(v is not None for v in values) else None
        code,time=op.code,op.time
        if scalar_leaves and tag=='raw_gather' and coords[0]==10:
            code='raw_payload'
        if scalar_leaves and tag=='node_xor' and coords[0]==10:
            time=None
        if scalar_leaves and constant and len(set(constant))==1 and 1026<=constant[0]<=1032:
            if tag=='setup' and code in ('const','add_imm'):
                imm='leaf_root'
                constant=tuple(op.args[0])
            elif tag=='constant_reload':
                constant=None
        key=(op.engine,code,tag,tuple(coords),time,imm,constant)
        occurrence=counts[key];counts[key]+=1
        yield key,occurrence


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source',help='Verified prefix, with .pkl and .sol')
    ap.add_argument('target',help='Changed prefix, with .pkl and .txt')
    ap.add_argument('--scalar-leaves',action='store_true',
                    help='Match equivalent deepest-leaf scalar payloads and XOR lanes')
    ap.add_argument('--seed',action='store_true',
                    help='Also transfer a full calendar, checking all dependencies and capacities')
    ap.add_argument('--ssa-destinations',action='store_true',
                    help='Match preserved SSA destinations, applying target raw-hash packet swaps')
    args=ap.parse_args()
    with open(args.source+'.pkl','rb') as stream:old=pickle.load(stream)
    with open(args.target+'.pkl','rb') as stream:new=pickle.load(stream)
    solution=list(map(int,Path(args.source+'.sol').read_text().split()))
    n,ng=solution[:2];assert (n,ng)==(len(old.ops),len(old.groups))
    times=solution[2:2+n]
    oldkeys=list(keys(old,args.scalar_leaves,args.ssa_destinations))
    swaps=(getattr(new,'raw_hash_swaps',{})
           if not getattr(old,'raw_hash_swaps',{}) else {})
    newkeys=list(keys(new,args.scalar_leaves,args.ssa_destinations,
                      swaps))
    mapping=dict(zip(oldkeys,times))
    lines=Path(args.target+'.txt').read_text().splitlines()
    assert int(lines[0].split()[0])==len(new.ops)
    matches=0
    for j,key in enumerate(newkeys):
        if key in mapping:
            engine,_=lines[j+1].split()
            lines[j+1]=f'{engine} {mapping[key]}'
            matches+=1
    output=Path(args.target+'.warm.txt')
    output.write_text('\n'.join(lines)+'\n')
    print({'matched':matches,'total':len(new.ops),'output':str(output)})
    if args.seed:
        assert all(key in mapping for key in newkeys), 'Unmatched seed operations'
        newtimes=[mapping[key] for key in newkeys]
        oldmodes=solution[2+n:]
        mode_map={tuple(oldkeys[j] for j in group[2]):oldmodes[g]
                  for g,group in enumerate(old.groups)}
        newmodes=[mode_map.get(tuple(newkeys[j] for j in group[2]),0)
                  for group in new.groups]
        for j,dd in enumerate(new.deps):
            for d,lag in dd.items():
                assert newtimes[j]>=newtimes[d]+lag,(d,j,lag)
        covered={j:g for g,group in enumerate(new.groups) if newmodes[g] for j in group[2]}
        use=defaultdict(lambda:defaultdict(int))
        caps={'alu':12,'valu':6,'load':2,'store':2,'flow':1}
        for j,op in enumerate(new.ops):
            engine=op.engine
            if j in covered:
                ids=new.groups[covered[j]][2]
                assert len({newtimes[i] for i in ids})==1
                if j!=ids[0]:continue
                engine='valu'
            use[newtimes[j]][engine]+=1
            assert use[newtimes[j]][engine]<=caps[engine]
        seed=Path(args.target+'.seed.sol')
        seed.write_text(f'{len(new.ops)} {len(new.groups)}\n'+' '.join(map(str,newtimes))+'\n'+' '.join(map(str,newmodes))+'\n')
        print({'seed':str(seed),'cycles':max(newtimes)+1})


if __name__=='__main__':main()
