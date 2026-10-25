"""Search the middle hash with a changed preceding XOR and real live values.

Let a=4097*p+C0, and allow the preceding XOR constant K to change for free.
The candidate reads a, p, and b_K=a XOR (a>>19) XOR K, and must produce
F(a XOR (a>>19) XOR C1) XOR O. Both K and O can be arbitrary.

For width <=19, use the legitimate full32 input subdomain 0<=a<2**width.
Then a>>19=0, and reduction modulo 2**width commutes with the admitted
MAD/XOR/AND/OR operations. A reduced SAT is never a full32/kernel result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import z3

from codex.search_cross_encoding863 import middle
from codex.synthesize_hash_cegis import choose


C0,C1,MASK = 0x7ed55d16,0xc761c23c,0xffffffff
OPS = ('mad','^','&','|')


def live_values(a,key,width):
    mask = (1 << width)-1
    prior = ((a-C0)*pow(4097,-1,1 << width))&mask
    # Do not encode literal 19 in a tiny bit-vector: e.g. width four would
    # truncate it to three. This domain explicitly has full32 a>>19 == 0.
    shifted = (a >> 19 if isinstance(a,int) else z3.LShR(a,19))if width==32 else 0
    return [a,prior,(a ^ shifted ^ key)&mask]


def execute(a,key,rows,width):
    mask = (1 << width)-1
    values = live_values(a & mask,key & mask,width)
    for row in rows:
        pool = values+row['constants']
        x,y,z = [pool[i]for i in row['operands']]
        value = x*y+z if row['op']=='mad' else x^y if row['op']=='^' else x&y if row['op']=='&' else x|y
        values.append(value & mask)
    return values[-1]


def expression(a,key,rows,width):
    values = live_values(a,key,width)
    for row in rows:
        pool = values+[z3.BitVecVal(v,width)for v in row['constants']]
        x,y,z = [pool[i]for i in row['operands']]
        values.append(x*y+z if row['op']=='mad' else x^y if row['op']=='^' else x&y if row['op']=='&' else x|y)
    return values[-1]


def original_program():
    return [dict(op='mad',operands=[2,3,4],constants=[33,0xe9f8cc1d]),
            dict(op='mad',operands=[2,4,5],constants=[16896,0xaccf6200]),
            dict(op='^',operands=[3,4,0],constants=[0,0]),
            dict(op='mad',operands=[5,6,7],constants=[9,0xfd7046c5])]


def reference(a,width,control=False):
    if control:
        _,prior,b = live_values(a,42,width)
        return (33*b+17)^(5*prior+29)
    shifted = (a >> 19 if isinstance(a,int) else z3.LShR(a,19))if width==32 else 0
    return middle(a ^ shifted ^ C1)


def search(order,width=11,seconds=20,check_ms=18000,*,pin_original=False,
           control=False,fixed_key=None,first_source=None,middle_source=None,
           last_operands=None):
    if not 1 <= len(order) <= 4 or any(op not in OPS for op in order):
        raise ValueError('one to four unshifted MAD/XOR/AND/OR instructions required')
    if width not in (*range(2,17),32) or seconds<=0 or check_ms<=0:
        raise ValueError('width 2..16 or 32 and positive limits required')
    if pin_original and (tuple(order)!=('mad','mad','^','mad') or control):
        raise ValueError('original control has four specified instructions')
    if first_source not in (None,'a','prior','keyed'):
        raise ValueError('invalid first source')
    if first_source and order[0]!='mad':
        raise ValueError('first-source restriction requires first MAD')
    if middle_source not in (None,'a','prior','keyed','key'):
        raise ValueError('invalid middle source')
    if (middle_source or last_operands) and tuple(order) not in tuple(('mad',op,'mad')for op in ('^','&','|')):
        raise ValueError('middle/last restrictions require MAD,bitwise,MAD')
    if last_operands is not None and (len(last_operands)!=3 or any(i not in range(7)for i in last_operands)):
        raise ValueError('last operands must be three indices in 0..6')
    mask = (1 << width)-1
    key,outkey = z3.BitVecs('keyed_live_input_key keyed_live_output_key',width)
    solver = z3.SolverFor('QF_BV')
    if fixed_key is not None:
        solver.add(key == (fixed_key&mask))
    spec = []
    for i,op in enumerate(order):
        a,b,c = z3.BitVecs(f'kl_a{i} kl_b{i} kl_c{i}',4)
        k0,k1 = z3.BitVecs(f'kl_k{i}_0 kl_k{i}_1',width)
        solver.add(*(z3.ULT(v,i+5)for v in(a,b,c)),z3.ULE(a,b))
        if op!='mad':solver.add(c==0)
        spec.append((op,a,b,c,k0,k1))
    if first_source:
        indices = (('a','prior','keyed').index(first_source),3,4)
        solver.add(*(v==n for v,n in zip(spec[0][1:4],indices)))
    if middle_source:
        other = ('a','prior','keyed','key').index(middle_source)
        if other==3:other=4
        solver.add(*(v==n for v,n in zip(spec[1][1:3],sorted((3,other)))))
    if last_operands is not None:
        solver.add(*(v==n for v,n in zip(spec[2][1:4],last_operands)))
    for i in range(len(order)-1):
        solver.add(z3.Or(*(v==i+3 for row in spec[i+1:]for v in row[1:4 if row[0]=='mad'else 3])))
    if pin_original:
        solver.add(key==C1&mask,outkey==0)
        for row,witness in zip(spec,original_program()):
            solver.add(*(v==n for v,n in zip(row[1:4],witness['operands'])),
                       *(v==(n&mask)for v,n in zip(row[4:],witness['constants'])))

    def candidate(a):
        values = live_values(a,key,width)
        for op,x,y,z,k0,k1 in spec:
            pool=values+[k0,k1]
            u,v,w = choose(x,pool),choose(y,pool),choose(z,pool)
            values.append(u*v+w if op=='mad'else u^v if op=='^'else u&v if op=='&'else u|v)
        return z3.simplify(values[-1])

    rng = random.Random(86451)
    samples = list(dict.fromkeys(v&mask for v in [*range(32),256,264,288,296,511,512,1023,1024,
        0x7ffff,0x80000,0x80001,0x80000000,MASK]+[rng.getrandbits(width)for _ in range(20)]))

    def constrain(v):
        a=z3.BitVecVal(v,width)
        solver.add(candidate(a)==(reference(a,width,control)^outkey))

    for v in samples:constrain(v)
    checks = list(range(1 << width))if width<32 else list(dict.fromkeys(samples+
        [1 << i for i in range(32)]+[rng.getrandbits(32)for _ in range(512)]))
    started=time.monotonic()
    report=dict(sequence=list(order),width=width,seconds_limit=seconds,check_ms=check_ms,
        live_pool=['a','prior=(a-C0)/4097','keyed=a XOR (a>>19) XOR K'],
        target='F(a XOR (a>>19) XOR C1) XOR O',arbitrary_keys=fixed_key is None,
        fixed_key=fixed_key,first_source=first_source,middle_source=middle_source,
        last_operands=last_operands,initial_samples=samples[:],control=control,
        pin_original=pin_original,events=[],result='deadline',new_executable=False,
        scope='Reduced widths restrict the full32 pre-shift a to the low domain; '
              'SAT does not imply a full32 identity or a cycle score.')
    while time.monotonic()-started<seconds:
        solver.set(timeout=min(check_ms,max(1,int(1000*(seconds-(time.monotonic()-started))))))
        status=solver.check()
        event=dict(samples=len(samples),synthesis=str(status));report['events'].append(event)
        if status!=z3.sat:
            report['result']='necessary_subdomain_unsat'if status==z3.unsat else 'unknown'
            if status==z3.unknown:report['reason']=solver.reason_unknown()
            break
        model=solver.model()
        kv,ov=[model.eval(v,model_completion=True).as_long()for v in(key,outkey)]
        rows=[dict(op=row[0],operands=[model.eval(v,model_completion=True).as_long()for v in row[1:4]],
              constants=[model.eval(v,model_completion=True).as_long()for v in row[4:]])for row in spec]
        report.update(input_key=kv,output_key=ov,program=rows)
        bad=[a for a in checks if execute(a,kv,rows,width)!=((reference(a,width,control)^ov)&mask)][:8]
        if bad:
            event['counterexamples']=bad
            for a in bad:
                assert a not in samples
                samples.append(a);constrain(a)
            continue
        if width<32:
            report.update(result='equivalent_on_restricted_low_domain',verified_inputs=len(checks));break
        a=z3.BitVec('keyed_live_full32_verify',32)
        verify=z3.SolverFor('QF_BV')
        verify.set(timeout=min(check_ms,max(1,int(1000*(seconds-(time.monotonic()-started))))))
        verify.add(expression(a,kv,rows,width)!=(reference(a,width,control)^ov))
        status=verify.check();event['full32_verification']=str(status)
        if status==z3.unsat:
            report['result']='proved_equivalent_full32';break
        if status==z3.unknown:
            report.update(result='verification_unknown',reason=verify.reason_unknown());break
        a=verify.model().eval(a,model_completion=True).as_long()
        event['symbolic_counterexample']=a
        assert a not in samples
        samples.append(a);constrain(a)
    report['seconds']=round(time.monotonic()-started,3)
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sequence',required=True)
    ap.add_argument('--width',type=int,default=11)
    ap.add_argument('--seconds',type=float,default=20)
    ap.add_argument('--check-ms',type=int,default=18000)
    ap.add_argument('--pin-original',action='store_true')
    ap.add_argument('--control',action='store_true')
    ap.add_argument('--fixed-key',type=lambda x:int(x,0))
    ap.add_argument('--first-source',choices=('a','prior','keyed'))
    ap.add_argument('--middle-source',choices=('a','prior','keyed','key'))
    ap.add_argument('--last-operands',type=lambda x:tuple(map(int,x.split(','))))
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    report=search(args.sequence.split(','),args.width,args.seconds,args.check_ms,
        pin_original=args.pin_original,control=args.control,fixed_key=args.fixed_key,
        first_source=args.first_source,middle_source=args.middle_source,last_operands=args.last_operands)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report.get(k)for k in('sequence','width','result','seconds','input_key','output_key')}))


if __name__=='__main__':main()
