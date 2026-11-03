"""Bounded searches for the 18 live-input one-MAD orders not closed by prefix lemma.

Every fixed order is first checked against an unpinned constructive positive
control. SAT controls need an independent full-domain equivalence check;
UNKNOWN is recorded, never treated as exclusion. The other nine orders have
two bitwise instructions before the MAD and are covered by the rectangle lemma.
"""
import argparse
from collections import Counter
import itertools
import json

from codex.search_live_middle863 import live_words, search


def bitwise(op,a,b):
    return {'^':lambda:a^b, '&':lambda:a&b, '|':lambda:a|b}[op]()


def control_function(sequence,width):
    def fn(x):
        _,a,p,_ = live_words(x,width)
        if sequence[0] == 'mad':
            return bitwise(sequence[2],bitwise(sequence[1],33*x+17,a),p)
        return bitwise(sequence[2],33*bitwise(sequence[0],x,17)+19,a)
    return fn


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout-ms',type=int,default=3000)
    parser.add_argument('--width',type=int,default=11)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    rows=[]
    for pos in (0,1):
        for ops in itertools.product(('^','&','|'),repeat=2):
            sequence=list(ops)
            sequence.insert(pos,'mad')
            control=search(sequence,args.width,args.timeout_ms,
                           target_fn=control_function(sequence,args.width))
            actual=search(sequence,args.width,args.timeout_ms)
            rows.append({'sequence':sequence,'positive_control':control,'target':actual})
            report={'width':args.width,'timeout_ms':args.timeout_ms,
                'prefix_closed_orders':9,'counts':dict(Counter(r['target']['result'] for r in rows)),
                'control_counts':dict(Counter(r['positive_control']['result'] for r in rows)),
                'rows':rows}
            with open(args.output,'w') as output:
                json.dump(report,output,indent=2)
                output.write('\n')
            print(json.dumps({'sequence':sequence,'control':control['result'],
                'control_verification':control.get('verification'),'target':actual['result'],
                'target_verification':actual.get('verification')}),flush=True)


if __name__ == '__main__':
    main()
