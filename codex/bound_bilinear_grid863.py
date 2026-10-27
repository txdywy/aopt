"""Exhaust a restricted d5 bilinear layout on a common 16-word grid.

Each d5 address has four d7 records at offsets 0,4,8,12. This checks only
the unchanged signed-bit form x[b0]*s2+y[b1]+v[b3]*s4, not arbitrary layouts.
"""
from functools import lru_cache
import argparse
import json
from pathlib import Path


def contributions(a,b):
    return (-2*a,-a,-2*b,-b)


@lru_cache(None)
def shapes(limit):
    result={}
    for a in range(-limit,limit+1):
        for b in range(-limit,limit+1):
            values=contributions(a,b)
            if len(set(values))!=4:continue
            low,high=min(values),max(values)
            if high-low>limit:continue
            normalized=tuple(sorted(v-low for v in values))
            result.setdefault(normalized,(a,b))
    return result


def compute():
    # A putative span below or equal to 50 grid cells has range <= 49.
    # Each pair (-2*a,-a) differs by |a|, so every coefficient is in
    # [-49,49]. A/B translation and exchanging y0,y1 do not affect span.
    values=shapes(49);best=51;witness=None;examined=0
    for left,lr in values.items():
        for right,rr in values.items():
            if left[-1]+right[-1]>=best:continue
            sums={a+b for a in left for b in right}
            if len(sums)!=16:continue
            examined+=1
            mask=sum(1<<s for s in sums)
            delta=1
            while mask & (mask<<delta):delta+=1
            extent=left[-1]+right[-1]+delta+1
            if extent<best:
                best=extent
                witness=dict(x=list(lr),v=list(rr),y_delta=delta)
    assert best==50 and witness is not None
    x0,x1=witness['x'];v0,v1=witness['v'];delta=witness['y_delta']
    addresses={x+y+v for x in contributions(x0,x1)
               for y in (0,delta) for v in contributions(v0,v1)}
    assert len(addresses)==32 and max(addresses)-min(addresses)+1==50
    return dict(shape_count=len(values),valid_sum_pairs_examined=examined,
        minimum_grid_cells=best,minimum_word_span=16*best,witness=witness,
        distinct_addresses=len(addresses),new_executable=False,
        scope='Only common-grid p5=x[b0]*s2+y[b1]+v[b3]*s4 with s2,s4 in {-2,-1}, '
              'four d7 records at offsets 0,4,8,12, and 32 distinct p5 addresses. '
              'Other encodings, record placements, arithmetic graphs and wrapping '
              'outside a compact real memory interval are not excluded.',
        proof='Any span <=50 grid cells bounds every x/v coefficient and the y '
              'difference by 49; all 4-element contribution shapes are enumerated. '
              'For each distinct 16-element pairwise sumset, the minimum disjoint '
              'translation of the two y choices gives the minimum possible span.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();row=compute()
    args.output.write_text(json.dumps(row,indent=2)+'\n')
    print(json.dumps({k:row[k] for k in ('shape_count','minimum_grid_cells','minimum_word_span','witness')}))


if __name__=='__main__':main()
