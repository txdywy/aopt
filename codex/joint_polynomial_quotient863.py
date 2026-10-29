"""Polynomial plus a prepared quotient: a prefix-length-independent cut."""
from functools import lru_cache

from codex.audit_one_high_tail863 import information_fibre
from codex.close_early_boundary863 import rectangle
from codex.search_early_boundary863 import MASK,NODE


@lru_cache(maxsize=1)
def certificate():
    rect=rectangle();delta=rect['target_mixed'];distance=rect['distance']
    small=[]
    for d in range(1,5):
        residue=(d*delta)&MASK;dist=min(residue,(-residue)&MASK)
        assert dist>2*(d-1)
        small.append(dict(divisor=d,target_mixed_times_divisor=residue,distance=dist,residual_bound=2*(d-1)))
    bound=(MASK+4)//5
    assert distance>2*bound
    rem=information_fibre()['prepared_remainder_divisor_max']-1
    assert distance>2*rem and distance>2*65535
    return dict(rectangle=rect,small_divisor_checks=small,divisor_at_least5_quotient_maximum=bound,
                remainder_maximum=rem,
                proof='Every integer polynomial in the rectangle coordinates has mixed difference0 modulo2^32, with any degree and constant coefficients. For Y=R+floor(P/D), D*Y=D*R+P-r,0<=r<D. For ceil, D*Y=D*R+P+r. Thus D*Delta(Y) is within2*(D-1) of0 modulo2^32, excluding D1..4. For D>=5, q<=ceil(MASK/5), so Delta(Y)=Delta(q) has distance<=2*qmax, a contradiction. REM has q<D<=the478-state bound; SHR has q<=65535. These also contradict the same rectangle.')


def applicable(row,node):
    if node!=NODE or row['order'][-1]!='mad':return None
    special=[i for i,op in enumerate(row['order']) if op in ('shr16','shr','//','%','cdiv')]
    if len(special)!=1 or any(op!='mad' for i,op in enumerate(row['order']) if i!=special[0]):return None
    n=8;q=n+special[0]
    def depends(index):
        if index==q:return True
        if index<n:return False
        i=index-n
        return any(depends(j) for j in row['wiring'][i] if j<n+i)
    a,b,c=row['wiring'][-1]
    if c!=q or depends(a) or depends(b):return None
    return certificate()
