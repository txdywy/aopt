"""Exact wide-product constraints for a prepared unsigned division sample."""
import z3

from codex.search_live_boundary863 import expression


def relation(numerator,result,divisor,op,width=32,tag='division'):
    def bv(v):return v if z3.is_bv(v) else z3.BitVecVal(int(v),width)
    n,q,d=map(bv,(numerator,result,divisor))
    nn,qq,dd=[z3.ZeroExt(width,v) for v in (n,q,d)]
    valid=d!=0
    if op=='//':
        product=qq*dd
        return [valid,z3.ULE(product,nn),z3.ULT(nn,product+dd)]
    if op=='cdiv':
        return [valid,z3.Or(z3.And(n==0,q==0),
                            z3.And(q!=0,z3.ULT((qq-1)*dd,nn),z3.ULE(nn,qq*dd)))]
    if op=='%':
        multiple=z3.BitVec(tag+'_multiple',width)
        product=z3.ZeroExt(width,multiple)*dd
        return [valid,z3.ULT(q,d),product+qq==nn]
    raise ValueError(op)


def sample(row,constants,initial,tag):
    values=list(initial);conditions=[]
    for i,(op,w,cc) in enumerate(zip(row['order'],row['wiring'],constants)):
        if op in ('//','%','cdiv'):
            q=z3.BitVec(f'{tag}_{i}_value',32)
            conditions.extend(relation(values[w[0]],q,cc[0],op,tag=f'{tag}_{i}'))
            values.append(q)
        else:
            part=dict(order=[op],wiring=[w])
            values.append(expression(part,[cc],values))
    return values,conditions
