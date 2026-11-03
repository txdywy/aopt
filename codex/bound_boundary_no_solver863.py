"""Relaxed prefix/suffix bounds on an expanded word-SSA graph, without CP-SAT."""
from collections import deque

from codex.packet_budget_bound import lane_weight


def order_and_distance(deps,succ):
    degree=list(map(len,deps))
    ready=deque(j for j,d in enumerate(degree) if not d)
    distance=[0]*len(deps);order=[]
    while ready:
        j=ready.popleft();order.append(j)
        for q,lag in succ[j]:
            distance[q]=max(distance[q],distance[j]+lag)
            degree[q]-=1
            if not degree[q]:ready.append(q)
    assert len(order)==len(deps),'cyclic expanded graph'
    return order,distance


def prefix(ir,width=6):
    _,earliest=order_and_distance(ir.deps,ir.succ)
    weights=[lane_weight(ir,j) for j in range(len(ir.ops))]
    per_prefix=[sum(w for w,t in zip(weights,earliest) if t<k) for k in range(1,width+1)]
    maximum=min(60*width,*(60*(width-k)+per_prefix[k-1] for k in range(1,width+1)))
    return dict(width=width,availability_by_cycle=per_prefix,objective_upper_bound=maximum,
        waste_lower_bound=60*width-maximum,
        scope='Grant all arithmetic with ASAP<k for free and all later slots full; '
              'ignore non-arithmetic and cross-node resource contention.')


def suffix(ir,width=3):
    deps=[dict(edges) for edges in ir.succ]
    succ=[list(edges.items()) for edges in ir.deps]
    order,earliest=order_and_distance(deps,succ)
    eligible={j for j,t in enumerate(earliest) if t<width}
    owners={};roots=[]
    for j in order:
        if j not in eligible:continue
        if not deps[j]:owners[j]={j};roots.append(j)
        else:owners[j]=set().union(*(owners[d] for d in deps[j]))
    store_roots={j for j in roots if ir.ops[j].engine=='store'}
    rewards={root:[0]*width for root in store_roots};free=0
    for j in eligible:
        if ir.ops[j].engine not in ('alu','valu'):continue
        weight=lane_weight(ir,j)
        if len(owners[j])!=1 or not owners[j]<=store_roots:
            free+=weight;continue
        root=next(iter(owners[j]))
        for t in range(width-earliest[j]):rewards[root][t]+=weight
    maximum=min(60*width,free+sum(sum(sorted((rewards[root][t] for root in store_roots),
                                  reverse=True)[:2]) for t in range(width)))
    return dict(width=width,store_roots=len(store_roots),free_work=free,
        objective_upper_bound=maximum,waste_lower_bound=60*width-maximum,
        scope='Two stores per cycle; shared/non-store-root arithmetic is granted '
              'free placement and all other resources are relaxed.')
