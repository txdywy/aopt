"""Close the sampled joint boundary graphs without treating timeouts as UNSAT."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def key(row):
    return tuple(row['order']),tuple(tuple(w) for w in row['wiring'])


def input_path_avoiding(row,cut):
    """Backwards paths retain every real raw-input bypass.

    Node 0 is x, node i+1 is instruction i. An operand at index >= i+1
    in instruction i is a private constant, not an SSA input word.
    """
    todo=[len(row['order'])];visited=set()
    while todo:
        node=todo.pop()
        if node==cut or node in visited:continue
        visited.add(node)
        if node==0:return True
        todo.extend(i for i in row['wiring'][node-1] if i<node)
    return False


def audit(source,refinement):
    a=json.loads(source.read_text());b=json.loads(refinement.read_text())
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert a['joint_last_mad'] and a['sampled_not_exhaustive']
    assert b['source_sha256']==digest(source) and b['node']==a['node']==0xA76E1971
    assert a['completed']==len(a['manifest'])==len(a['results'])==3552
    original={key(r):r for r in a['results']}
    assert len(original)==3552 and set(original)=={key(r) for r in a['manifest']}
    pending={key(r) for r in a['results'] if r['status']!='unsat'}
    assert pending=={key(r) for r in b['results']}
    assert b['completed']==len(b['results'])==len(pending)==23
    closed=[];unresolved=[];counts=Counter(a['counts'])
    counts=Counter(initial_sample_unsat=counts['unsat'])
    for row in b['results']:
        assert row['full32_semantics'] and row['joint_last_mad']
        if row['status']=='unsat':
            counts['refinement_unsat']+=1
            continue
        assert row['status'] not in ('proved_equivalent',), 'A verified macro must be integrated, not excluded'
        cuts=[i+1 for i,op in enumerate(row['order'])
              if op=='shr16' and not input_path_avoiding(row,i+1)]
        if cuts:
            counts['shift_wordcut_range_certificate']+=1
            closed.append(dict(order=row['order'],wiring=row['wiring'],cut=cuts[0],
                               cut_image_at_most=65536,target_image=1<<32,
                               certificate='Every input-to-output path passes through SHR16'))
        else:
            unresolved.append(row)
    return dict(node=a['node'],sampled_not_exhaustive=True,manifest_wirings=3552,
                counts=dict(counts),closed_by_structure=closed,unresolved=unresolved,
                remaining=len(unresolved),best_executable_cycles=864,new_executable=False,
                evidence_sha256={str(p):digest(p) for p in (source,refinement)},
                scope='Only the 3552 sampled four-op graphs; not all four-op programs or a kernel bound.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path('codex/root_boundary863_joint_scout.json'))
    ap.add_argument('--refinement',type=Path,default=Path('codex/root_boundary863_joint_refinement.json'))
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    result=audit(args.source,args.refinement)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(result['counts'],'remaining',result['remaining'])


if __name__=='__main__':main()
