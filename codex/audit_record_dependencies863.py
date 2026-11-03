"""Check the address dependency that defeats a proposed d7/d9 record fusion.

This is an audit of unexpanded research_stage864 IR, not a new kernel. Removing
a gather does not make its pointer dead when the next depth still uses it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle


def audit(ir):
    writers = {v: j for j, op in enumerate(ir.ops) if op.engine != 'store'
               for v in op.args[0]}
    rows = []
    for group in range(32):
        at9 = next(op for op in ir.ops if op.tag == ('raw_gather', (9, group, 0)))
        at10 = next(op for op in ir.ops if op.tag == ('raw_gather', (10, group, 0)))
        p9_id = writers[at9.args[1][0]]
        p10_id = writers[at10.args[1][0]]
        p9, p10 = ir.ops[p9_id], ir.ops[p10_id]
        assert p9.engine == 'valu' and p9.code == '-'
        assert p10.engine == 'valu' and p10.code == 'multiply_add'
        assert p10.args[1] == p9.args[0]
        rows.append(dict(group=group, depth9_address_op=p9_id,
            depth10_address_op=p10_id, retained_arithmetic_lanes=8,
            pointer_words=list(p9.args[0])))
    return dict(groups=len(rows), retained_arithmetic_lanes=sum(
        row['retained_arithmetic_lanes'] for row in rows), rows=rows,
        conclusion='All 32 depth-9 address vectors feed depth-10 addressing. '
            'Removing depth-9 gathers alone cannot remove these 256 arithmetic lanes.',
        scope='Dependency audit only; does not rule out a different depth-10 layout.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    with args.ir.open('rb') as stream:
        ir = pickle.load(stream)
    result = audit(ir)
    result['model_sha256'] = hashlib.sha256(args.ir.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
