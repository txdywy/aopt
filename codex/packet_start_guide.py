"""Write a small canonical bootstrap guide for packet-mode experiments."""
from __future__ import annotations

import argparse
from pathlib import Path
import pickle


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('model', type=Path)
    ap.add_argument('output', type=Path)
    args = ap.parse_args()
    ir = pickle.loads(args.model.read_bytes())
    pins, modes = {}, {}
    for j, op in enumerate(ir.ops):
        pin = None
        if op.tag in (('input', (0,)), ('input', (1,))):
            pin = 1
        elif op.tag == ('tree', (0,)):
            pin = 3
        elif op.engine == 'load':
            if op.code == 'const' and op.args[1] == 2318:
                pin = 0
            elif op.code == 'vload' and tuple(ir.known.get(v) for v in op.args[0][:7]) == (
                    16, 2047, 256, 10, 7, 2054, 2310):
                pin = 0
        elif op.engine == 'flow' and op.code == 'add_imm':
            pin = {0x7ed55d16: 0, 0xc761c23c: 1, 2899272192: 2,
                   0xfd7046c5: 3, 3925396509: 4,
                   0xb55a4f09: op.time}.get(ir.known.get(op.args[0][0]))
        elif op.engine == 'valu':
            value = ir.known.get(op.args[0][0])
            if op.code == 'vbroadcast':
                if value in (16, 256, 2310, 0x7ed55d16) or (
                        value is None and op.tag == ('setup', ())):
                    pin = 1
                elif value == 0xc761c23c:
                    pin = 2
            elif op.code == 'multiply_add' and value == 4097:
                pin = 2
        if pin is not None:
            pins[j] = pin
    for g, (code, aa, ids, _) in enumerate(ir.groups):
        values = {ir.known.get(v) for v in aa[0]}
        pin = 0 if code == '==' and values == {1} else 1 if code == '+' and values == {2} else None
        if pin is not None:
            modes[g] = 1
            pins.update({j: pin for j in ids})
    with args.output.open('w') as stream:
        stream.write(str(len(pins)) + '\n')
        for j, t in sorted(pins.items()):
            stream.write(f'{j} {t}\n')
        stream.write(str(len(modes)) + '\n')
        for g, md in sorted(modes.items()):
            stream.write(f'{g} {md}\n')
    print({'pinned_nodes': len(pins), 'native_groups': len(modes)})


if __name__ == '__main__':
    main()
