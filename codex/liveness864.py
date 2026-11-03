"""Lane-level backward liveness of the frozen 864 program over several seeds."""
import sys, random
from collections import Counter, defaultdict
sys.path.insert(0, '.'); sys.path.insert(0, 'tests')
from frozen_problem import Machine, build_mem_image, Tree, Input
from perf_takehome import KernelBuilder

kb = KernelBuilder(); kb.build_kernel(10, 2047, 256, 16); prog = kb.instrs
VL = 8


def trace(seed):
    random.seed(seed)
    f = Tree.generate(10); inp = Input.generate(f, 256, 16)
    mem = build_mem_image(f, inp)
    m = Machine(mem, prog, kb.debug_info(), n_cores=1)
    m.enable_pause = False; m.enable_debug = False
    core = m.cores[0]
    rows = []  # per cycle: list of (key, lanes) ; lane = (out_kind, out_addr, scr_reads, mem_reads)
    for t, ins in enumerate(prog):
        S = core.scratch
        ops = []
        for eng, slots in ins.items():
            if eng == 'debug':
                continue
            for si, s in enumerate(slots):
                key = (t, eng, si)
                lanes = []
                op = s[0]
                if eng == 'alu':
                    _, d, a, b = s
                    lanes.append(('s', d, (a, b), ()))
                elif eng == 'valu':
                    if op == 'vbroadcast':
                        _, d, src = s
                        lanes = [('s', d + i, (src,), ()) for i in range(VL)]
                    elif op == 'multiply_add':
                        _, d, a, b, c = s
                        lanes = [('s', d + i, (a + i, b + i, c + i), ()) for i in range(VL)]
                    else:
                        _, d, a, b = s
                        lanes = [('s', d + i, (a + i, b + i), ()) for i in range(VL)]
                elif eng == 'load':
                    if op == 'load':
                        _, d, a = s
                        lanes.append(('s', d, (a,), (S[a],)))
                    elif op == 'load_offset':
                        _, d, a, o = s
                        lanes.append(('s', d + o, (a + o,), (S[a + o],)))
                    elif op == 'vload':
                        _, d, a = s
                        lanes = [('s', d + i, (a,), (S[a] + i,)) for i in range(VL)]
                    elif op == 'const':
                        _, d, v = s
                        lanes.append(('s', d, (), ()))
                elif eng == 'store':
                    if op == 'store':
                        _, a, src = s
                        lanes.append(('m', S[a], (a, src), ()))
                    else:
                        _, a, src = s
                        lanes = [('m', S[a] + i, (a, src + i), ()) for i in range(VL)]
                elif eng == 'flow':
                    if op == 'vselect':
                        _, d, c, a, b = s
                        lanes = [('s', d + i, (c + i, a + i, b + i), ()) for i in range(VL)]
                    elif op == 'select':
                        _, d, c, a, b = s
                        lanes.append(('s', d, (c, a, b), ()))
                    elif op == 'add_imm':
                        _, d, a, _imm = s
                        lanes.append(('s', d, (a,), ()))
                    else:
                        raise RuntimeError(s)
                ops.append((key, op, lanes))
        rows.append(ops)
        m.step(ins, core)
    return rows, inp


def backward(rows, out_lo, out_hi):
    live_s, live_m = set(), set(range(out_lo, out_hi))
    alive = {}
    for t in range(len(rows) - 1, -1, -1):
        written_s, written_m, reads_s, reads_m = set(), set(), set(), set()
        for key, op, lanes in rows[t]:
            flags = []
            for kind, addr, rs, rm in lanes:
                live = (addr in live_m) if kind == 'm' else (addr in live_s)
                flags.append(live)
                (written_m if kind == 'm' else written_s).add(addr)
                if live:
                    reads_s.update(rs); reads_m.update(rm)
            alive[key] = flags
        live_s = (live_s - written_s) | reads_s
        live_m = (live_m - written_m) | reads_m
    return alive


def main():
    seeds = [int(x) for x in sys.argv[1:]] or [1, 2, 3]
    union = {}
    meta = {}
    for sd in seeds:
        rows, inp = trace(sd)
        alive = backward(rows, 2310, 2310 + 256)
        for t, ops in enumerate(rows):
            for key, op, lanes in ops:
                meta[key] = op
        for k, fl in alive.items():
            union[k] = [a or b for a, b in zip(union.get(k, [False] * len(fl)), fl)]
    stat = Counter(); partial = defaultdict(list)
    for k, fl in union.items():
        t, eng, si = k
        n = sum(fl)
        if eng == 'alu':
            stat['alu_dead' if n == 0 else 'alu_live'] += 1
            if n == 0:
                partial['alu_dead'].append((t, meta[k]))
        elif eng == 'valu':
            stat[f'valu_live{n}'] += 1
            if n < 8:
                partial['valu'].append((t, meta[k], n))
        elif eng in ('flow', 'load', 'store'):
            stat[f'{eng}_lanes{n}/{len(fl)}'] += 1
    for k in sorted(stat):
        print(k, stat[k])
    print('ALU dead ops:', partial['alu_dead'][:40])
    print('VALU partially dead ops (cycle, op, live lanes):', partial['valu'][:60])
    save = len(partial['alu_dead']) + sum(8 - n for _, _, n in partial['valu'])
    print('upper bound on lanes recoverable by dropping dead ALU ops and narrowing VALU ops:', save)


if __name__ == '__main__':
    main()
