"""Conservative whole-program register liveness, including indirect dispatch.

Every indirect jump may target EVERY packet. This loses precision but never
assumes a sampled dispatch path covers all inputs. Memory/control operations
are retained, even if their register results appear dead.
"""
import json
import random
import sys
from pathlib import Path
from codex.perf_takehome_983 import KernelBuilder
from codex.fuse_table_returns import accesses


def mask(words):
    return sum(1 << word for word in words)


def audit(program):
    reads, writes, succ, entries = [], [], [], []
    for pc, packet in enumerate(program):
        rd = wr = 0
        items = []
        successors = [pc+1] if pc+1 < len(program) else []
        for engine, slots in packet.items():
            for si, slot in enumerate(slots):
                op = slot[0]
                if engine == 'flow' and op in ('halt', 'jump', 'pause'):
                    if op == 'halt': successors = []
                    if op == 'jump': successors = [slot[1]]
                    continue
                if engine == 'debug':
                    raise ValueError('debug reads require explicit modeling')
                r, w = accesses(engine, slot)
                r, w = mask(r), mask(w)
                rd |= r; wr |= w
                items.append((engine, si, slot, r, w))
                if engine == 'flow' and op == 'jump_indirect':
                    successors = None
        reads.append(rd); writes.append(wr); succ.append(successors); entries.append(items)
    live = [0] * len(program)
    rounds = 0
    while True:
        rounds += 1
        universe = 0
        for bits in live: universe |= bits
        changed = False
        for pc in range(len(program)-1, -1, -1):
            out = universe if succ[pc] is None else 0
            if succ[pc] is not None:
                for target in succ[pc]: out |= live[target]
            value = reads[pc] | (out & ~writes[pc])
            if value != live[pc]:
                changed = True; live[pc] = value
        if not changed: break
    universe = 0
    for bits in live: universe |= bits
    dead = []
    for pc, items in enumerate(entries):
        out = universe if succ[pc] is None else 0
        if succ[pc] is not None:
            for target in succ[pc]: out |= live[target]
        for engine, si, slot, r, w in items:
            pure = engine in ('alu', 'valu') or (engine == 'flow' and slot[0] in ('add_imm', 'vselect'))
            if pure and w and not w & out:
                dead.append((pc, engine, si, slot))
    return dead, rounds


if __name__ == '__main__':
    builder = KernelBuilder(); builder.build_kernel(10, 2047, 256, 16)
    dead, rounds = audit(builder.instrs)
    report = {'fixed_point_passes': rounds, 'dead_operation_count': len(dead),
              'main_dead_operations': [item for item in dead if item[0] < 984],
              'indirect_targets': 'all program packets'}
    if '--verify' in sys.argv:
        for pc, engine, si, slot in sorted(dead, reverse=True):
            assert builder.instrs[pc][engine][si] == slot
            del builder.instrs[pc][engine][si]
        if '--early-loads' in sys.argv:
            addresses = [('-', 1112, 16, 856), ('-', 1116, 16, 860)]
            loads = [('load_offset', 1111, 1111, 1), ('load_offset', 1111, 1111, 5)]
            for slot in addresses:
                builder.instrs[972]['alu'].remove(slot)
            builder.instrs[966].setdefault('alu', []).extend(addresses)
            builder.instrs[967].setdefault('load', []).extend(loads)
            changed = 0
            for pc, packet in enumerate(builder.instrs):
                if pc == 973 or packet.get('flow') == [('jump', 974)]:
                    for slot in loads:
                        if slot in packet.get('load', []):
                            packet['load'].remove(slot)
                            changed += 1
            assert changed > 2, changed
            report['early_load_copies_removed'] = changed
            if '--early-hash' in sys.argv:
                address = ('-', 1113, 16, 857)
                builder.instrs[966]['alu'].append(address)
                for pc, packet in enumerate(builder.instrs):
                    if pc == 971 or packet.get('flow') == [('jump', 972)]:
                        packet['alu'].remove(address)
                for lane in (2, 6):
                    slot = ('load_offset', 1111, 1111, lane)
                    builder.instrs[972]['load'].remove(slot)
                    builder.instrs[970].setdefault('load', []).append(slot)
                xors = [('^', 599+i, 599+i, 1111+i) for i in range(8)]
                first_hash = ('multiply_add', 599, 599, 127, 71)
                for pc, packet in enumerate(builder.instrs):
                    if pc in (973, 974, 975) or packet.get('flow') in ([('jump', 974)], [('jump', 976)]):
                        for slot in xors:
                            if slot in packet.get('alu', []): packet['alu'].remove(slot)
                        if first_hash in packet.get('valu', []): packet['valu'].remove(first_hash)
                    if pc == 974:
                        packet.setdefault('valu', []).append(first_hash)
                builder.instrs[972]['alu'].extend(xors)
                from codex.fuse_table_returns import LIMITS
                for pc, packet in enumerate(builder.instrs):
                    assert all(len(slots) <= LIMITS[engine] for engine, slots in packet.items()), (pc, {e:len(s) for e,s in packet.items()})
                report['early_hash'] = True
        if '--reschedule-tail' in sys.argv:
            from codex.schedule_tail_window import schedule_tail
            tail_start = 974 if '--early-hash' in sys.argv else 973
            candidate, certificate = schedule_tail(
                builder.instrs, start=tail_start, stop=984, horizon=983-tail_start,
                allow_data_flow=True, allow_memory=True, timeout_ms=15000)
            report['tail_schedule'] = {k: v for k, v in certificate.items()
                                       if k not in ('critical_chain', 'store_earliest', 'issue_cycles')}
            if '--show-chain' in sys.argv:
                report['critical_chain'] = certificate.get('critical_chain')
            if candidate is None:
                print(json.dumps(report))
                raise SystemExit(0)
            builder.instrs = candidate
            report['tail_certificate'] = certificate
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
        from frozen_problem import Machine, Tree, Input, build_mem_image, reference_kernel2, N_CORES
        scores = []
        for seed in range(23):
            random.seed(seed)
            tree = Tree.generate(10); inp = Input.generate(tree, 256, 16)
            def value(i):
                return random.getrandbits(32) if seed < 20 else (0, 0xffffffff, 0xffffffff if i % 2 else 0)[seed-20]
            tree.values = [value(i) for i in range(len(tree.values))]
            inp.values = [value(i) for i in range(256)]
            mem = build_mem_image(tree, inp)
            machine = Machine(mem, builder.instrs, builder.debug_info(), n_cores=N_CORES)
            machine.enable_pause = machine.enable_debug = False
            machine.run()
            for ref in reference_kernel2(mem): pass
            assert machine.mem[ref[6]:ref[6]+256] == ref[ref[6]:ref[6]+256], seed
            scores.append(machine.cycle)
        report.update(cycles=scores, json_bytes=len(json.dumps(builder.instrs)))
    print(json.dumps(report))
