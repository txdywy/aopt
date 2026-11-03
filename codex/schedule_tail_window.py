"""Schedule a straight-line terminal window with precise packet hazards.

Requires optional z3-solver. Register RAW/WAW lag one; WAR lag zero because
the frozen ISA reads before writes. Stores are independent except for their
address/value register dependencies; this window has no memory loads.
"""

from collections import defaultdict
from copy import deepcopy
import hashlib
import json

from codex.fuse_table_returns import accesses, LIMITS


def schedule_tail(program, start=976, stop=986, horizon=9, timeout_ms=45000, certificate=None,
                  allow_data_flow=False, allow_memory=False, exit_pc=None,
                  advance_only=None):
    if advance_only is not None and (type(advance_only) is not int or advance_only < 0):
        raise ValueError("advance_only must be a nonnegative integer")
    ops, packets, edges = [], [], {}
    domains = []
    writer, readers = {}, defaultdict(set)
    memory_reads, memory_writes = [], []
    for pc in range(start, stop):
        packet_nodes = []
        for engine, slots in program[pc].items():
            for slot in slots:
                if engine == "flow" and slot == ("halt",):
                    continue
                if engine == "load" and slot[0] != "const" and not allow_memory:
                    raise ValueError("tail scheduler does not analyze memory loads")
                if engine == "flow" and not (allow_data_flow and slot[0] in ("vselect", "add_imm")):
                    raise ValueError("window must be straight line")
                read, write = accesses(engine, slot)
                i = len(ops)
                ops.append((engine, slot, read, write))
                if advance_only is not None:
                    domains.append((max(0, pc-start-advance_only), min(horizon-1, pc-start)))
                packet_nodes.append(i)
                if allow_memory:
                    if engine == "load" and slot[0] != "const":
                        for old in memory_writes:
                            edges[old, i] = 1
                    if engine == "store":
                        for old in memory_reads:
                            edges[old, i] = 0
                        for old in memory_writes:
                            # Same-packet stores retain their original slot
                            # order, preserving last-write-wins on overlap.
                            edges[old, i] = 0
                for word in read:
                    if word in writer:
                        edges[writer[word], i] = 1
                    readers[word].add(i)
        # All reads see the pre-packet values, including in-place operations.
        for i in packet_nodes:
            for word in ops[i][3]:
                if word in writer:
                    edges[writer[word], i] = 1
                for reader in readers[word]:
                    if reader != i:
                        edges[reader, i] = max(0, edges.get((reader, i), 0))
                readers[word].clear()
                writer[word] = i
        packets.append(packet_nodes)
        if allow_memory:
            # Same-packet memory reads precede writes too. Preserve this
            # even when addresses may alias, rather than relying on the
            # current kernel's disjoint tree/output regions.
            packet_reads = [i for i in packet_nodes if ops[i][0] == "load" and ops[i][1][0] != "const"]
            packet_writes = [i for i in packet_nodes if ops[i][0] == "store"]
            for child in packet_writes:
                for parent in packet_reads:
                    edges[parent, child] = max(0, edges.get((parent, child), 0))
            for parent, child in zip(packet_writes, packet_writes[1:]):
                edges[parent, child] = max(0, edges.get((parent, child), 0))
            memory_reads.extend(i for i in packet_nodes if ops[i][0] == "load" and ops[i][1][0] != "const")
            memory_writes.extend(i for i in packet_nodes if ops[i][0] == "store")

    digest = hashlib.sha256(json.dumps(
        {"ops": [(op[0], op[1]) for op in ops], "edges": sorted((a, b, lag) for (a, b), lag in edges.items())},
        separators=(",", ":")
    ).encode()).hexdigest()
    if certificate is not None:
        if certificate["graph_sha256"] != digest or certificate["horizon"] != horizon:
            raise ValueError("schedule certificate does not match the instruction graph")
        issue = certificate["issue_cycles"]
        report = dict(certificate)
    else:
        report, issue = solve(ops, edges, horizon, timeout_ms, domains=domains or None)
        if advance_only is not None:
            report["advance_only"] = advance_only
        report["graph_sha256"] = digest
        if issue is None:
            return None, report
        report["issue_cycles"] = issue
    # Independently validate certificates as well as solver-produced plans.
    if len(issue) != len(ops) or any(type(t) is not int or not 0 <= t < horizon for t in issue):
        raise ValueError("invalid issue cycles")
    if domains and any(not low <= t <= high for t, (low, high) in zip(issue, domains)):
        raise ValueError("schedule violates movement bounds")
    for (parent, child), lag in edges.items():
        if issue[child] < issue[parent] + lag:
            raise ValueError("schedule violates a register dependency")
    usage = defaultdict(int)
    usage[horizon-1, "flow"] = 1  # The inserted exit occupies a real slot.
    for time, op in zip(issue, ops):
        usage[time, op[0]] += 1
        if usage[time, op[0]] > LIMITS[op[0]]:
            raise ValueError("schedule exceeds engine capacity")
    result = deepcopy(program)
    for pc in range(start, stop):
        result[pc] = {}
    for cycle, (engine, slot, _, _) in zip(issue, ops):
        result[start+cycle].setdefault(engine, []).append(slot)
    ending = ("halt",) if exit_pc is None else ("jump", exit_pc)
    result[start+horizon-1]["flow"] = [ending]
    for pc in range(start+horizon, stop):
        result[pc] = {"flow": [ending]}
    return result, report


def solve(ops, edges, horizon, timeout_ms, domains=None):
    import z3
    earliest = [0] * len(ops)
    predecessor = {}
    for _ in ops:
        changed = False
        for (parent, child), lag in edges.items():
            if earliest[child] < earliest[parent] + lag:
                earliest[child] = earliest[parent] + lag
                predecessor[child] = parent
                changed = True
        if not changed:
            break
    else:
        raise ValueError("positive dependency cycle")
    last = max(range(len(ops)), key=earliest.__getitem__)
    chain, seen = [], set()
    while last not in seen:
        seen.add(last)
        chain.append({"engine": ops[last][0], "slot": ops[last][1], "earliest": earliest[last]})
        if last not in predecessor:
            break
        last = predecessor[last]
    dependency_floor = max(earliest)+1
    latest = [horizon-1] * len(ops)
    if domains:
        earliest = [max(t, bound[0]) for t, bound in zip(earliest, domains)]
        latest = [min(t, bound[1]) for t, bound in zip(latest, domains)]
    for _ in ops:
        changed = False
        for (parent, child), lag in edges.items():
            if latest[parent] > latest[child] - lag:
                latest[parent] = latest[child] - lag
                changed = True
        if not changed:
            break
    counts = {engine: sum(op[0] == engine for op in ops) for engine in LIMITS}
    for engine, count in counts.items():
        capacity = horizon * LIMITS[engine] - (engine == "flow")
        if count > capacity:
            return {"status": "unsat", "operations": len(ops), "edges": len(edges),
                    "horizon": horizon, "reason": "engine work exceeds window capacity",
                    "engine": engine, "required_slots": count, "available_slots": capacity}, None
    solver = z3.Solver()
    solver.set(timeout=timeout_ms)
    times = [z3.Int(f"tail_{i}") for i in range(len(ops))]
    for i, time in enumerate(times):
        solver.add(time >= earliest[i], time <= latest[i])
    for (parent, child), lag in edges.items():
        solver.add(times[child] >= times[parent] + lag)
    for cycle in range(horizon):
        for engine, cap in LIMITS.items():
            terms = [(times[i] == cycle, 1) for i, op in enumerate(ops)
                     if op[0] == engine and earliest[i] <= cycle <= latest[i]]
            if terms:
                solver.add(z3.PbLe(terms, cap - (engine == "flow" and cycle == horizon-1)))
    status = solver.check()
    report = {"status": str(status), "operations": len(ops), "edges": len(edges), "horizon": horizon,
              "dependency_floor": dependency_floor, "release_floor": max(earliest)+1,
              "critical_chain": list(reversed(chain)),
              "store_earliest": [{"slot": op[1], "cycle": earliest[i]} for i, op in enumerate(ops) if op[0] == "store"]}
    if status != z3.sat:
        report["reason"] = solver.reason_unknown() if status == z3.unknown else "fixed instruction graph infeasible"
        return report, None
    model = solver.model()
    return report, [model.eval(time).as_long() for time in times]
