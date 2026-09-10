"""Conservatively fuse an out-of-line return with a following indirect jump.

Addresses and table lengths stay fixed. Every variant returning to a given
main-stream packet must be mergeable; otherwise that entire site is skipped.
"""

from collections import defaultdict

LIMITS = {"alu": 12, "valu": 6, "load": 2, "store": 2, "flow": 1}


def accesses(engine, slot):
    op, *args = slot
    words = lambda base: set(range(base, base + 8))
    if engine == "alu":
        dst, a, b = args
        return {a, b}, {dst}
    if engine == "valu":
        dst, *sources = args
        reads = {sources[0]} if op == "vbroadcast" else set().union(*(words(s) for s in sources))
        return reads, words(dst)
    if engine == "load":
        dst, src, *extra = args
        if op == "const":
            return set(), {dst}
        if op == "load_offset":
            return {src + extra[0]}, {dst + extra[0]}
        if op in ("load", "vload"):
            return {src}, words(dst) if op == "vload" else {dst}
    if engine == "store":
        addr, src = args
        return {addr} | (words(src) if op == "vstore" else {src}), set()
    if engine == "flow" and op == "jump_indirect":
        return {args[0]}, set()
    if engine == "flow" and op == "vselect":
        dst, cond, a, b = args
        return words(cond) | words(a) | words(b), words(dst)
    if engine == "flow" and op == "add_imm":
        dst, src, _ = args
        return {src}, {dst}
    raise ValueError(f"unsupported fusion instruction: {engine} {slot}")


def fuse_returns(program, main_length):
    sites = defaultdict(list)
    for pc in range(main_length, len(program)):
        flow = program[pc].get("flow", [])
        if len(flow) == 1 and flow[0][0] == "jump":
            target = flow[0][1]
            if 0 <= target < main_length:
                following = program[target].get("flow", [])
                if len(following) == 1 and following[0][0] == "jump_indirect":
                    sites[target].append(pc)
    accepted = []
    rejected = defaultdict(int)
    capacity_sites = defaultdict(int)
    capacity_excess = defaultdict(int)
    for target, entries in sites.items():
        following = program[target]
        later_reads, later_writes = set(), set()
        try:
            for engine, slots in following.items():
                for slot in slots:
                    reads, writes = accesses(engine, slot)
                    later_reads |= reads
                    later_writes |= writes
            reason = None
            for pc in entries:
                earlier = program[pc]
                overflow = {e:len(earlier.get(e,[]))+len(following.get(e,[]))-cap
                            for e,cap in LIMITS.items() if e!='flow'
                            and len(earlier.get(e,[]))+len(following.get(e,[]))>cap}
                if overflow:
                    for engine,amount in overflow.items():
                        capacity_sites[engine]+=1
                        capacity_excess[engine]+=amount
                    reason = "capacity"
                    break
                if earlier.get("store") and (following.get("load") or following.get("store")):
                    reason = "memory_order"
                    break
                earlier_writes = set()
                for engine, slots in earlier.items():
                    if engine == "flow":
                        continue
                    for slot in slots:
                        _, writes = accesses(engine, slot)
                        earlier_writes |= writes
                if earlier_writes & (later_reads | later_writes):
                    reason = "register_dependency"
                    break
        except ValueError:
            reason = "unsupported_instruction"
        if reason:
            rejected[reason] += 1
            continue
        for pc in entries:
            merged = {engine: list(slots) for engine, slots in program[pc].items()}
            for engine, slots in following.items():
                if engine != "flow":
                    merged.setdefault(engine, []).extend(slots)
            merged["flow"] = list(following["flow"])
            program[pc] = merged
        accepted.append(target)
    return {"sites": len(sites), "fused_sites": accepted, "rejected": dict(rejected),
            "first_conflicting_variant_capacity_sites":dict(capacity_sites),
            "first_conflicting_variant_excess_slots":dict(capacity_excess)}
