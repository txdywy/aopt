"""Check actual ISA operands, including every out-of-line table variant."""
from functools import lru_cache
import json
from codex.inspect_feedback_tail import capture
from codex.fuse_table_returns import accesses


@lru_cache(maxsize=100000)
def overlap(engine, slot):
    if engine == 'flow' and slot[0] in ('jump','halt','pause'):
        return False, False
    if engine == 'debug':
        return False, False
    reads,writes=accesses(engine,slot)
    target=set(range(47,55))
    return bool(reads & target),bool(writes & target)


if __name__ == '__main__':
    program=capture(scalar_cache=True)
    readers=[];writers=[];read_count=write_count=0
    for pc,packet in enumerate(program):
        for engine,slots in packet.items():
            for slot in slots:
                read,write=overlap(engine,tuple(slot))
                if read:
                    read_count+=1
                    if len(readers)<20:readers.append((pc,engine,slot))
                if write:
                    write_count+=1
                    if len(writers)<20:writers.append((pc,engine,slot))
    print(json.dumps({'packets':len(program),'read_count':read_count,
                      'write_count':write_count,'readers':readers,'writers':writers}),flush=True)
