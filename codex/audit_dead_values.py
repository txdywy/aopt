"""Report unconsumed scratch definitions; not a deletion pass.

Table assembly may introduce consumers absent from the DAG, so these are
candidates requiring separate validation, never automatically removable ops.
"""
from collections import Counter


def unconsumed_definitions(ops):
    writers = {}
    consumed = set()
    for index, op in enumerate(ops):
        # Reads see the preceding definition, including in-place operations.
        for address in op.reads:
            if address in writers:
                consumed.add(writers[address])
        for address in op.writes:
            writers[address] = index
    return [index for index, op in enumerate(ops)
            if op.writes and index not in consumed]


def report(ops):
    candidates = unconsumed_definitions(ops)
    return {"warning": "DAG-only candidates; assembled tables may add consumers",
            "count": len(candidates),
            "by_engine_tag": dict(Counter(
                f"{ops[i].engine}:{ops[i].tag}" for i in candidates))}
