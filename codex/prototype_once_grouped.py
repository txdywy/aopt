"""Algorithm-only one-time grouping prototype, NOT a VLIW submission.

Uses Python sorting to validate dataflow and output permutation recovery.
Counts unique ancestor demand for a prospective descendant-prefetch design.
Python sorting and dictionary access must not be treated as free ISA work.
"""
import json
import random
from collections import Counter
from tests.frozen_problem import Tree, Input, myhash, reference_kernel


def run(seed, bits, pattern='random'):
    rng = random.Random(seed)
    tree = Tree(10, [rng.getrandbits(bits) for _ in range(2047)])
    initial = [rng.getrandbits(bits) for _ in range(256)]
    if pattern != 'random':
        assert pattern in ('zero', 'ones', 'alternating')
        sequence = {'zero': [0], 'ones': [0xFFFFFFFF],
                    'alternating': [0, 0xFFFFFFFF]}[pattern]
        tree.values = [sequence[i % len(sequence)] for i in range(2047)]
        initial = [sequence[i % len(sequence)] for i in range(256)]
    # Each record keeps its original output slot throughout the traversal.
    records = [(0, value, lane) for lane, value in enumerate(initial)]
    parent_counts = []
    ideal_loads = 0
    for rnd in range(16):
        if rnd == 4:
            records.sort(key=lambda record: record[0])
            parent_counts = [len({index for index, _, _ in records[i:i+8]})
                             for i in range(0,256,8)]
        updated = []
        for offset in range(0,256,8):
            group = records[offset:offset+8]
            nodes = {index: tree.values[index] for index,_,_ in group}
            if 4 <= rnd <= 10:
                ideal_loads += len(nodes)
            for index,value,lane in group:
                value = myhash(value ^ nodes[index])
                index = 2*index+1+(value & 1)
                updated.append((index if index < 2047 else 0, value, lane))
        records = updated
    output_values, output_indices = [None]*256, [None]*256
    for index,value,lane in records:
        assert output_values[lane] is None
        output_values[lane], output_indices[lane] = value,index
    expected = Input([0]*256, initial[:],16)
    reference_kernel(tree,expected)
    assert output_values == expected.values
    assert output_indices == expected.indices
    return {'ideal_saved_loads':1792-ideal_loads,
            'parent_counts':parent_counts,
            # All descendants are prefetched, not just those actually used.
            # Fall back to eight lane loads when prefetch demand exceeds eight.
            'prefetch_loads': {depth: sum(min(8, n*(1 << (depth-4)))
                                          for n in parent_counts)
                               for depth in (4,5,6)}}


if __name__ == '__main__':
    results = [run(seed,bits) for bits in (30,32) for seed in range(20)]
    for pattern in ('zero', 'ones', 'alternating'):
        run(0,32,pattern)
    histogram = Counter(n for result in results for n in result['parent_counts'])
    print(json.dumps({'verified_inputs':len(results),
                      'verified_boundary_patterns':3,
                      'parent_count_histogram':histogram,
                      'mean_prefetch_loads':{depth:sum(r['prefetch_loads'][depth] for r in results)/len(results)
                                             for depth in (4,5,6)},
                      'mean_ideal_saved_loads':sum(r['ideal_saved_loads'] for r in results)/len(results),
                      'warning':'Algorithm prototype only; sort, dictionary, routing and output recovery costs excluded'},indent=2))
