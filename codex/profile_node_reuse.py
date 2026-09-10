"""Read-only traversal reuse profile; ideal load savings exclude routing costs.

No kernel or simulator modifications. Final results are checked independently
against the frozen reference. Random samples are not worst-case guarantees.
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from tests.frozen_problem import Tree, Input, myhash, reference_kernel


def profile(seed, full32):
    rng = random.Random(seed)
    width = 32 if full32 else 30
    tree = Tree(10, [rng.getrandbits(width) for _ in range(2047)])
    initial = [rng.getrandbits(width) for _ in range(256)]
    values = initial[:]
    indices = [0]*256
    rows = []
    once_sorted_order = None
    for rnd in range(16):
        if rnd == 4:
            once_sorted_order = sorted(range(256), key=lambda lane: indices[lane])
        row = {'round': rnd, 'unique_global': len(set(indices))}
        if once_sorted_order is not None:
            ordered = [indices[lane] for lane in once_sorted_order]
            row['once_sorted_group8_saved_loads'] = sum(
                8-len(set(ordered[i:i+8])) for i in range(0,256,8))
        populations = Counter(indices)
        row['node_homogeneous_vectors'] = sum((n+7)//8 for n in populations.values())
        row['node_homogeneous_padding'] = 8*row['node_homogeneous_vectors']-256
        row['node_bucket_sizes'] = [8 for n in populations.values() for _ in range(n//8)] + [n%8 for n in populations.values() if n%8]
        for size in (2, 8, 16, 32, 256):
            row[f'ideal_saved_loads_group{size}'] = sum(
                size-len(set(indices[i:i+size])) for i in range(0, 256, size))
        rows.append(row)
        for lane in range(256):
            value = myhash(values[lane] ^ tree.values[indices[lane]])
            index = 2*indices[lane] + 1 + (value & 1)
            values[lane] = value
            indices[lane] = index if index < len(tree.values) else 0
    expected = Input([0]*256, initial[:], 16)
    reference_kernel(tree, expected)
    assert values == expected.values and indices == expected.indices
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=20)
    args = parser.parse_args()
    assert args.seeds > 0
    records = [profile(seed, full32) for full32 in (False, True)
               for seed in range(args.seeds)]
    totals = defaultdict(list)
    for rows in records:
        for size in (2, 8, 16, 32, 256):
            totals[size].append(sum(row[f'ideal_saved_loads_group{size}']
                                    for row in rows[4:11]))
    result = {'verified_inputs': len(records), 'load_rounds': list(range(4, 11)),
              'once_sorted_group8_savings': {
                  'min': min(sum(row['once_sorted_group8_saved_loads'] for row in rows[4:11]) for rows in records),
                  'mean': sum(sum(row['once_sorted_group8_saved_loads'] for row in rows[4:11]) for rows in records)/len(records),
                  'max': max(sum(row['once_sorted_group8_saved_loads'] for row in rows[4:11]) for rows in records)},
              'ideal_savings': {size: {'min': min(v), 'mean': sum(v)/len(v), 'max': max(v)}
                                for size, v in totals.items()},
              'full_group8_pair_comparisons': 7*32*28,
              'node_homogeneous_padding_by_round': {
                  rnd: {'min': min(rows[rnd]['node_homogeneous_padding'] for rows in records),
                        'mean': sum(rows[rnd]['node_homogeneous_padding'] for rows in records)/len(records),
                        'max': max(rows[rnd]['node_homogeneous_padding'] for rows in records)}
                  for rnd in range(4, 11)},
              'extra_hash_lane_work_if_ten_ops_per_padded_lane': {
                  'min': min(10*sum(row['node_homogeneous_padding'] for row in rows[4:11]) for rows in records),
                  'mean': sum(10*sum(row['node_homogeneous_padding'] for row in rows[4:11]) for rows in records)/len(records),
                  'max': max(10*sum(row['node_homogeneous_padding'] for row in rows[4:11]) for rows in records)},
              'warning': 'Ideal duplicate-load removal only; no grouping, detection, copy, branch, or register costs. Not a cycle result.'}
    print(json.dumps(result, indent=2))
