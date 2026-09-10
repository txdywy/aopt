"""Optimistic resource bound for node-homogeneous SIMD/scalar hash mixing.

For each bucket chunk, independently choose ALU/VALU for every hash operation.
Six basic operations cost r scalar slots, four MADs cost2*r scalar slots.
VALU costs one slot, including inactive lanes. All instruction dependencies,
routing, node XOR, memory, control, register pressure and setup are ignored.
Thus this is a necessary bound for THIS fixed hash lowering/bucketing policy,
not a universal bound on the challenge or on new arithmetic formulations.
"""
import itertools
import json
from codex.profile_node_reuse import profile


def optimal_bound(costs):
    # For a fixed number of vector ops, vectorize the largest scalar costs.
    costs = sorted(costs, reverse=True)
    scalar = sum(costs)
    best = ((scalar+11)//12, 0, scalar)
    for vectors, saving in enumerate(costs, 1):
        scalar -= saving
        best = min(best, (max((vectors+5)//6, (scalar+11)//12), vectors, scalar))
    return {'cycles_floor': best[0], 'valu_slots': best[1], 'alu_slots': best[2]}


def bound(rows, node_xor=False):
    chunks = [r for row in rows for r in
              (row['node_bucket_sizes'] if 4 <= row['round'] <= 10 else [8]*32)]
    return optimal_bound([cost for r in chunks for cost in [r]*(6+node_xor)+[2*r]*4])


if __name__ == '__main__':
    # Independent brute force oracle for the allocation algorithm.
    for costs in ([1,2,8], [16,3,3,1], [8]*8, [1]*10):
        oracle = min(max((sum(bits)+5)//6,
                         (sum(c for c,b in zip(costs,bits) if not b)+11)//12)
                     for bits in itertools.product((0,1), repeat=len(costs)))
        assert optimal_bound(costs)['cycles_floor'] == oracle
    traversals = [profile(seed, full32) for full32 in (False, True) for seed in range(20)]
    results = [bound(rows) for rows in traversals]
    with_xor = [bound(rows, node_xor=True)['cycles_floor'] for rows in traversals]
    floors = [r['cycles_floor'] for r in results]
    print(json.dumps({'verified_inputs':len(results), 'hash_only_floor':
                      {'min':min(floors), 'mean':sum(floors)/len(floors), 'max':max(floors)},
                      'hash_plus_node_xor_floor': {'min':min(with_xor), 'mean':sum(with_xor)/len(with_xor), 'max':max(with_xor)},
                      'sample0':results[0], 'brute_force_controls':4,
                      'scope':'Existing ten-op hash, homogeneous node buckets in rounds4..10, free per-op scalar/vector selection; all non-hash work ignored'}, indent=2))
