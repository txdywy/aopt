"""Screen data-dependent cheap-pair fast paths; no assumed input constants."""
from collections import Counter
import random
import json


def profile():
    counts = Counter({key:0 for key in ('identical','at_most_two_different_bits',
                      'only_low_or_sign_bit_differs','cached_at_most_two_different_bits')})
    popcounts = Counter()
    for seed in range(40):
        rng = random.Random(seed)
        nodes = [rng.getrandbits(32) for _ in range(2047)]
        # Every sibling pair, including the shallow cached pairs.
        for left in range(1,2047,2):
            delta = nodes[left]^nodes[left+1]
            depth = (left+1).bit_length()-1
            counts['pairs'] += 1
            popcounts[delta.bit_count()] += 1
            if delta == 0:
                counts['identical'] += 1
            if delta.bit_count() <= 2:
                counts['at_most_two_different_bits'] += 1
            if delta & 0x7ffffffe == 0:
                counts['only_low_or_sign_bit_differs'] += 1
            if depth <= 4:
                counts['cached_depth_1_to_4_pairs'] += 1
                if delta.bit_count() <= 2:
                    counts['cached_at_most_two_different_bits'] += 1
    return dict(counts=counts,xor_popcount_histogram=dict(sorted(popcounts.items())))


if __name__ == '__main__':
    result = profile()
    assert result['counts']['pairs'] == 40*1023
    assert result['counts']['cached_depth_1_to_4_pairs'] == 40*15
    print(json.dumps(result,indent=2))
