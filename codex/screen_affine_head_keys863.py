"""Keep the SAME output XOR key across all separated rectangles and residues.

This is a necessary-condition screen, not a synthesizer. The first affine
word is t=A*x+B. Any subsequent bitwise prefix over x,t,constants followed
by any integer polynomial must satisfy the recorded mixed differences.
"""
import argparse
import base64
import hashlib
from itertools import combinations
import json
from pathlib import Path
import time
import zlib

import numpy as np

from codex.screen_affine_head_rectangles863 import axis_separated, control, rectangles
from codex.search_cross_encoding863 import C1, middle


def encode(array):
    array = np.asarray(array, dtype='<u8')
    raw = array.tobytes()
    return dict(shape=list(array.shape), dtype='<u8', sha256=hashlib.sha256(raw).hexdigest(),
                zlib_base64=base64.b64encode(zlib.compress(raw, 9)).decode())


def decode(data):
    raw = zlib.decompress(base64.b64decode(data['zlib_base64']))
    assert hashlib.sha256(raw).hexdigest() == data['sha256']
    return np.frombuffer(raw, dtype=data['dtype']).reshape(data['shape'])


def key_mask(corners):
    keys = np.arange(256, dtype=np.int64)
    p, q, r, s = [value ^ keys for value in corners]
    allowed = ((s-q-r+p) & 255) == 0
    return np.packbits(allowed, bitorder='little').view('<u8')


def cubes(max_dimension=2):
    if max_dimension not in range(2, 9):
        raise ValueError('maximum dimension must be in 2..8')
    for base, h, k in rectangles():
        yield base, (h, k)
    for dimension in range(3, max_dimension+1):
        for bits in combinations(range(8), dimension):
            if 3*dimension+sum(bits) < 11:
                continue
            axes = tuple(1 << bit for bit in bits)
            union = sum(axes)
            for base in range(256):
                if not base & union:
                    yield base, axes


def offsets(axes):
    return [sum(axis for i, axis in enumerate(axes) if subset >> i & 1)
            for subset in range(1 << len(axes))]


def mixed_key_mask(corners):
    """Signed mixed difference on a power-of-two cube, corner order by subset."""
    dimension = (len(corners)-1).bit_length()
    assert len(corners) == 1 << dimension
    keys = np.arange(256, dtype=np.int64)
    deltas = np.zeros(256, dtype=np.int64)
    for subset, value in enumerate(corners):
        sign = (-1)**(dimension-subset.bit_count())
        deltas += sign*(value ^ keys)
    return np.packbits((deltas & 255) == 0, bitorder='little').view('<u8')


def cube_separated(a, b, base, axes):
    if len(axes) == 2:
        return axis_separated(a, b, base, *axes)
    p = (a*base+b) & 255
    deltas = [((a*(base+axis)+b) & 255) ^ p for axis in axes]
    combined = [delta | axis for delta, axis in zip(deltas, axes)]
    valid = np.ones(a.shape, dtype=np.bool_)
    for left, right in combinations(combined, 2):
        valid &= (left & right) == 0
    selected = np.flatnonzero(valid)
    if not len(selected):
        return valid
    aa, bb, pp = a[selected], b[selected], p[selected]
    ds = [delta[selected] for delta in deltas]
    good = np.ones(len(selected), dtype=np.bool_)
    for subset, offset in enumerate(offsets(axes)):
        predicted = pp.copy()
        for i, delta in enumerate(ds):
            if subset >> i & 1:
                predicted ^= delta
        good &= ((aa*(base+offset)+bb) & 255) == predicted
    valid[selected] = good
    return valid


def residue_screen(residue=0, positive=False, max_dimension=2):
    if residue not in range(8):
        raise ValueError('residue must be in 0..7')
    started = time.monotonic()
    target = control if positive else lambda x: middle(x ^ C1)
    values = [(target(8*y+residue) >> 3) & 255 for y in range(256)]
    assert len({target(8*y+residue) & 7 for y in range(256)}) == 1
    ids = np.arange(65536, dtype=np.int64)
    masks = np.full((65536, 4), np.iinfo(np.uint64).max, dtype=np.uint64)
    witnesses = []
    visited = 0
    for base, axes in cubes(max_dimension):
        visited += 1
        corners = [values[base+offset] for offset in offsets(axes)]
        allowed = mixed_key_mask(corners)
        if np.all(allowed == np.iinfo(np.uint64).max):
            continue
        a, b = ids >> 8, ids & 255
        separated = cube_separated(a, b, base, axes)
        selected = np.flatnonzero(separated)
        if not len(selected):
            continue
        after = masks[selected] & allowed
        changed = np.any(after != masks[selected], axis=1)
        if not np.any(changed):
            continue
        masks[selected] = after
        keep = np.any(masks, axis=1)
        witness = dict(base_y=base, axes_y=list(axes), target_upper_values=corners,
            allowed_upper_key_mask=[f'{int(word):016x}' for word in allowed],
            restricted_pairs=int(np.count_nonzero(changed)),
            removed_pairs=int(np.count_nonzero(~keep)))
        if len(axes) == 2:
            witness.update(h_y=axes[0], k_y=axes[1])
        witnesses.append(witness)
        ids, masks = ids[keep], masks[keep]
        if not len(ids):
            break
    full_table = np.zeros((65536, 4), dtype=np.uint64)
    full_table[ids] = masks
    known_id = 5*256+((5*residue+19) >> 3)
    report = dict(width=11, residue=residue, control=positive, max_dimension=max_dimension,
        input_key=None if positive else C1, total_affine_pairs=65536,
        remaining_affine_pairs=len(ids), rectangles_tested=visited,
        witnesses=witnesses, remaining_upper_key_assignments=int(np.bitwise_count(masks).sum()),
        positive_pair_and_zero_key_retained=bool(full_table[known_id, 0] & np.uint64(1)) if positive else None,
        table=encode(full_table), seconds=round(time.monotonic()-started, 3),
        result='affine_prefix_polynomial_tail_excluded' if not len(ids) else 'necessary_pairs_survive',
        new_executable=False,
        scope='t=A*x+B; arbitrary bitwise prefix over x,t,constants; arbitrary integer-polynomial '
              'continuation. Correlated output XOR key retained. Survivors are only necessary conditions.')
    return report


def joint_screen(positive=False, max_dimension=2):
    started = time.monotonic()
    reports = []
    a = b = masks = None
    for residue in range(8):
        row = residue_screen(residue, positive, max_dimension)
        table = decode(row['table'])
        if a is None:
            # Lift upper-domain pairs to ALL original 11-bit A,B, including
            # every A high part and B low part. No lost constant bits.
            pair_ids = np.flatnonzero(np.any(table, axis=1))
            pair_ids = np.repeat(pair_ids, 64)
            lifts = np.tile(np.arange(64, dtype=np.int64), len(pair_ids)//64)
            a = (pair_ids >> 8) + ((lifts >> 3) << 8)
            b = ((pair_ids & 255) << 3) + (lifts & 7)
            masks = table[pair_ids].copy()
        else:
            masks &= table[((a & 255) << 8) | (((a*residue+b) >> 3) & 255)]
        keep = np.any(masks, axis=1)
        a, b, masks = a[keep], b[keep], masks[keep]
        row['joint_remaining_full_pairs'] = len(a)
        row['joint_remaining_upper_key_assignments'] = int(np.bitwise_count(masks).sum())
        reports.append(row)
        print(json.dumps(dict(residue=residue, pairs=len(a),
            upper_key_assignments=row['joint_remaining_upper_key_assignments'])), flush=True)
        if not len(a):
            break
    known = np.flatnonzero((a == 5) & (b == 19))
    return dict(width=11, control=positive, max_dimension=max_dimension, total_full_pairs=1 << 22,
        remaining_full_pairs=len(a), residues=reports,
        remaining_upper_key_assignments=int(np.bitwise_count(masks).sum()),
        surviving_pairs=encode(np.column_stack((a, b))), surviving_key_masks=encode(masks),
        positive_pair_and_zero_key_retained=bool(len(known) and masks[known[0], 0] & np.uint64(1)) if positive else None,
        result='affine_prefix_polynomial_tail_excluded' if not len(a) else 'necessary_pairs_survive',
        new_executable=False, seconds=round(time.monotonic()-started, 3), scope=reports[0]['scope'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', action='store_true')
    parser.add_argument('--joint', action='store_true')
    parser.add_argument('--residue', type=int, default=0)
    parser.add_argument('--max-dimension', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = (joint_screen(args.control, args.max_dimension) if args.joint else
              residue_screen(args.residue, args.control, args.max_dimension))
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result.get(key) for key in ('result', 'remaining_affine_pairs',
        'remaining_full_pairs', 'remaining_upper_key_assignments', 'seconds',
        'positive_pair_and_zero_key_retained')}))


if __name__ == '__main__':
    main()
