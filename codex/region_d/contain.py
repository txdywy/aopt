"""Function-class containment for hard SHR-at-I3 wirings.

If t0 is read only by I2, and the pair (I1, I2) computes a family of functions of z
that one instruction already realizes for every constant choice, then the wiring
computes exactly the functions of the shorter program
    [I' = that single instruction, SHR, I4 renumbered, identity pad].
The pad is XOR with a private constant (equal to 0 for the identity). Solving the
reduced program with rdg (exact, every shift amount) therefore decides the original.

Collapsible pairs handled (k* are private constants, t0 = I1):
  affine o affine : I1 = z*k0+k1 (or k0*z+k1), I2 = z*k2+t0 | t0*k2+k3 | t0*k2+z  -> MAD(z,A,B)
  xor o xor       : I1 = z^k0, I2 = t0^k2                                         -> z^K'
Pool numbering: SHR at I3 sees [z, t0, t1, k4, k5]; I4 sees [z, t0, t1, u, k6, k7].
"""
import sys
sys.path.insert(0, '/home/user/aopt')


def affine_in_z(op, idx, pool_z=0, consts=(1, 2)):
    """Instruction over pool [z, k0, k1] that is z*k + k' (any arrangement)."""
    if op != 'mad':
        return False
    a, b, c = idx
    return sorted((a, b)) in ([0, 1], [0, 2]) and c in consts and c not in (a, b)


def reduce_pos2(order, wiring):
    """Return (order', wiring') of the equivalent pos=1 program, or None."""
    if len(order) != 4 or order[2] != '>>':
        return None
    i1, i2, i3, i4 = wiring
    if i3[0] != 2:                                  # shift must read t1
        return None
    if 1 in i3 or 1 in i4:                          # t0 must feed only I2
        return None
    o1, o2 = order[0], order[1]
    single = None
    if affine_in_z(o1, i1) and o2 == 'mad':
        a, b, c = i2                                # pool [z, t0, k2, k3] = [0, 1, 2, 3]
        forms = {(0, 2, 1), (2, 0, 1), (1, 2, 3), (2, 1, 3), (1, 2, 0), (2, 1, 0)}
        if (a, b, c) in forms:
            single = ('mad', (0, 1, 2))             # z*A + B
    elif o1 == '^' and tuple(i1) == (0, 1) and o2 == '^' and tuple(sorted(i2)) == (1, 2):
        single = ('^', (0, 1))                      # z ^ K'
    if single is None:
        return None
    # renumber I4: old pool [z, t0, t1, u, k6, k7] -> new I3 pool [z, t1, u, k4, k5]
    remap = {0: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    if any(j not in remap for j in i4):
        return None
    new_i3 = tuple(remap[j] for j in i4)
    order2 = (single[0], '>>', order[3], '^')
    wiring2 = (single[1], (1, 2), new_i3, (3, 4))   # SHR(t1', k0'); pad = t2' ^ k
    return order2, wiring2


if __name__ == '__main__':
    import time
    import rdg
    o, w = ('mad', 'mad', '>>', 'mad'), ((0, 1, 2), (0, 2, 1), (2, 3), (0, 3, 2))
    red = reduce_pos2(o, w)
    print('original', o, w)
    print('reduced ', red)
    t = time.time()
    r = rdg.solve(red[0], red[1], 8, 120)
    print('reduced program, every shift, c=8:', r['result'], '%.2fs' % (time.time() - t))
