"""Exact bit-0 conditions for the round-0 single-shift templates left open.

Context (grammar4_round0_863): with one shift w >> s, the output's bit 0 is a
function of v0 and bit s of w, so every solution satisfies

    bit_s(w(v)) == y19(v) ^ g(v0)   for every v < 2^30,  y19 = bit 19 of 4097(v^R)+C0.

w is a T-function of v, so at s = 19 only v and the constants mod 2^20 matter.
This module finds the exact solution sets of that condition for the shift
inputs whose z3 version times out, so the full search can pin their low bits.

1. aff(xk(u)) = mu*(u ^ K) + nu at s = 19 + d (mu = 2^d * odd, see
   grammar4_round0_pinned863). With z = u ^ K and e = K ^ R mod 2^20 the
   condition is bit19(mu' z + nu') == b(z ^ e) ^ g'(z0), b(x) = bit 19 of
   4097x + C0: AFFINE_C's check with root e. Flipping bit 0 of an even z moves
   mu' z + nu' by mu', which crosses bit 19 exactly on two antipodal arcs, so
   the set of even z whose bit 19 changes is Q u (Q + 2^19) for an arithmetic
   progression Q. On the target side the same set is X0 ^ e with
   X0 = {x even : b(x ^ 1) != b(x)} (it does not depend on g; its complement
   among even z, the other case, has the same form). Hence pi(X0 ^ e), its
   image mod 2^19, must be an arithmetic progression. AP_C tests every e
   (mod 2^19, bit 0 clear; bits 0 and 19 do not change the set). The survivors
   are then solved exactly with AFFINE_C for each of their 4 lifts.

2. Shift inputs with two constants a, b: BF2_C enumerates both mod 2^20 and
   checks every u < 2^20 for the survivors of an 8-sample filter.

3. prod(u, aff(u)) = a u^2 + b u + K. Flipping u19 moves it by 2^19 b mod 2^20,
   so b is odd at s = 19. For s = 20..29 the bit must not depend on input bits
   t = s, s-1, ..., 20 (y19 does not): flipping bit t moves w by +-2^t (2au + b)
   mod 2^32, and inductively this forces b = 0 mod 2^(s-19), a = 0 mod 2^(s-20);
   bit s is then bit 20 of A u^2 + 2B u + K1 (a = 2^(s-20) A, b = 2^(s-19) B),
   flipping u19 forces A even and B odd, and bit 20 of 2A'' u^2 + 2B u + K1 is
   bit 19 of A'' u^2 + B u + (K1 >> 1). So s = 20..29 reduce to s = 19, which
   QUAD_C settles: a and odd b enumerated, K by arc intersection (g and g ^ 1
   are the same up to K + 2^19, so g0 = 0).

4. Rigid differences, for s = 19..24. Flip input bit j >= max(s, 20) (still in
   the domain). Every value not computed from the shift is a T-function of v
   and moves by a multiple of 2^j, so modulo M = 2^(32-s) it is unchanged. The
   shift output i3 = w >> s has 32 - s bits and moves by 2^(j-s) * Z mod M,
   where 2^j Z is the move of w (j >= s keeps it exact). If the last
   instruction is affine in i3 (lin1, lin2, prod, the three MADs), the output
   moves by coeff * 2^(j-s) * Z mod M, coeff being 1, a constant, or a value
   unchanged mod M. When w is built so that Z mod 2^(32-j) depends only on
   v mod 2^(32-j) and v_j ("rigid": MAD, linear forms and products of rigid
   values, and XOR/AND/OR with constants or each other only on values whose
   move is a single bit flip), two inputs that agree mod M and on bit j move
   the output by the same amount mod M, for every choice of constants. The
   target moves by m * (T(x ^ 2^j) - T(x)), m odd; a witness pair for which
   those moves differ mod M excludes s. Its XOR with y >> 19 is what makes the
   target's move depend on y's top bits; XOR/AND/OR as the last instruction
   escapes the argument.

5. The same argument for any concrete shift amounts (multi-shift templates),
   by tracking each value's move D = V(v ^ 2^j) - V(v) as an integer: D = 2^t Z
   (mod 2^w), where Z mod 2^k is fixed by (v mod 2^k, v_j) for k <= w - t, and
   whether V mod 2^k is fixed by v mod 2^k (true for T-functions). u moves by
   exactly +-2^j. Affine and linear forms keep the move form; a product keeps
   it when the other factor is a T-function, and otherwise only the multiple of
   2^t survives; XOR/AND/OR of single-bit flips stay single-bit flips, else only
   the lowest changed bit is known. A shift by s needs t >= s and w >= s, gives
   2^(t-s) Z mod 2^(w-s), and is 0 when s is at least the value's width (u has
   30 bits). If the output's move is known mod 2^e (w >= e), a witness pair that
   agrees mod 2^e and on bit j but whose target moves differ mod 2^e excludes
   the shift amounts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import functools
import json
from pathlib import Path
import subprocess
import tempfile

from codex import grammar3_head863 as g
from codex import grammar4_round0_863 as h

AP_C = r"""
#include <stdint.h>
#include <stdio.h>
#define N (1u << 20)
#define H (1u << 19)
static int b(uint32_t x) { return (((4097u * x + %(C0)uu) & (N - 1)) >> 19) & 1; }
static uint8_t inset[H];
int main(void) {
  static uint32_t Y[H], Z[H]; int n = 0;
  for (uint32_t x = 0; x < H; x += 2)   /* x ^ 2^19 gives the same residue mod 2^19 */
    if (b(x ^ 1) != b(x)) Y[n++] = x;
  printf("size %%d\n", n);
  for (uint32_t e = 0; e < H; e += 2) {
    for (int i = 0; i < n; i++) { Z[i] = Y[i] ^ e; inset[Z[i]] = 1; }
    /* a progression has at most one element without a successor, so one of the
       first two elements has one and the step is a difference from it */
    int found = 0;
    for (int an = 0; an < 2 && !found; an++)
      for (int j = 0; j < n && !found; j++) {
        uint32_t d = (Z[j] - Z[an]) & (H - 1), miss = 0;
        if (!d) continue;
        for (int i = 0; i < n && miss < 2; i++) miss += !inset[(Z[i] + d) & (H - 1)];
        if (miss < 2) { found = 1; printf("key %%u %%u\n", e, d); }
      }
    for (int i = 0; i < n; i++) inset[Z[i]] = 0;
  }
  printf("done\n");
  return 0;
}
"""

# BF2_C checks bit 19 on u < 2^20. BFW_C is the same for bit B on u < 2^(B+1), with a odd and
# bit B of b clear (both forced by flipping u_B, see grammar4_round0_pinned_audit863 rule bit20).
BFW_C = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define B %(B)d
#define N (1u << (B + 1))
#define MSK (N - 1)
#define C0 %(C0)uu
static uint32_t R;
static inline uint32_t y19(uint32_t u) { return ((4097u * (u ^ R) + C0) >> 19) & 1; }
static inline uint32_t W(uint32_t u, uint32_t a, uint32_t b) { return (uint32_t)(%(EXPR)s); }
#define BLK 4096
int main(int argc, char **argv) {
  R = (uint32_t)strtoul(argv[1], 0, 0);
  uint32_t slice = atoi(argv[2]), nsl = atoi(argv[3]);
  uint32_t us[24], ys[24], x = 2463534242u, y8 = 0;
  for (int i = 0; i < 24; i++) {
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    us[i] = (x & MSK & ~1u) | (i & 1); ys[i] = y19(us[i]);
    if (i < 8) y8 |= ys[i] << i;
  }
  const uint32_t EV = 0x55, OD = 0xAA;
  static uint32_t mk[BLK];
  long cand = 0;
  for (uint32_t b = slice; b < N; b += nsl) {
    if ((b >> B) & 1) continue;
    for (uint32_t a0 = 1; a0 < N; a0 += 2 * BLK) {
      for (uint32_t k = 0; k < BLK; k++) {
        uint32_t a = a0 + 2 * k, m = 0;
        for (int i = 0; i < 8; i++) m |= ((W(us[i], a, b) >> B) & 1) << i;
        mk[k] = m ^ y8;
      }
      for (uint32_t k = 0; k < BLK; k++) {
        uint32_t m = mk[k], a = a0 + 2 * k, ok = 1;
        if (m != 0 && m != EV && m != OD && m != (EV | OD)) continue;
        uint32_t g0 = m & 1, g1 = (m >> 1) & 1;
        cand++;
        for (int i = 8; i < 24 && ok; i++)
          ok = (((W(us[i], a, b) >> B) & 1) ^ ys[i]) == ((us[i] & 1) ? g1 : g0);
        for (uint32_t u = 0; u < N && ok; u++)
          ok = (((W(u, a, b) >> B) & 1) ^ y19(u)) == ((u & 1) ? g1 : g0);
        if (ok) printf("sol %%u %%u %%u %%u\n", a, b, g0, g1);
      }
    }
  }
  printf("done %%ld\n", cand);
  return 0;
}
"""

BF2_C = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define N (1u << 20)
#define MSK (N - 1)
#define C0 %(C0)uu
static uint32_t R;
static inline uint32_t y19(uint32_t u) { return ((4097u * (u ^ R) + C0) >> 19) & 1; }
static inline uint32_t W(uint32_t u, uint32_t a, uint32_t b) { return (uint32_t)(%(EXPR)s); }
#define BLK 4096
int main(int argc, char **argv) {
  R = (uint32_t)strtoul(argv[1], 0, 0);
  uint32_t slice = atoi(argv[2]), nsl = atoi(argv[3]);
  uint32_t us[24], ys[24], x = 2463534242u, y8 = 0;
  for (int i = 0; i < 24; i++) {          /* parities alternate: even, odd, ... */
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    us[i] = (x & MSK & ~1u) | (i & 1); ys[i] = y19(us[i]);
    if (i < 8) y8 |= ys[i] << i;
  }
  const uint32_t EV = 0x55, OD = 0xAA;
  static uint32_t mk[BLK];
  long cand = 0;
  for (uint32_t b = slice; b < N; b += nsl)
    for (uint32_t a0 = 0; a0 < N; a0 += BLK) {
      for (uint32_t k = 0; k < BLK; k++) {
        uint32_t a = a0 + k, m = 0;
        for (int i = 0; i < 8; i++) m |= ((W(us[i], a, b) >> 19) & 1) << i;
        mk[k] = m ^ y8;
      }
      for (uint32_t k = 0; k < BLK; k++) {
        uint32_t m = mk[k], a = a0 + k, ok = 1;
        if (m != 0 && m != EV && m != OD && m != (EV | OD)) continue;
        uint32_t g0 = m & 1, g1 = (m >> 1) & 1;
        cand++;
        for (int i = 8; i < 24 && ok; i++)
          ok = (((W(us[i], a, b) >> 19) & 1) ^ ys[i]) == ((us[i] & 1) ? g1 : g0);
        for (uint32_t u = 0; u < N && ok; u++)
          ok = (((W(u, a, b) >> 19) & 1) ^ y19(u)) == ((u & 1) ? g1 : g0);
        if (ok) printf("sol %%u %%u %%u %%u\n", a, b, g0, g1);
      }
    }
  printf("done %%ld\n", cand);
  return 0;
}
"""

QUAD_C = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define N (1u << 20)
#define MSK (N - 1)
#define H (1u << 19)
#define C0 %(C0)uu
static uint32_t R;
static inline uint32_t y19(uint32_t u) { return ((4097u * (u ^ R) + C0) >> 19) & 1; }
/* intersect the arc [lo, lo + len) with [p, p + H); returns the new length */
static inline uint32_t meet(uint32_t *lo, uint32_t len, uint32_t p) {
  uint32_t d = (p - *lo) & MSK, e = (*lo - p) & MSK;
  if (d < len) { len -= d; *lo = p; return len < H ? len : H; }
  if (e < H) { uint32_t r = H - e; return r < len ? r : len; }
  return 0;
}
int main(int argc, char **argv) {
  R = (uint32_t)strtoul(argv[1], 0, 0);
  uint32_t slice = atoi(argv[2]), nsl = atoi(argv[3]);
  uint32_t us[32], ys[32], x = 88172645u;
  for (int i = 0; i < 32; i++) {
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    us[i] = x & MSK; ys[i] = y19(us[i]);
  }
  long cand = 0;
  for (uint32_t a = slice; a < N; a += nsl)
    for (uint32_t b = 1; b < N; b += 2)
      for (uint32_t g1 = 0; g1 < 2; g1++) {
        uint32_t lo = 0, len = N;
        for (int i = 0; i < 32; i++) {
          uint32_t u = us[i], t = ys[i] ^ (u & 1 & g1), p = (t * H - a * u * u - b * u) & MSK;
          if (len == N) { lo = p; len = H; } else if (!(len = meet(&lo, len, p))) break;
        }
        if (!len) continue;
        cand++;
        for (uint32_t u = 0; u < N && len; u++)
          len = meet(&lo, len, ((y19(u) ^ (u & 1 & g1)) * H - a * u * u - b * u) & MSK);
        if (len) printf("sol %%u %%u %%u %%u %%u\n", a, b, g1, lo, len);
      }
  printf("done %%ld\n", cand);
  return 0;
}
"""

# Reads of u and of an affine value (rule 6 of grammar4_round0_pinned_audit863):
#   G1: bit19(mu u + nu) == y19(u) ^ G(u0, u_t), t = 1..19, G depending on u_t;
#   P18: bit18(mu u + nu) == y19(u) ^ u19 ^ g(u0) on u < 2^19 (y19 ^ u19 depends on u mod 2^19).
PAIR_C = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define C0 %(C0)uu
static uint32_t R;
static inline uint32_t y19(uint32_t u) { return ((4097u * (u ^ R) + C0) >> 19) & 1; }
static inline uint32_t meet(uint32_t *lo, uint32_t len, uint32_t p, uint32_t msk, uint32_t h) {
  uint32_t d = (p - *lo) & msk, e = (*lo - p) & msk;
  if (d < len) { len -= d; *lo = p; return len < h ? len : h; }
  if (e < h) { uint32_t r = h - e; return r < len ? r : len; }
  return 0;
}
/* target bit for u under mode: G1 with (t, G), or P18 with g */
static int T1, TG, MODE;
static inline uint32_t tgt(uint32_t u) {
  if (MODE == 0) return y19(u) ^ ((TG >> ((u & 1) | (((u >> T1) & 1) << 1))) & 1);
  return y19(u) ^ ((u >> 19) & 1) ^ ((TG >> (u & 1)) & 1);
}
static long run(int bits, int bit) {
  uint32_t n = 1u << bits, msk = n - 1, h = 1u << bit, us[48], x = 2654435761u;
  long found = 0;
  for (int i = 0; i < 48; i++) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; us[i] = x & msk; }
  for (uint32_t mu = 1; mu < n; mu += 2) {
    uint32_t lo = 0, len = n;
    for (int i = 0; i < 48 && len; i++) {
      uint32_t p = (tgt(us[i]) * h - mu * us[i]) & msk;
      if (len == n) { lo = p; len = h; } else len = meet(&lo, len, p, msk, h);
    }
    for (uint32_t u = 0; u < n && len; u++)
      len = meet(&lo, len, (tgt(u) * h - mu * u) & msk, msk, h);
    if (len) { printf("sol %%d %%d %%d %%u %%u %%u\n", MODE, T1, TG, mu, lo, len); found++; }
  }
  return found;
}
int main(int argc, char **argv) {
  R = (uint32_t)strtoul(argv[1], 0, 0);
  MODE = 0;
  for (T1 = 1; T1 <= 19; T1++)
    for (TG = 0; TG < 16; TG++)
      if ((TG & 1) != ((TG >> 2) & 1) || ((TG >> 1) & 1) != ((TG >> 3) & 1)) run(20, 19);
  MODE = 1; T1 = 0;
  for (TG = 0; TG < 4; TG++) run(19, 18);
  printf("done\n");
  return 0;
}
"""


# G2: bit19(mu u + nu) == y19(u) ^ G(u0, bit_p(mu u + nu)), p = 0..18. On the inputs
# u = k * 2^(p+1) (k < 2^(19-p)) bit p of mu u + nu and u0 stay fixed, while bit 19 is the top
# bit of H + k * mu mod 2^(19-p). So y19(k * 2^(p+1)) ^ const must be such a top-bit sequence
# for some odd mu (mod 2^(19-p)) and H; SUB_C finds every (p, mu, const) that allows it.
SUB_C = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define C0 %(C0)uu
static uint32_t R;
static inline uint32_t y19(uint32_t u) { return ((4097u * (u ^ R) + C0) >> 19) & 1; }
int main(int argc, char **argv) {
  R = (uint32_t)strtoul(argv[1], 0, 0);
  long found = 0;
  for (int p = 0; p <= 18; p++) {
    uint32_t n = 1u << (19 - p), msk = n - 1, h = n >> 1;
    for (uint32_t mu = 1; mu < n; mu += 2)
      for (uint32_t c = 0; c < 2; c++) {
        uint32_t lo = 0, len = n;
        for (uint32_t k = 0; k < n && len; k++) {
          uint32_t t = y19(k << (p + 1)) ^ c, a = (t * h - mu * k) & msk;   /* arc [a, a + h) */
          if (len == n) { lo = a; len = h; continue; }
          uint32_t d = (a - lo) & msk, e = (lo - a) & msk;
          if (d < len) { len -= d; lo = a; if (len > h) len = h; }
          else if (e < h) { uint32_t r = h - e; if (r < len) len = r; }
          else len = 0;
        }
        if (len) { printf("sol %%d %%u %%u %%u %%u\n", p, mu, c, lo, len); found++; }
      }
  }
  /* p = 16, 17: the classes c0 = u mod 2^(p+1) must all allow one common mu mod 2^(19-p) */
  for (int p = 16; p <= 17; p++) {
    uint32_t n = 1u << (19 - p), msk = n - 1, h = n >> 1, common = 0;
    for (uint32_t mu = 1; mu < n; mu += 2) common |= 1u << mu;
    for (uint32_t c0 = 0; c0 < (1u << (p + 1)) && common; c0++) {
      uint32_t ok = 0;
      for (uint32_t mu = 1; mu < n; mu += 2)
        for (uint32_t c = 0; c < 2; c++) {
          uint32_t lo = 0, len = n;
          for (uint32_t k = 0; k < n && len; k++) {
            uint32_t t = y19(c0 + (k << (p + 1))) ^ c, a = (t * h - mu * k) & msk;
            if (len == n) { lo = a; len = h; continue; }
            uint32_t d = (a - lo) & msk, e = (lo - a) & msk;
            if (d < len) { len -= d; lo = a; if (len > h) len = h; }
            else if (e < h) { uint32_t r = h - e; if (r < len) len = r; }
            else len = 0;
          }
          if (len) ok |= 1u << mu;
        }
      common &= ok;
    }
    printf("common %%d %%u\n", p, common);
  }
  /* p = 18: bit19(V) ^ bit18(V) = bit19(V + 2^18), so every class is an arc condition */
  {
    uint32_t n = 1u << 20, msk = n - 1, h = 1u << 19;
    for (uint32_t G = 0; G < 16; G++) {
      uint32_t cc = ((G & 1) != ((G >> 2) & 1)) | ((((G >> 1) & 1) != ((G >> 3) & 1)) << 1);
      if (!cc) continue;                     /* G must depend on the bit-18 read */
      for (uint32_t mu = 1; mu < n; mu += 2) {
        uint32_t lo = 0, len = n;
        for (uint32_t u = 0; u < n && len; u++) {
          /* G(u0, b) = dep * b ^ g0, g0 = G(u0, 0); the read b moves the arc by 2^18 */
          uint32_t u0 = u & 1, dep = (cc >> u0) & 1, g0 = (G >> u0) & 1;
          uint32_t t = y19(u) ^ g0, a = (t * h - mu * u - dep * (h >> 1)) & msk;
          if (len == n) { lo = a; len = h; continue; }
          uint32_t d = (a - lo) & msk, e = (lo - a) & msk;
          if (d < len) { len -= d; lo = a; if (len > h) len = h; }
          else if (e < h) { uint32_t r = h - e; if (r < len) len = r; }
          else len = 0;
        }
        if (len) { printf("sol18 %%u %%u %%u %%u\n", G, mu, lo, len); found++; }
      }
    }
  }
  /* p = 17: with step d = 2^17 / mu (mod 2^20), mu u + nu moves by exactly 2^17, so on
     u0' + k d (k < 8) the left side runs through bit19 ^ c * bit17 of eight consecutive
     multiples of 2^17: a rotation of 01011010 (c = 1) or 00001111 (c = 0), up to complement */
  for (uint32_t m8 = 1; m8 < 8; m8 += 2) {
    uint32_t inv = (m8 * m8) & 7, dd = (inv << 17) & 0xFFFFF;     /* m8^-1 = m8 mod 8 */
    for (uint32_t cc = 1; cc < 4; cc++) {
      int alive = 1;
      for (uint32_t c0 = 0; c0 < (1u << 17) && alive; c0++) {
        uint32_t pat = ((cc >> (c0 & 1)) & 1) ? 0x5Au : 0x0Fu, seq = 0;   /* bits k = 0..7 */
        for (uint32_t k = 0; k < 8; k++) seq |= y19((c0 + k * dd) & 0xFFFFF) << k;
        int match = 0;
        for (uint32_t r = 0; r < 8 && !match; r++) {
          uint32_t rot = ((pat >> r) | (pat << (8 - r))) & 0xFF;
          match = seq == rot || seq == (rot ^ 0xFF);
        }
        alive = match;
      }
      if (alive) { printf("p17 %%u %%u\n", m8, cc); found++; }
    }
  }
  printf("done %%ld\n", found);
  return 0;
}
"""


# Window checks for templates that end in a final right shift by 1 or 2 (rule final-window of
# grammar4_round0_pinned_audit863), with c = u mod 8: E1 bit19(L) = y19 ^ g(c); E2 the same
# with dep_c * bit18(L) added; E3 bit19(L) ^ G_c(bit17(L), bit18(L)) = y19.
WIN_C = r"""/* Window checks for templates ending in a final right shift by 1 or 2 (class e).
   c = u mod 8.  E1: bit19(mu u + nu) == y19(u) ^ g(c).
   E2: bit19(V) ^ dep_c * bit18(V) == y19 ^ g(c), V = mu u + nu (bit19 ^ bit18 = bit19(V + 2^18)).
   E3: bit19(V) ^ G_c(bit17(V), bit18(V)) == y19: with step d = 2^17 / mu (mod 2^20) V moves by
       exactly 2^17, so along u0 + k d (k < 8) the left side is a rotation of a fixed 8-bit pattern. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define N (1u << 20)
#define MSK (N - 1)
#define H (1u << 19)
#define C0 %(C0)uu
static uint32_t R;
static inline uint32_t y19(uint32_t u) { return ((4097u * (u ^ R) + C0) >> 19) & 1; }
/* arc intersection over the class c: nu with bit19(mu u + nu + shift) == y19(u) for u = c mod 8 */
static int arc_class(uint32_t mu, uint32_t c, uint32_t shift, uint32_t *lo_out, uint32_t *len_out) {
  uint32_t lo = 0, len = N;
  for (uint32_t u = c; u < N && len; u += 8) {
    uint32_t a = (y19(u) * H - mu * u - shift) & MSK;
    if (len == N) { lo = a; len = H; continue; }
    uint32_t d = (a - lo) & MSK, e = (lo - a) & MSK;
    if (d < len) { len -= d; lo = a; if (len > H) len = H; }
    else if (e < H) { uint32_t r = H - e; if (r < len) len = r; }
    else len = 0;
  }
  *lo_out = lo; *len_out = len;
  return len != 0;
}
int main(int argc, char **argv) {
  R = (uint32_t)strtoul(argv[1], 0, 0);
  long e1 = 0, e2 = 0;
  /* E1 and E2 together: for each mu, each class c, the feasible nu mod 2^19 is the union over
     dep in {0,1} of (arc for shift dep * 2^18) mod 2^19 (g_c = 1 adds 2^19, invisible mod 2^19).
     A class with no arc for either dep kills mu. Survivors are reported for a full check. */
  for (uint32_t mu = 1; mu < N; mu += 2) {
    int alive1 = 1, alive2 = 1;
    for (uint32_t c = 0; c < 8 && (alive1 || alive2); c++) {
      uint32_t lo, len, lo2, len2;
      int a0 = arc_class(mu, c, 0, &lo, &len);
      int a1 = arc_class(mu, c, H >> 1, &lo2, &len2);
      if (!a0) alive1 = 0;
      if (!a0 && !a1) alive2 = 0;
    }
    if (alive1) { printf("e1 %%u\n", mu); e1++; }
    if (alive2) { printf("e2 %%u\n", mu); e2++; }
  }
  /* E3 */
  long e3 = 0;
  for (uint32_t m8 = 1; m8 < 8; m8 += 2) {
    uint32_t dd = ((m8 & 7) << 17) & MSK;          /* m8^-1 = m8 mod 8 */
    int ok_all = 1;
    for (uint32_t c = 0; c < 8 && ok_all; c++) {
      uint32_t aliveG = 0xFFFF;
      for (uint32_t G = 0; G < 16; G++) {
        uint32_t pat = 0;
        for (uint32_t W = 0; W < 8; W++) {
          uint32_t b17 = W & 1, b18 = (W >> 1) & 1, b19 = W >> 2;
          pat |= (b19 ^ ((G >> (b17 | (b18 << 1))) & 1)) << W;
        }
        for (uint32_t u0 = c; u0 < (1u << 17) && ((aliveG >> G) & 1); u0 += 8) {
          uint32_t seq = 0;
          for (uint32_t k = 0; k < 8; k++) seq |= y19((u0 + k * dd) & MSK) << k;
          int match = 0;
          for (uint32_t r = 0; r < 8 && !match; r++)
            match = seq == (((pat >> r) | (pat << (8 - r))) & 0xFF);
          if (!match) aliveG &= ~(1u << G);
        }
      }
      if (!aliveG) ok_all = 0;
    }
    if (ok_all) { printf("e3 %%u\n", m8); e3++; }
  }
  printf("done %%ld %%ld %%ld\n", e1, e2, e3);
  return 0;
}
"""


# shift inputs whose bit-0 condition at s = 19 is decided here by exhaustion
TWO_CONST = {
    'xk_lin2': ([('xk', (0,)), ('lin2', (0, 1))], 'u + a*(u ^ b)'),
    'andk_lin2': ([('andk', (0,)), ('lin2', (0, 1))], 'u + a*(u & b)'),
    'ork_lin2': ([('ork', (0,)), ('lin2', (0, 1))], 'u + a*(u | b)'),
}


RIGID_LAST = {'lin1', 'lin2', 'prod', 'mad_c', 'mad_b', 'mad_a'}


def rigid_shape(p):
    """True if the single-shift template meets the conditions of rule 4."""
    sh = [k for k, (f, _) in enumerate(p) if f == 'shrk']
    if len(sh) != 1 or sh[0] != len(p) - 2:
        return False
    k = sh[0]
    f, args = p[-1]
    if f not in RIGID_LAST or k + 1 not in args:
        return False
    # kind of each value's move under a flip of input bit j: 'bit' (a single bit
    # flip or nothing), 'rigid', or None
    kind = ['bit']
    for g_, a in p[:k]:
        ks = [kind[i] for i in a]
        if g_ in ('xk', 'andk', 'ork', 'xor', 'and', 'or'):
            kind.append('bit' if all(x == 'bit' for x in ks) else None)
        elif g_ in ('aff', 'lin1', 'lin2', 'prod', 'mad_c', 'mad_b', 'mad_a'):
            kind.append('rigid' if all(x in ('bit', 'rigid') for x in ks) else None)
        else:
            kind.append(None)
    (src,) = p[k][1]
    return kind[src] in ('bit', 'rigid')


INF = 10 ** 6
_BITOPS = {'xk', 'andk', 'ork', 'xor', 'and', 'or'}


def rigid_moves(p, amounts, e, j):
    """Rule 5: True if the output's move mod 2^e under a flip of input bit j is fixed by
    (v mod 2^e, v_j) for every choice of constants, with shift amounts {instr: s}."""
    st = [dict(t=j, w=INF, vw=30, low=True, bit=True)]
    for k, (f, args) in enumerate(p):
        ops = [st[a] for a in args]
        if any(o is None for o in ops):
            st.append(None)
            continue
        if f == 'shrk':
            sh, (a,) = amounts[k], ops
            if sh >= a['vw']:
                st.append(dict(t=INF, w=INF, vw=0, low=True, bit=True))
            elif sh == 0:
                st.append(a)
            elif a['t'] >= sh and a['w'] >= sh:
                st.append(dict(t=a['t'] - sh, w=a['w'] - sh, vw=a['vw'] - sh, low=False, bit=False))
            else:
                st.append(None)
            continue
        low = all(o['low'] for o in ops)
        tmin = min(o['t'] for o in ops)
        wmin = min(min(o['w'] for o in ops), 32)
        if f in _BITOPS:
            if all(o['bit'] for o in ops):
                st.append(dict(t=j, w=INF, vw=32, low=low, bit=True))
            else:
                st.append(dict(t=tmin, w=min(tmin, wmin), vw=32, low=low, bit=False))
            continue
        if f in ('aff', 'lin1', 'lin2'):
            st.append(dict(t=tmin, w=wmin, vw=32, low=low, bit=False))
            continue
        if f == 'prod':
            (x, y), add = ops, []
        else:
            a, b, c = ops
            (x, y), add = {'mad_c': ((a, b), [c]), 'mad_b': ((a, c), [b]), 'mad_a': ((b, c), [a])}[f]
        # D(xy) = x Dy + y' Dx: a term is known when its coefficient is a T-function
        terms = [(mv['t'], mv['w'] if coef['low'] else mv['t']) for coef, mv in ((x, y), (y, x))]
        terms += [(o['t'], o['w']) for o in add]
        st.append(dict(t=min(t for t, _ in terms), w=min(min(w for _, w in terms), 32),
                       vw=32, low=low, bit=False))
    return st[-1] is not None and st[-1]['w'] >= e


@functools.lru_cache(maxsize=None)
def move_witness(e, j, root):
    """(v1, v2) with v1 = v2 mod 2^e, the same bit j, and target moves that differ mod 2^e."""
    import random
    m = 1 << e

    def move(v):
        x = v ^ root
        return (g.target(x ^ (1 << j)) - g.target(x)) & g.MASK
    rng = random.Random(e * 1000 + j)
    for _ in range(3000):
        v1 = rng.randrange(g.D30)
        v2 = (v1 & (m - 1)) | (v1 & (1 << j)) | (rng.randrange(g.D30) & ~(m - 1) & ~(1 << j))
        if (move(v1) - move(v2)) % m:
            return v1, v2
    return None


def rigid_any(p, amounts, root):
    """First (e, j) with a witness for which rule 5 holds, else None."""
    for e in range(13, 0, -1):
        for j in range(20, 30):
            if move_witness(e, j, root) and rigid_moves(p, amounts, e, j):
                return e, j
    return None


@functools.lru_cache(maxsize=None)
def rigid_witnesses(root):
    """s -> (j, v1, v2): v1 = v2 mod 2^(32-s), same bit j, target moves differ mod 2^(32-s)."""
    import random
    def move(v, j):
        x = v ^ root
        return (g.target(x ^ (1 << j)) - g.target(x)) & g.MASK
    out = {}
    for s in range(19, 32):
        m = 1 << (32 - s)
        for j in range(29, max(s, 20) - 1, -1):
            rng = random.Random(s * 100 + j)
            for _ in range(4000):
                v1 = rng.randrange(g.D30)
                v2 = (v1 & (m - 1)) | (v1 & (1 << j)) | (rng.randrange(g.D30) & ~(m - 1) & ~(1 << j))
                if (move(v1, j) - move(v2, j)) % m:
                    out[s] = (j, v1, v2)
                    break
            if s in out:
                break
    return out


def _build(src, name, d):
    c, exe = Path(d) / f'{name}.c', Path(d) / name
    c.write_text(src)
    subprocess.run(['clang', '-O3', '-o', str(exe), str(c)], check=True)
    return exe


@functools.lru_cache(maxsize=None)
def xk_affine_keys():
    """Residues e (mod 2^19, bit 0 clear) for which pi(X0 ^ e) is a progression."""
    with tempfile.TemporaryDirectory() as d:
        out = subprocess.run([str(_build(AP_C % dict(C0=g.C0), 'ap', d))], check=True,
                             capture_output=True, text=True).stdout.split('\n')
    assert out[-2] == 'done' and out[0].startswith('size ')
    return tuple(int(l.split()[1]) for l in out if l.startswith('key '))


@functools.lru_cache(maxsize=None)
def xk_affine_solutions(root):
    """All (K mod 2^20, mu', nu') with bit19(mu'(u^K) + nu') == y19(u) ^ g(u0) for every u."""
    sols = []
    for ep in xk_affine_keys():
        for lift in (0, 1, 1 << 19, (1 << 19) | 1):
            e = ep | lift
            for mu, nu, _ in h.affine_refuted(e):
                sols.append(((root ^ e) & 0xFFFFF, mu, nu))
    return tuple(sorted(set(sols)))


def two_const_solutions(expr, root, workers=8):
    """All (a, b, g0, g1) mod 2^20 with bit19(W(u;a,b)) == y19(u) ^ g(u0) for every u < 2^20."""
    with tempfile.TemporaryDirectory() as d:
        exe = _build(BF2_C % dict(C0=g.C0, EXPR=expr), 'bf2', d)
        run = lambda k: subprocess.run([str(exe), str(root), str(k), str(workers)], check=True,
                                       capture_output=True, text=True).stdout.split('\n')
        with ThreadPoolExecutor(workers) as pool:
            outs = list(pool.map(run, range(workers)))
    sols = []
    for out in outs:
        assert out[-2].startswith('done'), out[-3:]
        sols += [tuple(map(int, l.split()[1:])) for l in out if l.startswith('sol ')]
    return sorted(sols)


def quad_solutions(root, workers=8):
    """All (a, b, g1, K arc start, arc length) for prod(u, aff(u)) at s = 19."""
    with tempfile.TemporaryDirectory() as d:
        exe = _build(QUAD_C % dict(C0=g.C0), 'quad', d)
        run = lambda k: subprocess.run([str(exe), str(root), str(k), str(workers)], check=True,
                                       capture_output=True, text=True).stdout.split('\n')
        with ThreadPoolExecutor(workers) as pool:
            outs = list(pool.map(run, range(workers)))
    sols = []
    for out in outs:
        assert out[-2].startswith('done'), out[-3:]
        sols += [tuple(map(int, l.split()[1:])) for l in out if l.startswith('sol ')]
    return sorted(sols)


@functools.lru_cache(maxsize=None)
def pair_read_solutions(root):
    """Solutions of the G1 and P18 conditions (both empty for the default root)."""
    with tempfile.TemporaryDirectory() as d:
        out = subprocess.run([str(_build(PAIR_C % dict(C0=g.C0), 'pair', d)), str(root)], check=True,
                             capture_output=True, text=True).stdout.split('\n')
    assert out[-2] == 'done', out[-3:]
    return tuple(tuple(map(int, l.split()[1:])) for l in out if l.startswith('sol '))


@functools.lru_cache(maxsize=None)
def sub_affine_solutions(root):
    """(p, mu, const, H, len) for which the G2 subsequence condition holds."""
    with tempfile.TemporaryDirectory() as d:
        out = subprocess.run([str(_build(SUB_C % dict(C0=g.C0), 'sub', d)), str(root)], check=True,
                             capture_output=True, text=True).stdout.split('\n')
    assert out[-2].startswith('done'), out[-3:]
    sols = tuple(tuple(map(int, l.split()[1:])) for l in out if l.startswith('sol '))
    common = {int(l.split()[1]): int(l.split()[2]) for l in out if l.startswith('common ')}
    sol18 = tuple(tuple(map(int, l.split()[1:])) for l in out if l.startswith('sol18 '))
    p17 = tuple(tuple(map(int, l.split()[1:])) for l in out if l.startswith('p17 '))
    return sols, common, sol18, p17


def g2_refuted(root):
    """True if G2 has no solution for any p = 0..18 (rule two-read-L)."""
    sols, common, sol18, p17 = sub_affine_solutions(root)
    return all(p >= 16 for p, *_ in sols) and common[16] == 0 and p17 == () and sol18 == ()


def bit_b_solutions(expr, bit, root, workers=8):
    """All (a odd, b with bit `bit` clear, g0, g1) with bit_bit(W(u)) == y19(u) ^ g(u0), u < 2^(bit+1)."""
    with tempfile.TemporaryDirectory() as d:
        exe = _build(BFW_C % dict(C0=g.C0, EXPR=expr, B=bit), 'bfw', d)
        run = lambda k: subprocess.run([str(exe), str(root), str(k), str(workers)], check=True,
                                       capture_output=True, text=True).stdout.split('\n')
        with ThreadPoolExecutor(workers) as pool:
            outs = list(pool.map(run, range(workers)))
    sols = []
    for out in outs:
        assert out[-2].startswith('done'), out[-3:]
        sols += [tuple(map(int, l.split()[1:])) for l in out if l.startswith('sol ')]
    return sorted(sols)


@functools.lru_cache(maxsize=None)
def window_checks(root):
    """(e1 survivors, e2 survivors, e3 surviving mu mod 8); all empty means refuted."""
    with tempfile.TemporaryDirectory() as d:
        out = subprocess.run([str(_build(WIN_C % dict(C0=g.C0), 'win', d)), str(root)], check=True,
                             capture_output=True, text=True).stdout.split('\n')
    assert out[-2].startswith('done'), out[-3:]
    pick = lambda tag: tuple(int(l.split()[1]) for l in out if l.startswith(tag + ' '))
    return pick('e1'), pick('e2'), pick('e3')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--root', type=lambda s: int(s, 0), default=h.ROOTS[0])
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--only', default='', help='comma list of TWO_CONST names (default all)')
    args = ap.parse_args()
    res = json.loads(args.output.read_text()) if args.output.exists() else {}
    res['xk_affine_keys'] = list(xk_affine_keys())
    res['xk_aff'] = [list(s) for s in xk_affine_solutions(args.root)]
    print(json.dumps(dict(xk_affine_keys=res['xk_affine_keys'], xk_aff=len(res['xk_aff']))), flush=True)
    for name in (args.only.split(',') if args.only else [*TWO_CONST, 'aff_prod', 'ork_lin2_b20']):
        if name == 'aff_prod':
            sols = quad_solutions(args.root, args.workers)
        elif name == 'ork_lin2_b20':
            sols = bit_b_solutions('u + a*(u | b)', 20, args.root, args.workers)
        else:
            sols = two_const_solutions(TWO_CONST[name][1], args.root, args.workers)
        res[name] = [list(s) for s in sols]
        print(json.dumps(dict(name=name, solutions=len(sols))), flush=True)
        args.output.write_text(json.dumps(res, indent=0) + '\n')
    args.output.write_text(json.dumps(res, indent=0) + '\n')


if __name__ == '__main__':
    main()
