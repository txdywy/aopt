/* Lemma A of grammar4_head_c5_863: count a' odd, b' (mod 2^20) and g: {0,1} -> {0,1} with

       bit19(a' * u + b') == y19(u ^ KEY) ^ g(u & 1)   for every u < 2^20,
       y19(x) = bit 19 of 4097 * x + C0.

   For fixed a' and g, one u allows b' only in a half-circle arc of Z/2^20. Arcs of
   length at most 2^19 intersect in one arc, so the arcs of 4096 inputs are intersected
   exactly; every b' left in the arc is then checked on all 2^20 inputs.
   Usage: lemma KEY   (prints survivors after the sample arcs and full solutions) */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define N (1u << 20)
#define H (1u << 19)

static uint8_t ty[N];

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  uint32_t key = (uint32_t)strtoul(argv[1], 0, 0);
  long survivors = 0, solutions = 0;
  for (uint32_t u = 0; u < N; u++) ty[u] = (((4097u * (u ^ key) + 0x7ED55D16u) & (N - 1)) >> 19) & 1;
  for (uint32_t a = 1; a < N; a += 2)
    for (int g = 0; g < 4; g++) {
      /* u = 0 gives the starting arc [lo, lo + H) */
      uint32_t lo = (ty[0] ^ (g & 1)) ? H : 0, len = H;
      int ok = 1;
      for (uint32_t i = 1, u = 12345u; i < 4096 && ok; i++, u = (u * 2654435761u + 12345u) & (N - 1)) {
        int t = ty[u] ^ ((g >> (u & 1)) & 1);
        uint32_t c = (a * u) & (N - 1);
        uint32_t st = (t ? H - c : N - c) & (N - 1); /* b' in [st, st + H) */
        uint32_t off = (st - lo) & (N - 1), back = (lo - st) & (N - 1);
        if (off < len) { /* the new arc starts inside the current one */
          len = len - off < H ? len - off : H;
          lo = st;
        } else if (back < H) { /* the current arc starts inside the new one */
          len = H - back < len ? H - back : len;
        } else {
          ok = 0;
        }
        if (!len) ok = 0;
      }
      if (!ok) continue;
      survivors++;
      for (uint32_t k = 0; k < len; k++) {
        uint32_t b = (lo + k) & (N - 1);
        int good = 1;
        for (uint32_t u = 0; u < N && good; u++)
          if (((((a * u + b) & (N - 1)) >> 19) & 1) != (ty[u] ^ ((g >> (u & 1)) & 1))) good = 0;
        solutions += good;
      }
    }
  printf("key=%#x survivors=%ld solutions=%ld\n", key, survivors, solutions);
  return 0;
}
