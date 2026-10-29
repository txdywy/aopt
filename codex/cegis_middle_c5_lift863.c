/* Exact bit-lifting for the middle+C5 wirings (cegis_middle_c5_863.py).
 *
 * All values of these programs are T-functions of z, so a program matches the
 * target  F(z ^ K) ^ D  mod 2^m  iff its constants and K do mod 2^m. A depth-first
 * search fixes bit m-1 of every unknown at level m and keeps a branch only if the
 * program matches the target on all 2^m inputs mod 2^m. If no branch reaches the
 * top width, no constants and no K make the wiring work: a complete proof without
 * a solver.
 *
 * usage: lift WIDTH SPEC [PIN...]
 *   SPEC  instructions separated by ';', each "op:a,b[,c]" with op in m ^ & |
 *         and pool indices as in cegis_c1_absorb863 (z, t0.., k0, k1)
 *   PIN   "i.s=v" fixes constant slot s of instruction i to v (-1 = all ones);
 *         "D=v" replaces the output key (for controls)
 * prints: result (unsat/survivors), the level reached, nodes per level.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXI 4
#define MAXU 9

static const uint32_t K1 = 0xE9F8CC1Du, K2 = 0xACCF6200u, C4 = 0xFD7046C5u;
static uint32_t D = 0xB55AFA53u;   /* "D=v" on the command line overrides (controls) */

static int n;                   /* instructions */
static char op[MAXI];
static int arity[MAXI], arg[MAXI][3];
static int nu;                  /* unknowns: used constants then K */
static int uins[MAXU], uslot[MAXU];
static int pinned[MAXU];
static uint32_t pinval[MAXU];
static int slot_of[MAXI][2];    /* unknown index of (instr, slot) or -1 */
static int width;
static uint32_t ftab[1 << 16];  /* middle(y) for y < 2^width, taken mod 2^m */
static uint64_t nodes[33];
static uint32_t val[MAXU];
static uint32_t killer[33][4];

static uint32_t middle(uint32_t y) {
    return 9u * ((33u * y + K1) ^ (16896u * y + K2)) + C4;
}

static uint32_t run(uint32_t z, uint32_t mask) {
    uint32_t t[MAXI + 1];
    t[0] = z;
    for (int i = 0; i < n; i++) {
        uint32_t x[3];
        for (int j = 0; j < arity[i]; j++) {
            int p = arg[i][j];
            x[j] = p <= i ? t[p] : val[slot_of[i][p - i - 1]];
        }
        uint32_t v;
        switch (op[i]) {
        case 'm': v = x[0] * x[1] + x[2]; break;
        case '^': v = x[0] ^ x[1]; break;
        case '&': v = x[0] & x[1]; break;
        default: v = x[0] | x[1]; break;
        }
        t[i + 1] = v & mask;
    }
    return t[n];
}

static int matches(int m) {
    uint32_t mask = m == 32 ? 0xFFFFFFFFu : (1u << m) - 1, key = val[nu - 1];
    /* try the inputs that refuted recent branches first */
    for (int k = 0; k < 4; k++) {
        uint32_t y = killer[m][k] & mask;
        if (run(y, mask) != ((ftab[(y ^ key) & mask] ^ D) & mask)) return 0;
    }
    for (uint32_t y = 0; y <= mask; y++) {
        if (run(y, mask) != ((ftab[(y ^ key) & mask] ^ D) & mask)) {
            killer[m][3] = killer[m][2]; killer[m][2] = killer[m][1];
            killer[m][1] = killer[m][0]; killer[m][0] = y;
            return 0;
        }
    }
    return 1;
}

static int best;

static int search(int m) {      /* bits below m-1 are set; choose bit m-1 */
    if (m > width) return 1;
    int frees[MAXU], nf = 0;
    uint32_t bit = 1u << (m - 1);
    for (int u = 0; u < nu; u++) {
        if (pinned[u]) val[u] = (val[u] & ~bit) | (pinval[u] & bit);
        else frees[nf++] = u;
    }
    for (uint32_t e = 0; e < (1u << nf); e++) {
        for (int f = 0; f < nf; f++)
            val[frees[f]] = (val[frees[f]] & ~bit) | (((e >> f) & 1) ? bit : 0);
        nodes[m]++;
        if (!matches(m)) continue;
        if (m > best) best = m;
        if (search(m + 1)) return 1;
    }
    for (int f = 0; f < nf; f++) val[frees[f]] &= ~bit;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: lift WIDTH SPEC [PIN...]\n"); return 2; }
    width = atoi(argv[1]);
    char *spec = strdup(argv[2]), *save = NULL;
    memset(slot_of, -1, sizeof slot_of);
    for (char *ins = strtok_r(spec, ";", &save); ins; ins = strtok_r(NULL, ";", &save)) {
        op[n] = ins[0];
        arity[n] = 0;
        for (char *p = strchr(ins, ':') + 1; *p; ) {
            arg[n][arity[n]++] = (int)strtol(p, &p, 10);
            if (*p == ',') p++;
        }
        for (int j = 0; j < arity[n]; j++) {
            int p = arg[n][j];
            if (p > n && slot_of[n][p - n - 1] < 0) {
                slot_of[n][p - n - 1] = nu;
                uins[nu] = n; uslot[nu] = p - n - 1; nu++;
            }
        }
        n++;
    }
    nu++;                       /* K */
    for (int a = 3; a < argc; a++) {
        int i, s; long long v;
        if (sscanf(argv[a], "D=%lld", &v) == 1) { D = (uint32_t)v; continue; }
        if (sscanf(argv[a], "%d.%d=%lld", &i, &s, &v) == 3 && slot_of[i][s] >= 0) {
            pinned[slot_of[i][s]] = 1;
            pinval[slot_of[i][s]] = (uint32_t)v;
        }
    }
    for (uint32_t y = 0; y < (1u << width); y++) ftab[y] = middle(y);
    int ok = search(1);
    printf("%s level=%d nodes=", ok ? "survivors" : "unsat", ok ? width : best + 1);
    for (int m = 1; m <= width; m++) printf("%s%llu", m > 1 ? "," : "", (unsigned long long)nodes[m]);
    printf("\n");
    return 0;
}
