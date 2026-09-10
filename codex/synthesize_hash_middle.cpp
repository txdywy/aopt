// Bit-lifting synthesis for (A*x+B) XOR (D*x+F), all arithmetic uint32.
// At x=0, F = target(0) XOR B. Simultaneously complementing both affine
// expressions preserves XOR, so B bit 0 may be fixed to zero without loss.
// An empty frontier disproves this entire template using the tested inputs.
// A frontier limit is inconclusive; surviving full-width programs need proof.
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

struct State { uint32_t a, b, d; };
uint32_t target(uint32_t x, bool control, bool nested, bool input_xor, bool absorb_c1 = false) {
    if (absorb_c1) x ^= 0xC761C23Cu;
    uint32_t u = 33u*x + 0xE9F8CC1Du;
    uint32_t v = 16896u*x + 0xACCF6200u;
    uint32_t c = u ^ v;
    if (absorb_c1) return c;
    if (control && input_xor) return (33u*(x ^ 0xE9F8CC1Du)+0xACCF6200u)^0xFD7046C5u;
    return control ? (nested ? 9u*(u ^ x) + 0xFD7046C5u : c) : 9u*c + 0xFD7046C5u;
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "xor";
    const bool control = mode == "control" || mode == "nested-control" || mode == "input-xor-control";
    const bool nested = mode == "nested" || mode == "nested-control" || mode == "absorb-c1-nested";
    const bool input_xor = mode == "input-xor" || mode == "input-xor-control" || mode == "absorb-c1-input-xor";
    const bool absorb_c1 = mode == "absorb-c1" || mode == "absorb-c1-nested" || mode == "absorb-c1-input-xor";
    const size_t cap = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 2000000;
    std::vector<uint32_t> xs;
    for (uint32_t i = 0; i < 2048; ++i) xs.push_back(i);
    std::mt19937 rng(900);
    for (int i = 0; i < 128; ++i) xs.push_back(rng());
    std::vector<uint32_t> ys;
    for (auto x : xs) ys.push_back(target(x, control, nested, input_xor, absorb_c1));
    std::vector<State> states{{0, 0, 0}};
    const uint32_t y0 = ys[0];
    for (unsigned bit = 0; bit < 32; ++bit) {
        std::vector<State> next;
        const uint32_t flag = uint32_t(1) << bit;
        for (const auto& old : states) {
            for (unsigned choice = 0; choice < 8; ++choice) {
                if (bit == 0 && (choice & 2)) continue;
                State s{old.a | ((choice & 1) ? flag : 0),
                        old.b | ((choice & 2) ? flag : 0),
                        old.d | ((choice & 4) ? flag : 0)};
                // Nested template: D*((A*x+B) XOR x)+F. The x=0
                // constraint infers F. Complementing the inner affine,
                // negating D and subtracting D from F again normalizes B0.
                // Input-XOR template: (A*(x XOR B)+D) XOR F.
                // B -> ~B, A -> -A, D -> D-A preserves the inner affine.
                const uint32_t f = input_xor ? y0 ^ (s.a*s.b+s.d)
                                             : nested ? y0 - s.d*s.b : y0 ^ s.b;
                bool valid = true;
                for (size_t i = 1; i < xs.size(); ++i) {
                    uint32_t got = input_xor ? (s.a*(xs[i]^s.b)+s.d)^f
                                          : nested ? s.d*((s.a*xs[i] + s.b) ^ xs[i])+f
                                          : (s.a*xs[i] + s.b) ^ (s.d*xs[i] + f);
                    if ((got ^ ys[i]) & flag) { valid = false; break; }
                }
                if (valid) {
                    next.push_back(s);
                    if (next.size() > cap) {
                        std::cout << "frontier_limit bit=" << bit << " cap=" << cap
                                  << " result=inconclusive\n";
                        return 2;
                    }
                }
            }
        }
        std::cout << "bits=" << bit+1 << " survivors=" << next.size() << std::endl;
        if (next.empty()) {
            std::cout << "result=template_unsatisfiable_on_samples control=" << control << "\n";
            return control ? 3 : 0;
        }
        if (control) {
            // The known solution must remain after symmetry normalization.
            uint32_t mask = bit == 31 ? UINT32_MAX : (flag << 1) - 1;
            State known{uint32_t(-33) & mask, ~uint32_t(0xE9F8CC1D) & mask,
                        (input_xor ? 0xACCF6200u-33u : nested ? uint32_t(-9) : uint32_t(-16896)) & mask};
            bool found = false;
            for (const auto& s : next)
                if (s.a == known.a && s.b == known.b && s.d == known.d) found = true;
            if (!found) { std::cerr << "positive control lost\n"; return 4; }
        }
        states.swap(next);
    }
    std::cout << "result=survivors_need_full_equivalence_proof count=" << states.size() << "\n";
    for (size_t i = 0; i < states.size() && i < 8; ++i) {
        const auto& s = states[i];
        std::cout << std::hex << s.a << " " << s.b << " " << s.d << " "
                  << (input_xor ? y0 ^ (s.a*s.b+s.d) : nested ? y0 - s.d*s.b : y0 ^ s.b) << "\n";
    }
}
