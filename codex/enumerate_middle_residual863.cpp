// Exhaust odd inverse coefficients against necessary finite residual bounds.
// This is a research verifier, never part of the Python submission.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

struct Case { unsigned id, width; std::vector<uint32_t> residuals; };
struct Model { uint32_t a; unsigned high; };

void solve(const Case& c, std::mutex& output) {
    auto begin = std::chrono::steady_clock::now();
    uint64_t limit = (uint64_t(1) << c.width)-1;
    uint32_t mask = uint32_t(limit), half = uint32_t(uint64_t(1) << (c.width-1));
    std::vector<Model> models;
    uint64_t evaluated = 0;
    bool complete = true;
    for (uint64_t base = 1; base <= limit; base += 8) {
        uint32_t counts[4] = {0, 0, 0, 0};
        size_t prefix = std::min(size_t(16), c.residuals.size());
#if defined(__aarch64__)
        const uint32_t av[4] = {uint32_t(base), uint32_t(base+2), uint32_t(base+4), uint32_t(base+6)};
        uint32x4_t a = vld1q_u32(av), count = vdupq_n_u32(0);
        for (size_t i = 0; i < prefix; ++i) {
            uint32x4_t q = vandq_u32(vmulq_n_u32(a, c.residuals[i]), vdupq_n_u32(mask));
            count = vsubq_u32(count, vcgtq_u32(q, vdupq_n_u32(half)));
            if (vminvq_u32(count) >= 2) break;
        }
        vst1q_u32(counts, count);
#else
        for (unsigned lane = 0; lane < 4; ++lane) {
            for (size_t i = 0; i < prefix; ++i) {
                uint32_t q = uint32_t((base+2*lane)*c.residuals[i]) & mask;
                counts[lane] += q > half;
                if (counts[lane] >= 2) break;
            }
        }
#endif
        for (unsigned lane = 0; lane < 4; ++lane) {
            unsigned high = counts[lane];
            uint32_t a = uint32_t(base+2*lane);
            if (high < 2) {
                for (size_t i = prefix; i < c.residuals.size(); ++i) {
                    uint32_t q = uint32_t(uint64_t(a)*c.residuals[i]) & mask;
                    high += q > half;
                    if (high >= 2) break;
                }
                if (high < 2) models.push_back({a, high});
            }
            ++evaluated;
            if (models.size() >= 50000) { complete = false; break; }
        }
        if (!complete) break;
    }
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    std::lock_guard<std::mutex> lock(output);
    std::cout << "{\"id\":" << c.id << ",\"width\":" << c.width
              << ",\"complete\":" << (complete ? "true" : "false")
              << ",\"evaluated_odd_parameters\":" << evaluated
              << ",\"seconds\":" << seconds << ",\"models\":[";
    for (size_t i = 0; i < models.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << "[" << models[i].a << ',' << models[i].high << ']';
    }
    std::cout << "]}" << std::endl;
}

int main(int argc, char** argv) {
    unsigned jobs = argc > 1 ? unsigned(std::stoul(argv[1])) : 4;
    size_t n;
    if (!(std::cin >> n)) return 2;
    std::vector<Case> cases(n);
    for (auto& c : cases) {
        size_t samples;
        if (!(std::cin >> c.id >> c.width >> samples) || c.width < 3 || c.width > 32) return 2;
        c.residuals.resize(samples);
        uint64_t mask = (uint64_t(1) << c.width)-1;
        for (auto& r : c.residuals) if (!(std::cin >> r) || r > mask) return 2;
        // Input residuals must be distinct: the one-high condition counts
        // different quotient values, not repeated observations.
        auto sorted = c.residuals;
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) return 2;
    }
    std::atomic<size_t> next{0};
    std::mutex output;
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < std::max(1u, std::min(jobs, unsigned(n))); ++i) {
        threads.emplace_back([&] {
            while (true) {
                size_t index = next.fetch_add(1);
                if (index >= cases.size()) break;
                solve(cases[index], output);
            }
        });
    }
    for (auto& t : threads) t.join();
    return 0;
}
