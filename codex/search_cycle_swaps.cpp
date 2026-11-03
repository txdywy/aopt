#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

struct Edge {
    int parent;
    int child;
    int lag;
    bool cross;
};

struct Score {
    int excess;
    std::int64_t total;
    int count;
    bool operator<(const Score& other) const {
        return std::tie(excess, total, count) <
               std::tie(other.excess, other.total, other.count);
    }
};

// A move is evaluated as an atomic state transition.  In particular, a
// cycle rotation can temporarily violate a precedence relation halfway
// through the rotation even though its final state is legal, so never check
// or commit its members one at a time.
struct Move {
    int node;
    int target;
};

int main(int argc, char** argv) {
    if (argc < 9) {
        std::cerr << "usage: search_cycle_swaps problem output-prefix "
                     "iterations radius seed max-uphill plateau-chance cap "
                     "[minimum-horizon]\n";
        return 2;
    }
    const std::string problem_path = argv[1];
    const std::string output_prefix = argv[2];
    const int iterations = std::stoi(argv[3]);
    const int radius = std::max(1, std::stoi(argv[4]));
    const unsigned seed = static_cast<unsigned>(std::stoul(argv[5]));
    const int max_uphill = std::max(0, std::stoi(argv[6]));
    const int plateau_chance = std::max(1, std::stoi(argv[7]));
    const int cap = std::stoi(argv[8]);
    const int minimum_horizon = argc >= 10 ? std::stoi(argv[9]) : 0;
    // Keep the optimization cap independent from the admissibility guard.
    // A seed assembled from independently optimal engine calendars can have
    // a large cross-engine lag while still being a perfectly valid resource
    // assignment.  Rejecting it before the first ejection move prevented the
    // search from repairing precisely those seeds.  The guard is now only a
    // generous sanity bound; ``cap`` remains the score threshold.
    const int hard_max = std::max(cap + 1, 4096);

    std::ifstream input(problem_path);
    int n, m, engine_count;
    input >> n >> m >> engine_count;
    std::vector<int> original(n), engine_of(n), earliest(n), tail(n), cycles(n);
    int horizon = 0;
    for (int i = 0; i < n; ++i) {
        input >> original[i] >> engine_of[i] >> earliest[i] >> tail[i] >> cycles[i];
        horizon = std::max(horizon, cycles[i] + 1);
    }
    horizon = std::max(horizon, minimum_horizon);
    std::vector<Edge> edges;
    edges.reserve(m);
    std::vector<std::vector<int>> incident(n);
    for (int edge_index = 0; edge_index < m; ++edge_index) {
        int parent, child, lag;
        input >> parent >> child >> lag;
        edges.push_back(
            {parent, child, lag, engine_of[parent] != engine_of[child]}
        );
        incident[parent].push_back(edge_index);
        incident[child].push_back(edge_index);
    }
    std::vector<std::string> engine_names(engine_count);
    std::vector<int> capacities(engine_count);
    for (int engine = 0; engine < engine_count; ++engine) {
        int count;
        input >> engine_names[engine] >> capacities[engine] >> count;
        int ignored;
        for (int i = 0; i < count; ++i) input >> ignored;
    }
    if (!input) {
        std::cerr << "invalid problem input\n";
        return 2;
    }

    std::vector<std::vector<std::vector<int>>> occupancy(engine_count);
    for (int engine = 0; engine < engine_count; ++engine) {
        occupancy[engine].resize(horizon);
    }
    for (int node = 0; node < n; ++node) {
        occupancy[engine_of[node]][cycles[node]].push_back(node);
    }
    for (int engine = 0; engine < engine_count; ++engine) {
        for (int cycle = 0; cycle < horizon; ++cycle) {
            if (static_cast<int>(occupancy[engine][cycle].size()) >
                capacities[engine]) {
                std::cerr << "initial resource overflow\n";
                return 2;
            }
        }
    }

    auto edge_amount = [&](const Edge& edge) {
        return cycles[edge.parent] + edge.lag - cycles[edge.child];
    };
    auto score_all = [&]() {
        Score score{0, 0, 0};
        for (const Edge& edge : edges) {
            int amount = edge_amount(edge);
            if (!edge.cross) {
                if (amount > 0) return Score{std::numeric_limits<int>::max(), 0, 0};
                continue;
            }
            if (amount > hard_max) {
                return Score{std::numeric_limits<int>::max(), 0, 0};
            }
            if (amount > 0) {
                score.total += amount;
                ++score.count;
            }
            if (amount > cap) score.excess += amount - cap;
        }
        return score;
    };

    Score current = score_all();
    if (current.excess == std::numeric_limits<int>::max()) {
        std::cerr << "initial schedule violates same-engine edges or hard max\n";
        return 2;
    }
    Score best = current;
    std::vector<int> best_cycles = cycles;
    std::vector<int> critical_edges;
    auto rebuild_critical = [&]() {
        critical_edges.clear();
        for (int edge_index = 0; edge_index < m; ++edge_index) {
            const Edge& edge = edges[edge_index];
            if (edge.cross && edge_amount(edge) > cap) {
                critical_edges.push_back(edge_index);
            }
        }
    };
    rebuild_critical();
    std::cout << "start excess=" << current.excess << " total=" << current.total
              << " count=" << current.count << " critical="
              << critical_edges.size() << std::endl;

    std::mt19937 rng(seed);
    std::vector<int> marks(m, 0), affected;
    int generation = 0;
    int stall = 0;
    int rotation_cooldown = 0;

    auto erase_node = [](std::vector<int>& values, int node) {
        auto found = std::find(values.begin(), values.end(), node);
        if (found == values.end()) std::abort();
        *found = values.back();
        values.pop_back();
    };

    // Score a simultaneous set of moves and leave ``cycles`` unchanged.
    // This is deliberately more general than a swap: a rotation/ejection
    // chain must be validated only after every endpoint has moved.
    auto evaluate_moves = [&](const std::vector<Move>& moves, Score* trial) {
        if (moves.empty()) return false;
        std::vector<int> old_cycles;
        old_cycles.reserve(moves.size());
        for (std::size_t left = 0; left < moves.size(); ++left) {
            const Move& move = moves[left];
            if (move.node < 0 || move.node >= n || move.target < 0 ||
                move.target >= horizon || move.target == cycles[move.node] ||
                move.target < earliest[move.node] ||
                move.target + tail[move.node] >= horizon) {
                return false;
            }
            for (std::size_t right = 0; right < left; ++right) {
                if (moves[right].node == move.node) return false;
            }
            old_cycles.push_back(cycles[move.node]);
        }

        // Check the final calendar occupancy, accounting for all removals
        // before all insertions.  This also permits a two-node ejection into
        // a genuine slack slot near the two horizon boundaries.
        for (std::size_t pivot = 0; pivot < moves.size(); ++pivot) {
            for (int cycle : {old_cycles[pivot], moves[pivot].target}) {
                const int engine = engine_of[moves[pivot].node];
                int used = static_cast<int>(occupancy[engine][cycle].size());
                for (std::size_t index = 0; index < moves.size(); ++index) {
                    if (engine_of[moves[index].node] != engine) continue;
                    if (old_cycles[index] == cycle) --used;
                    if (moves[index].target == cycle) ++used;
                }
                if (used > capacities[engine]) return false;
            }
        }

        ++generation;
        affected.clear();
        for (const Move& move : moves) {
            for (int edge_index : incident[move.node]) {
                if (marks[edge_index] == generation) continue;
                marks[edge_index] = generation;
                affected.push_back(edge_index);
            }
        }

        *trial = current;
        for (int edge_index : affected) {
            const Edge& edge = edges[edge_index];
            if (!edge.cross) continue;
            const int amount = edge_amount(edge);
            if (amount > 0) {
                trial->total -= amount;
                --trial->count;
            }
            if (amount > cap) trial->excess -= amount - cap;
        }

        for (const Move& move : moves) cycles[move.node] = move.target;
        bool valid = true;
        for (int edge_index : affected) {
            const Edge& edge = edges[edge_index];
            const int amount = edge_amount(edge);
            if (!edge.cross) {
                if (amount > 0) {
                    valid = false;
                    break;
                }
                continue;
            }
            if (amount > hard_max) {
                valid = false;
                break;
            }
            if (amount > 0) {
                trial->total += amount;
                ++trial->count;
            }
            if (amount > cap) trial->excess += amount - cap;
        }
        for (std::size_t index = 0; index < moves.size(); ++index) {
            cycles[moves[index].node] = old_cycles[index];
        }
        return valid;
    };

    auto commit_moves = [&](const std::vector<Move>& moves, const Score& score) {
        std::vector<int> old_cycles;
        old_cycles.reserve(moves.size());
        for (const Move& move : moves) old_cycles.push_back(cycles[move.node]);
        // Remove every source first.  This matters for a 3-cycle where one
        // target is another member's source.
        for (std::size_t index = 0; index < moves.size(); ++index) {
            erase_node(
                occupancy[engine_of[moves[index].node]][old_cycles[index]],
                moves[index].node
            );
        }
        for (const Move& move : moves) {
            cycles[move.node] = move.target;
            occupancy[engine_of[move.node]][move.target].push_back(move.node);
        }
        current = score;
        rebuild_critical();
    };

    auto remember_best = [&](int iteration, const char* kind,
                             int engine, int source_cycle, int target_cycle) {
        if (current < best) {
            const int previous_excess = best.excess;
            best = current;
            best_cycles = cycles;
            stall = 0;
            if ((best.excess < previous_excess &&
                 (best.excess < 20 || best.excess % 10 == 0)) ||
                iteration % 100000 == 0) {
                std::cout << "best excess=" << best.excess
                          << " total=" << best.total << " count=" << best.count
                          << " iteration=" << iteration << " engine="
                          << engine_names[engine] << " " << kind << "="
                          << source_cycle << ':' << target_cycle << std::endl;
            }
        } else {
            ++stall;
        }
    };

    for (int iteration = 0; iteration < iterations && best.excess > 0; ++iteration) {
        // A saturated VLIW calendar often has no improving two-operation
        // exchange.  Enumerate compact ejection chains rooted at every
        // current cap edge before falling back to the stochastic pair walk.
        // R=3 already exposes many strict descents at the max-26 plateau;
        // R=8 is enough for the earlier max-27 platform.
        if (!critical_edges.empty() && rotation_cooldown == 0) {
            Score best_rotation{std::numeric_limits<int>::max(), 0, 0};
            std::vector<Move> rotation_moves;
            const int rotation_radius = std::min(radius, 8);
            for (int edge_index : critical_edges) {
                const Edge& edge = edges[edge_index];
                const std::pair<int, int> seeds[] = {
                    {edge.parent, cycles[edge.parent] - 1},
                    {edge.child, cycles[edge.child] + 1},
                };
                for (const auto& seed_move : seeds) {
                    const int mover = seed_move.first;
                    const int source_cycle = cycles[mover];
                    const int target_cycle = seed_move.second;
                    if (target_cycle < 0 || target_cycle >= horizon ||
                        target_cycle < earliest[mover] ||
                        target_cycle + tail[mover] >= horizon) {
                        continue;
                    }
                    const int engine = engine_of[mover];
                    const auto& target_bucket = occupancy[engine][target_cycle];
                    if (target_bucket.empty()) {
                        Score trial;
                        const std::vector<Move> moves{{mover, target_cycle}};
                        if (evaluate_moves(moves, &trial) && trial < best_rotation) {
                            best_rotation = trial;
                            rotation_moves = moves;
                        }
                        continue;
                    }
                    for (int partner : target_bucket) {
                        std::vector<int> ejection_cycles;
                        for (int distance = 1;
                             distance <= rotation_radius; ++distance) {
                            for (int base : {source_cycle, target_cycle}) {
                                for (int sign : {-1, 1}) {
                                    const int candidate = base + sign * distance;
                                    if (candidate < 0 || candidate >= horizon ||
                                        candidate == source_cycle ||
                                        candidate == target_cycle ||
                                        std::find(
                                            ejection_cycles.begin(),
                                            ejection_cycles.end(), candidate
                                        ) != ejection_cycles.end()) {
                                        continue;
                                    }
                                    ejection_cycles.push_back(candidate);
                                }
                            }
                        }
                        for (int ejection_cycle : ejection_cycles) {
                            const auto& ejection_bucket =
                                occupancy[engine][ejection_cycle];
                            if (ejection_bucket.empty()) {
                                Score trial;
                                const std::vector<Move> moves{
                                    {mover, target_cycle},
                                    {partner, ejection_cycle},
                                };
                                if (evaluate_moves(moves, &trial) &&
                                    trial < best_rotation) {
                                    best_rotation = trial;
                                    rotation_moves = moves;
                                }
                                continue;
                            }
                            for (int third : ejection_bucket) {
                                if (third == mover || third == partner) continue;
                                Score trial;
                                const std::vector<Move> moves{
                                    {mover, target_cycle},
                                    {partner, ejection_cycle},
                                    {third, source_cycle},
                                };
                                if (evaluate_moves(moves, &trial) &&
                                    trial < best_rotation) {
                                    best_rotation = trial;
                                    rotation_moves = moves;
                                }
                            }
                        }
                    }
                }
            }
            if (!rotation_moves.empty() && best_rotation < current) {
                const int mover = rotation_moves.front().node;
                const int source_cycle = cycles[mover];
                const int target_cycle = rotation_moves.front().target;
                const int engine = engine_of[mover];
                commit_moves(rotation_moves, best_rotation);
                rotation_cooldown = 0;
                remember_best(
                    iteration, "rotation", engine, source_cycle, target_cycle
                );
                continue;
            }
            // Avoid re-enumerating the same plateau on every rejected random
            // pair proposal.  Any accepted pair resets this and exposes its
            // new local rotation neighborhood immediately.
            rotation_cooldown = 2048;
        }
        if (rotation_cooldown > 0) --rotation_cooldown;

        int mover;
        int target_cycle;
        if (!critical_edges.empty() && (rng() % 10U) != 0) {
            const Edge& anchor = edges[critical_edges[rng() % critical_edges.size()]];
            int distance = 1 + static_cast<int>(rng() % radius);
            if (rng() & 1U) {
                mover = anchor.parent;
                target_cycle = cycles[mover] - distance;
            } else {
                mover = anchor.child;
                target_cycle = cycles[mover] + distance;
            }
        } else {
            mover = rng() % n;
            int distance = 1 + static_cast<int>(rng() % radius);
            target_cycle = cycles[mover] + ((rng() & 1U) ? distance : -distance);
        }
        if (target_cycle < 0 || target_cycle >= horizon) continue;
        const int engine = engine_of[mover];
        const int source_cycle = cycles[mover];
        if (target_cycle == source_cycle) continue;
        auto& targets = occupancy[engine][target_cycle];
        int partner = -1;
        if (static_cast<int>(targets.size()) >= capacities[engine]) {
            partner = targets[rng() % targets.size()];
            if (partner == mover) continue;
        }
        std::vector<Move> moves{{mover, target_cycle}};
        if (partner >= 0) moves.push_back({partner, source_cycle});
        Score trial;
        const bool valid = evaluate_moves(moves, &trial);
        bool accept = valid && trial < current;
        if (!accept && valid && stall > 1000 &&
            trial.excess <= best.excess + max_uphill &&
            (rng() % static_cast<unsigned>(plateau_chance)) == 0) {
            accept = true;
        }
        if (!accept) {
            ++stall;
            continue;
        }
        commit_moves(moves, trial);
        rotation_cooldown = 0;
        remember_best(iteration, "pair", engine, source_cycle, target_cycle);
    }

    cycles = best_cycles;
    for (int engine = 0; engine < engine_count; ++engine) {
        std::ofstream output(output_prefix + "-" + engine_names[engine] + ".json");
        output << "{\"engine\":\"" << engine_names[engine]
               << "\",\"horizon\":" << horizon << ",\"cycles\":{";
        bool first = true;
        for (int node = 0; node < n; ++node) {
            if (engine_of[node] != engine) continue;
            if (!first) output << ',';
            first = false;
            output << '\"' << original[node] << "\":" << cycles[node];
        }
        output << "}}\n";
    }
    std::cout << "final excess=" << best.excess << " total=" << best.total
              << " count=" << best.count << " output=" << output_prefix
              << std::endl;
    return best.excess == 0 ? 0 : 1;
}
