#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

struct Edge {
    int to;
    int lag;
};

struct Score {
    int cyclic_nodes = std::numeric_limits<int>::max();
    int span = std::numeric_limits<int>::max();
    int critical_count = std::numeric_limits<int>::max();
    std::int64_t pressure = std::numeric_limits<std::int64_t>::max();
    std::int64_t sum = std::numeric_limits<std::int64_t>::max();
    bool operator<(const Score& other) const {
        return std::tie(cyclic_nodes, span, critical_count, pressure, sum) <
               std::tie(other.cyclic_nodes, other.span, other.critical_count,
                        other.pressure, other.sum);
    }
    bool operator==(const Score& other) const {
        return cyclic_nodes == other.cyclic_nodes && span == other.span &&
               critical_count == other.critical_count &&
               pressure == other.pressure && sum == other.sum;
    }
};

struct BridgePair {
    int upstream_engine;
    int upstream_rank;
    int downstream_engine;
    int downstream_rank;
};

struct Evaluation {
    Score score;
    std::vector<int> starts;
    // `critical` keeps the original near-critical fanout.  `focus` is the
    // narrower set of resource edges on exact makespan endpoints, used only
    // by the optional coarse neighbourhood to avoid sampling 9k+ unrelated
    // edges on a wide schedule frontier.
    std::vector<std::pair<int, int>> critical;
    std::vector<std::pair<int, int>> focus;
    // Resource edges immediately around a true dependency transition on an
    // exact critical path.  These are the places where ALU/VALU chains hand
    // work to one another; changing a random edge deep inside a long pure
    // resource run rarely affects the makespan.
    std::vector<std::pair<int, int>> bridge;
    // A paired view of both resource edges bordering one true DAG dependency
    // on an exact critical path.  It enables coordinated ALU/VALU edits.
    std::vector<BridgePair> bridge_pairs;
};

struct Move {
    int engine;
    int from;
    int to;
};

static void apply_move(std::vector<int>& order, int from, int to) {
    if (from == to) return;
    int value = order[from];
    if (from < to) {
        std::move(order.begin() + from + 1, order.begin() + to + 1,
                  order.begin() + from);
    } else {
        std::move_backward(order.begin() + to, order.begin() + from,
                           order.begin() + from + 1);
    }
    order[to] = value;
}

// Keep the original insertion Move above as the default neighbourhood.  The
// block neighbourhood below is deliberately separate: it lets callers opt in
// to coarser resource-chain edits without changing legacy search behaviour.
// `last` is exclusive.  Reversal is its own inverse; rotate is undone by
// rotating the same range in the opposite direction.
enum class BlockKind : unsigned char { kReverse, kRotate };

struct BlockMove {
    int engine;
    int first;
    int last;
    int shift;
    BlockKind kind;
};

static void apply_block_move(std::vector<int>& order, const BlockMove& move) {
    auto first = order.begin() + move.first;
    auto last = order.begin() + move.last;
    if (move.kind == BlockKind::kReverse) {
        std::reverse(first, last);
        return;
    }
    std::rotate(first, first + move.shift, last);
}

static void undo_block_move(std::vector<int>& order, const BlockMove& move) {
    auto first = order.begin() + move.first;
    auto last = order.begin() + move.last;
    if (move.kind == BlockKind::kReverse) {
        std::reverse(first, last);
        return;
    }
    std::rotate(first, last - move.shift, last);
}

int main(int argc, char** argv) {
    if (argc < 8) {
        std::cerr << "usage: search_resource_orders problem output-prefix "
                     "iterations candidates radius seed max-uphill [batch] "
                     "[neighborhood] [max-block] [anneal-temperature] "
                     "[shuffle-same-cycle]\n"
                     "  neighborhood: 0=legacy insertion, 1=reverse, "
                     "2=rotate, 3=mixed, 4=capacity-bundle reverse, "
                     "5=critical-edge rotate, 6=bridge-local swap, "
                     "7=paired bridge rotate, 8=paired bridge swap\n";
        return 2;
    }
    std::ifstream input(argv[1]);
    if (!input) {
        std::cerr << "cannot open problem input: " << argv[1] << '\n';
        return 2;
    }
    const std::string output_prefix = argv[2];
    const int iterations = std::stoi(argv[3]);
    const int candidates_per_iteration = std::stoi(argv[4]);
    const int radius = std::stoi(argv[5]);
    const unsigned seed = static_cast<unsigned>(std::stoul(argv[6]));
    const int max_uphill = std::stoi(argv[7]);
    const int move_batch = argc >= 9 ? std::max(1, std::stoi(argv[8])) : 1;
    const int neighborhood = argc >= 10 ? std::clamp(std::stoi(argv[9]), 0, 8) : 0;
    // 2..16 is large enough to rewire a capacity-12 ALU chain while still
    // preserving a local neighbourhood.  Larger reversals usually create
    // cycles and give the evaluator little useful signal.
    const int max_block = argc >= 11 ? std::clamp(std::stoi(argv[10]), 2, 16) : 16;
    const double anneal_temperature =
        argc >= 12 ? std::max(0.0, std::stod(argv[11])) : 1.25;
    // Restarts normally reshuffle semantically unordered same-cycle jobs to
    // expose fresh lane matchings.  Set this final argument to 0 when feeding
    // back a hand-picked/previous-best resource order and preserving it
    // exactly matters more than diversification.
    const bool shuffle_same_cycle = argc < 13 || std::stoi(argv[12]) != 0;

    int n, m, engine_count;
    input >> n >> m >> engine_count;
    std::vector<int> original(n), engine_of(n), earliest(n), tail(n), hint(n);
    for (int i = 0; i < n; ++i) {
        input >> original[i] >> engine_of[i] >> earliest[i] >> tail[i] >> hint[i];
    }
    std::vector<std::vector<Edge>> children(n);
    std::vector<int> base_indegree(n, 0);
    for (int i = 0; i < m; ++i) {
        int parent, child, lag;
        input >> parent >> child >> lag;
        children[parent].push_back({child, lag});
        ++base_indegree[child];
    }
    std::vector<std::string> engine_names(engine_count);
    std::vector<int> capacities(engine_count);
    std::vector<std::vector<int>> orders(engine_count);
    for (int e = 0; e < engine_count; ++e) {
        int count;
        input >> engine_names[e] >> capacities[e] >> count;
        orders[e].resize(count);
        for (int& node : orders[e]) input >> node;
    }
    if (!input) {
        std::cerr << "invalid problem input\n";
        return 2;
    }

    std::mt19937 rng(seed);
    // Same-cycle operations are semantically unordered.  Different initial
    // matchings expose different resource-chain neighborhoods at zero cost.
    if (shuffle_same_cycle) {
        for (auto& order : orders) {
            for (int first = 0; first < static_cast<int>(order.size());) {
                int last = first + 1;
                while (last < static_cast<int>(order.size()) &&
                       hint[order[last]] == hint[order[first]]) {
                    ++last;
                }
                std::shuffle(order.begin() + first, order.begin() + last, rng);
                first = last;
            }
        }
    }

    auto evaluate = [&](const std::vector<std::vector<int>>& candidate_orders,
                        bool keep_starts) -> Evaluation {
        std::vector<int> resource_child(n, -1), resource_engine(n, -1),
            resource_rank(n, -1);
        std::vector<int> indegree = base_indegree;
        for (int e = 0; e < engine_count; ++e) {
            const auto& order = candidate_orders[e];
            const int capacity = capacities[e];
            for (int rank = 0; rank + capacity < static_cast<int>(order.size()); ++rank) {
                int parent = order[rank];
                int child = order[rank + capacity];
                resource_child[parent] = child;
                resource_engine[parent] = e;
                resource_rank[parent] = rank;
                ++indegree[child];
            }
        }

        std::vector<int> starts = earliest;
        std::vector<int> reason_parent(n, -1), reason_engine(n, -2),
            reason_rank(n, -1), queue(n);
        int head = 0, end = 0;
        for (int i = 0; i < n; ++i) {
            if (indegree[i] == 0) queue[end++] = i;
        }
        int seen = 0;
        while (head < end) {
            int parent = queue[head++];
            ++seen;
            for (const Edge edge : children[parent]) {
                int candidate = starts[parent] + edge.lag;
                if (candidate > starts[edge.to]) {
                    starts[edge.to] = candidate;
                    reason_parent[edge.to] = parent;
                    reason_engine[edge.to] = -1;
                    reason_rank[edge.to] = -1;
                }
                if (--indegree[edge.to] == 0) queue[end++] = edge.to;
            }
            int child = resource_child[parent];
            if (child >= 0) {
                int candidate = starts[parent] + 1;
                if (candidate >= starts[child]) {
                    starts[child] = candidate;
                    reason_parent[child] = parent;
                    reason_engine[child] = resource_engine[parent];
                    reason_rank[child] = resource_rank[parent];
                }
                if (--indegree[child] == 0) queue[end++] = child;
            }
        }
        if (seen != n) {
            Score score;
            score.cyclic_nodes = 0;
            score.span = 0;
            score.critical_count = 0;
            score.pressure = 0;
            score.sum = 0;
            std::vector<int> index(n, -1), low(n), component(n, -1), stack;
            std::vector<unsigned char> on_stack(n, 0);
            std::vector<int> component_sizes;
            int next_index = 0;
            auto visit = [&](auto&& self, int node) -> void {
                index[node] = low[node] = next_index++;
                stack.push_back(node);
                on_stack[node] = 1;
                auto follow = [&](int child) {
                    if (child < 0 || indegree[child] <= 0) return;
                    if (index[child] < 0) {
                        self(self, child);
                        low[node] = std::min(low[node], low[child]);
                    } else if (on_stack[child]) {
                        low[node] = std::min(low[node], index[child]);
                    }
                };
                for (const Edge edge : children[node]) follow(edge.to);
                follow(resource_child[node]);
                if (low[node] != index[node]) return;
                int size = 0;
                while (true) {
                    int member = stack.back();
                    stack.pop_back();
                    on_stack[member] = 0;
                    component[member] = component_sizes.size();
                    ++size;
                    if (member == node) break;
                }
                component_sizes.push_back(size);
            };
            std::vector<std::pair<int, int>> critical;
            for (int node = 0; node < n; ++node) {
                if (indegree[node] <= 0) continue;
                score.sum += starts[node];
                if (index[node] < 0) visit(visit, node);
            }
            std::vector<unsigned char> cyclic_component(
                component_sizes.size(), 0
            );
            for (int id = 0; id < static_cast<int>(component_sizes.size()); ++id) {
                if (component_sizes[id] > 1) {
                    cyclic_component[id] = 1;
                    score.cyclic_nodes += component_sizes[id];
                }
            }
            for (int node = 0; node < n; ++node) {
                if (indegree[node] <= 0 ||
                    !cyclic_component[component[node]]) continue;
                for (const Edge edge : children[node]) {
                    if (edge.to >= 0 && component[edge.to] == component[node]) {
                        ++score.pressure;
                    }
                }
                int child = resource_child[node];
                if (child >= 0 && component[child] == component[node]) {
                    ++score.pressure;
                    critical.push_back(
                        {resource_engine[node], resource_rank[node]});
                }
            }
            if (!keep_starts) starts.clear();
            return {score, std::move(starts), std::move(critical), {}, {}, {}};
        }

        Score score;
        score.cyclic_nodes = 0;
        score.span = starts[0] + tail[0] + 1;
        score.critical_count = 0;
        score.pressure = 0;
        score.sum = 0;
        for (int i = 0; i < n; ++i) {
            score.sum += starts[i];
            int completion = starts[i] + tail[i] + 1;
            if (completion > score.span) {
                score.span = completion;
            }
        }
        const int pressure_floor = score.span - 64;
        for (int i = 0; i < n; ++i) {
            int completion = starts[i] + tail[i] + 1;
            score.critical_count += completion == score.span;
            if (completion > pressure_floor) {
                std::int64_t excess = completion - pressure_floor;
                score.pressure += excess * excess;
            }
        }
        // Trial candidates are ranked only by Score.  Reconstructing every
        // near-critical path for them is quadratic in a wide frontier and
        // can dominate the actual DAG evaluation.  Build the detailed path
        // sets only for accepted/current/best states where they guide the
        // next neighbourhood sample.
        if (!keep_starts) {
            return {score, {}, {}, {}, {}, {}};
        }
        std::vector<std::pair<int, int>> critical;
        std::vector<unsigned char> traced(n, 0);
        for (int terminal = 0; terminal < n; ++terminal) {
            if (starts[terminal] + tail[terminal] + 1 < score.span - 2) continue;
            for (int node = terminal; node >= 0 && !traced[node];
                 node = reason_parent[node]) {
                traced[node] = 1;
                if (reason_engine[node] >= 0) {
                    critical.push_back(
                        {reason_engine[node], reason_rank[node]});
                }
            }
        }
        std::vector<std::pair<int, int>> focus;
        std::vector<unsigned char> focus_traced(n, 0);
        for (int terminal = 0; terminal < n; ++terminal) {
            if (starts[terminal] + tail[terminal] + 1 != score.span) continue;
            for (int node = terminal; node >= 0 && !focus_traced[node];
                 node = reason_parent[node]) {
                focus_traced[node] = 1;
                if (reason_engine[node] >= 0) {
                    focus.push_back({reason_engine[node], reason_rank[node]});
                }
            }
        }
        std::vector<std::pair<int, int>> bridge;
        std::vector<unsigned char> bridge_mark(n, 0);
        for (int terminal = 0; terminal < n; ++terminal) {
            if (starts[terminal] + tail[terminal] + 1 != score.span) continue;
            std::vector<int> recent_resources;
            recent_resources.reserve(16);
            int upstream_budget = 0;
            for (int node = terminal; node >= 0; node = reason_parent[node]) {
                if (reason_engine[node] >= 0) {
                    recent_resources.push_back(node);
                    if (static_cast<int>(recent_resources.size()) > 16) {
                        recent_resources.erase(recent_resources.begin());
                    }
                    if (upstream_budget > 0) {
                        bridge_mark[node] = 1;
                        --upstream_budget;
                    }
                } else if (reason_parent[node] >= 0) {
                    // The previous resource run is downstream of this DAG
                    // dependency in forward time; the next run we traverse
                    // is upstream.  Mark both short boundaries.
                    for (int resource_node : recent_resources) {
                        bridge_mark[resource_node] = 1;
                    }
                    recent_resources.clear();
                    upstream_budget = 16;
                }
            }
        }
        for (int node = 0; node < n; ++node) {
            if (bridge_mark[node] && reason_engine[node] >= 0) {
                bridge.push_back({reason_engine[node], reason_rank[node]});
            }
        }
        std::vector<BridgePair> bridge_pairs;
        for (int terminal = 0; terminal < n; ++terminal) {
            if (starts[terminal] + tail[terminal] + 1 != score.span) continue;
            int downstream_engine = -1;
            int downstream_rank = -1;
            int pending_downstream_engine = -1;
            int pending_downstream_rank = -1;
            bool need_upstream = false;
            for (int node = terminal; node >= 0; node = reason_parent[node]) {
                if (reason_engine[node] >= 0) {
                    if (need_upstream) {
                        bridge_pairs.push_back(
                            {reason_engine[node], reason_rank[node],
                             pending_downstream_engine,
                             pending_downstream_rank}
                        );
                        need_upstream = false;
                    }
                    downstream_engine = reason_engine[node];
                    downstream_rank = reason_rank[node];
                } else if (reason_parent[node] >= 0 &&
                           downstream_engine >= 0) {
                    pending_downstream_engine = downstream_engine;
                    pending_downstream_rank = downstream_rank;
                    need_upstream = true;
                }
            }
        }
        return {score, std::move(starts), std::move(critical), std::move(focus),
                std::move(bridge), std::move(bridge_pairs)};
    };

    Evaluation current = evaluate(orders, true);
    Evaluation best = current;
    auto best_orders = orders;
    std::cout << "start cyclic=" << best.score.cyclic_nodes
              << " span=" << best.score.span << " endpoints="
              << best.score.critical_count << " pressure="
              << best.score.pressure << " sum=" << best.score.sum
              << " critical=" << best.critical.size()
              << " focus=" << best.focus.size()
              << " bridge=" << best.bridge.size()
              << " bridge-pairs=" << best.bridge_pairs.size() << std::endl;

    int stall = 0;
    if (neighborhood == 0) {
    for (int iteration = 0; iteration < iterations; ++iteration) {
        std::vector<Move> chosen;
        Evaluation chosen_eval;
        for (int candidate_number = 0; candidate_number < candidates_per_iteration;
             ++candidate_number) {
            std::vector<Move> moves;
            for (int batch_index = 0; batch_index < move_batch; ++batch_index) {
                int e, anchor;
                if (!current.critical.empty() &&
                    candidate_number * 4 < candidates_per_iteration * 3) {
                    const auto [critical_engine, critical_rank] =
                        current.critical[rng() % current.critical.size()];
                    e = critical_engine;
                    anchor = critical_rank + ((rng() & 1U) ? capacities[e] : 0);
                } else {
                    e = rng() % engine_count;
                    anchor = rng() % orders[e].size();
                }
                int from = std::clamp(
                    anchor + static_cast<int>(rng() % 17) - 8, 0,
                    static_cast<int>(orders[e].size()) - 1);
                int distance = 1 + static_cast<int>(rng() % std::max(1, radius));
                int to = from + ((rng() & 1U) ? distance : -distance);
                to = std::clamp(to, 0, static_cast<int>(orders[e].size()) - 1);
                if (from == to) continue;
                moves.push_back({e, from, to});
                apply_move(orders[e], from, to);
            }
            if (moves.empty()) continue;
            Evaluation trial = evaluate(orders, false);
            for (auto move = moves.rbegin(); move != moves.rend(); ++move) {
                apply_move(orders[move->engine], move->to, move->from);
            }
            if (trial.score.span == std::numeric_limits<int>::max()) continue;
            if (chosen.empty() || trial.score < chosen_eval.score) {
                chosen = std::move(moves);
                chosen_eval = std::move(trial);
            }
        }
        if (chosen.empty()) continue;
        bool accept = chosen_eval.score < current.score;
        if (!accept && stall >= 4 && chosen_eval.score.span <= best.score.span + max_uphill) {
            // Plateau/uphill moves are essential: deleting one critical
            // cycle usually requires a sequence of resource-edge reversals.
            std::uniform_int_distribution<int> chance(0, std::max(1, max_uphill + 1));
            accept = chance(rng) == 0 || chosen_eval.score.span <= current.score.span;
        }
        if (!accept) {
            ++stall;
            continue;
        }
        for (const Move move : chosen) {
            apply_move(orders[move.engine], move.from, move.to);
        }
        current = evaluate(orders, true);
        ++stall;
        if (current.score < best.score) {
            int previous_cyclic = best.score.cyclic_nodes;
            int previous_span = best.score.span;
            best = current;
            best_orders = orders;
            stall = 0;
            if (best.score.cyclic_nodes < previous_cyclic ||
                best.score.span < previous_span || iteration % 1000 == 0) {
                std::cout << "best cyclic=" << best.score.cyclic_nodes
                          << " span=" << best.score.span << " sum="
                          << best.score.sum << " pressure="
                          << best.score.pressure << " endpoints="
                          << best.score.critical_count << " iteration=" << iteration
                          << " engine=" << engine_names[chosen.front().engine]
                          << " move=" << chosen.front().from << ':'
                          << chosen.front().to << " batch=" << chosen.size()
                          << " critical=" << best.critical.size() << std::endl;
            }
        }
    }
    } else {
        // Coarse resource-order moves complement the insertion neighbourhood
        // above.  A capacity-C chain is only changed when a move crosses an
        // element roughly C ranks away, so an adjacent insertion is often too
        // weak to escape a critical-path basin.  We therefore target a traced
        // resource edge and reverse/rotate a 2..16 element window containing
        // both ends whenever possible.
        auto choose_block = [&](int candidate_number) -> BlockMove {
            int engine = -1;
            int critical_rank = -1;
            const auto& focal_edges =
                !current.bridge.empty() && (rng() % 10U) < 7U
                ? current.bridge
                : (!current.focus.empty() && (rng() % 10U) < 8U
                    ? current.focus
                    : current.critical);
            const bool target_critical =
                !focal_edges.empty() &&
                candidate_number * 5 < candidates_per_iteration * 4;
            if (target_critical) {
                const auto [selected_engine, selected_rank] =
                    focal_edges[rng() % focal_edges.size()];
                if (selected_engine >= 0 && selected_engine < engine_count &&
                    orders[selected_engine].size() >= 2) {
                    engine = selected_engine;
                    critical_rank = selected_rank;
                }
            }
            if (engine < 0) {
                for (int tries = 0; tries < engine_count * 2; ++tries) {
                    const int selected_engine = rng() % engine_count;
                    if (orders[selected_engine].size() >= 2) {
                        engine = selected_engine;
                        break;
                    }
                }
            }
            if (engine < 0) return {-1, 0, 0, 0, BlockKind::kReverse};

            const int size = static_cast<int>(orders[engine].size());
            const int block_limit = std::min(max_block, size);
            const bool use_capacity_bundle =
                neighborhood == 4 ||
                (neighborhood == 3 && (rng() % 10U) < 3U);
            if (use_capacity_bundle && capacities[engine] <= block_limit) {
                // A full capacity-sized group is one resource cycle in the
                // flattened order.  Reversing just that group changes the
                // predecessor/successor lane matching, while retaining the
                // group's coarse placement.  It is a much safer large move
                // than reordering several complete cycles at once.
                const int capacity = capacities[engine];
                int anchor = rng() % size;
                if (critical_rank >= 0) {
                    anchor = critical_rank +
                        ((rng() & 1U) ? capacity : 0);
                    anchor = std::clamp(anchor, 0, size - 1);
                }
                int first = (anchor / capacity) * capacity;
                if (first + capacity > size) first -= capacity;
                if (first >= 0) {
                    return {engine, first, first + capacity, 0,
                            BlockKind::kReverse};
                }
            }
            const bool use_critical_edge_rotate =
                neighborhood == 5 ||
                (neighborhood == 3 && (rng() % 10U) < 4U);
            if (use_critical_edge_rotate && critical_rank >= 0 &&
                critical_rank + capacities[engine] < size &&
                capacities[engine] + 1 <= block_limit) {
                // Rotate [parent, ..., child] around a traced resource edge.
                // Unlike a full reversal, this keeps the C-1 interior jobs in
                // their relative order and directly changes which endpoint
                // inherits the critical lane predecessor.
                const int length = capacities[engine] + 1;
                const int shift = (rng() & 1U) ? 1 : capacities[engine];
                return {engine, critical_rank, critical_rank + length, shift,
                        BlockKind::kRotate};
            }
            if (neighborhood == 6 && critical_rank >= 0 && size >= 2) {
                // The 2-item reversal is the finest block move.  Anchor it
                // directly at a bridge edge endpoint rather than sampling a
                // generic adjacent swap elsewhere in a 10k-item machine
                // order.  This is useful after coarse rotations have found
                // the right region but leave a one-cycle tie unresolved.
                int anchor = critical_rank +
                    ((rng() & 1U) ? capacities[engine] : 0);
                anchor = std::clamp(anchor, 0, size - 1);
                int first = anchor;
                if (first == size - 1 || ((rng() & 1U) && first > 0)) --first;
                return {engine, first, first + 2, 0, BlockKind::kReverse};
            }
            int length_min = 2;
            int first_min = 0;
            int first_max = size - 2;
            if (critical_rank >= 0 &&
                critical_rank + capacities[engine] < size &&
                capacities[engine] + 1 <= block_limit && (rng() & 1U)) {
                // Include both endpoints of one traced resource edge.  This
                // is particularly important for ALU's capacity-12 chain.
                length_min = capacities[engine] + 1;
            }
            const int length = length_min + static_cast<int>(rng() %
                static_cast<unsigned>(block_limit - length_min + 1));
            if (critical_rank >= 0 &&
                critical_rank + capacities[engine] < size &&
                length >= capacities[engine] + 1) {
                const int edge_child = critical_rank + capacities[engine];
                first_min = std::max(0, edge_child - length + 1);
                first_max = std::min(critical_rank, size - length);
            } else {
                int anchor = rng() % size;
                if (critical_rank >= 0) {
                    anchor = critical_rank +
                        ((rng() & 1U) ? capacities[engine] : 0);
                    anchor = std::clamp(anchor, 0, size - 1);
                }
                first_min = std::max(0, anchor - length + 1);
                first_max = std::min(anchor, size - length);
            }
            const int first = first_min + static_cast<int>(rng() %
                static_cast<unsigned>(first_max - first_min + 1));
            BlockKind kind = BlockKind::kReverse;
            if (neighborhood == 2 ||
                (neighborhood == 3 && (rng() % 10U) >= 6U)) {
                kind = BlockKind::kRotate;
            }
            const int shift = kind == BlockKind::kRotate
                ? 1 + static_cast<int>(rng() % static_cast<unsigned>(length - 1))
                : 0;
            return {engine, first, first + length, shift, kind};
        };

        auto append_edge_rotate = [&](std::vector<BlockMove>& moves,
                                      int engine, int rank) {
            if (engine < 0 || engine >= engine_count || rank < 0) return false;
            const int capacity = capacities[engine];
            const int size = static_cast<int>(orders[engine].size());
            if (rank + capacity >= size || capacity + 1 > max_block) return false;
            const int shift = (rng() & 1U) ? 1 : capacity;
            moves.push_back(
                {engine, rank, rank + capacity + 1, shift, BlockKind::kRotate}
            );
            return true;
        };

        auto append_bridge_swap = [&](std::vector<BlockMove>& moves,
                                      int engine, int rank) {
            if (engine < 0 || engine >= engine_count || rank < 0) return false;
            const int size = static_cast<int>(orders[engine].size());
            if (size < 2) return false;
            int anchor = rank + ((rng() & 1U) ? capacities[engine] : 0);
            anchor = std::clamp(anchor, 0, size - 1);
            int first = anchor;
            if (first == size - 1 || ((rng() & 1U) && first > 0)) --first;
            moves.push_back({engine, first, first + 2, 0, BlockKind::kReverse});
            return true;
        };

        auto accept_annealed = [&](const Score& trial, int iteration,
                                   int local_stall) {
            if (trial < current.score) return true;
            if (trial.cyclic_nodes != 0 ||
                trial.span > best.score.span + max_uphill) return false;
            if (trial == current.score) return true;
            if (anneal_temperature <= 0.0) return false;

            // Scores are lexicographic, but SA needs a continuous energy.
            // Normalizing the secondary terms keeps a one-cycle uphill move
            // possible early in the search without letting pressure dominate
            // it.  The small stall term reheats a local basin deterministically.
            const double endpoint_scale =
                std::max(8.0, static_cast<double>(current.score.critical_count) / 8.0);
            const double pressure_scale =
                std::max(1.0, static_cast<double>(current.score.pressure) / 16.0);
            const double sum_scale =
                std::max(1.0, static_cast<double>(current.score.sum) / 128.0);
            double energy = std::max(0, trial.span - current.score.span);
            energy += std::max(0, trial.critical_count - current.score.critical_count) /
                endpoint_scale;
            energy += static_cast<double>(
                std::max<std::int64_t>(0, trial.pressure - current.score.pressure)) /
                pressure_scale;
            energy += static_cast<double>(
                std::max<std::int64_t>(0, trial.sum - current.score.sum)) /
                sum_scale;
            const double progress = static_cast<double>(iteration) /
                std::max(1, iterations);
            const double reheat = 1.0 + 0.15 * std::min(local_stall, 8);
            const double temperature = anneal_temperature * reheat *
                (0.08 + 0.92 * (1.0 - progress));
            std::uniform_real_distribution<double> unit(0.0, 1.0);
            return unit(rng) < std::exp(-energy / temperature);
        };

        for (int iteration = 0; iteration < iterations; ++iteration) {
            std::vector<BlockMove> chosen;
            Evaluation chosen_eval;
            std::vector<BlockMove> exploratory;
            Evaluation exploratory_eval;
            int feasible_candidates = 0;
            for (int candidate_number = 0;
                 candidate_number < candidates_per_iteration;
                 ++candidate_number) {
                std::vector<BlockMove> moves;
                for (int batch_index = 0; batch_index < move_batch; ++batch_index) {
                    if ((neighborhood == 7 || neighborhood == 8) &&
                        !current.bridge_pairs.empty()) {
                        const BridgePair* selected_pair = nullptr;
                        // Prefer an actual ALU<->VALU handoff.  Same-engine
                        // pairs are retained as a fallback because a DAG
                        // dependency can still split two useful machine runs.
                        for (int tries = 0;
                             tries < static_cast<int>(current.bridge_pairs.size());
                             ++tries) {
                            const BridgePair& candidate = current.bridge_pairs[
                                rng() % current.bridge_pairs.size()
                            ];
                            if (candidate.upstream_engine !=
                                candidate.downstream_engine) {
                                selected_pair = &candidate;
                                break;
                            }
                            if (!selected_pair) selected_pair = &candidate;
                        }
                        if (selected_pair) {
                            std::vector<BlockMove> paired;
                            const bool upstream_ok = neighborhood == 7
                                ? append_edge_rotate(
                                    paired, selected_pair->upstream_engine,
                                    selected_pair->upstream_rank
                                )
                                : append_bridge_swap(
                                    paired, selected_pair->upstream_engine,
                                    selected_pair->upstream_rank
                                );
                            const bool downstream_ok = neighborhood == 7
                                ? append_edge_rotate(
                                    paired, selected_pair->downstream_engine,
                                    selected_pair->downstream_rank
                                )
                                : append_bridge_swap(
                                    paired, selected_pair->downstream_engine,
                                    selected_pair->downstream_rank
                                );
                            if (upstream_ok && downstream_ok && paired.size() == 2) {
                                for (const BlockMove& move : paired) {
                                    apply_block_move(orders[move.engine], move);
                                }
                                moves.insert(moves.end(), paired.begin(), paired.end());
                                continue;
                            }
                        }
                    }
                    BlockMove move = choose_block(candidate_number);
                    if (move.engine < 0 || move.last - move.first < 2) continue;
                    moves.push_back(move);
                    apply_block_move(orders[move.engine], move);
                }
                if (moves.empty()) continue;
                Evaluation trial = evaluate(orders, false);
                for (auto move = moves.rbegin(); move != moves.rend(); ++move) {
                    undo_block_move(orders[move->engine], *move);
                }
                if (trial.score.span == std::numeric_limits<int>::max() ||
                    trial.score.cyclic_nodes != 0) continue;
                ++feasible_candidates;
                // Reservoir sampling gives the annealer an unbiased feasible
                // alternative.  Selecting only the lexicographically best
                // candidate turns a nominal SA loop back into hill-climbing:
                // tiny pressure improvements would otherwise suppress every
                // temporary uphill route out of the current basin.
                if (feasible_candidates == 1 ||
                    rng() % static_cast<unsigned>(feasible_candidates) == 0U) {
                    exploratory = moves;
                    exploratory_eval = trial;
                }
                if (chosen.empty() || trial.score < chosen_eval.score) {
                    chosen = std::move(moves);
                    chosen_eval = std::move(trial);
                }
            }
            if (chosen.empty()) {
                ++stall;
                continue;
            }
            const bool structural_best =
                chosen_eval.score.span < current.score.span ||
                chosen_eval.score.critical_count < current.score.critical_count;
            if (!structural_best && !exploratory.empty() &&
                anneal_temperature > 0.0 && (rng() & 3U) == 0U) {
                chosen = std::move(exploratory);
                chosen_eval = std::move(exploratory_eval);
            }
            if (!accept_annealed(chosen_eval.score, iteration, stall)) {
                ++stall;
                continue;
            }
            for (const BlockMove& move : chosen) {
                apply_block_move(orders[move.engine], move);
            }
            current = evaluate(orders, true);
            ++stall;
            if (current.score < best.score) {
                const int previous_cyclic = best.score.cyclic_nodes;
                const int previous_span = best.score.span;
                best = current;
                best_orders = orders;
                stall = 0;
                if (best.score.cyclic_nodes < previous_cyclic ||
                    best.score.span < previous_span || iteration % 100 == 0) {
                    const BlockMove& move = chosen.front();
                    std::cout << "best cyclic=" << best.score.cyclic_nodes
                              << " span=" << best.score.span << " sum="
                              << best.score.sum << " pressure="
                              << best.score.pressure << " endpoints="
                              << best.score.critical_count << " iteration="
                              << iteration << " engine="
                              << engine_names[move.engine] << " block="
                              << move.first << ':' << move.last << " kind="
                              << (move.kind == BlockKind::kReverse ? "reverse" : "rotate")
                              << " batch=" << chosen.size() << " critical="
                              << best.critical.size() << std::endl;
                }
            }
        }
    }

    best = evaluate(best_orders, true);
    if (best.score.cyclic_nodes != 0) {
        std::cout << "final cyclic=" << best.score.cyclic_nodes
                  << " pressure=" << best.score.pressure
                  << " critical=" << best.critical.size() << std::endl;
        return 1;
    }
    for (int e = 0; e < engine_count; ++e) {
        std::ofstream output(output_prefix + "-" + engine_names[e] + ".json");
        output << "{\"engine\":\"" << engine_names[e] << "\",\"horizon\":"
               << best.score.span << ",\"cycles\":{";
        bool first = true;
        for (int node = 0; node < n; ++node) {
            if (engine_of[node] != e) continue;
            if (!first) output << ',';
            first = false;
            output << '\"' << original[node] << "\":" << best.starts[node];
        }
        output << "},\"order\":[";
        for (std::size_t rank = 0; rank < best_orders[e].size(); ++rank) {
            if (rank) output << ',';
            output << original[best_orders[e][rank]];
        }
        output << "]}\n";
    }
    std::cout << "final cyclic=0 span=" << best.score.span << " endpoints="
              << best.score.critical_count << " pressure="
              << best.score.pressure << " sum=" << best.score.sum
              << " output=" << output_prefix << std::endl;
    return 0;
}
