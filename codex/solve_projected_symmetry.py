"""CP-SAT experiments for projected capacity-two engines.

The ordinary projected solver leaves a large permutation symmetry among the
eight lane-equivalent gather operations.  This standalone variant optionally
adds only *provable* symmetry breakers: operations with identical weighted
predecessor signatures, release/deadline window and tail are ordered by their
stable graph index.  It is a diagnostic tool; no production code imports it.
"""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path

from ortools.sat.python import cp_model


def main() -> None:
    payload = json.loads(Path(os.environ["GRAPH"]).read_text())
    records = payload["ops"]
    horizon = int(os.environ.get("TARGET", payload["horizon"]))
    capacity = int(os.environ.get("CAPACITY", payload.get("capacity") or 2))
    ids = [int(record["index"]) for record in records]
    position = {index: rank for rank, index in enumerate(ids)}
    earliest = {int(record["index"]): int(record["earliest"]) for record in records}
    latest = {int(record["index"]): min(horizon - 1, int(record["latest"])) for record in records}
    parents = {
        int(record["index"]): {
            int(parent): int(lag) for parent, lag in record["parents"].items()
        }
        for record in records
    }

    model = cp_model.CpModel()
    permutation = bool(int(os.environ.get("PERMUTATION", "0"))) and capacity == 2
    if permutation:
        # Encode the two-slot calendar as one permutation.  A position p
        # occupies cycle floor(p/2); AllDifferent removes the huge cumulative
        # interval symmetry and leaves only the legal lane pairing freedom.
        positions = {
            index: model.new_int_var(0, 2 * horizon - 1, f"p{index}")
            for index in ids
        }
        model.add_all_different(list(positions.values()))
        starts = {
            index: model.new_int_var(earliest[index], latest[index], f"s{index}")
            for index in ids
        }
        for index in ids:
            model.add_division_equality(starts[index], positions[index], 2)
    else:
        starts = {
            index: model.new_int_var(earliest[index], latest[index], f"s{index}")
            for index in ids
        }
    for child in ids:
        for parent, lag in parents[child].items():
            model.add(starts[child] >= starts[parent] + lag)

    anchor_raw = os.environ.get("ANCHOR_WINDOWS", "")
    if anchor_raw:
        # Keep the teacher's stable topological/resource order for jobs
        # outside the explicitly released windows.  Unlike fixing cycles,
        # these are only valid +1 resource-order hints; CP-SAT may retime all
        # anchored jobs.  The operation's topological position makes the
        # order safe even when two zero-lag jobs shared a teacher cycle.
        record_position = {
            int(record["index"]): int(record.get("position", rank))
            for rank, record in enumerate(records)
        }
        hint_payload = None
        hint_path = os.environ.get("HINT")
        if hint_path:
            hint_payload = json.loads(Path(hint_path).read_text())["cycles"]
        windows = []
        for raw in anchor_raw.split(","):
            if not raw:
                continue
            left, right = (int(value) for value in raw.split(":"))
            windows.append((left, right))
        def released(index: int) -> bool:
            if hint_payload is None:
                return False
            cycle = int(hint_payload.get(str(index), -10**9))
            return any(left <= cycle <= right for left, right in windows)
        anchored = [index for index in ids if not released(index)]
        if hint_payload is not None:
            anchored.sort(key=lambda index: (int(hint_payload.get(str(index), 0)), record_position[index]))
            for left, right in zip(anchored, anchored[2:]):
                model.add(starts[right] >= starts[left] + 1)
        print(f"anchor_windows={anchor_raw} anchored={len(anchored)}", flush=True)

    symmetry_mode = os.environ.get("SYMMETRY", "none")
    classes: dict[tuple[object, ...], list[int]] = defaultdict(list)
    if symmetry_mode != "none":
        for record in records:
            index = int(record["index"])
            parent_signature = tuple(sorted(parents[index].items()))
            if symmetry_mode == "parents":
                key = (parent_signature, earliest[index], latest[index], int(record["tail"]))
            elif symmetry_mode == "tag":
                key = (record["tag"], parent_signature, earliest[index], latest[index], int(record["tail"]))
            elif symmetry_mode == "group_round":
                key = (record["tag"], int(record["group"]), int(record["round"]), parent_signature)
            else:
                raise ValueError(f"unknown SYMMETRY={symmetry_mode}")
            classes[key].append(index)
        for members in classes.values():
            if len(members) < 2:
                continue
            members.sort()
            for left, right in zip(members, members[1:]):
                model.add(starts[left] <= starts[right])
            if bool(int(os.environ.get("PACK_IDENTICAL", "0"))) and len(members) % capacity == 0:
                # Optional aggressive research restriction: identical jobs
                # occupy consecutive capacity-sized bundles.  This is not a
                # general theorem for arbitrary DAGs, so it is opt-in and is
                # used only to test whether the gather-dominated instance
                # has a compact phase schedule.
                anchor = members[0]
                for offset, member in enumerate(members[1:], 1):
                    model.add(starts[member] == starts[anchor] + offset // capacity)
    if permutation:
        pass
    elif bool(int(os.environ.get("LANES", "0"))) and capacity == 2:
        # An explicit two-lane formulation gives CP-SAT stronger propagation
        # than one large cumulative: each operation chooses exactly one lane,
        # and each lane is a true NoOverlap sequence.  It is equivalent to a
        # capacity-two calendar, not an architectural restriction.
        lane_zero = []
        lane_one = []
        for index in ids:
            lane = model.new_bool_var(f"lane_{index}")
            zero = model.new_optional_fixed_size_interval_var(
                starts[index], 1, lane.Not(), f"z{index}"
            )
            one = model.new_optional_fixed_size_interval_var(
                starts[index], 1, lane, f"o{index}"
            )
            lane_zero.append(zero)
            lane_one.append(one)
        model.add_no_overlap(lane_zero)
        model.add_no_overlap(lane_one)
    else:
        intervals = [
            model.new_fixed_size_interval_var(starts[index], 1, f"i{index}")
            for index in ids
        ]
        model.add_cumulative(intervals, [1] * len(intervals), capacity)

    hint_path = os.environ.get("HINT")
    if hint_path:
        hint = json.loads(Path(hint_path).read_text())["cycles"]
        hinted_order = sorted(ids, key=lambda index: (int(hint.get(str(index), 0)), index))
        hinted_rank = {index: rank for rank, index in enumerate(hinted_order)}
        for index in ids:
            if str(index) in hint:
                hinted_cycle = min(latest[index], max(earliest[index], int(hint[str(index)])))
                model.add_hint(starts[index], hinted_cycle)
                if permutation:
                    model.add_hint(positions[index], hinted_rank[index])

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(os.environ.get("TIME_LIMIT", "60"))
    solver.parameters.num_workers = int(os.environ.get("WORKERS", "8"))
    solver.parameters.random_seed = int(os.environ.get("SEED", "1"))
    solver.parameters.randomize_search = bool(int(os.environ.get("RANDOMIZE", "0")))
    solver.parameters.log_search_progress = bool(int(os.environ.get("LOG", "0")))
    status = solver.solve(model)
    print(
        f"status={solver.status_name(status)} jobs={len(ids)} "
        f"symmetry={symmetry_mode} classes={sum(len(v)>1 for v in classes.values())} "
        f"wall={solver.wall_time:.3f}",
        flush=True,
    )
    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return
    cycles = {str(index): solver.value(starts[index]) for index in ids}
    output = Path(os.environ.get("OUT", "/tmp/aopt-symmetry-schedule.json"))
    output.write_text(json.dumps({"engine": payload["engine"], "horizon": horizon, "cycles": cycles}))
    print(f"span={max(cycles.values()) + 1} output={output}")


if __name__ == "__main__":
    main()
