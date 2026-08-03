"""Drive monotone ALU/VALU soft-cap repair through successive plateaus.

The projected schedules are individually feasible at the requested horizon.
This driver repeatedly asks ``solve_projected_engines`` to minimize only the
cross-engine dependence excess above the next lower cap, while retaining a
hard guard at the current cap.  Broad plateaus use CP-SAT's LNS portfolio;
small residual critical sets switch back to exact neighborhood search.
Every accepted state is checkpointed, so a long run can be resumed safely.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys


METRICS_RE = re.compile(
    r"alternating_joint_metrics max_sum_count=\((\d+), (\d+), (\d+)\)"
)
OBJECTIVE_RE = re.compile(
    r"alternating_joint status=(?:FEASIBLE|OPTIMAL) objective=([0-9.]+)"
)


def run_candidate(
    *,
    env_file: Path,
    alu: Path,
    valu: Path,
    prefix: Path,
    radius: int,
    window: str,
    seed: int,
    lns_only: bool,
    hard_cap: bool,
    time_limit: int,
    workers: int,
) -> dict[str, object]:
    assignments = {
        "ENGINES": "alu,valu",
        "TARGET": "959",
        "PROJECTED_HINTS": f"{alu},{valu}",
        "ALTERNATING_REPAIR": "1",
        "ALTERNATING_JOINT": "1",
        "ALTERNATING_JOINT_TIME_LIMIT": str(time_limit),
        "ALTERNATING_LOCAL_RADIUS": str(radius),
        "ALTERNATING_FORCE_MAX_DROP": "1",
        "ALTERNATING_EXPLICIT_DOMAINS": "1",
        "RANDOM_SEED": str(seed),
        "WORKERS": str(workers),
        "OUT_PREFIX": str(prefix),
    }
    if hard_cap:
        assignments["ALTERNATING_JOINT_FEASIBILITY"] = "1"
    else:
        assignments["ALTERNATING_JOINT_SOFT_CAP"] = "1"
    if lns_only:
        assignments["ALTERNATING_JOINT_LNS_ONLY"] = "1"
    if window:
        assignments["ALTERNATING_JOINT_WINDOWS"] = window
    command = "\n".join(
        [f"source {shlex.quote(str(env_file))}"]
        + [f"export {key}={shlex.quote(value)}" for key, value in assignments.items()]
        + ["python3 -m codex.solve_projected_engines"]
    )
    completed = subprocess.run(
        ["zsh", "-lc", command],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    metrics = METRICS_RE.findall(completed.stdout)
    objectives = OBJECTIVE_RE.findall(completed.stdout)
    result: dict[str, object] = {
        "radius": radius,
        "window": window,
        "seed": seed,
        "lns_only": lns_only,
        "returncode": completed.returncode,
        "output": completed.stdout,
        "prefix": str(prefix),
    }
    if metrics and objectives:
        maximum, total, count = map(int, metrics[-1])
        result.update(
            {
                "maximum": maximum,
                "total": total,
                "count": count,
                "objective": float(objectives[-1]),
                "alu": str(prefix) + "-alu.json",
                "valu": str(prefix) + "-valu.json",
            }
        )
    return result


def stage(iteration: int) -> tuple[list[int], bool]:
    stages = [
        ([1, 2, 4, 6], True),
        ([1, 2, 4, 6], True),
        ([2, 4, 8, 16], False),
        ([8, 12, 16, 32], False),
        ([1, 2, 4, 6], False),
    ]
    return stages[iteration % len(stages)]


def write_checkpoint(path: Path, state: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alu", type=Path, required=True)
    parser.add_argument("--valu", type=Path, required=True)
    parser.add_argument("--maximum", type=int, required=True)
    parser.add_argument("--until", type=int, default=0)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, default=Path("/tmp/aopt-auto"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("/tmp/aopt-auto-checkpoint.json")
    )
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument(
        "--window",
        action="append",
        default=[],
        help=(
            "periodic engine:left:right repair window; repeat for parallel "
            "window candidates"
        ),
    )
    parser.add_argument("--window-period", type=int, default=6)
    parser.add_argument(
        "--exact-windows",
        action="store_true",
        help="use complete CP-SAT search rather than LNS for window stages",
    )
    parser.add_argument(
        "--hard-windows",
        action="store_true",
        help="solve windows as hard next-cap feasibility problems",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume the saved checkpoint instead of replacing it",
    )
    args = parser.parse_args()

    if args.resume and args.checkpoint.exists():
        state = json.loads(args.checkpoint.read_text())
        for engine in ("alu", "valu"):
            if not Path(str(state[engine])).exists():
                raise FileNotFoundError(
                    f"checkpoint {engine} schedule is missing: {state[engine]}"
                )
        print(
            f"drive_resume maximum={state['maximum']} "
            f"objective={state['objective']} accepted={state['accepted']}",
            flush=True,
        )
        state.setdefault("attempts", 0)
    else:
        state = {
            "alu": str(args.alu.resolve()),
            "valu": str(args.valu.resolve()),
            "maximum": args.maximum,
            "objective": float("inf"),
            "level_iteration": 0,
            "accepted": 0,
            "attempts": 0,
        }
        write_checkpoint(args.checkpoint, state)

    for global_iteration in range(args.max_iterations):
        current_maximum = int(state["maximum"])
        if current_maximum <= args.until:
            print(
                f"drive_complete maximum={current_maximum} "
                f"alu={state['alu']} valu={state['valu']}",
                flush=True,
            )
            return
        level_iteration = int(state["level_iteration"])
        attempt = int(state["attempts"])
        use_windows = bool(args.window) and (
            level_iteration % args.window_period == args.window_period - 1
        )
        radii, lns_only = (
            # Broad windows are diversification stages.  LNS preserves and
            # perturbs the incumbent inside those large neighborhoods;
            # full exact search spends most of its budget rebuilding an
            # equivalent feasible calendar before improving the soft cap.
            ([0] * len(args.window), not args.exact_windows)
            if use_windows
            else stage(level_iteration)
        )
        windows = args.window if use_windows else [""] * len(radii)
        print(
            f"drive_start global={global_iteration} maximum={current_maximum} "
            f"level_iteration={level_iteration} radii={radii} lns={int(lns_only)}",
            flush=True,
        )
        jobs = []
        for rank, (radius, window) in enumerate(zip(radii, windows)):
            candidate_prefix = Path(
                f"{args.prefix}-m{current_maximum}-i{level_iteration}"
                f"-a{attempt}-{'w' + str(rank) if window else 'r' + str(radius)}"
                f"-{'lns' if lns_only else 'cp'}"
            )
            jobs.append(
                {
                    "env_file": args.env_file.resolve(),
                    "alu": Path(str(state["alu"])),
                    "valu": Path(str(state["valu"])),
                    "prefix": candidate_prefix,
                    "radius": radius,
                    "window": window,
                    "seed": 10_007 + attempt * 101 + rank,
                    "lns_only": lns_only,
                    "hard_cap": args.hard_windows,
                    "time_limit": args.time_limit,
                    "workers": args.workers,
                }
            )
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            results = list(executor.map(lambda job: run_candidate(**job), jobs))
        state["attempts"] = attempt + 1
        feasible = [result for result in results if "maximum" in result]
        for result in results:
            print(
                f"drive_candidate radius={result['radius']} "
                f"window={result['window'] or '-'} "
                f"lns={int(bool(result['lns_only']))} "
                f"maximum={result.get('maximum', 'NA')} "
                f"objective={result.get('objective', 'NA')} "
                f"total={result.get('total', 'NA')}",
                flush=True,
            )
            if "maximum" not in result:
                print(str(result["output"])[-2000:], flush=True)
        if not feasible:
            state["level_iteration"] = level_iteration + 1
            write_checkpoint(args.checkpoint, state)
            continue
        best = min(
            feasible,
            key=lambda result: (
                int(result["maximum"]),
                float(result["objective"]),
                int(result["total"]),
                int(result["count"]),
            ),
        )
        incumbent_key = (
            current_maximum,
            float(state["objective"]),
            sys.maxsize,
            sys.maxsize,
        )
        candidate_key = (
            int(best["maximum"]),
            float(best["objective"]),
            int(best["total"]),
            int(best["count"]),
        )
        if candidate_key < incumbent_key:
            state.update(
                {
                    "alu": best["alu"],
                    "valu": best["valu"],
                    "maximum": best["maximum"],
                    "objective": best["objective"],
                    "total": best["total"],
                    "count": best["count"],
                    "accepted": int(state["accepted"]) + 1,
                }
            )
            if int(best["maximum"]) < current_maximum:
                state["objective"] = float("inf")
                state["level_iteration"] = 0
            else:
                state["level_iteration"] = level_iteration + 1
            print(
                f"drive_accept maximum={state['maximum']} "
                f"objective={best['objective']} total={best['total']} "
                f"alu={best['alu']} valu={best['valu']}",
                flush=True,
            )
        else:
            state["level_iteration"] = level_iteration + 1
        write_checkpoint(args.checkpoint, state)

    raise SystemExit("maximum driver iterations exhausted")


if __name__ == "__main__":
    main()
