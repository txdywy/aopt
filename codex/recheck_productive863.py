"""Re-evaluate swept configurations with productive lane accounting.

sweep_stage863 charges every lane of a vector op as work. When a group may run
as scalar lanes, lanes whose results nobody reads cost nothing, so the
productive work (as in scan_frontier863) can be lower. This recomputes
productive work and productive prefix/suffix waste for configurations whose
LOAD and FLOW stream bounds still allow 863. The result is still a necessary
lower bound, never a score.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import traceback


def recheck(changes):
    from functools import lru_cache
    from types import SimpleNamespace
    from codex import stride4_family as family
    from codex import scan_stage863 as scan
    from codex import research_stage864 as stage
    from codex import research_frontier863 as frontier
    from codex.packet_budget_bound import boundary, lane_weight
    from codex.packet_modes import expand
    from codex.word_ssa import observed, rebuild
    from codex.sweep_stage863 import DEEP
    if not getattr(family.capture, 'cache_info', None):
        family.capture = lru_cache(maxsize=1)(family.capture)
    if not hasattr(frontier, '_orig_build'):
        frontier._orig_build = frontier.build
    deep = {k: v for k, v in changes.items() if k in DEEP}

    def patched(args, _deep=deep):
        for k, v in _deep.items():
            setattr(args, k, v)
        return frontier._orig_build(args)
    frontier.build = patched
    try:
        config = scan.baseline(**{k: v for k, v in changes.items() if k not in DEEP})
        ir, _ = stage.build(SimpleNamespace(**config))
        expanded = expand(ir)
        expanded.views = [a for _, aa, _, _ in expanded.groups for a in aa]
        rebuild(expanded)
        needed = observed(expanded)
        productive = sum(lane_weight(expanded, j, True, needed) for j in range(len(expanded.ops)))
        head = boundary(expanded, 6, False, 4, 2, True)
        tail = boundary(expanded, 3, True, 3, 2, True)
        eff = productive + head['waste_lower_bound'] + tail['waste_lower_bound']
        return dict(changes=changes, ok=True, productive=productive,
                    head=head['waste_lower_bound'], tail=tail['waste_lower_bound'],
                    effective=eff, bound=(eff + 59) // 60)
    except Exception as exc:
        return dict(changes=changes, ok=False, error=f'{type(exc).__name__}: {exc}'[:160],
                    where=traceback.format_exc().strip().splitlines()[-2][:160])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('inputs', nargs='+', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--all', action='store_true', help='include stream-infeasible configurations')
    args = ap.parse_args()
    rows = [json.loads(line) for f in args.inputs for line in f.open()]
    keep = [r['changes'] for r in rows if r['ok'] and (args.all or (
        r['streams']['load'] <= 863 and r['streams']['flow'] <= 863))]
    keep.append({})  # baseline control
    print(json.dumps(dict(configs=len(keep))), flush=True)
    with args.output.open('w') as out, ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(recheck, keep):
            out.write(json.dumps(row) + '\n')
            out.flush()
            if row.get('ok') and row['bound'] <= 863:
                print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
