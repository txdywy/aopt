"""Optimistic resource ceiling for the address/uniform exchange menu.

This is a bounded menu model, not a universal kernel bound. It even grants
free 4097 roots and perfect placement; startup delays only worsen the result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def enumerate_menu():
    best = -1
    witnesses = []
    cases = 0
    for changed in range(12):
        # Eleven native uniform vectors: ten constants plus the initial root.
        # Grant six free independent scalars, including 4097, which actually
        # needs construction. At least m-6 other roots remain necessary.
        roots = max(0,changed-6)
        for root_load in range(roots+1):
            for root_flow in range(roots-root_load+1):
                root_alu = roots-root_load-root_flow
                for stages in range(min(changed,4-root_load)+1):
                    literals = stages+root_load
                    trade_flow = min(literals,3)
                    trade_alu = int(literals == 4)
                    for pairs in range((changed-stages)//2+1):
                        thirds = changed-stages-2*pairs
                        new_flow = trade_flow+root_flow+pairs+thirds
                        fused = max(0,(new_flow-1)//2)
                        # Grant the complete dead-256 saving unconditionally,
                        # even without selecting/reconstructing 4097.
                        saved = (8*changed+8-root_alu-trade_alu
                                 -6*pairs-2*thirds-8*fused)
                        row = dict(changed=changed,roots=roots,root_load=root_load,
                            root_flow=root_flow,root_alu=root_alu,stages=stages,
                            pairs=pairs,thirds=thirds,fused_digits=fused,
                            new_flow=new_flow,literal_exchanges=literals,
                            arithmetic_saving=saved)
                        cases += 1
                        if saved > best:
                            best,witnesses = saved,[row]
                        elif saved == best:
                            witnesses.append(row)
    return dict(model='optimistic native-uniform materialization + d7 digit fusion',
        cases=cases,maximum_arithmetic_saving=best,witnesses=witnesses,
        baseline_arithmetic=51665,minimum_arithmetic_in_menu=51665-best,
        first_cycle_waste=52,final_three_cycle_waste=116,
        cycle_lower_bound=(51665-best+52+116+59)//60,
        required_saving_for_863=53,new_executable=False,
        assumptions=[
            'Only the eleven existing native uniform vectors may change; ones stays native.',
            'A uniform uses native 8C, STORE/reload 1L, third pack 2C+1F, or paired pack 6C+1F per pair.',
            'At least max(0,changed-6) independent roots require one C/F/L each; 4097 root is granted free.',
            'Three existing literal LOADs can become FLOW; the fourth becomes a one-ALU input pointer.',
            'Critical LOAD capacity allows no net additional LOAD; each load root or stage needs an exchange.',
            'The unchanged final FLOW dependencies allow two net additional FLOW at horizon 863.',
            'Each d7 digit fusion costs 8C and refunds 2F.',
            'Grant an extra 8C saving for deleting vector 256 unconditionally.',
            'All new work is placed perfectly with no startup penalty beyond the existing 52/116.',
            'Changing record layouts, eliminating anchor packs, new hash identities, and new primitives are outside this menu.'])


def independent_cp_optimum():
    """A separate integer model checks the enumeration and rounding edges."""
    from ortools.sat.python import cp_model
    model = cp_model.CpModel()
    changed,stages,pairs,thirds,ra,rf,rl,fused = [model.NewIntVar(0,11,n)
        for n in ('changed','stages','pairs','thirds','ra','rf','rl','fused')]
    used, fourth = model.NewBoolVar('used'),model.NewBoolVar('fourth')
    loads,trade_flow = model.NewIntVar(0,4,'loads'),model.NewIntVar(0,3,'trade_flow')
    model.Add(changed <= 11*used)
    model.Add(changed >= used)
    model.Add(stages+2*pairs+thirds == changed)
    model.Add(ra+rf+rl >= changed-6)
    model.Add(loads == stages+rl)
    model.Add(loads == 4).OnlyEnforceIf(fourth)
    model.Add(loads <= 3).OnlyEnforceIf(fourth.Not())
    model.Add(trade_flow == loads-fourth)
    model.Add(trade_flow+rf+pairs+thirds <= 2+2*fused)
    saving = 8*changed+8-ra-fourth-6*pairs-2*thirds-8*fused
    model.Maximize(saving)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5
    solver.parameters.num_search_workers = 1
    status = solver.Solve(model)
    return dict(status=solver.StatusName(status),objective=solver.ObjectiveValue(),
                upper_bound=solver.BestObjectiveBound())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args = ap.parse_args()
    result = enumerate_menu()
    result['independent_cp'] = independent_cp_optimum()
    assert result['maximum_arithmetic_saving'] == result['independent_cp']['upper_bound']
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('cases','maximum_arithmetic_saving','cycle_lower_bound','independent_cp')}))


if __name__ == '__main__':
    main()
