"""Optimistic enumeration of a *limited* uniform-chain exchange menu.

This is not a bound for all kernels. It grants free roots, free dead-code
savings and perfect placement; even long chains not assembled in a candidate
are admitted. Boundary losses are explicitly conditional on the old graph.
"""
from functools import lru_cache
import argparse
import json
from pathlib import Path


@lru_cache(None)
def partitions(n, low=2):
    """Partition n into singleton thirds and nondecreasing chains >= 2."""
    result=[(n,())]
    for size in range(low,n+1):
        for singles,chains in partitions(n-size,size):
            result.append((singles,(size,)+chains))
    return tuple(result)


def enumerate_menu(allow_joint=True):
    best=-1;witnesses=[];cases=0
    for changed in range(12):
        roots=max(0,changed-6)
        for root_load in range(roots+1):
            for root_flow in range(roots-root_load+1):
                root_alu=roots-root_load-root_flow
                for staged in range(min(changed,4-root_load)+1):
                    loads=staged+root_load
                    flow_trades=min(loads,3)
                    fourth=int(loads==4)
                    for joint in range(changed-staged+1 if allow_joint else 1):
                        for thirds,chains in partitions(changed-staged-joint):
                            flow_ops=(flow_trades+root_flow+joint+thirds+
                                      sum(size-1 for size in chains))
                            fused=max(0,(flow_ops-1)//2)
                            saving=(8*staged+6*thirds+sum(7*n-4 for n in chains)
                                    +(7*joint+2 if joint else 0)+8-root_alu-fourth-8*fused)
                            row=dict(changed=changed,roots=roots,root_load=root_load,
                                root_flow=root_flow,root_alu=root_alu,staged=staged,
                                joint_constants=joint,thirds=thirds,chains=list(chains),
                                extra_flow=flow_ops,fused_digits=fused,saving=saving)
                            cases+=1
                            if saving>best:best=saving;witnesses=[]
                            if saving==best:witnesses.append(row)
    return dict(cases=cases,maximum_saving=best,witnesses=witnesses)


def report():
    no_joint=enumerate_menu(False);joint=enumerate_menu(True)
    return dict(without_joint=no_joint,with_joint=joint,
        baseline_arithmetic=51665,conditional_old_prefix_waste=52,
        conditional_old_suffix_waste=116,
        conditional_cycle_lower_bound=(51665-joint['maximum_saving']+52+116+59)//60,
        new_executable=False,
        scope='A limited menu with optimistic idealizations; no universal lower bound. '
              'The 52/116 boundary losses apply only if independently preserved.',
        assumptions=[
            'At most eleven original native uniforms are changed.',
            'At least max(0,changed-6) independent roots use one ALU/FLOW/LOAD each; six are granted free.',
            'A staged uniform saves 8 arithmetic lanes and consumes a LOAD exchange.',
            'A third uniform saves 6 lanes and uses 1 FLOW.',
            'A normal m-uniform chain is granted cost m+4 lanes and m-1 FLOWs.',
            'A self-masked ones-plus-k chain is granted saving 7k+2 lanes and k FLOWs.',
            'At most four LOAD exchanges; the fourth costs one arithmetic lane.',
            'Two additional net FLOWs are allowed; each digit fusion refunds two at cost 8 lanes.',
            'An extra 8-lane dead-uniform saving is granted unconditionally.',
            'No startup, layout, liveness, instruction-order or scratch penalties are charged.',
            'New record layouts and changes to the hash, boundary or LOAD/FLOW streams are outside the menu.'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();result=report()
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('conditional_cycle_lower_bound','new_executable')} |
                     {'without_joint':result['without_joint']['maximum_saving'],
                      'with_joint':result['with_joint']['maximum_saving']}))


if __name__=='__main__':main()
