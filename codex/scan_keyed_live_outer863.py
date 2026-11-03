"""Search two MADs followed by XOR with arbitrary keyed earlier live inputs."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import Counter
import argparse
import hashlib
from itertools import combinations_with_replacement
import json
from pathlib import Path
import subprocess
import sys

from codex.scan_keyed_live_nonlinear863 import first_forms


def cases():
    return [dict(first_source=s)for s in('a','keyed')]+[
        dict(first_operands=role)for _,role in first_forms()]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seconds',type=float,default=5)
    ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--refine',type=Path)
    args=ap.parse_args()
    jobs=cases();previous=None
    covered=[]
    if args.refine:
        old=json.loads(args.refine.read_text())
        pending=[r for r in old['results']if r['result']=='unknown']
        assert len(pending)==1 and pending[0]['first_source']=='keyed'
        jobs=[]
        for a,b in combinations_with_replacement(range(6),2):
            for c in range(6):
                if not set((a,b,c)).intersection((0,1)):
                    covered.append([a,b,c]);continue
                jobs.append(dict(first_source='keyed',second_operands=(a,b,c)))
        proof=Path(__file__).with_name('free_keys863_complete_grammar.json')
        previous=dict(path=args.refine.name,sha256=hashlib.sha256(args.refine.read_bytes()).hexdigest(),
            single_input_proof=proof.name,single_input_sha256=hashlib.sha256(proof.read_bytes()).hexdigest())
    report=dict(width=11,total_queries=len(jobs),new_executable=False,results=[],
        refinement=previous,second_roles_covered_by_single_input_proof=covered,
        scope='MAD/MAD/XOR, last XOR combines the two MAD outputs; first affine in a/keyed '
              'or one of 28 mixed/product roles, arbitrary input/output keys and second MAD. '
              'Low-a domain; UNKNOWN stays open and SAT needs full32 checking.')
    def run(case):
        code=('from codex.search_keyed_live_middle863 import search;import json;'
              f'print(json.dumps(search(("mad","mad","^"),11,{args.seconds!r},'
              f'{int(args.seconds*990)!r},last_operands=(3,4,0),**{case!r})))')
        done=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        if done.returncode:raise RuntimeError(done.stderr)
        return json.loads(done.stdout)
    with ThreadPoolExecutor(max_workers=args.workers)as pool:
        futures=[pool.submit(run,case)for case in jobs]
        for future in as_completed(futures):
            report['results'].append(future.result())
            report['counts']=dict(Counter(r['result']for r in report['results']))
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            if len(report['results'])%5==0:
                print(json.dumps(dict(done=len(report['results']),counts=report['counts'])),flush=True)


if __name__=='__main__':main()
