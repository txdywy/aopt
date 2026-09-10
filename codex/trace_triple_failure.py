"""Replay failed arity3 prototype with reaching-definition diagnostics only."""
import json
from pathlib import Path
import runpy
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
import frozen_problem as frozen
from codex.fuse_table_returns import accesses

old_step = frozen.Machine.step
last = {}


def trace(self, instr, core):
    pending = []
    for engine, slots in instr.items():
        for slot in slots:
            if engine == 'flow' and slot[0] in ('jump', 'pause', 'halt'): continue
            reads, writes = accesses(engine, slot)
            node = {'pc': core.pc-1, 'slot': slot,
                    'inputs': [(r, core.scratch[r], last.get(r)) for r in sorted(reads)]}
            pending.extend((w, node) for w in writes)
    try:
        old_step(self, instr, core)
    except IndexError:
        def simplify(node, depth=2):
            if node is None: return None
            return {'pc':node['pc'], 'slot':node['slot'],
                    'inputs':[(r,v,simplify(parent,depth-1) if depth else None)
                              for r,v,parent in node['inputs']]}
        for slot in instr.get('load', []):
            if slot[0] == 'load_offset':
                src = slot[2]+slot[3]
                if core.scratch[src] >= len(self.mem):
                    print(json.dumps({'failing_pc':core.pc-1, 'load':slot,
                                      'definition':simplify(last.get(src))}),flush=True)
        raise
    for word,node in pending: last[word] = node


frozen.Machine.step = trace
args = shlex.split(json.loads(Path('codex/table_981_result.json').read_text())['command'])[1:]
args[args.index('--arity')+1] = '3'
args.remove('--vector-index')
args[args.index('--seeds')+1] = '1'
sys.argv = args
runpy.run_path(args[0], run_name='__main__')
