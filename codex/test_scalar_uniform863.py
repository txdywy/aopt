"""Guard mode legality and reproducibility of the scalar-uniform graphs."""
from copy import deepcopy
import hashlib
import json
import unittest

from codex.research_scalar_uniform863 import build
from codex.packet_modes import expand


def fingerprint(ir):
    rows=[(op.engine,op.code,op.args,op.tag)for op in ir.ops]
    return hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()


class ScalarUniformTests(unittest.TestCase):
    def test_scalar_sources_cannot_be_remerged_as_vector_operands(self):
        ir,_,converted=build(full=['cache0',0xb55a4f09,-2,0xc761c23c])
        self.assertEqual([32,96,96,512],[row['converted_vectors']for row in converted])
        expand(ir)
        grouped={j for _,_,ids,_ in ir.groups for j in ids}
        scalar_ids={j for j,op in enumerate(ir.ops)if op.tag[0]=='scalar_uniform_use'}
        self.assertEqual(8*(32+96+96+512),len(scalar_ids))
        self.assertFalse(grouped&scalar_ids)

    def test_saved_json_config_rebuilds_identical_graph(self):
        cfg=dict(full=['cache0',0xb55a4f09,-2,0xc761c23c],
            methods={9:'stage',19:'stage',3:'third'},prolog19=8,init4097='flow')
        ir,_,converted=build(**cfg)
        replay,_,again=build(**json.loads(json.dumps(cfg)))
        self.assertEqual(fingerprint(ir),fingerprint(replay))
        self.assertEqual(converted,again)
        self.assertEqual(8,converted[-1]['converted_vectors'])
        self.assertTrue(converted[-1]['partial'])

    def test_removed_constant_epochs_have_no_dangling_consumers(self):
        ir,_,_=build(full=['cache0',0xb55a4f09])
        reloads=[op for op in ir.ops if op.tag[0]=='constant_reload']
        self.assertNotIn(('constant_reload',(238,)),[op.tag for op in reloads])
        self.assertNotIn(('constant_reload',(1215,)),[op.tag for op in reloads])
        for j,deps in enumerate(ir.deps):
            self.assertNotIn(j,deps)
        expanded=deepcopy(ir);expand(expanded)
        # The expanded mode graph must preserve both removed memory epochs.
        self.assertEqual(sum(op.engine=='load'for op in ir.ops),
                         sum(op.engine=='load'for op in expanded.ops))


if __name__=='__main__':unittest.main()
