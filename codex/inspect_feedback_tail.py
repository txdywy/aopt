"""Inspect assembled feedback candidate without replacing submission artifacts."""
import contextlib
import io
import json
from pathlib import Path
import shlex
import sys
from codex import experiment_paired_base as driver


def capture():
    class Captured(Exception):
        pass
    programs = []
    class CaptureMachine:
        def __init__(self, mem, program, debug, **kwargs):
            programs.append(program)
            raise Captured()
    driver.Machine = CaptureMachine
    command = json.loads(Path(__file__).with_name('table_981_result.json').read_text())['command']
    sys.argv = shlex.split(command)[1:] + ['--feedback-rounds','4','--feedback-percent','10']
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            driver.main()
    except Captured:
        pass
    return programs[0]


if __name__ == '__main__':
    combine = '--combine' in sys.argv
    program = capture()
    if not combine:
        print(json.dumps({'instructions':len(program),
                          'tail':[(i,program[i]) for i in range(966,981)]}))
    else:
        from codex.schedule_tail_window import schedule_tail
        prefix = [{e:list(s) for e,s in p.items()} for p in program[:1305]]
        moved = [('multiply_add',1079,567,135,87),('multiply_add',567,567,95,103)]
        mapping = {693:(437,685),694:(438,686),949:(453,685),950:(454,686)}
        count = 0
        for pc,packet in enumerate(prefix):
            if pc != 968 and packet.get('flow') != [('jump',969)]:
                continue
            for slot in moved: packet['valu'].remove(slot)
            packet['valu'] += [('^',429,685,677),('^',445,941,677)]
            packet['alu'] = [('^',mapping[s[1]][0],mapping[s[1]][1],s[2])
                             if s[0]=='|' and s[1] in mapping else s for s in packet['alu']]
            count += 1
        assert count == 65,count
        assert prefix[969]['flow'] == [('vselect',1191,1191,687,943)]
        prefix[969]['flow'] = [('vselect',679,1191,431,447)]
        prefix[970]['valu'].remove(('^',679,679,1191))
        prefix.insert(969,{'valu':moved})
        for slot in [('>>',1111,599,55),('^',599,599,79)]:
            prefix[970]['valu'].remove(slot)
            prefix[970].setdefault('alu',[]).extend(
                (slot[0],slot[1]+n,slot[2]+n,slot[3]+n) for n in range(8))
        result, certificate = schedule_tail(prefix,start=969,stop=981,horizon=10,
                                            allow_data_flow=True,timeout_ms=30000)
        assert result is not None,certificate
        result = result[:980]+result[981:]
        assert len(result)==1305
        program[:1305]=result
        import random
        from frozen_problem import Machine,Tree,Input,build_mem_image,reference_kernel2,N_CORES
        scores=[]
        for seed in range(103):
            random.seed(seed);tree=Tree.generate(10);inp=Input.generate(tree,256,16)
            def value(i):
                return random.getrandbits(32) if seed<100 else (0,0xffffffff,0xffffffff if i%2 else 0)[seed-100]
            tree.values=[value(i) for i in range(len(tree.values))]
            inp.values=[value(i) for i in range(256)]
            mem=build_mem_image(tree,inp)
            machine=Machine(mem,program,None,n_cores=N_CORES)
            machine.enable_pause=machine.enable_debug=False;machine.run()
            for ref in reference_kernel2(mem):pass
            assert machine.mem[ref[6]:ref[6]+256]==ref[ref[6]:ref[6]+256],seed
            assert machine.cycle==979,(seed,machine.cycle)
            scores.append(machine.cycle)
        print(json.dumps({'cycles':scores,'certificate':certificate}),flush=True)
        import types
        import runpy
        from problem import DebugInfo
        class VerifiedBuilder:
            def build_kernel(self,*shape):
                assert shape == (10,2047,256,16)
                self.instrs=program
            def debug_info(self):
                return DebugInfo(scratch_map={})
        module=types.ModuleType('perf_takehome')
        module.KernelBuilder=VerifiedBuilder
        sys.modules['perf_takehome']=module
        sys.argv=[str(Path(__file__).resolve().parents[1]/'tests/submission_tests.py')]
        runpy.run_path(sys.argv[0],run_name='__main__')
