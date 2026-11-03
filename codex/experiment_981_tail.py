"""Speculate the final node XOR into the last special table packet."""
import base64
import json
import random
from pathlib import Path
import sys
import zlib
from codex.perf_takehome_981 import _CODE
from codex.schedule_tail_window import schedule_tail


def prepare():
    prefix, recipes = json.loads(zlib.decompress(base64.b85decode(_CODE)))
    prefix = [{e: [tuple(s) for s in slots] for e, slots in p.items()} for p in prefix]
    moved = [('multiply_add', 1079, 567, 135, 87), ('multiply_add', 567, 567, 95, 103)]
    def rewrite(packet):
        for slot in moved:
            packet['valu'].remove(slot)
        packet['valu'] += [('^',429,685,677), ('^',445,941,677)]
        mapping = {693:(437,685), 694:(438,686), 949:(453,685), 950:(454,686)}
        packet['alu'] = [('^',mapping[s[1]][0],mapping[s[1]][1],s[2])
                         if s[0]=='|' and s[1] in mapping else s for s in packet['alu']]
    rewritten = 0
    for pc, packet in enumerate(prefix):
        if pc == 969 or packet.get('flow') == [('jump',970)]:
            rewrite(packet)
            rewritten += 1
    assert rewritten == 65, rewritten
    assert prefix[970]['flow'] == [('vselect',1191,1191,687,943)]
    prefix[970]['flow'] = [('vselect',679,1191,431,447)]
    prefix[971]['valu'].remove(('^',679,679,1191))
    # A virtual earlier packet establishes the moved producers before their
    # existing consumers. The scheduler may then pack them legally; placing
    # them in the consumer's original packet would assert false same-cycle reads.
    prefix.insert(970, {'valu': moved})
    return prefix, recipes


if __name__ == '__main__':
    prefix, recipes = prepare()
    for slot in [('>>',1111,599,55), ('^',599,599,79)]:
        prefix[971]['valu'].remove(slot)
        prefix[971].setdefault('alu', []).extend(
            (slot[0],slot[1]+lane,slot[2]+lane,slot[3]+lane) for lane in range(8))
    result, certificate = schedule_tail(prefix, start=970, stop=982, horizon=10,
                                        allow_data_flow=True, timeout_ms=30000)
    assert result is not None, certificate
    # Delete the virtual producer packet's extra space, preserving every
    # absolute table address. Packet980 is unreachable after the new halt979.
    result = result[:981] + result[982:]
    assert len(result) == 1306
    payload = base64.b85encode(zlib.compress(json.dumps([result,recipes],separators=(',',':')).encode(),9)).decode()
    source = Path(__file__).with_name('perf_takehome_981.py').read_text()
    source = '\n'.join('_CODE = '+repr(payload) if line.startswith('_CODE = ') else line
                       for line in source.splitlines())+'\n'
    source = source.replace('Standalone 981-cycle', 'Standalone 980-cycle')
    namespace = {'__name__':'standalone_candidate'}
    exec(source, namespace)
    builder = namespace['KernelBuilder']()
    builder.build_kernel(10,2047,256,16)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
    from frozen_problem import Machine,Tree,Input,build_mem_image,reference_kernel2,N_CORES
    for seed in range(103):
        random.seed(seed)
        tree=Tree.generate(10); inp=Input.generate(tree,256,16)
        def value(i):
            return random.getrandbits(32) if seed<100 else (0,0xffffffff,0xffffffff if i%2 else 0)[seed-100]
        tree.values=[value(i) for i in range(len(tree.values))]
        inp.values=[value(i) for i in range(256)]
        mem=build_mem_image(tree,inp)
        machine=Machine(mem,builder.instrs,builder.debug_info(),n_cores=N_CORES)
        machine.enable_pause=machine.enable_debug=False
        machine.run()
        for ref in reference_kernel2(mem):pass
        assert machine.mem[ref[6]:ref[6]+256]==ref[ref[6]:ref[6]+256], seed
        assert machine.cycle==980,(seed,machine.cycle)
    if '--export' in sys.argv:
        print('*** Begin Patch\n*** Add File: /Users/yiwei/x/aopt/codex/perf_takehome_980.py')
        print('\n'.join('+'+line for line in source.splitlines()))
        print('*** End Patch')
    else:
        print(json.dumps({'cycles':980,'full32_seeds':100,'boundary_patterns':3,
                          'certificate':certificate}),flush=True)
