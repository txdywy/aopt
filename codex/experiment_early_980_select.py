"""Free FLOW951 with scalar selection; test earlier group31 node readiness."""
import base64
import json
import random
import zlib
from pathlib import Path
import sys
from codex import perf_takehome_980 as base


def main():
    prefix, recipes = json.loads(zlib.decompress(base64.b85decode(base._CODE)))
    # Dead output storage: check no remaining reads/writes in the main stream
    # before the existing speculative tail reclaims these words at969.
    from codex.fuse_table_returns import accesses, LIMITS
    reserved = set(range(431,432)) | set(range(439,447))
    for packet in prefix[947:969]:
        for engine, slots in packet.items():
            for slot in slots:
                if slot[0] in ('jump','halt'): continue
                reads,writes=accesses(engine,slot)
                assert not reserved & (reads|writes), slot
    prefix[947].setdefault('alu',[]).append(['-',431,247,279])
    prefix[950]['alu'] += [['*',711+n,687+n,431] for n in range(8)]
    prefix[953]['alu'] += [['+',711+n,711+n,279] for n in range(8)]
    assert prefix[951]['flow']==[['vselect',711,687,247,279]]
    prefix[951]['flow']=[['vselect',439,687,263,295]]
    prefix[957]['flow']=[['vselect',439,943,711,439]]
    prefix[958]['flow']=[['vselect',1191,703,719,439]]
    del prefix[959]['flow']
    for packet in prefix:
        assert all(len(slots)<=LIMITS[engine] for engine,slots in packet.items())
    # These packets are not copied into any out-of-line entry in this recipe.
    for packet, patches in recipes:
        flow = packet.get('flow', [])
        assert len(flow)==1 and flow[0][0]=='jump', flow
        if not 947 <= flow[0][1]-1 < 969:
            continue
        for engine,slots in packet.items():
            for slot in slots:
                if slot[0] in ('jump','halt'):continue
                reads,writes=accesses(engine,slot)
                assert not reserved & (reads|writes), slot
    if '--advance-hash' in sys.argv:
        prefix[959]['valu'].remove(['^',591,591,1103])
        prefix[959]['valu'].append(['^',679,679,1191])
        for lane in range(8):
            prefix[960]['alu'].remove(['^',679+lane,679+lane,1191+lane])
            prefix[960]['alu'].append(['^',591+lane,591+lane,1103+lane])
        prefix[960]['valu'].remove(['multiply_add',591,591,143,111])
        prefix[960]['valu'].append(['multiply_add',679,679,127,71])
        targets=[prefix[961]]+[packet for packet,_ in recipes
                              if packet.get('flow')==[['jump',962]]]
        assert len(targets)==2
        for packet in targets:
            packet['valu'].remove(['multiply_add',679,679,127,71])
            packet['valu'].append(['multiply_add',591,591,143,111])
        if '--advance-front' in sys.argv:
            def packets_at(pc):
                return [prefix[pc]] + [packet for packet,_ in recipes
                                      if packet.get('flow')==[['jump',pc+1]]] + [
                    packet for packet in prefix[981:]
                    if packet.get('flow')==[['jump',pc+1]]]
            def move(slot, old, new, engine='valu'):
                for packet in packets_at(old): packet[engine].remove(slot)
                for packet in packets_at(new): packet.setdefault(engine,[]).append(slot)
            move(['^',639,639,1151],961,962)
            move(['multiply_add',583,583,143,111],961,963)
            move(['>>',1191,679,55],962,961)
            move(['^',679,679,79],962,961)
            move(['^',679,679,1191],963,962)
            if '--advance-middle' in sys.argv:
                move(['multiply_add',583,583,143,111],963,964)
                move(['^',551,551,1063],963,964)
                move(['vstore',15,551],964,965,'store')
                move(['multiply_add',1191,679,135,87],964,963)
                move(['multiply_add',679,679,95,103],964,963)
                move(['^',575,575,1087],964,965)
                move(['^',679,679,1191],965,964)
                move(['^',583,583,119],965,966)
                move(['multiply_add',679,679,143,111],966,965)
                move(['^',607,607,79],966,967)
                move(['>>',1191,679,63],967,966)
                move(['^',591,591,1103],967,968)
                move(['^',679,679,1191],968,967)
                if '--advance-candidates' in sys.argv:
                    move(['^',567,567,1079],968,969)
                    move(['^',639,639,119],968,969)
                    move(['&',1191,679,23],969,968)
                    move(['^',429,685,677],969,968)
                    move(['^',445,941,677],969,968)
                    for packet in packets_at(968):
                        packet['valu'].remove(['^',429,685,677])
                        packet['valu'].remove(['^',445,941,677])
                        # Full aligned writes avoid clobbering cached nodes429/430
                        # before table969 reads them. Its last two lanes replace
                        # the provisional values in437/438 and453/454 next cycle.
                        packet['valu'].extend([['^',431,687,679],['^',447,943,679]])
                        packet['valu'].remove(['^',591,591,1103])
                        packet.setdefault('alu',[]).extend(
                            [['^',591+n,591+n,1103+n] for n in range(8)])
        for pc, packet in enumerate(prefix+[p for p,_ in recipes]):
            assert all(len(slots)<=LIMITS[engine] for engine,slots in packet.items()), (pc,packet)
    base._CODE=base64.b85encode(zlib.compress(json.dumps([prefix,recipes]).encode())).decode()
    builder=base.KernelBuilder();builder.build_kernel(10,2047,256,16)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
    from frozen_problem import Machine,Tree,Input,build_mem_image,reference_kernel2,N_CORES
    scores=[]
    for seed in range(23):
        random.seed(seed);tree=Tree.generate(10);inp=Input.generate(tree,256,16)
        def value(i):
            return random.getrandbits(32) if seed<20 else (0,0xffffffff,0xffffffff if i%2 else 0)[seed-20]
        tree.values=[value(i) for i in range(len(tree.values))]
        inp.values=[value(i) for i in range(len(inp.values))]
        mem=build_mem_image(tree,inp)
        machine=Machine(mem,builder.instrs,builder.debug_info(),n_cores=N_CORES)
        machine.enable_pause=machine.enable_debug=False;machine.run()
        for ref in reference_kernel2(mem):pass
        assert machine.mem[ref[6]:ref[6]+256]==ref[ref[6]:ref[6]+256],seed
        scores.append(machine.cycle)
    print(json.dumps({'cycles':scores,'extra_alu':17,'flow_removed':1,
                      'node_select_pc':958,'original_node_select_pc':959,
                      'early_candidates':'--advance-candidates' in sys.argv,
                      'first_hash_mad_pc':960 if '--advance-hash' in sys.argv else 961}))


if __name__=='__main__':main()
