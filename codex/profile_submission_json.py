"""Measure default-json payload, without expanding huge recipe tables."""
import base64
import importlib
import json
import zlib


def recipe_json_size(prefix,recipes,size=65536):
    count=len(prefix)+len(recipes)*size
    total=sum(len(json.dumps(p).encode()) for p in prefix)
    for packet,patches in recipes:
        total+=size*len(json.dumps(packet).encode())
        seen=set()
        for engine,si,ai,shift,values in patches:
            assert (engine,si,ai) not in seen
            seen.add((engine,si,ai))
            assert len(values)==16 and all(type(v)==int for v in values)
            assert size % (16<<shift)==0
            original=packet[engine][si][ai]
            total+=(size//16)*sum(len(str(v))-len(str(original)) for v in values)
    return total+2+max(0,count-1)*2


if __name__=='__main__':
    # Independently check the formula on all16 variants of a tiny recipe.
    packet={'alu':[['|',10,7,7]]}
    values=list(range(95,111))
    recipes=[(packet,[('alu',0,2,0,values)])]
    expanded=[{'alu':[['|',10,v,7]]} for v in values]
    assert recipe_json_size([],recipes,16)==len(json.dumps(expanded).encode())
    reports=[]
    for name in ('perf_takehome_983','perf_takehome_979'):
        mod=importlib.import_module('codex.'+name)
        if hasattr(mod,'_CODE'):
            prefix,recipes=json.loads(zlib.decompress(base64.b85decode(mod._CODE)))
            length=recipe_json_size(prefix,recipes)
            count=len(prefix)+65536*len(recipes)
            method='exact recipe length arithmetic; no huge JSON allocation'
        else:
            builder=mod.KernelBuilder();builder.build_kernel(10,2047,256,16)
            length=len(json.dumps(builder.instrs).encode())
            count=len(builder.instrs)
            method='actual json.dumps'
        reports.append({'file':name+'.py','packets':count,'json_bytes':length,
                        'json_mib':round(length/1048576,3),'method':method})
    print(json.dumps(reports,indent=2))
