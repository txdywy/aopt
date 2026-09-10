"""Exact affine-coordinate analysis of the original mirrored heap recurrence.

This proves only recurrences; it does not claim a faster executable kernel.
"""
import json
import z3

a,p=z3.BitVecs('a p',32)
next_a=2*a-5-p
results={}
for name,encoded,next_encoded,wanted in [
    ('positive_shift6',a-6,next_a-6,2*(a-6)+(1-p)),
    ('negative_shift5',5-a,5-next_a,2*(5-a)+p),
]:
    solver=z3.Solver()
    solver.add(z3.Or(p==0,p==1),next_encoded!=wanted)
    answer=str(solver.check())
    assert answer=='unsat',(name,answer)
    results[name]=answer
print(json.dumps({
    'uint32_equivalence':results,
    'caveat':'Encoded coordinates are not original memory addresses.',
    'positive_mapping':'original address a must be read at encoded address a-6',
    'negative_mapping':'5-a wraps to large uint32 for original tree addresses>=7',
    'relocation_lower_estimate':{
        'nodes':2047,'vector_width':8,'load_packets_ceil':256,'store_packets_ceil':256,
        'note':'Assumes all nodes moved with vector loads/stores; overlap with existing preprocessing must be counted separately.'
    }
},indent=2))
