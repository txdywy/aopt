"""Optimistic whole-level relocation screen, not a global kernel lower bound."""
import itertools
import json

# Measured gather_address DAG lane counts for the low-load991 configuration.
# Depth4 includes both rounds4 and15. Existing preprocessing visits depths4..7.
savings={4:160,5:256,6:256,7:256,8:256,9:256,10:256}
extra_loads={d:0 if d<=7 else (1<<d)//8 for d in savings}
budget=899*2-1791
rows=[]
for mask in itertools.product((False,True),repeat=7):
    depths=[d for d,on in zip(savings,mask) if on]
    loads=sum(extra_loads[d] for d in depths)
    if loads<=budget:
        rows.append((sum(savings[d] for d in depths),depths,loads))
best=max(rows)
assert best==(928,[4,5,6,7],0)
print(json.dumps({'target_cycles':899,'spare_load_slots':budget,
                  'best_address_work_saved':best[0],'depths':best[1],
                  'remaining_arithmetic_deficit':2191-best[0],
                  'assumptions':['whole levels only','all existing preprocessing loads reusable',
                                 'no extra conversion, parity, copy or store cost',
                                 'only gather_address removal credited'],
                  'scope':'Optimistic screen of this relocation family, not all address algorithms.'},indent=2))
