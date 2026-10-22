"""Extend a packed prefix by small joint stream/compute CP windows.

The output is an ISA calendar, not a score until packet_modes check performs
allocation and frozen-machine replay. Failed windows leave the last valid
calendar intact. Optional OR-Tools is needed only by the child solver.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import subprocess
import sys


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prefix',type=Path,required=True)
    ap.add_argument('--output-prefix',type=Path,required=True)
    ap.add_argument('--start',type=int,required=True)
    ap.add_argument('--stop',type=int,default=800)
    ap.add_argument('--step',type=int,default=30)
    ap.add_argument('--overlap',type=int,default=10)
    ap.add_argument('--radius',type=int,default=4)
    ap.add_argument('--seconds',type=float,default=40)
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--seed',type=int,default=1)
    ap.add_argument('--extra-waste',type=int,default=0,
                    help='Total additional arithmetic waste allowed across all rolling steps')
    args=ap.parse_args()
    assert args.start<args.stop and args.step>0 and args.overlap>0
    source=str(args.prefix);dest=str(args.output_prefix)
    assert Path(source).resolve()!=Path(dest).resolve()
    Path(dest+'.pkl').write_bytes(Path(source+'.pkl').read_bytes())
    Path(dest+'.sol').write_bytes(Path(source+'.sol').read_bytes())
    ir=pickle.loads(Path(dest+'.pkl').read_bytes())
    def stats(path,boundary):
        data=list(map(int,path.read_text().split()));n,ng=data[:2]
        assert (n,ng)==(len(ir.ops),len(ir.groups))
        times=data[2:2+n]
        lanes=Counter()
        for j,op in enumerate(ir.ops):
            if op.engine in ('alu','valu'):
                lanes[times[j]]+=8 if op.engine=='valu' else 1
        return max(times)+1,60*boundary-sum(lanes[t] for t in range(boundary))
    horizon,waste=stats(Path(dest+'.sol'),args.start)
    current_waste=waste
    boundary=args.start;steps=[]
    while boundary<args.stop:
        target=min(args.stop,boundary+args.step)
        high=min(horizon,target+args.overlap)
        output=Path(f'{dest}-{target}.sol')
        if output.exists():
            raise FileExistsError(output)
        command=[sys.executable,'-m','codex.exact_compute_packets','--prefix',dest,
                 '--output',str(output),'--lo',str(boundary),'--hi',str(high),
                 '--horizon',str(horizon),'--radius',str(args.radius),'--move-streams',
                 '--pack-before',str(target),'--max-waste',str(waste+args.extra_waste-current_waste),
                 '--seconds',str(args.seconds),
                 '--workers',str(args.workers),'--seed',str(args.seed+target)]
        print({'rolling_window':[boundary,high],'pack_before':target},flush=True)
        result=subprocess.run(command,check=False)
        if result.returncode or not output.exists():
            print({'stopped_at':boundary,'unrealized_target':target},flush=True)
            break
        h,actual=stats(output,target)
        assert h==horizon and current_waste<=actual<=waste+args.extra_waste,(h,actual,horizon,waste)
        current_waste=actual
        Path(dest+'.sol').write_bytes(output.read_bytes())
        steps.append({'lo':boundary,'hi':high,'packed_before':target,'prefix_waste':actual})
        boundary=target
        Path(dest+'.progress.json').write_text(json.dumps(dict(
            horizon=horizon,packed_before=boundary,prefix_waste=current_waste,steps=steps,
            scope='Calendar validated by each solver; requires independent allocation and replay.'),indent=2)+'\n')
    print({'horizon':horizon,'packed_before':boundary,'prefix_waste':current_waste,
           'calendar':dest+'.sol'},flush=True)


if __name__=='__main__':main()
