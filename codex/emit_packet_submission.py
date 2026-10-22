"""Package an independently verified ISA program as a standalone submission."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import textwrap
import zlib

from codex.verify_compact import validate_isa


def emit(program,path,cycles):
    assert len(program) == cycles, (len(program), cycles)
    counts,maximum=validate_isa(program)
    raw=json.dumps(program,separators=(',',':')).encode()
    digest=hashlib.sha256(raw).hexdigest()
    encoded=base64.b85encode(zlib.compress(raw,9)).decode()
    payload='\n'.join('    '+repr(line) for line in textwrap.wrap(encoded,100,break_on_hyphens=False))
    source=f'''"""Standalone {cycles}-cycle fixed-shape VLIW/SIMD kernel.

Workload: height10, 2047 nodes, 256 zero-initialized indices, 16 rounds;
the Without Indices contract compares final values. Consumed header/tree/
index memory and loaded input-value slots hold tree records and anchor banks.

All runtime tree transforms, addressing and hashes execute as original ISA
instructions. The compressed payload contains instructions only. Building
requires Python's standard library and the original problem module.

The readable compiler and search tools are in codex/stride4_family.py,
codex/research_sub864.py, codex/research_bootstrap864.py,
codex/research_stage864.py, codex/word_ssa.py and codex/packet_modes.py.
Signed-parity bilinear addresses
and stride-four records were informed by Josu San Martin's public research:
https://github.com/josusanmartin/josusanmartin.github.io/blob/master/vliw/chat.log
The preserved Sakauma 880 kernel supplied initial scheduling priorities:
https://github.com/Sakauma/my_original_performance_takehome/commit/cb2e0554735ef7d0dfde1a04bc2c12c4bb2ed38d

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
publish or redistribute solutions, consistent with the original source notice.
"""
import base64
from functools import lru_cache
import hashlib
import json
import zlib

from problem import DebugInfo, N_CORES, SCRATCH_SIZE, SLOT_LIMITS, VLEN

EXPECTED_CYCLES = {cycles}
PROGRAM_SHA256 = {digest!r}
SCRATCH_WORDS = {maximum+1}

_PAYLOAD = (
{payload}
)


@lru_cache(maxsize=1)
def _program():
    raw = zlib.decompress(base64.b85decode(_PAYLOAD))
    assert hashlib.sha256(raw).hexdigest() == PROGRAM_SHA256
    program = json.loads(raw)
    assert len(program) == EXPECTED_CYCLES
    return program


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch_debug = {{}}

    def build_kernel(self, forest_height, n_nodes, batch_size, rounds):
        assert (forest_height, n_nodes, batch_size, rounds) == (10, 2047, 256, 16)
        assert (N_CORES, SCRATCH_SIZE, VLEN) == (1, 1536, 8)
        assert {{e: SLOT_LIMITS[e] for e in ('alu','valu','load','store','flow')}} == {{
            'alu':12, 'valu':6, 'load':2, 'store':2, 'flow':1,
        }}
        self.instrs = [{{e: [tuple(slot) for slot in slots] for e, slots in packet.items()}}
                       for packet in _program()]
        self.scratch_ptr = self.ssa_peak = SCRATCH_WORDS

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)
'''
    path.write_text(source)
    return dict(cycles=cycles,source_bytes=path.stat().st_size,
                json_bytes=len(json.dumps(program).encode()),scratch_words=maximum+1,
                operations=counts,program_sha256=digest)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('program',type=Path);ap.add_argument('output',type=Path)
    ap.add_argument('--cycles',type=int,required=True)
    args=ap.parse_args()
    print(json.dumps(emit(json.loads(args.program.read_text()),args.output,args.cycles),indent=2))


if __name__=='__main__':main()
