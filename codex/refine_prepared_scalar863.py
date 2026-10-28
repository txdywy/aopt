"""Independent full scalar-domain slice for the remaining e bypass."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from codex.audit_prepared_numerator863 import vector_inverse, distinct_tag_slice, low_scalar_filter
from codex.search_early_boundary863 import NODE, early_inputs, wanted


def audit():
    path = Path('codex/prepared_numerator863_audit.json')
    previous = json.loads(path.read_text())
    assert [r['source'] for r in previous['scalar_initial_products'] if r['status'] != 'excluded'] == [0]
    rng = np.random.default_rng(8642881)
    xx = rng.choice(65536, size=1024, replace=False).astype(np.uint64) << 16
    bs = vector_inverse(0, xx); yy = wanted(early_inputs(bs)); selected = distinct_tag_slice(xx, yy)
    lows, history = low_scalar_filter(xx[selected], yy[selected])
    used = selected[:len(history)]
    return dict(node=NODE, source=0, source_name='e', status='excluded' if not lows else 'open',
                previous_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                primary_b=[int(bs[i]) for i in used], initial_factors=[int(xx[i]) for i in used],
                targets=[int(yy[i]) for i in used], history=history, low16_survivors=lows,
                low16_scalars_enumerated=65536, represented_word_scalars=1 << 32,
                omitted_high16_bits_annihilated=True,
                scope='Y=G*e+DIV/REM/CDIV(C,P), arbitrary32-bit prepared G/C, arbitrary nonzero denominator P.',
                best_executable_cycles=864, new_executable=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); result = audit()
    args.output.write_text(json.dumps(result, indent=2)+'\n'); print(result['status'], flush=True)


if __name__ == '__main__':
    main()
