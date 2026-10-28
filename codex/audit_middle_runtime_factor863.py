"""Close the special fused q*v+q tail left by parity-factor screening."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from codex.audit_division_first_t863 import CLASSES
from codex.audit_runtime_shift_core863 import base_fields, monomials
from codex.audit_small_numerator863 import bit_fits
from codex.screen_middle_division_residual863 import pool
from codex.screen_xor_mad_polynomial import consistent
from codex.search_early_boundary863 import MASK, NAMES, NODE, early_inputs, wanted


def audit():
    even_coefficients = []
    for source, name in enumerate(NAMES):
        if source == 3:
            assert all(early_inputs(b)[source] & 1 == 0 for b in CLASSES)
            continue
        b = next(b for b in CLASSES if early_inputs(b)[source] & 1)
        pair = [b, b ^ 65536]
        assert all((early_inputs(x)[source]+1) & 1 == 0 for x in pair)
        even_coefficients.append(dict(source=source, source_name=name, primary_pair=pair))
    rows = []
    for b in pool():
        v = early_inputs(b); y = wanted(v)
        assert v[3] & 511 == 0
        q = (y*pow(v[3]+1, -1, 1 << 32)) & MASK
        assert q*(v[3]+1) & MASK == y
        if q > 1 << 31:
            rows.append((b, v, q))
    matrix = [monomials(base_fields(b, v)) for b, v, q in rows]
    targets = [q for b, v, q in rows]
    assert len(set(targets)) > 1
    polynomial = consistent(matrix, targets, 32); bit = bit_fits([v for b, v, q in rows], targets)
    assert not polynomial and not bit
    assert consistent(matrix, [v[0] for b, v, q in rows], 32)
    return dict(node=NODE, input_names=NAMES, other_plus_one_factor_cases=even_coefficients,
                v_plus_one_always_odd=True,
                high_forced_values=[dict(primary_b=b, initial_words=v, quotient=q) for b, v, q in rows],
                first_MAD_relaxation=dict(base_fields=5, coefficients=21, consistent=polynomial, positive_control=True),
                first_bitop_fits=bit,
                computed_numerator='Every uint32 DIV/REM/CDIV result>2^31 equals its numerator. On these inputs a computed numerator prefix P must equal the forced q. No first MAD or bitop can interpolate them.',
                prepared_numerator='All high q must equal the same prepared numerator. The recorded high values are distinct, so no prepared numerator is possible.',
                initial_numerator='At a legal zero of any initial numerator, q and q*v+q are0; the target is nonzero at all nine recorded zero witnesses in middle_division_residual863_screen.json.',
                remaining_in_stated_family=0, best_executable_cycles=864, new_executable=False,
                scope='T prefix, one middle DIV/REM/CDIV, final q*initial+q. All factors except v fail parity; v+1 uses high-output forcing. Initial denominator zeros and self-division cases reduce as in prior division audits. No other final tail is claimed closed.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); result = audit()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('q*initial+q family closed; forced high quotient points', len(result['high_forced_values']), flush=True)


if __name__ == '__main__':
    main()
