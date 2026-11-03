"""Check first-instruction coverage for two-op, one-input arithmetic programs.

Scope: the 14 arithmetic opcodes in search_hash_transport863, arbitrary
preloaded uint32 constants, one variable x, and no other live inputs. This
does NOT cover FLOW select, memory lookup, branch-dependent programs, or
joint rewrites with additional live values. No whole-kernel bound follows.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

import z3
from z3.z3util import get_vars

from codex.search_hash_transport863 import FIRST_FORMS, OPS, first_operation, operation
from codex.prove_middle_quadratic863 import certificate


def prove():
    x = z3.BitVec('coverage_x',32)
    constants = z3.BitVecs('coverage_k0 coverage_k1 coverage_k2',32)
    zero,one = z3.BitVecVal(0,32),z3.BitVecVal(1,32)
    rows = []
    for code in OPS:
        arity = 3 if code == 'mad' else 2
        for flags in itertools.product((False,True),repeat=arity):
            args = [x if flag else constants[i] for i,flag in enumerate(flags)]
            while len(args) < 3: args.append(zero)
            lhs,valid = operation(code,*args)
            if code in ('//','%','cdiv') and flags[1]:
                # A straight-line instruction with denominator x faults at
                # x=0, regardless of every preloaded constant.
                at_zero = z3.simplify(z3.substitute(valid,(x,zero)))
                assert z3.is_false(at_zero)
                rows.append(dict(op=code,variable_operands=list(flags),
                    classification='invalid_at_input_zero'))
                continue
            cases = []

            def add(form,d=zero,e=zero,guard=valid):
                # Synthesized constants must not smuggle x into a literal.
                assert not any(z3.eq(v,x) for expr in (d,e) for v in get_vars(expr))
                cases.append((form,d,e,guard))

            if not any(flags): add('affine',zero,lhs)
            elif code == 'mad':
                if flags[0] and flags[1]:
                    add('square-input') if flags[2] else add('square',args[2])
                elif flags[0] or flags[1]:
                    multiplier = args[1] if flags[0] else args[0]
                    add('affine',multiplier+one,zero) if flags[2] else add('affine',multiplier,args[2])
                else: add('affine',one,args[0]*args[1])
            elif all(flags):
                if code == '+': add('affine',z3.BitVecVal(2,32))
                elif code in ('-','^','<'): add('affine')
                elif code in ('&','|'): add('affine',one)
                elif code == '==': add('affine',zero,one)
                elif code == '*': add('square')
                elif code in ('<<','>>'): add('self-shl' if code == '<<' else 'self-shr')
                else: raise AssertionError(code)
            else:
                k = args[1] if flags[0] else args[0]
                if code == '+': add('affine',one,k)
                elif code == '-':
                    add('affine',one,-k) if flags[0] else add('affine',-one,k)
                elif code == '*': add('affine',k)
                elif code in ('^','&','|','=='):
                    add('equal' if code == '==' else code,k)
                elif code == '<': add('less-than' if flags[0] else 'greater-than',k)
                elif code == '<<':
                    add('affine',one << k) if flags[0] else add('constant-shl',k)
                elif code == '>>':
                    if not flags[0]: add('constant-shr',k)
                    else:
                        add('affine',one,guard=k==0)
                        add('shr',k,guard=z3.And(z3.UGE(k,1),z3.ULE(k,31)))
                        add('affine',guard=z3.UGE(k,32))
                elif code in ('//','%','cdiv'): add(code,k)
                else: raise AssertionError(code)

            cover = z3.SolverFor('QF_BV')
            cover.add(valid,z3.Not(z3.Or(*(row[3] for row in cases))))
            assert cover.check() == z3.unsat
            for form,d,e,guard in cases:
                rhs,rhs_valid = first_operation(form,x,d,e)
                solver = z3.SolverFor('QF_BV')
                solver.set(timeout=10000)
                solver.add(guard,z3.Or(z3.Not(rhs_valid),lhs!=rhs))
                status = solver.check()
                assert status == z3.unsat,(code,flags,form,status)
            rows.append(dict(op=code,variable_operands=list(flags),
                classification='normalized',families=[row[0] for row in cases],
                proof='full_uint32_symbolic_constants'))
    return dict(first_instruction_shapes=len(rows),
        normalized=sum(row['classification']=='normalized' for row in rows),
        invalid_at_zero=sum(row['classification']=='invalid_at_input_zero' for row in rows),
        scope=__doc__.strip(),rows=rows)


def audit_results(root):
    """Join stored exclusions to the normalization proof, requiring every cell."""
    table = {}
    files = []
    controls = 0
    raw_count = 0
    statuses = Counter()
    for pattern in ('hash_transport863*_result.json',
                    'hash_middle_tail863*_result.json',
                    'hash_head_xor863*_result.json',
                    'hash_extended863*_result.json'):
        for path in sorted(root.glob(pattern)):
            data = json.loads(path.read_text())
            objective = data.get('objective','transport')
            first = data.get('first')
            if first is None:
                assert path.name == 'hash_transport863_result.json'
                first = 'shr16'
            rows = {row['op']:row for row in data['results'] if row['control'] is None}
            assert set(rows) == set(OPS),path
            for row in data.get('followup',[]): rows[row['op']] = row
            checks = [row for row in data['results'] if row['control'] is not None]
            assert len(checks)==2 and all(row['status']=='proved_equivalent' for row in checks),path
            controls += len(checks)
            for op,row in rows.items():
                key = objective,first,op
                assert key not in table,key
                table[key] = row['status']
                statuses[row['status']] += 1
                raw_count += 1
            files.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    proof = certificate()
    saved = json.loads((root/'hash_middle_tail863_quadratic_certificate.json').read_text())
    assert saved == proof and proof['status']=='polynomial_template_excluded'
    forms = set(FIRST_FORMS)-{'shr16'}
    independent = 0
    for objective,first,op in itertools.product(('transport','middle-tail','head-xor'),sorted(forms),OPS):
        status = table[objective,first,op]  # Missing cells are errors, not exclusions.
        if status == 'unsat': continue
        assert (objective,first,op,status)==('middle-tail','affine','mad','unknown')
        independent += 1
    coverage = prove()
    normalized_forms = {form for row in coverage['rows'] for form in row.get('families',[])}
    assert normalized_forms == forms
    return dict(raw_template_checks=raw_count,distinct_normalized_templates=3*len(forms)*len(OPS),
        raw_smt_statuses=dict(statuses),positive_controls=controls,
        independent_polynomial_exclusions=independent,
        checked_first_shapes=coverage['first_instruction_shapes'],
        complete_for='Two straight-line arithmetic instructions, one uint32 input and arbitrary preloaded constants; the 14 OPS only',
        excluded_from_scope='FLOW select, memory operations, branches, multiple inputs, additional live intermediates, whole-kernel optimality',
        minimum_instructions_per_isolated_objective=3,verified_kernel_cycles=864,
        files=files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--results-dir',type=Path,
                        help='Audit all stored searches as well as the first-instruction proof')
    args = parser.parse_args()
    result = audit_results(args.results_dir) if args.results_dir else prove()
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','files')}))
