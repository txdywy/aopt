"""Archive current source fingerprints and the scoped search validation."""
import argparse
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy
import z3

from codex.close_prepared_joint863 import audit


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('regression-log','closure-log','enum-log','output'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    record,open_div,open_magic=audit()
    for value,path in [(record,'codex/prepared_joint863_audit.json'),
                       (open_div,'codex/joint_prepared_division863_open_20261029.json'),
                       (open_magic,'codex/joint_prepared_division863_magic_open_20261029.json')]:
        assert value==json.loads(Path(path).read_text())
    log=args.regression_log.read_text().strip();assert 'Ran 118 tests' in log and log.endswith('OK')
    protected={p:sha(p) for p in ['perf_takehome.py','codex/perf_takehome_864.py','codex/perf_takehome_865.py']}
    assert protected['perf_takehome.py']==protected['codex/perf_takehome_864.py']=='98d0f4ce1ecae152378e31ddae70d52f0ed4cb6ace5b027ce191d4027346a884'
    assert protected['codex/perf_takehome_865.py']=='58ce2efca049a18ef343ba82ceaeb72fd74f638bd7c8bbc23a14e6e796d72f44'
    assert not subprocess.check_output(['git','diff','origin/main','--','tests/','problem.py'])
    sources=['README.md','SUB863_PREPARED_OPERANDS_20261029.md','audit_one_high_tail863.py',
             'audit_quotient_pair863.py','close_prepared_joint863.py','validate_prepared_joint863.py',
             'joint_division_relation863.py','joint_polynomial_quotient863.py','search_prepared_shift_joint863.py',
             'search_prepared_division_joint863.py','search_joint_live_boundary863.py','search_live_boundary863.py',
             'search_focused_joint_boundary863.py','test_live_boundary863.py','test_prepared_shift_joint863.py',
             'test_prepared_division_joint863.py','test_one_high_tail863.py','test_joint_polynomial_quotient863.py',
             'test_quotient_pair863.py','enumerate_middle_residual863.cpp','run_middle_residual_enum863.py',
             'search_early_boundary863.py','search_root_boundary863.py']
    artifacts=['one_high_tail863_full_enum_20261029.json','one_high_tail863_audit.json','quotient_pair863_audit.json',
               'joint_prepared_shift863_scout_20261029.json.gz','joint_prepared_shift863_refinement_20261029.json',
               'joint_prepared_shift863_prefix_20261029.json','joint_prepared_shift863_split_20261029.json.gz',
               'joint_prepared_division863_scout_20261029.json.gz','joint_prepared_division863_refinement_20261029.json.gz',
               'joint_prepared_division863_product_20261029.json.gz','joint_prepared_division863_magic_20261029.json.gz',
               'joint_prepared_division863_open_20261029.json','joint_prepared_division863_magic_open_20261029.json',
               'prepared_joint863_audit.json','claude_prepared_refresh_20261029.json']
    archives=[]
    for name in artifacts:
        if not name.endswith('.gz'):continue
        path=Path('codex')/name;raw=path.read_bytes();decoded=gzip.decompress(raw)
        archives.append(dict(path=str(path),sha256=sha(path),decoded_sha256=hashlib.sha256(decoded).hexdigest(),
                             encoded_bytes=len(raw),decoded_bytes=len(decoded)))
    enum=json.loads(Path('codex/one_high_tail863_full_enum_20261029.json').read_text())
    result=dict(date='2026-10-29',validated_at_utc=datetime.now(timezone.utc).isoformat(),
                base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                best_executable_cycles=864,new_executable=False,no_global_optimality_claim=True,
                runtime=dict(python=platform.python_version(),numpy=numpy.__version__,z3=z3.get_version_string()),
                regression=dict(tests=118,new_tests=21,result='OK',summary=log,log_sha256=sha(args.regression_log),
                                new_four_op_frozen_controls=216,new_frozen_lanes=1728),
                component_replay=dict(summary=args.closure_log.read_text().strip(),log_sha256=sha(args.closure_log),
                                      independent_prefix_guard=True),
                one_high_tail=dict(forms=913,fresh_complete_odd_parameter_scan=True,enumerated_parameters=19327352832,
                                   all_one_high_excluded=enum['all_one_high_excluded'],seconds=enum['seconds'],
                                   scanner_source_sha256=enum['scanner_source_sha256'],binary_sha256=enum['binary_sha256'],
                                   log_sha256=sha(args.enum_log)),
                information_fibre=record['information_fibre'],complete_divisor_sum_scans=record['complete_divisor_sum_scans'],
                prepared_shift=dict(sampled_connections=1384,final_counts=record['prepared_shift_final_counts'],
                                    all_sampled_excluded=True,not_complete_four_op_grammar=True),
                prepared_division=dict(sampled_connections=528,final_counts=record['prepared_division_final_counts'],
                                       unresolved_connections=127,not_complete=True,remaining_by_opcode=record['remaining_division_by_opcode']),
                pinned_divisors=dict(values=[65535,65537],cases=254,counts=record['pinned_magic_counts'],unresolved=110,
                                    does_not_close_full_divisor_scope=True),
                current_validated_source_sha256={'codex/'+p:sha('codex/'+p) for p in sources},
                artifact_sha256={'codex/'+p:sha('codex/'+p) for p in artifacts},compressed_archives=archives,
                component_evidence=record['evidence'],protected_files_sha256=protected,frozen_tests_and_problem_unchanged=True,
                prior_official_evidence=dict(path='codex/root_boundary863_validation.json',
                                            sha256=sha('codex/root_boundary863_validation.json'),tests=9,cycles=864,not_rerun_this_turn=True),
                scope='Fixed allowed raw root0x12345678; sampled four-op joint windows on eight specified initial words, no free e. Prepared shifts and DIV/REM/CDIV. Additional exact families have their separate stated scopes. Current source fingerprints describe the validated final implementation; intermediate search runs keep their original observations.',
                remaining=record['still_open'])
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('118 tests OK; protected sources unchanged; remaining full-divisor / pinned branches:127 /110',flush=True)


if __name__=='__main__':main()
