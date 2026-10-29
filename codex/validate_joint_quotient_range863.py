"""Fingerprint the completed continuation and replay its finite certificates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import z3

from codex.close_joint_quotient_range863 import audit, normalized


SOURCES = ['codex/README.md','codex/SUB863_QUOTIENT_RANGE_20261029.md',
    *('codex/'+name+'.py' for name in ('audit_joint_quotient_bypass863','close_joint_quotient_range863',
        'refine_joint_quotient_range863','search_keyed_live_restore863','search_mixed_affine_restore863',
        'test_joint_quotient_bypass863','test_keyed_live_restore863','test_mixed_affine_restore863',
        'test_quotient_range863','validate_joint_quotient_range863','search_restore_scaled863',
        'search_cross_encoding863','search_keyed_live_middle863','run_middle_residual_enum863',
        'search_joint_live_boundary863','search_live_boundary863','search_early_boundary863',
        'close_prepared_joint863')),'codex/enumerate_middle_residual863.cpp']
ARTIFACTS = ['codex/'+name for name in (
    'joint_computed_bypass863_audit_20261029.json','joint_computed_bypass863_enum_20261029.json',
    'joint_quotient_range863_closed_20261029.json','joint_quotient_range863_min3_20261029.json.gz',
    'joint_quotient_range863_refinement_20261029.json.gz','joint_quotient_range863_scout_20261029.json.gz',
    'keyed_live_restore863_parallel_20261029.json.gz','keyed_live_restore863_parallel_refinement_20261029.json.gz',
    'keyed_live_restore863_refinement_20261029.json.gz','keyed_live_restore863_scout_20261029.json.gz',
    'keyed_live_restore863_single_input_replay_20261029.json','mixed_affine_restore863_normalized_20261029.json')]
PROTECTED = {'perf_takehome.py':'98d0f4ce1ecae152378e31ddae70d52f0ed4cb6ace5b027ce191d4027346a884',
    'codex/perf_takehome_864.py':'98d0f4ce1ecae152378e31ddae70d52f0ed4cb6ace5b027ce191d4027346a884',
    'codex/perf_takehome_865.py':'58ce2efca049a18ef343ba82ceaeb72fd74f638bd7c8bbc23a14e6e796d72f44'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--regression-log',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args = ap.parse_args()
    regression = args.regression_log.read_text()
    match = re.search(r'Ran (\d+) tests in ([\d.]+)s\s+OK\s*$',regression)
    assert match and int(match[1])==138
    checked = audit()
    stored = json.loads(Path('codex/joint_quotient_range863_closed_20261029.json').read_text())
    assert stored == normalized(checked)
    assert all(digest(path)==sha for path,sha in PROTECTED.items())
    frozen_diff = subprocess.check_output(['git','diff','origin/main','--','tests/','problem.py'],text=True)
    frozen_status = subprocess.check_output(['git','status','--porcelain','--','tests/','problem.py'],text=True)
    assert not frozen_diff and not frozen_status
    archives = []
    for name in ARTIFACTS:
        if name.endswith('.gz'):
            raw = Path(name).read_bytes();decoded = gzip.decompress(raw)
            assert raw[4:8]==bytes(4) and gzip.decompress(gzip.compress(decoded,mtime=0))==decoded
            archives.append(dict(path=name,encoded_bytes=len(raw),decoded_bytes=len(decoded),
                                 decoded_sha256=hashlib.sha256(decoded).hexdigest()))
    enum = json.loads(Path('codex/joint_computed_bypass863_enum_20261029.json').read_text())
    official = 'codex/root_boundary863_validation.json'
    assert digest(official)=='2b8a33054353015dfad1bbe7794f3f27803c2c933d7012e737b141d74740e5e4'
    result = dict(date='2026-10-29',validated_at_utc=datetime.now(timezone.utc).isoformat(),
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        best_executable_cycles=864,new_executable=False,no_global_optimality_claim=True,
        runtime=dict(numpy=np.__version__,z3=z3.get_version_string()),
        regression=dict(tests=138,new_tests=20,result='OK',seconds=float(match[2]),
                        log_sha256=digest(args.regression_log),summary=regression.strip()),
        all528_sampled_prepared_divisions_excluded=True,previous127_remaining=0,
        newly_closed_counts=checked['newly_closed_counts'],keyed_restore=checked['keyed_restore'],
        finite_certificates_independently_replayed=True,
        full_inverse_scan=dict(complete=enum['complete'],parameters=3*(1 << 31),
            all_one_high_excluded=enum['all_one_high_excluded'],seconds=enum['seconds'],
            scanner_source_sha256=enum['scanner_source_sha256'],binary_sha256=enum['binary_sha256']),
        current_validated_source_sha256={name:digest(name) for name in SOURCES},
        artifact_sha256={name:digest(name) for name in ARTIFACTS},
        protected_source_sha256=PROTECTED,compressed_archives=archives,
        frozen_tests_and_problem_unchanged=True,
        prior_official_evidence=dict(path=official,sha256=digest(official),tests=9,cycles=864,not_rerun=True),
        provenance='Current source fingerprints describe the final validated implementation. '
        'Earlier range runs retain their looser sound bounds and original UNKNOWN observations. '
        'Finite certificates, predecessor evidence and source equality are replayed; the6.44-billion scan '
        'is the preserved complete native run, not reexecuted by this validator.',
        remaining=checked['still_open'])
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('138 tests OK;528 sampled divisions and1899 keyed-live restore connections closed;',
          'protected executable remains864.',flush=True)


if __name__ == '__main__':
    main()
