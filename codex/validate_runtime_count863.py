"""Replay the completed count proofs and fingerprint this continuation."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import z3

from codex.audit_runtime_shift_band863 import audit as band_audit
from codex.audit_runtime_count_tail863 import audit as tail_audit
from codex.close_runtime_count_prefix863 import audit as prefix_audit
from codex.search_first_runtime_count863 import count_manifest, interpolate, solve, fibres
from codex.validate_joint_quotient_range863 import PROTECTED


SOURCES=['README.md','codex/README.md','codex/SUB863_RUNTIME_COUNT_20261029.md',
    *('codex/'+name+'.py' for name in (
        'audit_runtime_shift_band863','audit_runtime_count_tail863','close_runtime_count_prefix863',
        'enumerate_runtime_count_tail863','search_first_runtime_count863','search_runtime_count_prefix863',
        'search_runtime_count_tail863','test_runtime_count_prefix863','validate_runtime_count863',
        'audit_division_first_t863','cegis_c1_absorb863','search_early_boundary863',
        'search_joint_live_boundary863','search_live_boundary863','search_root_boundary863',
        'scout_joint_wirings863','test_live_boundary863'))]
ARTIFACTS=['codex/'+name for name in (
    'runtime_shift_band863_audit_20261029.json','runtime_count_prefix863_closed_20261029.json',
    'runtime_count_prefix863_constant_20261029.json.gz','runtime_count_prefix863_binary_20261029.json.gz',
    'runtime_count_prefix863_mad_20261029.json.gz','first_runtime_count863_scout_20261029.json.gz',
    'first_runtime_count863_closed_20261029.json','runtime_count_tail863_scout_20261029.json.gz',
    'runtime_count_tail863_rectangles_20261029.json','runtime_count_tail863_inverse_20261029.json',
    'runtime_count_tail863_audit_20261029.json')]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(value):return json.loads(json.dumps(value))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--regression-log',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    regression=args.regression_log.read_text()
    match=re.search(r'Ran (\d+) tests in ([\d.]+)s\s+OK\s*$',regression)
    assert match and int(match[1])==50
    assert all(digest(path)==sha for path,sha in PROTECTED.items())
    assert not subprocess.check_output(['git','diff','origin/main','--','tests/','problem.py'],text=True)
    assert not subprocess.check_output(['git','status','--porcelain','--','tests/','problem.py'],text=True)
    band=band_audit();prefix=prefix_audit();tail=tail_audit()
    for path,result in (('codex/runtime_shift_band863_audit_20261029.json',band),
                        ('codex/runtime_count_prefix863_closed_20261029.json',prefix),
                        ('codex/runtime_count_tail863_audit_20261029.json',tail)):
        assert json.loads(Path(path).read_text())==normalized(result),path
    scout_path=Path('codex/first_runtime_count863_scout_20261029.json.gz')
    scout=json.loads(gzip.decompress(scout_path.read_bytes()))
    first=json.loads(Path('codex/first_runtime_count863_closed_20261029.json').read_text())
    rows=count_manifest()
    key=lambda r:json.dumps({k:r[k] for k in ('op','source','other','wiring') if k in r},sort_keys=True)
    expected={key(r) for r in rows}
    assert scout['completed']==len(scout['results'])==len(rows)==552
    assert {key(r) for r in scout['results']}=={key(r) for r in scout['manifest']}==expected
    assert scout['primary_fibres']==fibres()
    assert scout['universal_count_band']==normalized(band)
    assert all(r['status']=='nonconstant_count_unsat' for r in scout['results'])
    assert first['source']==str(scout_path) and first['source_sha256']==digest(scout_path)
    assert {key(r) for r in first['checks']}==expected
    assert all(r['status']=='finite_constancy_implies_full32_constant' for r in first['checks'])
    assert first['every_one_T_count_must_be_root_constant']
    # Fresh necessary-count queries validate the strongest reduction using
    # the final implementation, independently of the earlier bit0 screens.
    with ProcessPoolExecutor(2) as pool:
        futures=[pool.submit(solve,row,5) for row in rows]
        for i,future in enumerate(as_completed(futures),1):
            result=future.result()
            assert result['status']=='nonconstant_count_unsat',result
            if i%100==0:print('finite count replay',i,'/552',flush=True)
    for i,row in enumerate(rows,1):
        result=interpolate(row,5)
        assert result['status']=='finite_constancy_implies_full32_constant',result
    archives=[]
    for path in ARTIFACTS:
        if not path.endswith('.gz'):continue
        raw=Path(path).read_bytes();decoded=gzip.decompress(raw)
        assert raw[4:8]==bytes(4) and gzip.decompress(gzip.compress(decoded,mtime=0))==decoded
        archives.append(dict(path=path,encoded_bytes=len(raw),decoded_bytes=len(decoded),
                             decoded_sha256=hashlib.sha256(decoded).hexdigest()))
    official='codex/root_boundary863_validation.json'
    assert digest(official)=='2b8a33054353015dfad1bbe7794f3f27803c2c933d7012e737b141d74740e5e4'
    result=dict(date='2026-10-29',validated_at_utc=datetime.now(timezone.utc).isoformat(),
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        best_executable_cycles=864,new_executable=False,no_global_optimality_claim=True,
        runtime=dict(numpy=np.__version__,z3=z3.get_version_string()),
        regression=dict(tests=50,new_tests=22,result='OK',seconds=float(match[2]),
                        log_sha256=digest(args.regression_log)),
        complete_count_band=dict(low16=[16,23],quotient_maximum=65535,
                                 signatures=32768,primary_inputs=33554432,minimum_low9_image=387),
        first_count=dict(shapes=552,finite_queries_freshly_replayed=True,
                         full32_constancy_queries_freshly_replayed=True,all_counts_are_constant=True),
        first_prefixes=dict(connections=5520,remaining=0,original_unknowns_preserved=50,
            independently_reduced_by_full32_count_constancy_and64_raw_shift_pairs=True,
            original_bit0_screens_preserved_not_reexecuted=True),
        final_MAD=dict(connections=1656,remaining=0,original_unknowns_preserved=194,
                       all552_free_parameter_rectangle_checks_replayed=True,
                       inverse_parameters=4718592,vector_scan_replayed=True,
                       independent_scalar_full_word_scan_preserved=True),
        current_validated_source_sha256={path:digest(path) for path in SOURCES},
        artifact_sha256={path:digest(path) for path in ARTIFACTS},
        protected_source_sha256=PROTECTED,compressed_archives=archives,
        frozen_tests_and_problem_unchanged=True,
        prior_official_evidence=dict(path=official,sha256=digest(official),tests=9,cycles=864,not_rerun=True),
        provenance='Earlier UNKNOWN observations are retained. Current source/proof fingerprints '
        'are validated after all edits. The strongest552 count screens and their full32 '
        'interpolation queries are reexecuted; this independently reduces all5520 prefixes. '
        'The complete inverse scan was run with two independent implementations; this '
        'validator reexecutes its vector form and verifies its final scanner fingerprint.',
        still_open=['two-or-more-T count prefixes outside these final MAD families',
            'final tails not reading the count','multiple non-T operations',
            'other live words/encodings','new layouts or whole-kernel algorithms'])
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('50 tests OK;552 count constancy proofs,5520 prefixes and1656 tails closed; executable864.',flush=True)


if __name__=='__main__':main()
