from __future__ import annotations
import hashlib
import os
import platform
import sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np


def changed_provenance_files(before: dict, after: dict) -> list[str]:
    changed = []
    for key, prefix in (('source_sha256', 'fdge_repro/'),
                        ('reference_sha256', '')):
        previous, current = before[key], after[key]
        changed.extend(prefix + name for name in sorted(previous.keys() | current.keys())
                       if previous.get(name) != current.get(name))
    return changed


def provenance() -> dict:
    package = Path(__file__).parent
    references = list(package.parent.glob('*.pdf')) + list((package.parent / 'docs' / 'references').glob('*.pdf'))
    return {
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
        'machine': platform.machine(),
        'thread_environment': {name: os.environ.get(name) for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package.glob('*.py'))},
        'reference_sha256': {str(p.relative_to(package.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(references)},
        'clock': 'time.perf_counter_ns',
    }
