"""Generate isolated CUDA candidate; refuse an unreviewed base source."""
from pathlib import Path
import hashlib
BASE = Path('/home/shiba/multikernel-experiment/research/Strata-dsv41-experiment/src/dsv41/ds4_iq.cu')
OUT = Path(__file__).resolve().parent
assert hashlib.sha256(BASE.read_bytes()).hexdigest() == 'fc2eecbe95a382faa02f25328332afa2b9ab2733da28e840f566969e05f44055'
s = BASE.read_text()
changes = [
 ('#if !defined(STRATA_USE_HIP)\n#include <mma.h>', '#if !defined(STRATA_USE_HIP) && !defined(STRATA_DS4_SCALAR_ONLY)\n#include <mma.h>'),
 ('#if !defined(STRATA_USE_HIP)\n// the same on the tensor cores', '#if !defined(STRATA_USE_HIP) && !defined(STRATA_DS4_SCALAR_ONLY)\n// the same on the tensor cores'),
 ('#if defined(STRATA_USE_HIP)\n    return false;\n#else\n    // (per device:', '#if defined(STRATA_USE_HIP) || defined(STRATA_DS4_SCALAR_ONLY)\n    return false;\n#else\n    // (per device:'),
 ('#if !defined(STRATA_USE_HIP)\n    if (tc_ok()) {', '#if !defined(STRATA_USE_HIP) && !defined(STRATA_DS4_SCALAR_ONLY)\n    if (tc_ok()) {'),
]
for old, new in changes:
 assert s.count(old) == 1, old
 s = s.replace(old, new)
(OUT / 'ds4_iq-sm61.cu').write_text(s)
