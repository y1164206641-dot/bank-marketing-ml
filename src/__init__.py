"""Project package.

Keep matplotlib's cache inside the workspace: the default user cache dir is
not writable under the DSH sandbox.
"""

import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplcache")
)

# The DSH sandbox blocks named pipes, which breaks joblib's process *and*
# threading pools (both go through multiprocessing.SimpleQueue on Windows).
# Forcing one OpenMP/BLAS thread keeps sklearn's internal Parallel calls
# sequential, so HistGradientBoosting and friends run fine.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
