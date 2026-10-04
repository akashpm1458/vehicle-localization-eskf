"""Entry point for ``python -m vehicle_localization``.

BLAS thread limits are set *before* NumPy is imported: the filter multiplies tiny
15x15 matrices, and multi-threaded BLAS only adds overhead (oversubscription) there.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from vehicle_localization.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
