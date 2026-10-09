"""BLAS thread limits for the work on small dense matrices.

A reduced model's sweep, its beam blocks and the joins of reduced sections
do many dense operations on matrices of a few hundred rows. At that size a
multithreaded BLAS spends its time starting and synchronising threads:
measured on 16 cores, the beam join of a module of eight reduced cavities ran
10x slower on every thread than on four, and ~100x slower with three such
processes running at once. :func:`small_dense_blas` caps the threads for the
duration of such work; large factorisations (the full-order solve, the
coupled solve of a big joined model) keep every thread.
"""

from __future__ import annotations

from contextlib import contextmanager

#: BLAS threads for work on small dense matrices
SMALL_DENSE_THREADS = 1


@contextmanager
def small_dense_blas(limit: int = SMALL_DENSE_THREADS):
    """Limit the BLAS threads (OpenBLAS, MKL, ...) to ``limit`` in this block.

    A no-op when threadpoolctl cannot control the BLAS library.
    """
    try:
        from threadpoolctl import threadpool_limits
        ctl = threadpool_limits(limits=limit, user_api='blas')
    except Exception:
        ctl = None
    try:
        yield
    finally:
        if ctl is not None:
            ctl.unregister()
