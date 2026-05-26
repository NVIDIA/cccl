# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Numba-integrated task context manager for STF tests/examples.
Not shipped in the wheel. Uses task.args_cai() (CAI from cuda.stf._experimental), converts to
numba.cuda device arrays so cuda.stf._experimental has no Numba dependency. Requires numba-cuda.

Mirrors pytorch_task.py which yields torch.Tensor objects.

Example
-------
>>> from tests.stf.numba_task import numba_task
>>> with numba_task(ctx, lX.read(), lY.rw()) as (args, stream):
...     cuda.compute.reduce_into(args[0], args[1], OpKind.PLUS, N, h_init, stream=stream)
"""

from __future__ import annotations


def _to_numba(cai):
    """Convert an stf_cai object to a numba.cuda device array."""
    from numba import cuda

    return cuda.from_cuda_array_interface(cai, owner=None, sync=False)


def numba_task(ctx, *args, symbol=None):
    """Context manager: ctx.task(*args) yielding (numba_arrays, stf_stream).

    numba_arrays is a tuple of numba.cuda device arrays (one per non-token dep),
    converted from stf_cai via the CUDA Array Interface.

    stf_stream implements __cuda_stream__ so it can be passed as stream=
    to cuda.compute algorithms.

    Example
    -------
    >>> with numba_task(ctx, lA.read(), lB.read(), lC.rw()) as (args, stream):
    ...     cuda.compute.binary_transform(args[0], args[1], args[2], OpKind.PLUS, N, stream=stream)
    """
    t = ctx.task(*args, symbol=symbol)

    class _NumbaTaskContext:
        def __enter__(self):
            t.start()
            cais = t.args_cai()
            stream = t.stream_ptr()
            if cais is None:
                return ((), stream)
            if isinstance(cais, tuple):
                return (tuple(_to_numba(c) for c in cais), stream)
            return ((_to_numba(cais),), stream)

        def __exit__(self, exc_type, exc_val, exc_tb):
            t.end()
            return False

    return _NumbaTaskContext()


__all__ = ["numba_task"]
