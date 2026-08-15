# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Sparse conjugate gradient with the CSR matrix as ONE bundle dependency.

Python port of ``cudax/examples/stf/linear_algebra/cg_csr.cu``. The earlier
``cg.py`` port simplified the problem to a dense matrix because a CSR matrix
is three arrays (values / column indices / row offsets) and every task had to
spell all three dependencies. A :class:`bundle` makes the sparse port direct:
the matrix is one object, tasks depend on it with a single argument, and the
structure arrays carry a read-only ceiling (``constant``) so a whole-bundle
``rw()`` could never write them.

Every task below receives the matrix as one slot: ``t.get(0)`` yields a
namespace with ``vals`` / ``colind`` / ``rowptr`` views.
"""

import numpy as np
import pytest

pytest.importorskip("cuda.stf._experimental._stf_bindings")
numba = pytest.importorskip("numba")
from numba import cuda  # noqa: E402

import cuda.stf._experimental as stf  # noqa: E402
from cuda.stf._experimental.bundles import constant  # noqa: E402


def _nb(view):
    """Adapt a task view (CUDA Array Interface) to a numba device array."""
    return cuda.from_cuda_array_interface(
        view.__cuda_array_interface__, owner=None, sync=False
    )


def _stream(t):
    """The task's CUDA stream as a numba stream (kernels must run on it)."""
    return cuda.external_stream(t.stream_ptr())


def _nb_views(bundle_view):
    """Adapt a whole bundle view: a namedtuple of numba device arrays.

    Numba types namedtuples of arrays, so the bundle stays ONE kernel
    argument — the device-side analog of the C++ tuple view.
    """
    return type(bundle_view)(*map(_nb, bundle_view))


@cuda.jit
def _spmv_kernel(a, x, y):
    """The CSR matrix arrives as ONE argument (a namedtuple of arrays)."""
    i = cuda.grid(1)
    if i < y.shape[0]:
        acc = 0.0
        for k in range(a.rowptr[i], a.rowptr[i + 1]):
            acc += a.vals[k] * x[a.colind[k]]
        y[i] = acc


@cuda.jit
def _zero1(out):
    out[0] = 0.0


@cuda.jit
def _dot_kernel(a, b, out):
    i = cuda.grid(1)
    if i < a.shape[0]:
        cuda.atomic.add(out, 0, a[i] * b[i])


@cuda.jit
def _axpy_kernel(alpha_num, alpha_den, x, y):
    """y += (alpha_num / alpha_den) * x"""
    i = cuda.grid(1)
    if i < y.shape[0]:
        y[i] += (alpha_num[0] / alpha_den[0]) * x[i]


@cuda.jit
def _xpay_kernel(beta_num, beta_den, x, y):
    """y = x + (beta_num / beta_den) * y"""
    i = cuda.grid(1)
    if i < y.shape[0]:
        y[i] = x[i] + (beta_num[0] / beta_den[0]) * y[i]


def spmv(ctx, A, lx, ly, n):
    """y = A @ x — the CSR matrix is a single dependency."""
    nb = (n + 127) // 128
    with ctx.task(A.read(), lx.read(), ly.rw()) as t:
        _spmv_kernel[nb, 128, _stream(t)](
            _nb_views(t.get(0)), _nb(t.get(1)), _nb(t.get(2))
        )


def dot(ctx, la, lb, lres, n):
    nb = (n + 127) // 128
    with ctx.task(la.read(), lb.read(), lres.rw()) as t:
        s = _stream(t)
        out = _nb(t.get(2))
        _zero1[1, 1, s](out)
        _dot_kernel[nb, 128, s](_nb(t.get(0)), _nb(t.get(1)), out)


def test_cg_csr_bundle():
    n = 256

    # Tridiagonal SPD system (2 on the diagonal, -1 off-diagonal)
    rowptr, colind, vals = [0], [], []
    for i in range(n):
        for j, v in ((i - 1, -1.0), (i, 2.0), (i + 1, -1.0)):
            if 0 <= j < n:
                colind.append(j)
                vals.append(v)
        rowptr.append(len(colind))

    rng = np.random.default_rng(7)
    b = rng.standard_normal(n)

    ctx = stf.context()

    # The matrix is one bundle: values tracked, structure read-only ceilinged
    A = ctx.bundle(
        vals=ctx.logical_data(np.array(vals, dtype=np.float64)),
        colind=constant(ctx.logical_data(np.array(colind, dtype=np.int32))),
        rowptr=constant(ctx.logical_data(np.array(rowptr, dtype=np.int32))),
    )

    x_host = np.zeros(n)
    lx = ctx.logical_data(x_host)
    lr = ctx.logical_data(b.copy())  # r = b - A@0 = b
    lp = ctx.logical_data(b.copy())  # p = r
    lap = ctx.logical_data(np.zeros(n))
    lrsold = ctx.logical_data(np.zeros(1))
    lrsnew = ctx.logical_data(np.zeros(1))
    lpap = ctx.logical_data(np.zeros(1))

    nb = (n + 127) // 128
    dot(ctx, lr, lr, lrsold, n)

    for _ in range(2 * n):
        spmv(ctx, A, lp, lap, n)
        dot(ctx, lp, lap, lpap, n)

        with ctx.task(lrsold.read(), lpap.read(), lp.read(), lx.rw()) as t:
            _axpy_kernel[nb, 128, _stream(t)](
                _nb(t.get(0)), _nb(t.get(1)), _nb(t.get(2)), _nb(t.get(3))
            )
        with ctx.task(lrsold.read(), lpap.read(), lap.read(), lr.rw()) as t:
            # r -= alpha * Ap
            _xpay_neg[nb, 128, _stream(t)](
                _nb(t.get(0)), _nb(t.get(1)), _nb(t.get(2)), _nb(t.get(3))
            )
        dot(ctx, lr, lr, lrsnew, n)
        with ctx.task(lrsnew.read(), lrsold.read(), lr.read(), lp.rw()) as t:
            _xpay_kernel[nb, 128, _stream(t)](
                _nb(t.get(0)), _nb(t.get(1)), _nb(t.get(2)), _nb(t.get(3))
            )
        with ctx.task(lrsnew.read(), lrsold.rw()) as t:
            _copy1[1, 1, _stream(t)](_nb(t.get(0)), _nb(t.get(1)))

    ctx.finalize()

    # x_host received the write-back at finalize; verify A @ x ~= b
    ax = np.zeros(n)
    rp = np.array(rowptr)
    ci = np.array(colind)
    vv = np.array(vals)
    xh = np.asarray(x_host)
    for i in range(n):
        ax[i] = np.dot(vv[rp[i] : rp[i + 1]], xh[ci[rp[i] : rp[i + 1]]])
    np.testing.assert_allclose(ax, b, atol=1e-6)


@cuda.jit
def _xpay_neg(alpha_num, alpha_den, x, y):
    """y -= (alpha_num / alpha_den) * x"""
    i = cuda.grid(1)
    if i < y.shape[0]:
        y[i] -= (alpha_num[0] / alpha_den[0]) * x[i]


@cuda.jit
def _copy1(src, dst):
    dst[0] = src[0]


if __name__ == "__main__":
    test_cg_csr_bundle()
    print("cg_csr_bundle OK")
