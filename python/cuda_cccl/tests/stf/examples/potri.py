#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python implementation of tiled POTRI (matrix inversion via Cholesky) using CUDA
STF with direct cuBLAS / cuSOLVER bindings from nvmath-python.

POTRI computes the inverse of a symmetric positive definite matrix using its
Cholesky factorization:
    1. Cholesky factorization: A = L*L^T     (``PDPOTRF`` → ``cusolverDnDpotrf``)
    2. Triangular inversion:   L^(-1)        (``PDTRTRI`` → ``cusolverDnXtrtri``)
    3. Compute A^(-1) = L^(-T) * L^(-1)      (``PDLAUUM``)

This example demonstrates:
- Tiled matrix operations with STF logical data.
- In-place calls to cuSOLVER (``potrf``, ``xtrtri``) and cuBLAS (``trsm``,
  ``syrk``, ``gemm``, ``trmm``, ``symm``) through ``nvmath.bindings`` — no
  hidden CuPy temporary allocations or workspace churn through CuPy's
  memory pool on the numerical path.
- STF-managed per-task scratch buffers declared as ``logical_data_empty`` with
  ``.write()`` access, mirroring the C++ pattern in
  ``cudax/examples/stf/linear_algebra/07-potri.cu`` (e.g. DPOTRF / DTRTRI
  workspaces, DLAAUM triangular scratch).
- A single tiny ``cp.RawKernel`` compiled once for the triangular
  copy-with-zero-fill used by ``DLAAUM`` (``cusolverDnDlacpy`` is not
  exposed by nvmath-python).
- Multi-device execution with automatic data placement.

Storage convention: tiles are numpy/CuPy row-major (``shape=(mb, nb)``). cuBLAS
/ cuSOLVER are column-major, so every call flips ``uplo``, ``side``, and (for
symmetric-update ops like ``dsyrk``) the transpose parameter using the
standard row-major-wrapper trick. ``dtrmm`` / ``dtrsm`` / ``dgemm`` keep the
original transpose as-is because they apply it to a full operand buffer.
"""

import ctypes
import sys

import numpy as np
import pytest

try:
    import cupy as cp
except ImportError:
    raise ImportError(
        "This example requires CuPy. Install it with: pip install cupy-cuda13x (or cupy-cuda12x)"
    ) from None

try:
    from nvmath.bindings import cublas as _cb
    from nvmath.bindings import cusolverDn as _cdn
except ImportError:
    raise ImportError(
        "This example requires nvmath-python. Install it with: pip install 'nvmath-python[cu13]'"
    ) from None

# Skip if the compiled CUDASTF bindings are unavailable (e.g. Windows wheels).
pytest.importorskip("cuda.stf._experimental._stf_bindings")
import cuda.stf._experimental as stf  # noqa: E402

# ---------------------------------------------------------------------------
# Direct cuBLAS / cuSOLVER helpers
# ---------------------------------------------------------------------------
#
# One handle per process is enough: all nvmath submissions are serialized by
# the Python GIL, and ``set_stream`` is called at the top of each task body so
# the asynchronous work lands on ``t.stream_ptr()``.
_cublas_handle = 0
_cusolver_handle = 0


def _cublas():
    global _cublas_handle
    if _cublas_handle == 0:
        _cublas_handle = _cb.create()
    return _cublas_handle


def _cusolver():
    global _cusolver_handle
    if _cusolver_handle == 0:
        _cusolver_handle = _cdn.create()
    return _cusolver_handle


# Row-major → column-major parameter translation tables.
# cuSOLVER reuses cuBLAS' enum types for uplo / diag, so we share them.
_FILL_FLIP = {"L": int(_cb.FillMode.UPPER), "U": int(_cb.FillMode.LOWER)}
_SIDE_FLIP = {"L": int(_cb.SideMode.RIGHT), "R": int(_cb.SideMode.LEFT)}
_OP_SAME = {
    "N": int(_cb.Operation.N),
    "T": int(_cb.Operation.T),
    "C": int(_cb.Operation.T),
}
_OP_FLIP = {
    "N": int(_cb.Operation.T),
    "T": int(_cb.Operation.N),
    "C": int(_cb.Operation.N),
}
_DIAG = {"N": int(_cb.DiagType.NON_UNIT), "U": int(_cb.DiagType.UNIT)}
_CUDA_R_64F = 1  # cudaDataType_t, per <library_types.h>


def _scalar_ptr(val):
    """Return ``(ptr, owner)`` for an f64 scalar passed by reference to cuBLAS.

    cuBLAS' default pointer mode is HOST, so the scalar is read synchronously
    during the API call — keeping ``owner`` alive until after the call
    returns is sufficient.
    """
    owner = np.array([val], dtype=np.float64)
    return owner.ctypes.data, owner


# Triangular copy with zero-fill on the complement.
#
# ``dst[i, j] = src[i, j]`` if the element belongs to the requested triangle
# (inclusive of diagonal), else ``dst[i, j] = 0``. Equivalent to LAPACK's
# ``dlacpy(uplo)`` but written once as a CuPy ``RawKernel`` so it is compiled
# only on first use and does not allocate on invocation — only the raw device
# pointers we pass in are touched.
_TRICOPY_SRC = r"""
extern "C" __global__
void tricopy_lower(const double* __restrict__ src,
                   double* __restrict__ dst,
                   int n, int ldsrc, int lddst) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;
    dst[i * lddst + j] = (i >= j) ? src[i * ldsrc + j] : 0.0;
}
extern "C" __global__
void tricopy_upper(const double* __restrict__ src,
                   double* __restrict__ dst,
                   int n, int ldsrc, int lddst) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;
    dst[i * lddst + j] = (i <= j) ? src[i * ldsrc + j] : 0.0;
}
"""
_tricopy_module = None


def _tricopy_kernel(uplo):
    """Return a cached ``(cupy.RawKernel)`` for triangular copy with zero-fill."""
    global _tricopy_module
    if _tricopy_module is None:
        _tricopy_module = cp.RawModule(code=_TRICOPY_SRC)
    name = "tricopy_lower" if uplo == "L" else "tricopy_upper"
    return _tricopy_module.get_function(name)


def cai_to_numpy(cai_dict):
    """Convert CUDA Array Interface dict to NumPy array (for host memory)."""

    # Extract CAI fields
    data_ptr, readonly = cai_dict["data"]
    shape = cai_dict["shape"]
    typestr = cai_dict["typestr"]

    # Convert typestr to NumPy dtype
    dtype = np.dtype(typestr)

    # Calculate total size in bytes
    itemsize = dtype.itemsize
    size = np.prod(shape) * itemsize

    # Create ctypes buffer from pointer
    buffer = (ctypes.c_byte * size).from_address(data_ptr)

    # Create NumPy array from buffer
    arr = np.frombuffer(buffer, dtype=dtype).reshape(shape)

    return arr


class BlockRef:
    """Reference to a specific block in a tiled matrix."""

    def __init__(self, matrix, row, col):
        self.matrix = matrix
        self.row = row
        self.col = col
        self._handle = matrix.handle(row, col)
        self._devid = matrix.get_preferred_devid(row, col)

    def handle(self):
        """Get the STF logical data handle for this block."""
        return self._handle

    def devid(self):
        """Get the preferred device ID for this block."""
        return self._devid

    def __repr__(self):
        return f"BlockRef({self.matrix.symbol}[{self.row},{self.col}])"


class TiledMatrix:
    """
    Tiled matrix class that splits a matrix into blocks for parallel processing.
    Each block is managed as an STF logical data object.
    Uses tiled storage format for contiguous blocks.
    """

    def __init__(
        self,
        ctx,
        nrows,
        ncols,
        blocksize_rows,
        blocksize_cols,
        is_symmetric=False,
        symbol="matrix",
        dtype=np.float64,
    ):
        self.ctx = ctx
        self.symbol = symbol
        self.dtype = dtype
        self.sym_matrix = is_symmetric

        self.m = nrows
        self.n = ncols
        self.mb = blocksize_rows
        self.nb = blocksize_cols

        assert self.m % self.mb == 0, (
            f"nrows {nrows} must be divisible by blocksize_rows {blocksize_rows}"
        )
        assert self.n % self.nb == 0, (
            f"ncols {ncols} must be divisible by blocksize_cols {blocksize_cols}"
        )

        # Number of blocks
        self.mt = self.m // self.mb
        self.nt = self.n // self.nb

        # Allocate pinned host memory for faster transfers (in tiled format)
        self.h_array = cp.cuda.alloc_pinned_memory(
            self.m * self.n * np.dtype(dtype).itemsize
        )
        self.h_array_np = np.frombuffer(self.h_array, dtype=dtype).reshape(
            self.m, self.n
        )

        # Dictionary to store logical data handles for each block
        self.handles = {}

        # Determine device layout
        self.ndevs = cp.cuda.runtime.getDeviceCount()
        self.grid_p, self.grid_q = self._compute_device_grid(self.ndevs)

        print(
            f"[{self.symbol}] {self.m}x{self.n} matrix, {self.mt}x{self.nt} blocks of {self.mb}x{self.nb}"
        )
        print(
            f"[{self.symbol}] Using {self.ndevs} devices in {self.grid_p}x{self.grid_q} grid"
        )

    def _compute_device_grid(self, ndevs):
        """Compute 2D device grid dimensions (as close to square as possible)"""
        grid_p = 1
        grid_q = ndevs
        for a in range(1, int(np.sqrt(ndevs)) + 1):
            if ndevs % a == 0:
                grid_p = a
                grid_q = ndevs // a
        return grid_p, grid_q

    def get_preferred_devid(self, row, col):
        """Get preferred device ID for a given block using cyclic distribution"""
        return (row % self.grid_p) + (col % self.grid_q) * self.grid_p

    def handle(self, row, col):
        """Get the logical data handle for a block."""
        return self.handles[(row, col)]

    def block(self, row, col):
        """Get a BlockRef for block (row, col)"""
        return BlockRef(self, row, col)

    def _get_index(self, row, col):
        """Convert (row, col) to linear index in tiled storage"""
        tile_row = row // self.mb
        tile_col = col // self.nb
        tile_size = self.mb * self.nb
        tile_start = (tile_row + self.mt * tile_col) * tile_size
        offset = (row % self.mb) + (col % self.nb) * self.mb
        return tile_start + offset

    def _get_block_h(self, brow, bcol):
        """Get a view of the host data for block (brow, bcol)"""
        # For tiled storage, blocks are stored contiguously
        start_idx = (brow + self.mt * bcol) * self.mb * self.nb
        end_idx = start_idx + self.mb * self.nb
        flat_view = self.h_array_np.ravel()
        return flat_view[start_idx:end_idx].reshape(self.mb, self.nb)

    def fill(self, func):
        """
        Fill the matrix blocks using a function func(row, col) -> value.
        Creates STF logical data from host arrays and lets STF handle transfers.
        """
        print(f"[{self.symbol}] Filling matrix on host...")
        for colb in range(self.nt):
            low_rowb = colb if self.sym_matrix else 0
            for rowb in range(low_rowb, self.mt):
                # Fill host block
                h_block = self._get_block_h(rowb, colb)
                for lrow in range(self.mb):
                    for lcol in range(self.nb):
                        row = lrow + rowb * self.mb
                        col = lcol + colb * self.nb
                        h_block[lrow, lcol] = func(row, col)

                handle = self.ctx.logical_data(
                    h_block, name=f"{self.symbol}_{rowb}_{colb}"
                )
                self.handles[(rowb, colb)] = handle


# ============================================================================
# Block-level operations (BLAS/LAPACK)
# ============================================================================


def DPOTRF(ctx, a):
    """Cholesky factorization of a diagonal block: A = L*L^T (row-major lower).

    cuSOLVER is column-major, so "lower row-major" == "upper column-major";
    we therefore call ``dpotrf`` with ``uplo=UPPER``. The scratch buffer and
    ``devInfo`` are declared as STF workspaces via ``logical_data_empty(...)``
    + ``.write()`` — STF allocates them for this task and drops them at end,
    exactly like the C++ example.
    """
    n = a.matrix.mb

    # Pointer/value-invariant size query: safe to run outside any task.
    lwork = _cdn.dpotrf_buffer_size(_cusolver(), _FILL_FLIP["L"], n, 0, n)

    potrf_buffer = ctx.logical_data_empty(
        (lwork,), np.float64, name=f"DPOTRF_ws_{a.row}_{a.col}"
    )
    dev_info = ctx.logical_data_empty(
        (1,), np.int32, name=f"DPOTRF_info_{a.row}_{a.col}"
    )

    with ctx.task(
        stf.exec_place.device(a.devid()),
        a.handle().rw(),
        potrf_buffer.write(),
        dev_info.write(),
    ) as t:
        _cdn.set_stream(_cusolver(), t.stream_ptr())
        _cdn.dpotrf(
            _cusolver(),
            _FILL_FLIP["L"],
            n,
            t.get_arg_cai(0).ptr,
            n,
            t.get_arg_cai(1).ptr,
            lwork,
            t.get_arg_cai(2).ptr,
        )


def DTRSM(ctx, a, b, side="L", uplo="L", transa="N", diag="N", alpha=1.0):
    """Triangular solve via cuBLAS dtrsm (in-place on B).

    Row-major → column-major: swap ``side``, flip ``uplo``, keep ``trans`` and
    ``diag``, and exchange ``m`` / ``n``.
    """
    mb_b, nb_b = b.matrix.mb, b.matrix.nb
    nb_a = a.matrix.nb  # square diagonal block
    alpha_ptr, _alpha_owner = _scalar_ptr(alpha)

    with ctx.task(
        stf.exec_place.device(b.devid()), a.handle().read(), b.handle().rw()
    ) as t:
        _cb.set_stream(_cublas(), t.stream_ptr())
        _cb.dtrsm(
            _cublas(),
            _SIDE_FLIP[side],
            _FILL_FLIP[uplo],
            _OP_SAME[transa],
            _DIAG[diag],
            nb_b,
            mb_b,
            alpha_ptr,
            t.get_arg_cai(0).ptr,
            nb_a,
            t.get_arg_cai(1).ptr,
            nb_b,
        )


def DTRTRI(ctx, a, uplo="L", diag="N"):
    """In-place triangular inversion A := A^(-1) via cusolverDnXtrtri.

    ``xtrtri`` needs both a device and a host workspace. We declare them as
    STF-owned scratch buffers: the device one via ``logical_data_empty`` /
    ``.write()``, and the host one via ``.write(stf.data_place.managed())`` so
    STF materializes it on managed memory reachable by the CPU-side workspace
    argument — mirroring the C++ ``h_buffer.write(data_place::managed())``
    pattern.
    """
    n = a.matrix.mb

    uplo_cm = _FILL_FLIP[uplo]
    diag_cm = _DIAG[diag]
    dev_bytes, host_bytes = _cdn.xtrtri_buffer_size(
        _cusolver(), uplo_cm, diag_cm, n, _CUDA_R_64F, 0, n
    )

    # STF cannot allocate zero-sized buffers; clamp both workspaces to at
    # least 1 byte and pass the true size (``host_bytes``/``dev_bytes``)
    # through to cuSOLVER.
    dev_bytes_alloc = max(dev_bytes, 1)
    host_bytes_alloc = max(host_bytes, 1)

    d_buffer = ctx.logical_data_empty(
        (dev_bytes_alloc,), np.int8, name=f"DTRTRI_dws_{a.row}_{a.col}"
    )
    h_buffer = ctx.logical_data_empty(
        (host_bytes_alloc,), np.int8, name=f"DTRTRI_hws_{a.row}_{a.col}"
    )
    dev_info = ctx.logical_data_empty(
        (1,), np.int32, name=f"DTRTRI_info_{a.row}_{a.col}"
    )

    with ctx.task(
        stf.exec_place.device(a.devid()),
        a.handle().rw(),
        d_buffer.write(),
        h_buffer.write(stf.data_place.managed()),
        dev_info.write(),
    ) as t:
        _cdn.set_stream(_cusolver(), t.stream_ptr())
        _cdn.xtrtri(
            _cusolver(),
            uplo_cm,
            diag_cm,
            n,
            _CUDA_R_64F,
            t.get_arg_cai(0).ptr,
            n,
            t.get_arg_cai(1).ptr,
            dev_bytes,
            t.get_arg_cai(2).ptr,
            host_bytes,
            t.get_arg_cai(3).ptr,
        )


def DGEMM(ctx, a, b, c, transa="N", transb="N", alpha=1.0, beta=1.0):
    """General matrix multiplication: C = alpha * op(A) * op(B) + beta * C.

    Row-major → column-major: swap A↔B, swap ``transa``↔``transb``, swap
    ``m``↔``n`` (``k`` stays); leading dims are the row-major column counts.
    """
    mb_c, nb_c = c.matrix.mb, c.matrix.nb
    nb_a = a.matrix.nb
    nb_b = b.matrix.nb
    # k = cols of op(A) row-major = rows of op(B) row-major
    k = a.matrix.nb if transa == "N" else a.matrix.mb
    alpha_ptr, _alpha_owner = _scalar_ptr(alpha)
    beta_ptr, _beta_owner = _scalar_ptr(beta)

    with ctx.task(
        stf.exec_place.device(c.devid()),
        a.handle().read(),
        b.handle().read(),
        c.handle().rw(),
    ) as t:
        _cb.set_stream(_cublas(), t.stream_ptr())
        _cb.dgemm(
            _cublas(),
            _OP_SAME[transb],
            _OP_SAME[transa],
            nb_c,
            mb_c,
            k,
            alpha_ptr,
            t.get_arg_cai(1).ptr,
            nb_b,
            t.get_arg_cai(0).ptr,
            nb_a,
            beta_ptr,
            t.get_arg_cai(2).ptr,
            nb_c,
        )


def DSYRK(ctx, a, c, uplo="L", trans="N", alpha=1.0, beta=1.0):
    """Symmetric rank-k update: C = alpha * op(A) @ op(A)^T + beta * C.

    Row-major → column-major: flip ``uplo`` and flip ``trans`` (N↔T).
    """
    n = c.matrix.mb
    k = a.matrix.nb if trans == "N" else a.matrix.mb
    nb_a = a.matrix.nb
    alpha_ptr, _alpha_owner = _scalar_ptr(alpha)
    beta_ptr, _beta_owner = _scalar_ptr(beta)

    with ctx.task(
        stf.exec_place.device(c.devid()), a.handle().read(), c.handle().rw()
    ) as t:
        _cb.set_stream(_cublas(), t.stream_ptr())
        _cb.dsyrk(
            _cublas(),
            _FILL_FLIP[uplo],
            _OP_FLIP[trans],
            n,
            k,
            alpha_ptr,
            t.get_arg_cai(0).ptr,
            nb_a,
            beta_ptr,
            t.get_arg_cai(1).ptr,
            n,
        )


def DTRMM(ctx, a, b, side="L", uplo="L", transa="N", diag="N", alpha=1.0):
    """Triangular matrix multiplication via ``cublasDtrmm`` (in-place).

    Row-major semantics:
        side='L':  B := alpha * op(A) * B
        side='R':  B := alpha * B * op(A)

    cuBLAS is column-major, so we flip ``side``, ``uplo`` and ``transa`` using
    the standard row-major wrapper trick. ``cublasDtrmm`` is natively
    out-of-place but supports in-place by passing the same buffer/ldb for ``B``
    and ``C``.
    """
    m_row = b.matrix.mb  # rows of B in row-major
    n_row = b.matrix.nb  # cols of B in row-major
    lda = a.matrix.mb  # A is square mb x mb (lda in col-major = row-major ncols)
    ldb = n_row
    alpha_ptr, _alpha_owner = _scalar_ptr(alpha)

    with ctx.task(
        stf.exec_place.device(b.devid()), a.handle().read(), b.handle().rw()
    ) as t:
        _cb.set_stream(_cublas(), t.stream_ptr())
        _cb.dtrmm(
            _cublas(),
            _SIDE_FLIP[side],
            _FILL_FLIP[uplo],
            _OP_SAME[transa],
            _DIAG[diag],
            n_row,  # m_cm (swapped)
            m_row,  # n_cm
            alpha_ptr,
            t.get_arg_cai(0).ptr,
            lda,
            t.get_arg_cai(1).ptr,
            ldb,
            t.get_arg_cai(1).ptr,
            ldb,
        )


def DSYMM(ctx, a, b, c, side="L", uplo="L", alpha=1.0, beta=1.0):
    """Symmetric matrix multiplication via ``cublasDsymm``.

    Row-major:
        side='L':  C := alpha * A * B + beta * C
        side='R':  C := alpha * B * A + beta * C
    where A is symmetric; cuBLAS only reads the ``uplo`` triangle of A.

    Standard row-major wrapper: flip ``side`` and ``uplo``, swap (m, n). No
    operand swap is needed because ``dsymm`` has no transpose parameter.
    """
    m_row = c.matrix.mb
    n_row = c.matrix.nb
    lda = a.matrix.mb
    ldb = b.matrix.nb
    ldc = n_row
    alpha_ptr, _alpha_owner = _scalar_ptr(alpha)
    beta_ptr, _beta_owner = _scalar_ptr(beta)

    with ctx.task(
        stf.exec_place.device(c.devid()),
        a.handle().read(),
        b.handle().read(),
        c.handle().rw(),
    ) as t:
        _cb.set_stream(_cublas(), t.stream_ptr())
        _cb.dsymm(
            _cublas(),
            _SIDE_FLIP[side],
            _FILL_FLIP[uplo],
            n_row,  # m_cm
            m_row,  # n_cm
            alpha_ptr,
            t.get_arg_cai(0).ptr,
            lda,
            t.get_arg_cai(1).ptr,
            ldb,
            beta_ptr,
            t.get_arg_cai(2).ptr,
            ldc,
        )


# ============================================================================
# Tiled operations
# ============================================================================


def PDPOTRF(ctx, A, uplo="L"):
    """Parallel tiled Cholesky factorization"""
    print("\n[PDPOTRF] Starting Cholesky factorization...")
    assert uplo == "L", "Only lower triangular factorization supported"

    for k in range(A.nt):
        # Factorize diagonal block
        DPOTRF(ctx, A.block(k, k))

        # Update column below diagonal
        for m in range(k + 1, A.mt):
            DTRSM(
                ctx,
                A.block(k, k),
                A.block(m, k),
                side="R",
                uplo="L",
                transa="T",
                diag="N",
                alpha=1.0,
            )

        # Update trailing submatrix
        for n in range(k + 1, A.nt):
            DSYRK(
                ctx,
                A.block(n, k),
                A.block(n, n),
                uplo="L",
                trans="N",
                alpha=-1.0,
                beta=1.0,
            )

            for m in range(n + 1, A.mt):
                DGEMM(
                    ctx,
                    A.block(m, k),
                    A.block(n, k),
                    A.block(m, n),
                    transa="N",
                    transb="T",
                    alpha=-1.0,
                    beta=1.0,
                )

    print("[PDPOTRF] Completed")


def PDTRTRI(ctx, A, uplo="L", diag="N"):
    """Parallel tiled triangular matrix inversion"""
    print("\n[PDTRTRI] Starting triangular inversion...")
    assert uplo == "L", "Only lower triangular inversion supported"

    for k in range(A.nt):
        # Step 1: Update A[m,k] for m > k
        for m in range(k + 1, A.mt):
            DTRSM(
                ctx,
                A.block(k, k),
                A.block(m, k),
                side="R",
                uplo="L",
                transa="N",
                diag=diag,
                alpha=-1.0,
            )

        # Step 2: Update A[m,n] for m > k, n < k
        for m in range(k + 1, A.mt):
            for n in range(k):
                DGEMM(
                    ctx,
                    A.block(m, k),
                    A.block(k, n),
                    A.block(m, n),
                    transa="N",
                    transb="N",
                    alpha=1.0,
                    beta=1.0,
                )

        # Step 3: Update A[k,n] for n < k
        for n in range(k):
            DTRSM(
                ctx,
                A.block(k, k),
                A.block(k, n),
                side="L",
                uplo="L",
                transa="N",
                diag=diag,
                alpha=1.0,
            )

        # Step 4: Invert diagonal block A[k,k]
        DTRTRI(ctx, A.block(k, k), uplo=uplo, diag=diag)

    print("[PDTRTRI] Completed")


def DLAAUM(ctx, a, uplo="L"):
    """In-place ``lauum`` on a triangular block via an STF-managed scratch.

    Lower: ``A := L^T * L``   (L = tril(A))
    Upper: ``A := U   * U^T`` (U = triu(A))

    Mirrors the C++ ``cublasDnDlaaum`` reference implementation in
    ``cudax/examples/stf/linear_algebra/07-potri.cu``: zero a workspace, copy
    the relevant triangle of ``A`` into it (zeros on the complement),
    multiply in-place with ``cublasDtrmm``, and copy the result's triangle
    back. ``cusolverDnDlacpy`` is not exposed by nvmath-python, so the
    triangle-copy uses a tiny ``cp.RawKernel`` compiled once on first use.
    The scratch buffer is an ``STF logical_data_empty(...).write()``, matching
    the C++ pattern.
    """
    n = a.matrix.mb

    scratch = ctx.logical_data_empty(
        (n, n), np.float64, name=f"DLAAUM_ws_{a.row}_{a.col}"
    )
    alpha_ptr, _alpha_owner = _scalar_ptr(1.0)

    # Row-major: for Lower we compute B := L^T * B (side=L, transa=T).
    # Row-major: for Upper we compute B := B * U^T (side=R, transa=T).
    side_row = "L" if uplo == "L" else "R"

    with ctx.task(
        stf.exec_place.device(a.devid()),
        a.handle().rw(),
        scratch.write(),
    ) as t:
        a_ptr = t.get_arg_cai(0).ptr
        s_ptr = t.get_arg_cai(1).ptr
        stream = t.stream_ptr()

        nbytes = n * n * np.dtype(np.float64).itemsize

        with cp.cuda.ExternalStream(stream):
            cp.cuda.runtime.memsetAsync(s_ptr, 0, nbytes, stream)
            kernel = _tricopy_kernel(uplo)
            block = (16, 16, 1)
            grid = ((n + 15) // 16, (n + 15) // 16, 1)
            kernel(
                grid,
                block,
                (a_ptr, s_ptr, np.int32(n), np.int32(n), np.int32(n)),
            )

        _cb.set_stream(_cublas(), stream)
        _cb.dtrmm(
            _cublas(),
            _SIDE_FLIP[side_row],
            _FILL_FLIP[uplo],
            _OP_SAME["T"],
            _DIAG["N"],
            n,
            n,
            alpha_ptr,
            a_ptr,
            n,
            s_ptr,
            n,
            s_ptr,
            n,
        )

        with cp.cuda.ExternalStream(stream):
            kernel = _tricopy_kernel(uplo)
            block = (16, 16, 1)
            grid = ((n + 15) // 16, (n + 15) // 16, 1)
            kernel(
                grid,
                block,
                (s_ptr, a_ptr, np.int32(n), np.int32(n), np.int32(n)),
            )


def PDLAUUM(ctx, A, uplo="L"):
    """Parallel tiled computation of A^T * A for lower triangular A"""
    print("\n[PDLAUUM] Starting LAUUM (A^T * A)...")
    assert uplo == "L", "Only lower triangular LAUUM supported"

    for k in range(A.mt):
        # Step 1: Update off-diagonal blocks
        for n in range(k):
            # Update A[n,n] with A[k,n]^T * A[k,n]
            DSYRK(
                ctx,
                A.block(k, n),
                A.block(n, n),
                uplo="L",
                trans="T",
                alpha=1.0,
                beta=1.0,
            )

            # Update A[m,n] with A[k,m]^T * A[k,n]
            for m in range(n + 1, k):
                DGEMM(
                    ctx,
                    A.block(k, m),
                    A.block(k, n),
                    A.block(m, n),
                    transa="T",
                    transb="N",
                    alpha=1.0,
                    beta=1.0,
                )

        # Step 2: Update A[k,n] = A[k,k]^T * A[k,n]
        for n in range(k):
            DTRMM(
                ctx,
                A.block(k, k),
                A.block(k, n),
                side="L",
                uplo="L",
                transa="T",
                diag="N",
                alpha=1.0,
            )

        # Step 3: Update diagonal block A[k,k] = A[k,k]^T * A[k,k]
        DLAAUM(ctx, A.block(k, k), uplo=uplo)

    print("[PDLAUUM] Completed")


def PDGEMM(ctx, A, B, C, transa="N", transb="N", alpha=1.0, beta=1.0):
    """Parallel tiled matrix multiplication"""
    print("\n[PDGEMM] Starting matrix multiplication...")

    for m in range(C.mt):
        for n in range(C.nt):
            inner_k = A.nt if transa == "N" else A.mt

            if alpha == 0.0 or inner_k == 0:
                # Just scale C
                DGEMM(
                    ctx,
                    A.block(0, 0),
                    B.block(0, 0),
                    C.block(m, n),
                    transa=transa,
                    transb=transb,
                    alpha=0.0,
                    beta=beta,
                )
            elif transa == "N":
                if transb == "N":
                    for k in range(A.nt):
                        zbeta = beta if k == 0 else 1.0
                        DGEMM(
                            ctx,
                            A.block(m, k),
                            B.block(k, n),
                            C.block(m, n),
                            transa="N",
                            transb="N",
                            alpha=alpha,
                            beta=zbeta,
                        )
                else:
                    for k in range(A.nt):
                        zbeta = beta if k == 0 else 1.0
                        DGEMM(
                            ctx,
                            A.block(m, k),
                            B.block(n, k),
                            C.block(m, n),
                            transa="N",
                            transb="T",
                            alpha=alpha,
                            beta=zbeta,
                        )
            else:  # transa in ['T', 'C']
                if transb == "N":
                    for k in range(A.mt):
                        zbeta = beta if k == 0 else 1.0
                        DGEMM(
                            ctx,
                            A.block(k, m),
                            B.block(k, n),
                            C.block(m, n),
                            transa="T",
                            transb="N",
                            alpha=alpha,
                            beta=zbeta,
                        )
                else:
                    for k in range(A.mt):
                        zbeta = beta if k == 0 else 1.0
                        DGEMM(
                            ctx,
                            A.block(k, m),
                            B.block(n, k),
                            C.block(m, n),
                            transa="T",
                            transb="T",
                            alpha=alpha,
                            beta=zbeta,
                        )

    print("[PDGEMM] Completed")


def PDSYMM(ctx, A, B, C, side="L", uplo="L", alpha=1.0, beta=1.0):
    """Parallel tiled symmetric matrix multiplication"""
    print("\n[PDSYMM] Starting symmetric matrix multiplication...")

    for m in range(C.mt):
        for n in range(C.nt):
            if side == "L":
                if uplo == "L":
                    for k in range(C.mt):
                        zbeta = beta if k == 0 else 1.0
                        if k < m:
                            DGEMM(
                                ctx,
                                A.block(m, k),
                                B.block(k, n),
                                C.block(m, n),
                                transa="N",
                                transb="N",
                                alpha=alpha,
                                beta=zbeta,
                            )
                        else:
                            if k == m:
                                DSYMM(
                                    ctx,
                                    A.block(k, k),
                                    B.block(k, n),
                                    C.block(m, n),
                                    side=side,
                                    uplo=uplo,
                                    alpha=alpha,
                                    beta=zbeta,
                                )
                            else:
                                DGEMM(
                                    ctx,
                                    A.block(k, m),
                                    B.block(k, n),
                                    C.block(m, n),
                                    transa="T",
                                    transb="N",
                                    alpha=alpha,
                                    beta=zbeta,
                                )
            else:  # side == 'R'
                # Similar logic for right multiplication
                pass

    print("[PDSYMM] Completed")


def compute_norm(ctx, matrix):
    """Compute Frobenius norm of matrix using host tasks"""
    norm_sq = 0.0

    for colb in range(matrix.nt):
        low_rowb = colb if matrix.sym_matrix else 0
        for rowb in range(low_rowb, matrix.mt):
            handle = matrix.handle(rowb, colb)

            # Host task to read the block and compute norm
            def compute_block_norm(h_block):
                nonlocal norm_sq
                norm_sq += np.sum(h_block * h_block)

            with ctx.task(stf.exec_place.host(), handle.read()) as t:
                # Synchronize the stream before reading data
                cp.cuda.runtime.streamSynchronize(t.stream_ptr())

                h_block = cai_to_numpy(t.get_arg_cai(0))
                compute_block_norm(h_block)

    return np.sqrt(norm_sq)


def main(N=512, NB=128, check_result=False):
    assert N % NB == 0, f"Matrix size {N} must be divisible by block size {NB}"

    print("=" * 60)
    print("Tiled POTRI (Matrix Inversion) with CUDA STF + CuPy")
    print("=" * 60)
    print(f"Matrix size: {N}x{N}")
    print(f"Block size: {NB}x{NB}")
    print(f"Number of blocks: {N // NB}x{N // NB}")
    print(f"Check result: {check_result}")
    print("=" * 60)

    # Create STF context
    ctx = stf.context()

    # Create matrices
    A = TiledMatrix(ctx, N, N, NB, NB, is_symmetric=True, symbol="A")

    if check_result:
        Aref = TiledMatrix(ctx, N, N, NB, NB, is_symmetric=False, symbol="Aref")

    print("\n" + "=" * 60)
    print("Initializing matrices...")
    print("=" * 60)

    # Hilbert matrix + diagonal dominance for numerical stability
    def hilbert(row, col):
        return 1.0 / (col + row + 1.0) + 2.0 * N * (col == row)

    A.fill(hilbert)
    if check_result:
        Aref.fill(hilbert)

    # Measure performance
    import time

    start_time = time.time()

    print("\n" + "=" * 60)
    print("Performing POTRI (inversion via Cholesky)...")
    print("=" * 60)

    # Step 1: Cholesky factorization A = L*L^T
    PDPOTRF(ctx, A, uplo="L")

    # Step 2: Triangular inversion L^(-1)
    PDTRTRI(ctx, A, uplo="L", diag="N")

    # Step 3: Compute A^(-1) = L^(-T) * L^(-1)
    PDLAUUM(ctx, A, uplo="L")

    if check_result:
        print("\n" + "=" * 60)
        print("Verifying result...")
        print("=" * 60)

        # Create test vector B
        B_potri = TiledMatrix(ctx, N, 1, NB, 1, is_symmetric=False, symbol="B_potri")
        Bref_potri = TiledMatrix(
            ctx, N, 1, NB, 1, is_symmetric=False, symbol="Bref_potri"
        )

        def rhs_vals(row, col):
            return 1.0 * (row + 1)

        B_potri.fill(rhs_vals)
        Bref_potri.fill(rhs_vals)

        # Compute norm of B
        b_norm = compute_norm(ctx, Bref_potri)

        # Create temporary matrix for result
        B_tmp = TiledMatrix(ctx, N, 1, NB, 1, is_symmetric=False, symbol="B_tmp")

        def zero_vals(row, col):
            return 0.0

        B_tmp.fill(zero_vals)

        # Compute B_tmp = A^(-1) * B
        PDSYMM(ctx, A, B_potri, B_tmp, side="L", uplo="L", alpha=1.0, beta=0.0)

        # Compute residual: Bref = Aref * B_tmp - Bref
        PDGEMM(
            ctx, Aref, B_tmp, Bref_potri, transa="N", transb="N", alpha=1.0, beta=-1.0
        )

        # Compute residual norm
        res_norm = compute_norm(ctx, Bref_potri)

    print("\n" + "=" * 60)
    print("Finalizing STF context...")
    print("=" * 60)
    ctx.finalize()

    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000.0

    # Compute FLOPS for POTRI
    # POTRF: (1/3) * N^3
    # TRTRI: (1/3) * N^3
    # LAUUM: (1/3) * N^3
    # Total: N^3
    flops = float(N) ** 3
    gflops = flops / (elapsed_ms / 1000.0) / 1e9

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"[POTRI] Elapsed time: {elapsed_ms:.2f} ms")
    print(f"[POTRI] Performance: {gflops:.2f} GFLOPS")

    if check_result:
        residual = res_norm / b_norm
        print(f"\n[POTRI] ||A * (A^(-1) * B) - B||: {res_norm:.6e}")
        print(f"[POTRI] ||B||: {b_norm:.6e}")
        print(f"[POTRI] Residual (||A * (A^(-1) * B) - B||/||B||): {residual:.6e}")

        if residual < 0.01:
            print("\n✅ Algorithm converged successfully!")
            return 0
        else:
            print(f"\n❌ Algorithm did not converge (residual {residual:.6e} >= 0.01)")
            return 1

    print("=" * 60)
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Tiled POTRI (matrix inversion via Cholesky) with CUDA STF"
    )
    parser.add_argument(
        "N", type=int, nargs="?", default=512, help="Matrix size (default: 512)"
    )
    parser.add_argument(
        "NB", type=int, nargs="?", default=128, help="Block size (default: 128)"
    )
    parser.add_argument("--check", action="store_true", help="Check result (slower)")
    args = parser.parse_args()

    sys.exit(main(N=args.N, NB=args.NB, check_result=args.check))
