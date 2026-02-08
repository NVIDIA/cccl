#!/usr/bin/env python3
"""
Python implementation of POTRI (matrix inversion via Cholesky) using CUDA STF and CuPy.

POTRI computes the inverse of a symmetric positive definite matrix using its Cholesky factorization:
1. Cholesky factorization: A = L*L^T
2. Triangular inversion: L^(-1)
3. Compute A^(-1) = L^(-T) * L^(-1)

This example demonstrates:
- Tiled matrix operations with STF logical data
- Integration of CuPy's CUBLAS and CUSOLVER functions with STF tasks
- Multi-device execution with automatic data placement
- Task-based parallelism for linear algebra operations
"""

import sys

import cupy as cp
import numpy as np
from cupyx.scipy import linalg as cp_linalg

import cuda.stf as stf


class CAIWrapper:
    """Wrapper to expose CUDA Array Interface dict as a proper CAI object."""

    def __init__(self, cai_dict):
        self.__cuda_array_interface__ = cai_dict


def get_cupy_arrays(task):
    """
    Get all CuPy arrays from STF task arguments.

    Usage:
        d_a, d_b, d_c = get_cupy_arrays(t)
    """
    arrays = []
    idx = 0
    while True:
        try:
            arrays.append(cp.asarray(CAIWrapper(task.get_arg_cai(idx))))
            idx += 1
        except Exception:
            break
    return tuple(arrays) if len(arrays) > 1 else arrays[0] if arrays else None


def cai_to_numpy(cai_dict):
    """Convert CUDA Array Interface dict to NumPy array (for host memory)."""
    import ctypes

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

                handle = self.ctx.logical_data(h_block)
                handle.set_symbol(f"{self.symbol}_{rowb}_{colb}")

                self.handles[(rowb, colb)] = handle


# ============================================================================
# Block-level operations (BLAS/LAPACK)
# ============================================================================


def DPOTRF(ctx, a):
    """Cholesky factorization of a diagonal block: A = L*L^T (lower triangular)"""
    with ctx.task(stf.exec_place.device(a.devid()), a.handle().rw()) as t:
        d_block = get_cupy_arrays(t)
        with cp.cuda.ExternalStream(t.stream_ptr()):
            d_block[:] = cp.linalg.cholesky(d_block)


def DTRSM(ctx, a, b, side="L", uplo="L", transa="N", diag="N", alpha=1.0):
    """Triangular solve: B = alpha * op(A)^(-1) * B"""
    with ctx.task(
        stf.exec_place.device(b.devid()), a.handle().read(), b.handle().rw()
    ) as t:
        d_a, d_b = get_cupy_arrays(t)
        with cp.cuda.ExternalStream(t.stream_ptr()):
            lower = uplo == "L"
            trans = transa != "N"
            result = cp_linalg.solve_triangular(d_a, d_b, lower=lower, trans=trans)
            if alpha != 1.0:
                d_b[:] = alpha * result
            else:
                d_b[:] = result


def DTRTRI(ctx, a, uplo="L", diag="N"):
    """Triangular matrix inversion: A = A^(-1)"""
    with ctx.task(stf.exec_place.device(a.devid()), a.handle().rw()) as t:
        d_block = get_cupy_arrays(t)
        with cp.cuda.ExternalStream(t.stream_ptr()):
            lower = uplo == "L"
            unit_diagonal = diag == "U"
            # CuPy doesn't have trtri directly, use solve with identity
            n = d_block.shape[0]
            identity = cp.eye(n, dtype=d_block.dtype)
            d_block[:] = cp_linalg.solve_triangular(
                d_block, identity, lower=lower, unit_diagonal=unit_diagonal
            )


def DGEMM(ctx, a, b, c, transa="N", transb="N", alpha=1.0, beta=1.0):
    """General matrix multiplication: C = alpha * op(A) * op(B) + beta * C"""
    with ctx.task(
        stf.exec_place.device(c.devid()),
        a.handle().read(),
        b.handle().read(),
        c.handle().rw(),
    ) as t:
        d_a, d_b, d_c = get_cupy_arrays(t)
        with cp.cuda.ExternalStream(t.stream_ptr()):
            op_a = d_a.T if transa != "N" else d_a
            op_b = d_b.T if transb != "N" else d_b

            if beta == 0.0:
                d_c[:] = alpha * (op_a @ op_b)
            elif beta == 1.0:
                d_c[:] += alpha * (op_a @ op_b)
            else:
                d_c[:] = alpha * (op_a @ op_b) + beta * d_c


def DSYRK(ctx, a, c, uplo="L", trans="N", alpha=1.0, beta=1.0):
    """Symmetric rank-k update: C = alpha * op(A) @ op(A).T + beta * C"""
    with ctx.task(
        stf.exec_place.device(c.devid()), a.handle().read(), c.handle().rw()
    ) as t:
        d_a, d_c = get_cupy_arrays(t)
        with cp.cuda.ExternalStream(t.stream_ptr()):
            op_a = d_a.T if trans != "N" else d_a

            if beta == 0.0:
                d_c[:] = alpha * (op_a @ op_a.T)
            elif beta == 1.0:
                d_c[:] += alpha * (op_a @ op_a.T)
            else:
                d_c[:] = alpha * (op_a @ op_a.T) + beta * d_c


def DTRMM(ctx, a, b, side="L", uplo="L", transa="N", diag="N", alpha=1.0):
    """Triangular matrix multiplication: B = alpha * op(A) * B (side='L') or B = alpha * B * op(A) (side='R')"""
    with ctx.task(
        stf.exec_place.device(b.devid()), a.handle().read(), b.handle().rw()
    ) as t:
        d_a, d_b = get_cupy_arrays(t)
        with cp.cuda.ExternalStream(t.stream_ptr()):
            lower = uplo == "L"
            trans = transa != "N"

            # Extract triangle from A
            if lower:
                tri_a = cp.tril(d_a)
            else:
                tri_a = cp.triu(d_a)

            if trans:
                tri_a = tri_a.T

            if side == "L":
                d_b[:] = alpha * (tri_a @ d_b)
            else:  # side == 'R'
                d_b[:] = alpha * (d_b @ tri_a)


def DSYMM(ctx, a, b, c, side="L", uplo="L", alpha=1.0, beta=1.0):
    """Symmetric matrix multiplication: C = alpha * A * B + beta * C (side='L') or C = alpha * B * A + beta * C (side='R')
    where A is symmetric."""
    with ctx.task(
        stf.exec_place.device(c.devid()),
        a.handle().read(),
        b.handle().read(),
        c.handle().rw(),
    ) as t:
        d_a, d_b, d_c = get_cupy_arrays(t)
        with cp.cuda.ExternalStream(t.stream_ptr()):
            # Reconstruct full symmetric matrix from lower/upper triangle
            if uplo == "L":
                # Lower triangle is stored
                sym_a = cp.tril(d_a) + cp.tril(d_a, -1).T
            else:
                # Upper triangle is stored
                sym_a = cp.triu(d_a) + cp.triu(d_a, 1).T

            if side == "L":
                result = alpha * (sym_a @ d_b)
            else:  # side == 'R'
                result = alpha * (d_b @ sym_a)

            if beta == 0.0:
                d_c[:] = result
            elif beta == 1.0:
                d_c[:] += result
            else:
                d_c[:] = result + beta * d_c


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
    """Compute A^T * A for a triangular block (lauum operation)"""
    with ctx.task(stf.exec_place.device(a.devid()), a.handle().rw()) as t:
        d_block = get_cupy_arrays(t)
        with cp.cuda.ExternalStream(t.stream_ptr()):
            # lauum: compute L * L^T for lower triangular L
            if uplo == "L":
                L = cp.tril(d_block)
                d_block[:] = L @ L.T
            else:
                U = cp.triu(d_block)
                d_block[:] = U.T @ U


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


def main():
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

    N = args.N
    NB = args.NB
    check_result = args.check

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
    sys.exit(main())
