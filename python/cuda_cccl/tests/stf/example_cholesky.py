#!/usr/bin/env python3
"""
Python implementation of Cholesky decomposition using CUDA STF and CuPy (CUBLAS/CUSOLVER).

This example demonstrates:
- Tiled matrix operations with STF logical data
- Integration of CuPy's CUBLAS and CUSOLVER functions with STF tasks
- Multi-device execution with automatic data placement
- Task-based parallelism for linear algebra operations

Note: CUDASTF automatically manages device context within tasks via exec_place.device().
There's no need to manually set the current device in task bodies - just use the STF stream.
"""

import sys
import numpy as np
import cupy as cp
from cupyx.scipy import linalg as cp_linalg
import cuda.stf as stf


class TiledMatrix:
    """
    Tiled matrix class that splits a matrix into blocks for parallel processing.
    Each block is managed as an STF logical data object.
    """
    
    def __init__(self, ctx, nrows, ncols, block_rows, block_cols, is_symmetric=False, symbol="matrix", dtype=np.float64):
        """
        Initialize a tiled matrix.
        
        Args:
            ctx: STF context
            nrows: Total number of rows
            ncols: Total number of columns
            block_rows: Block size (rows)
            block_cols: Block size (columns)
            is_symmetric: If True, only stores lower triangular blocks
            symbol: Name/symbol for the matrix
            dtype: Data type (default: np.float64)
        """
        self.ctx = ctx
        self.symbol = symbol
        self.dtype = dtype
        
        self.m = nrows
        self.n = ncols
        self.mb = block_rows
        self.nb = block_cols
        self.sym_matrix = is_symmetric
        
        assert self.m % self.mb == 0, f"nrows ({self.m}) must be divisible by block_rows ({self.mb})"
        assert self.n % self.nb == 0, f"ncols ({self.n}) must be divisible by block_cols ({self.nb})"
        
        # Number of blocks
        self.mt = self.m // self.mb
        self.nt = self.n // self.nb
        
        # Allocate host memory (pinned for faster transfers)
        self.h_array = cp.cuda.alloc_pinned_memory(self.m * self.n * np.dtype(dtype).itemsize)
        self.h_array_np = np.frombuffer(self.h_array, dtype=dtype).reshape(self.m, self.n)
        
        # Create logical data handles for each block
        self.handles = {}
        
        # Get available devices for mapping
        self.ndevs = cp.cuda.runtime.getDeviceCount()
        self.grid_p, self.grid_q = self._compute_device_grid(self.ndevs)
        
        print(f"[{symbol}] {self.m}x{self.n} matrix, {self.mt}x{self.nt} blocks of {self.mb}x{self.nb}")
        print(f"[{symbol}] Using {self.ndevs} devices in {self.grid_p}x{self.grid_q} grid")
        
        # Create blocks
        for colb in range(self.nt):
            low_rowb = colb if self.sym_matrix else 0
            for rowb in range(low_rowb, self.mt):
                # Get tile data from host array (using tiled storage)
                tile_data = self._get_block_h(rowb, colb)
                
                # Create CuPy array on the preferred device
                devid = self.get_preferred_devid(rowb, colb)
                with cp.cuda.Device(devid):
                    # Create device array for this block
                    d_block = cp.asarray(tile_data)
                    
                    # Create STF logical data
                    device_place = stf.data_place.device(devid)
                    handle = self.ctx.logical_data(d_block, device_place)
                    handle.set_symbol(f"{symbol}_{rowb}_{colb}")
                    
                    self.handles[(rowb, colb)] = (handle, d_block)
    
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
        """Get the logical data handle for block (row, col)"""
        return self.handles[(row, col)][0]
    
    def device_array(self, row, col):
        """Get the CuPy device array for block (row, col)"""
        return self.handles[(row, col)][1]
    
    def _get_index(self, row, col):
        """Convert (row, col) to linear index in tiled storage"""
        # Find which tile contains this element
        tile_row = row // self.mb
        tile_col = col // self.nb
        
        tile_size = self.mb * self.nb
        
        # Index of the beginning of the tile
        tile_start = (tile_row + self.mt * tile_col) * tile_size
        
        # Offset within the tile
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
        """Fill matrix using a function func(row, col) -> value"""
        print(f"[{self.symbol}] Filling matrix...")
        
        for colb in range(self.nt):
            low_rowb = colb if self.sym_matrix else 0
            for rowb in range(low_rowb, self.mt):
                devid = self.get_preferred_devid(rowb, colb)
                handle = self.handle(rowb, colb)
                d_array = self.device_array(rowb, colb)
                
                # Fill on host then copy to device
                h_block = self._get_block_h(rowb, colb)
                for lrow in range(self.mb):
                    for lcol in range(self.nb):
                        row = lrow + rowb * self.mb
                        col = lcol + colb * self.nb
                        h_block[lrow, lcol] = func(row, col)
                
                # Copy to device
                with cp.cuda.Device(devid):
                    cp.copyto(d_array, cp.asarray(h_block))
    
    def finalize(self):
        """Copy all blocks back to host memory"""
        print(f"[{self.symbol}] Finalizing (copying back to host)...")
        for colb in range(self.nt):
            low_rowb = colb if self.sym_matrix else 0
            for rowb in range(low_rowb, self.mt):
                devid = self.get_preferred_devid(rowb, colb)
                d_array = self.device_array(rowb, colb)
                h_block = self._get_block_h(rowb, colb)
                
                with cp.cuda.Device(devid):
                    cp.copyto(h_block, cp.asnumpy(d_array))


# BLAS/LAPACK operations wrapped in STF tasks

def DPOTRF(ctx, A, row, col):
    """Cholesky factorization of block (row, col) using CUSOLVER"""
    handle = A.handle(row, col)
    d_block = A.device_array(row, col)
    devid = A.get_preferred_devid(row, col)
    
    with ctx.task(stf.exec_place.device(devid), handle.rw()) as t:
        # STF automatically sets the current device, just use the stream
        stream_ptr = t.stream_ptr()
        cp_stream = cp.cuda.ExternalStream(stream_ptr)
        
        with cp_stream:
            # Perform Cholesky factorization (lower triangular) - IN PLACE
            # CuPy's cholesky returns L where A = L @ L.T
            d_block[:] = cp.linalg.cholesky(d_block)

def DTRSM(ctx, A, a_row, a_col, B, b_row, b_col, side='L', uplo='L', transa='T', diag='N', alpha=1.0):
    """Triangular solve: B = alpha * op(A)^{-1} @ B or B = alpha * B @ op(A)^{-1}"""
    handle_a = A.handle(a_row, a_col)
    handle_b = B.handle(b_row, b_col)
    d_a = A.device_array(a_row, a_col)
    d_b = B.device_array(b_row, b_col)
    devid = B.get_preferred_devid(b_row, b_col)
    
    with ctx.task(stf.exec_place.device(devid), handle_a.read(), handle_b.rw()) as t:
        # STF automatically sets the current device
        stream_ptr = t.stream_ptr()
        cp_stream = cp.cuda.ExternalStream(stream_ptr)
        
        with cp_stream:
            # Triangular solve using CuPy
            # For side='L': solve op(A) @ X = B for X, then X = alpha * X
            if side == 'L':
                if transa == 'N':
                    # Solve A @ X = B where A is lower/upper triangular
                    d_b[:] = cp_linalg.solve_triangular(d_a, d_b, lower=(uplo == 'L'))
                else:
                    # Solve A^T @ X = B where A is lower triangular
                    # This is equivalent to solving U @ X = B where U = A^T is upper
                    d_b[:] = cp_linalg.solve_triangular(d_a.T, d_b, lower=(uplo != 'L'))
                if alpha != 1.0:
                    d_b *= alpha
            else:
                # For side='R': solve X @ op(A) = B
                # Rewrite as op(A)^T @ X^T = B^T
                if transa == 'N':
                    d_b[:] = cp_linalg.solve_triangular(d_a.T, d_b.T, lower=(uplo != 'L')).T
                else:
                    d_b[:] = cp_linalg.solve_triangular(d_a, d_b.T, lower=(uplo == 'L')).T
                if alpha != 1.0:
                    d_b *= alpha

def DGEMM(ctx, A, a_row, a_col, B, b_row, b_col, C, c_row, c_col, 
          transa='N', transb='N', alpha=1.0, beta=1.0):
    """Matrix multiplication: C = alpha * op(A) @ op(B) + beta * C"""
    handle_a = A.handle(a_row, a_col)
    handle_b = B.handle(b_row, b_col)
    handle_c = C.handle(c_row, c_col)
    d_a = A.device_array(a_row, a_col)
    d_b = B.device_array(b_row, b_col)
    d_c = C.device_array(c_row, c_col)
    devid = C.get_preferred_devid(c_row, c_col)
    
    with ctx.task(stf.exec_place.device(devid), handle_a.read(), handle_b.read(), handle_c.rw()) as t:
        # STF automatically sets the current device
        stream_ptr = t.stream_ptr()
        cp_stream = cp.cuda.ExternalStream(stream_ptr)
        
        with cp_stream:
            # Apply transposes
            op_a = d_a.T if transa != 'N' else d_a
            op_b = d_b.T if transb != 'N' else d_b
            
            # C = alpha * op(A) @ op(B) + beta * C (IN PLACE)
            if beta == 0.0:
                d_c[:] = alpha * (op_a @ op_b)
            elif beta == 1.0:
                d_c[:] += alpha * (op_a @ op_b)
            else:
                d_c[:] = alpha * (op_a @ op_b) + beta * d_c

def DSYRK(ctx, A, a_row, a_col, C, c_row, c_col, uplo='L', trans='N', alpha=1.0, beta=1.0):
    """Symmetric rank-k update: C = alpha * op(A) @ op(A).T + beta * C"""
    handle_a = A.handle(a_row, a_col)
    handle_c = C.handle(c_row, c_col)
    d_a = A.device_array(a_row, a_col)
    d_c = C.device_array(c_row, c_col)
    devid = C.get_preferred_devid(c_row, c_col)
    
    with ctx.task(stf.exec_place.device(devid), handle_a.read(), handle_c.rw()) as t:
        # STF automatically sets the current device
        stream_ptr = t.stream_ptr()
        cp_stream = cp.cuda.ExternalStream(stream_ptr)
        
        with cp_stream:
            # Apply transpose
            op_a = d_a.T if trans != 'N' else d_a
            
            # C = alpha * op(A) @ op(A).T + beta * C (IN PLACE)
            if beta == 0.0:
                d_c[:] = alpha * (op_a @ op_a.T)
            elif beta == 1.0:
                d_c[:] += alpha * (op_a @ op_a.T)
            else:
                d_c[:] = alpha * (op_a @ op_a.T) + beta * d_c


# High-level algorithms

def PDPOTRF(ctx, A):
    """Parallel tiled Cholesky factorization (blocked algorithm)"""
    print(f"\n[PDPOTRF] Starting Cholesky factorization...")
    
    assert A.m == A.n, "Matrix must be square"
    assert A.mt == A.nt, "Block grid must be square"
    assert A.sym_matrix, "Matrix must be symmetric"
    
    nblocks = A.mt
    
    for k in range(nblocks):
        # Factor diagonal block
        DPOTRF(ctx, A, k, k)
        
        # Solve triangular systems for blocks in column k
        for row in range(k + 1, nblocks):
            DTRSM(ctx, A, k, k, A, row, k, side='R', uplo='L', transa='T', diag='N', alpha=1.0)
            
            # Update trailing matrix
            for col in range(k + 1, row):
                DGEMM(ctx, A, row, k, A, col, k, A, row, col, transa='N', transb='T', alpha=-1.0, beta=1.0)
            
            # Symmetric rank-k update of diagonal block
            DSYRK(ctx, A, row, k, A, row, row, uplo='L', trans='N', alpha=-1.0, beta=1.0)
    
    print(f"[PDPOTRF] Completed")

def PDTRSM(ctx, A, B, side='L', uplo='L', trans='N', diag='N', alpha=1.0):
    """Parallel tiled triangular solve"""
    print(f"\n[PDTRSM] Starting triangular solve...")
    
    if side == 'L':
        if uplo == 'L':
            if trans == 'N':
                # Forward substitution
                for k in range(B.mt):
                    lalpha = alpha if k == 0 else 1.0
                    for n in range(B.nt):
                        DTRSM(ctx, A, k, k, B, k, n, side='L', uplo='L', transa='N', diag=diag, alpha=lalpha)
                    for m in range(k + 1, B.mt):
                        for n in range(B.nt):
                            DGEMM(ctx, A, m, k, B, k, n, B, m, n, transa='N', transb='N', alpha=-1.0, beta=lalpha)
            else:  # trans == 'T' or 'C'
                # Backward substitution
                for k in range(B.mt):
                    lalpha = alpha if k == 0 else 1.0
                    for n in range(B.nt):
                        DTRSM(ctx, A, B.mt - k - 1, B.mt - k - 1, B, B.mt - k - 1, n, 
                              side='L', uplo='L', transa='T', diag=diag, alpha=lalpha)
                    for m in range(k + 1, B.mt):
                        for n in range(B.nt):
                            DGEMM(ctx, A, B.mt - k - 1, B.mt - 1 - m, B, B.mt - k - 1, n, 
                                  B, B.mt - 1 - m, n, transa='T', transb='N', alpha=-1.0, beta=lalpha)
    
    print(f"[PDTRSM] Completed")

def PDPOTRS(ctx, A, B, uplo='L'):
    """Solve A @ X = B where A is factored by Cholesky (A = L @ L.T)"""
    print(f"\n[PDPOTRS] Solving linear system...")
    
    # First solve: L @ Y = B
    PDTRSM(ctx, A, B, side='L', uplo=uplo, trans='N' if uplo == 'L' else 'T', diag='N', alpha=1.0)
    
    # Second solve: L.T @ X = Y
    PDTRSM(ctx, A, B, side='L', uplo=uplo, trans='T' if uplo == 'L' else 'N', diag='N', alpha=1.0)
    
    print(f"[PDPOTRS] Completed")

def PDGEMM(ctx, A, B, C, transa='N', transb='N', alpha=1.0, beta=1.0):
    """Parallel tiled matrix multiplication"""
    print(f"\n[PDGEMM] Starting matrix multiplication...")
    
    for m in range(C.mt):
        for n in range(C.nt):
            inner_k = A.nt if transa == 'N' else A.mt
            
            if alpha == 0.0 or inner_k == 0:
                # Just scale C
                DGEMM(ctx, A, 0, 0, B, 0, 0, C, m, n, transa=transa, transb=transb, alpha=0.0, beta=beta)
            elif transa == 'N':
                if transb == 'N':
                    for k in range(A.nt):
                        zbeta = beta if k == 0 else 1.0
                        DGEMM(ctx, A, m, k, B, k, n, C, m, n, transa='N', transb='N', alpha=alpha, beta=zbeta)
                else:
                    for k in range(A.nt):
                        zbeta = beta if k == 0 else 1.0
                        DGEMM(ctx, A, m, k, B, n, k, C, m, n, transa='N', transb='T', alpha=alpha, beta=zbeta)
            else:
                if transb == 'N':
                    for k in range(A.mt):
                        zbeta = beta if k == 0 else 1.0
                        DGEMM(ctx, A, k, m, B, k, n, C, m, n, transa='T', transb='N', alpha=alpha, beta=zbeta)
                else:
                    for k in range(A.mt):
                        zbeta = beta if k == 0 else 1.0
                        DGEMM(ctx, A, k, m, B, n, k, C, m, n, transa='T', transb='T', alpha=alpha, beta=zbeta)
    
    print(f"[PDGEMM] Completed")

def compute_norm(matrix):
    """Compute Frobenius norm of matrix"""
    norm_sq = 0.0
    for colb in range(matrix.nt):
        low_rowb = colb if matrix.sym_matrix else 0
        for rowb in range(low_rowb, matrix.mt):
            d_block = matrix.device_array(rowb, colb)
            norm_sq += float(cp.sum(d_block * d_block))
    return np.sqrt(norm_sq)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Tiled Cholesky decomposition with CUDA STF')
    parser.add_argument('N', type=int, nargs='?', default=1024, help='Matrix size (default: 1024)')
    parser.add_argument('NB', type=int, nargs='?', default=128, help='Block size (default: 128)')
    parser.add_argument('--check', action='store_true', help='Check result (slower)')
    args = parser.parse_args()
    
    N = args.N
    NB = args.NB
    check_result = args.check
    
    assert N % NB == 0, f"Matrix size {N} must be divisible by block size {NB}"
    
    print("="*60)
    print("Tiled Cholesky Decomposition with CUDA STF + CuPy")
    print("="*60)
    print(f"Matrix size: {N}x{N}")
    print(f"Block size: {NB}x{NB}")
    print(f"Number of blocks: {N//NB}x{N//NB}")
    print(f"Check result: {check_result}")
    print("="*60)
    
    # Create STF context
    ctx = stf.context()
    
    # Create matrices
    A = TiledMatrix(ctx, N, N, NB, NB, is_symmetric=True, symbol="A")
    
    if check_result:
        Aref = TiledMatrix(ctx, N, N, NB, NB, is_symmetric=False, symbol="Aref")
    
    # Fill with Hilbert matrix + diagonal dominance
    # H_{i,j} = 1/(i+j+1) + 2*N if i==j
    def hilbert(row, col):
        return 1.0 / (row + col + 1.0) + (2.0 * N if row == col else 0.0)
    
    print("\n" + "="*60)
    print("Initializing matrices...")
    print("="*60)
    
    A.fill(hilbert)
    if check_result:
        Aref.fill(hilbert)
    
    # Create right-hand side
    if check_result:
        B = TiledMatrix(ctx, N, 1, NB, 1, is_symmetric=False, symbol="B")
        Bref = TiledMatrix(ctx, N, 1, NB, 1, is_symmetric=False, symbol="Bref")
        
        def rhs_vals(row, col):
            return 1.0 * (row + 1)
        
        B.fill(rhs_vals)
        Bref.fill(rhs_vals)
        
        # Compute ||B|| for residual calculation
        Bref_norm = compute_norm(Bref)
    
    # Synchronize before timing
    cp.cuda.runtime.deviceSynchronize()
    
    # Record start time
    start_event = cp.cuda.Event()
    stop_event = cp.cuda.Event()
    start_event.record()
    
    # Perform Cholesky factorization
    print("\n" + "="*60)
    print("Performing Cholesky factorization...")
    print("="*60)
    PDPOTRF(ctx, A)
    
    # Record stop time
    stop_event.record()
    
    # Solve system if checking
    if check_result:
        print("\n" + "="*60)
        print("Solving linear system...")
        print("="*60)
        PDPOTRS(ctx, A, B, uplo='L')
        
        print("\n" + "="*60)
        print("Computing residual...")
        print("="*60)
        # Compute residual: Bref = Aref @ B - Bref
        PDGEMM(ctx, Aref, B, Bref, transa='N', transb='N', alpha=1.0, beta=-1.0)
        
        # Compute ||residual||
        res_norm = compute_norm(Bref)
    
    # Finalize STF context
    print("\n" + "="*60)
    print("Finalizing STF context...")
    print("="*60)
    ctx.finalize()
    
    # Wait for completion
    stop_event.synchronize()
    
    # Compute timing
    elapsed_ms = cp.cuda.get_elapsed_time(start_event, stop_event)
    gflops = (1.0/3.0 * N * N * N) / 1e9
    gflops_per_sec = gflops / (elapsed_ms / 1000.0)
    
    print("\n" + "="*60)
    print("Results")
    print("="*60)
    print(f"[PDPOTRF] Elapsed time: {elapsed_ms:.2f} ms")
    print(f"[PDPOTRF] Performance: {gflops_per_sec:.2f} GFLOPS")
    
    if check_result:
        residual = res_norm / Bref_norm
        print(f"\n[POTRS] ||AX - B||: {res_norm:.6e}")
        print(f"[POTRS] ||B||: {Bref_norm:.6e}")
        print(f"[POTRS] Residual (||AX - B||/||B||): {residual:.6e}")
        
        if residual >= 0.01:
            print("\n❌ Algorithm did not converge (residual >= 0.01)")
            return 1
        else:
            print("\n✅ Algorithm converged successfully!")
    
    print("="*60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

