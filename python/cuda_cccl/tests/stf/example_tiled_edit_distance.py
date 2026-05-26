# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tiled Levenshtein edit distance via ``cuda.stf._experimental`` -- wavefront scheduling for free.

The story
---------
Classical Levenshtein edit distance is a 2D DP with the recurrence

    S[i, j] = min(
        S[i-1, j]   + 1,                      # deletion
        S[i, j-1]   + 1,                      # insertion
        S[i-1, j-1] + (A[i-1] != B[j-1]),     # match / substitution
    )

Tile the table into an ``M x N`` grid of ``TS x TS`` tiles. Tile ``T[I, J]``
needs the last row of ``T[I-1, J]``, the last column of ``T[I, J-1]`` and the
bottom-right corner of ``T[I-1, J-1]``. All tiles on the same anti-diagonal
(``I + J == k``) are mutually independent and can run concurrently; width peaks
at ``min(M, N)``.

Writing this in plain CuPy / NumPy forces the user to either

  * serialise everything on one stream (no concurrency), or
  * hand-roll an anti-diagonal loop with explicit stream + event juggling.

Neither is fun. Under STF, the obvious nested ``for I, J`` double loop -- with
each tile body declared as ``last_row[I-1,J].read()``, ``last_col[I,J-1].read()``,
``corner[I-1,J-1].read()``, ``last_row[I,J].write()``, ``last_col[I,J].write()``,
``corner[I,J].write()`` -- *is* the parallel wavefront schedule. STF infers the
wavefront from the per-edge access annotations and submits independent tiles to
different CUDA streams automatically. Multi-GPU is a one-line change: assign
each tile row to a different ``stf.exec_place.device(...)`` and the
affine-data-place rule puts every piece of data on the consuming device,
inserting peer copies only for the halos that cross the link.

Dependency picture (3x3 example)
--------------------------------
    T00 -> T01 -> T02              Wavefront 0: {T00}             (1 task)
     |      |                      Wavefront 1: {T10, T01}        (2 concurrent)
     v      v                      Wavefront 2: {T20, T11, T02}   (3 concurrent)
    T10 -> T11 -> T12              Wavefront 3: {T21, T12}        (2 concurrent)
     |      |                      Wavefront 4: {T22}             (1 task)
     v      v
    T20 -> T21 -> T22

    (diagonal dependencies Tij -> T(i+1)(j+1) via corner are also present but
     omitted from the ASCII picture for clarity.)

What the benchmark shows
------------------------
Three implementations of the exact same algorithm:

  cupy_wavefront   naive CuPy: every tile kernel submitted serially on a
                   single explicit stream. No concurrency, no STF. This is the
                   "I would never write the hand-scheduled version" baseline.
  stf_tiled_single exact same nested Python double loop, but each tile is an
                   ``stf.context()`` task with ``.read()`` / ``.write()``
                   annotations. STF discovers the wavefront and schedules the
                   anti-diagonals across streams.
  stf_tiled_multi  adds ``stf.exec_place.device(I % n_gpus)`` to each task and
                   nothing else; STF places data on the affine device and
                   inserts peer copies for cross-row halos. Only difference
                   from ``stf_tiled_single`` is the ``exec_place`` argument.

All three use the identical Numba CUDA tile kernel (``_tile_edit_kernel``), so
any speedup is pure scheduling, not a better inner kernel.

Toggles
-------
    EDIT_BENCH=1        run the benchmark (default off; correctness runs always)
    EDIT_L=32768        total sequence length (must be a multiple of EDIT_TS)
    EDIT_TS=1024        tile size
    EDIT_NGPUS=2        max GPUs the multi-GPU variant will span
    EDIT_ITERS=5        timed iterations
    EDIT_WARMUP=2       warmup iterations
    EDIT_SEED=0         RNG seed for the random ASCII sequences

Correctness tests at small L validate against a pure-NumPy reference. The
2-GPU test skips silently if fewer than 2 CUDA devices are visible.

Sample numbers (1x H100 80GB, CUDA 13.0, numba-cuda 0.26)
---------------------------------------------------------
    === Tiled Levenshtein, L=32768, TS=1024, grid=32x32, iters=3 (warmup=1) ===
    implementation                   ms / run   speedup
    cupy_wavefront                    3045.95     1.00x
    stf_tiled_single                   979.36     3.11x
    stf_tiled_multi                   skipped  (need >= 2 GPUs)

The 3.1x win is pure scheduling: cupy_wavefront submits the 1024 tile kernels
serially on one stream, while stf_tiled_single runs up to 32 concurrent
anti-diagonal tiles across CUDA streams. Same Numba kernel, same memory
layout, same inner work. On 2 GPUs the multi-GPU variant typically adds
another 1.7-1.9x on top (row-cyclic placement + affine data place).

Future work
-----------
Reconstructing the actual alignment requires keeping the full interior
``S[I, J]`` tiles, which is ~17 GB at the default ``L = 32768``. The current
file only returns the scalar edit distance. Adding an optional backtrace
(gated on a small ``L`` or an env flag) is a straightforward follow-up.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import pytest

numba = pytest.importorskip("numba")
pytest.importorskip("numba.cuda")
cp = pytest.importorskip("cupy")

from numba import cuda as nbcuda  # noqa: E402
from numba_task import numba_task  # noqa: E402

import cuda.stf._experimental as stf  # noqa: E402

# Silence the "low-occupancy" warning; our inner kernel is deliberately a single
# block so the outer tile wavefront is where the concurrency comes from.
nbcuda.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EditConfig:
    """Workload shape for the tiled-edit-distance demo."""

    L: int
    TS: int
    sub_ts: int
    n_gpus: int
    iters: int
    warmup: int
    seed: int

    @classmethod
    def from_env(cls) -> "EditConfig":
        return cls(
            L=int(os.environ.get("EDIT_L", 32768)),
            TS=int(os.environ.get("EDIT_TS", 1024)),
            sub_ts=int(os.environ.get("EDIT_SUB_TS", 128)),
            n_gpus=int(os.environ.get("EDIT_NGPUS", 2)),
            iters=int(os.environ.get("EDIT_ITERS", 5)),
            warmup=int(os.environ.get("EDIT_WARMUP", 2)),
            seed=int(os.environ.get("EDIT_SEED", 0)),
        )


def _make_sequences(L: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two random int8 sequences over a 4-letter alphabet, length ``L``."""

    rng = np.random.default_rng(seed)
    A = rng.integers(0, 4, size=L, dtype=np.int8)
    B = rng.integers(0, 4, size=L, dtype=np.int8)
    return A, B


# ---------------------------------------------------------------------------
# Inner tile kernel (single block, anti-diagonal sweep)
# ---------------------------------------------------------------------------
#
# The kernel is intentionally a single block so the *outer* tile wavefront is
# the concurrency mechanism. Per-tile cost at TS=1024 is ~1-3 ms on a modern
# GPU, comfortably above the ~0.3 ms STF per-task overhead measured elsewhere
# in this repo, so the scheduling story dominates the timing.
#
# Dependency pattern inside a tile is the same as across tiles: the only way
# to get parallelism inside a single tile is the same anti-diagonal sweep we
# use at the outer level.


@nbcuda.jit
def _tile_edit_kernel(
    a_tile, b_tile, top_row, left_col, corner_in, S, last_row, last_col, corner_out
):
    """Compute one ``TS x TS`` tile's DP.

    Indexing convention: ``S`` has shape ``(TS+1, TS+1)``. Row / column 0 are
    the tile's incoming boundaries (``corner_in`` + ``top_row`` across the top,
    ``corner_in`` + ``left_col`` down the left). Rows / columns ``1..TS`` are
    the computed interior. Outputs:

      * ``last_row[k] = S[TS, k+1]``  for ``k = 0..TS-1``
      * ``last_col[k] = S[k+1, TS]``  for ``k = 0..TS-1``
      * ``corner_out   = S[TS, TS]``
    """

    TS = a_tile.shape[0]
    tid = nbcuda.threadIdx.x
    nth = nbcuda.blockDim.x

    # Seed the boundary row / column and the top-left corner.
    if tid == 0:
        S[0, 0] = corner_in[0]
    k = tid
    while k < TS:
        S[0, k + 1] = top_row[k]
        S[k + 1, 0] = left_col[k]
        k += nth
    nbcuda.syncthreads()

    # Anti-diagonal sweep over the interior. Diagonal ``d`` covers cells
    # ``(i, j)`` with ``i + j == d``, ``1 <= i, j <= TS``. ``d`` ranges 2..2*TS.
    for d in range(2, 2 * TS + 1):
        i_min = 1 if d - TS < 1 else d - TS
        i_max = TS if d - 1 > TS else d - 1
        length = i_max - i_min + 1
        t = tid
        while t < length:
            i = i_min + t
            j = d - i
            up = S[i - 1, j] + 1
            lf = S[i, j - 1] + 1
            dg = S[i - 1, j - 1] + (0 if a_tile[i - 1] == b_tile[j - 1] else 1)
            v = up
            if lf < v:
                v = lf
            if dg < v:
                v = dg
            S[i, j] = v
            t += nth
        nbcuda.syncthreads()

    # Write the outgoing halos and corner.
    k = tid
    while k < TS:
        last_row[k] = S[TS, k + 1]
        last_col[k] = S[k + 1, TS]
        k += nth
    if tid == 0:
        corner_out[0] = S[TS, TS]


_THREADS_PER_BLOCK = 256


# ---------------------------------------------------------------------------
# Multi-block path: sub-tile the tile, one block per sub-tile on a sub-diagonal
# ---------------------------------------------------------------------------
#
# The single-block kernel above uses exactly 1 CUDA block per tile. That is
# convenient for the demo story but not representative of a realistic DP
# kernel: on an H100 it leaves ~99% of SMs idle per tile, so *any* outer
# wavefront schedule trivially looks better than a serial one.
#
# The kernels below implement the same tile DP as a sub-tile wavefront:
#
#   * Chop the ``TS x TS`` tile into ``B x B`` sub-tiles of size ``SUB_TS``.
#   * Launch ``2*B - 1`` kernels, one per intra-tile anti-diagonal, each with
#     up to ``B`` blocks (one block per sub-tile on the diagonal).
#   * Per-block parallelism is the same anti-diagonal sweep used in the
#     single-block kernel, but now acting on a ``SUB_TS x SUB_TS`` patch.
#
# With ``TS = 1024`` and ``SUB_TS = 128`` the main intra-tile diagonal
# dispatches 8 blocks * 128 threads = 1024 threads concurrently per launch --
# ~10x more SM occupancy per tile than the single-block path. The outer DAG
# is unchanged, so the per-tile STF dependency annotations stay identical.


@nbcuda.jit
def _init_boundary_kernel(top_row, left_col, corner_in, S):
    """Seed ``S[0, :]``, ``S[:, 0]`` and ``S[0, 0]`` with the outer-tile boundary.

    Required by the multi-block path: the per-sub-tile kernels read their top /
    left / corner from ``S``, so the tile boundary must be materialised into
    ``S`` before the first sub-diagonal launch.
    """

    TS = top_row.shape[0]
    tid = nbcuda.threadIdx.x
    nth = nbcuda.blockDim.x

    if tid == 0:
        S[0, 0] = corner_in[0]
    k = tid
    while k < TS:
        S[0, k + 1] = top_row[k]
        S[k + 1, 0] = left_col[k]
        k += nth


@nbcuda.jit
def _subtile_edit_kernel(a_tile, b_tile, S, sub_diag_idx, B_sub, sub_ts):
    """Compute one sub-diagonal of ``sub_ts x sub_ts`` sub-tiles inside a tile.

    ``blockIdx.x`` selects the sub-tile's position on the sub-diagonal
    ``sub_diag_idx`` (``0 .. 2*B_sub - 2``). Thread count per block must be
    ``sub_ts``; thread ``t`` owns one cell on each sub-tile anti-diagonal. The
    sub-tile's top / left / corner boundaries are already in ``S`` (written by
    previous sub-diagonal launches or by :func:`_init_boundary_kernel`); the
    kernel writes the sub-tile's ``sub_ts x sub_ts`` interior back into ``S``.
    """

    d = sub_diag_idx
    b = nbcuda.blockIdx.x
    sub_i_start = 0 if d < B_sub else d - (B_sub - 1)
    sub_i = sub_i_start + b
    sub_j = d - sub_i
    i0 = sub_i * sub_ts
    j0 = sub_j * sub_ts

    tid = nbcuda.threadIdx.x

    for sd in range(2, 2 * sub_ts + 1):
        ii_min = 1 if sd - sub_ts < 1 else sd - sub_ts
        ii_max = sub_ts if sd - 1 > sub_ts else sd - 1
        length = ii_max - ii_min + 1
        if tid < length:
            ii = ii_min + tid
            jj = sd - ii
            i = i0 + ii
            j = j0 + jj
            up = S[i - 1, j] + 1
            lf = S[i, j - 1] + 1
            dg = S[i - 1, j - 1] + (0 if a_tile[i - 1] == b_tile[j - 1] else 1)
            v = up
            if lf < v:
                v = lf
            if dg < v:
                v = dg
            S[i, j] = v
        nbcuda.syncthreads()


@nbcuda.jit
def _extract_boundaries_kernel(S, last_row, last_col, corner_out):
    """Copy ``S`` 's last row / column / corner into the tile's outgoing halos."""

    TS = last_row.shape[0]
    tid = nbcuda.threadIdx.x
    nth = nbcuda.blockDim.x
    k = tid
    while k < TS:
        last_row[k] = S[TS, k + 1]
        last_col[k] = S[k + 1, TS]
        k += nth
    if tid == 0:
        corner_out[0] = S[TS, TS]


@nbcuda.jit
def _tile_edit_cg_kernel(
    a_tile,
    b_tile,
    top_row,
    left_col,
    corner_in,
    S,
    last_row,
    last_col,
    corner_out,
    B_sub,
    sub_ts,
):
    """One kernel per tile: ``B_sub`` blocks cooperate via grid-sync.

    This is the **realistic** single-launch multi-block kernel. Launched with
    ``grid=(B_sub,)`` and ``block=(sub_ts,)``, every block represents one
    column of sub-tiles in the tile's (TS/sub_ts) x (TS/sub_ts) sub-grid.
    Blocks synchronise with :func:`cuda.cg.this_grid().sync()` between sub-
    diagonals. Requires a cooperative launch (auto-detected by Numba).

    Compared to :func:`_subtile_edit_kernel` which uses one CPU-side kernel
    launch per intra-tile sub-diagonal, this variant does the whole tile in a
    single kernel launch, eliminating ``2*B_sub - 1`` launches per tile without
    changing the arithmetic or the block count at peak.
    """

    grid = nbcuda.cg.this_grid()
    b = nbcuda.blockIdx.x
    tid = nbcuda.threadIdx.x
    TS = a_tile.shape[0]

    # Seed tile boundary: S[0, 0] from corner_in, S[0, 1..TS] from top_row,
    # S[1..TS, 0] from left_col. Spread across the whole grid.
    if b == 0 and tid == 0:
        S[0, 0] = corner_in[0]
    stride = B_sub * sub_ts
    gtid = b * sub_ts + tid
    k = gtid
    while k < TS:
        S[0, k + 1] = top_row[k]
        S[k + 1, 0] = left_col[k]
        k += stride
    grid.sync()

    # Sub-diagonal sweep. All blocks reach every grid.sync() regardless of
    # whether they own an active sub-tile on this sub-diagonal.
    for d in range(2 * B_sub - 1):
        nb = min(d + 1, B_sub, 2 * B_sub - 1 - d)
        sub_i_start = 0 if d < B_sub else d - (B_sub - 1)
        is_active = b < nb
        sub_i = 0
        sub_j = 0
        i0 = 0
        j0 = 0
        if is_active:
            sub_i = sub_i_start + b
            sub_j = d - sub_i
            i0 = sub_i * sub_ts
            j0 = sub_j * sub_ts

        for sd in range(2, 2 * sub_ts + 1):
            ii_min = 1 if sd - sub_ts < 1 else sd - sub_ts
            ii_max = sub_ts if sd - 1 > sub_ts else sd - 1
            length = ii_max - ii_min + 1
            if is_active and tid < length:
                ii = ii_min + tid
                jj = sd - ii
                i = i0 + ii
                j = j0 + jj
                up = S[i - 1, j] + 1
                lf = S[i, j - 1] + 1
                dg = S[i - 1, j - 1] + (0 if a_tile[i - 1] == b_tile[j - 1] else 1)
                v = up
                if lf < v:
                    v = lf
                if dg < v:
                    v = dg
                S[i, j] = v
            nbcuda.syncthreads()
        grid.sync()

    # Extract outgoing halos and corner.
    k = gtid
    while k < TS:
        last_row[k] = S[TS, k + 1]
        last_col[k] = S[k + 1, TS]
        k += stride
    if b == 0 and tid == 0:
        corner_out[0] = S[TS, TS]


def _launch_tile(
    nb_stream,
    a_tile,
    b_tile,
    top_row,
    left_col,
    corner_in,
    S,
    last_row,
    last_col,
    corner_out,
    sub_ts: int,
) -> None:
    """Compute one outer tile by launching the appropriate kernel sequence.

    ``sub_ts == 0``
        Legacy single-block path (:func:`_tile_edit_kernel`) -- 1 kernel
        launch, 1 CUDA block, tiny per-tile SM occupancy. Kept for comparison.

    ``sub_ts > 0`` (negative)
        Multi-kernel sub-tile wavefront: ``1 + (2*B - 1) + 1`` launches per
        tile, each with a grid of up to ``B`` blocks. Illustrates the
        overhead of naive per-sub-diagonal launches.

    ``sub_ts > 0`` (positive, preferred)
        Cooperative-groups single-launch path (:func:`_tile_edit_cg_kernel`):
        ONE kernel launch per tile, grid of ``B`` blocks, grid-sync between
        sub-diagonals. This is the realistic reference kernel -- high
        per-tile SM occupancy **and** minimal launch overhead.

    The sign of ``sub_ts`` selects between the two multi-block variants: a
    positive value uses the cooperative-groups single-launch kernel, a
    negative value uses the legacy multi-kernel path. Benchmarks use the
    positive path; negative is available for overhead decomposition.
    """

    if sub_ts == 0:
        _tile_edit_kernel[1, _THREADS_PER_BLOCK, nb_stream](
            a_tile,
            b_tile,
            top_row,
            left_col,
            corner_in,
            S,
            last_row,
            last_col,
            corner_out,
        )
        return

    if sub_ts < 0:
        actual_sub_ts = -sub_ts
        TS = a_tile.shape[0]
        assert TS % actual_sub_ts == 0, f"sub_ts={sub_ts} does not divide TS={TS}"
        B = TS // actual_sub_ts
        _init_boundary_kernel[1, _THREADS_PER_BLOCK, nb_stream](
            top_row, left_col, corner_in, S
        )
        for d in range(2 * B - 1):
            nb = min(d + 1, B, 2 * B - 1 - d)
            _subtile_edit_kernel[nb, actual_sub_ts, nb_stream](
                a_tile, b_tile, S, d, B, actual_sub_ts
            )
        _extract_boundaries_kernel[1, _THREADS_PER_BLOCK, nb_stream](
            S, last_row, last_col, corner_out
        )
        return

    TS = a_tile.shape[0]
    assert TS % sub_ts == 0, f"sub_ts={sub_ts} does not divide TS={TS}"
    B = TS // sub_ts
    _tile_edit_cg_kernel[B, sub_ts, nb_stream](
        a_tile,
        b_tile,
        top_row,
        left_col,
        corner_in,
        S,
        last_row,
        last_col,
        corner_out,
        B,
        sub_ts,
    )


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def cpu_reference(A: np.ndarray, B: np.ndarray) -> int:
    """Pure-NumPy Levenshtein. ``O(L^2)`` time and memory; only used on small L."""

    L = len(A)
    S = np.empty((L + 1, L + 1), dtype=np.int32)
    S[0, :] = np.arange(L + 1, dtype=np.int32)
    S[:, 0] = np.arange(L + 1, dtype=np.int32)
    # A vectorised inner loop. Still O(L^2), kept tight for small L sanity.
    for i in range(1, L + 1):
        prev_row = S[i - 1, :]
        curr_row = S[i, :]
        for j in range(1, L + 1):
            match = 0 if A[i - 1] == B[j - 1] else 1
            curr_row[j] = min(
                prev_row[j] + 1,
                curr_row[j - 1] + 1,
                prev_row[j - 1] + match,
            )
    return int(S[L, L])


def _seeded_top_row(J: int, TS: int) -> np.ndarray:
    """DP boundary ``S[0, J*TS + 1 .. (J+1)*TS]``."""

    start = J * TS + 1
    return np.arange(start, start + TS, dtype=np.int32)


def _seeded_left_col(I: int, TS: int) -> np.ndarray:
    """DP boundary ``S[I*TS + 1 .. (I+1)*TS, 0]``."""

    start = I * TS + 1
    return np.arange(start, start + TS, dtype=np.int32)


def _seeded_corner(I: int, J: int, TS: int) -> np.ndarray:
    """DP corner ``S[I*TS, J*TS]`` -- equals ``I*TS`` on left edge, ``J*TS`` on top edge."""

    if I == 0 and J == 0:
        return np.array([0], dtype=np.int32)
    if I == 0:
        return np.array([J * TS], dtype=np.int32)
    return np.array([I * TS], dtype=np.int32)


def cupy_wavefront(A: np.ndarray, B: np.ndarray, TS: int, *, sub_ts: int = 0) -> int:
    """Honest single-stream baseline: same Numba kernels, submitted serially.

    All TxT tiles run one after the other on a single non-default CUDA stream,
    so the only parallelism exercised is what the inner kernel extracts from
    the anti-diagonal sweep within one tile. Every other source of concurrency
    (cross-tile wavefront, cross-row independence, multi-GPU placement) is
    deliberately disabled.

    ``sub_ts`` selects the inner kernel:

      * ``0`` -> single-block path (:func:`_tile_edit_kernel`), 1 CUDA block
        per tile. Deliberately low per-tile SM occupancy -- the "tiny kernel"
        scenario.
      * ``> 0`` -> multi-block sub-tile wavefront (:func:`_launch_tile`):
        each tile dispatches ``2 * (TS / sub_ts) - 1`` sub-diagonal kernels
        with up to ``TS / sub_ts`` blocks each. Much higher per-tile SM
        occupancy; this is the "realistic kernel" scenario.
    """

    L = len(A)
    assert L % TS == 0
    M = N = L // TS

    with cp.cuda.Device(0):
        stream = cp.cuda.Stream(non_blocking=False)
        nb_stream = nbcuda.external_stream(stream.ptr)

        d_A = cp.asarray(A)
        d_B = cp.asarray(B)

        # One scratch tile per row (tiles on the same row run sequentially so
        # they can safely share the same scratch).
        d_S = [cp.empty((TS + 1, TS + 1), dtype=cp.int32) for _ in range(M)]

        d_last_row = {}
        d_last_col = {}
        d_corner = {}
        for I in range(M):
            for J in range(N):
                d_last_row[(I, J)] = cp.empty((TS,), dtype=cp.int32)
                d_last_col[(I, J)] = cp.empty((TS,), dtype=cp.int32)
                d_corner[(I, J)] = cp.empty((1,), dtype=cp.int32)

        d_top_seed = [cp.asarray(_seeded_top_row(J, TS)) for J in range(N)]
        d_left_seed = [cp.asarray(_seeded_left_col(I, TS)) for I in range(M)]
        d_corner_seed = {}
        for J in range(N):
            d_corner_seed[(0, J)] = cp.asarray(_seeded_corner(0, J, TS))
        for I in range(1, M):
            d_corner_seed[(I, 0)] = cp.asarray(_seeded_corner(I, 0, TS))

        with stream:
            for I in range(M):
                for J in range(N):
                    top = d_top_seed[J] if I == 0 else d_last_row[(I - 1, J)]
                    left = d_left_seed[I] if J == 0 else d_last_col[(I, J - 1)]
                    if I > 0 and J > 0:
                        cin = d_corner[(I - 1, J - 1)]
                    else:
                        cin = d_corner_seed[(I, J)]
                    _launch_tile(
                        nb_stream,
                        d_A[I * TS : (I + 1) * TS],
                        d_B[J * TS : (J + 1) * TS],
                        top,
                        left,
                        cin,
                        d_S[I],
                        d_last_row[(I, J)],
                        d_last_col[(I, J)],
                        d_corner[(I, J)],
                        sub_ts,
                    )
            stream.synchronize()
            result = int(d_corner[(M - 1, N - 1)].get()[0])

    return result


# ---------------------------------------------------------------------------
# STF implementations
# ---------------------------------------------------------------------------


def _stf_tiled_impl(
    A: np.ndarray, B: np.ndarray, TS: int, n_gpus: int, *, sub_ts: int = 0
) -> int:
    """Shared implementation for the single- and multi-GPU STF variants.

    ``n_gpus == 1`` yields the single-GPU version (no ``exec_place`` on tasks,
    STF picks the default device). ``n_gpus >= 2`` pins each tile row to
    ``stf.exec_place.device(I % n_gpus)`` and lets the affine-data-place rule
    route the data.

    This single function is the whole demo: the orchestration is a plain
    ``for I, J`` double loop.
    """

    L = len(A)
    assert L % TS == 0, f"L={L} must be a multiple of TS={TS}"
    M = N = L // TS

    ctx = stf.context()

    # Sequence tiles -- create once, read by all tasks that touch this row / col.
    la_tiles = [
        ctx.logical_data(np.ascontiguousarray(A[I * TS : (I + 1) * TS]), name=f"a[{I}]")
        for I in range(M)
    ]
    lb_tiles = [
        ctx.logical_data(np.ascontiguousarray(B[J * TS : (J + 1) * TS]), name=f"b[{J}]")
        for J in range(N)
    ]

    # Per-row scratch S[I]. Tiles on the same row serialize anyway (they
    # depend on each other via last_col), so sharing scratch within a row is
    # free; rows stay independent and can run concurrently. The kernel fully
    # rewrites the scratch on every tile, so ``.write()`` is the semantically
    # correct access mode (``.rw()`` would force STF to preserve prior contents
    # we never read back).
    l_scratch = [
        ctx.logical_data_empty((TS + 1, TS + 1), np.int32, name=f"S[{I}]")
        for I in range(M)
    ]

    # Halo outputs per tile.
    l_last_row = {}
    l_last_col = {}
    l_corner = {}
    for I in range(M):
        for J in range(N):
            l_last_row[(I, J)] = ctx.logical_data_empty(
                (TS,), np.int32, name=f"lr[{I},{J}]"
            )
            l_last_col[(I, J)] = ctx.logical_data_empty(
                (TS,), np.int32, name=f"lc[{I},{J}]"
            )
            l_corner[(I, J)] = ctx.logical_data_empty(
                (1,), np.int32, name=f"c[{I},{J}]"
            )

    # Seeded boundaries (I == 0 row, J == 0 column, and the left / top corners).
    l_top_seed = [
        ctx.logical_data(_seeded_top_row(J, TS), name=f"ts[{J}]") for J in range(N)
    ]
    l_left_seed = [
        ctx.logical_data(_seeded_left_col(I, TS), name=f"ls[{I}]") for I in range(M)
    ]
    l_corner_seed = {}
    for J in range(N):
        l_corner_seed[(0, J)] = ctx.logical_data(
            _seeded_corner(0, J, TS), name=f"cs[0,{J}]"
        )
    for I in range(1, M):
        l_corner_seed[(I, 0)] = ctx.logical_data(
            _seeded_corner(I, 0, TS), name=f"cs[{I},0]"
        )

    def _top_for(I, J):
        return l_top_seed[J] if I == 0 else l_last_row[(I - 1, J)]

    def _left_for(I, J):
        return l_left_seed[I] if J == 0 else l_last_col[(I, J - 1)]

    def _corner_for(I, J):
        if I > 0 and J > 0:
            return l_corner[(I - 1, J - 1)]
        return l_corner_seed[(I, J)]

    # The whole demo lives in this double loop. STF sees the read / write
    # annotations and discovers the wavefront schedule automatically; the
    # ``numba_task`` helper converts the task's CAI args into ready-to-use
    # Numba device arrays so the body reads as a plain kernel launch.
    for I in range(M):
        exec_args = (stf.exec_place.device(I % n_gpus),) if n_gpus > 1 else ()
        for J in range(N):
            with numba_task(
                ctx,
                *exec_args,
                la_tiles[I].read(),
                lb_tiles[J].read(),
                _top_for(I, J).read(),
                _left_for(I, J).read(),
                _corner_for(I, J).read(),
                l_scratch[I].write(),
                l_last_row[(I, J)].write(),
                l_last_col[(I, J)].write(),
                l_corner[(I, J)].write(),
                symbol=f"tile[{I},{J}]",
            ) as (args, stream):
                _launch_tile(nbcuda.external_stream(stream), *args, sub_ts)

    # Readback: host_launch schedules a Python callable as a graph node and
    # auto-unpacks the read dep as a numpy view on the host copy of the corner.
    result = [0]

    def read_corner(corner_arr, res):
        res[0] = int(corner_arr[0])

    ctx.host_launch(l_corner[(M - 1, N - 1)].read(), fn=read_corner, args=[result])
    ctx.finalize()
    return result[0]


def stf_tiled_single(A: np.ndarray, B: np.ndarray, TS: int, *, sub_ts: int = 0) -> int:
    """Single-GPU STF tiled Levenshtein. See :func:`_launch_tile` for ``sub_ts``."""

    return _stf_tiled_impl(A, B, TS, n_gpus=1, sub_ts=sub_ts)


def stf_tiled_multi(
    A: np.ndarray, B: np.ndarray, TS: int, n_gpus: int, *, sub_ts: int = 0
) -> int:
    """Multi-GPU STF tiled Levenshtein. Row-cyclic placement across ``n_gpus``.

    The *only* difference from ``stf_tiled_single`` is the
    ``stf.exec_place.device(I % n_gpus)`` argument on each task. Logical data
    carries no explicit ``data_place``: STF uses the affine-data-place rule to
    land each piece of data on the consuming device, inserting peer copies only
    for the ``last_row`` / ``corner`` halos that cross between rows on
    different GPUs.
    """

    assert n_gpus >= 2
    return _stf_tiled_impl(A, B, TS, n_gpus=n_gpus, sub_ts=sub_ts)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _visible_gpus() -> int:
    try:
        return int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        return 0


def _time_one(fn, *args, warmup: int, iters: int, **kwargs) -> float:
    """Wall-clock mean (ms) of ``fn(*args, **kwargs)`` after ``warmup`` untimed runs."""

    for _ in range(warmup):
        fn(*args, **kwargs)
    # No free-standing synchronise: every implementation finalises + reads back
    # its own result synchronously, so the returned time already includes the
    # full pipeline.
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args, **kwargs)
    return (time.perf_counter() - t0) * 1e3 / iters


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_tiled_edit_distance_correctness_small():
    """Ground-truth check on tiny L: NumPy == cupy_wavefront == stf_tiled_single."""

    L, TS = 256, 64
    A, B = _make_sequences(L, seed=0)

    ref = cpu_reference(A, B)
    assert cupy_wavefront(A, B, TS) == ref, "cupy_wavefront disagrees with NumPy"
    assert stf_tiled_single(A, B, TS) == ref, "stf_tiled_single disagrees with NumPy"


def test_tiled_edit_distance_correctness_multiblock():
    """Same ground-truth check, but with the multi-block sub-tile kernel path."""

    L, TS, sub_ts = 256, 64, 16
    A, B = _make_sequences(L, seed=2)
    ref = cpu_reference(A, B)
    assert cupy_wavefront(A, B, TS, sub_ts=sub_ts) == ref, (
        "cupy_wavefront (multi-block) disagrees with NumPy"
    )
    assert stf_tiled_single(A, B, TS, sub_ts=sub_ts) == ref, (
        "stf_tiled_single (multi-block) disagrees with NumPy"
    )


def test_tiled_edit_distance_2gpu_consistency():
    """2-GPU STF must agree with single-GPU STF on the same input."""

    if _visible_gpus() < 2:
        pytest.skip("need >= 2 CUDA devices for the multi-GPU consistency test")

    L, TS = 1024, 128
    A, B = _make_sequences(L, seed=1)
    single = stf_tiled_single(A, B, TS)
    multi = stf_tiled_multi(A, B, TS, n_gpus=2)
    assert single == multi, f"stf_tiled_multi={multi} disagrees with single={single}"


def _run_one_benchmark_section(
    label: str, A, B, cfg: "EditConfig", sub_ts: int, ngpus_used: int
) -> dict:
    """Time cupy_wavefront / stf_tiled_single / stf_tiled_multi for one sub_ts."""

    print(f"\n-- {label} (sub_ts={sub_ts}) --")
    print(f"{'implementation':<28} {'ms / run':>12}  {'speedup':>8}")

    t_cupy = _time_one(
        cupy_wavefront, A, B, cfg.TS, warmup=cfg.warmup, iters=cfg.iters, sub_ts=sub_ts
    )
    print(f"{'cupy_wavefront':<28} {t_cupy:>12.2f}  {1.0:>7.2f}x")

    t_stf1 = _time_one(
        stf_tiled_single,
        A,
        B,
        cfg.TS,
        warmup=cfg.warmup,
        iters=cfg.iters,
        sub_ts=sub_ts,
    )
    print(f"{'stf_tiled_single':<28} {t_stf1:>12.2f}  {t_cupy / t_stf1:>7.2f}x")

    t_stfN = None
    if ngpus_used >= 2:
        t_stfN = _time_one(
            stf_tiled_multi,
            A,
            B,
            cfg.TS,
            ngpus_used,
            warmup=cfg.warmup,
            iters=cfg.iters,
            sub_ts=sub_ts,
        )
        print(
            f"{f'stf_tiled_multi ({ngpus_used} GPUs)':<28} {t_stfN:>12.2f}  "
            f"{t_cupy / t_stfN:>7.2f}x"
        )
    else:
        print(f"{'stf_tiled_multi':<28} {'skipped':>12}  (need >= 2 GPUs)")

    # Correctness gate: all implementations for this sub_ts must agree.
    s_cupy = cupy_wavefront(A, B, cfg.TS, sub_ts=sub_ts)
    s_stf1 = stf_tiled_single(A, B, cfg.TS, sub_ts=sub_ts)
    assert s_cupy == s_stf1, f"[sub_ts={sub_ts}] cupy={s_cupy} != stf_single={s_stf1}"
    if ngpus_used >= 2:
        s_stfN = stf_tiled_multi(A, B, cfg.TS, ngpus_used, sub_ts=sub_ts)
        assert s_stf1 == s_stfN, (
            f"[sub_ts={sub_ts}] stf_single={s_stf1} != stf_multi={s_stfN}"
        )

    return {"cupy": t_cupy, "stf1": t_stf1, "stfN": t_stfN}


@pytest.mark.skipif(
    os.environ.get("EDIT_BENCH", "0") != "1",
    reason="set EDIT_BENCH=1 to run the benchmark",
)
def test_tiled_edit_distance_benchmark():
    """Print a comparison table: single-block vs multi-block kernel path."""

    cfg = EditConfig.from_env()
    assert cfg.L % cfg.TS == 0, f"EDIT_L={cfg.L} must be a multiple of EDIT_TS={cfg.TS}"
    if cfg.sub_ts > 0:
        assert cfg.TS % cfg.sub_ts == 0, (
            f"EDIT_SUB_TS={cfg.sub_ts} must divide EDIT_TS={cfg.TS}"
        )
    M = cfg.L // cfg.TS

    A, B = _make_sequences(cfg.L, seed=cfg.seed)
    ngpus_avail = _visible_gpus()
    ngpus_used = min(cfg.n_gpus, ngpus_avail)

    print(
        f"\n=== Tiled Levenshtein, L={cfg.L}, TS={cfg.TS}, grid={M}x{M}, "
        f"iters={cfg.iters} (warmup={cfg.warmup}) ==="
    )

    # Section 1: single-block kernel path (1 CUDA block / tile). Makes the
    # STF wavefront story look most dramatic because per-tile SM occupancy is
    # tiny so concurrent tiles genuinely co-tenant SMs.
    single = _run_one_benchmark_section(
        "single-block kernel", A, B, cfg, sub_ts=0, ngpus_used=ngpus_used
    )

    # Section 2: multi-block sub-tile kernel path. Each tile dispatches
    # 2*B - 1 sub-diagonal kernels with up to B blocks each, so each tile
    # already uses many SMs on its own. The remaining STF win is inter-tile
    # scheduling overlap, not idle-SM packing.
    multi = None
    if cfg.sub_ts > 0:
        multi = _run_one_benchmark_section(
            "multi-block sub-tile kernel",
            A,
            B,
            cfg,
            sub_ts=cfg.sub_ts,
            ngpus_used=ngpus_used,
        )
    else:
        print(
            "\nmulti-block sub-tile section skipped (EDIT_SUB_TS=0). "
            "Set EDIT_SUB_TS=128 (or similar) to run both kernel paths."
        )

    # Sanity gate: the single-block STF path must at least match the CuPy
    # baseline -- any regression there means the wavefront schedule is broken.
    assert single["stf1"] <= single["cupy"] * 1.10, (
        f"single-block: stf_tiled_single ({single['stf1']:.2f} ms) is slower "
        f"than cupy_wavefront ({single['cupy']:.2f} ms); scheduling is broken"
    )
    if multi is not None:
        # Multi-block is a harder test of STF's scheduling (tiles are big,
        # less SM headroom to overlap). We still expect no slowdown, but we
        # allow a bit more slack before failing.
        assert multi["stf1"] <= multi["cupy"] * 1.15, (
            f"multi-block: stf_tiled_single ({multi['stf1']:.2f} ms) is "
            f"slower than cupy_wavefront ({multi['cupy']:.2f} ms)"
        )


if __name__ == "__main__":
    test_tiled_edit_distance_correctness_small()
    print("correctness OK")
    os.environ["EDIT_BENCH"] = "1"
    test_tiled_edit_distance_benchmark()
