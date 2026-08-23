# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Real mixed-backend kernels used by the activation integration test."""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata, util
from pathlib import Path
from threading import Barrier

import numpy as np
import torch
from cutlass import cute
from cutlass.cute import runtime as cute_runtime
from numba_cuda_mlir import cuda as numba_cuda
from numba_cuda_mlir import types

import cuda.coop.cutlass as cutlass_coop
import cuda.coop.numba_mlir as numba_coop
from cuda import coop
from cuda.coop._core import root_api

_THREADS = 32


def _assert_cuda_coop_provenance() -> None:
    """Prove installed-wheel lanes did not substitute the source package."""

    if os.environ.get("CUDA_COOP_EXAMPLES_USE_INSTALLED_CUDA_COOP") != "1":
        return
    spec = util.find_spec("cuda.coop._core")
    assert spec is not None and spec.origin is not None
    expected = metadata.distribution("cuda-coop").locate_file(
        "cuda/coop/_core/__init__.py"
    )
    assert Path(spec.origin).resolve() == Path(expected).resolve()
    assert Path(spec.origin).resolve().is_relative_to(Path(sys.prefix).resolve())


@cute.kernel
def _cutlass_kernel(
    values_in: cute.Tensor,
    common_out: cute.Tensor,
    qualified_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    common_out[tidx] = coop.sum(coop.this_block(), values_in[tidx])
    qualified_out[tidx] = cutlass_coop.sum(cutlass_coop.this_block(), values_in[tidx])


@cute.jit
def _run_cutlass_kernel(
    values_in: cute.Tensor,
    common_out: cute.Tensor,
    qualified_out: cute.Tensor,
):
    _cutlass_kernel(values_in, common_out, qualified_out).launch(
        grid=(1, 1, 1),
        block=(_THREADS, 1, 1),
    )


@cute.kernel
def _cutlass_pair_kernel(
    keys_in: cute.Tensor,
    values_in: cute.Tensor,
    keys_out: cute.Tensor,
    values_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    keys = coop.ThreadData(1, dtype=int)
    values = coop.ThreadData(1, dtype=np.float32)
    keys[0] = keys_in[tidx]
    values[0] = values_in[tidx]
    block = coop.this_block()
    merge_keys, merge_values = coop.merge_sort_pairs(block, keys, values)
    radix_keys, radix_values = coop.radix_sort_pairs(block, keys, values)
    max_keys, max_values = coop.topk_max_pairs(block, keys, values, 1)
    min_keys, min_values = coop.topk_min_pairs(block, keys, values, 1)
    keys_out[tidx] = merge_keys[0] + radix_keys[0]
    values_out[tidx] = merge_values[0] + radix_values[0]
    if tidx == 0:
        keys_out[tidx] += max_keys[0] + min_keys[0]
        values_out[tidx] += max_values[0] + min_values[0]


@cute.jit
def _run_cutlass_pair_kernel(
    keys_in: cute.Tensor,
    values_in: cute.Tensor,
    keys_out: cute.Tensor,
    values_out: cute.Tensor,
):
    _cutlass_pair_kernel(keys_in, values_in, keys_out, values_out).launch(
        grid=(1, 1, 1),
        block=(_THREADS, 1, 1),
    )


@numba_cuda.jit
def _numba_kernel(values_in, common_out, qualified_out):
    tidx = numba_cuda.threadIdx.x
    common_out[tidx] = coop.sum(coop.this_block(), values_in[tidx])
    qualified_out[tidx] = numba_coop.sum(numba_coop.this_block(), values_in[tidx])


@numba_cuda.jit
def _numba_pair_kernel(keys_in, values_in, keys_out, values_out):
    tidx = numba_cuda.threadIdx.x
    keys = coop.ThreadData(1, dtype=types.int32)
    values = coop.ThreadData(1, dtype=types.float32)
    keys[0] = keys_in[tidx]
    values[0] = values_in[tidx]
    block = coop.this_block()
    merge_keys, merge_values = coop.merge_sort_pairs(block, keys, values)
    radix_keys, radix_values = coop.radix_sort_pairs(block, keys, values)
    max_keys, max_values = coop.topk_max_pairs(block, keys, values, 1)
    min_keys, min_values = coop.topk_min_pairs(block, keys, values, 1)
    keys_out[tidx] = merge_keys[0] + radix_keys[0]
    values_out[tidx] = merge_values[0] + radix_values[0]
    if tidx == 0:
        keys_out[tidx] += max_keys[0] + min_keys[0]
        values_out[tidx] += max_values[0] + min_values[0]


def _pair_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keys = ((np.arange(_THREADS, dtype=np.int32) * 13 + 7) % _THREADS).astype(np.int32)
    values = (keys * 3).astype(np.float32)
    expected_keys = 2 * np.arange(_THREADS, dtype=np.int32)
    expected_values = 6 * np.arange(_THREADS, dtype=np.float32)
    expected_keys[0] += _THREADS - 1
    expected_values[0] += 3 * (_THREADS - 1)
    return keys, values, expected_keys, expected_values


def _assert_compiler_scope_is_clear() -> None:
    assert root_api._ACTIVE_BACKEND_MODULE.get() is None


def _compile_cutlass() -> tuple[int, int]:
    import cutlass

    cutlass.cuda.initialize_cuda_context()
    values_host = torch.arange(1, _THREADS + 1, dtype=torch.int32)
    values_in = values_host.cuda()
    common_out = torch.zeros_like(values_in)
    qualified_out = torch.zeros_like(values_in)
    _run_cutlass_kernel(
        cute_runtime.from_dlpack(values_in),
        cute_runtime.from_dlpack(common_out),
        cute_runtime.from_dlpack(qualified_out),
    )
    torch.cuda.synchronize()

    expected = torch.full_like(values_host, int(values_host.sum()))
    torch.testing.assert_close(common_out.cpu(), expected, atol=0, rtol=0)
    torch.testing.assert_close(qualified_out.cpu(), expected, atol=0, rtol=0)

    pair_keys, pair_values, expected_keys, expected_values = _pair_inputs()
    pair_keys_in = torch.from_numpy(pair_keys).cuda()
    pair_values_in = torch.from_numpy(pair_values).cuda()
    pair_keys_out = torch.zeros_like(pair_keys_in)
    pair_values_out = torch.zeros_like(pair_values_in)
    _run_cutlass_pair_kernel(
        cute_runtime.from_dlpack(pair_keys_in),
        cute_runtime.from_dlpack(pair_values_in),
        cute_runtime.from_dlpack(pair_keys_out),
        cute_runtime.from_dlpack(pair_values_out),
    )
    torch.cuda.synchronize()
    np.testing.assert_array_equal(pair_keys_out.cpu().numpy(), expected_keys)
    np.testing.assert_array_equal(pair_values_out.cpu().numpy(), expected_values)
    _assert_compiler_scope_is_clear()
    return int(common_out.sum()), int(qualified_out.sum())


def _compile_numba() -> tuple[int, int]:
    values_in = np.arange(1, _THREADS + 1, dtype=np.int32)
    common_out = np.zeros_like(values_in)
    qualified_out = np.zeros_like(values_in)
    _numba_kernel[1, _THREADS](values_in, common_out, qualified_out)
    numba_cuda.synchronize()

    expected = np.full_like(values_in, values_in.sum())
    np.testing.assert_array_equal(common_out, expected)
    np.testing.assert_array_equal(qualified_out, expected)

    pair_keys, pair_values, expected_keys, expected_values = _pair_inputs()
    pair_keys_out = np.zeros_like(pair_keys)
    pair_values_out = np.zeros_like(pair_values)
    _numba_pair_kernel[1, _THREADS](
        pair_keys,
        pair_values,
        pair_keys_out,
        pair_values_out,
    )
    numba_cuda.synchronize()
    np.testing.assert_array_equal(pair_keys_out, expected_keys)
    np.testing.assert_array_equal(pair_values_out, expected_values)
    _assert_compiler_scope_is_clear()
    return int(common_out.sum()), int(qualified_out.sum())


def _run_in_order(first, second) -> None:
    first_result = first()
    second_result = second()
    assert first_result == second_result


def _run_concurrently() -> None:
    ready = Barrier(2)

    def run(compiler):
        ready.wait()
        return compiler()

    with ThreadPoolExecutor(max_workers=2) as executor:
        cutlass_future = executor.submit(run, _compile_cutlass)
        numba_future = executor.submit(run, _compile_numba)
        assert cutlass_future.result() == numba_future.result()
    _assert_compiler_scope_is_clear()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("cutlass-first", "numba-first", "concurrent"),
    )
    mode = parser.parse_args().mode

    _assert_cuda_coop_provenance()
    _assert_compiler_scope_is_clear()
    if mode == "cutlass-first":
        _run_in_order(_compile_cutlass, _compile_numba)
    elif mode == "numba-first":
        _run_in_order(_compile_numba, _compile_cutlass)
    else:
        _run_concurrently()
    _assert_compiler_scope_is_clear()


if __name__ == "__main__":
    main()
