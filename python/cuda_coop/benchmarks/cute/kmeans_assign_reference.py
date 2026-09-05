# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reference fused k-means assignment-shaped tensor-core baselines for benchmark comparison."""

from __future__ import annotations

import functools
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

from examples.cutlass._runtime import require_runtime

BATCHED_QUERY_COUNT = 4096
CENTROIDS_PER_TILE = 256
FEATURE_DIM = 128
ROW_MIN_ROWS_PER_CTA = 8
# Each score row occupies one complete physical warp. The group-first
# ``this_warp()`` reduction intentionally has no logical-subwarp width.
ROW_MIN_ROW_THREADS = 32
ROW_MIN_ITEMS_PER_THREAD = CENTROIDS_PER_TILE // ROW_MIN_ROW_THREADS
ROW_MIN_BLOCK_THREADS = ROW_MIN_ROWS_PER_CTA * ROW_MIN_ROW_THREADS
ROW_MIN_TOPK_K = 1
SM120_DENSE_GEMM_RELATIVE_PATH = Path(
    "examples/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py"
)


class PreparedReference(NamedTuple):
    step: Callable[[], None]
    synchronize: Callable[[], None]
    validate: Callable[[], dict[str, Any]]
    measure_cuda_event_us: Callable[..., float]


def _expected_nearest_centroids_from_inputs(
    query: Any,
    centroids: Any,
) -> Any:
    import torch

    # The benchmark materializes fp16 GEMM cross-terms before converting to
    # fp32 scores. Recompute the same input-derived formula independently of
    # the stored cross_terms/scores tensors so validation catches orientation
    # and score-correction bugs without changing the scenario's precision.
    independent_cross_terms = torch.mm(query, centroids.t())
    independent_centroid_norms = centroids.float().square().sum(dim=1)
    independent_scores = (
        independent_centroid_norms[None, :] - 2.0 * independent_cross_terms.float()
    )
    return independent_scores.argmin(dim=1)


def prepare_torch_gemm_argmin_reference() -> PreparedReference:
    """Prepare a tensor-core GEMM plus row-argmin reference baseline.

    This intentionally materializes the full score tile and launches separate
    PyTorch kernels. It is not a fused fused k-means assignment implementation; it is a
    performance reference for how close the assignment shape gets once the dot
    product is tensor-core backed.
    """

    import torch

    torch.manual_seed(0)
    query = torch.randn(
        (BATCHED_QUERY_COUNT, FEATURE_DIM),
        device="cuda",
        dtype=torch.float16,
    )
    centroids = torch.randn(
        (CENTROIDS_PER_TILE, FEATURE_DIM),
        device="cuda",
        dtype=torch.float16,
    )
    centroids_t = centroids.t().contiguous()
    centroid_norms = (centroids.float() * centroids.float()).sum(dim=1)

    cross_terms = torch.empty(
        (BATCHED_QUERY_COUNT, CENTROIDS_PER_TILE),
        device="cuda",
        dtype=torch.float16,
    )
    scores = torch.empty(
        (BATCHED_QUERY_COUNT, CENTROIDS_PER_TILE),
        device="cuda",
        dtype=torch.float32,
    )
    best_centroids = torch.empty(
        (BATCHED_QUERY_COUNT,),
        device="cuda",
        dtype=torch.int64,
    )
    has_run_step = False

    def step() -> None:
        nonlocal has_run_step
        torch.mm(query, centroids_t, out=cross_terms)
        torch.mul(cross_terms, -2.0, out=scores)
        torch.add(scores, centroid_norms[None, :], out=scores)
        torch.argmin(scores, dim=1, out=best_centroids)
        has_run_step = True

    def synchronize() -> None:
        torch.cuda.synchronize()

    def validate() -> dict[str, Any]:
        if not has_run_step:
            step()
        synchronize()
        expected = _expected_nearest_centroids_from_inputs(query, centroids)
        torch.testing.assert_close(best_centroids, expected)
        return {
            "query_count": BATCHED_QUERY_COUNT,
            "centroid_count": CENTROIDS_PER_TILE,
            "feature_dim": FEATURE_DIM,
            "topk_k": 1,
            "top_centroids": [int(x) for x in best_centroids[:5].cpu().tolist()],
        }

    def measure_cuda_event_us(*, warmup_iters: int, measure_iters: int) -> float:
        for _ in range(max(1, int(warmup_iters))):
            step()
        synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(max(1, int(measure_iters))):
            step()
        end.record()
        synchronize()
        return start.elapsed_time(end) * 1000.0 / max(1, int(measure_iters))

    return PreparedReference(
        step=step,
        synchronize=synchronize,
        validate=validate,
        measure_cuda_event_us=measure_cuda_event_us,
    )


def _load_sm120_gemm_kernel() -> Any:
    for entry in sys.path:
        if not entry:
            continue
        candidate = Path(entry) / SM120_DENSE_GEMM_RELATIVE_PATH
        if not candidate.is_file():
            continue
        spec = importlib.util.spec_from_file_location(
            "_cuda_coop_kmeans_assign_sm120_dense_gemm",
            candidate,
        )
        if spec is None or spec.loader is None:
            break
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.Sm120GemmKernel

    searched = ", ".join(
        str(Path(entry) / SM120_DENSE_GEMM_RELATIVE_PATH) for entry in sys.path if entry
    )
    raise RuntimeError(
        "kmeans_assign_cute_gemm_coop_argmin_reference requires the CUTLASS "
        f"CuTe DSL SM120 dense GEMM sample on PYTHONPATH; searched: {searched}"
    )


@functools.lru_cache(maxsize=1)
def _make_coop_row_min_runner() -> tuple[Any, ...]:
    """Build the CuTe row-min runner used after the tensor-core GEMM."""

    cutlass, cute, torch, from_dlpack, coop = require_runtime()

    globals()["cutlass"] = cutlass
    globals()["cute"] = cute
    globals()["coop"] = coop

    @cute.kernel
    def _row_min_kernel(
        cross_terms: cute.Tensor,
        centroid_norms: cute.Tensor,
        best_score_out: cute.Tensor,
        best_centroid_out: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        query_group_idx, _, _ = cute.arch.block_idx()
        row_in_cta = tidx // cutlass.Int32(ROW_MIN_ROW_THREADS)
        row_tidx = tidx - row_in_cta * cutlass.Int32(ROW_MIN_ROW_THREADS)
        query_idx = query_group_idx * cutlass.Int32(ROW_MIN_ROWS_PER_CTA) + row_in_cta
        centroid = row_tidx.to(cutlass.Int32)
        warp_idx = tidx // cutlass.Int32(ROW_MIN_ROW_THREADS)
        lane = tidx - warp_idx * cutlass.Int32(ROW_MIN_ROW_THREADS)

        cross_term = cross_terms[query_idx, centroid, 0].to(cutlass.Float32)
        best_score = centroid_norms[centroid] - cutlass.Float32(2.0) * cross_term
        best_centroid = centroid

        for item_idx in cutlass.range_constexpr(1, ROW_MIN_ITEMS_PER_THREAD):
            item_centroid = centroid + cutlass.Int32(item_idx * ROW_MIN_ROW_THREADS)
            item_cross_term = cross_terms[query_idx, item_centroid, 0].to(
                cutlass.Float32
            )
            item_score = (
                centroid_norms[item_centroid] - cutlass.Float32(2.0) * item_cross_term
            )
            if item_score < best_score:
                best_score = item_score
                best_centroid = item_centroid
            if item_score == best_score:
                if item_centroid < best_centroid:
                    best_centroid = item_centroid

        warp_best_score = coop.reduce(
            coop.this_warp(),
            best_score,
            binary_op="min",
        )
        if best_score != warp_best_score:
            best_centroid = cutlass.Int32(CENTROIDS_PER_TILE)
        warp_best_centroid = coop.reduce(
            coop.this_warp(),
            best_centroid,
            binary_op="min",
        )

        if lane == cutlass.Int32(0):
            best_score_out[query_idx] = warp_best_score
            best_centroid_out[query_idx] = warp_best_centroid

    @cute.jit
    def _run_row_min(
        cross_terms: cute.Tensor,
        centroid_norms: cute.Tensor,
        best_score_out: cute.Tensor,
        best_centroid_out: cute.Tensor,
    ):
        _row_min_kernel(
            cross_terms,
            centroid_norms,
            best_score_out,
            best_centroid_out,
        ).launch(
            grid=(BATCHED_QUERY_COUNT // ROW_MIN_ROWS_PER_CTA, 1, 1),
            block=(ROW_MIN_BLOCK_THREADS, 1, 1),
        )

    return _run_row_min, torch, from_dlpack, cutlass, cute


def prepare_cute_gemm_coop_argmin_reference() -> PreparedReference:
    """Prepare an SM120 CuTe GEMM plus cuda.coop row-argmin baseline.

    This keeps the tensor-core dot product in the stock Blackwell GeForce CuTe
    GEMM sample, then launches a small score-preserving warp-group reduction
    kernel over the materialized cross-term tile. It is still not the final
    fused GEMM epilogue, but it removes the PyTorch score-correction and argmin
    kernels from the reference path.
    """

    import cutlass.torch as cutlass_torch

    Sm120GemmKernel = _load_sm120_gemm_kernel()

    (
        run_row_min,
        torch,
        from_dlpack,
        cutlass,
        cute,
    ) = _make_coop_row_min_runner()
    cutlass.cuda.initialize_cuda_context()

    torch.manual_seed(0)
    a_dtype = cutlass.Float16
    b_dtype = cutlass.Float16
    c_dtype = cutlass.Float16
    acc_dtype = cutlass.Float32
    tile_shape_mnk = (128, 64, 64)
    a_mode0_major = True
    b_mode0_major = True
    c_mode0_major = False
    batch_count = 1

    def create_cute_tensor(data_ref: Any, cutlass_dtype: Any) -> tuple[Any, Any]:
        cute_tensor, torch_tensor = cutlass_torch.cute_tensor_like(
            data_ref,
            cutlass_dtype,
            True,
            16,
        )
        return cute_tensor, torch_tensor

    query_cpu = cutlass_torch.matrix(
        batch_count,
        BATCHED_QUERY_COUNT,
        FEATURE_DIM,
        a_mode0_major,
        a_dtype,
    )
    centroids_cpu = cutlass_torch.matrix(
        batch_count,
        CENTROIDS_PER_TILE,
        FEATURE_DIM,
        b_mode0_major,
        b_dtype,
    )
    cross_terms_cpu = cutlass_torch.matrix(
        batch_count,
        BATCHED_QUERY_COUNT,
        CENTROIDS_PER_TILE,
        c_mode0_major,
        c_dtype,
    )

    query_tensor, query_gpu = create_cute_tensor(query_cpu, a_dtype)
    centroids_tensor, centroids_gpu = create_cute_tensor(centroids_cpu, b_dtype)
    cross_terms_tensor, cross_terms_gpu = create_cute_tensor(cross_terms_cpu, c_dtype)

    centroid_norms = (
        centroids_gpu[:, :, 0].float() * centroids_gpu[:, :, 0].float()
    ).sum(dim=1)
    centroid_norms = centroid_norms.contiguous()
    best_scores = torch.empty(
        (BATCHED_QUERY_COUNT,),
        device="cuda",
        dtype=torch.float32,
    )
    best_centroids = torch.empty(
        (BATCHED_QUERY_COUNT,),
        device="cuda",
        dtype=torch.int32,
    )
    centroid_norms_arg = from_dlpack(centroid_norms)
    best_scores_arg = from_dlpack(best_scores)
    best_centroids_arg = from_dlpack(best_centroids)

    gemm = Sm120GemmKernel(acc_dtype, tile_shape_mnk)
    hardware_info = cutlass.utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(1)
    stream = cutlass_torch.default_stream()
    compiled_gemm = cute.compile(
        gemm,
        query_tensor,
        centroids_tensor,
        cross_terms_tensor,
        max_active_clusters,
        stream,
    )
    compiled_row_min = cute.compile(
        run_row_min,
        cross_terms_tensor,
        centroid_norms_arg,
        best_scores_arg,
        best_centroids_arg,
    )

    def step() -> None:
        compiled_gemm(query_tensor, centroids_tensor, cross_terms_tensor, stream)
        compiled_row_min(
            cross_terms_tensor,
            centroid_norms_arg,
            best_scores_arg,
            best_centroids_arg,
        )

    def synchronize() -> None:
        torch.cuda.synchronize()

    def validate() -> dict[str, Any]:
        synchronize()
        scores = centroid_norms[None, :] - 2.0 * cross_terms_gpu[:, :, 0].float()
        expected_scores, _ = torch.min(scores, dim=1)
        expected_centroids = _expected_nearest_centroids_from_inputs(
            query_gpu[:, :, 0],
            centroids_gpu[:, :, 0],
        )
        selected_scores = scores[
            torch.arange(BATCHED_QUERY_COUNT, device="cuda"),
            best_centroids.to(torch.int64),
        ]
        assert bool(torch.all(best_centroids >= 0).item())
        assert bool(torch.all(best_centroids < CENTROIDS_PER_TILE).item())
        torch.testing.assert_close(best_centroids.to(torch.int64), expected_centroids)
        torch.testing.assert_close(selected_scores, expected_scores, atol=0.0, rtol=0.0)
        torch.testing.assert_close(best_scores, expected_scores, atol=0.0, rtol=0.0)
        return {
            "query_count": BATCHED_QUERY_COUNT,
            "centroid_count": CENTROIDS_PER_TILE,
            "feature_dim": FEATURE_DIM,
            "topk_k": ROW_MIN_TOPK_K,
            "top_centroids": [int(x) for x in best_centroids[:5].cpu().tolist()],
        }

    def measure_cuda_event_us(*, warmup_iters: int, measure_iters: int) -> float:
        for _ in range(max(1, int(warmup_iters))):
            step()
        synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(max(1, int(measure_iters))):
            step()
        end.record()
        synchronize()
        return start.elapsed_time(end) * 1000.0 / max(1, int(measure_iters))

    return PreparedReference(
        step=step,
        synchronize=synchronize,
        validate=validate,
        measure_cuda_event_us=measure_cuda_event_us,
    )
