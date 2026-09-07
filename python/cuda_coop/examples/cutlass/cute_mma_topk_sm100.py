# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Fuse ``cuda.coop.cutlass`` TopK into a Blackwell SM100/SM103 GEMM epilogue.

This example dynamically imports CUTLASS's Blackwell ``DenseGemmKernel``
sample.  The kernel performs ``tcgen05.mma`` into TMEM and owns the policy for
the corresponding ``tcgen05.ld`` TMEM-to-RMEM transfer: its copy atom, thread
partition, register layout, readiness, and lifetime.

The default ``tmem_loader`` mode passes ``TileTopKConsumer`` through the
kernel's ``accumulator_consumer`` hook.  CUTLASS gives it a producer-owned
source capability, and ``ThreadData.load`` invokes that source's load hook to
trigger the exact ``cute.copy`` selected by CUTLASS before calling BlockTopK.
The ``post_t2r`` mode is an explicit baseline: CUTLASS triggers the transfer
first, then the ordinary epilogue callback wraps its register ``TensorSSA`` in
``PostT2RAccumulatorSource`` for the same ``ThreadData.load`` and TopK body.
No full GEMM result is materialized before selection in either mode.

The stock kernel releases its mainloop shared-memory partition before the
epilogue.  This example therefore uses a single 128 x 32 TMA-store epilogue
tile: its post-release shared-memory budget accommodates both C staging and
BlockTopK scratch, and the callback still sees the complete output tile in one
invocation.

The stock ``DenseGemmKernel`` emits ``setsmemsize`` for early shared-memory
release.  A CUTLASS DSL build that preserves this instruction's PTX extension
descriptor through external LTO linking is therefore required; releases
without that fix cannot final-link this example.

The inputs are deterministic and produce ``C[i, j] = i + j / 256`` for one
128 x 32 output tile.  The callback keeps the tile-wide Top8 values and
replaces every other output with ``NON_CANDIDATE``.  Runtime validation compares
the selected values with ``torch.topk(torch.bmm(A, B).flatten(), 8)``.

Run on an SM100 or SM103 GPU:

.. code-block:: bash

    python -m examples.cutlass.cute_mma_topk_sm100
    python -m examples.cutlass.cute_mma_topk_sm100 --selector warp_merge
    python -m examples.cutlass.cute_mma_topk_sm100 --mode post_t2r

Select the native architecture when compiling without a launch:

.. code-block:: bash

    CUTE_DSL_ARCH=sm_103a CUTE_DSL_DRYRUN=1 \
      python -m examples.cutlass.cute_mma_topk_sm100 --compile-only

Benchmark a launch with enough independent tiles to occupy the GPU:

.. code-block:: bash

    python -m examples.cutlass.cute_mma_topk_sm100 \
      --benchmark --batch-count 4096 --selector warp_merge

Compile without launching:

.. code-block:: bash

    CUTE_DSL_ARCH=sm_100a CUTE_DSL_DRYRUN=1 \
      python -m examples.cutlass.cute_mma_topk_sm100 --compile-only
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from examples.cutlass._runtime import require_runtime

M = 128
N = 32
K = 64
BATCH_COUNT = 1
TOPK = 8
BLOCK_THREADS = 128
WARP_THREADS = 32
ITEMS_PER_THREAD = M * N // BLOCK_THREADS
CHUNK_ITEMS_PER_THREAD = 8
CHUNKS_PER_THREAD = ITEMS_PER_THREAD // CHUNK_ITEMS_PER_THREAD
CANDIDATE_COUNT = CHUNKS_PER_THREAD * TOPK
MERGE_ITEMS_PER_THREAD = (CANDIDATE_COUNT + BLOCK_THREADS - 1) // BLOCK_THREADS
MMA_TILER_MN = (128, 32)
CLUSTER_SHAPE_MN = (1, 1)
USE_2CTA_INSTRS = False
USE_TMA_STORE = True
NON_CANDIDATE = -3.4028234663852886e38
TOPK_TEMP_STORAGE_BYTES = 10240
TMEM_LOADER_MODE = "tmem_loader"
POST_T2R_MODE = "post_t2r"
MODES = (TMEM_LOADER_MODE, POST_T2R_MODE)
BLOCK_MERGE_SELECTOR = "block_merge"
WARP_MERGE_SELECTOR = "warp_merge"
SELECTORS = (
    BLOCK_MERGE_SELECTOR,
    WARP_MERGE_SELECTOR,
)
SUPPORTED_COMPUTE_CAPABILITIES = ((10, 0), (10, 3))

DENSE_GEMM_RELATIVE_PATH = Path(
    "examples/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm.py"
)


def _load_dense_gemm_module() -> Any:
    """Load CUTLASS's stock Blackwell dense-GEMM sample from ``sys.path``."""

    for entry in sys.path:
        if not entry:
            continue
        candidate = Path(entry) / DENSE_GEMM_RELATIVE_PATH
        if not candidate.is_file():
            continue
        spec = importlib.util.spec_from_file_location(
            "_cuda_coop_blackwell_dense_gemm",
            candidate,
        )
        if spec is None or spec.loader is None:
            break
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    searched = ", ".join(
        str(Path(entry) / DENSE_GEMM_RELATIVE_PATH) for entry in sys.path if entry
    )
    raise RuntimeError(
        "cute_mma_topk_sm100 requires CUTLASS's Blackwell DenseGemmKernel "
        f"sample on PYTHONPATH; searched: {searched}"
    )


def _require_runtime() -> tuple[Any, ...]:
    return require_runtime(include_int32=True)


def _validate_mode(mode: str) -> str:
    if mode not in MODES:
        choices = ", ".join(MODES)
        raise ValueError(f"mode must be one of: {choices}; got {mode!r}")
    return mode


def _validate_selector(selector: str) -> str:
    if selector not in SELECTORS:
        choices = ", ".join(SELECTORS)
        raise ValueError(f"selector must be one of: {choices}; got {selector!r}")
    return selector


def _validate_batch_count(batch_count: int) -> int:
    if isinstance(batch_count, bool) or not isinstance(batch_count, int):
        raise TypeError("batch_count must be an integer")
    if batch_count <= 0:
        raise ValueError("batch_count must be positive")
    return batch_count


def _require_supported_compute_capability(torch: Any) -> tuple[int, int]:
    capability = tuple(torch.cuda.get_device_capability())
    if capability not in SUPPORTED_COMPUTE_CAPABILITIES:
        supported = ", ".join(
            f"{major}.{minor}" for major, minor in SUPPORTED_COMPUTE_CAPABILITIES
        )
        raise RuntimeError(
            "cute_mma_topk_sm100 requires an SM100 or SM103 GPU for execution; "
            f"supported compute capabilities are {supported}, found "
            f"{capability[0]}.{capability[1]}"
        )
    return capability


def _selector_geometry(selector: str) -> tuple[int, int, int, int]:
    _validate_selector(selector)
    return (
        CHUNK_ITEMS_PER_THREAD,
        CHUNKS_PER_THREAD,
        CANDIDATE_COUNT,
        TOPK_TEMP_STORAGE_BYTES,
    )


def _metadata(
    mode: str,
    selector: str,
    batch_count: int,
    *,
    device_compute_capability: tuple[int, int] | None = None,
    compile_target: str | None = None,
) -> dict[str, Any]:
    mode = _validate_mode(mode)
    selector = _validate_selector(selector)
    batch_count = _validate_batch_count(batch_count)
    (
        chunk_items_per_thread,
        chunk_count,
        candidate_count,
        temp_storage_bytes,
    ) = _selector_geometry(selector)
    is_tmem_loader = mode == TMEM_LOADER_MODE
    return {
        "mode": mode,
        "selector": selector,
        "batch_count": batch_count,
        "chunk_items_per_thread": chunk_items_per_thread,
        "chunk_count": chunk_count,
        "candidate_count": candidate_count,
        "temp_storage_bytes": temp_storage_bytes,
        "mma": "tcgen05.mma",
        "transfer": "tcgen05.ld",
        "transfer_policy_owner": "cutlass_dense_gemm",
        "transfer_trigger": (
            "thread_data_load_source_hook"
            if is_tmem_loader
            else "cutlass_dense_gemm_epilogue"
        ),
        "thread_data_source": (
            "producer_tmem_accumulator" if is_tmem_loader else "post_t2r_register"
        ),
        "shape_mnk": (M, N, K),
        "topk": TOPK,
        "block_threads": BLOCK_THREADS,
        "cluster_shape_mn": CLUSTER_SHAPE_MN,
        "use_2cta_instrs": USE_2CTA_INSTRS,
        "use_tma_store": USE_TMA_STORE,
        "device_compute_capability": device_compute_capability,
        "compile_target": compile_target,
    }


@dataclass(frozen=True)
class PreparedExample:
    metadata: dict[str, Any]
    step: Callable[[], None]
    synchronize: Callable[[], None]
    validate: Callable[[], dict[str, Any]]
    measure_cuda_event_us: Callable[..., float]


@dataclass(frozen=True)
class PostT2RAccumulatorSource:
    """Expose a producer-owned, already-loaded register accumulator payload.

    Stock ``DenseGemmKernel`` calls the epilogue after its selected T2R copy.
    This baseline capability therefore adapts only that post-LDTM register
    value.  The default producer source implements the same protocol while
    carrying the copy plan and enforcing readiness/lifetime itself.
    """

    register_payload: Any

    @property
    def shape(self) -> Any:
        """Static shape of the already-loaded register payload."""

        return self.register_payload.shape

    @property
    def dtype(self) -> Any:
        """Element type of the already-loaded register payload."""

        return self.register_payload.dtype

    def __cuda_coop_thread_data_load__(self) -> Any:
        """Return the register payload for ``ThreadData.load`` to adapt."""

        return self.register_payload


@functools.lru_cache(maxsize=None)
def make_runner(
    mode: str = TMEM_LOADER_MODE,
    selector: str = BLOCK_MERGE_SELECTOR,
    batch_count: int = BATCH_COUNT,
) -> PreparedExample:
    """Compile SM100 GEMM+TopK using the selected accumulator-source mode."""

    mode = _validate_mode(mode)
    selector = _validate_selector(selector)
    batch_count = _validate_batch_count(batch_count)
    (
        chunk_items_per_thread,
        chunk_count,
        candidate_count,
        _temp_storage_bytes,
    ) = _selector_geometry(selector)
    merge_items_per_thread = (candidate_count + BLOCK_THREADS - 1) // BLOCK_THREADS
    cutlass, cute, torch, from_dlpack, coop, Int32 = _require_runtime()
    dense_gemm = _load_dense_gemm_module()
    from cutlass.cute.runtime import make_fake_stream
    from cutlass.cutlass_dsl import BaseDSL

    from cuda.bindings import driver as cuda

    globals()["cutlass"] = cutlass
    globals()["cute"] = cute
    globals()["coop"] = coop
    globals()["Int32"] = Int32
    capability = tuple(torch.cuda.get_device_capability())
    compile_target = BaseDSL._get_dsl().get_arch_enum().name
    metadata = _metadata(
        mode,
        selector,
        batch_count,
        device_compute_capability=capability,
        compile_target=compile_target,
    )

    class TileTopKConsumer:
        """Select tile-wide TopK from either accumulator source capability."""

        @cute.jit
        def __call__(self, accumulator_source: Any) -> Any:
            tidx, _, _ = cute.arch.thread_idx()
            block = coop.this_block()
            accumulator_values = coop.ThreadData.load(accumulator_source)
            selected = coop.ThreadData.from_fn(
                ITEMS_PER_THREAD,
                lambda _item: cutlass.Float32(NON_CANDIDATE),
                dtype=cutlass.Float32,
            )

            candidate_ptr = cute.arch.alloc_smem(
                cutlass.Float32,
                candidate_count,
            )
            candidate_scores = cute.make_tensor(
                candidate_ptr,
                cute.make_layout((candidate_count,)),
            )

            # Each BlockTopK call handles one register chunk from every
            # thread. Their union is sufficient for the complete tile's TopK.
            for chunk in cutlass.range_constexpr(chunk_count):
                chunk_scores = coop.ThreadData.from_fn(
                    chunk_items_per_thread,
                    lambda item: accumulator_values[
                        chunk * chunk_items_per_thread + item
                    ],
                    dtype=cutlass.Float32,
                )
                chunk_top = coop.topk_max_keys(
                    block,
                    chunk_scores,
                    cutlass.const_expr(TOPK),
                )
                for item in cutlass.range_constexpr(chunk_items_per_thread):
                    chunk_position = tidx * cutlass.Int32(
                        chunk_items_per_thread
                    ) + cutlass.Int32(item)
                    if chunk_position < cutlass.Int32(TOPK):
                        candidate_position = (
                            cutlass.Int32(chunk * TOPK) + chunk_position
                        )
                        candidate_scores[candidate_position] = chunk_top[item]

            cute.arch.sync_threads()
            if cutlass.const_expr(selector == BLOCK_MERGE_SELECTOR):
                merge_input = coop.ThreadData.from_fn(
                    merge_items_per_thread,
                    lambda _item: cutlass.Float32(NON_CANDIDATE),
                    dtype=cutlass.Float32,
                )
                for item in cutlass.range_constexpr(merge_items_per_thread):
                    candidate_position = tidx * cutlass.Int32(
                        merge_items_per_thread
                    ) + cutlass.Int32(item)
                    value = cutlass.Float32(NON_CANDIDATE)
                    if candidate_position < cutlass.Int32(candidate_count):
                        value = candidate_scores[candidate_position]
                    merge_input[item] = value

                top_scores = coop.topk_max_keys(
                    block,
                    merge_input,
                    cutlass.const_expr(TOPK),
                    valid_items=cutlass.const_expr(candidate_count),
                    begin_bit=0,
                    end_bit=None,
                )
                for item in cutlass.range_constexpr(merge_items_per_thread):
                    selected_position = tidx * cutlass.Int32(
                        merge_items_per_thread
                    ) + cutlass.Int32(item)
                    if selected_position < cutlass.Int32(TOPK):
                        candidate_scores[selected_position] = top_scores[item]
            else:
                # One physical warp can sort the 32 candidates without
                # involving the rest of the CTA in another block TopK. Build
                # the ThreadData on every lane so its traced type is stable.
                warp_merge_value = cutlass.Float32(NON_CANDIDATE)
                if tidx < cutlass.Int32(candidate_count):
                    warp_merge_value = candidate_scores[tidx]
                warp_merge_input = coop.ThreadData.from_values(
                    warp_merge_value,
                    dtype=cutlass.Float32,
                )
                warp_top_score = cutlass.Float32(NON_CANDIDATE)
                if tidx < cutlass.Int32(WARP_THREADS):
                    warp_top_score = coop.merge_sort_keys(
                        coop.this_warp(),
                        warp_merge_input,
                        descending=True,
                    )[0]
                if tidx < cutlass.Int32(TOPK):
                    candidate_scores[tidx] = warp_top_score

            cute.arch.sync_threads()
            for item in cutlass.range_constexpr(ITEMS_PER_THREAD):
                flat_item = tidx * cutlass.Int32(ITEMS_PER_THREAD) + cutlass.Int32(item)
                value = cutlass.Float32(NON_CANDIDATE)
                if flat_item < cutlass.Int32(TOPK):
                    value = candidate_scores[flat_item]
                selected[item] = value
            return selected.to_tensor_ssa(like=accumulator_source)

    class IdentityEpilogue:
        """Leave the accumulator-consumer result unchanged."""

        @cute.jit
        def __call__(self, value: Any) -> Any:
            return value

    class PostT2RTopKAdapter:
        """Adapt the stock post-LDTM epilogue callback to the source protocol."""

        def __init__(self, consumer: Any):
            self.consumer = consumer

        @cute.jit
        def __call__(self, accumulator: Any) -> Any:
            return self.consumer(PostT2RAccumulatorSource(accumulator))

    row = torch.arange(M, device="cuda", dtype=torch.float16)
    column = torch.arange(N, device="cuda", dtype=torch.float16)
    a = torch.zeros(
        (batch_count, M, K),
        device="cuda",
        dtype=torch.float16,
    )
    b = torch.zeros(
        (batch_count, K, N),
        device="cuda",
        dtype=torch.float16,
    )
    a[:, :, 0] = row
    a[:, :, 1] = 1
    b[:, 0, :] = 1
    b[:, 1, :] = column / 256
    candidates = torch.empty(
        (batch_count, M, N),
        device="cuda",
        dtype=torch.float32,
    )

    a_arg = from_dlpack(a, assumed_align=16).mark_layout_dynamic(leading_dim=2)
    b_arg = from_dlpack(b, assumed_align=16).mark_layout_dynamic(leading_dim=2)
    candidates_arg = from_dlpack(candidates, assumed_align=16).mark_layout_dynamic(
        leading_dim=2
    )

    gemm = dense_gemm.DenseGemmKernel(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=USE_2CTA_INSTRS,
        mma_tiler_mn=MMA_TILER_MN,
        cluster_shape_mn=CLUSTER_SHAPE_MN,
        use_tma_store=USE_TMA_STORE,
    )
    topk_consumer = TileTopKConsumer()
    if mode == TMEM_LOADER_MODE:
        epilogue = IdentityEpilogue()
        accumulator_consumer = topk_consumer
    else:
        epilogue = PostT2RTopKAdapter(topk_consumer)
        accumulator_consumer = None
    compile_args = (
        dense_gemm.bmm,
        gemm,
        a_arg,
        b_arg,
        candidates_arg,
        make_fake_stream(),
        epilogue,
    )
    if accumulator_consumer is not None:
        compile_args = (*compile_args, accumulator_consumer)
    compiled = cute.compile(*compile_args)

    def step() -> None:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled(a_arg, b_arg, candidates_arg, stream)

    def synchronize() -> None:
        torch.cuda.synchronize()

    def validate() -> dict[str, Any]:
        synchronize()
        expected = torch.topk(
            torch.bmm(a.float(), b.float()).flatten(start_dim=1),
            TOPK,
            dim=1,
            sorted=False,
        ).values
        actual = torch.topk(
            candidates.flatten(start_dim=1),
            TOPK,
            dim=1,
            sorted=False,
        ).values
        actual_sorted = torch.sort(actual.float(), dim=1).values
        expected_sorted = torch.sort(expected.float(), dim=1).values
        torch.testing.assert_close(
            actual_sorted,
            expected_sorted,
            atol=0,
            rtol=0,
        )
        non_sentinel_counts = torch.count_nonzero(
            candidates != NON_CANDIDATE,
            dim=(1, 2),
        )
        if not bool(torch.all(non_sentinel_counts == TOPK).item()):
            raise AssertionError(
                f"expected {TOPK} selected outputs per batch, got "
                f"{non_sentinel_counts.detach().cpu().tolist()}"
            )
        return {
            **metadata,
            "top_scores": [
                float(value) for value in actual_sorted[0].detach().cpu().tolist()
            ],
            "expected_scores": [
                float(value) for value in expected_sorted[0].detach().cpu().tolist()
            ],
            "non_sentinel_count": int(non_sentinel_counts[0].item()),
            "non_sentinel_counts_match": True,
        }

    def measure_cuda_event_us(
        *,
        warmup_iters: int = 16,
        measure_iters: int = 100,
    ) -> float:
        warmup_iters = max(1, int(warmup_iters))
        measure_iters = max(1, int(measure_iters))
        for _ in range(warmup_iters):
            step()
        synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(measure_iters):
            step()
        end.record()
        synchronize()
        return start.elapsed_time(end) * 1000.0 / measure_iters

    return PreparedExample(
        metadata=metadata,
        step=step,
        synchronize=synchronize,
        validate=validate,
        measure_cuda_event_us=measure_cuda_event_us,
    )


def compile_example(
    mode: str = TMEM_LOADER_MODE,
    selector: str = BLOCK_MERGE_SELECTOR,
    batch_count: int = BATCH_COUNT,
) -> dict[str, Any]:
    """Compile the fused SM100/SM103 kernel without launching it."""

    mode = _validate_mode(mode)
    selector = _validate_selector(selector)
    batch_count = _validate_batch_count(batch_count)
    prepared = make_runner(mode, selector, batch_count)
    return dict(prepared.metadata)


def run_example(
    mode: str = TMEM_LOADER_MODE,
    selector: str = BLOCK_MERGE_SELECTOR,
    batch_count: int = BATCH_COUNT,
) -> dict[str, Any]:
    """Run and validate the fused SM100/SM103 TMEM-to-RMEM GEMM+TopK example."""

    mode = _validate_mode(mode)
    selector = _validate_selector(selector)
    batch_count = _validate_batch_count(batch_count)
    _, _, torch, _, _, _ = _require_runtime()
    _require_supported_compute_capability(torch)
    prepared = make_runner(mode, selector, batch_count)
    prepared.step()
    return prepared.validate()


def benchmark_example(
    mode: str = TMEM_LOADER_MODE,
    selector: str = BLOCK_MERGE_SELECTOR,
    batch_count: int = 4096,
    *,
    warmup_iters: int = 16,
    measure_iters: int = 100,
) -> dict[str, Any]:
    """Measure steady GPU time for independent GEMM+TopK tiles."""

    mode = _validate_mode(mode)
    selector = _validate_selector(selector)
    batch_count = _validate_batch_count(batch_count)
    _, _, torch, _, _, _ = _require_runtime()
    _require_supported_compute_capability(torch)
    prepared = make_runner(mode, selector, batch_count)
    prepared.step()
    validation = prepared.validate()
    kernel_us = prepared.measure_cuda_event_us(
        warmup_iters=warmup_iters,
        measure_iters=measure_iters,
    )
    return {
        **validation,
        "kernel_us": kernel_us,
        "output_tiles_per_second": batch_count * 1.0e6 / kernel_us,
        "warmup_iters": max(1, int(warmup_iters)),
        "measure_iters": max(1, int(measure_iters)),
    }


def prepare_block_merge_benchmark() -> PreparedExample:
    """Prepare the occupancy-scaled known-good selector benchmark."""

    return make_runner(
        TMEM_LOADER_MODE,
        BLOCK_MERGE_SELECTOR,
        batch_count=4096,
    )


def prepare_warp_merge_benchmark() -> PreparedExample:
    """Prepare the occupancy-scaled one-warp candidate-merge benchmark."""

    return make_runner(
        TMEM_LOADER_MODE,
        WARP_MERGE_SELECTOR,
        batch_count=4096,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="compile the fused kernel without launching it",
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default=TMEM_LOADER_MODE,
        help="choose who triggers the producer-selected TMEM-to-register transfer",
    )
    parser.add_argument(
        "--selector",
        choices=SELECTORS,
        default=BLOCK_MERGE_SELECTOR,
        help="choose the per-thread chunking and final TopK merge strategy",
    )
    parser.add_argument(
        "--batch-count",
        type=int,
        default=BATCH_COUNT,
        help="number of independent GEMM output tiles",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="report CUDA-event time after warmup",
    )
    parser.add_argument("--warmup-iters", type=int, default=16)
    parser.add_argument("--measure-iters", type=int, default=100)
    args = parser.parse_args(argv)
    if args.compile_only:
        result = compile_example(args.mode, args.selector, args.batch_count)
    elif args.benchmark:
        result = benchmark_example(
            args.mode,
            args.selector,
            args.batch_count,
            warmup_iters=args.warmup_iters,
            measure_iters=args.measure_iters,
        )
    else:
        result = run_example(args.mode, args.selector, args.batch_count)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
