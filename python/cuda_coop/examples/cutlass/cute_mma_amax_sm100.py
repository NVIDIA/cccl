# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Fuse a cooperative absolute-maximum reduction into an SM100/SM103 GEMM.

CUTLASS performs ``tcgen05.mma`` into TMEM and owns the corresponding
TMEM-to-register copy. The default ``tmem_loader`` mode gives the epilogue
consumer a producer-owned source. ``ThreadData.load`` invokes that source's
copy before ``cuda.coop.cutlass.reduce`` computes one absolute maximum across
the complete ``128 x 32`` output tile.

The example broadcasts the tile amax back through CUTLASS's normal dense output
path. That makes the result easy to validate without adding a side-output ABI.
A quantizing epilogue would normally store one statistic per tensor or tile
instead.

The ``post_t2r`` mode lets CUTLASS issue the same copy before the callback, then
wraps the register payload in the same source protocol. Both modes execute the
same ThreadData and cooperative-reduction body.

Run on an SM100 or SM103 GPU:

.. code-block:: bash

    python -m examples.cutlass.cute_mma_amax_sm100
    python -m examples.cutlass.cute_mma_amax_sm100 --mode post_t2r

Select the native architecture when compiling without a launch:

.. code-block:: bash

    CUTE_DSL_ARCH=sm_103a CUTE_DSL_DRYRUN=1 \
      python -m examples.cutlass.cute_mma_amax_sm100 --compile-only

Benchmark enough independent output tiles to occupy the GPU:

.. code-block:: bash

    python -m examples.cutlass.cute_mma_amax_sm100 \
      --benchmark --batch-count 4096
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
BLOCK_THREADS = 128
ITEMS_PER_THREAD = M * N // BLOCK_THREADS
MMA_TILER_MN = (128, 32)
CLUSTER_SHAPE_MN = (1, 1)
USE_2CTA_INSTRS = False
USE_TMA_STORE = True
TMEM_LOADER_MODE = "tmem_loader"
POST_T2R_MODE = "post_t2r"
MODES = (TMEM_LOADER_MODE, POST_T2R_MODE)
SUPPORTED_COMPUTE_CAPABILITIES = ((10, 0), (10, 3))

DENSE_GEMM_RELATIVE_PATH = Path(
    "examples/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm.py"
)


def _load_dense_gemm_module() -> Any:
    """Load CUTLASS's Blackwell dense-GEMM sample from ``sys.path``."""

    for entry in sys.path:
        if not entry:
            continue
        candidate = Path(entry) / DENSE_GEMM_RELATIVE_PATH
        if not candidate.is_file():
            continue
        spec = importlib.util.spec_from_file_location(
            "_cuda_coop_blackwell_dense_gemm_amax",
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
        "cute_mma_amax_sm100 requires CUTLASS's Blackwell DenseGemmKernel "
        f"sample on PYTHONPATH; searched: {searched}"
    )


def _require_runtime() -> tuple[Any, ...]:
    return require_runtime()


def _validate_mode(mode: str) -> str:
    if mode not in MODES:
        choices = ", ".join(MODES)
        raise ValueError(f"mode must be one of: {choices}; got {mode!r}")
    return mode


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
            "cute_mma_amax_sm100 requires an SM100 or SM103 GPU for execution; "
            f"supported compute capabilities are {supported}, found "
            f"{capability[0]}.{capability[1]}"
        )
    return capability


def _metadata(
    mode: str,
    batch_count: int,
    *,
    device_compute_capability: tuple[int, int] | None = None,
    compile_target: str | None = None,
) -> dict[str, Any]:
    mode = _validate_mode(mode)
    batch_count = _validate_batch_count(batch_count)
    is_tmem_loader = mode == TMEM_LOADER_MODE
    return {
        "mode": mode,
        "batch_count": batch_count,
        "mma": "tcgen05.mma",
        "transfer": "tcgen05.ld",
        "cooperative_primitive": "block_reduce_max",
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
        "block_threads": BLOCK_THREADS,
        "items_per_thread": ITEMS_PER_THREAD,
        "cluster_shape_mn": CLUSTER_SHAPE_MN,
        "use_2cta_instrs": USE_2CTA_INSTRS,
        "use_tma_store": USE_TMA_STORE,
        "output_contract": "broadcast_tile_amax",
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
    """Adapt an already-loaded register accumulator to ``ThreadData.load``."""

    register_payload: Any

    @property
    def shape(self) -> Any:
        """Static shape of the register payload."""

        return self.register_payload.shape

    @property
    def dtype(self) -> Any:
        """Element type of the register payload."""

        return self.register_payload.dtype

    def __cuda_coop_thread_data_load__(self) -> Any:
        """Return the register payload selected and loaded by CUTLASS."""

        return self.register_payload


@functools.lru_cache(maxsize=None)
def make_runner(
    mode: str = TMEM_LOADER_MODE,
    batch_count: int = BATCH_COUNT,
) -> PreparedExample:
    """Compile SM100/SM103 GEMM+amax with the selected source mode."""

    mode = _validate_mode(mode)
    batch_count = _validate_batch_count(batch_count)
    cutlass, cute, torch, from_dlpack, coop = _require_runtime()
    dense_gemm = _load_dense_gemm_module()
    from cutlass.cute.runtime import make_fake_stream
    from cutlass.cutlass_dsl import BaseDSL

    from cuda.bindings import driver as cuda

    globals()["cutlass"] = cutlass
    globals()["cute"] = cute
    globals()["coop"] = coop
    capability = tuple(torch.cuda.get_device_capability())
    compile_target = BaseDSL._get_dsl().get_arch_enum().name
    metadata = _metadata(
        mode,
        batch_count,
        device_compute_capability=capability,
        compile_target=compile_target,
    )

    class TileAmaxConsumer:
        """Replace each output element with the complete tile's absmax."""

        @cute.jit
        def __call__(self, accumulator_source: Any) -> Any:
            accumulator = coop.ThreadData.load(accumulator_source)
            thread_amax = cutlass.Float32(0.0)
            for item in cutlass.range_constexpr(ITEMS_PER_THREAD):
                value = accumulator[item]
                magnitude = cute.arch.fmax(value, -value)
                thread_amax = cute.arch.fmax(thread_amax, magnitude)

            tile_amax = coop.reduce(
                coop.this_block(),
                thread_amax,
                binary_op="max",
            )
            output = coop.ThreadData.from_fn(
                ITEMS_PER_THREAD,
                lambda _item: tile_amax,
                dtype=cutlass.Float32,
            )
            return output.to_tensor_ssa(like=accumulator_source)

    class IdentityEpilogue:
        @cute.jit
        def __call__(self, value: Any) -> Any:
            return value

    class PostT2RAmaxAdapter:
        def __init__(self, consumer: Any):
            self.consumer = consumer

        @cute.jit
        def __call__(self, accumulator: Any) -> Any:
            return self.consumer(PostT2RAccumulatorSource(accumulator))

    row = torch.arange(M, device="cuda", dtype=torch.float16) - M // 2
    column = torch.arange(N, device="cuda", dtype=torch.float16) - N // 2
    batch_scale = (
        torch.arange(batch_count, device="cuda", dtype=torch.int32).remainder(8) + 1
    ).to(torch.float16)
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
    a[:, :, 0] = batch_scale[:, None] * row[None, :]
    b[:, 0, :] = column
    output = torch.empty(
        (batch_count, M, N),
        device="cuda",
        dtype=torch.float32,
    )

    a_arg = from_dlpack(a, assumed_align=16).mark_layout_dynamic(leading_dim=2)
    b_arg = from_dlpack(b, assumed_align=16).mark_layout_dynamic(leading_dim=2)
    output_arg = from_dlpack(output, assumed_align=16).mark_layout_dynamic(
        leading_dim=2
    )

    gemm = dense_gemm.DenseGemmKernel(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=USE_2CTA_INSTRS,
        mma_tiler_mn=MMA_TILER_MN,
        cluster_shape_mn=CLUSTER_SHAPE_MN,
        use_tma_store=USE_TMA_STORE,
    )
    amax_consumer = TileAmaxConsumer()
    if mode == TMEM_LOADER_MODE:
        epilogue = IdentityEpilogue()
        accumulator_consumer = amax_consumer
    else:
        epilogue = PostT2RAmaxAdapter(amax_consumer)
        accumulator_consumer = None
    compile_args = (
        dense_gemm.bmm,
        gemm,
        a_arg,
        b_arg,
        output_arg,
        make_fake_stream(),
        epilogue,
    )
    if accumulator_consumer is not None:
        compile_args = (*compile_args, accumulator_consumer)
    compiled = cute.compile(*compile_args)

    def step() -> None:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled(a_arg, b_arg, output_arg, stream)

    def synchronize() -> None:
        torch.cuda.synchronize()

    def validate() -> dict[str, Any]:
        synchronize()
        expected_amax = torch.abs(torch.bmm(a.float(), b.float())).amax(dim=(1, 2))
        expected_output = expected_amax[:, None, None].expand_as(output)
        torch.testing.assert_close(
            output,
            expected_output,
            atol=0,
            rtol=0,
        )
        return {
            **metadata,
            "tile_amax": float(output[0, 0, 0].item()),
            "expected_tile_amax": float(expected_amax[0].item()),
            "distinct_tile_amax_count": int(torch.unique(expected_amax).numel()),
            "all_batches_match": True,
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
    batch_count: int = BATCH_COUNT,
) -> dict[str, Any]:
    """Compile the fused SM100/SM103 kernel without launching it."""

    mode = _validate_mode(mode)
    batch_count = _validate_batch_count(batch_count)
    prepared = make_runner(mode, batch_count)
    return dict(prepared.metadata)


def run_example(
    mode: str = TMEM_LOADER_MODE,
    batch_count: int = BATCH_COUNT,
) -> dict[str, Any]:
    """Run and validate the fused SM100/SM103 GEMM+amax example."""

    mode = _validate_mode(mode)
    batch_count = _validate_batch_count(batch_count)
    _, _, torch, _, _ = _require_runtime()
    _require_supported_compute_capability(torch)
    prepared = make_runner(mode, batch_count)
    prepared.step()
    return prepared.validate()


def benchmark_example(
    mode: str = TMEM_LOADER_MODE,
    batch_count: int = 4096,
    *,
    warmup_iters: int = 16,
    measure_iters: int = 100,
) -> dict[str, Any]:
    """Measure steady GPU time for independent GEMM+amax tiles."""

    mode = _validate_mode(mode)
    batch_count = _validate_batch_count(batch_count)
    _, _, torch, _, _ = _require_runtime()
    _require_supported_compute_capability(torch)
    prepared = make_runner(mode, batch_count)
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


def prepare_benchmark() -> PreparedExample:
    """Prepare an occupancy-scaled loader-mode benchmark."""

    return make_runner(TMEM_LOADER_MODE, batch_count=4096)


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
        result = compile_example(args.mode, args.batch_count)
    elif args.benchmark:
        result = benchmark_example(
            args.mode,
            args.batch_count,
            warmup_iters=args.warmup_iters,
            measure_iters=args.measure_iters,
        )
    else:
        result = run_example(args.mode, args.batch_count)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
