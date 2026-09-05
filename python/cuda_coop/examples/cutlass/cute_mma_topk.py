# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Fuse ``cuda.coop.cutlass`` TopK into a CuTe tensor-core GEMM epilogue.

This example imports CUTLASS's BSD-licensed Ampere ``TensorOpGemm`` sample and
specializes its executable epilogue callback.  The underlying kernel performs
the real tensor-core operation here, in ``TensorOpGemm.kernel``::

    cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)

Immediately after the MMA mainloop, ``TensorOpGemm.kernel`` calls
``epilogue_op(tCrC.load())``.  ``TileTopKEpilogue`` receives that register
accumulator as a CuTe ``TensorSSA`` and invokes CUB BlockTopK through
``cuda.coop.cutlass``.  No full GEMM result is materialized before selection.

The demo uses one 128 x 128 output tile. The first ``TOPK`` logical positions of
the epilogue fragment contain the selected values; all other positions are
replaced by the minimum finite fp16 value.  The stock GEMM store path writes
that compact candidate tile to C, and validation compares its largest values
with ``torch.topk(A @ B)``.

The same boundary applies to SM100 ``tcgen05`` kernels: their accumulator is in
TMEM, so the GEMM epilogue must first perform its normal TMEM-to-RMEM copy and
then pass the resulting register fragment to the cooperative algorithm.
"""

from __future__ import annotations

import functools
import importlib.util
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from examples.cutlass._runtime import require_runtime

M = 128
N = 128
K = 32
BATCH_COUNT = 1
TOPK = 8
BLOCK_THREADS = 128
ITEMS_PER_THREAD = M * N // BLOCK_THREADS
CHUNK_ITEMS_PER_THREAD = 8
CHUNKS_PER_THREAD = ITEMS_PER_THREAD // CHUNK_ITEMS_PER_THREAD
ATOM_LAYOUT_MNK = (2, 2, 1)
NON_CANDIDATE = -65504.0
TOPK_TEMP_STORAGE_BYTES = 10240

TENSOROP_GEMM_RELATIVE_PATH = Path(
    "examples/CuTeDSL/cute/ampere/kernel/dense_gemm/tensorop_gemm.py"
)


def _load_tensorop_gemm_module() -> Any:
    """Load the CUTLASS DSL tensor-core GEMM sample from ``sys.path``."""

    for entry in sys.path:
        if not entry:
            continue
        candidate = Path(entry) / TENSOROP_GEMM_RELATIVE_PATH
        if not candidate.is_file():
            continue
        spec = importlib.util.spec_from_file_location(
            "_cuda_coop_tensorop_gemm",
            candidate,
        )
        if spec is None or spec.loader is None:
            break
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    searched = ", ".join(
        str(Path(entry) / TENSOROP_GEMM_RELATIVE_PATH) for entry in sys.path if entry
    )
    raise RuntimeError(
        "cute_mma_topk requires CUTLASS's TensorOpGemm CuTe sample on "
        f"PYTHONPATH; searched: {searched}"
    )


def _require_runtime() -> tuple[Any, ...]:
    return require_runtime(include_int32=True)


@functools.lru_cache(maxsize=1)
def make_runner() -> tuple[Any, ...]:
    """Compile the tensor-core GEMM with its cooperative TopK epilogue."""

    cutlass, cute, torch, _from_dlpack, coop, Int32 = _require_runtime()
    tensorop_gemm = _load_tensorop_gemm_module()
    from cutlass.cute.runtime import make_fake_stream

    from cuda.bindings import driver as cuda

    globals()["cutlass"] = cutlass
    globals()["cute"] = cute
    globals()["coop"] = coop
    globals()["Int32"] = Int32

    class TileTopKEpilogue:
        """Select the tile-wide TopK directly from the MMA accumulator."""

        @cute.jit
        def __call__(self, accumulator: Any) -> Any:
            tidx, _, _ = cute.arch.thread_idx()
            block = coop.this_block()
            candidate_ptr = cute.arch.alloc_smem(
                cutlass.Float32,
                CHUNKS_PER_THREAD * TOPK,
            )
            candidate_scores = cute.make_tensor(
                candidate_ptr,
                cute.make_layout((CHUNKS_PER_THREAD * TOPK,)),
            )

            # TopK each register chunk. The union of every chunk's TopK is a
            # sufficient candidate set for the TopK of the complete MMA tile.
            for chunk in cutlass.range_constexpr(CHUNKS_PER_THREAD):
                chunk_scores = coop.ThreadData.from_fn(
                    CHUNK_ITEMS_PER_THREAD,
                    lambda item: accumulator[chunk * CHUNK_ITEMS_PER_THREAD + item],
                    dtype=cutlass.Float32,
                )
                chunk_top = coop.topk_max_keys(
                    block,
                    chunk_scores,
                    cutlass.const_expr(TOPK),
                )
                if tidx == cutlass.Int32(0):
                    for item in cutlass.range_constexpr(TOPK):
                        candidate_scores[chunk * TOPK + item] = chunk_top[item]

            cute.arch.sync_threads()
            merge_input = coop.ThreadData.from_values(
                candidate_scores[tidx],
                dtype=cutlass.Float32,
            )
            top_scores = coop.topk_max_keys(
                block,
                merge_input,
                cutlass.const_expr(TOPK),
            )
            if tidx < cutlass.Int32(TOPK):
                candidate_scores[tidx] = top_scores[0]

            cute.arch.sync_threads()
            selected = coop.ThreadData.from_fn(
                ITEMS_PER_THREAD,
                lambda _item: cutlass.Float32(NON_CANDIDATE),
                dtype=cutlass.Float32,
            )
            for item in cutlass.range_constexpr(ITEMS_PER_THREAD):
                flat_item = tidx * cutlass.Int32(ITEMS_PER_THREAD) + cutlass.Int32(item)
                value = cutlass.Float32(NON_CANDIDATE)
                if flat_item < cutlass.Int32(TOPK):
                    value = candidate_scores[flat_item]
                selected[item] = value
            return selected.to_tensor_ssa(
                dtype=cutlass.Float32,
                shape=accumulator.shape,
            )

    torch.manual_seed(7)
    a = torch.randint(
        -3,
        4,
        (BATCH_COUNT, M, K),
        device="cuda",
        dtype=torch.int32,
    ).to(torch.float16)
    b = torch.randint(
        -3,
        4,
        (BATCH_COUNT, K, N),
        device="cuda",
        dtype=torch.int32,
    ).to(torch.float16)
    candidates = torch.empty(
        (BATCH_COUNT, M, N),
        device="cuda",
        dtype=torch.float16,
    )

    a_arg, b_arg, candidates_arg = tensorop_gemm.mark_dynamic_layout(
        a,
        b,
        candidates,
        2,
        2,
        2,
        cutlass.Float16,
        cutlass.Float16,
    )
    gemm = tensorop_gemm.TensorOpGemm(
        cutlass.Float16,
        cutlass.Float16,
        cutlass.Float32,
        ATOM_LAYOUT_MNK,
    )
    epilogue = TileTopKEpilogue()
    compiled = cute.compile(
        tensorop_gemm.bmm,
        gemm,
        a_arg,
        b_arg,
        candidates_arg,
        make_fake_stream(),
        epilogue,
    )

    def step() -> None:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled(a_arg, b_arg, candidates_arg, stream)

    return step, torch, a, b, candidates


@dataclass(frozen=True)
class PreparedExample:
    step: Callable[[], None]
    synchronize: Callable[[], None]
    validate: Callable[[], dict[str, Any]]


def prepare_example() -> PreparedExample:
    """Prepare a deterministic one-tile GEMM+TopK launch."""

    step, torch, a, b, candidates = make_runner()

    def synchronize() -> None:
        torch.cuda.synchronize()

    def validate() -> dict[str, Any]:
        synchronize()
        reference = torch.bmm(a.float(), b.float()).to(torch.float16)
        expected = torch.topk(reference.flatten(), TOPK, sorted=False).values
        actual = torch.topk(candidates.flatten(), TOPK, sorted=False).values
        torch.testing.assert_close(
            torch.sort(actual.float()).values,
            torch.sort(expected.float()).values,
            atol=0,
            rtol=0,
        )
        return {
            "mma": "cute.gemm",
            "shape_mnk": (M, N, K),
            "topk": TOPK,
            "top_scores": [float(value) for value in actual.cpu().tolist()],
        }

    return PreparedExample(step=step, synchronize=synchronize, validate=validate)


def run_example() -> dict[str, Any]:
    """Run and validate the fused CuTe MMA+TopK example."""

    prepared = prepare_example()
    prepared.step()
    return prepared.validate()


def main() -> int:
    print(run_example())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
