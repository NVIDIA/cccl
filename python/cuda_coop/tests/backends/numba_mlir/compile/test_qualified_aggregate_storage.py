# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from ..support.compile import compile_for_launch

pytestmark = pytest.mark.gpu


def test_opaque_extension_collective_fails_before_cubin_materialization(
    numba_mlir_cuda_available,
):
    del numba_mlir_cuda_available
    cuda = pytest.importorskip("numba_cuda_mlir.cuda")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda import typing as cuda_typing

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._single_phase_rewrites import (
        CoopSinglePhaseRewriteError,
    )

    from ..support.padded_extension import padded_opaque_type

    @cuda.jit
    def kernel(output):
        items = coop.ThreadData(2, dtype=padded_opaque_type)
        exchanged = coop.exchange(coop.this_block(), items)
        output[0] = len(exchanged)

    signature = cuda_typing.signature(
        types.none,
        types.Array(types.int32, 1, "C"),
    )
    with pytest.raises(CoopSinglePhaseRewriteError) as exc_info:
        compile_for_launch(kernel, signature, block=32)

    cause = exc_info.value.__cause__
    assert isinstance(cause, TypeError)
    assert str(cause) == (
        "cuda.coop.numba_mlir cannot safely materialize CUB storage for dtype "
        "PaddedOpaque: exact ABI size and alignment are unavailable; use a "
        "compiler-native dtype, Numba-CUDA-MLIR AggregateType, or matching "
        "registered CUDA and MLIR StructModels with inspectable member types"
    )
