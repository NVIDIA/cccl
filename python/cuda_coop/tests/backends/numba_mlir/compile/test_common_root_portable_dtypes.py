# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from ..support.compile import compile_for_launch

pytestmark = pytest.mark.gpu

_THREADS = 32


@pytest.mark.parametrize("explicit_dtype", [False, True], ids=["inferred", "explicit"])
def test_common_thread_data_rejects_complex128_during_compilation(
    numba_mlir_cuda_available,
    explicit_dtype,
):
    del numba_mlir_cuda_available
    cuda = pytest.importorskip("numba_cuda_mlir.cuda")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda import typing as cuda_typing

    from cuda import coop
    from cuda.coop.numba_mlir._single_phase_rewrites import (
        CoopSinglePhaseRewriteError,
    )

    if explicit_dtype:

        @cuda.jit
        def kernel(source, output):
            items = coop.ThreadData(1, dtype=types.complex128)
            items[0] = source[0]
            output[0] = items[0]

    else:

        @cuda.jit
        def kernel(source, output):
            items = coop.ThreadData(1)
            items[0] = source[0]
            output[0] = items[0]

    array_type = types.Array(types.complex128, 1, "C")
    signature = cuda_typing.signature(types.none, array_type, array_type)
    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match=(
            r"cuda\.coop\.ThreadData common V1 supports dtypes uint8, int32, "
            r"uint32, int64, uint64, float32, float64; use a "
            r"backend-qualified import for backend-specific dtypes"
        ),
    ):
        compile_for_launch(kernel, signature, block=_THREADS)


@pytest.mark.parametrize("operation", ["load", "store", "exchange"])
def test_common_operation_rejects_qualified_complex128_payload(
    numba_mlir_cuda_available,
    operation,
):
    del numba_mlir_cuda_available
    cuda = pytest.importorskip("numba_cuda_mlir.cuda")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda import typing as cuda_typing

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop
    from cuda.coop.numba_mlir._single_phase_rewrites import (
        CoopSinglePhaseRewriteError,
    )

    if operation == "load":

        @cuda.jit
        def kernel(source, output):
            items = numba_coop.ThreadData(1, dtype=types.complex128)
            coop.load(coop.this_block(), source, items)
            output[0] = items[0]

    elif operation == "store":

        @cuda.jit
        def kernel(source, output):
            items = numba_coop.ThreadData(1, dtype=types.complex128)
            items[0] = source[0]
            coop.store(coop.this_block(), output, items)

    else:

        @cuda.jit
        def kernel(source, output):
            items = numba_coop.ThreadData(1, dtype=types.complex128)
            items[0] = source[cuda.threadIdx.x]
            exchanged = coop.exchange(coop.this_block(), items)
            output[cuda.threadIdx.x] = exchanged[0]

    array_type = types.Array(types.complex128, 1, "C")
    signature = cuda_typing.signature(types.none, array_type, array_type)
    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match=(
            rf"cuda\.coop\.{operation} common V1 supports dtypes uint8, int32, "
            r"uint32, int64, uint64, float32, float64; use a "
            r"backend-qualified import for backend-specific dtypes"
        ),
    ):
        compile_for_launch(kernel, signature, block=_THREADS)


@pytest.mark.parametrize(
    "operation",
    [
        "reduce",
        "sum",
        "scan",
        "exclusive_sum",
        "inclusive_sum",
        "exclusive_scan",
        "inclusive_scan",
        "adjacent_difference",
        "discontinuity",
        "shuffle",
    ],
)
def test_common_collective_rejects_qualified_complex128_payload(
    numba_mlir_cuda_available,
    operation,
):
    del numba_mlir_cuda_available
    cuda = pytest.importorskip("numba_cuda_mlir.cuda")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda import typing as cuda_typing

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop
    from cuda.coop.numba_mlir._single_phase_rewrites import (
        CoopSinglePhaseRewriteError,
    )

    collective = getattr(coop, operation)
    if operation in {"reduce", "sum"}:

        @cuda.jit
        def kernel(source, output):
            items = numba_coop.ThreadData(1, dtype=types.complex128)
            items[0] = source[cuda.threadIdx.x]
            output[cuda.threadIdx.x] = collective(coop.this_block(), items)

    else:

        @cuda.jit
        def kernel(source, output):
            items = numba_coop.ThreadData(1, dtype=types.complex128)
            items[0] = source[cuda.threadIdx.x]
            result = collective(coop.this_block(), items)
            output[cuda.threadIdx.x] = result[0]

    array_type = types.Array(types.complex128, 1, "C")
    signature = cuda_typing.signature(types.none, array_type, array_type)
    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match=(
            rf"cuda\.coop\.{operation} common V1 supports dtypes uint8, int32, "
            r"uint32, int64, uint64, float32, float64; use a "
            r"backend-qualified import for backend-specific dtypes"
        ),
    ):
        compile_for_launch(kernel, signature, block=_THREADS)


def test_common_thread_data_accepts_python_float_alias_during_compilation(
    numba_mlir_cuda_available,
):
    del numba_mlir_cuda_available
    cuda = pytest.importorskip("numba_cuda_mlir.cuda")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda import typing as cuda_typing

    from cuda import coop

    @cuda.jit
    def kernel(source, output):
        items = coop.ThreadData(1, dtype=float)
        items[0] = source[0]
        output[0] = items[0]

    array_type = types.Array(types.float32, 1, "C")
    signature = cuda_typing.signature(types.none, array_type, array_type)
    _, result = compile_for_launch(kernel, signature, block=_THREADS)
    cubin = result.metadata.get("cubin")
    assert isinstance(cubin, bytes)
    assert cubin.startswith(b"\x7fELF")


@pytest.mark.parametrize(
    ("dtype_alias", "resolved_dtype"),
    [(bool, "boolean"), (complex, "complex128")],
    ids=["bool", "complex"],
)
def test_common_thread_data_builtin_extensions_reach_profile_diagnostic(
    numba_mlir_cuda_available,
    dtype_alias,
    resolved_dtype,
):
    del numba_mlir_cuda_available
    cuda = pytest.importorskip("numba_cuda_mlir.cuda")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda import typing as cuda_typing

    from cuda import coop
    from cuda.coop.numba_mlir._single_phase_rewrites import (
        CoopSinglePhaseRewriteError,
    )

    @cuda.jit
    def kernel(source, output):
        items = coop.ThreadData(1, dtype=dtype_alias)
        items[0] = source[0]
        output[0] = items[0]

    array_type = types.Array(getattr(types, resolved_dtype), 1, "C")
    signature = cuda_typing.signature(types.none, array_type, array_type)
    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match=(
            r"cuda\.coop\.ThreadData common V1 supports dtypes uint8, int32, "
            r"uint32, int64, uint64, float32, float64; use a "
            r"backend-qualified import for backend-specific dtypes"
        ),
    ):
        compile_for_launch(kernel, signature, block=_THREADS)


@pytest.mark.parametrize(
    ("case", "source_dtype", "destination_dtype", "operation"),
    [
        ("load_source", "complex128", "int32", "load"),
        ("store_destination", "int32", "complex128", "store"),
        ("store_scalar_value", "complex128", "int32", "store"),
    ],
)
def test_common_load_store_validate_each_data_operand(
    numba_mlir_cuda_available,
    case,
    source_dtype,
    destination_dtype,
    operation,
):
    del numba_mlir_cuda_available
    cuda = pytest.importorskip("numba_cuda_mlir.cuda")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda import typing as cuda_typing

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop
    from cuda.coop.numba_mlir._single_phase_rewrites import (
        CoopSinglePhaseRewriteError,
    )

    if case == "load_source":

        @cuda.jit
        def kernel(source, destination):
            items = numba_coop.ThreadData(1, dtype=types.int32)
            loaded = coop.load(coop.this_block(), source, items)
            destination[0] = loaded[0]

    elif case == "store_destination":

        @cuda.jit
        def kernel(source, destination):
            items = numba_coop.ThreadData(1, dtype=types.int32)
            items[0] = source[0]
            coop.store(coop.this_block(), destination, items)

    else:

        @cuda.jit
        def kernel(source, destination):
            coop.store(coop.this_block(), destination, source[0])

    signature = cuda_typing.signature(
        types.none,
        types.Array(getattr(types, source_dtype), 1, "C"),
        types.Array(getattr(types, destination_dtype), 1, "C"),
    )
    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match=(
            rf"cuda\.coop\.{operation} common V1 supports dtypes uint8, int32, "
            r"uint32, int64, uint64, float32, float64; use a "
            r"backend-qualified import for backend-specific dtypes"
        ),
    ):
        compile_for_launch(kernel, signature, block=_THREADS)
