# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import Counter

import pytest

from ..support.compile import compile_for_launch

pytestmark = pytest.mark.gpu

_BLOCK = (8, 4, 2)
_RUNS_PER_THREAD = 2
_DECODED_ITEMS_PER_THREAD = 3


def _signature(types, *dtypes):
    from numba_cuda_mlir.numba_cuda import typing as cuda_typing

    return cuda_typing.signature(types.none, *dtypes)


def _array_type(types, dtype):
    return types.Array(dtype, 1, "C")


@pytest.mark.evidence_for(
    "group.run_length_decode",
    backend="numba_mlir",
    evidence="compile",
)
def test_common_and_qualified_run_length_decode_compile_to_shared_provider_plan(
    numba_mlir_cuda_available,
):
    del numba_mlir_cuda_available
    cuda = pytest.importorskip("numba_cuda_mlir.cuda")

    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    @cuda.jit
    def kernel(values, lengths, decoded, relative_offsets, total, window_offset):
        tid = (
            cuda.threadIdx.x
            + cuda.threadIdx.y * cuda.blockDim.x
            + cuda.threadIdx.z * cuda.blockDim.x * cuda.blockDim.y
        )
        common_values = coop.ThreadData(
            _RUNS_PER_THREAD,
            dtype=types.uint8,
        )
        common_lengths = coop.ThreadData(
            _RUNS_PER_THREAD,
            dtype=types.uint64,
        )
        qualified_values = numba_coop.ThreadData(
            _RUNS_PER_THREAD,
            dtype=types.uint8,
        )
        qualified_lengths = numba_coop.ThreadData(
            _RUNS_PER_THREAD,
            dtype=types.uint64,
        )
        for item in range(_RUNS_PER_THREAD):
            index = tid * _RUNS_PER_THREAD + item
            value = values[index]
            length = lengths[index]
            common_values[item] = value
            common_lengths[item] = length
            qualified_values[item] = value
            qualified_lengths[item] = length

        common = coop.run_length_decode(
            coop.this_block(),
            common_values,
            common_lengths,
            decoded_items_per_thread=_DECODED_ITEMS_PER_THREAD,
            decoded_window_offset=window_offset,
        )
        qualified_plain = numba_coop.run_length_decode(
            numba_coop.this_block(),
            qualified_values,
            qualified_lengths,
            decoded_items_per_thread=_DECODED_ITEMS_PER_THREAD,
            decoded_window_offset=window_offset,
        )
        qualified_relative = numba_coop.ThreadData(
            _DECODED_ITEMS_PER_THREAD,
            dtype=types.uint64,
        )
        qualified_total = numba_coop.ThreadData(1, dtype=types.uint64)
        qualified = numba_coop.run_length_decode(
            numba_coop.this_block(),
            qualified_values,
            qualified_lengths,
            decoded_items_per_thread=_DECODED_ITEMS_PER_THREAD,
            decoded_window_offset=window_offset,
            relative_offsets=qualified_relative,
            total_decoded_size=qualified_total,
        )
        for item in range(_DECODED_ITEMS_PER_THREAD):
            index = tid * _DECODED_ITEMS_PER_THREAD + item
            decoded[index] = common[item] + qualified_plain[item] + qualified[item]
            relative_offsets[index] = qualified_relative[item]
        total[tid] = qualified_total[0]

    signature = _signature(
        types,
        _array_type(types, types.uint8),
        _array_type(types, types.uint64),
        _array_type(types, types.uint8),
        _array_type(types, types.uint64),
        _array_type(types, types.uint64),
        types.uint64,
    )
    inspect_key, result = compile_for_launch(kernel, signature, block=_BLOCK)

    cubin = result.metadata.get("cubin")
    assert isinstance(cubin, bytes)
    assert cubin.startswith(b"\x7fELF")
    link_plan = result.metadata["link_plan"]
    assert link_plan.has_external_link_items
    assert link_plan.has_ltoir_link_items
    assert result.metadata["linked_external_link_items"]

    records = result.metadata["__cuda_coop_numba_mlir_materialized_specializations__"]
    run_length_records = [
        record
        for record in records
        if record[0].split("<", 1)[0] == "BlockRunLengthDecodeDriver"
    ]
    # The common and otherwise-identical qualified calls deduplicate to the
    # same DecodeAt provider; the side-output call needs one offsets variant.
    assert Counter(record[1] for record in run_length_records) == {
        "DecodeAt": 1,
        "DecodeWithOffsetsAt": 1,
    }
    assert all(len(record[5]) == 1 for record in run_length_records)
    assert kernel.get_metadata(inspect_key)["cubin"] == cubin


@pytest.mark.parametrize("offset_dtype_name", ["float32", "boolean"])
def test_common_run_length_decode_rejects_dynamic_noninteger_offsets(
    numba_mlir_cuda_available,
    offset_dtype_name,
):
    del numba_mlir_cuda_available
    cuda = pytest.importorskip("numba_cuda_mlir.cuda")

    from numba_cuda_mlir import types

    from cuda import coop
    from cuda.coop.numba_mlir._single_phase_rewrites import (
        CoopSinglePhaseRewriteError,
    )

    @cuda.jit
    def kernel(values, lengths, output, window_offset):
        tid = cuda.threadIdx.x
        run_values = coop.ThreadData(1, dtype=types.int32)
        run_lengths = coop.ThreadData(1, dtype=types.uint32)
        run_values[0] = values[tid]
        run_lengths[0] = lengths[tid]
        decoded = coop.run_length_decode(
            coop.this_block(),
            run_values,
            run_lengths,
            decoded_items_per_thread=1,
            decoded_window_offset=window_offset,
        )
        output[tid] = decoded[0]

    signature = _signature(
        types,
        _array_type(types, types.int32),
        _array_type(types, types.uint32),
        _array_type(types, types.int32),
        getattr(types, offset_dtype_name),
    )
    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match=(
            r"cuda\.coop\.run_length_decode decoded_window_offset must have "
            r"an integer dtype"
        ),
    ):
        compile_for_launch(kernel, signature, block=64)


@pytest.mark.parametrize("offset_dtype_name", ["float32", "boolean"])
def test_qualified_group_run_length_decode_rejects_dynamic_noninteger_offsets(
    numba_mlir_cuda_available,
    offset_dtype_name,
):
    del numba_mlir_cuda_available
    cuda = pytest.importorskip("numba_cuda_mlir.cuda")

    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as numba_coop
    from cuda.coop.numba_mlir._single_phase_rewrites import (
        CoopSinglePhaseRewriteError,
    )

    @cuda.jit
    def kernel(values, lengths, output, window_offset):
        tid = cuda.threadIdx.x
        run_values = numba_coop.ThreadData(1, dtype=types.int32)
        run_lengths = numba_coop.ThreadData(1, dtype=types.uint32)
        run_values[0] = values[tid]
        run_lengths[0] = lengths[tid]
        decoded = numba_coop.run_length_decode(
            numba_coop.this_block(),
            run_values,
            run_lengths,
            decoded_items_per_thread=1,
            decoded_window_offset=window_offset,
        )
        output[tid] = decoded[0]

    signature = _signature(
        types,
        _array_type(types, types.int32),
        _array_type(types, types.uint32),
        _array_type(types, types.int32),
        getattr(types, offset_dtype_name),
    )
    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match=(
            r"cuda\.coop\.numba_mlir\.run_length_decode "
            r"decoded_window_offset must have an integer dtype"
        ),
    ):
        compile_for_launch(kernel, signature, block=64)


@pytest.mark.parametrize(
    ("run_values_dtype_name", "run_lengths_dtype_name", "parameter"),
    [
        ("float32", "uint32", "run_values"),
        ("int32", "uint16", "run_lengths"),
    ],
)
def test_common_run_length_decode_rejects_nonportable_dtypes_during_compilation(
    numba_mlir_cuda_available,
    run_values_dtype_name,
    run_lengths_dtype_name,
    parameter,
):
    del numba_mlir_cuda_available
    cuda = pytest.importorskip("numba_cuda_mlir.cuda")

    from numba_cuda_mlir import types

    from cuda import coop
    from cuda.coop.numba_mlir._single_phase_rewrites import (
        CoopSinglePhaseRewriteError,
    )

    run_values_dtype = getattr(types, run_values_dtype_name)
    run_lengths_dtype = getattr(types, run_lengths_dtype_name)

    @cuda.jit
    def kernel(values, lengths, output):
        tid = cuda.threadIdx.x
        run_values = coop.ThreadData(1, dtype=run_values_dtype)
        run_lengths = coop.ThreadData(1, dtype=run_lengths_dtype)
        run_values[0] = values[tid]
        run_lengths[0] = lengths[tid]
        decoded = coop.run_length_decode(
            coop.this_block(),
            run_values,
            run_lengths,
            decoded_items_per_thread=1,
        )
        output[tid] = decoded[0]

    signature = _signature(
        types,
        _array_type(types, run_values_dtype),
        _array_type(types, run_lengths_dtype),
        _array_type(types, run_values_dtype),
    )
    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match=(rf"cuda\.coop\.run_length_decode common V1 supports {parameter} dtypes"),
    ):
        compile_for_launch(kernel, signature, block=64)


@pytest.mark.parametrize("common", [True, False])
@pytest.mark.parametrize(
    ("run_length_dtype_name", "maximum_offset"),
    [
        ("int32", (1 << 31) - 1),
        ("uint32", (1 << 32) - 1),
    ],
)
def test_group_run_length_decode_checks_static_offset_representability(
    numba_mlir_cuda_available,
    common,
    run_length_dtype_name,
    maximum_offset,
):
    del numba_mlir_cuda_available
    cuda = pytest.importorskip("numba_cuda_mlir.cuda")

    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop
    from cuda.coop.numba_mlir._single_phase_rewrites import (
        CoopSinglePhaseRewriteError,
    )

    api = coop if common else numba_coop
    run_length_dtype = getattr(types, run_length_dtype_name)

    def make_kernel(window_offset):
        @cuda.jit
        def kernel(values, lengths, output):
            tid = cuda.threadIdx.x
            run_values = api.ThreadData(1, dtype=types.int32)
            run_lengths = api.ThreadData(1, dtype=run_length_dtype)
            run_values[0] = values[tid]
            run_lengths[0] = lengths[tid]
            decoded = api.run_length_decode(
                api.this_block(),
                run_values,
                run_lengths,
                decoded_items_per_thread=1,
                decoded_window_offset=window_offset,
            )
            output[tid] = decoded[0]

        return kernel

    signature = _signature(
        types,
        _array_type(types, types.int32),
        _array_type(types, run_length_dtype),
        _array_type(types, types.int32),
    )
    accepted = make_kernel(maximum_offset)
    _inspect_key, result = compile_for_launch(accepted, signature, block=64)
    assert result.metadata["cubin"].startswith(b"\x7fELF")

    rejected = make_kernel(maximum_offset + 1)
    scope = r"cuda\.coop" if common else r"cuda\.coop\.numba_mlir"
    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match=(
            rf"{scope}\.run_length_decode decoded_window_offset must be "
            r"representable in the run_lengths dtype"
        ),
    ):
        compile_for_launch(rejected, signature, block=64)


def test_qualified_run_length_decode_retains_float_value_support(
    numba_mlir_cuda_available,
):
    del numba_mlir_cuda_available
    cuda = pytest.importorskip("numba_cuda_mlir.cuda")

    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as numba_coop

    @cuda.jit
    def kernel(values, lengths, output):
        tid = cuda.threadIdx.x
        run_values = numba_coop.ThreadData(1, dtype=types.float32)
        run_lengths = numba_coop.ThreadData(1, dtype=types.uint32)
        run_values[0] = values[tid]
        run_lengths[0] = lengths[tid]
        decoded = numba_coop.run_length_decode(
            numba_coop.this_block(),
            run_values,
            run_lengths,
            decoded_items_per_thread=1,
        )
        output[tid] = decoded[0]

    signature = _signature(
        types,
        _array_type(types, types.float32),
        _array_type(types, types.uint32),
        _array_type(types, types.float32),
    )
    _inspect_key, result = compile_for_launch(kernel, signature, block=64)
    assert result.metadata["cubin"].startswith(b"\x7fELF")


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("total", "total_decoded_size dtype must match run_lengths dtype"),
        ("relative", "relative_offsets dtype must match run_lengths dtype"),
        ("offset", "decoded offset dtype must match run_lengths dtype"),
    ],
)
def test_qualified_group_run_length_decode_rejects_mismatched_control_dtypes(
    numba_mlir_cuda_available,
    mismatch,
    message,
):
    del numba_mlir_cuda_available
    cuda = pytest.importorskip("numba_cuda_mlir.cuda")

    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as numba_coop
    from cuda.coop.numba_mlir._single_phase_rewrites import (
        CoopSinglePhaseRewriteError,
    )

    if mismatch == "total":

        @cuda.jit
        def kernel(values, lengths, output):
            tid = cuda.threadIdx.x
            run_values = numba_coop.ThreadData(1, dtype=types.int32)
            run_lengths = numba_coop.ThreadData(1, dtype=types.uint64)
            total = numba_coop.ThreadData(1, dtype=types.uint32)
            run_values[0] = values[tid]
            run_lengths[0] = lengths[tid]
            decoded = numba_coop.run_length_decode(
                numba_coop.this_block(),
                run_values,
                run_lengths,
                decoded_items_per_thread=1,
                total_decoded_size=total,
            )
            output[tid] = decoded[0]

    elif mismatch == "relative":

        @cuda.jit
        def kernel(values, lengths, output):
            tid = cuda.threadIdx.x
            run_values = numba_coop.ThreadData(1, dtype=types.int32)
            run_lengths = numba_coop.ThreadData(1, dtype=types.uint64)
            relative = numba_coop.ThreadData(1, dtype=types.uint32)
            run_values[0] = values[tid]
            run_lengths[0] = lengths[tid]
            decoded = numba_coop.run_length_decode(
                numba_coop.this_block(),
                run_values,
                run_lengths,
                decoded_items_per_thread=1,
                relative_offsets=relative,
            )
            output[tid] = decoded[0]

    else:

        @cuda.jit
        def kernel(values, lengths, output):
            tid = cuda.threadIdx.x
            run_values = numba_coop.ThreadData(1, dtype=types.int32)
            run_lengths = numba_coop.ThreadData(1, dtype=types.uint64)
            run_values[0] = values[tid]
            run_lengths[0] = lengths[tid]
            decoded = numba_coop.run_length_decode(
                numba_coop.this_block(),
                run_values,
                run_lengths,
                decoded_items_per_thread=1,
                decoded_offset_dtype=types.uint32,
            )
            output[tid] = decoded[0]

    signature = _signature(
        types,
        _array_type(types, types.int32),
        _array_type(types, types.uint64),
        _array_type(types, types.int32),
    )
    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match=rf"cuda\.coop\.numba_mlir\.run_length_decode {message}",
    ):
        compile_for_launch(kernel, signature, block=64)
