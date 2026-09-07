# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import Counter

import pytest

from ..support.compile import compile_for_launch

pytestmark = pytest.mark.gpu

_THREADS = 32
_ITEMS_PER_THREAD = 2


@pytest.mark.evidence_for(
    "group.topk_max_keys",
    backend="numba_mlir",
    evidence="compile",
)
@pytest.mark.evidence_for(
    "group.topk_min_keys",
    backend="numba_mlir",
    evidence="compile",
)
def test_common_and_qualified_topk_compile_to_shared_provider_plans(
    numba_mlir_cuda_available,
):
    del numba_mlir_cuda_available
    cuda = pytest.importorskip("numba_cuda_mlir.cuda")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda import typing as cuda_typing

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    @cuda.jit
    def kernel(source, output, k, valid_items, begin_bit, end_bit):
        tid = cuda.threadIdx.x
        common_keys = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
        qualified_keys = numba_coop.ThreadData(
            _ITEMS_PER_THREAD,
            dtype=types.int32,
        )
        for item in range(_ITEMS_PER_THREAD):
            index = tid * _ITEMS_PER_THREAD + item
            value = source[index]
            common_keys[item] = value
            qualified_keys[item] = value

        common_max = coop.topk_max_keys(
            coop.this_block(),
            common_keys,
            k,
            valid_items=valid_items,
            begin_bit=begin_bit,
            end_bit=end_bit,
        )
        qualified_max = numba_coop.topk_max_keys(
            numba_coop.this_block(),
            qualified_keys,
            k,
            valid_items=valid_items,
            begin_bit=begin_bit,
            end_bit=end_bit,
        )
        common_min = coop.topk_min_keys(
            coop.this_block(),
            common_keys,
            k,
            valid_items=valid_items,
            begin_bit=begin_bit,
            end_bit=end_bit,
        )
        qualified_min = numba_coop.topk_min_keys(
            numba_coop.this_block(),
            qualified_keys,
            k,
            valid_items=valid_items,
            begin_bit=begin_bit,
            end_bit=end_bit,
        )
        output[tid] = (
            common_max[0] + qualified_max[0] + common_min[0] + qualified_min[0]
        )

    array_type = types.Array(types.int32, 1, "C")
    signature = cuda_typing.signature(
        types.none,
        array_type,
        array_type,
        types.int32,
        types.int32,
        types.int32,
        types.int32,
    )
    inspect_key, result = compile_for_launch(kernel, signature, block=_THREADS)

    cubin = result.metadata.get("cubin")
    assert isinstance(cubin, bytes)
    assert cubin.startswith(b"\x7fELF")
    link_plan = result.metadata["link_plan"]
    assert link_plan.has_external_link_items
    assert link_plan.has_ltoir_link_items
    assert result.metadata["linked_external_link_items"]

    records = result.metadata["__cuda_coop_numba_mlir_materialized_specializations__"]
    topk_records = [
        record for record in records if record[0].split("<", 1)[0] == "BlockTopKCoop"
    ]
    assert Counter(record[1] for record in topk_records) == {
        "max_keys_partial": 1,
        "min_keys_partial": 1,
    }
    assert all(len(record[5]) == 1 for record in topk_records)
    assert kernel.get_metadata(inspect_key)["cubin"] == cubin


@pytest.mark.evidence_for(
    "group.topk_max_pairs",
    backend="numba_mlir",
    evidence="compile",
)
@pytest.mark.evidence_for(
    "group.topk_min_pairs",
    backend="numba_mlir",
    evidence="compile",
)
def test_common_topk_pairs_compile_with_inferred_end_bit(
    numba_mlir_cuda_available,
):
    del numba_mlir_cuda_available
    cuda = pytest.importorskip("numba_cuda_mlir.cuda")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda import typing as cuda_typing

    from cuda import coop

    @cuda.jit
    def kernel(source, output, k):
        tid = cuda.threadIdx.x
        keys = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
        values = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.float64)
        for item in range(_ITEMS_PER_THREAD):
            index = tid * _ITEMS_PER_THREAD + item
            keys[item] = source[index]
            values[item] = index

        max_keys, max_values = coop.topk_max_pairs(
            coop.this_block(),
            keys,
            values,
            k,
            begin_bit=4,
        )
        min_keys, min_values = coop.topk_min_pairs(
            coop.this_block(),
            keys,
            values,
            k,
            begin_bit=4,
        )
        output[tid] = max_keys[0] + min_keys[0] + max_values[0] + min_values[0]

    source_type = types.Array(types.int32, 1, "C")
    output_type = types.Array(types.int32, 1, "C")
    signature = cuda_typing.signature(
        types.none,
        source_type,
        output_type,
        types.int32,
    )
    _inspect_key, result = compile_for_launch(kernel, signature, block=_THREADS)

    records = result.metadata["__cuda_coop_numba_mlir_materialized_specializations__"]
    topk_records = [
        record for record in records if record[0].split("<", 1)[0] == "BlockTopKCoop"
    ]
    assert Counter(record[1] for record in topk_records) == {
        "max_pairs_full": 1,
        "min_pairs_full": 1,
    }
    assert all(len(record[5]) == 1 for record in topk_records)


@pytest.mark.evidence_for(
    "group.topk_max_keys",
    backend="numba_mlir",
    evidence="compile",
)
@pytest.mark.evidence_for(
    "group.topk_min_keys",
    backend="numba_mlir",
    evidence="compile",
)
@pytest.mark.evidence_for(
    "group.topk_max_pairs",
    backend="numba_mlir",
    evidence="compile",
)
@pytest.mark.evidence_for(
    "group.topk_min_pairs",
    backend="numba_mlir",
    evidence="compile",
)
def test_common_and_qualified_topk_materialize_with_reusable_temp_storage(
    numba_mlir_cuda_available,
):
    del numba_mlir_cuda_available
    cuda = pytest.importorskip("numba_cuda_mlir.cuda")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda import typing as cuda_typing

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop as common_coop

    @cuda.jit
    def kernel(source, output, k):
        tid = cuda.threadIdx.x
        storage = numba_coop.TempStorage()
        keys = numba_coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
        values = numba_coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.float32)
        for item in range(_ITEMS_PER_THREAD):
            index = tid * _ITEMS_PER_THREAD + item
            keys[item] = source[index]
            values[item] = source[index]

        max_keys = common_coop.topk_max_keys(
            common_coop.this_block(),
            keys,
            k,
            temp_storage=storage,
        )
        min_keys = numba_coop.topk_min_keys(
            numba_coop.this_block(),
            keys,
            k,
            begin_bit=1,
            temp_storage=storage,
        )
        max_pair_keys, max_pair_values = numba_coop.topk_max_pairs(
            numba_coop.this_block(),
            keys,
            values,
            k,
            temp_storage=storage,
        )
        min_pair_keys, min_pair_values = numba_coop.topk_min_pairs(
            numba_coop.this_block(),
            keys,
            values,
            k,
            temp_storage=storage,
        )
        output[tid] = (
            max_keys[0]
            + min_keys[0]
            + max_pair_keys[0]
            + max_pair_values[0]
            + min_pair_keys[0]
            + min_pair_values[0]
        )

    source_type = types.Array(types.int32, 1, "C")
    output_type = types.Array(types.float32, 1, "C")
    signature = cuda_typing.signature(
        types.none,
        source_type,
        output_type,
        types.int32,
    )
    _inspect_key, result = compile_for_launch(kernel, signature, block=_THREADS)

    cubin = result.metadata.get("cubin")
    assert isinstance(cubin, bytes)
    assert cubin.startswith(b"\x7fELF")
    records = result.metadata["__cuda_coop_numba_mlir_materialized_specializations__"]
    topk_records = [
        record for record in records if record[0].split("<", 1)[0] == "BlockTopKCoop"
    ]
    assert Counter(record[1] for record in topk_records) == {
        "max_keys_full": 1,
        "min_keys_full": 1,
        "max_pairs_full": 1,
        "min_pairs_full": 1,
    }


@pytest.mark.parametrize(
    ("begin_bit", "message"),
    [
        (-1, r"begin_bit must be non-negative"),
        (32, r"begin_bit must be < 32"),
    ],
)
def test_qualified_topk_rejects_out_of_range_begin_bit_with_omitted_end_bit(
    numba_mlir_cuda_available,
    begin_bit,
    message,
):
    del numba_mlir_cuda_available
    cuda = pytest.importorskip("numba_cuda_mlir.cuda")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda import typing as cuda_typing

    import cuda.coop.numba_mlir as numba_coop
    from cuda.coop.numba_mlir._single_phase_rewrites import (
        CoopSinglePhaseRewriteError,
    )

    @cuda.jit
    def kernel(source, output):
        tid = cuda.threadIdx.x
        keys = numba_coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
        for item in range(_ITEMS_PER_THREAD):
            index = tid * _ITEMS_PER_THREAD + item
            keys[item] = source[index]
        selected = numba_coop.topk_max_keys(
            numba_coop.this_block(),
            keys,
            1,
            begin_bit=begin_bit,
        )
        output[tid] = selected[0]

    array_type = types.Array(types.int32, 1, "C")
    signature = cuda_typing.signature(types.none, array_type, array_type)
    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match=rf"cuda\.coop\.numba_mlir\.topk_max_keys {message}",
    ):
        compile_for_launch(kernel, signature, block=_THREADS)


@pytest.mark.parametrize("dtype_name", ["float32", "int16"])
def test_common_topk_rejects_nonportable_key_dtypes_during_compilation(
    numba_mlir_cuda_available,
    dtype_name,
):
    del numba_mlir_cuda_available
    cuda = pytest.importorskip("numba_cuda_mlir.cuda")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda import typing as cuda_typing

    from cuda import coop
    from cuda.coop.numba_mlir._single_phase_rewrites import (
        CoopSinglePhaseRewriteError,
    )

    dtype = getattr(types, dtype_name)

    @cuda.jit
    def kernel(source, output):
        tid = cuda.threadIdx.x
        keys = coop.ThreadData(_ITEMS_PER_THREAD, dtype=dtype)
        for item in range(_ITEMS_PER_THREAD):
            index = tid * _ITEMS_PER_THREAD + item
            keys[item] = source[index]
        selected = coop.topk_max_keys(coop.this_block(), keys, 1)
        output[tid] = selected[0]

    array_type = types.Array(dtype, 1, "C")
    signature = cuda_typing.signature(types.none, array_type, array_type)
    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match=(
            r"cuda\.coop\.topk_max_keys common V1 supports key dtypes int32, "
            r"uint32, int64, uint64"
        ),
    ):
        compile_for_launch(kernel, signature, block=_THREADS)


@pytest.mark.parametrize(
    "control_name",
    ["k", "valid_items", "begin_bit", "end_bit"],
)
@pytest.mark.parametrize("qualified", [False, True], ids=["common", "qualified"])
def test_group_first_topk_rejects_dynamic_noninteger_controls(
    numba_mlir_cuda_available,
    control_name,
    qualified,
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

    api = numba_coop if qualified else coop

    @cuda.jit
    def kernel(source, output, control):
        tid = cuda.threadIdx.x
        keys = api.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
        for item in range(_ITEMS_PER_THREAD):
            index = tid * _ITEMS_PER_THREAD + item
            keys[item] = source[index]
        if control_name == "k":
            selected = api.topk_min_keys(api.this_block(), keys, control)
        elif control_name == "valid_items":
            selected = api.topk_min_keys(
                api.this_block(),
                keys,
                1,
                valid_items=control,
            )
        elif control_name == "begin_bit":
            selected = api.topk_min_keys(
                api.this_block(),
                keys,
                1,
                begin_bit=control,
                end_bit=32,
            )
        else:
            selected = api.topk_min_keys(
                api.this_block(),
                keys,
                1,
                begin_bit=0,
                end_bit=control,
            )
        output[tid] = selected[0]

    array_type = types.Array(types.int32, 1, "C")
    signature = cuda_typing.signature(
        types.none,
        array_type,
        array_type,
        types.float32,
    )
    scope = r"cuda\.coop\.numba_mlir" if qualified else r"cuda\.coop"
    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match=rf"{scope}\.topk_min_keys {control_name} must have an integer dtype",
    ):
        compile_for_launch(kernel, signature, block=_THREADS)
