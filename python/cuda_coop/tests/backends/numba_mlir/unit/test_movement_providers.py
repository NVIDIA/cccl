# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]


def _parameter_names(specialization):
    return [
        [type(parameter).__name__ for parameter in overload]
        for overload in specialization.parameters
    ]


def test_block_load_store_adapters_preserve_offset_overloads():
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types

    from cuda.coop._core.block import make_block_load_spec, make_block_store_spec
    from cuda.coop.numba_mlir._core_adapter import NumbaMlirCoreAdapter

    load = NumbaMlirCoreAdapter().materialize(
        make_block_load_spec(
            dtype=types.int32,
            block_dim=(16, 2, 1),
            items_per_thread=3,
            algorithm="striped",
            valid_items=True,
            oob_default=True,
            include_full_tile=True,
            include_pointer_offset=True,
        ).specialization
    )
    assert _parameter_names(load) == [
        ["Pointer", "Pointer", "Array"],
        ["Pointer", "Pointer", "Array", "Value", "Reference"],
        [
            "Pointer",
            "Pointer",
            "Array",
            "Value",
            "Reference",
            "PointerOffset",
        ],
        ["Pointer", "Pointer", "Array", "PointerOffset"],
    ]

    store = NumbaMlirCoreAdapter().materialize(
        make_block_store_spec(
            dtype=types.int32,
            block_dim=(32, 1, 1),
            items_per_thread=2,
            algorithm="direct",
            include_pointer_offset=True,
        ).specialization
    )
    assert _parameter_names(store) == [
        ["Pointer", "Pointer", "Array"],
        ["Pointer", "Pointer", "Array", "PointerOffset"],
    ]


def test_block_exchange_adapter_preserves_in_and_out_of_place_forms():
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types

    from cuda.coop._core.block import make_block_exchange_spec
    from cuda.coop.numba_mlir._core_adapter import NumbaMlirCoreAdapter

    exchange = NumbaMlirCoreAdapter().materialize(
        make_block_exchange_spec(
            dtype=types.int32,
            block_dim=(16, 2, 1),
            items_per_thread=3,
            mode="scatter_to_striped_flagged",
            value_form="both",
            warp_time_slicing=True,
            rank_dtype=types.int32,
            valid_flag_dtype=types.uint8,
        ).specialization
    )
    assert exchange.method_name == "ScatterToStripedFlagged"
    assert _parameter_names(exchange) == [
        ["Pointer", "Array", "Array", "Array"],
        ["Pointer", "Array", "Array", "Array", "Array"],
    ]


def test_block_shuffle_adapter_preserves_static_distance_and_boundary():
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types

    from cuda.coop._core import ArgumentBinding
    from cuda.coop._core.block import make_block_shuffle_spec
    from cuda.coop.numba_mlir._core_adapter import NumbaMlirCoreAdapter

    scalar = NumbaMlirCoreAdapter().materialize(
        make_block_shuffle_spec(
            dtype=types.int32,
            block_dim=(32, 1, 1),
            mode="offset",
            distance=ArgumentBinding.static(-2),
        ).specialization
    )
    assert scalar.method_name == "Offset"
    assert scalar.fake_return
    assert scalar.parameters[0][-1].cpp == "-2"

    array = NumbaMlirCoreAdapter().materialize(
        make_block_shuffle_spec(
            dtype=types.int32,
            block_dim=(32, 1, 1),
            mode="down",
            items_per_thread=2,
            block_prefix=True,
        ).specialization
    )
    assert array.method_name == "Down"
    assert _parameter_names(array) == [
        ["Pointer", "Array", "Array", "PointerReference"]
    ]


@pytest.mark.parametrize(
    ("module_name", "operation", "factory_kwargs"),
    [
        (
            "cuda.coop.numba_mlir._block._block_exchange",
            "exchange",
            {"threads_per_block": 32, "use_output_items": True},
        ),
        (
            "cuda.coop.numba_mlir._warp._warp_exchange",
            "warp_exchange",
            {"threads_in_warp": 8, "threads_per_block": 64},
        ),
    ],
)
def test_exchange_factories_define_aggregate_storage(
    monkeypatch,
    module_name,
    operation,
    factory_kwargs,
):
    import importlib

    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types

    module = importlib.import_module(module_name)
    captured = {}

    def capture(specialization, **kwargs):
        captured.update(kwargs)
        return specialization

    monkeypatch.setattr(module, "make_invocable_from_specialization", capture)
    specialization = getattr(module, operation)(
        dtype=types.complex128,
        items_per_thread=2,
        **factory_kwargs,
    )

    assert "struct __align__(8) storage_t" in specialization.type_definitions[0].code
    assert "char data[16]" in specialization.type_definitions[0].code
    if operation == "warp_exchange":
        assert captured == {"threads": 8, "block_threads": 64}
