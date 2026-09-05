# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib

import pytest


def test_numba_mlir_block_load_store_adapter_preserves_offset_overloads():
    pytest.importorskip("numba_cuda_mlir", exc_type=ImportError)
    from numba_cuda_mlir import types

    from cuda.coop._core.block import make_block_load_spec, make_block_store_spec
    from cuda.coop.numba_mlir._core_adapter import NumbaMlirCoreAdapter

    block_load = make_block_load_spec(
        dtype=types.int32,
        block_dim=(16, 2, 1),
        items_per_thread=3,
        algorithm="striped",
        valid_items=True,
        oob_default=True,
        include_full_tile=True,
        include_pointer_offset=True,
    )
    load_specialization = NumbaMlirCoreAdapter().materialize(block_load.specialization)
    assert [
        [type(parameter).__name__ for parameter in method]
        for method in load_specialization.parameters
    ] == [
        ["Pointer", "Pointer", "Array"],
        ["Pointer", "Pointer", "Array", "Value", "Reference"],
        ["Pointer", "Pointer", "Array", "Value", "Reference", "PointerOffset"],
        ["Pointer", "Pointer", "Array", "PointerOffset"],
    ]
    assert load_specialization.parameters[2][-1].pointer_arg_index == 0

    block_store = make_block_store_spec(
        dtype=types.int32,
        block_dim=(32, 1, 1),
        items_per_thread=2,
        algorithm="direct",
        include_pointer_offset=True,
    )
    store_specialization = NumbaMlirCoreAdapter().materialize(
        block_store.specialization
    )
    assert [
        [type(parameter).__name__ for parameter in method]
        for method in store_specialization.parameters
    ] == [
        ["Pointer", "Pointer", "Array"],
        ["Pointer", "Pointer", "Array", "PointerOffset"],
    ]
    assert store_specialization.parameters[1][-1].pointer_arg_index == 0


def test_numba_mlir_provider_symbols_include_the_complete_overload_set():
    pytest.importorskip("numba_cuda_mlir", exc_type=ImportError)
    from numba_cuda_mlir import types

    from cuda.coop._core.block import make_block_store_spec
    from cuda.coop.numba_mlir._core_adapter import NumbaMlirCoreAdapter

    def materialize(*, valid_items):
        spec = make_block_store_spec(
            dtype=types.int32,
            block_dim=(32, 1, 1),
            items_per_thread=2,
            algorithm="transpose",
            valid_items=valid_items,
            include_full_tile=valid_items,
            include_pointer_offset=True,
        )
        return NumbaMlirCoreAdapter().materialize(spec.specialization)

    full = materialize(valid_items=False)
    full_clone = materialize(valid_items=False)
    partial = materialize(valid_items=True)

    # CUB template arguments alone do not distinguish these interfaces. The
    # private namespace must also account for their emitted overload sets.
    assert full.c_name == partial.c_name
    full._source_code()
    full_clone._source_code()
    partial._source_code()

    assert full._private_symbol_digest == full_clone._private_symbol_digest
    assert full._private_symbol_digest != partial._private_symbol_digest
    assert full._temp_storage_symbol_names() == full_clone._temp_storage_symbol_names()
    assert full._temp_storage_symbol_names() != partial._temp_storage_symbol_names()
    assert full.mangled_name(full.parameters[0]) == full_clone.mangled_name(
        full_clone.parameters[0]
    )
    assert full.mangled_name(full.parameters[0]) != partial.mangled_name(
        partial.parameters[0]
    )


@pytest.mark.parametrize(
    ("module_name", "operation", "factory_kwargs"),
    [
        (
            "cuda.coop.numba_mlir._block._block_load_store",
            "load",
            {"threads_per_block": 64},
        ),
        (
            "cuda.coop.numba_mlir._block._block_load_store",
            "store",
            {"threads_per_block": 64},
        ),
        (
            "cuda.coop.numba_mlir._warp._warp_load_store",
            "warp_load",
            {"threads_per_block": 64},
        ),
        (
            "cuda.coop.numba_mlir._warp._warp_load_store",
            "warp_store",
            {"threads_per_block": 64},
        ),
    ],
)
def test_numba_mlir_load_store_factories_define_aggregate_storage(
    monkeypatch,
    module_name,
    operation,
    factory_kwargs,
):
    pytest.importorskip("numba_cuda_mlir", exc_type=ImportError)
    from numba_cuda_mlir import types

    module = importlib.import_module(module_name)
    monkeypatch.setattr(
        module,
        "make_invocable_from_specialization",
        lambda specialization, **_kwargs: specialization,
    )

    specialization = getattr(module, operation)(
        types.complex128,
        items_per_thread=2,
        algorithm="transpose",
        **factory_kwargs,
    )

    assert len(specialization.type_definitions) == 1
    assert "struct __align__(8) storage_t" in specialization.type_definitions[0].code
    assert "char data[16]" in specialization.type_definitions[0].code
