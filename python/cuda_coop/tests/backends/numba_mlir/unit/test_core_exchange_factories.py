# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib

import pytest


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
            {"threads_per_block": 64},
        ),
    ],
)
def test_numba_mlir_exchange_factories_define_aggregate_storage(
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
        dtype=types.complex128,
        items_per_thread=2,
        **factory_kwargs,
    )

    assert len(specialization.type_definitions) == 1
    assert "struct __align__(8) storage_t" in specialization.type_definitions[0].code
    assert "char data[16]" in specialization.type_definitions[0].code


@pytest.mark.parametrize(
    ("module_name", "operation", "factory_kwargs"),
    [
        (
            "cuda.coop.numba_mlir._block._block_exchange",
            "exchange",
            {"threads_per_block": 32},
        ),
        (
            "cuda.coop.numba_mlir._warp._warp_exchange",
            "warp_exchange",
            {"threads_per_block": 64},
        ),
    ],
)
def test_numba_mlir_exchange_factories_forward_custom_methods(
    monkeypatch,
    module_name,
    operation,
    factory_kwargs,
):
    pytest.importorskip("numba_cuda_mlir", exc_type=ImportError)
    from numba_cuda_mlir import types

    module = importlib.import_module(module_name)
    methods = {"construct": object(), "assign": object()}
    captured = []

    def capture_wrapper(dtype, *, methods=None):
        captured.append((dtype, methods))
        return object()

    class CapturingAdapter:
        def materialize(self, specialization, *, extra_type_definitions):
            assert len(extra_type_definitions) == 1
            return specialization

    monkeypatch.setattr(module, "numba_type_to_wrapper", capture_wrapper)
    monkeypatch.setattr(module, "NumbaMlirCoreAdapter", CapturingAdapter)
    monkeypatch.setattr(
        module,
        "make_invocable_from_specialization",
        lambda specialization, **_kwargs: specialization,
    )

    getattr(module, operation)(
        dtype=types.complex128,
        items_per_thread=2,
        methods=methods,
        **factory_kwargs,
    )

    assert captured == [(types.complex128, methods)]
