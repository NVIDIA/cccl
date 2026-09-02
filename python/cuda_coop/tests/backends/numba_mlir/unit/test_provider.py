# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("numba_cuda_mlir")

from numba_cuda_mlir import types

from cuda.coop._core.block.reduce import make_block_reduce_spec
from cuda.coop._core.warp.reduce import make_warp_reduce_spec
from cuda.coop.numba_mlir._compiler import _nvrtc
from cuda.coop.numba_mlir._compiler._operations import factory_operation
from cuda.coop.numba_mlir._lowering import _reduce


def _spec(**changes):
    values = {
        "dtype": types.int32,
        "block_dim": (32, 2, 1),
        "operation": "reduce",
        "binary_op": "max",
        "algorithm": "raking",
        "valid_items": True,
    }
    values.update(changes)
    return make_block_reduce_spec(**values)


def _context(**changes):
    values = {
        "nvrtc_path": "/cuda/lib/libnvrtc.so.13",
        "nvrtc_builtins_path": "/cuda/lib/libnvrtc-builtins.so.13.3",
        "nvrtc_version": (13, 3),
        "include_dirs": ("/cccl/include", "/cuda/include"),
        "header_identity": "headers-a",
        "architecture": "90",
    }
    values.update(changes)
    return _nvrtc.CompileContext(**values)


def _marker_spec(**changes):
    spec = _spec(dtype=None, valid_items=False, **changes)
    return _reduce._MarkerSpec(
        group_kind="block",
        block_dim=spec.block_dim,
        operation=spec.operation,
        binary_op=spec.binary_op,
        algorithm=spec.algorithm,
        valid_items=spec.valid_items,
    )


def _warp_spec(**changes):
    values = {
        "dtype": types.int32,
        "block_dim": (8, 4, 2),
        "operation": "reduce",
        "binary_op": "max",
        "valid_items": True,
    }
    values.update(changes)
    return make_warp_reduce_spec(**values)


def _warp_marker_spec(**changes):
    spec = _warp_spec(dtype=None, valid_items=False, **changes)
    return _reduce._MarkerSpec(
        group_kind="warp",
        block_dim=spec.block_dim,
        operation=_reduce.BlockReduceOperation(spec.operation.value),
        binary_op=spec.binary_op,
        algorithm=None,
        valid_items=spec.valid_items,
    )


def test_generated_provider_is_direct_root_only_cub_block_reduce():
    source = _reduce._source(
        symbol="test_symbol",
        cpp_type="::cuda::std::int32_t",
        spec=_spec(),
    )

    assert "#include <cub/block/block_reduce.cuh>" in source
    assert "#include <cuda/functional>" in source
    assert "#include <cuda/std/cstdint>" in source
    assert "#include <cuda/std/functional>" in source
    assert "::cub::BlockReduce<" in source
    assert "::cub::BLOCK_REDUCE_RAKING" in source
    assert ".Reduce(value, ::cuda::maximum<T>{}, valid_items)" in source
    assert "__syncthreads();" in source
    assert 'extern "C" __device__' in source


@pytest.mark.parametrize("operation", ["sum", "reduce"])
def test_sum_operator_selects_cub_sum_without_a_callback(operation):
    source = _reduce._source(
        symbol="test_symbol",
        cpp_type="float",
        spec=_spec(
            dtype=types.float32,
            operation=operation,
            binary_op="sum",
            algorithm="warp_reductions",
            valid_items=False,
        ),
    )

    assert ".Sum(value)" in source
    assert "BLOCK_REDUCE_WARP_REDUCTIONS" in source


def test_generated_provider_is_direct_per_physical_warp_cub_reduce():
    source = _reduce._source(
        symbol="test_symbol",
        cpp_type="::cuda::std::int32_t",
        spec=_warp_spec(),
    )

    assert "#include <cub/warp/warp_reduce.cuh>" in source
    assert "::cub::WarpReduce<T, 32>" in source
    assert "TempStorage storage[2]" in source
    assert (
        "threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)" in source
    )
    assert "warp_id = linear_thread_rank / 32" in source
    assert "WarpReduce(storage[warp_id]).Reduce(" in source
    assert "::cuda::maximum<T>{}, valid_items" in source
    assert "__syncwarp();" in source
    assert "__syncthreads();" not in source


def test_warp_reduce_with_sum_operator_uses_cub_sum():
    source = _reduce._source(
        symbol="test_symbol",
        cpp_type="float",
        spec=_warp_spec(
            dtype=types.float32,
            operation="reduce",
            binary_op="sum",
            valid_items=False,
        ),
    )

    assert ".Sum(value)" in source
    assert ".Reduce(" not in source


def test_lowering_factories_are_registered_by_exact_identity():
    sum_metadata = factory_operation(_reduce.sum)
    reduce_metadata = factory_operation(_reduce.block_reduce_builtin)

    assert (sum_metadata.operation, sum_metadata.namespace) == ("sum", "block")
    assert (reduce_metadata.operation, reduce_metadata.namespace) == (
        "block_reduce_builtin",
        "block",
    )
    warp_sum_metadata = factory_operation(_reduce.warp_sum)
    warp_reduce_metadata = factory_operation(_reduce.warp_reduce_builtin)
    assert (warp_sum_metadata.operation, warp_sum_metadata.namespace) == (
        "warp_sum",
        "warp",
    )
    assert (warp_reduce_metadata.operation, warp_reduce_metadata.namespace) == (
        "warp_reduce_builtin",
        "warp",
    )


def test_compile_context_requires_every_direct_header_and_exact_toolkit(
    monkeypatch,
):
    requested = []
    validated = []
    paths = SimpleNamespace(
        cuda=(Path("/cuda/include"),),
        as_tuple=lambda: (Path("/cccl/include"), Path("/cuda/include")),
    )
    libraries = SimpleNamespace(
        nvrtc_path="/cuda/lib/libnvrtc.so.13",
        nvrtc_builtins_path="/cuda/lib/libnvrtc-builtins.so.13.3",
        toolkit_version=(13, 3),
    )
    monkeypatch.setattr(
        _nvrtc,
        "resolve_include_paths",
        lambda **kwargs: requested.append(kwargs) or paths,
    )
    monkeypatch.setattr(
        _nvrtc,
        "preload_toolkit_compiler_libraries",
        lambda include_dirs: validated.append(("preload", include_dirs)) or libraries,
    )
    monkeypatch.setattr(_nvrtc, "_load_nvrtc", lambda: object())
    monkeypatch.setattr(_nvrtc, "_version", lambda nvrtc: (13, 3))
    monkeypatch.setattr(
        _nvrtc,
        "validate_nvrtc_version",
        lambda selected, version: validated.append((selected, version)),
    )
    monkeypatch.setattr(
        _nvrtc,
        "include_dirs_identity",
        lambda include_dirs: SimpleNamespace(digest="headers-a"),
    )
    monkeypatch.setattr(
        _nvrtc.cuda,
        "get_current_device",
        lambda: pytest.fail("the explicit compiler target must be authoritative"),
    )
    state = SimpleNamespace(metadata={"targetoptions": {"chip": "sm_90a"}})

    context = _nvrtc.resolve_compile_context(state)

    assert requested[0]["required_headers"] == (
        "cub/block/block_reduce.cuh",
        "cub/warp/warp_reduce.cuh",
        "cuda/functional",
        "cuda/std/cstdint",
        "cuda/std/functional",
    )
    assert validated[0] == ("preload", paths.cuda)
    assert validated[1] == (libraries, (13, 3))
    assert context == _context(architecture="90a")


@pytest.mark.parametrize(
    "dtype",
    [
        types.int8,
        types.uint8,
        types.int16,
        types.uint16,
        types.int32,
        types.uint32,
        types.int64,
        types.uint64,
        types.float32,
        types.float64,
    ],
)
def test_provider_declares_the_supported_scalar_numeric_types(dtype):
    assert dtype in _reduce._CPP_TYPES


@pytest.mark.parametrize("dtype", [types.boolean, types.complex64, types.complex128])
def test_provider_rejects_bool_and_complex_before_nvrtc(monkeypatch, dtype):
    monkeypatch.setattr(
        _reduce,
        "_provider_for_context",
        lambda *args: pytest.fail("unsupported types must fail before NVRTC"),
    )

    with pytest.raises(TypeError, match="does not support scalar type"):
        _reduce._typed_provider(_marker_spec(), dtype, _context())


def test_provider_rejects_bitwise_float_before_nvrtc(monkeypatch):
    monkeypatch.setattr(
        _reduce,
        "_provider_for_context",
        lambda *args: pytest.fail("invalid operators must fail before NVRTC"),
    )

    with pytest.raises(TypeError, match="requires an integer scalar"):
        _reduce._typed_provider(
            _marker_spec(binary_op="bit_xor"), types.float32, _context()
        )


def test_generic_marker_specializes_from_the_overload_value_type(monkeypatch):
    registrations = []
    provider_requests = []
    provider = _reduce._Provider(
        extern=lambda value: value,
        ltoir_path="/tmp/provider.ltoir",
    )

    def overload(marker, **kwargs):
        assert kwargs == {
            "inline": "always",
            "typing_registry": _reduce.typing_registry,
        }

        def register(typer):
            registrations.append((marker, typer))
            return typer

        return register

    monkeypatch.setattr(_reduce, "overload", overload)
    monkeypatch.setattr(
        _reduce,
        "_typed_provider",
        lambda spec, dtype, context: (
            provider_requests.append((spec, dtype, context)) or provider
        ),
    )
    marker_spec = _marker_spec()
    context = _context()
    _reduce._marker_for.cache_clear()
    try:
        marker = _reduce._marker_for(marker_spec, context)
        assert provider_requests == []

        implementation = registrations[0][1](types.int32)
    finally:
        _reduce._marker_for.cache_clear()

    assert registrations[0][0] is marker
    assert provider_requests == [(marker_spec, types.int32, context)]
    assert implementation(types.int32) is types.int32


def test_typed_provider_is_governed_by_the_concrete_portable_plan(monkeypatch):
    plans = []
    providers = []
    original = _reduce.plan_group_primitive
    expected = _reduce._Provider(extern=object(), ltoir_path="/tmp/provider.ltoir")

    def plan(call, launch):
        result = original(call, launch)
        plans.append(result)
        return result

    monkeypatch.setattr(_reduce, "plan_group_primitive", plan)
    monkeypatch.setattr(
        _reduce,
        "_provider_for_context",
        lambda spec, cpp_type, context: (
            providers.append((spec, cpp_type, context)) or expected
        ),
    )
    context = _context()

    actual = _reduce._typed_provider(_marker_spec(), types.int32, context)

    assert actual is expected
    assert len(plans) == 1
    portable = plans[0].require_supported()
    assert portable.call.operation.dtype is types.int32
    assert portable.implementation is not None
    assert providers == [(portable.implementation, "::cuda::std::int32_t", context)]


def test_warp_typed_provider_replans_with_actual_dtype_and_group(monkeypatch):
    plans = []
    providers = []
    original = _reduce.plan_group_primitive
    expected = _reduce._Provider(extern=object(), ltoir_path="/tmp/warp.ltoir")

    def plan(call, launch):
        result = original(call, launch)
        plans.append(result)
        return result

    monkeypatch.setattr(_reduce, "plan_group_primitive", plan)
    monkeypatch.setattr(
        _reduce,
        "_provider_for_context",
        lambda spec, cpp_type, context: (
            providers.append((spec, cpp_type, context)) or expected
        ),
    )
    context = _context()

    actual = _reduce._typed_provider(
        _warp_marker_spec(binary_op="max"), types.int32, context
    )

    assert actual is expected
    assert len(plans) == 1
    portable = plans[0].require_supported()
    assert portable.resolved_group.kind == "warp"
    assert portable.call.operation.dtype is types.int32
    assert portable.implementation is not None
    assert providers == [(portable.implementation, "::cuda::std::int32_t", context)]


def test_provider_cache_and_symbol_include_the_exact_compile_context(monkeypatch):
    compiled = []
    declared = []
    monkeypatch.setattr(
        _reduce._nvrtc,
        "compile_lto_ir",
        lambda source, context: compiled.append((source, context)) or b"ltoir",
    )
    monkeypatch.setattr(
        _reduce.cuda,
        "declare_device",
        lambda name, sig, **kwargs: declared.append((name, sig, kwargs)) or object(),
    )
    spec = _spec(valid_items=False)
    contexts = (
        _context(),
        _context(nvrtc_builtins_path="/cuda/lib/libnvrtc-builtins.so.13.2"),
        _context(header_identity="headers-b"),
        _context(architecture="75"),
    )
    _reduce._provider_for_context.cache_clear()
    try:
        first = _reduce._provider_for_context(
            spec, _reduce._CPP_TYPES[spec.dtype], contexts[0]
        )
        assert (
            _reduce._provider_for_context(
                spec, _reduce._CPP_TYPES[spec.dtype], contexts[0]
            )
            is first
        )
        remaining = [
            _reduce._provider_for_context(
                spec,
                _reduce._CPP_TYPES[spec.dtype],
                context,
            )
            for context in contexts[1:]
        ]
    finally:
        _reduce._provider_for_context.cache_clear()

    assert [context for _, context in compiled] == list(contexts)
    assert first.ltoir_path.endswith(".ltoir")
    assert all(provider.ltoir_path.endswith(".ltoir") for provider in remaining)
    assert len({declaration[0] for declaration in declared}) == len(contexts)
    assert declared[0][2]["abi"] == "c"
    assert declared[0][2]["link"] == [first.ltoir_path]
