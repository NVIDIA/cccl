# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from types import SimpleNamespace

import pytest

pytest.importorskip("numba_cuda_mlir")

from numba_cuda_mlir import types

from cuda.coop.numba_mlir._compiler import _nvrtc, _provider


def _spec(**changes):
    values = {
        "block_dim": (32, 2, 1),
        "operation": "reduce",
        "binary_op": "max",
        "algorithm": "raking",
        "has_valid_items": True,
    }
    values.update(changes)
    return _provider.ReductionMarkerSpec(**values)


def test_generated_provider_is_direct_root_only_cub_block_reduce():
    source = _provider._source(
        symbol="test_symbol",
        cpp_type="::cuda::std::int32_t",
        spec=_spec(),
    )

    assert "#include <cub/block/block_reduce.cuh>" in source
    assert "#include <cuda/functional>" in source
    assert "::cub::BlockReduce<" in source
    assert "::cub::BLOCK_REDUCE_RAKING" in source
    assert ".Reduce(value, ::cuda::maximum<T>{}, valid_items)" in source
    assert "__syncthreads();" in source
    assert 'extern "C" __device__' in source


def test_sum_provider_selects_sum_without_a_callback():
    source = _provider._source(
        symbol="test_symbol",
        cpp_type="float",
        spec=_spec(
            operation="sum",
            binary_op="sum",
            algorithm="warp_reductions",
            has_valid_items=False,
        ),
    )

    assert ".Sum(value)" in source
    assert "BLOCK_REDUCE_WARP_REDUCTIONS" in source


def test_default_reduce_selects_sum_without_a_callback():
    source = _provider._source(
        symbol="test_symbol",
        cpp_type="float",
        spec=_spec(
            operation="reduce",
            binary_op="sum",
            has_valid_items=False,
        ),
    )

    assert ".Sum(value)" in source


def test_header_resolution_requires_all_provider_headers(monkeypatch):
    requested = []
    monkeypatch.setattr(
        _nvrtc,
        "resolve_include_paths",
        lambda **kwargs: (
            requested.append(kwargs)
            or SimpleNamespace(as_tuple=lambda: ("/cccl/include", "/cuda/include"))
        ),
    )

    assert _nvrtc.include_paths() == ("/cccl/include", "/cuda/include")
    assert requested[0]["required_headers"] == (
        "cub/block/block_reduce.cuh",
        "cuda/functional",
        "cuda/std/cstdint",
        "cuda/std/functional",
    )


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
    assert dtype in _provider._CPP_TYPES


@pytest.mark.parametrize("dtype", [types.boolean, types.complex64, types.complex128])
def test_provider_rejects_bool_and_complex_before_nvrtc(monkeypatch, dtype):
    monkeypatch.setattr(
        _provider._nvrtc,
        "include_paths",
        lambda: pytest.fail("unsupported types must fail before NVRTC"),
    )

    with pytest.raises(TypeError, match="does not support scalar type"):
        _provider._provider(dtype, _spec(has_valid_items=False))


def test_provider_rejects_bitwise_float_before_nvrtc(monkeypatch):
    monkeypatch.setattr(
        _provider._nvrtc,
        "include_paths",
        lambda: pytest.fail("invalid operators must fail before NVRTC"),
    )

    with pytest.raises(TypeError, match="requires an integer scalar"):
        _provider._provider(
            types.float32,
            _spec(binary_op="bit_xor", has_valid_items=False),
        )


def test_provider_cache_includes_the_resolved_compiler_context(monkeypatch):
    compiled = []
    declared = []
    include_paths = [
        ("/cccl/first", "/cuda/include"),
        ("/cccl/second", "/cuda/include"),
    ]
    current_includes = [include_paths[0]]
    current_cc = [(9, 0)]
    monkeypatch.setattr(_provider._nvrtc, "include_paths", lambda: current_includes[0])
    monkeypatch.setattr(_provider._nvrtc, "version", lambda: (13, 3))
    monkeypatch.setattr(
        _provider._nvrtc,
        "compile_lto_ir",
        lambda source, cc, includes: (
            compiled.append((source, cc, includes)) or b"ltoir"
        ),
    )
    monkeypatch.setattr(
        _provider.cuda,
        "get_current_device",
        lambda: SimpleNamespace(compute_capability=current_cc[0]),
    )
    monkeypatch.setattr(
        _provider.cuda,
        "declare_device",
        lambda name, sig, **kwargs: declared.append((name, sig, kwargs)) or object(),
    )
    spec = _spec(has_valid_items=False)
    _provider._provider_for_context.cache_clear()
    try:
        first = _provider._provider(types.int32, spec)
        assert _provider._provider(types.int32, spec) is first
        current_includes[0] = include_paths[1]
        second = _provider._provider(types.int32, spec)
        current_cc[0] = (7, 5)
        third = _provider._provider(types.int32, spec)
    finally:
        _provider._provider_for_context.cache_clear()

    assert len(compiled) == 3
    assert compiled[0][1:] == ("90", include_paths[0])
    assert compiled[1][1:] == ("90", include_paths[1])
    assert compiled[2][1:] == ("75", include_paths[1])
    assert first.ltoir_path.endswith(".ltoir")
    assert second.ltoir_path.endswith(".ltoir")
    assert third.ltoir_path.endswith(".ltoir")
    assert len({declaration[0] for declaration in declared}) == 3
    assert declared[0][2]["abi"] == "c"
    assert declared[0][2]["link"] == [first.ltoir_path]


def test_provider_context_uses_the_explicit_compiler_target(monkeypatch):
    monkeypatch.setattr(_provider._nvrtc, "version", lambda: (13, 3))
    monkeypatch.setattr(
        _provider._nvrtc,
        "include_paths",
        lambda: ("/cccl/include", "/cuda/include"),
    )
    monkeypatch.setattr(
        _provider.cuda,
        "get_current_device",
        lambda: pytest.fail("an explicit compiler target must be authoritative"),
    )
    state = SimpleNamespace(
        metadata={"targetoptions": {"chip": "sm_90a"}},
    )

    context = _provider.resolve_provider_context(state)

    assert context == _provider.ProviderContext(
        architecture="90a",
        nvrtc_version=(13, 3),
        include_paths=("/cccl/include", "/cuda/include"),
    )


def test_marker_identity_includes_the_resolved_provider_context():
    spec = _spec(has_valid_items=False)
    first_context = _provider.ProviderContext(
        architecture="90",
        nvrtc_version=(13, 3),
        include_paths=("/cccl/include", "/cuda/include"),
    )
    second_context = _provider.ProviderContext(
        architecture="75",
        nvrtc_version=(13, 3),
        include_paths=("/cccl/include", "/cuda/include"),
    )

    first = _provider.marker_for(spec, first_context)

    assert _provider.marker_for(spec, first_context) is first
    assert _provider.marker_for(spec, second_context) is not first
