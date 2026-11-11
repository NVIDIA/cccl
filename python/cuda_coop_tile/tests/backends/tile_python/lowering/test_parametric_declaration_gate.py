# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib
import os

import pytest

from ..support.e2e import configure_e2e_environment
from .test_cudatile_ir import _FakeCudaArray, _get_compile_result


def _varint(value: int) -> bytes:
    encode_varint = importlib.import_module(
        "cuda.tile._bytecode.basic"
    ).encode_varint
    buffer = bytearray()
    encode_varint(value, buffer)
    return bytes(buffer)


@pytest.mark.link
def test_tilelib_call_only_stack_resolves_sum_across_dtypes_and_extents(
    monkeypatch,
):
    """Gate declaration-free calls against the CCCL-owned TileLib."""
    include_path_env = "CUDA_COOP_TILE_PROVIDER_INCLUDE_PATHS"
    monkeypatch.setenv(include_path_env, os.environ.get(include_path_env, ""))
    configure_e2e_environment(monkeypatch)
    ct = importlib.import_module("cuda.tile")
    coop = importlib.import_module("cuda.coop.tile")
    default_tile_context = importlib.import_module(
        "cuda.tile._cext"
    ).default_tile_context
    preview_call_type = importlib.import_module(
        "cuda.tile_preview"
    ).PreviewForeignCall
    token_type = importlib.import_module("cuda.tile._ir.type").TokenTy
    tile_library = coop.get_cpp_tile_library_path().resolve()

    @ct.kernel
    def mixed_sum_kernel(
        i32_small_in: ct.Array,
        i32_large_in: ct.Array,
        f32_small_in: ct.Array,
        f32_large_in: ct.Array,
        i32_small_out: ct.Array,
        i32_large_out: ct.Array,
        f32_small_out: ct.Array,
        f32_large_out: ct.Array,
    ):
        i32_small = ct.load(i32_small_in, (0,), shape=(128,))
        i32_large = ct.load(i32_large_in, (0,), shape=(256,))
        f32_small = ct.load(f32_small_in, (0,), shape=(128,))
        f32_large = ct.load(f32_large_in, (0,), shape=(256,))
        ct.store(i32_small_out, (0,), coop.sum(i32_small, broadcast=True))
        ct.store(i32_large_out, (0,), coop.sum(i32_large, broadcast=True))
        ct.store(f32_small_out, (0,), coop.sum(f32_small, broadcast=True))
        ct.store(f32_large_out, (0,), coop.sum(f32_large, broadcast=True))

    arguments = (
        _FakeCudaArray((128,), typestr="<i4"),
        _FakeCudaArray((256,), typestr="<i4"),
        _FakeCudaArray((128,), typestr="<f4"),
        _FakeCudaArray((256,), typestr="<f4"),
        _FakeCudaArray((128,), typestr="<i4"),
        _FakeCudaArray((256,), typestr="<i4"),
        _FakeCudaArray((128,), typestr="<f4"),
        _FakeCudaArray((256,), typestr="<f4"),
    )
    result = _get_compile_result(
        mixed_sum_kernel._pyfunc,
        arguments,
        default_tile_context,
        # Force bytecode generation and TileIR linking so this does not stop at
        # an in-memory PreviewForeignCall shape check.
        return_bytecode=True,
        return_cubin=True,
        sm_arch=os.environ.get("CUDA_COOP_TILE_E2E_SM_ARCH"),
    )
    assert result.bytecode is not None
    assert result.cubin
    assert int.from_bytes(result.bytecode[10:12], "little") & 1
    assert _varint(16386) in result.bytecode
    assert b"cuda_coop_tile_sum_batched" in result.bytecode
    final_ir = result.final_ir[0]
    foreign_calls = [
        op for op in final_ir.traverse() if isinstance(op, preview_call_type)
    ]
    assert len(foreign_calls) == 4
    assert {op.symbol_name for op in foreign_calls} == {
        "cuda_coop_tile_sum_batched"
    }
    assert all(len(op.tilelibs) == 1 for op in foreign_calls)
    assert all(op.tilelibs[0][1] for op in foreign_calls)
    assert {op.tilelibs[0][0].resolve() for op in foreign_calls} == {
        tile_library
    }
    assert all(len(op.inputs) == 2 for op in foreign_calls)
    assert all(len(op.result_vars) == 2 for op in foreign_calls)
    assert all(op.token is not None for op in foreign_calls)
    assert all(
        isinstance(op.result_vars[-1].get_type(), token_type)
        for op in foreign_calls
    )
    assert not any(
        type(op).__name__ == "ForeignDeclCallIROp" for op in final_ir.traverse()
    )
    assert {
        (
            op.constant_params[0],
            op.constant_params[1].name,
        )
        for op in foreign_calls
    } == {
        (128, "int32"),
        (256, "int32"),
        (128, "float32"),
        (256, "float32"),
    }
