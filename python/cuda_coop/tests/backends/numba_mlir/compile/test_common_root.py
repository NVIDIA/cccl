# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from ..support.compile import compile_for_launch


def test_common_root_planner_requests_configured_launch(optional_backend):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import compiler, types

    from cuda import coop
    from cuda.coop._core import root_api

    def common_root_reduce(value):
        coop.reduce(coop.this_block(), value)

    expected = (
        "whole-function planner requires launch metadata; "
        "compile through a configured kernel launch"
    )
    assert root_api._backend_module_name() is None
    with pytest.raises(RuntimeError) as exc_info:
        compiler.compile(
            common_root_reduce,
            types.void(types.int32),
            device=False,
            abi="numba",
            cc=(8, 0),
        )

    assert type(exc_info.value) is RuntimeError
    assert str(exc_info.value) == expected
    assert exc_info.value.__cause__ is None
    assert root_api._backend_module_name() is None


@pytest.mark.gpu
def test_portable_root_sum_example_compiles_from_its_source_module(
    source_examples,
    numba_mlir_cuda_available,
):
    del source_examples, numba_mlir_cuda_available
    pytest.importorskip("numba_cuda_mlir.cuda")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda import typing as cuda_typing

    import cuda.coop.numba_mlir  # noqa: F401
    from examples.numba_mlir import portable_root_sum

    array_type = types.Array(types.int32, 1, "C")
    signature = cuda_typing.signature(types.none, array_type, array_type, array_type)
    inspect_key, result = compile_for_launch(
        portable_root_sum.portable_root_sum_kernel,
        signature,
        block=portable_root_sum.THREADS,
    )

    cubin = result.metadata.get("cubin")
    assert isinstance(cubin, bytes)
    assert cubin.startswith(b"\x7fELF")
    records = result.metadata["__cuda_coop_numba_mlir_materialized_specializations__"]
    assert {record[0].split("<", 1)[0] for record in records} == {
        "BlockLoad",
        "BlockStore",
    }
    assert (
        portable_root_sum.portable_root_sum_kernel.get_metadata(inspect_key)["cubin"]
        == cubin
    )
