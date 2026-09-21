# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualified backends preserve the common kernel surface and call shapes."""

import importlib
import inspect

import pytest

from cuda.coop._core import api as common_api

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]

_COMMON_FUNCTIONS = tuple(
    name
    for name in common_api.__all__
    if inspect.isfunction(getattr(common_api, name))
    and name not in {"TempStorage", "ThreadData"}
)
_POSITIONAL_KINDS = {
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
}


@pytest.fixture(params=("cutlass", "numba_mlir"))
def qualified_api(request):
    runtime = "cutlass" if request.param == "cutlass" else "numba_cuda_mlir"
    pytest.importorskip(runtime)
    return importlib.import_module(f"cuda.coop.{request.param}")


def test_common_exports(qualified_api):
    assert set(common_api.__all__) <= set(qualified_api.__all__)
    for name in common_api.__all__:
        assert hasattr(qualified_api, name), name


@pytest.mark.parametrize("name", _COMMON_FUNCTIONS)
def test_common_function_call_shape(qualified_api, name):
    common = inspect.signature(getattr(common_api, name)).parameters
    qualified = inspect.signature(getattr(qualified_api, name)).parameters
    assert tuple(parameter for parameter in qualified if parameter in common) == tuple(
        common
    )
    for parameter, expected in common.items():
        actual = qualified[parameter]
        assert actual.kind == expected.kind
        assert actual.default == expected.default

    common_positional = tuple(
        parameter
        for parameter, value in common.items()
        if value.kind in _POSITIONAL_KINDS
    )
    qualified_positional = tuple(
        parameter
        for parameter, value in qualified.items()
        if value.kind in _POSITIONAL_KINDS
    )
    assert qualified_positional[: len(common_positional)] == common_positional
    for parameter, value in qualified.items():
        if parameter not in common:
            assert value.default is not inspect.Parameter.empty or value.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }
