# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check qualified frontends preserve the common callable surface.

Each available compiler frontend must export every common name. Functions
keep the common parameter order, kinds, and defaults; qualified APIs may
add optional parameters. These checks compare signatures, not runtime
behavior or types.
"""

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
    """Import each frontend only when its compiler runtime is present.

    The module and runtime names differ for Numba. Skipping a missing runtime
    lets this shared surface check run in either backend environment.
    """

    runtime = "cutlass" if request.param == "cutlass" else "numba_cuda_mlir"
    pytest.importorskip(runtime)
    return importlib.import_module(f"cuda.coop.{request.param}")


def test_common_exports(qualified_api):
    assert set(common_api.__all__) <= set(qualified_api.__all__)
    for name in common_api.__all__:
        assert hasattr(qualified_api, name), name


@pytest.mark.parametrize("name", _COMMON_FUNCTIONS)
def test_common_function_call_shape(qualified_api, name):
    """Preserve common call shapes while allowing optional backend extensions.

    Common parameters keep their relative order, kinds, and defaults. Their
    positional order must also remain the prefix of the qualified signature,
    and added parameters cannot introduce a required argument.
    """

    common = inspect.signature(getattr(common_api, name)).parameters
    qualified = inspect.signature(getattr(qualified_api, name)).parameters
    assert tuple(
        parameter for parameter in qualified if parameter in common
    ) == tuple(common)
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
            assert (
                value.default is not inspect.Parameter.empty
                or value.kind
                in {
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                }
            )
