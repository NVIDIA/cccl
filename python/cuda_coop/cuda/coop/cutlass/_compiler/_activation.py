# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Lightweight CUTLASS trace-context activation for qualified imports."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager

from cuda.coop._core.api._dispatch import _compiler_scope

from ._runtime import (
    CutlassRuntimeDependencyError,
    validate_cutlass_runtime,
)

_BACKEND_MODULE = "cuda.coop.cutlass"
_DISPATCHER_ATTR = "_cuda_coop_cutlass_trace_context_dispatcher"
_TARGET_ATTR = "_cuda_coop_cutlass_trace_context_target"


@contextmanager
def trace_context(compiler: object, function_name: str) -> Iterator[None]:
    """Select CUTLASS while its compiler processes one kernel function."""

    del compiler, function_name
    with _compiler_scope(_BACKEND_MODULE):
        yield


def _trace_context_dispatcher(
    compiler: object,
    function_name: str,
) -> AbstractContextManager[None]:
    dsl = validate_cutlass_runtime().dsl_type._get_dsl()
    target = getattr(dsl, _TARGET_ATTR)
    return target(compiler, function_name)


def register_trace_context() -> None:
    """Register common-root activation directly with the CUTLASS compiler."""

    runtime = validate_cutlass_runtime()
    dsl = runtime.dsl_type._get_dsl()
    register = getattr(dsl, "register_trace_context_factory", None)
    if not callable(register):
        raise CutlassRuntimeDependencyError(
            "backend-runtime-incompatible",
            "cuda.coop.cutlass requires callable "
            "register_trace_context_factory() support.",
            missing_capabilities=(
                "cutlass.cutlass_dsl.CuTeDSL.register_trace_context_factory",
            ),
        )
    if getattr(dsl, _DISPATCHER_ATTR, None) is None:
        register(_trace_context_dispatcher)
        setattr(dsl, _DISPATCHER_ATTR, _trace_context_dispatcher)
    setattr(dsl, _TARGET_ATTR, trace_context)


__all__ = ["register_trace_context", "trace_context"]
