# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Select CUTLASS only while its compiler owns the active environment."""

from __future__ import annotations

from cuda.coop._core.api._dispatch import _register_compiler_context_probe

from ._runtime import CutlassRuntimeDependencyError, validate_cutlass_runtime

_BACKEND_MODULE = "cuda.coop.cutlass"


def register_trace_context() -> None:
    """Register ownership after the optional runtime has initialized.

    The probe captures initialized runtime objects. Common host calls neither
    import a compiler nor instantiate a DSL just to select a backend.
    """

    runtime = validate_cutlass_runtime()
    dsl = runtime.dsl_type._get_dsl()
    environment = getattr(dsl, "envar", None)
    if environment is None:
        raise CutlassRuntimeDependencyError(
            "backend-runtime-incompatible",
            "cuda.coop.cutlass requires a compiler environment owned by CuTeDSL.",
            missing_capabilities=("cutlass.cutlass_dsl.CuTeDSL.envar",),
        )
    current_environment = runtime.common.get_current_env_manager

    def is_current_cutlass_environment() -> bool:
        return current_environment() is environment

    _register_compiler_context_probe(_BACKEND_MODULE, is_current_cutlass_environment)


__all__ = ["register_trace_context"]
