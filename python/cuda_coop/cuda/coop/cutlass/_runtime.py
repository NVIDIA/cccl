# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUTLASS DSL runtime validation for the qualified backend."""

from __future__ import annotations

import functools
import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any

_INSTALL_HINT = "cuda-coop[cutlass]"


class CutlassRuntimeDependencyError(ImportError):
    """A structured failure to activate the qualified CUTLASS backend."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        cause: BaseException | None = None,
        **details: Any,
    ) -> None:
        super().__init__(message)
        self.backend = "cutlass"
        self.reason_code = reason_code
        self.details = details
        if cause is not None:
            self.__cause__ = cause


@dataclass(frozen=True)
class CutlassRuntime:
    """Validated modules and compiler type used by the provider."""

    cutlass_dsl: ModuleType
    cute: ModuleType
    compiler: ModuleType
    common: ModuleType
    dsl_type: type


def _missing_capabilities(
    cutlass_dsl: ModuleType,
    cute: ModuleType,
    compiler: ModuleType,
    common: ModuleType,
) -> tuple[str, ...]:
    dsl_type = getattr(cutlass_dsl, "CuTeDSL", None)
    missing: list[str] = []
    if not isinstance(dsl_type, type):
        missing.append("cutlass.cutlass_dsl.CuTeDSL")
    else:
        for name in (
            "_get_dsl",
            "register_trace_finalize_hook",
            "trace_finalize_hooks",
        ):
            if not callable(getattr(dsl_type, name, None)):
                missing.append(f"cutlass.cutlass_dsl.CuTeDSL.{name}")
    if not callable(getattr(cute, "_get_launch_facts", None)):
        missing.append("cutlass.cute._get_launch_facts")
    if not callable(getattr(common, "get_current_env_manager", None)):
        missing.append("cutlass.base_dsl.common.get_current_env_manager")
    link_libraries = getattr(compiler, "LinkLibraries", None)
    if not callable(link_libraries):
        missing.append("cutlass.base_dsl.compiler.LinkLibraries")
    elif getattr(link_libraries, "_option_name", None) != "link-libraries":
        missing.append(
            "cutlass.base_dsl.compiler.LinkLibraries._option_name=link-libraries"
        )
    if not callable(getattr(compiler, "GPUArch", None)):
        missing.append("cutlass.base_dsl.compiler.GPUArch")
    return tuple(missing)


@functools.lru_cache(maxsize=1)
def validate_cutlass_runtime() -> CutlassRuntime:
    """Return a CUTLASS runtime with the capabilities required by ``cuda.coop``."""

    try:
        cutlass_dsl = importlib.import_module("cutlass.cutlass_dsl")
        cute = importlib.import_module("cutlass.cute")
        compiler = importlib.import_module("cutlass.base_dsl.compiler")
        common = importlib.import_module("cutlass.base_dsl.common")
    except ImportError as error:
        missing = getattr(error, "name", None)
        raise CutlassRuntimeDependencyError(
            "backend-runtime-missing",
            "cuda.coop.cutlass requires a compatible CUTLASS Python DSL; "
            f"install '{_INSTALL_HINT}'. Missing import: {missing!r}.",
            cause=error,
            missing=missing,
        ) from error
    except Exception as error:
        raise CutlassRuntimeDependencyError(
            "backend-runtime-import-failed",
            "cuda.coop.cutlass found CUTLASS, but importing its compiler runtime "
            f"failed with {type(error).__name__}.",
            cause=error,
            exception_type=type(error).__name__,
        ) from error

    missing_capabilities = _missing_capabilities(
        cutlass_dsl,
        cute,
        compiler,
        common,
    )
    if missing_capabilities:
        raise CutlassRuntimeDependencyError(
            "backend-runtime-incompatible",
            "cuda.coop.cutlass requires CUTLASS scoped trace finalization, a "
            "current compiler environment, exact launch facts, and link-library "
            "merging; missing: " + ", ".join(missing_capabilities),
            missing_capabilities=missing_capabilities,
        )

    return CutlassRuntime(
        cutlass_dsl=cutlass_dsl,
        cute=cute,
        compiler=compiler,
        common=common,
        dsl_type=cutlass_dsl.CuTeDSL,
    )


def is_current_cutlass_environment() -> bool:
    """Return whether CUTLASS owns the compiler environment in this context."""

    runtime = validate_cutlass_runtime()
    dsl = runtime.dsl_type._get_dsl()
    try:
        return runtime.common.get_current_env_manager() is dsl.envar
    except Exception:
        return False


__all__ = [
    "CutlassRuntime",
    "CutlassRuntimeDependencyError",
    "is_current_cutlass_environment",
    "validate_cutlass_runtime",
]
