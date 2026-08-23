# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Validate and diagnose the separately installed CUTLASS DSL runtime."""

from __future__ import annotations

import functools
import importlib
import importlib.metadata
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Any

_MINIMUM_RUNTIME_VERSION = "4.8"
_RUNTIME_EXTRA = "cuda-coop[cutlass]"
_RUNTIME_DISTRIBUTIONS = ("nvidia-cutlass-dsl",)
_RUNTIME_MODULES = (
    "cutlass.cutlass_dsl",
    "cutlass.cute",
    "cutlass.base_dsl.compiler",
)


class CutlassRuntimeDependencyError(ImportError):
    """Structured failure to load a qualified CUTLASS backend."""

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
    """Validated CUTLASS modules and compiler type used by ``cuda.coop``."""

    cutlass_dsl: ModuleType
    cute: ModuleType
    compiler: ModuleType
    dsl_type: type


def _runtime_requirement() -> str:
    version = None
    distribution = None
    for candidate in _RUNTIME_DISTRIBUTIONS:
        try:
            version = importlib.metadata.version(candidate)
        except importlib.metadata.PackageNotFoundError:
            continue
        distribution = candidate
        break

    detected = "no CUTLASS DSL distribution was detected"
    if isinstance(version, str) and version:
        detected = f"detected {distribution}=={version}"
    return (
        f"{detected}; the public package floor is nvidia-cutlass-dsl>="
        f"{_MINIMUM_RUNTIME_VERSION}. Install '{_RUNTIME_EXTRA}'."
    )


def _runtime_import_error(error: ImportError) -> CutlassRuntimeDependencyError:
    missing = getattr(error, "name", None)
    if missing == "cutlass":
        return CutlassRuntimeDependencyError(
            "backend-runtime-missing",
            "cuda.coop.cutlass requires a compatible CUTLASS DSL runtime. "
            f"{_runtime_requirement()}",
            cause=error,
            missing=missing,
        )
    if isinstance(missing, str) and any(
        missing == module_name or missing.startswith(f"{module_name}.")
        for module_name in _RUNTIME_MODULES
    ):
        return CutlassRuntimeDependencyError(
            "conflicting-backend-runtime",
            "cuda.coop.cutlass found a package named 'cutlass', but it does not "
            "provide the complete CUTLASS DSL compiler runtime; missing "
            f"{missing!r}. Remove the conflicting package. "
            f"{_runtime_requirement()}",
            cause=error,
            missing=missing,
        )
    return CutlassRuntimeDependencyError(
        "transitive-runtime-import-failed",
        "cuda.coop.cutlass found the CUTLASS DSL runtime, but importing it "
        "failed at dependency "
        f"{missing!r}. {_runtime_requirement()}",
        cause=error,
        missing=missing,
    )


def _missing_capabilities(
    cutlass_dsl: ModuleType,
    cute: ModuleType,
    compiler: ModuleType,
) -> tuple[str, ...]:
    dsl_type = getattr(cutlass_dsl, "CuTeDSL", None)
    missing: list[str] = []
    if not isinstance(dsl_type, type):
        missing.append("cutlass.cutlass_dsl.CuTeDSL")
    else:
        for name in (
            "_get_dsl",
            "register_trace_context_factory",
            "register_trace_finalize_hook",
        ):
            if not callable(getattr(dsl_type, name, None)):
                missing.append(f"cutlass.cutlass_dsl.CuTeDSL.{name}")

    if not callable(getattr(cute, "_get_launch_facts", None)):
        missing.append("cutlass.cute._get_launch_facts")

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
    """Return CUTLASS modules with the capabilities required by ``cuda.coop``."""

    try:
        cutlass_dsl = importlib.import_module("cutlass.cutlass_dsl")
        cute = importlib.import_module("cutlass.cute")
        compiler = importlib.import_module("cutlass.base_dsl.compiler")
    except ImportError as error:
        raise _runtime_import_error(error) from error
    except Exception as error:
        raise CutlassRuntimeDependencyError(
            "transitive-runtime-import-failed",
            "cuda.coop.cutlass found the CUTLASS DSL runtime, but importing it "
            "failed with "
            f"{type(error).__name__}. "
            f"{_runtime_requirement()}",
            cause=error,
            exception_type=type(error).__name__,
        ) from error

    missing_capabilities = _missing_capabilities(cutlass_dsl, cute, compiler)
    if missing_capabilities:
        raise CutlassRuntimeDependencyError(
            "backend-runtime-incompatible",
            "cuda.coop.cutlass requires CUTLASS trace contexts, trace "
            "finalization, exact launch facts, architecture selection, and "
            "link-library merging; missing: "
            + ", ".join(missing_capabilities)
            + f". {_runtime_requirement()}",
            missing_capabilities=missing_capabilities,
        )

    return CutlassRuntime(
        cutlass_dsl=cutlass_dsl,
        cute=cute,
        compiler=compiler,
        dsl_type=cutlass_dsl.CuTeDSL,
    )


def raise_for_missing_cutlass_runtime(error: ImportError) -> None:
    missing = getattr(error, "name", None)
    if not isinstance(missing, str):
        return
    if missing == "cutlass" or missing.startswith("cutlass."):
        translated = _runtime_import_error(error)
        raise translated from error


def guard_cutlass_runtime(function: Callable[..., Any]) -> Callable[..., Any]:
    """Translate CUTLASS import failures from lazily imported provider modules."""

    @functools.wraps(function)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except ImportError as error:
            raise_for_missing_cutlass_runtime(error)
            raise

    return guarded


__all__ = [
    "CutlassRuntime",
    "CutlassRuntimeDependencyError",
    "guard_cutlass_runtime",
    "raise_for_missing_cutlass_runtime",
    "validate_cutlass_runtime",
]
