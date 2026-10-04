# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Activate the Numba backend after checking its compiler runtime.

Load the compiler package, check its supported version, and import the CUDA
compiler interface before registering ``CoopWholeFunctionPlanner``. The
registration entry point acquires ``_activation_lock`` itself and imports all
planner dependencies before registering the planner as its final step. Failed
imports can therefore be retried without leaving a partially registered pass.

Runtime-load and version diagnostics include installation details, and wrapped
import failures retain their original causes. Planner-dependency import errors
propagate unchanged. Explicit activation raises these failures; automatic
backend discovery reports them as warnings.
"""

from __future__ import annotations

import importlib
from threading import RLock
from types import ModuleType
from typing import cast

from ._numba_mlir_compat import (
    NumbaMlirBackendImportError,
    _require_numba_mlir_version,
    _runtime_requirement,
)

assert __package__ is not None
_BACKEND_PACKAGE = __package__.removesuffix("._compiler")
_activation_lock = RLock()


def _load_runtime() -> (
    tuple[ModuleType, None] | tuple[None, NumbaMlirBackendImportError]
):
    """Load the Numba CUDA compiler and classify activation failures.

    Import ``numba_cuda_mlir``, validate its version, then import
    ``numba_cuda_mlir.cuda``. Classify import failures to distinguish
    a missing runtime, an incomplete or shadowed package, and a dependency
    that fails while either module is imported. A supported Numba-CUDA-MLIR
    installation includes ``.cuda``; its absence is an import-path or broken
    installation diagnostic, not a supported compiler configuration. For
    example, a local ``numba_cuda_mlir`` directory can shadow the installed
    package while distribution metadata still reports a supported version.
    This checks the Python compiler installation; it does not check GPU
    availability, driver support, or the installed CUDA Toolkit.

    Return import failures with their original exception as the cause for
    ``_require_runtime`` to raise. Version-validation failures raise directly,
    before importing the ``.cuda`` submodule. In either case, explicit
    registration and qualified imports propagate an ``ImportError``.
    Automatic registration catches it and warns so the common import works.

    Returns
    -------
    runtime : module or None
        The ``numba_cuda_mlir.cuda`` module on success, otherwise ``None``.
    error : NumbaMlirBackendImportError or None
        Failure with a reason code and import details, otherwise ``None``.
        Exactly one of the two return values is non-``None``.

    Raises
    ------
    NumbaMlirBackendImportError
        The runtime version cannot be established, falls outside the supported
        range, or the version checker's ``packaging`` dependency is missing.
    """

    try:
        runtime = importlib.import_module("numba_cuda_mlir")
    except ImportError as exc:
        missing = getattr(exc, "name", None)
        if missing == "numba_cuda_mlir":
            return (
                None,
                NumbaMlirBackendImportError(
                    "backend-runtime-missing",
                    "cuda.coop.numba_mlir requires a compatible "
                    f"Numba-CUDA-MLIR runtime. {_runtime_requirement()}",
                    cause=exc,
                    missing=missing,
                ),
            )
        return (
            None,
            NumbaMlirBackendImportError(
                "transitive-runtime-import-failed",
                "cuda.coop.numba_mlir found the Numba-CUDA-MLIR runtime, but "
                f"importing it failed at dependency {missing!r}. "
                f"{_runtime_requirement()}",
                cause=exc,
                missing=missing,
            ),
        )
    except Exception as exc:  # noqa: BLE001 - preserve backend import context
        return (
            None,
            NumbaMlirBackendImportError(
                "transitive-runtime-import-failed",
                "cuda.coop.numba_mlir found the Numba-CUDA-MLIR runtime, but "
                "importing it failed with "
                f"{type(exc).__name__}. {_runtime_requirement()}",
                cause=exc,
                exception_type=type(exc).__name__,
            ),
        )

    _require_numba_mlir_version(runtime)

    try:
        cuda_module = importlib.import_module("numba_cuda_mlir.cuda")
    except ImportError as exc:
        missing = getattr(exc, "name", None)
        if missing == "numba_cuda_mlir.cuda":
            return (
                None,
                NumbaMlirBackendImportError(
                    "conflicting-backend-runtime",
                    "cuda.coop.numba_mlir found a package named "
                    "'numba_cuda_mlir', but it does not provide the CUDA "
                    "compiler runtime at 'numba_cuda_mlir.cuda'. Remove the "
                    f"conflicting package. {_runtime_requirement(runtime)}",
                    cause=exc,
                    missing=missing,
                ),
            )
        return (
            None,
            NumbaMlirBackendImportError(
                "transitive-runtime-import-failed",
                "cuda.coop.numba_mlir found the Numba-CUDA-MLIR runtime, but "
                f"its CUDA compiler failed to import dependency {missing!r}. "
                f"{_runtime_requirement(runtime)}",
                cause=exc,
                missing=missing,
            ),
        )
    except Exception as exc:  # noqa: BLE001 - preserve backend import context
        return (
            None,
            NumbaMlirBackendImportError(
                "transitive-runtime-import-failed",
                "cuda.coop.numba_mlir found the Numba-CUDA-MLIR runtime, but "
                "its CUDA compiler failed to import with "
                f"{type(exc).__name__}. {_runtime_requirement(runtime)}",
                cause=exc,
                exception_type=type(exc).__name__,
            ),
        )
    return cuda_module, None


_cuda_module: ModuleType | None = None


def _require_runtime() -> ModuleType:
    """Return the validated CUDA compiler module, loading it on first use.

    Cache successful loads only. Explicit activation propagates the classified
    import or version error. Automatic registration catches it and warns.

    Returns
    -------
    ModuleType
        The ``numba_cuda_mlir.cuda`` module.

    Raises
    ------
    NumbaMlirBackendImportError
        The runtime is absent, unsupported, or fails to import. The diagnostic
        retains the reason and original exception established by the loader.
    """

    global _cuda_module
    if _cuda_module is None:
        runtime, error = _load_runtime()
        if error is not None:
            raise error
        _cuda_module = cast(ModuleType, runtime)
    return _cuda_module


def _initialize_runtime_hooks() -> None:
    """Register the planner after all compiler support has loaded.

    Serialize this backend's activation attempts and require a supported
    runtime before importing its implementation. Those imports do not register
    planner or rewrite hooks. Registering the single planner as the final step
    therefore keeps failed imports from leaving partially activated
    cooperative hooks; a later attempt can reuse successfully loaded modules.
    The runtime deduplicates repeated registration of the same planner class.

    Raises
    ------
    NumbaMlirBackendImportError
        The runtime cannot be loaded or its version is unsupported.
    ImportError
        A compiler integration dependency cannot be imported. The original
        import error propagates; automatic registration converts it to a
        warning, while explicit activation leaves it visible to the caller.
    """

    with _activation_lock:
        _require_runtime()
        planner_module = importlib.import_module(
            f"{_BACKEND_PACKAGE}._compiler._planner"
        )
        from numba_cuda_mlir.extending import register_planner

        register_planner(planner_module.CoopWholeFunctionPlanner)


__all__ = [
    "_initialize_runtime_hooks",
    "_require_runtime",
]
