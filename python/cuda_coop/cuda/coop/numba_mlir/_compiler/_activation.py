# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Load compiler support before registering the cooperative planner."""

from __future__ import annotations

import importlib
from threading import RLock
from typing import Any

from ._numba_mlir_compat import (
    _NumbaMlirBackendImportError,
    _require_numba_mlir_version,
    _runtime_requirement,
)

assert __package__ is not None
_BACKEND_PACKAGE = __package__.removesuffix("._compiler")
_activation_lock = RLock()


def _load_runtime() -> tuple[Any, _NumbaMlirBackendImportError | None]:
    try:
        runtime = importlib.import_module("numba_cuda_mlir")
    except ImportError as exc:
        missing = getattr(exc, "name", None)
        if missing == "numba_cuda_mlir":
            return (
                None,
                _NumbaMlirBackendImportError(
                    "backend-runtime-missing",
                    "cuda.coop.numba_mlir requires a compatible "
                    f"Numba-CUDA-MLIR runtime. {_runtime_requirement()}",
                    cause=exc,
                    missing=missing,
                ),
            )
        return (
            None,
            _NumbaMlirBackendImportError(
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
            _NumbaMlirBackendImportError(
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
                _NumbaMlirBackendImportError(
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
            _NumbaMlirBackendImportError(
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
            _NumbaMlirBackendImportError(
                "transitive-runtime-import-failed",
                "cuda.coop.numba_mlir found the Numba-CUDA-MLIR runtime, but "
                "its CUDA compiler failed to import with "
                f"{type(exc).__name__}. {_runtime_requirement(runtime)}",
                cause=exc,
                exception_type=type(exc).__name__,
            ),
        )
    return cuda_module, None


_cuda_module = None


def _require_runtime():
    global _cuda_module
    if _cuda_module is None:
        runtime, error = _load_runtime()
        if error is not None:
            raise error
        _cuda_module = runtime
    return _cuda_module


def _initialize_runtime_hooks() -> None:
    """Register the planner after all compiler support has loaded.

    Implementation modules do not register compiler hooks during import.
    A failed import can therefore be retried without undoing registrations
    or removing successfully imported modules. The runtime deduplicates
    repeated registration of the same planner class.
    """

    with _activation_lock:
        _require_runtime()
        planner_module = importlib.import_module(
            f"{_BACKEND_PACKAGE}._compiler._planner"
        )
        from numba_cuda_mlir.extending import register_planner

        register_planner(planner_module.CoopWholeFunctionPlanner)


__all__: tuple[str, ...] = ()
