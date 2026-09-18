# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Load Numba-CUDA-MLIR and transactionally activate compiler hooks.

Qualified backend import owns activation, while this module owns runtime
capability diagnostics and rollback-safe compiler registry mutation.
"""

from __future__ import annotations

import importlib
import sys
from threading import RLock
from typing import Any

from ._numba_mlir_compat import (
    _get_numba_mlir_compat,
    _NumbaMlirBackendImportError,
    _NumbaMlirCompilerCompat,
    _RegistrationSnapshot,
    _runtime_requirement,
)

_BACKEND_PACKAGE = __package__.removesuffix("._compiler")
_REGISTRATION_MODULES = frozenset(
    {
        f"{_BACKEND_PACKAGE}._compiler._group_planner",
        f"{_BACKEND_PACKAGE}._compiler._rewrite",
    }
)
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
    """Register compiler planners and rewrites for this qualified import."""

    with _activation_lock:
        _initialize_runtime_hooks_transaction()


def _initialize_runtime_hooks_transaction() -> None:
    """Run one compiler-hook registration transaction."""

    _require_runtime()
    compat = _get_numba_mlir_compat()
    snapshot = _snapshot_registrations(compat)
    package_name = _BACKEND_PACKAGE
    loaded_backend_modules = frozenset(
        name for name in sys.modules if name.startswith(f"{package_name}.")
    )
    try:
        planner_module = importlib.import_module(f"{package_name}._compiler._rewrite")
        group_planner_module = importlib.import_module(
            f"{package_name}._compiler._group_planner"
        )
        _verify_registration_postconditions(
            snapshot,
            planner_module,
            group_planner_module,
            compat=compat,
        )
    except BaseException:
        _restore_registrations(snapshot)
        for name in tuple(sys.modules):
            if (
                name.startswith(f"{package_name}.")
                and name not in loaded_backend_modules
            ):
                sys.modules.pop(name, None)
        raise


def _verify_registration_postconditions(
    snapshot: _RegistrationSnapshot,
    planner_module: Any,
    group_planner_module: Any,
    *,
    compat: _NumbaMlirCompilerCompat | None = None,
) -> None:
    expected_planners = (
        (
            "CoopGroupHierarchyPlanner",
            getattr(group_planner_module, "CoopGroupHierarchyPlanner", None),
        ),
        (
            "CoopWholeFunctionPlanner",
            getattr(planner_module, "CoopWholeFunctionPlanner", None),
        ),
    )
    if compat is None:
        compat = _get_numba_mlir_compat()
    registration_counts = compat.registration_counts(
        snapshot,
        expected_planners,
        (
            "CoopSinglePhaseRewrite",
            getattr(planner_module, "CoopSinglePhaseRewrite", None),
        ),
    )
    invalid = tuple(
        f"{name}={count}" for name, count in registration_counts.items() if count != 1
    )
    if invalid:
        raise _NumbaMlirBackendImportError(
            "registration-postcondition-failed",
            "cuda.coop.numba_mlir called the compiler registration APIs, but "
            "the required planner and rewrite hooks were not each registered "
            "exactly once: " + ", ".join(invalid),
            registration_counts=registration_counts,
        )


def _snapshot_registrations(
    compat: _NumbaMlirCompilerCompat | None = None,
) -> _RegistrationSnapshot:
    """Snapshot compiler registries populated during backend activation."""

    if compat is None:
        compat = _get_numba_mlir_compat()
    try:
        return compat.snapshot_registrations()
    except (AttributeError, TypeError) as exc:
        raise _NumbaMlirBackendImportError(
            "registration-transaction-unavailable",
            "cuda.coop.numba_mlir cannot transactionally register its "
            "planners and rewrite with the installed numba-cuda-mlir runtime.",
            cause=exc,
        ) from exc


def _restore_registrations(snapshot: _RegistrationSnapshot) -> None:
    """Remove only this backend's additions after a failed import.

    Other extensions may register with Numba-CUDA-MLIR while this backend is
    importing.  Replacing either registry with its pre-import snapshot would
    erase those independent additions.  Instead, retain the occurrences that
    existed in the snapshot and delete only new classes owned by the two
    activation modules.  Registrations from every other module remain in
    place, including ones appended concurrently during rollback.
    """

    _NumbaMlirCompilerCompat.restore_registrations(
        snapshot,
        owned_modules=_REGISTRATION_MODULES,
    )


__all__: tuple[str, ...] = ()
