# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Load Numba-CUDA-MLIR and transactionally activate compiler hooks.

Qualified backend import owns activation, while this module owns runtime
capability diagnostics and rollback-safe compiler registry mutation.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
from dataclasses import dataclass
from threading import RLock
from typing import Any

_MINIMUM_RUNTIME_VERSION = "0.5.0"
_RUNTIME_INSTALL_HINT = (
    "'cuda-coop[numba-cuda-mlir-cu12]' for CUDA 12 or "
    "'cuda-coop[numba-cuda-mlir-cu13]' for CUDA 13"
)
_BACKEND_PACKAGE = __package__.removesuffix("._compiler")
_REGISTRATION_MODULES = frozenset(
    {
        f"{_BACKEND_PACKAGE}._compiler._group_planner",
        f"{_BACKEND_PACKAGE}._compiler._rewrite",
    }
)
_activation_lock = RLock()


@dataclass(frozen=True)
class _RegistrationSnapshot:
    planner_registry: Any
    planners: tuple[type, ...]
    rewrite_registry: Any
    rewrites: dict[str, tuple[type, ...]]


class _NumbaMlirBackendImportError(ImportError):
    """Structured qualified-backend import failure."""

    def __init__(self, reason_code, message, *, cause=None, **details):
        super().__init__(message)
        self.backend = "numba-cuda-mlir"
        self.reason_code = reason_code
        self.details = details
        if cause is not None:
            self.__cause__ = cause


def _runtime_requirement(runtime=None) -> str:
    version = getattr(runtime, "__version__", None)
    if not isinstance(version, str) or not version:
        try:
            version = importlib.metadata.version("numba-cuda-mlir")
        except importlib.metadata.PackageNotFoundError:
            version = None

    detected = "no Numba-CUDA-MLIR distribution was detected"
    if isinstance(version, str) and version:
        detected = f"detected numba-cuda-mlir=={version}"
    return (
        f"{detected}; the public package floor is numba-cuda-mlir>="
        f"{_MINIMUM_RUNTIME_VERSION}. Install {_RUNTIME_INSTALL_HINT}."
    )


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
    try:
        extending = importlib.import_module("numba_cuda_mlir.extending")
        rewrites = importlib.import_module("numba_cuda_mlir.numba_cuda.core.rewrites")
    except ImportError as exc:
        raise _NumbaMlirBackendImportError(
            "runtime-hook-api-import-failed",
            "cuda.coop.numba_mlir could not import the required compiler "
            "hook API from numba-cuda-mlir.",
            cause=exc,
            missing=getattr(exc, "name", None),
        ) from exc

    required = {
        "numba_cuda_mlir.extending": (
            "WholeFunctionPlanner",
            "refresh_registries",
            "register_planner",
            "require_launch_config",
            "set_required_dynamic_shared_memory",
        ),
        "numba_cuda_mlir.numba_cuda.core.rewrites": (
            "Rewrite",
            "register_rewrite",
        ),
    }
    modules = {
        "numba_cuda_mlir.extending": extending,
        "numba_cuda_mlir.numba_cuda.core.rewrites": rewrites,
    }
    missing = tuple(
        f"{module_name}.{name}"
        for module_name, names in required.items()
        for name in names
        if not callable(getattr(modules[module_name], name, None))
    )
    if missing:
        raise _NumbaMlirBackendImportError(
            "incomplete-runtime-hook-api",
            "cuda.coop.numba_mlir requires compiler capabilities that are "
            "missing from the installed runtime: " + ", ".join(missing),
            missing_capabilities=missing,
        )

    snapshot = _snapshot_registrations(rewrites)
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
    expected_rewrite = getattr(planner_module, "CoopSinglePhaseRewrite", None)
    with snapshot.planner_registry._lock:
        planner_counts = {
            name: snapshot.planner_registry._planners.count(planner)
            if planner is not None
            else 0
            for name, planner in expected_planners
        }
    rewrite_count = (
        snapshot.rewrite_registry.rewrites.get("before-inference", []).count(
            expected_rewrite
        )
        if expected_rewrite is not None
        else 0
    )
    registration_counts = {
        **planner_counts,
        "CoopSinglePhaseRewrite": rewrite_count,
    }
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


def _snapshot_registrations(rewrites: Any) -> _RegistrationSnapshot:
    """Snapshot compiler registries populated during backend activation."""

    try:
        planner_module = importlib.import_module(
            "numba_cuda_mlir._whole_function_planners"
        )
        planner_registry = planner_module._planner_registry
        rewrite_registry = rewrites.rewrite_registry
        with planner_registry._lock:
            planners = tuple(planner_registry._planners)
        rewrite_lists = rewrite_registry.rewrites.copy()
        registered_rewrites = {
            kind: tuple(rewrite_classes)
            for kind, rewrite_classes in rewrite_lists.items()
        }
    except (ImportError, AttributeError, TypeError) as exc:
        raise _NumbaMlirBackendImportError(
            "registration-transaction-unavailable",
            "cuda.coop.numba_mlir cannot transactionally register its "
            "planners and rewrite with the installed numba-cuda-mlir runtime.",
            cause=exc,
        ) from exc
    return _RegistrationSnapshot(
        planner_registry=planner_registry,
        planners=planners,
        rewrite_registry=rewrite_registry,
        rewrites=registered_rewrites,
    )


def _restore_registrations(snapshot: _RegistrationSnapshot) -> None:
    """Remove only this backend's additions after a failed import.

    Other extensions may register with Numba-CUDA-MLIR while this backend is
    importing.  Replacing either registry with its pre-import snapshot would
    erase those independent additions.  Instead, retain the occurrences that
    existed in the snapshot and delete only new classes owned by the two
    activation modules.  Registrations from every other module remain in
    place, including ones appended concurrently during rollback.
    """

    with snapshot.planner_registry._lock:
        _remove_backend_additions(
            snapshot.planner_registry._planners,
            snapshot.planners,
        )

    registered_rewrites = snapshot.rewrite_registry.rewrites
    for kind, rewrite_classes in registered_rewrites.copy().items():
        _remove_backend_additions(
            rewrite_classes,
            snapshot.rewrites.get(kind, ()),
        )


def _remove_backend_additions(
    registrations: list[type],
    baseline: tuple[type, ...],
) -> None:
    """Delete post-snapshot backend registrations without replacing a list."""

    baseline_counts: dict[int, int] = {}
    for registration in baseline:
        identity = id(registration)
        baseline_counts[identity] = baseline_counts.get(identity, 0) + 1

    removal_indices: list[int] = []
    for index, registration in enumerate(registrations):
        identity = id(registration)
        remaining = baseline_counts.get(identity, 0)
        if remaining:
            baseline_counts[identity] = remaining - 1
        elif getattr(registration, "__module__", None) in _REGISTRATION_MODULES:
            removal_indices.append(index)

    # Public registration APIs append to these lists.  Removing by index from
    # the tail preserves any foreign append that races with this cleanup.
    for index in reversed(removal_indices):
        del registrations[index]


__all__: tuple[str, ...] = ()
