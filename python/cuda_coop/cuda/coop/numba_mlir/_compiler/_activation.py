# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Transactionally activate the Numba-CUDA-MLIR planner."""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
from dataclasses import dataclass
from typing import Any

_MINIMUM_RUNTIME_VERSION = "0.5.0"
_MAXIMUM_RUNTIME_VERSION = "0.6"
_RUNTIME_INSTALL_HINT = (
    "'cuda-coop[numba-cuda-mlir-cu12]' for CUDA 12 or "
    "'cuda-coop[numba-cuda-mlir-cu13]' for CUDA 13"
)


class NumbaMlirBackendImportError(ImportError):
    """Structured qualified-backend import failure."""

    def __init__(self, reason_code: str, message: str, *, cause=None, **details):
        super().__init__(message)
        self.backend = "numba-cuda-mlir"
        self.reason_code = reason_code
        self.details = details
        if cause is not None:
            self.__cause__ = cause


@dataclass(frozen=True)
class _RegistrationSnapshot:
    registry: Any
    planners: tuple[type, ...]


def _runtime_requirement(runtime: Any = None) -> str:
    version = getattr(runtime, "__version__", None)
    if not isinstance(version, str) or not version:
        try:
            version = importlib.metadata.version("numba-cuda-mlir")
        except importlib.metadata.PackageNotFoundError:
            version = None
    detected = "no Numba-CUDA-MLIR distribution was detected"
    if version:
        detected = f"detected numba-cuda-mlir=={version}"
    return (
        f"{detected}; cuda.coop requires numba-cuda-mlir>="
        f"{_MINIMUM_RUNTIME_VERSION},<{_MAXIMUM_RUNTIME_VERSION}. "
        f"Install {_RUNTIME_INSTALL_HINT}."
    )


def _load_runtime() -> Any:
    try:
        runtime = importlib.import_module("numba_cuda_mlir")
    except ImportError as error:
        missing = getattr(error, "name", None)
        reason = (
            "backend-runtime-missing"
            if missing == "numba_cuda_mlir"
            else "transitive-runtime-import-failed"
        )
        raise NumbaMlirBackendImportError(
            reason,
            "cuda.coop.numba_mlir could not import Numba-CUDA-MLIR; "
            + _runtime_requirement(),
            cause=error,
            missing=missing,
        ) from error
    except Exception as error:  # noqa: BLE001 - preserve import diagnostics
        raise NumbaMlirBackendImportError(
            "transitive-runtime-import-failed",
            "cuda.coop.numba_mlir found Numba-CUDA-MLIR but importing it "
            f"failed with {type(error).__name__}; " + _runtime_requirement(),
            cause=error,
            exception_type=type(error).__name__,
        ) from error
    try:
        cuda_module = importlib.import_module("numba_cuda_mlir.cuda")
    except ImportError as error:
        missing = getattr(error, "name", None)
        reason = (
            "conflicting-backend-runtime"
            if missing == "numba_cuda_mlir.cuda"
            else "transitive-runtime-import-failed"
        )
        raise NumbaMlirBackendImportError(
            reason,
            "cuda.coop.numba_mlir found Numba-CUDA-MLIR but its CUDA "
            "compiler runtime is unavailable; " + _runtime_requirement(runtime),
            cause=error,
            missing=missing,
        ) from error
    except Exception as error:  # noqa: BLE001 - preserve import diagnostics
        raise NumbaMlirBackendImportError(
            "transitive-runtime-import-failed",
            "cuda.coop.numba_mlir found Numba-CUDA-MLIR but its CUDA "
            f"compiler failed to import with {type(error).__name__}; "
            + _runtime_requirement(runtime),
            cause=error,
            exception_type=type(error).__name__,
        ) from error
    return cuda_module


def _snapshot_registration() -> _RegistrationSnapshot:
    try:
        planner_module = importlib.import_module(
            "numba_cuda_mlir._whole_function_planners"
        )
        registry = planner_module._planner_registry
        with registry._lock:
            planners = tuple(registry._planners)
    except (ImportError, AttributeError, TypeError) as error:
        raise NumbaMlirBackendImportError(
            "registration-transaction-unavailable",
            "cuda.coop.numba_mlir cannot transactionally register its "
            "planner with this Numba-CUDA-MLIR runtime.",
            cause=error,
        ) from error
    return _RegistrationSnapshot(registry=registry, planners=planners)


def _restore_registration(snapshot: _RegistrationSnapshot) -> None:
    with snapshot.registry._lock:
        snapshot.registry._planners[:] = snapshot.planners


_initialized = False


def _initialize_runtime_hooks() -> None:
    """Register exactly one whole-function planner, with rollback on failure."""

    global _initialized
    if _initialized:
        return
    _load_runtime()
    try:
        extending = importlib.import_module("numba_cuda_mlir.extending")
    except ImportError as error:
        raise NumbaMlirBackendImportError(
            "runtime-hook-api-import-failed",
            "cuda.coop.numba_mlir could not import the Numba-CUDA-MLIR "
            "compiler extension API.",
            cause=error,
        ) from error
    required = (
        "WholeFunctionPlanner",
        "overload",
        "refresh_registries",
        "register_planner",
        "require_launch_config",
    )
    missing = tuple(
        name for name in required if not callable(getattr(extending, name, None))
    )
    if not hasattr(extending, "typing_registry"):
        missing += ("typing_registry",)
    if missing:
        raise NumbaMlirBackendImportError(
            "incomplete-runtime-hook-api",
            "cuda.coop.numba_mlir requires missing compiler capabilities: "
            + ", ".join(missing),
            missing_capabilities=missing,
        )

    snapshot = _snapshot_registration()
    package_name = __package__.removesuffix("._compiler")
    loaded_modules = frozenset(
        name for name in sys.modules if name.startswith(f"{package_name}._compiler.")
    )
    phase = "planner import"
    try:
        planner_module = importlib.import_module(f"{package_name}._compiler._planner")
        planner = planner_module.CoopBlockReducePlanner
        with snapshot.registry._lock:
            count = snapshot.registry._planners.count(planner)
        if count != 1:
            raise NumbaMlirBackendImportError(
                "registration-postcondition-failed",
                "cuda.coop.numba_mlir did not register exactly one block "
                f"reduction planner (count={count}).",
                registration_count=count,
            )
        phase = "registry refresh"
        extending.refresh_registries()
    except BaseException as error:
        _restore_registration(snapshot)
        for name in tuple(sys.modules):
            if (
                name.startswith(f"{package_name}._compiler.")
                and name not in loaded_modules
            ):
                sys.modules.pop(name, None)
        if isinstance(
            error,
            (NumbaMlirBackendImportError, KeyboardInterrupt, SystemExit),
        ):
            raise
        raise NumbaMlirBackendImportError(
            "backend-hook-activation-failed",
            f"cuda.coop.numba_mlir failed during {phase}; " + _runtime_requirement(),
            cause=error,
            activation_phase=phase,
            exception_type=type(error).__name__,
        ) from error
    _initialized = True


__all__ = ["NumbaMlirBackendImportError"]
