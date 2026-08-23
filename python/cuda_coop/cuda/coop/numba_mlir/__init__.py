# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Numba-CUDA-MLIR-qualified cooperative group building blocks."""

import importlib
import importlib.metadata
import operator
import struct
import sys
from dataclasses import dataclass
from typing import Any

from ._thread_group import (
    Hierarchy,
    ThreadGroup,
    ThreadHierarchy,
    this_block,
    this_cluster,
    this_grid,
    this_thread,
    this_warp,
)

_MINIMUM_RUNTIME_VERSION = "0.5.0"
_RUNTIME_EXTRA = "cuda-coop[numba-cuda-mlir]"


class _DefaultThreadDataAlignment:
    """Distinguish an omitted legacy keyword while preserving its signature."""

    def __repr__(self):
        return "8"


_DEFAULT_THREAD_DATA_ALIGNMENT = _DefaultThreadDataAlignment()


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
        f"{_MINIMUM_RUNTIME_VERSION}. Install '{_RUNTIME_EXTRA}'."
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
                f"importing it failed with {type(exc).__name__}. "
                f"{_runtime_requirement()}",
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


_cuda_module, _runtime_import_error = _load_runtime()


def _require_runtime():
    if _runtime_import_error is not None:
        raise _runtime_import_error
    return _cuda_module


_require_runtime()


def ThreadData(
    items_per_thread,
    dtype=None,
    *,
    alignas=_DEFAULT_THREAD_DATA_ALIGNMENT,
    alignment=None,
):
    """Create fixed-size thread-local storage for cooperative operations."""

    if isinstance(items_per_thread, bool):
        raise TypeError("items_per_thread must be an integer")
    try:
        items_per_thread = operator.index(items_per_thread)
    except TypeError as exc:
        raise TypeError("items_per_thread must be an integer") from exc
    if items_per_thread <= 0:
        raise ValueError("items_per_thread must be a positive integer")

    if alignas is _DEFAULT_THREAD_DATA_ALIGNMENT:
        alignas = 8 if alignment is None else alignment
    elif alignment is not None:
        if alignas != alignment:
            raise ValueError("alignas and alignment must match when both are set")

    if isinstance(alignas, bool):
        raise TypeError("alignment must be an integer")
    try:
        alignas = operator.index(alignas)
    except TypeError as exc:
        raise TypeError("alignment must be an integer") from exc
    if alignas <= 0:
        raise ValueError("alignment must be a positive integer")
    if alignas & (alignas - 1):
        raise ValueError("alignment must be a power of 2")
    pointer_size = struct.calcsize("P")
    if alignas % pointer_size:
        raise ValueError(f"alignment must be a multiple of {pointer_size}")

    return _require_runtime().local.array(
        items_per_thread,
        dtype,
        alignment=alignas,
    )


class TempStorage:
    """Shared-memory requirements for cooperative operations in one kernel."""

    def __init__(
        self,
        size_in_bytes=None,
        alignment=None,
        auto_sync=None,
        sharing="shared",
    ):
        if size_in_bytes is not None:
            if not isinstance(size_in_bytes, int) or isinstance(size_in_bytes, bool):
                raise TypeError("TempStorage size_in_bytes must be an integer or None.")
            if size_in_bytes <= 0:
                raise ValueError(
                    "TempStorage size_in_bytes must be a positive integer."
                )

        if alignment is not None:
            if not isinstance(alignment, int) or isinstance(alignment, bool):
                raise TypeError("TempStorage alignment must be an integer or None.")
            if alignment <= 0:
                raise ValueError("TempStorage alignment must be a positive integer.")
            if alignment & (alignment - 1):
                raise ValueError("TempStorage alignment must be a power of 2.")

        if not isinstance(sharing, str):
            raise TypeError(
                "TempStorage sharing must be a string: 'shared' or 'exclusive'."
            )
        sharing_value = sharing.strip().lower()
        if sharing_value not in {"shared", "exclusive"}:
            raise ValueError("TempStorage sharing must be 'shared' or 'exclusive'.")

        if auto_sync is not None and not isinstance(auto_sync, bool):
            raise TypeError("TempStorage auto_sync must be None/True/False.")
        if sharing_value == "exclusive" and auto_sync is True:
            raise ValueError(
                "TempStorage with sharing='exclusive' does not support auto_sync=True."
            )

        self.size_in_bytes = size_in_bytes
        self.alignment = alignment
        self.sharing = sharing_value
        self.auto_sync = (
            False
            if sharing_value == "exclusive"
            else (True if auto_sync is None else auto_sync)
        )


__all__ = [
    "BlockLoadAlgorithm",
    "BlockScanAlgorithm",
    "BlockStoreAlgorithm",
    "Hierarchy",
    "StatefulFunction",
    "TempStorage",
    "ThreadData",
    "ThreadGroup",
    "ThreadHierarchy",
    "WarpLoadAlgorithm",
    "WarpStoreAlgorithm",
    "adjacent_difference",
    "discontinuity",
    "exchange",
    "exclusive_scan",
    "exclusive_sum",
    "gpu_dataclass",
    "gpu_dataclass_argument_handler",
    "inclusive_scan",
    "inclusive_sum",
    "load",
    "local",
    "merge_sort_keys",
    "merge_sort_pairs",
    "radix_rank",
    "radix_sort_keys",
    "radix_sort_pairs",
    "reduce",
    "scan",
    "shared",
    "shuffle",
    "store",
    "sum",
    "this_block",
    "this_cluster",
    "this_grid",
    "this_thread",
    "this_warp",
    "topk_max_keys",
    "topk_max_pairs",
    "topk_min_keys",
    "topk_min_pairs",
]


def _initialize_runtime_hooks() -> None:
    """Register compiler planners and rewrites for this qualified import."""

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
    loaded_backend_modules = frozenset(
        name for name in sys.modules if name.startswith(f"{__name__}.")
    )
    try:
        planner_module = importlib.import_module(f"{__name__}._single_phase_rewrites")
        group_planner_module = importlib.import_module(f"{__name__}._group_rewrites")
        _verify_registration_postconditions(
            snapshot,
            planner_module,
            group_planner_module,
        )
    except BaseException:
        _restore_registrations(snapshot)
        for name in tuple(sys.modules):
            if name.startswith(f"{__name__}.") and name not in loaded_backend_modules:
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
        registered_rewrites = {
            kind: tuple(rewrite_classes)
            for kind, rewrite_classes in rewrite_registry.rewrites.items()
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
    """Roll compiler registries back after a failed backend import."""

    with snapshot.planner_registry._lock:
        snapshot.planner_registry._planners[:] = snapshot.planners
    registered_rewrites = snapshot.rewrite_registry.rewrites
    registered_rewrites.clear()
    registered_rewrites.update(
        {
            kind: list(rewrite_classes)
            for kind, rewrite_classes in snapshot.rewrites.items()
        }
    )


def __getattr__(name):
    if name == "local":
        value = _require_runtime().local
        globals()[name] = value
        return value
    if name == "shared":
        value = _require_runtime().shared
        globals()[name] = value
        return value
    if name == "StatefulFunction":
        value = importlib.import_module(f"{__name__}._types").StatefulFunction
        globals()[name] = value
        return value
    if name in {"gpu_dataclass", "gpu_dataclass_argument_handler"}:
        value = getattr(importlib.import_module(f"{__name__}._dataclass"), name)
        globals()[name] = value
        return value
    if name in {
        "adjacent_difference",
        "discontinuity",
        "exchange",
        "exclusive_scan",
        "exclusive_sum",
        "inclusive_scan",
        "inclusive_sum",
        "load",
        "merge_sort_keys",
        "merge_sort_pairs",
        "radix_rank",
        "radix_sort_keys",
        "radix_sort_pairs",
        "reduce",
        "scan",
        "shuffle",
        "store",
        "sum",
        "topk_max_keys",
        "topk_max_pairs",
        "topk_min_keys",
        "topk_min_pairs",
    }:
        value = getattr(importlib.import_module(f"{__name__}._group_ops"), name)
        globals()[name] = value
        return value
    if name in {
        "BlockLoadAlgorithm",
        "BlockScanAlgorithm",
        "BlockStoreAlgorithm",
        "WarpLoadAlgorithm",
        "WarpStoreAlgorithm",
    }:
        value = getattr(importlib.import_module(f"{__name__}._enums"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)


_initialize_runtime_hooks()
