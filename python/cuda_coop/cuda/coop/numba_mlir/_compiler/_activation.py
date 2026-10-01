# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Register compiler hooks and roll back this backend's additions on failure."""

from __future__ import annotations

import importlib
import sys
from collections.abc import MutableSequence
from dataclasses import dataclass
from threading import RLock
from typing import Any

from ._numba_mlir_compat import (
    _NumbaMlirBackendImportError,
    _require_numba_mlir_version,
    _runtime_requirement,
)

assert __package__ is not None
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


def _load_runtime() -> tuple[Any, _NumbaMlirBackendImportError | None]:
    """Import the CUDA runtime and classify failures for backend activation.

    Import the top-level package before its CUDA module so an absent runtime,
    a conflicting package without CUDA support, and a broken dependency can
    produce different diagnostics. Preserve the original exception as the
    structured error's cause. Import failures are returned here so
    ``_require_runtime`` can raise them at the activation boundary.

    Returns
    -------
    runtime : module or None
        The ``numba_cuda_mlir.cuda`` module on success, otherwise ``None``.
    error : _NumbaMlirBackendImportError or None
        Failure with a reason code and import details, otherwise ``None``.
        Exactly one of the two return values is non-``None``.
    """

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
    """Register compiler planners and rewrites for this qualified import."""

    with _activation_lock:
        _initialize_runtime_hooks_transaction()


def _initialize_runtime_hooks_transaction() -> None:
    """Import compiler passes as one recoverable registration attempt.

    Require the runtime and compatibility layer before taking snapshots of
    the compiler registries and currently loaded backend submodules. Importing
    the rewrite and group-planner modules registers their classes as a side
    effect; verify that both planners and the before-inference rewrite are
    present exactly once, including when imports reuse existing modules.

    If importing or verification raises, remove this backend's new registry
    entries and newly loaded submodules before re-raising. This lets a later
    activation retry execute the registration code, while preserving modules
    and registrations that predate the attempt or belong to other extensions.
    Runtime imports and compatibility objects are outside this rollback.

    The caller must hold ``_activation_lock`` to serialize this backend's
    registration attempts. The rollback also runs for ``BaseException``
    subclasses, including interruption of an import.
    """

    _require_runtime()
    snapshot = _snapshot_registrations()
    package_name = _BACKEND_PACKAGE
    loaded_backend_modules = frozenset(
        name for name in sys.modules if name.startswith(f"{package_name}.")
    )
    try:
        planner_module = importlib.import_module(
            f"{package_name}._compiler._rewrite"
        )
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
    """Require one live registration for each mandatory compiler pass.

    Import success alone does not show that the compiler will run a pass:
    registration APIs may have changed or registered the same class twice.
    Check the class objects exported by the imported modules against the
    registries used by this activation attempt. Missing exports count as zero.

    Parameters
    ----------
    snapshot : _RegistrationSnapshot
        Snapshot holding references to the live registries being checked.
        Counts come from their current contents, not the saved baseline.
    planner_module : module
        Rewrite module exporting ``CoopWholeFunctionPlanner`` and
        ``CoopSinglePhaseRewrite``.
    group_planner_module : module
        Module exporting ``CoopGroupHierarchyPlanner``.
    compat : _NumbaMlirCompilerCompat or None, optional
        Compiler adapter used to inspect registrations. ``None`` loads the
        active runtime's cached adapter.

    Raises
    ------
    _NumbaMlirBackendImportError
        A required planner or before-inference rewrite is absent or duplicated.
        The error records all three counts for activation diagnostics.
    """

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
    registration_counts = _registration_counts(
        snapshot,
        expected_planners,
        (
            "CoopSinglePhaseRewrite",
            getattr(planner_module, "CoopSinglePhaseRewrite", None),
        ),
    )
    invalid = tuple(
        f"{name}={count}"
        for name, count in registration_counts.items()
        if count != 1
    )
    if invalid:
        raise _NumbaMlirBackendImportError(
            "registration-postcondition-failed",
            "cuda.coop.numba_mlir called the compiler registration APIs, but "
            "the required planner and rewrite hooks were not each registered "
            "exactly once: " + ", ".join(invalid),
            registration_counts=registration_counts,
        )


def _snapshot_registrations() -> _RegistrationSnapshot:
    """Capture registration baselines without copying the registry objects.

    Activation needs both the original entries and the live containers: the
    former identify additions to roll back, while the latter remain the
    compiler's authoritative registries. Capture planner entries under the
    planner lock and copy each rewrite bucket to a tuple. The two registries
    are sampled separately; this is not an atomic snapshot across both.

    Returns
    -------
    _RegistrationSnapshot
        References to the live planner and rewrite registries, plus saved
        sequences of their current class entries. Later appends do not
        change those sequences; duplicate entries retain their multiplicity.

    Raises
    ------
    _NumbaMlirBackendImportError
        A required registry cannot be inspected using the supported compiler
        interface. The original attribute or type error is retained as its cause.
    """

    from numba_cuda_mlir._whole_function_planners import _planner_registry
    from numba_cuda_mlir.numba_cuda.core.rewrites import rewrite_registry

    try:
        with _planner_registry._lock:
            planners = tuple(_planner_registry._planners)
        rewrites = {
            kind: tuple(rewrite_classes)
            for kind, rewrite_classes in (
                rewrite_registry.rewrites.copy().items()
            )
        }
        return _RegistrationSnapshot(
            planner_registry=_planner_registry,
            planners=planners,
            rewrite_registry=rewrite_registry,
            rewrites=rewrites,
        )
    except (AttributeError, TypeError) as exc:
        raise _NumbaMlirBackendImportError(
            "registration-transaction-unavailable",
            "cuda.coop.numba_mlir cannot transactionally register its "
            "planners and rewrite with the installed numba-cuda-mlir runtime.",
            cause=exc,
        ) from exc


def _registration_counts(
    snapshot: _RegistrationSnapshot,
    planners: tuple[tuple[str, type | None], ...],
    rewrite: tuple[str, type | None],
) -> dict[str, int]:
    """Count the expected pass classes in the snapshot's live registries.

    Activation uses these counts to detect both missing hooks and duplicate
    registration. The saved baseline entries do not affect the counts.

    Parameters
    ----------
    snapshot : _RegistrationSnapshot
        Identifies the registry objects to inspect at the time of this call.
    planners : tuple of (str, type or None)
        Diagnostic names paired with expected planner classes. ``None``
        represents a missing export and produces a count of zero.
    rewrite : tuple of (str, type or None)
        Diagnostic name and expected rewrite class. Only the
        ``"before-inference"`` bucket is checked; ``None`` counts as zero.

    Returns
    -------
    dict of str to int
        Occurrences of each supplied class, keyed by its diagnostic name.
        Planner counts are read while holding the planner registry lock.
    """

    with snapshot.planner_registry._lock:
        counts = {
            name: snapshot.planner_registry._planners.count(planner)
            if planner is not None
            else 0
            for name, planner in planners
        }
    rewrite_name, rewrite_type = rewrite
    counts[rewrite_name] = (
        snapshot.rewrite_registry.rewrites.get("before-inference", []).count(
            rewrite_type
        )
        if rewrite_type is not None
        else 0
    )
    return counts


def _restore_registrations(snapshot: _RegistrationSnapshot) -> None:
    """Remove only this backend's additions after a failed import.

    Other extensions may register with Numba-CUDA-MLIR while this backend is
    importing. Replacing either registry with its pre-import snapshot would
    erase those independent additions. Instead, retain the occurrences that
    existed in the snapshot and delete only new classes owned by the two
    activation modules. Registrations from every other module remain in
    place, including ones appended concurrently during rollback.

    Planner entries are filtered under the registry lock. Each current rewrite
    bucket is filtered in place, including buckets created after the snapshot;
    empty buckets may remain. Entries removed by other code are not reinserted.

    Parameters
    ----------
    snapshot : _RegistrationSnapshot
        Original registry objects and their pre-import entries. The live
        lists are edited in place; saved entries are used to distinguish
        preexisting occurrences from this attempt's additions.
    """

    with snapshot.planner_registry._lock:
        _remove_backend_additions(
            snapshot.planner_registry._planners,
            snapshot.planners,
            owned_modules=_REGISTRATION_MODULES,
        )

    registered_rewrites = snapshot.rewrite_registry.rewrites
    for kind, rewrite_classes in registered_rewrites.copy().items():
        _remove_backend_additions(
            rewrite_classes,
            snapshot.rewrites.get(kind, ()),
            owned_modules=_REGISTRATION_MODULES,
        )


def _remove_backend_additions(
    registrations: MutableSequence[type],
    baseline: tuple[type, ...],
    *,
    owned_modules: frozenset[str],
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
        elif getattr(registration, "__module__", None) in owned_modules:
            removal_indices.append(index)

    # Public registration APIs append to these lists. Removing by index from
    # the tail preserves any foreign append that races with this cleanup.
    for index in reversed(removal_indices):
        del registrations[index]


__all__: tuple[str, ...] = ()
