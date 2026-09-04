# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compatibility boundary for private Numba-CUDA-MLIR 0.5 APIs.

The backend otherwise imports documented, top-level Numba-CUDA-MLIR APIs
directly.  Keep every dependency on the runtime's private registries and its
vendored Numba implementation in this module so a future runtime API can
replace this shim as one unit.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import re
from collections.abc import MutableMapping, MutableSequence
from dataclasses import dataclass
from typing import Any

_MINIMUM_RUNTIME_VERSION = "0.5.0"
_MAXIMUM_RUNTIME_VERSION = "0.6"
_SUPPORTED_RUNTIME_SERIES_PREFIX = _MINIMUM_RUNTIME_VERSION.rsplit(".", 1)[0]
_SUPPORTED_RUNTIME_SERIES = f"{_SUPPORTED_RUNTIME_SERIES_PREFIX}.x"
_SUPPORTED_VERSION = re.compile(
    rf"^{re.escape(_SUPPORTED_RUNTIME_SERIES_PREFIX)}(?:\.|$)"
)
_RUNTIME_INSTALL_HINT = (
    "'cuda-coop[numba-cuda-mlir-cu12]' for CUDA 12 or "
    "'cuda-coop[numba-cuda-mlir-cu13]' for CUDA 13"
)


class _NumbaMlirBackendImportError(ImportError):
    """Structured qualified-backend import failure."""

    def __init__(self, reason_code, message, *, cause=None, **details):
        super().__init__(message)
        self.backend = "numba-cuda-mlir"
        self.reason_code = reason_code
        self.details = details
        if cause is not None:
            self.__cause__ = cause


@dataclass(frozen=True)
class _RegistrationSnapshot:
    planner_registry: Any
    planners: tuple[type, ...]
    rewrite_registry: Any
    rewrites: dict[str, tuple[type, ...]]


@dataclass(frozen=True)
class _NumbaMlirCompilerCompat:
    """Validated private compiler surface for Numba-CUDA-MLIR 0.5.x."""

    version: str
    planner_registry: Any
    rewrite_registry: Any
    overload_function_template: type
    make_overload_template: Any
    numba_errors: Any
    numba_typeof: Any
    numba_ir: Any
    rewrite_type: type
    register_rewrite: Any

    def snapshot_registrations(self) -> _RegistrationSnapshot:
        """Snapshot the registries populated during backend activation."""

        with self.planner_registry._lock:
            planners = tuple(self.planner_registry._planners)
        rewrites = {
            kind: tuple(rewrite_classes)
            for kind, rewrite_classes in self.rewrite_registry.rewrites.copy().items()
        }
        return _RegistrationSnapshot(
            planner_registry=self.planner_registry,
            planners=planners,
            rewrite_registry=self.rewrite_registry,
            rewrites=rewrites,
        )

    @staticmethod
    def registration_counts(
        snapshot: _RegistrationSnapshot,
        planners: tuple[tuple[str, type | None], ...],
        rewrite: tuple[str, type | None],
    ) -> dict[str, int]:
        """Count expected registrations without exposing registry internals."""

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

    @staticmethod
    def restore_registrations(
        snapshot: _RegistrationSnapshot,
        *,
        owned_modules: frozenset[str],
    ) -> None:
        """Remove only backend-owned additions made after ``snapshot``."""

        with snapshot.planner_registry._lock:
            _remove_backend_additions(
                snapshot.planner_registry._planners,
                snapshot.planners,
                owned_modules=owned_modules,
            )

        registered_rewrites = snapshot.rewrite_registry.rewrites
        for kind, rewrite_classes in registered_rewrites.copy().items():
            _remove_backend_additions(
                rewrite_classes,
                snapshot.rewrites.get(kind, ()),
                owned_modules=owned_modules,
            )


@dataclass(frozen=True)
class _NumbaMlirDatamodelCompat:
    """Private datamodel APIs used for exact storage ABI inspection."""

    mlir_ir: Any
    default_manager: Any
    cuda_struct_model: type
    cuda_data_manager: Any


def _detected_version(runtime: Any) -> str | None:
    version = getattr(runtime, "__version__", None)
    if isinstance(version, str) and version:
        return version
    try:
        return importlib.metadata.version("numba-cuda-mlir")
    except importlib.metadata.PackageNotFoundError:
        return None


def _is_supported_runtime_version(version: str | None) -> bool:
    """Return whether ``version`` is covered by the private compatibility shim."""

    return (
        isinstance(version, str)
        and bool(version)
        and _SUPPORTED_VERSION.match(version) is not None
    )


def _runtime_requirement(runtime: Any = None) -> str:
    """Describe the supported runtime series and the detected installation."""

    version = _detected_version(runtime)
    detected = "no Numba-CUDA-MLIR distribution was detected"
    if version is not None:
        detected = f"detected numba-cuda-mlir=={version}"
    return (
        f"{detected}; the supported series is numba-cuda-mlir>="
        f"{_MINIMUM_RUNTIME_VERSION},<{_MAXIMUM_RUNTIME_VERSION}. Install "
        f"{_RUNTIME_INSTALL_HINT}."
    )


def _import_compat_module(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise _NumbaMlirBackendImportError(
            "runtime-hook-api-import-failed",
            "cuda.coop.numba_mlir could not import its required "
            f"Numba-CUDA-MLIR 0.5 compatibility API at {module_name!r}.",
            cause=exc,
            missing=getattr(exc, "name", None),
            module=module_name,
        ) from exc


def _load_numba_mlir_compat(runtime: Any) -> _NumbaMlirCompilerCompat:
    """Load and validate the private compiler shape supported by this shim."""

    version = _detected_version(runtime)
    if not _is_supported_runtime_version(version):
        detected = "an unknown version" if version is None else f"version {version}"
        raise _NumbaMlirBackendImportError(
            "unsupported-runtime-version",
            "cuda.coop.numba_mlir supports numba-cuda-mlir "
            f"{_SUPPORTED_RUNTIME_SERIES}, but "
            f"{detected} was detected.",
            detected_version=version,
            supported_series=_SUPPORTED_RUNTIME_SERIES,
        )

    extending = _import_compat_module("numba_cuda_mlir.extending")
    planner_module = _import_compat_module("numba_cuda_mlir._whole_function_planners")
    rewrites = _import_compat_module("numba_cuda_mlir.numba_cuda.core.rewrites")
    errors = _import_compat_module("numba_cuda_mlir.numba_cuda.core.errors")
    typeof_module = _import_compat_module("numba_cuda_mlir.numba_cuda.typing.typeof")
    templates = _import_compat_module("numba_cuda_mlir.numba_cuda.typing.templates")
    transforms = _import_compat_module("numba_cuda_mlir.numbair_transforms")

    callable_requirements = {
        "numba_cuda_mlir.extending": (
            "WholeFunctionPlanner",
            "register_planner",
            "require_launch_config",
            "set_required_dynamic_shared_memory",
        ),
        "numba_cuda_mlir.numba_cuda.core.rewrites": (
            "Rewrite",
            "register_rewrite",
        ),
        "numba_cuda_mlir.numba_cuda.typing.typeof": ("typeof",),
        "numba_cuda_mlir.numba_cuda.typing.templates": ("make_overload_template",),
    }
    modules = {
        "numba_cuda_mlir.extending": extending,
        "numba_cuda_mlir.numba_cuda.core.rewrites": rewrites,
        "numba_cuda_mlir.numba_cuda.typing.typeof": typeof_module,
        "numba_cuda_mlir.numba_cuda.typing.templates": templates,
    }
    missing = [
        f"{module_name}.{name}"
        for module_name, names in callable_requirements.items()
        for name in names
        if not callable(getattr(modules[module_name], name, None))
    ]

    overload_template = getattr(
        extending,
        "_NumbaCudaMlirOverloadFunctionTemplate",
        None,
    )
    overload_base = getattr(templates, "_OverloadFunctionTemplate", None)
    overload_template_valid = (
        isinstance(overload_template, type)
        and isinstance(overload_base, type)
        and issubclass(overload_template, overload_base)
        and callable(getattr(overload_template, "_get_jit_decorator", None))
    )
    if overload_template_valid:
        try:
            get_jit_parameters = tuple(
                inspect.signature(overload_template._get_jit_decorator).parameters
            )
        except (TypeError, ValueError):
            get_jit_parameters = ()
        overload_template_valid = get_jit_parameters == ("self",)
    if not overload_template_valid:
        missing.append(
            "numba_cuda_mlir.extending._NumbaCudaMlirOverloadFunctionTemplate"
        )
    make_overload_template = getattr(templates, "make_overload_template", None)
    if callable(make_overload_template):
        try:
            parameters = inspect.signature(make_overload_template).parameters
        except (TypeError, ValueError):
            parameters = {}
        expected_parameters = (
            "func",
            "overload_func",
            "jit_options",
            "strict",
            "inline",
            "prefer_literal",
            "base",
        )
        positional = {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
        keyword = {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
        ordered_parameters = tuple(parameters.values())
        valid_signature = (
            tuple(parameters)[: len(expected_parameters)] == expected_parameters
            and all(
                parameter.kind in positional for parameter in ordered_parameters[:5]
            )
            and all(parameter.kind in keyword for parameter in ordered_parameters[5:7])
        )
        if not valid_signature:
            missing.append(
                "numba_cuda_mlir.numba_cuda.typing.templates."
                "make_overload_template signature"
            )
    constant_inference_error = getattr(errors, "ConstantInferenceError", None)
    if not isinstance(constant_inference_error, type) or not issubclass(
        constant_inference_error, Exception
    ):
        missing.append("numba_cuda_mlir.numba_cuda.core.errors.ConstantInferenceError")
    numba_ir = getattr(transforms, "ir", None)
    if numba_ir is None:
        missing.append("numba_cuda_mlir.numbair_transforms.ir")
    if missing:
        missing_capabilities = tuple(missing)
        raise _NumbaMlirBackendImportError(
            "incomplete-runtime-hook-api",
            "cuda.coop.numba_mlir requires Numba-CUDA-MLIR 0.5 compiler "
            "capabilities that are missing or malformed: "
            + ", ".join(missing_capabilities),
            missing_capabilities=missing_capabilities,
        )

    planner_registry = getattr(planner_module, "_planner_registry", None)
    planner_lock = getattr(planner_registry, "_lock", None)
    planners = getattr(planner_registry, "_planners", None)
    rewrite_registry = getattr(rewrites, "rewrite_registry", None)
    rewrite_lists = getattr(rewrite_registry, "rewrites", None)
    valid_rewrite_lists = isinstance(rewrite_lists, MutableMapping) and all(
        isinstance(value, MutableSequence) for value in rewrite_lists.values()
    )
    if (
        not hasattr(planner_lock, "__enter__")
        or not hasattr(planner_lock, "__exit__")
        or not isinstance(planners, MutableSequence)
        or not valid_rewrite_lists
        or not isinstance(rewrite_lists.get("before-inference"), MutableSequence)
    ):
        raise _NumbaMlirBackendImportError(
            "registration-transaction-unavailable",
            "cuda.coop.numba_mlir cannot transactionally register its "
            "compiler hooks with the installed numba-cuda-mlir 0.5 runtime.",
        )

    return _NumbaMlirCompilerCompat(
        version=version,
        planner_registry=planner_registry,
        rewrite_registry=rewrite_registry,
        overload_function_template=overload_template,
        make_overload_template=make_overload_template,
        numba_errors=errors,
        numba_typeof=typeof_module.typeof,
        numba_ir=numba_ir,
        rewrite_type=rewrites.Rewrite,
        register_rewrite=rewrites.register_rewrite,
    )


_compiler_compat: _NumbaMlirCompilerCompat | None = None


def _get_numba_mlir_compat() -> _NumbaMlirCompilerCompat:
    global _compiler_compat
    if _compiler_compat is None:
        runtime = _import_compat_module("numba_cuda_mlir")
        _compiler_compat = _load_numba_mlir_compat(runtime)
    return _compiler_compat


_datamodel_compat: _NumbaMlirDatamodelCompat | None = None


def _get_numba_mlir_datamodel_compat() -> _NumbaMlirDatamodelCompat:
    global _datamodel_compat
    if _datamodel_compat is None:
        mlir_module = _import_compat_module("numba_cuda_mlir._mlir")
        datamodel = _import_compat_module("numba_cuda_mlir.numba_cuda.datamodel")
        datamodel_models = _import_compat_module(
            "numba_cuda_mlir.numba_cuda.datamodel.models"
        )
        cuda_models = _import_compat_module("numba_cuda_mlir.numba_cuda.models")
        _datamodel_compat = _NumbaMlirDatamodelCompat(
            mlir_ir=mlir_module.ir,
            default_manager=datamodel.default_manager,
            cuda_struct_model=datamodel_models.StructModel,
            cuda_data_manager=cuda_models.cuda_data_manager,
        )
    return _datamodel_compat


def _get_numba_mlir_devices() -> Any:
    return _import_compat_module("numba_cuda_mlir.numba_cuda.cudadrv.devices")


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
