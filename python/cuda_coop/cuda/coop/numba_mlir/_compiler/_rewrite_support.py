# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# The family rewrites import this module's private support names explicitly.
# ruff: noqa: F401

"""Shared single-phase provider rewriting for Numba-CUDA-MLIR.

This compiler layer preserves launch inference, device-function deferral,
payload provenance, temporary-storage planning, and invocable coalescing.  It
dispatches to semantic provider modules through exact callable identities.
"""

from __future__ import annotations

import hashlib
import operator
import struct
from dataclasses import dataclass, field, replace
from itertools import count

import numpy as np
from numba_cuda_mlir import cuda as _cuda_module
from numba_cuda_mlir.extending import (
    WholeFunctionPlanner,
    register_planner,
    require_launch_config,
    set_required_dynamic_shared_memory,
)
from numba_cuda_mlir.numba_cuda.core import errors as _numba_errors
from numba_cuda_mlir.numba_cuda.core.rewrites import Rewrite, register_rewrite
from numba_cuda_mlir.numba_cuda.typing.typeof import typeof as _numba_typeof
from numba_cuda_mlir.numbair_transforms import ir

from cuda.coop._core import api as _portable_api

from .. import _lowering  # noqa: F401 - registers factories
from .._types import (
    _hash_symbol_value,
    algo_coalesce_key,
    collect_specializations,
    make_invocable_from_specialization,
    prepare_ltoir_bundle,
)
from ._operations import FactoryOperation, factory_operation
from ._parameters import normalize_dim_param, normalize_dtype_param

_INFERENCE_EXCEPTIONS = (
    KeyError,
    ValueError,
    TypeError,
    AttributeError,
    _numba_errors.ConstantInferenceError,
)
_GLOBAL_NAME_COUNTER = count()
_UNRESOLVED = object()
_MIN_TEMP_STORAGE_ALIGNMENT = max(1, struct.calcsize("P"))
_DEFAULT_STATIC_SHARED_MEMORY_BYTES = 48 * 1024


class CoopSinglePhaseRewriteError(Exception):
    """Raised when a matched one-shot coop call cannot be rewritten."""


class _DeferredCoopRewrite(Exception):
    """Leave launch-dependent cooperative IR for whole-function planning."""


def _next_global_name(stem: str) -> str:
    return f"__cuda_coop_numba_mlir_{stem}_{next(_GLOBAL_NAME_COUNTER)}__"


def _phi_incoming_values(definition):
    if not hasattr(definition, "incoming_values"):
        raise CoopSinglePhaseRewriteError(
            "Unsupported Numba phi expression shape: missing incoming_values."
        )
    incoming_values = definition.incoming_values
    if not isinstance(incoming_values, (list, tuple)):
        raise CoopSinglePhaseRewriteError(
            "Unsupported Numba phi expression shape: incoming_values is not a sequence."
        )
    return tuple(incoming_values)


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    return (value + alignment - 1) // alignment * alignment


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _default_temp_storage_alignment(required_alignment: int) -> int:
    return max(_MIN_TEMP_STORAGE_ALIGNMENT, _next_power_of_two(required_alignment))


def _normalize_temp_storage_alignment(
    alignment: int, *, context: str = "TempStorage alignment"
) -> int:
    if alignment <= 0:
        raise CoopSinglePhaseRewriteError(f"{context} must be a positive integer.")
    if alignment & alignment - 1 != 0:
        raise CoopSinglePhaseRewriteError(f"{context} must be a power of 2.")
    return max(_MIN_TEMP_STORAGE_ALIGNMENT, alignment)


def _dtype_values_match(lhs, rhs) -> bool:
    try:
        lhs = normalize_dtype_param(lhs)
        rhs = normalize_dtype_param(rhs)
    except (TypeError, ValueError, AttributeError):
        pass
    return lhs == rhs


def _validate_temp_storage_alignment(
    alignment: int, *, context: str = "TempStorage alignment"
) -> None:
    alignment = _normalize_temp_storage_alignment(alignment, context=context)
    if alignment % _MIN_TEMP_STORAGE_ALIGNMENT != 0:
        raise CoopSinglePhaseRewriteError(
            f"{context} must be a multiple of {_MIN_TEMP_STORAGE_ALIGNMENT}."
        )


def _check_driver_error(err, op: str) -> None:
    if err.value != 0:
        raise RuntimeError(f"{op} failed with CUDA driver error {err}")


def _query_device_shared_memory_limits() -> dict[str, int]:
    from numba_cuda_mlir.numba_cuda.cudadrv import devices

    from cuda.bindings import driver

    context = devices.get_context()
    (err,) = driver.cuInit(0)
    _check_driver_error(err, "cuInit")
    err, device = driver.cuDeviceGet(int(context.device.id))
    _check_driver_error(err, "cuDeviceGet")
    err, max_default = driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
        device,
    )
    _check_driver_error(err, "cuDeviceGetAttribute(MAX_SHARED_MEMORY_PER_BLOCK)")
    err, max_optin = driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        device,
    )
    _check_driver_error(err, "cuDeviceGetAttribute(MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)")
    if int(max_optin) <= 0:
        max_optin = max_default
    return {
        "device_id": int(context.device.id),
        "max_default_shared_memory_per_block": int(max_default),
        "max_optin_shared_memory_per_block": int(max_optin),
    }


@dataclass(frozen=True)
class _RewriteMatch:
    op_name: str
    factory: object
    factory_metadata: FactoryOperation
    func_var_name: str
    func_var_name_extra: str | None
    runtime_args: tuple[ir.Var, ...]
    runtime_temp_storage_var: ir.Var | None
    factory_kwargs: dict[str, object]
    factory_kw_value_vars: tuple[ir.Var, ...]
    loc: ir.Loc
    family_metadata: object = None


@dataclass(frozen=True)
class _ResolvedCallTarget:
    factory: object
    factory_metadata: FactoryOperation
    func_var_name: str
    func_var_name_extra: str | None
    getitem_temp_storage: ir.Var | None

    @property
    def operation(self) -> str:
        return self.factory_metadata.operation


@dataclass(frozen=True)
class _ThreadDataSpec:
    items_per_thread: object | None
    dtype: object | None
    common_root: bool = False
    alignment: int | None = None

    def __post_init__(self) -> None:
        if self.dtype is None:
            return
        try:
            canonical = normalize_dtype_param(self.dtype)
        except (TypeError, ValueError):
            return
        object.__setattr__(self, "dtype", canonical)


@dataclass(frozen=True)
class _TempStorageCtorSpec:
    size_in_bytes: int | None
    alignment: int | None
    auto_sync: bool | None
    sharing: str


@dataclass(frozen=True)
class _TempStorageUseRequirement:
    call_assign: ir.Assign
    order: int
    size_in_bytes: int
    alignment: int


@dataclass
class _TempStorageRequirementSummary:
    max_size_in_bytes: int = 0
    max_alignment: int = 1
    uses: list[_TempStorageUseRequirement] = field(default_factory=list)


@dataclass(frozen=True)
class _TempStorageSlice:
    offset: int
    size_in_bytes: int


@dataclass(frozen=True)
class _TempStoragePlan:
    size_in_bytes: int
    alignment: int
    sharing: str
    auto_sync: bool
    slices_by_call_id: dict[int, _TempStorageSlice]
    base_offset: int = 0


@dataclass(frozen=True)
class _TempStorageGlobalPlan:
    total_size: int
    max_alignment: int
    uses_dynamic_smem: bool
    dynamic_shared_bytes: int
    max_default_smem: int
    max_optin_smem: int


# Support consumers import the private names they use explicitly.
