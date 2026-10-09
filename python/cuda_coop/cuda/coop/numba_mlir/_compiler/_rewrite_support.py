# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Share call-rewrite records, alignment rules, and compiler support helpers.

Match records carry resolved provider inputs; payload and storage records
retain facts needed between the whole-function scan and block replacement.
Storage requirements describe sizes and alignments before plans assign backing
offsets. ``_UNRESOLVED`` distinguishes failed static inference from an
explicit ``None`` value. ``_DeferredCoopRewrite`` asks for a retry once
launch metadata is available; it does not report an invalid user call.
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import count
from typing import TYPE_CHECKING, Any

from numba_cuda_mlir.numba_cuda.core.errors import ConstantInferenceError
from numba_cuda_mlir.numba_cuda.core.rewrites import (
    Rewrite as Rewrite,  # noqa: PLC0414 - Re-export for typing.
)

from cuda.coop._core import GroupLoweringPlan

from ._operations import FactoryOperation
from ._parameters import normalize_dtype_param

if TYPE_CHECKING:
    from numba_cuda_mlir.numba_cuda.core import ir
else:
    from numba_cuda_mlir.numbair_transforms import ir

_INFERENCE_EXCEPTIONS = (
    KeyError,
    ValueError,
    TypeError,
    AttributeError,
    ConstantInferenceError,
)
_GLOBAL_NAME_COUNTER = count()
_UNRESOLVED = object()
_MIN_TEMP_STORAGE_ALIGNMENT = max(1, struct.calcsize("P"))
_DEFAULT_STATIC_SHARED_MEMORY_BYTES = 48 * 1024
# numba-cuda-mlir declares its dynamic shared-memory window with 16-byte
# alignment; a larger request cannot be honored once the backing goes dynamic.
_DYNAMIC_SHARED_MEMORY_ALIGNMENT = 16


class CoopSinglePhaseRewriteError(Exception):
    """Report a matched cooperative call that cannot be rewritten."""


class _DeferredCoopRewrite(Exception):
    """Signal that a compiler pass must leave a cooperative call for later.

    ``CoopSinglePhaseRewrite.match`` catches this when launch-dependent work
    needs exact launch metadata and records the deferral on the rewrite.
    ``_CallRewriting._rewrite_calls`` reads that flag and retries kernel work
    with the metadata. Deferral leaves the affected IR unchanged; it is not an
    application error.
    """


def _next_global_name(stem: str) -> str:
    """Give an injected Python object a process-unique IR global name."""

    return f"__cuda_coop_numba_mlir_{stem}_{next(_GLOBAL_NAME_COUNTER)}__"


def _phi_incoming_values(definition):
    """Read the alternatives of a control-flow merge expression.

    Check the compiler IR shape at this boundary so unsupported phi forms
    produce a rewrite diagnostic instead of losing an incoming value.
    """

    if not hasattr(definition, "incoming_values"):
        raise CoopSinglePhaseRewriteError(
            "Unsupported Numba phi expression shape: missing incoming_values."
        )
    incoming_values = definition.incoming_values
    if not isinstance(incoming_values, (list, tuple)):
        raise CoopSinglePhaseRewriteError(
            "Unsupported Numba phi expression shape: incoming_values is not "
            "a sequence."
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
    """Round alignment up to a power of two, at least pointer size."""

    return max(
        _MIN_TEMP_STORAGE_ALIGNMENT, _next_power_of_two(required_alignment)
    )


def _normalize_temp_storage_alignment(
    alignment: int, *, context: str = "TempStorage alignment"
) -> int:
    """Validate power-of-two alignment and apply the pointer-size minimum.

    ``context`` names the setting in diagnostics. A smaller valid request is
    raised to the minimum needed by the generated storage pointer.
    """

    if alignment <= 0:
        raise CoopSinglePhaseRewriteError(
            f"{context} must be a positive integer."
        )
    if alignment & alignment - 1 != 0:
        raise CoopSinglePhaseRewriteError(f"{context} must be a power of 2.")
    return max(_MIN_TEMP_STORAGE_ALIGNMENT, alignment)


def _dtype_values_match(lhs, rhs) -> bool:
    """Compare dtype spellings after normalization when possible.

    If normalization fails, use the original or partially normalized values.
    Callers remain responsible for validating accepted dtypes.
    """

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
    """Read per-block shared-memory limits for the current CUDA device.

    Obtain the active context and initialize the driver before querying
    default and opt-in capacities. A nonpositive opt-in value falls back to
    the default limit. Driver failures raise ``RuntimeError``.
    """

    from numba_cuda_mlir.numba_cuda.cudadrv import devices

    import cuda.bindings.driver as _driver

    context = devices.get_context()
    (err,) = _driver.cuInit(0)
    _check_driver_error(err, "cuInit")
    err, device = _driver.cuDeviceGet(int(context.device.id))
    _check_driver_error(err, "cuDeviceGet")
    err, max_default = _driver.cuDeviceGetAttribute(
        _driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
        device,
    )
    _check_driver_error(
        err, "cuDeviceGetAttribute(MAX_SHARED_MEMORY_PER_BLOCK)"
    )
    err, max_optin = _driver.cuDeviceGetAttribute(
        _driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        device,
    )
    _check_driver_error(
        err, "cuDeviceGetAttribute(MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)"
    )
    if int(max_optin) <= 0:
        max_optin = max_default
    return {
        "device_id": int(context.device.id),
        "max_default_shared_memory_per_block": int(max_default),
        "max_optin_shared_memory_per_block": int(max_optin),
    }


@dataclass(frozen=True)
class _RewriteMatch:
    """Keep one validated provider call ready for compilation and emission.

    ``factory`` and ``factory_metadata`` identify the implementation and
    declared call contract. ``factory_kwargs`` holds the resolved Python
    values and binding descriptors passed to the host factory to specialize
    the provider. ``runtime_args`` keeps device operands without storage,
    which is tracked separately in ``runtime_temp_storage_var``.

    ``factory_kw_value_vars`` retains original IR variables consumed at
    compile time so their assignments can be removed if no rewritten block
    uses them. These are cleanup candidates, not another set of factory
    inputs. Inferred keywords need no source variable; lowering metadata and
    omitted controls can leave a cleanup candidate without a factory keyword.

    ``func_var_name`` and its optional extra alias identify callee assignments
    that may also become unused after replacement. ``family_metadata`` carries
    a hook's analysis, such as scalar boxing. ``lowering_plan`` keeps the shared
    group contract after its private keyword is removed from factory inputs.
    ``op_name`` and ``loc`` identify the operation and source site.
    """

    op_name: str
    factory: Callable[..., Any]
    factory_metadata: FactoryOperation
    func_var_name: str
    func_var_name_extra: str | None
    runtime_args: tuple[ir.Var, ...]
    runtime_temp_storage_var: ir.Var | None
    factory_kwargs: dict[str, object]
    factory_kw_value_vars: tuple[ir.Var, ...]
    loc: ir.Loc
    family_metadata: object = None
    lowering_plan: GroupLoweringPlan | None = None


@dataclass(frozen=True)
class _ResolvedCallTarget:
    """Separate callee recognition from argument validation.

    Keep the registered factory and contract, plus the callee names that
    replacement may remove. ``getitem_temp_storage`` records an operand from
    subscript syntax; later descriptor-use checks decide whether the call may
    use that syntax.
    """

    factory: Callable[..., Any]
    factory_metadata: FactoryOperation
    func_var_name: str
    func_var_name_extra: str | None
    getitem_temp_storage: ir.Var | None

    @property
    def operation(self) -> str:
        return self.factory_metadata.operation


@dataclass(frozen=True)
class _ThreadDataSpecification:
    """Carry known payload facts while ordinary typing is incomplete.

    ``items_per_thread`` and ``dtype`` may be unknown. Native local/shared
    array queries also use this record, so its presence alone does not
    establish public ``ThreadData`` origin. ``common_root`` records a common
    API constructor for later numeric validation. ``alignment`` is an optional
    byte alignment. Construction normalizes recognized dtypes and leaves
    unsupported values for later validation.
    """

    items_per_thread: int | None
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
class _TempStorageCtorSpecification:
    """Retain a descriptor's requested storage policy before layout.

    Capacity and alignment are in bytes and may be inferred later.
    ``auto_sync=None`` has the effective value false. ``sharing`` selects
    reuse among compatible calls or separate storage for every call.
    """

    size_in_bytes: int | None
    alignment: int | None
    auto_sync: bool | None
    sharing: str


@dataclass(frozen=True)
class _TempStorageUseRequirement:
    """Record one call's per-group scratch requirement before placement.

    Size and alignment are in bytes. ``call_assign`` retains the original IR
    identity for slice lookup; ``order`` is the whole-function scan order, not
    runtime execution order. ``lowering_plan`` supplies group instances and
    reuse rules when the shared planner produced the call. ``reservation``
    identifies a typed array that follows its descriptor's sharing policy.
    """

    call_assign: ir.Assign
    order: int
    size_in_bytes: int
    alignment: int
    lowering_plan: GroupLoweringPlan | None = None
    reservation: bool = False


@dataclass
class _TempStorageRequirementSummary:
    """Accumulate calls that share one descriptor or implicit region.

    The maxima summarize individual byte and alignment requirements. ``uses``
    retains each call so layout can account for alignment gaps, group
    instances, and whether storage can be reused.
    """

    max_size_in_bytes: int = 0
    max_alignment: int = 1
    uses: list[_TempStorageUseRequirement] = field(default_factory=list)


@dataclass(frozen=True)
class _TempStorageSlice:
    """Describe one call's view before the region's base offset is added.

    ``offset`` and ``size_in_bytes`` locate its bytes within the region.
    Multiple group ``instances`` are separated by ``stride`` bytes; a missing
    stride falls back to the call's size during emission. The lowering plan
    supplies the formula that selects the current instance.
    """

    offset: int
    size_in_bytes: int
    stride: int | None = None
    instances: int = 1
    lowering_plan: GroupLoweringPlan | None = None


@dataclass(frozen=True)
class _TempStoragePlan:
    """Place one explicit descriptor or the implicit scratch region.

    Capacity and alignment include all of the region's calls and group
    instances. ``slices_by_call_id`` maps IR assignment identities to their
    views. ``base_offset`` places the region within the global byte array;
    individual slice offsets remain relative to this region. ``sharing`` and
    ``auto_sync`` retain the reuse policy for call emission.
    """

    size_in_bytes: int
    alignment: int
    sharing: str
    auto_sync: bool
    slices_by_call_id: dict[int, _TempStorageSlice]
    base_offset: int = 0


@dataclass(frozen=True)
class _TempStorageGlobalPlan:
    """Describe the backing allocation shared by all scratch regions.

    ``total_size`` includes padding to ``max_alignment``; both are in bytes.
    ``uses_dynamic_smem`` selects dynamic placement, in which case
    ``dynamic_shared_bytes`` is the launch requirement. Static placement
    leaves that requirement at zero.
    """

    total_size: int
    max_alignment: int
    uses_dynamic_smem: bool
    dynamic_shared_bytes: int


__all__ = [
    "_DEFAULT_STATIC_SHARED_MEMORY_BYTES",
    "_DYNAMIC_SHARED_MEMORY_ALIGNMENT",
    "_GLOBAL_NAME_COUNTER",
    "_INFERENCE_EXCEPTIONS",
    "_MIN_TEMP_STORAGE_ALIGNMENT",
    "_UNRESOLVED",
    "CoopSinglePhaseRewriteError",
    "Rewrite",
    "_DeferredCoopRewrite",
    "_ResolvedCallTarget",
    "_RewriteMatch",
    "_TempStorageCtorSpecification",
    "_TempStorageGlobalPlan",
    "_TempStoragePlan",
    "_TempStorageRequirementSummary",
    "_TempStorageSlice",
    "_TempStorageUseRequirement",
    "_ThreadDataSpecification",
    "_align_up",
    "_default_temp_storage_alignment",
    "_dtype_values_match",
    "_next_global_name",
    "_normalize_temp_storage_alignment",
    "_phi_incoming_values",
    "_query_device_shared_memory_limits",
    "_validate_temp_storage_alignment",
    "ir",
]
