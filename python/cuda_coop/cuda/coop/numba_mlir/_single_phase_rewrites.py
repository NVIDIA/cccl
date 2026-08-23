# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import hashlib
import operator
import struct
from dataclasses import dataclass, field, replace
from itertools import count
from numbers import Integral

import numpy as np
from numba_cuda_mlir import cuda as _cuda_module
from numba_cuda_mlir.numba_cuda.core import errors as _numba_errors
from numba_cuda_mlir.numba_cuda.core.rewrites import Rewrite, register_rewrite
from numba_cuda_mlir.numba_cuda.typing.typeof import typeof as _numba_typeof
from numba_cuda_mlir.numbair_transforms import ir

from cuda.coop._core import root_api as _common_root_api
from cuda.coop._core.block import (
    normalize_positive_int,
    normalize_radix_order,
    resolve_static_radix_end_bit,
)

from ._common import normalize_dim_param, normalize_dtype_param
from ._types import (
    _hash_symbol_value,
    algo_coalesce_key,
    collect_specializations,
    make_invocable_from_specialization,
    prepare_ltoir_bundle,
)


class CoopSinglePhaseRewriteError(Exception):
    """Raised when a matched one-shot coop call cannot be rewritten."""


class _DeferredCoopRewrite(Exception):
    """Internal signal to leave device-function coop IR for caller rewriting."""


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
    return ((value + alignment - 1) // alignment) * alignment


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
    if (alignment & (alignment - 1)) != 0:
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
    if (alignment % _MIN_TEMP_STORAGE_ALIGNMENT) != 0:
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
    func_var_name: str
    func_var_name_extra: str | None
    runtime_args: tuple[ir.Var, ...]
    runtime_temp_storage_var: ir.Var | None
    factory_kwargs: dict[str, object]
    factory_kw_value_vars: tuple[ir.Var, ...]
    loc: ir.Loc
    runtime_arg_constant_replacements: tuple[tuple[int, object], ...] = ()
    physical_warp_tile_origin: bool = False
    preserve_root_store_payload: bool = False
    root_store_scalar: bool = False


@dataclass(frozen=True)
class _ResolvedCallTarget:
    factory: object
    func_var_name: str
    func_var_name_extra: str | None
    getitem_temp_storage: ir.Var | None


@dataclass(frozen=True)
class _DirectInvocableTempStorageCall:
    invocable: object
    temp_storage_var: ir.Var
    rewritten_kws: tuple[tuple[str, ir.Var], ...]


@dataclass(frozen=True)
class _ThreadDataSpec:
    items_per_thread: object | None
    dtype: object | None
    common_v1: bool = False

    def __post_init__(self) -> None:
        # Constructor kwargs arrive in user spellings (np.int32, "int32",
        # int) while typing inference records Numba types. Canonicalize once
        # here so spec equality means type equality at every comparison site.
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


@dataclass(frozen=True)
class _ParentCtorSpec:
    parent_kind: str
    factory_kwargs_items: tuple[tuple[str, object], ...]
    captured_var_items: tuple[tuple[str, str], ...]

    def factory_kwargs(self) -> dict[str, object]:
        return dict(self.factory_kwargs_items)

    def captured_vars(self) -> dict[str, str]:
        return dict(self.captured_var_items)


@register_rewrite("before-inference")
class CoopSinglePhaseRewrite(Rewrite):
    """Rewrite narrow one-shot coop._block calls into two-phase invocable calls."""

    _CUDA_ROOT_MODULES = frozenset(
        {
            "cuda",
            "numba_cuda_mlir.cuda",
        }
    )

    _TEMP_STORAGE_RUNTIME_KW_OPS = frozenset(
        {
            "adjacent_difference",
            "shuffle",
            "discontinuity",
            "radix_rank",
            "exchange",
            "load",
            "store",
            "scan",
            "exclusive_sum",
            "inclusive_sum",
            "exclusive_scan",
            "inclusive_scan",
            "reduce",
            "sum",
            "merge_sort_keys",
            "merge_sort_pairs",
            "radix_sort_keys",
            "radix_sort_keys_descending",
            "radix_sort_pairs",
            "radix_sort_pairs_descending",
            "topk_max_keys",
            "topk_min_keys",
            "topk_max_pairs",
            "topk_min_pairs",
            "warp_load",
            "warp_store",
            "warp_exchange",
            "warp_reduce",
            "warp_sum",
            "warp_max",
            "warp_min",
            "warp_exclusive_sum",
            "warp_inclusive_sum",
            "warp_exclusive_scan",
            "warp_inclusive_scan",
            "warp_merge_sort_keys",
            "warp_merge_sort_pairs",
            "_common_merge_sort_keys",
            "_common_merge_sort_pairs",
            "_common_warp_merge_sort_keys",
            "_common_warp_merge_sort_pairs",
            "_common_radix_sort_keys",
            "_common_radix_sort_pairs",
            "_common_topk_max_keys",
            "_common_topk_min_keys",
            "_common_topk_max_pairs",
            "_common_topk_min_pairs",
            "_qualified_group_topk_max_keys",
            "_qualified_group_topk_min_keys",
            "_qualified_group_topk_max_pairs",
            "_qualified_group_topk_min_pairs",
            "_common_radix_rank",
        }
    )

    _OP_SPECS = {
        "group_reduce": {
            "namespace": "group",
            "runtime_arg_counts": {1},
            "allowed_factory_kwargs": {
                "dtype",
                "group",
                "binary_op",
                "items_per_thread",
                "broadcast",
                "methods",
            },
            "required_factory_kwargs": {"dtype", "group"},
        },
        "block_reduce_builtin": {
            "namespace": "block",
            "runtime_arg_counts": {1, 2},
            "runtime_factory_kwargs": ("num_valid",),
            "allowed_factory_kwargs": {
                "dtype",
                "threads_per_block",
                "binary_op",
                "items_per_thread",
                "algorithm",
                "num_valid",
            },
            "required_factory_kwargs": {
                "dtype",
                "threads_per_block",
                "binary_op",
            },
        },
        "warp_reduce_builtin": {
            "namespace": "warp",
            "runtime_arg_counts": {1, 2},
            "runtime_factory_kwargs": ("valid_items",),
            "allowed_factory_kwargs": {
                "dtype",
                "binary_op",
                "threads_in_warp",
                "threads_per_block",
                "valid_items",
            },
            "required_factory_kwargs": {"dtype", "binary_op"},
        },
        "load": {
            "namespace": "block",
            "runtime_arg_counts": {2, 3, 4},
            "runtime_factory_kwargs": ("num_valid_items", "oob_default"),
            "runtime_factory_kw_prerequisites": {"oob_default": "num_valid_items"},
            "allowed_factory_kwargs": {
                "dtype",
                "threads_per_block",
                "items_per_thread",
                "algorithm",
                "num_valid_items",
                "oob_default",
                "offset",
                "_common_profile_operation",
            },
            "required_factory_kwargs": {"dtype", "threads_per_block"},
        },
        "store": {
            "namespace": "block",
            "runtime_arg_counts": {2, 3},
            "runtime_factory_kwargs": ("num_valid_items",),
            "allowed_factory_kwargs": {
                "dtype",
                "threads_per_block",
                "items_per_thread",
                "algorithm",
                "num_valid_items",
                "offset",
                "_group_root_store",
                "_common_profile_operation",
            },
            "required_factory_kwargs": {"dtype", "threads_per_block"},
        },
        "scan": {
            "namespace": "block",
            "runtime_arg_counts": {2, 3},
            "allowed_factory_kwargs": {
                "dtype",
                "threads_per_block",
                "items_per_thread",
                "mode",
                "scan_op",
                "initial_value",
                "block_prefix_callback_op",
                "prefix_op",
                "block_aggregate",
                "algorithm",
                "methods",
            },
            "required_factory_kwargs": {"dtype", "threads_per_block"},
        },
        "exclusive_sum": {
            "namespace": "block",
            "runtime_arg_counts": {2, 3},
            "allowed_factory_kwargs": {
                "dtype",
                "threads_per_block",
                "items_per_thread",
                "block_prefix_callback_op",
                "prefix_op",
                "algorithm",
                "methods",
            },
            "required_factory_kwargs": {"dtype", "threads_per_block"},
        },
        "inclusive_sum": {
            "namespace": "block",
            "runtime_arg_counts": {2, 3},
            "allowed_factory_kwargs": {
                "dtype",
                "threads_per_block",
                "items_per_thread",
                "block_prefix_callback_op",
                "prefix_op",
                "algorithm",
                "methods",
            },
            "required_factory_kwargs": {"dtype", "threads_per_block"},
        },
        "exclusive_scan": {
            "namespace": "block",
            "runtime_arg_counts": {2, 3},
            "allowed_factory_kwargs": {
                "dtype",
                "threads_per_block",
                "items_per_thread",
                "scan_op",
                "initial_value",
                "block_prefix_callback_op",
                "prefix_op",
                "algorithm",
                "methods",
            },
            "required_factory_kwargs": {"dtype", "threads_per_block"},
        },
        "inclusive_scan": {
            "namespace": "block",
            "runtime_arg_counts": {2, 3},
            "allowed_factory_kwargs": {
                "dtype",
                "threads_per_block",
                "items_per_thread",
                "scan_op",
                "initial_value",
                "block_prefix_callback_op",
                "prefix_op",
                "algorithm",
                "methods",
            },
            "required_factory_kwargs": {"dtype", "threads_per_block"},
        },
        "reduce": {
            "namespace": "block",
            "runtime_arg_counts": {1, 2},
            "runtime_factory_kwargs": ("num_valid",),
            "allowed_factory_kwargs": {
                "dtype",
                "threads_per_block",
                "binary_op",
                "items_per_thread",
                "algorithm",
                "num_valid",
                "methods",
            },
            "required_factory_kwargs": {"dtype", "threads_per_block", "binary_op"},
        },
        "sum": {
            "namespace": "block",
            "runtime_arg_counts": {1, 2},
            "runtime_factory_kwargs": ("num_valid",),
            "allowed_factory_kwargs": {
                "dtype",
                "threads_per_block",
                "items_per_thread",
                "algorithm",
                "num_valid",
                "methods",
            },
            "required_factory_kwargs": {"dtype", "threads_per_block"},
        },
        "adjacent_difference": {
            "namespace": "block",
            "runtime_arg_counts": {2, 3, 4},
            "allowed_factory_kwargs": {
                "block_adjacent_difference_type",
                "dtype",
                "threads_per_block",
                "items_per_thread",
                "difference_op",
                "methods",
                "valid_items",
                "tile_predecessor_item",
                "tile_successor_item",
            },
            "required_factory_kwargs": {
                "dtype",
                "threads_per_block",
                "difference_op",
            },
        },
        "shuffle": {
            "namespace": "block",
            "runtime_arg_counts": {1, 2, 3},
            "allowed_factory_kwargs": {
                "block_shuffle_type",
                "dtype",
                "threads_per_block",
                "items_per_thread",
                "distance",
                "block_prefix",
                "block_suffix",
                "methods",
            },
            "required_factory_kwargs": {"dtype", "threads_per_block"},
        },
        "discontinuity": {
            "namespace": "block",
            "runtime_arg_counts": {2, 3, 4, 5},
            "allowed_factory_kwargs": {
                "dtype",
                "threads_per_block",
                "items_per_thread",
                "flag_op",
                "flag_dtype",
                "block_discontinuity_type",
                "methods",
                "tile_predecessor_item",
                "tile_successor_item",
            },
            "required_factory_kwargs": {"dtype", "threads_per_block", "flag_op"},
        },
        "radix_rank": {
            "namespace": "block",
            "runtime_arg_counts": {2, 3},
            "runtime_factory_kwargs": ("exclusive_digit_prefix",),
            "allowed_factory_kwargs": {
                "dtype",
                "threads_per_block",
                "items_per_thread",
                "begin_bit",
                "end_bit",
                "descending",
                "exclusive_digit_prefix",
            },
            "required_factory_kwargs": {
                "dtype",
                "threads_per_block",
                "begin_bit",
                "end_bit",
            },
        },
        "exchange": {
            "namespace": "block",
            "runtime_arg_counts": {1, 2, 3, 4},
            "allowed_factory_kwargs": {
                "block_exchange_type",
                "dtype",
                "threads_per_block",
                "items_per_thread",
                "warp_time_slicing",
                "offset_dtype",
                "valid_flag_dtype",
                "use_output_items",
                "methods",
                "_common_profile_operation",
            },
            "required_factory_kwargs": {"dtype", "threads_per_block"},
        },
        "merge_sort_keys": {
            "namespace": "block",
            "runtime_arg_counts": {1, 3},
            "runtime_factory_kwargs": ("valid_items", "oob_default"),
            "runtime_factory_kw_prerequisites": {"oob_default": "valid_items"},
            "allowed_factory_kwargs": {
                "dtype",
                "threads_per_block",
                "items_per_thread",
                "compare_op",
                "valid_items",
                "oob_default",
                "methods",
            },
            "required_factory_kwargs": {"dtype", "threads_per_block", "compare_op"},
        },
        "merge_sort_pairs": {
            "namespace": "block",
            "runtime_arg_counts": {2, 4},
            "runtime_factory_kwargs": ("valid_items", "oob_default"),
            "runtime_factory_kw_prerequisites": {"oob_default": "valid_items"},
            "allowed_factory_kwargs": {
                "keys",
                "values",
                "threads_per_block",
                "items_per_thread",
                "compare_op",
                "valid_items",
                "oob_default",
                "methods",
            },
            "required_factory_kwargs": {
                "keys",
                "values",
                "threads_per_block",
                "compare_op",
            },
        },
        "radix_sort_keys": {
            "namespace": "block",
            "runtime_arg_counts": {1, 3},
            "runtime_only_kwargs": ("begin_bit", "end_bit"),
            "runtime_factory_kw_prerequisites": {
                "begin_bit": "end_bit",
                "end_bit": "begin_bit",
            },
            "allowed_factory_kwargs": {
                "dtype",
                "threads_per_block",
                "items_per_thread",
                "blocked_to_striped",
            },
            "required_factory_kwargs": {"dtype", "threads_per_block"},
        },
        "radix_sort_keys_descending": {
            "namespace": "block",
            "runtime_arg_counts": {1, 3},
            "runtime_only_kwargs": ("begin_bit", "end_bit"),
            "runtime_factory_kw_prerequisites": {
                "begin_bit": "end_bit",
                "end_bit": "begin_bit",
            },
            "allowed_factory_kwargs": {
                "dtype",
                "threads_per_block",
                "items_per_thread",
                "blocked_to_striped",
            },
            "required_factory_kwargs": {"dtype", "threads_per_block"},
        },
        "radix_sort_pairs": {
            "namespace": "block",
            "runtime_arg_counts": {2, 4},
            "runtime_only_kwargs": ("begin_bit", "end_bit"),
            "runtime_factory_kw_prerequisites": {
                "begin_bit": "end_bit",
                "end_bit": "begin_bit",
            },
            "allowed_factory_kwargs": {
                "keys",
                "values",
                "key_dtype",
                "value_dtype",
                "threads_per_block",
                "items_per_thread",
                "blocked_to_striped",
            },
            "required_factory_kwargs": {
                "threads_per_block",
            },
        },
        "radix_sort_pairs_descending": {
            "namespace": "block",
            "runtime_arg_counts": {2, 4},
            "runtime_only_kwargs": ("begin_bit", "end_bit"),
            "runtime_factory_kw_prerequisites": {
                "begin_bit": "end_bit",
                "end_bit": "begin_bit",
            },
            "allowed_factory_kwargs": {
                "keys",
                "values",
                "key_dtype",
                "value_dtype",
                "threads_per_block",
                "items_per_thread",
                "blocked_to_striped",
            },
            "required_factory_kwargs": {
                "threads_per_block",
            },
        },
        "topk_max_keys": {
            "namespace": "block",
            "runtime_arg_counts": {2, 3, 4, 5},
            "runtime_factory_kwargs": ("num_valid", "begin_bit", "end_bit"),
            "runtime_factory_kw_prerequisites": {
                "begin_bit": "end_bit",
                "end_bit": "begin_bit",
            },
            "allowed_factory_kwargs": {
                "dtype",
                "threads_per_block",
                "items_per_thread",
                "num_valid",
                "begin_bit",
                "end_bit",
            },
            "required_factory_kwargs": {"dtype", "threads_per_block"},
        },
        "topk_min_keys": {
            "namespace": "block",
            "runtime_arg_counts": {2, 3, 4, 5},
            "runtime_factory_kwargs": ("num_valid", "begin_bit", "end_bit"),
            "runtime_factory_kw_prerequisites": {
                "begin_bit": "end_bit",
                "end_bit": "begin_bit",
            },
            "allowed_factory_kwargs": {
                "dtype",
                "threads_per_block",
                "items_per_thread",
                "num_valid",
                "begin_bit",
                "end_bit",
            },
            "required_factory_kwargs": {"dtype", "threads_per_block"},
        },
        "topk_max_pairs": {
            "namespace": "block",
            "runtime_arg_counts": {3, 4, 5, 6},
            "runtime_factory_kwargs": ("num_valid", "begin_bit", "end_bit"),
            "runtime_factory_kw_prerequisites": {
                "begin_bit": "end_bit",
                "end_bit": "begin_bit",
            },
            "allowed_factory_kwargs": {
                "keys",
                "values",
                "key_dtype",
                "value_dtype",
                "threads_per_block",
                "items_per_thread",
                "num_valid",
                "begin_bit",
                "end_bit",
            },
            "required_factory_kwargs": {
                "threads_per_block",
            },
        },
        "topk_min_pairs": {
            "namespace": "block",
            "runtime_arg_counts": {3, 4, 5, 6},
            "runtime_factory_kwargs": ("num_valid", "begin_bit", "end_bit"),
            "runtime_factory_kw_prerequisites": {
                "begin_bit": "end_bit",
                "end_bit": "begin_bit",
            },
            "allowed_factory_kwargs": {
                "keys",
                "values",
                "key_dtype",
                "value_dtype",
                "threads_per_block",
                "items_per_thread",
                "num_valid",
                "begin_bit",
                "end_bit",
            },
            "required_factory_kwargs": {
                "threads_per_block",
            },
        },
        "warp_reduce": {
            "namespace": "warp",
            "runtime_arg_counts": {1, 2},
            "runtime_factory_kwargs": ("valid_items",),
            "allowed_factory_kwargs": {
                "dtype",
                "binary_op",
                "threads_in_warp",
                "threads_per_block",
                "methods",
            },
            "required_factory_kwargs": {"dtype", "binary_op"},
        },
        "warp_sum": {
            "namespace": "warp",
            "runtime_arg_counts": {1, 2},
            "runtime_factory_kwargs": ("valid_items",),
            "allowed_factory_kwargs": {
                "dtype",
                "threads_in_warp",
                "threads_per_block",
            },
            "required_factory_kwargs": {"dtype"},
        },
        "warp_max": {
            "namespace": "warp",
            "runtime_arg_counts": {1, 2},
            "runtime_factory_kwargs": ("valid_items",),
            "allowed_factory_kwargs": {
                "dtype",
                "threads_in_warp",
                "threads_per_block",
            },
            "required_factory_kwargs": {"dtype"},
        },
        "warp_min": {
            "namespace": "warp",
            "runtime_arg_counts": {1, 2},
            "runtime_factory_kwargs": ("valid_items",),
            "allowed_factory_kwargs": {
                "dtype",
                "threads_in_warp",
                "threads_per_block",
            },
            "required_factory_kwargs": {"dtype"},
        },
        "warp_exclusive_sum": {
            "namespace": "warp",
            "runtime_arg_counts": {1, 2},
            "runtime_factory_kwargs": ("warp_aggregate",),
            "allowed_factory_kwargs": {
                "dtype",
                "threads_in_warp",
                "threads_per_block",
                "warp_aggregate",
            },
            "required_factory_kwargs": {"dtype"},
        },
        "warp_inclusive_sum": {
            "namespace": "warp",
            "runtime_arg_counts": {1, 2},
            "runtime_factory_kwargs": ("warp_aggregate",),
            "allowed_factory_kwargs": {
                "dtype",
                "threads_in_warp",
                "threads_per_block",
                "warp_aggregate",
            },
            "required_factory_kwargs": {"dtype"},
        },
        "warp_exclusive_scan": {
            "namespace": "warp",
            "runtime_arg_counts": {1, 2, 3},
            "runtime_factory_kwargs": ("valid_items", "warp_aggregate"),
            "allowed_factory_kwargs": {
                "dtype",
                "scan_op",
                "initial_value",
                "threads_in_warp",
                "threads_per_block",
                "valid_items",
                "warp_aggregate",
            },
            "required_factory_kwargs": {"dtype", "scan_op"},
        },
        "warp_inclusive_scan": {
            "namespace": "warp",
            "runtime_arg_counts": {1, 2, 3},
            "runtime_factory_kwargs": ("valid_items", "warp_aggregate"),
            "allowed_factory_kwargs": {
                "dtype",
                "scan_op",
                "initial_value",
                "threads_in_warp",
                "threads_per_block",
                "valid_items",
                "warp_aggregate",
            },
            "required_factory_kwargs": {"dtype", "scan_op"},
        },
        "warp_load": {
            "namespace": "warp",
            "runtime_arg_counts": {2, 3, 4},
            "runtime_factory_kwargs": ("num_valid_items", "oob_default"),
            "runtime_factory_kw_prerequisites": {"oob_default": "num_valid_items"},
            "allowed_factory_kwargs": {
                "dtype",
                "items_per_thread",
                "threads_in_warp",
                "threads_per_block",
                "algorithm",
                "methods",
                "offset",
                "_physical_warp_tile_origin",
                "_common_profile_operation",
            },
            "required_factory_kwargs": {"dtype"},
        },
        "warp_store": {
            "namespace": "warp",
            "runtime_arg_counts": {2, 3},
            "runtime_factory_kwargs": ("num_valid_items",),
            "allowed_factory_kwargs": {
                "dtype",
                "items_per_thread",
                "threads_in_warp",
                "threads_per_block",
                "algorithm",
                "methods",
                "offset",
                "_physical_warp_tile_origin",
                "_group_root_store",
                "_common_profile_operation",
            },
            "required_factory_kwargs": {"dtype"},
        },
        "warp_exchange": {
            "namespace": "warp",
            "runtime_arg_counts": {2, 3},
            "allowed_factory_kwargs": {
                "dtype",
                "items_per_thread",
                "threads_in_warp",
                "threads_per_block",
                "warp_exchange_type",
                "offset_dtype",
                "use_output_items",
                "methods",
                "_common_profile_operation",
            },
            "required_factory_kwargs": {"dtype"},
        },
        "warp_merge_sort_keys": {
            "namespace": "warp",
            "runtime_arg_counts": {1, 3},
            "runtime_factory_kwargs": ("valid_items", "oob_default"),
            "runtime_factory_kw_prerequisites": {"oob_default": "valid_items"},
            "allowed_factory_kwargs": {
                "dtype",
                "items_per_thread",
                "compare_op",
                "threads_in_warp",
                "threads_per_block",
                "valid_items",
                "oob_default",
                "methods",
            },
            "required_factory_kwargs": {"dtype", "items_per_thread", "compare_op"},
        },
        "warp_merge_sort_pairs": {
            "namespace": "warp",
            "runtime_arg_counts": {2, 4},
            "runtime_factory_kwargs": ("valid_items", "oob_default"),
            "runtime_factory_kw_prerequisites": {"oob_default": "valid_items"},
            "allowed_factory_kwargs": {
                "keys",
                "values",
                "items_per_thread",
                "compare_op",
                "threads_in_warp",
                "threads_per_block",
                "valid_items",
                "oob_default",
                "methods",
            },
            "required_factory_kwargs": {
                "keys",
                "values",
                "items_per_thread",
                "compare_op",
            },
        },
    }

    for _private_name, _public_name in {
        "_common_merge_sort_keys": "merge_sort_keys",
        "_common_merge_sort_pairs": "merge_sort_pairs",
        "_common_warp_merge_sort_keys": "warp_merge_sort_keys",
        "_common_warp_merge_sort_pairs": "warp_merge_sort_pairs",
        "_common_radix_sort_keys": "radix_sort_keys",
        "_common_radix_sort_pairs": "radix_sort_pairs",
        "_common_radix_rank": "radix_rank",
        "_common_topk_max_keys": "topk_max_keys",
        "_common_topk_min_keys": "topk_min_keys",
        "_common_topk_max_pairs": "topk_max_pairs",
        "_common_topk_min_pairs": "topk_min_pairs",
    }.items():
        _public_spec = _OP_SPECS[_public_name]
        _OP_SPECS[_private_name] = {
            **_public_spec,
            "allowed_factory_kwargs": set(_public_spec["allowed_factory_kwargs"]),
            "required_factory_kwargs": set(_public_spec["required_factory_kwargs"]),
        }
        if _private_name in {"_common_radix_sort_keys", "_common_radix_sort_pairs"}:
            _OP_SPECS[_private_name]["allowed_factory_kwargs"].add("descending")
        if _private_name in {
            "_common_topk_max_keys",
            "_common_topk_min_keys",
            "_common_topk_max_pairs",
            "_common_topk_min_pairs",
        }:
            _OP_SPECS[_private_name]["runtime_factory_kw_prerequisites"] = {
                "end_bit": "begin_bit"
            }
    del _private_name, _public_name, _public_spec

    for _private_name, _public_name in {
        "_qualified_group_topk_max_keys": "topk_max_keys",
        "_qualified_group_topk_min_keys": "topk_min_keys",
        "_qualified_group_topk_max_pairs": "topk_max_pairs",
        "_qualified_group_topk_min_pairs": "topk_min_pairs",
    }.items():
        _public_spec = _OP_SPECS[_public_name]
        _OP_SPECS[_private_name] = {
            **_public_spec,
            "runtime_factory_kw_prerequisites": {"end_bit": "begin_bit"},
            "allowed_factory_kwargs": set(_public_spec["allowed_factory_kwargs"]),
            "required_factory_kwargs": set(_public_spec["required_factory_kwargs"]),
        }
    del _private_name, _public_name, _public_spec

    for _spec in _OP_SPECS.values():
        if "dtype" in _spec["allowed_factory_kwargs"]:
            _spec["allowed_factory_kwargs"].add("_common_profile_operation")
        if (
            _spec["namespace"] == "block"
            and "threads_per_block" in _spec["allowed_factory_kwargs"]
        ):
            _spec["allowed_factory_kwargs"].add("dim")
    del _spec

    _PARENT_SPECS = {
        "histogram": {
            "namespace": "block",
            "allowed_factory_kwargs": {"algorithm", "_common_profile_operation"},
        },
        "run_length": {
            "namespace": "block",
            "allowed_factory_kwargs": {"decoded_offset_dtype", "temp_storage"},
        },
        "_common_run_length": {
            "namespace": "block",
            "allowed_factory_kwargs": {
                "decoded_offset_dtype",
                "_static_decoded_window_offset",
            },
        },
        "_qualified_group_run_length": {
            "namespace": "block",
            "allowed_factory_kwargs": {
                "decoded_offset_dtype",
                "_static_decoded_window_offset",
            },
        },
    }

    _BLOCK_OPS = frozenset(
        name for name, spec in _OP_SPECS.items() if spec["namespace"] == "block"
    )
    _WARP_OPS = frozenset(
        name for name, spec in _OP_SPECS.items() if spec["namespace"] == "warp"
    )

    @staticmethod
    def _require_matching_items_per_thread(
        op_name: str,
        lhs_name: str,
        lhs_spec: _ThreadDataSpec | None,
        rhs_name: str,
        rhs_spec: _ThreadDataSpec | None,
    ) -> None:
        if lhs_spec is None or rhs_spec is None:
            return
        lhs_items = lhs_spec.items_per_thread
        rhs_items = rhs_spec.items_per_thread
        if lhs_items is not None and rhs_items is not None and lhs_items != rhs_items:
            raise CoopSinglePhaseRewriteError(
                f"coop single-phase '{op_name}' requires {lhs_name}/{rhs_name} "
                "arrays to have matching items_per_thread."
            )

    def __init__(self, state):
        super().__init__(state)
        self._state = state
        self._func_ir = state.func_ir
        self._block: ir.Block | None = None
        self._block_defs: dict[str, object] = {}
        self._matches: dict[ir.Assign, _RewriteMatch] = {}
        self._temp_storage_assigns: set[ir.Assign] = set()
        self._parent_ctor_assigns: set[ir.Assign] = set()
        self._direct_invocable_temp_storage_calls: dict[
            ir.Assign, _DirectInvocableTempStorageCall
        ] = {}
        self._parent_ctor_specs: dict[str, _ParentCtorSpec] = {}
        self._parent_ctor_func_vars: set[str] = set()
        self._temp_storage_func_vars: set[str] = set()
        self._temp_storage_ctor_specs: dict[str, _TempStorageCtorSpec] = {}
        self._temp_storage_ctor_order: dict[str, int] = {}
        self._thread_data_func_vars: set[str] = set()
        self._typed_group_payload_func_vars: set[str] = set()
        self._thread_data_specs: dict[str, _ThreadDataSpec] = {}
        self._func_ir_identity: int | None = None
        self._func_temp_storage_requirements: dict[
            str, _TempStorageRequirementSummary
        ] = {}
        self._temp_storage_plans: dict[str, _TempStoragePlan] = {}
        self._temp_storage_global_plan: _TempStorageGlobalPlan | None = None
        self._compiletime_context = self._build_compiletime_context()
        self._arg_type_map = self._build_arg_type_map()
        self._arg_value_map = self._build_arg_value_map()
        self._invocable_cache: dict[
            tuple[str, tuple[tuple[str, str, str], ...]], object
        ] = {}
        self._dataclass_invocable_getattrs: dict[ir.Assign, object] = {}
        self._dataclass_invocable_func_vars: set[str] = set()
        self._local_array_literal_shape_rewrites: dict[ir.Assign, _ThreadDataSpec] = {}
        self._prebundled_specializations: dict[
            tuple[str, tuple[tuple[str, str, str], ...]],
            tuple[object, int | None, int | tuple[int, ...] | None],
        ] = {}
        self._deferred_launch_dim_inference = False

    def _infer_constant(self, value):
        return self._func_ir.infer_constant(value)

    def _build_compiletime_context(self) -> dict[str, object]:
        func = self._state.func_id.func
        context: dict[str, object] = {}

        context.update(getattr(func, "__globals__", {}))
        freevars = getattr(func.__code__, "co_freevars", ())
        closure = getattr(func, "__closure__", None) or ()
        for name, cell in zip(freevars, closure):
            try:
                context[name] = cell.cell_contents
            except ValueError:
                # Empty closure cell.
                continue

        return context

    def _build_arg_type_map(self) -> dict[str, object]:
        arg_names = tuple(getattr(self._func_ir, "arg_names", ()) or ())
        arg_types = tuple(getattr(self._state, "args", ()) or ())
        if len(arg_names) != len(arg_types):
            return {}
        return dict(zip(arg_names, arg_types))

    def _build_arg_value_map(self) -> dict[str, object]:
        arg_names = tuple(getattr(self._func_ir, "arg_names", ()) or ())
        targetoptions = getattr(self._state, "metadata", {}).get("targetoptions") or {}
        arg_values = tuple(targetoptions.get("__launch_args__", ()) or ())
        if len(arg_names) != len(arg_values):
            return {}
        return dict(zip(arg_names, arg_values))

    def _lookup_definition(self, value):
        if isinstance(value, ir.Var):
            if value.name in self._block_defs:
                return self._block_defs[value.name]
            try:
                return self._func_ir.get_definition(value)
            except KeyError:
                return None
        if isinstance(value, str):
            if value in self._block_defs:
                return self._block_defs[value]
            try:
                return self._func_ir.get_definition(value)
            except KeyError:
                return None
        return value

    def _lookup_definitions(self, value) -> list[object]:
        defs: list[object] = []
        seen_ids: set[int] = set()

        def add(candidate) -> None:
            if candidate is None:
                return
            cid = id(candidate)
            if cid in seen_ids:
                return
            seen_ids.add(cid)
            defs.append(candidate)

        if isinstance(value, ir.Var):
            if value.name in self._block_defs:
                add(self._block_defs[value.name])
            for definition in (getattr(self._func_ir, "_definitions", {}) or {}).get(
                value.name, ()
            ):
                add(definition)
            return defs

        if isinstance(value, str):
            if value in self._block_defs:
                add(self._block_defs[value])
            for definition in (getattr(self._func_ir, "_definitions", {}) or {}).get(
                value, ()
            ):
                add(definition)
            return defs

        return [value]

    def _resolve_attribute_chain(self, func_var):
        attrs: list[str] = []
        current = self._lookup_definition(func_var)
        if current is None:
            return None
        while isinstance(current, ir.Expr) and current.op == "getattr":
            attrs.append(current.attr)
            current = self._lookup_definition(current.value)
            if current is None:
                return None

        if isinstance(current, (ir.Global, ir.FreeVar, ir.Const)):
            root = current.value
        else:
            return None

        attrs.reverse()
        return root, attrs

    def _resolve_python_value(self, value):
        chain = self._resolve_attribute_chain(value)
        if chain is None:
            return None
        root, attrs = chain
        obj = root
        try:
            for attr in attrs:
                obj = getattr(obj, attr)
        except (AttributeError, ImportError):
            return None
        return obj

    def _is_common_root_member(self, value, name: str) -> bool:
        member = getattr(_common_root_api, name)
        return (
            self._resolve_python_value(value) is member
            and getattr(member, "__cuda_coop_backend_member__", None) == name
        )

    def _chain_can_be_coop(self, root, attrs: list[str]) -> bool:
        if not attrs:
            return False

        public_name = attrs[-1]
        root_name = getattr(root, "__name__", "")

        namespace = None
        if len(attrs) >= 2 and attrs[-2] in {"_block", "_warp"}:
            namespace = attrs[-2].removeprefix("_")
        elif root_name in {"cuda.coop.numba_mlir._block", "cuda.coop.numba_mlir._warp"}:
            namespace = root_name.rsplit(".", 1)[-1].removeprefix("_")

        if namespace is None:
            return False

        op_name = public_name
        if namespace == "warp":
            # Warp factories intentionally use unique internal function names to
            # avoid collisions with block factories.
            op_name = {
                "load": "warp_load",
                "store": "warp_store",
                "exchange": "warp_exchange",
                "reduce": "warp_reduce",
                "sum": "warp_sum",
                "max": "warp_max",
                "min": "warp_min",
                "exclusive_sum": "warp_exclusive_sum",
                "inclusive_sum": "warp_inclusive_sum",
                "exclusive_scan": "warp_exclusive_scan",
                "inclusive_scan": "warp_inclusive_scan",
                "merge_sort_keys": "warp_merge_sort_keys",
                "merge_sort_pairs": "warp_merge_sort_pairs",
            }.get(public_name, public_name)

        if op_name not in self._OP_SPECS:
            return False
        if self._OP_SPECS[op_name]["namespace"] != namespace:
            return False

        namespace_attr = f"_{namespace}"
        if (
            attrs == [namespace_attr, public_name]
            and root_name == "cuda.coop.numba_mlir"
        ):
            return True
        if (
            attrs == ["numba_mlir", namespace_attr, public_name]
            and root_name == "cuda.coop"
        ):
            return True
        if (
            attrs == ["coop", "numba_mlir", namespace_attr, public_name]
            and root_name in self._CUDA_ROOT_MODULES
        ):
            return True
        if attrs == [public_name] and root_name == (
            f"cuda.coop.numba_mlir.{namespace_attr}"
        ):
            return True
        return False

    def _is_supported_factory(self, obj) -> bool:
        name = getattr(obj, "__name__", None)
        module_name = getattr(obj, "__module__", "")
        if not callable(obj) or name not in self._OP_SPECS:
            return False
        expected_ns = self._OP_SPECS[name]["namespace"]
        if expected_ns == "group":
            return module_name == "cuda.coop.numba_mlir._group_provider"
        private_namespace = f"_{expected_ns}"
        return module_name == f"cuda.coop.numba_mlir.{private_namespace}" or (
            module_name.startswith(f"cuda.coop.numba_mlir.{private_namespace}.")
        )

    def _resolve_factory_from_var(self, func_var):
        direct = None
        direct_def = self._lookup_definition(func_var)
        if isinstance(direct_def, (ir.Global, ir.FreeVar, ir.Const)):
            direct = direct_def.value
        elif callable(direct_def):
            direct = direct_def
        elif direct_def is None:
            try:
                direct = self._infer_constant(func_var)
            except _INFERENCE_EXCEPTIONS:
                direct = None
        if self._is_supported_factory(direct):
            return direct

        chain = self._resolve_attribute_chain(func_var)
        if chain is None:
            return None
        root, attrs = chain
        if not self._chain_can_be_coop(root, attrs):
            return None

        obj = root
        for attr in attrs:
            obj = getattr(obj, attr)
        if self._is_supported_factory(obj):
            return obj
        return None

    def _resolve_invocable_from_var(self, func_var):
        from ._types import Invocable

        direct = None
        direct_def = self._lookup_definition(func_var)
        if isinstance(direct_def, (ir.Global, ir.FreeVar, ir.Const)):
            direct = direct_def.value
        elif isinstance(direct_def, Invocable):
            direct = direct_def
        elif direct_def is None:
            try:
                direct = self._infer_constant(func_var)
            except _INFERENCE_EXCEPTIONS:
                direct = None
        if isinstance(direct, Invocable):
            return direct
        return None

    def _resolve_direct_invocable_temp_storage_call(
        self, call: ir.Expr
    ) -> _DirectInvocableTempStorageCall | None:
        invocable = self._resolve_invocable_from_var(call.func)
        if invocable is None:
            return None

        temp_storage_var = None
        rewritten_kws = []
        for name, value in call.kws:
            if name != "temp_storage":
                rewritten_kws.append((name, value))
                continue
            if temp_storage_var is not None:
                raise CoopSinglePhaseRewriteError(
                    "Duplicate coop invocable temp_storage keyword."
                )
            if not isinstance(value, ir.Var):
                raise CoopSinglePhaseRewriteError(
                    "coop invocable temp_storage must be a variable."
                )
            temp_storage_var = value

        if temp_storage_var is None:
            return None
        return _DirectInvocableTempStorageCall(
            invocable=invocable,
            temp_storage_var=temp_storage_var,
            rewritten_kws=tuple(rewritten_kws),
        )

    def _resolve_dataclass_invocable_getattr(self, expr: ir.Expr):
        if expr.op != "getattr" or not isinstance(expr.value, ir.Var):
            return None

        dc = self._arg_value_map.get(expr.value.name)
        if not getattr(dc, "__cuda_coop_numba_mlir_gpu_dataclass__", False):
            return None

        try:
            value = getattr(dc, expr.attr)
        except AttributeError:
            return None

        from ._types import Invocable

        if isinstance(value, Invocable):
            return value
        return None

    def _chain_can_be_parent_factory(self, root, attrs: list[str]) -> bool:
        if not attrs:
            return False

        public_name = attrs[-1]
        if public_name not in self._PARENT_SPECS:
            return False

        root_name = getattr(root, "__name__", "")
        namespace = self._PARENT_SPECS[public_name]["namespace"]
        namespace_attr = f"_{namespace}"

        if (
            attrs == [namespace_attr, public_name]
            and root_name == "cuda.coop.numba_mlir"
        ):
            return True
        if (
            attrs == ["numba_mlir", namespace_attr, public_name]
            and root_name == "cuda.coop"
        ):
            return True
        if (
            attrs == ["coop", "numba_mlir", namespace_attr, public_name]
            and root_name in self._CUDA_ROOT_MODULES
        ):
            return True
        if attrs == [public_name] and root_name == (
            f"cuda.coop.numba_mlir.{namespace_attr}"
        ):
            return True
        return False

    def _is_supported_parent_factory(self, obj) -> bool:
        name = getattr(obj, "__name__", None)
        module_name = getattr(obj, "__module__", "")
        if not callable(obj) or name not in self._PARENT_SPECS:
            return False
        expected_ns = self._PARENT_SPECS[name]["namespace"]
        private_namespace = f"_{expected_ns}"
        return module_name == f"cuda.coop.numba_mlir.{private_namespace}" or (
            module_name.startswith(f"cuda.coop.numba_mlir.{private_namespace}.")
        )

    def _resolve_parent_factory_from_var(self, func_var):
        direct = None
        direct_def = self._lookup_definition(func_var)
        if isinstance(direct_def, (ir.Global, ir.FreeVar, ir.Const)):
            direct = direct_def.value
        elif callable(direct_def):
            direct = direct_def
        elif direct_def is None:
            try:
                direct = self._infer_constant(func_var)
            except _INFERENCE_EXCEPTIONS:
                direct = None
        if self._is_supported_parent_factory(direct):
            return direct

        chain = self._resolve_attribute_chain(func_var)
        if chain is None:
            return None
        root, attrs = chain
        if not self._chain_can_be_parent_factory(root, attrs):
            return None

        obj = root
        for attr in attrs:
            obj = getattr(obj, attr)
        if self._is_supported_parent_factory(obj):
            return obj
        return None

    def _extract_1d_extent_literal(self, value_ref):
        try:
            value = self._infer_constant(value_ref)
        except _INFERENCE_EXCEPTIONS:
            value = self._resolve_dataclass_field_value_ref(value_ref)
            if value is None:
                return None

        if isinstance(value, int):
            return value

        if isinstance(value, tuple) and len(value) == 1 and isinstance(value[0], int):
            return int(value[0])
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], int):
            return int(value[0])

        return None

    def _resolve_dataclass_field_value_ref(self, value_ref):
        if not isinstance(value_ref, ir.Var):
            return None
        definition = self._lookup_definition(value_ref)
        if not (isinstance(definition, ir.Expr) and definition.op == "getattr"):
            return None
        if not isinstance(definition.value, ir.Var):
            return None

        dc = self._arg_value_map.get(definition.value.name)
        if not getattr(dc, "__cuda_coop_numba_mlir_gpu_dataclass__", False):
            return None

        try:
            return getattr(dc, definition.attr)
        except AttributeError:
            return None

    @staticmethod
    def _call_shape_ref(call: ir.Expr):
        if call.args:
            return call.args[0]
        for name, value in call.kws:
            if name == "shape":
                return value
        return None

    def _is_temp_storage_ctor_call(self, call: ir.Expr) -> bool:
        if self._is_common_root_member(call.func, "TempStorage"):
            return True
        chain = self._resolve_attribute_chain(call.func)
        if chain is None:
            return False
        root, attrs = chain
        root_name = getattr(root, "__name__", "")
        if attrs == ["TempStorage"] and root_name == "cuda.coop.numba_mlir":
            return True
        if attrs == ["numba_mlir", "TempStorage"] and root_name == "cuda.coop":
            return True
        if (
            attrs == ["coop", "numba_mlir", "TempStorage"]
            and root_name in self._CUDA_ROOT_MODULES
        ):
            return True
        return False

    def _is_thread_data_ctor_call(self, call: ir.Expr) -> bool:
        if self._is_common_root_member(call.func, "ThreadData"):
            return True
        chain = self._resolve_attribute_chain(call.func)
        if chain is None:
            return False
        root, attrs = chain
        if not attrs:
            from . import ThreadData

            return root is ThreadData
        root_name = getattr(root, "__name__", "")
        if attrs == ["ThreadData"] and root_name == "cuda.coop.numba_mlir":
            return True
        if attrs == ["numba_mlir", "ThreadData"] and root_name == "cuda.coop":
            return True
        if (
            attrs == ["coop", "numba_mlir", "ThreadData"]
            and root_name in self._CUDA_ROOT_MODULES
        ):
            return True
        return False

    def _is_typed_group_payload_ctor_call(self, call: ir.Expr) -> bool:
        chain = self._resolve_attribute_chain(call.func)
        if chain is None:
            return False
        root, attrs = chain
        if attrs:
            return False
        from ._group_rewrites import _typed_group_payload_like

        return root is _typed_group_payload_like

    def _is_typed_group_payload_var(self, value: ir.Var) -> bool:
        return any(
            isinstance(definition, ir.Expr)
            and definition.op == "call"
            and self._is_typed_group_payload_ctor_call(definition)
            for definition in self._lookup_definitions(value)
        )

    def _extract_typed_group_payload_spec(
        self,
        call: ir.Expr,
        *,
        seen: set[str] | None = None,
    ) -> _ThreadDataSpec:
        if seen is None:
            seen = set()
        if len(call.args) not in {3, 4} or call.kws:
            raise CoopSinglePhaseRewriteError(
                "typed group payload marker requires prototype, array-kind, "
                "dtype-policy, and optional explicit-extent arguments"
            )
        prototype, is_array_ref, dtype_policy_ref = call.args[:3]
        if not isinstance(prototype, ir.Var):
            raise CoopSinglePhaseRewriteError(
                "typed group payload prototype must be a variable"
            )
        try:
            is_array = self._infer_constant(is_array_ref)
            dtype_policy = self._infer_constant(dtype_policy_ref)
        except _INFERENCE_EXCEPTIONS as exc:
            raise CoopSinglePhaseRewriteError(
                "typed group payload shape and dtype policy must be "
                "compile-time constants"
            ) from exc
        if not isinstance(is_array, bool):
            raise CoopSinglePhaseRewriteError(
                "typed group payload array-kind must be a compile-time bool"
            )

        from ._group_rewrites import (
            _PAYLOAD_DTYPE_BOOL,
            _PAYLOAD_DTYPE_INT32,
            _PAYLOAD_DTYPE_LIKE,
        )

        if dtype_policy not in {
            _PAYLOAD_DTYPE_LIKE,
            _PAYLOAD_DTYPE_BOOL,
            _PAYLOAD_DTYPE_INT32,
        }:
            raise CoopSinglePhaseRewriteError(
                f"unknown typed group payload dtype policy {dtype_policy!r}"
            )

        prototype_spec = self._resolve_array_spec_from_var(prototype, seen=set(seen))
        if len(call.args) == 4:
            try:
                items_per_thread = self._infer_constant(call.args[3])
            except _INFERENCE_EXCEPTIONS as exc:
                raise CoopSinglePhaseRewriteError(
                    "typed group payload explicit extent must be a "
                    "compile-time positive integer"
                ) from exc
            if (
                isinstance(items_per_thread, bool)
                or not isinstance(items_per_thread, int)
                or items_per_thread < 1
            ):
                raise CoopSinglePhaseRewriteError(
                    "typed group payload explicit extent must be a "
                    "compile-time positive integer"
                )
        elif is_array:
            items_per_thread = (
                prototype_spec.items_per_thread if prototype_spec is not None else None
            )
        else:
            items_per_thread = 1

        if dtype_policy == _PAYLOAD_DTYPE_BOOL:
            from numba_cuda_mlir import types as numba_mlir_types

            dtype = numba_mlir_types.boolean
        elif dtype_policy == _PAYLOAD_DTYPE_INT32:
            from numba_cuda_mlir import types as numba_mlir_types

            dtype = numba_mlir_types.int32
        else:
            dtype = prototype_spec.dtype if prototype_spec is not None else None
            if dtype is None:
                dtype = self._resolve_var_dtype(prototype)

        return _ThreadDataSpec(
            items_per_thread=items_per_thread,
            dtype=dtype,
            common_v1=(
                prototype_spec.common_v1 if prototype_spec is not None else False
            ),
        )

    def _extract_thread_data_spec(self, call: ir.Expr) -> _ThreadDataSpec:
        kw_map = {name: value for name, value in call.kws}
        is_common_root = self._is_common_root_member(call.func, "ThreadData")
        allowed_keywords = {"items_per_thread", "dtype"}
        if not is_common_root:
            allowed_keywords.add("alignas")
        unexpected_keywords = sorted(set(kw_map) - allowed_keywords)
        if unexpected_keywords:
            names = ", ".join(unexpected_keywords)
            scope = "cuda.coop" if is_common_root else "cuda.coop.numba_mlir"
            raise CoopSinglePhaseRewriteError(
                f"{scope}.ThreadData got unexpected keyword(s): {names}"
            )

        extent_refs = []
        if call.args:
            extent_refs.append(("positional items_per_thread", call.args[0]))
        if "items_per_thread" in kw_map:
            extent_refs.append(("items_per_thread", kw_map["items_per_thread"]))

        if len(call.args) > 2:
            raise CoopSinglePhaseRewriteError(
                "coop.ThreadData accepts at most items_per_thread and dtype "
                "positional arguments."
            )
        if len(extent_refs) > 1:
            names = " and ".join(name for name, _ in extent_refs)
            raise CoopSinglePhaseRewriteError(
                f"coop.ThreadData received both {names}; specify only one."
            )
        if not extent_refs:
            raise CoopSinglePhaseRewriteError(
                "coop.ThreadData requires items_per_thread."
            )
        items_ref = extent_refs[0][1]

        dtype_ref = None
        if len(call.args) == 2:
            dtype_ref = call.args[1]
        if "dtype" in kw_map:
            if dtype_ref is not None:
                raise CoopSinglePhaseRewriteError(
                    "coop.ThreadData received dtype both positionally and by keyword."
                )
            dtype_ref = kw_map["dtype"]

        alignas_ref = kw_map.get("alignas")
        if alignas_ref is not None:
            try:
                alignas = self._infer_constant(alignas_ref)
            except _INFERENCE_EXCEPTIONS as exc:
                raise CoopSinglePhaseRewriteError(
                    "cuda.coop.numba_mlir.ThreadData alignas must be a "
                    "compile-time positive integer"
                ) from exc
            if isinstance(alignas, bool) or not isinstance(alignas, int) or alignas < 1:
                raise CoopSinglePhaseRewriteError(
                    "cuda.coop.numba_mlir.ThreadData alignas must be a "
                    "compile-time positive integer"
                )

        try:
            raw_items_per_thread = self._infer_constant(items_ref)
        except _INFERENCE_EXCEPTIONS as exc:
            raw_items_per_thread = self._resolve_dataclass_field_value_ref(items_ref)
            if raw_items_per_thread is None:
                raise CoopSinglePhaseRewriteError(
                    "items_per_thread must be a compile-time integer"
                ) from exc
        if isinstance(raw_items_per_thread, bool):
            raise CoopSinglePhaseRewriteError("items_per_thread must be an integer")
        try:
            items_per_thread = operator.index(raw_items_per_thread)
        except TypeError as exc:
            raise CoopSinglePhaseRewriteError(
                "items_per_thread must be an integer"
            ) from exc
        if items_per_thread <= 0:
            raise CoopSinglePhaseRewriteError(
                "items_per_thread must be a positive integer"
            )

        dtype = None
        if dtype_ref is not None:
            dtype = self._resolve_dtype_ref(dtype_ref)
            if any(dtype is alias for alias in (bool, int, float, complex)):
                dtype = normalize_dtype_param(dtype)

        return _ThreadDataSpec(
            items_per_thread=items_per_thread,
            dtype=dtype,
            common_v1=is_common_root,
        )

    @staticmethod
    def _merge_thread_data_specs(
        existing: _ThreadDataSpec | None, observed: _ThreadDataSpec
    ) -> _ThreadDataSpec:
        if existing is None:
            return observed

        if (
            existing.items_per_thread is not None
            and observed.items_per_thread is not None
            and existing.items_per_thread != observed.items_per_thread
        ):
            raise CoopSinglePhaseRewriteError(
                "Inconsistent items_per_thread across merged coop.ThreadData aliases."
            )
        if (
            existing.dtype is not None
            and observed.dtype is not None
            and existing.dtype != observed.dtype
        ):
            raise CoopSinglePhaseRewriteError(
                "Inconsistent dtype across merged coop.ThreadData aliases."
            )

        items_per_thread = existing.items_per_thread
        if items_per_thread is None:
            items_per_thread = observed.items_per_thread

        dtype = existing.dtype
        if dtype is None:
            dtype = observed.dtype

        return _ThreadDataSpec(
            items_per_thread=items_per_thread,
            dtype=dtype,
            common_v1=existing.common_v1 or observed.common_v1,
        )

    @staticmethod
    def _merge_temp_storage_ctor_specs(
        existing: _TempStorageCtorSpec | None, observed: _TempStorageCtorSpec
    ) -> _TempStorageCtorSpec:
        if existing is None:
            return observed
        if existing != observed:
            raise CoopSinglePhaseRewriteError(
                "Inconsistent TempStorage constructor metadata across merged aliases."
            )
        return existing

    def _record_inferred_thread_data_dtype(
        self, value: ir.Var, dtype, seen: set[str] | None = None
    ) -> None:
        if not isinstance(value, ir.Var) or dtype is None:
            return
        if seen is None:
            seen = set()
        if value.name in seen:
            return
        seen.add(value.name)

        spec = self._thread_data_specs.get(value.name)
        if spec is None:
            spec = self._resolve_thread_data_spec(value)
            if spec is not None:
                self._thread_data_specs[value.name] = spec
        if spec is not None:
            if spec.dtype is None:
                self._thread_data_specs[value.name] = _ThreadDataSpec(
                    items_per_thread=spec.items_per_thread,
                    dtype=dtype,
                    common_v1=spec.common_v1,
                )
            elif spec.dtype != dtype:
                raise CoopSinglePhaseRewriteError(
                    "Inconsistent inferred dtype for coop.ThreadData usage."
                )

        for definition in self._lookup_definitions(value):
            if isinstance(definition, ir.Var):
                self._record_inferred_thread_data_dtype(definition, dtype, seen)
                continue
            if not isinstance(definition, ir.Expr):
                continue
            if definition.op == "cast":
                cast_value = getattr(definition, "value", None)
                if isinstance(cast_value, ir.Var):
                    self._record_inferred_thread_data_dtype(cast_value, dtype, seen)
            elif definition.op == "phi":
                for incoming in _phi_incoming_values(definition):
                    if isinstance(incoming, ir.Var):
                        self._record_inferred_thread_data_dtype(incoming, dtype, seen)

    def _infer_consistent_dtype_from_runtime_args(
        self, runtime_args: list[ir.Var], exclude_indices: tuple[int, ...]
    ):
        inferred = None
        for idx, arg in enumerate(runtime_args):
            if idx in exclude_indices or not isinstance(arg, ir.Var):
                continue
            dtype = self._resolve_var_dtype(arg)
            if dtype is None:
                continue
            if inferred is None:
                inferred = dtype
                continue
            if inferred != dtype:
                raise CoopSinglePhaseRewriteError(
                    "Failed to infer a consistent dtype from runtime arguments."
                )
        return inferred

    def _extract_temp_storage_ctor_spec(self, call: ir.Expr) -> _TempStorageCtorSpec:
        kw_map = {name: value for name, value in call.kws}
        parameter_names = (
            "size_in_bytes",
            "alignment",
            "auto_sync",
            "sharing",
        )
        if len(call.args) > len(parameter_names):
            raise CoopSinglePhaseRewriteError(
                "TempStorage accepts at most size_in_bytes, alignment, auto_sync, "
                "and sharing positional arguments."
            )
        unexpected_keywords = sorted(set(kw_map) - set(parameter_names))
        if unexpected_keywords:
            names = ", ".join(unexpected_keywords)
            raise CoopSinglePhaseRewriteError(
                f"TempStorage got unexpected keyword(s): {names}"
            )

        refs = dict(zip(parameter_names, call.args))
        for name, value_ref in call.kws:
            if name in refs:
                raise CoopSinglePhaseRewriteError(
                    f"TempStorage got multiple values for argument {name!r}"
                )
            refs[name] = value_ref

        size_ref = refs.get("size_in_bytes")
        alignment_ref = refs.get("alignment")
        auto_sync_ref = refs.get("auto_sync")
        sharing_ref = refs.get("sharing")

        def infer_constant(value_ref, *, name: str):
            try:
                return self._infer_constant(value_ref)
            except _INFERENCE_EXCEPTIONS as exc:
                raise CoopSinglePhaseRewriteError(
                    f"TempStorage {name} must be a compile-time literal."
                ) from exc

        size_in_bytes = None
        if size_ref is not None:
            raw_size_in_bytes = infer_constant(size_ref, name="size_in_bytes")
            if raw_size_in_bytes is not None and (
                not isinstance(raw_size_in_bytes, int)
                or isinstance(raw_size_in_bytes, bool)
            ):
                raise CoopSinglePhaseRewriteError(
                    "TempStorage size_in_bytes must be an integer or None."
                )
            size_in_bytes = raw_size_in_bytes
            if size_in_bytes is not None and size_in_bytes <= 0:
                raise CoopSinglePhaseRewriteError(
                    "TempStorage size_in_bytes must be a positive integer."
                )

        alignment = None
        if alignment_ref is not None:
            raw_alignment = infer_constant(alignment_ref, name="alignment")
            if raw_alignment is not None and (
                not isinstance(raw_alignment, int) or isinstance(raw_alignment, bool)
            ):
                raise CoopSinglePhaseRewriteError(
                    "TempStorage alignment must be an integer or None."
                )
            if raw_alignment is not None:
                alignment = _normalize_temp_storage_alignment(raw_alignment)

        auto_sync = None
        if auto_sync_ref is not None:
            auto_sync = infer_constant(auto_sync_ref, name="auto_sync")
            if auto_sync is not None and not isinstance(auto_sync, bool):
                raise CoopSinglePhaseRewriteError(
                    "TempStorage auto_sync must be None/True/False."
                )

        sharing = "shared"
        if sharing_ref is not None:
            sharing = infer_constant(sharing_ref, name="sharing")
            if not isinstance(sharing, str):
                raise CoopSinglePhaseRewriteError(
                    "TempStorage sharing must be a string: 'shared' or 'exclusive'."
                )
            sharing = sharing.strip().lower()
        if sharing not in {"shared", "exclusive"}:
            raise CoopSinglePhaseRewriteError(
                "TempStorage sharing must be 'shared' or 'exclusive'."
            )

        return _TempStorageCtorSpec(
            size_in_bytes=size_in_bytes,
            alignment=alignment,
            auto_sync=auto_sync,
            sharing=sharing,
        )

    def _collect_temp_storage_ctor_keys(
        self, value: ir.Var, seen: set[str]
    ) -> set[str]:
        if not isinstance(value, ir.Var):
            return set()
        if value.name in seen:
            return set()
        if value.name in self._temp_storage_ctor_specs:
            return {value.name}

        seen.add(value.name)
        keys: set[str] = set()
        for definition in self._lookup_definitions(value):
            if isinstance(definition, ir.Expr):
                if definition.op == "call" and self._is_temp_storage_ctor_call(
                    definition
                ):
                    spec = self._extract_temp_storage_ctor_spec(definition)
                    self._temp_storage_ctor_specs[value.name] = (
                        self._merge_temp_storage_ctor_specs(
                            self._temp_storage_ctor_specs.get(value.name),
                            spec,
                        )
                    )
                    keys.add(value.name)
                    continue
                if definition.op == "cast":
                    cast_value = getattr(definition, "value", None)
                    if isinstance(cast_value, ir.Var):
                        keys.update(
                            self._collect_temp_storage_ctor_keys(cast_value, seen)
                        )
                    continue
                if definition.op == "phi":
                    for incoming in _phi_incoming_values(definition):
                        if isinstance(incoming, ir.Var):
                            keys.update(
                                self._collect_temp_storage_ctor_keys(incoming, seen)
                            )
                    continue
            if isinstance(definition, ir.Var):
                keys.update(self._collect_temp_storage_ctor_keys(definition, seen))

        return keys

    def _resolve_temp_storage_ctor_key(self, value: ir.Var) -> str | None:
        if not isinstance(value, ir.Var):
            return None
        keys = self._collect_temp_storage_ctor_keys(value, seen=set())
        if not keys:
            return None
        if len(keys) == 1:
            return next(iter(keys))

        merged_spec: _TempStorageCtorSpec | None = None
        ordered_keys = sorted(
            keys,
            key=lambda key: (self._temp_storage_ctor_order.get(key, 1 << 30), key),
        )
        for key in ordered_keys:
            spec = self._temp_storage_ctor_specs.get(key)
            if spec is None:
                continue
            merged_spec = self._merge_temp_storage_ctor_specs(merged_spec, spec)
        if merged_spec is None:
            return None

        # Materialize merged metadata on a stable canonical root key.
        canonical_key = ordered_keys[0]
        self._temp_storage_ctor_specs[canonical_key] = merged_spec
        alias_orders = [
            self._temp_storage_ctor_order[key]
            for key in ordered_keys
            if key in self._temp_storage_ctor_order
        ]
        if alias_orders:
            self._temp_storage_ctor_order[canonical_key] = min(alias_orders)
        return canonical_key

    def _resolve_temp_storage_ctor_spec(
        self, value: ir.Var
    ) -> _TempStorageCtorSpec | None:
        key = self._resolve_temp_storage_ctor_key(value)
        if key is None:
            return None
        return self._temp_storage_ctor_specs.get(key)

    def _resolve_temp_storage_plan(self, value: ir.Var) -> _TempStoragePlan | None:
        key = self._resolve_temp_storage_ctor_key(value)
        if key is None:
            return None
        if self._temp_storage_global_plan is None and self._temp_storage_ctor_specs:
            self._ensure_temp_storage_global_plan()
        return self._finalize_temp_storage_plan_for_var(key)

    def _finalize_temp_storage_plan_for_var(self, var_name: str) -> _TempStoragePlan:
        cached = self._temp_storage_plans.get(var_name)
        if cached is not None:
            return cached

        ctor_spec = self._temp_storage_ctor_specs.get(var_name)
        if ctor_spec is None:
            raise CoopSinglePhaseRewriteError(
                f"Missing TempStorage constructor metadata for variable '{var_name}'."
            )

        requirements = self._func_temp_storage_requirements.get(var_name)
        uses = list(requirements.uses) if requirements is not None else []
        uses.sort(key=lambda entry: entry.order)

        required_alignment = (
            max(_MIN_TEMP_STORAGE_ALIGNMENT, *(entry.alignment for entry in uses))
            if uses
            else max(_MIN_TEMP_STORAGE_ALIGNMENT, int(ctor_spec.alignment or 1))
        )

        slices_by_call_id: dict[int, _TempStorageSlice] = {}
        if ctor_spec.sharing == "shared":
            required_size = max((entry.size_in_bytes for entry in uses), default=0)
            for entry in uses:
                slices_by_call_id[id(entry.call_assign)] = _TempStorageSlice(
                    offset=0,
                    size_in_bytes=entry.size_in_bytes,
                )
        else:
            required_size = 0
            for entry in uses:
                required_size = _align_up(required_size, max(1, int(entry.alignment)))
                slices_by_call_id[id(entry.call_assign)] = _TempStorageSlice(
                    offset=required_size,
                    size_in_bytes=entry.size_in_bytes,
                )
                required_size += entry.size_in_bytes

        if ctor_spec.size_in_bytes is None:
            if required_size <= 0:
                raise CoopSinglePhaseRewriteError(
                    "Could not infer TempStorage size_in_bytes from primitive uses; "
                    "pass an explicit size_in_bytes."
                )
            size_in_bytes = required_size
        else:
            size_in_bytes = int(ctor_spec.size_in_bytes)
        if size_in_bytes <= 0:
            raise CoopSinglePhaseRewriteError(
                "TempStorage size_in_bytes must be a positive integer."
            )
        if required_size > 0 and size_in_bytes < required_size:
            raise CoopSinglePhaseRewriteError(
                "TempStorage size_in_bytes is smaller than required by primitive "
                f"uses ({size_in_bytes} < {required_size})."
            )

        if ctor_spec.alignment is None:
            alignment = _default_temp_storage_alignment(required_alignment)
        else:
            alignment = int(ctor_spec.alignment)
        _validate_temp_storage_alignment(alignment)
        if required_alignment > 0 and alignment < required_alignment:
            raise CoopSinglePhaseRewriteError(
                "TempStorage alignment is smaller than required by primitive uses "
                f"({alignment} < {required_alignment})."
            )

        if ctor_spec.sharing == "exclusive":
            if ctor_spec.auto_sync is True:
                raise CoopSinglePhaseRewriteError(
                    "TempStorage with sharing='exclusive' does not support "
                    "auto_sync=True."
                )
            auto_sync = False
        else:
            auto_sync = True if ctor_spec.auto_sync is None else ctor_spec.auto_sync

        plan = _TempStoragePlan(
            size_in_bytes=size_in_bytes,
            alignment=alignment,
            sharing=ctor_spec.sharing,
            auto_sync=auto_sync,
            slices_by_call_id=slices_by_call_id,
        )
        self._temp_storage_plans[var_name] = plan
        return plan

    def _is_local_array_ctor_call(self, call: ir.Expr) -> bool:
        chain = self._resolve_attribute_chain(call.func)
        if chain is None:
            return False
        root, attrs = chain
        root_name = getattr(root, "__name__", "")
        if attrs == ["local", "array"] and (
            root_name in self._CUDA_ROOT_MODULES or root_name == "cuda.coop.numba_mlir"
        ):
            return True
        if attrs == ["array"] and root_name in {
            "numba_cuda_mlir.cuda.local",
            "cuda.local",
        }:
            return True
        return False

    def _extract_local_array_spec(self, call: ir.Expr) -> _ThreadDataSpec:
        kw_map = {name: value for name, value in call.kws}

        items_ref = None
        if call.args:
            items_ref = call.args[0]
        elif "shape" in kw_map:
            items_ref = kw_map["shape"]

        dtype_ref = None
        if len(call.args) >= 2:
            dtype_ref = call.args[1]
        elif "dtype" in kw_map:
            dtype_ref = kw_map["dtype"]

        items_per_thread = None
        if items_ref is not None:
            items_per_thread = self._extract_1d_extent_literal(items_ref)

        dtype = None
        if dtype_ref is not None:
            dtype = self._resolve_dtype_ref(dtype_ref)

        return _ThreadDataSpec(items_per_thread=items_per_thread, dtype=dtype)

    def _is_shared_array_ctor_call(self, call: ir.Expr) -> bool:
        chain = self._resolve_attribute_chain(call.func)
        if chain is None:
            return False
        root, attrs = chain
        root_name = getattr(root, "__name__", "")
        if attrs == ["shared", "array"] and (
            root_name in self._CUDA_ROOT_MODULES or root_name == "cuda.coop.numba_mlir"
        ):
            return True
        if attrs == ["array"] and root_name in {
            "numba_cuda_mlir.cuda.shared",
            "cuda.shared",
        }:
            return True
        return False

    def _extract_shared_array_spec(self, call: ir.Expr) -> _ThreadDataSpec:
        kw_map = {name: value for name, value in call.kws}

        shape_ref = None
        if call.args:
            shape_ref = call.args[0]
        elif "shape" in kw_map:
            shape_ref = kw_map["shape"]

        dtype_ref = None
        if len(call.args) >= 2:
            dtype_ref = call.args[1]
        elif "dtype" in kw_map:
            dtype_ref = kw_map["dtype"]

        extent = None
        if shape_ref is not None:
            extent = self._extract_1d_extent_literal(shape_ref)

        dtype = None
        if dtype_ref is not None:
            dtype = self._resolve_dtype_ref(dtype_ref)

        return _ThreadDataSpec(items_per_thread=extent, dtype=dtype)

    def _resolve_array_spec_from_var(
        self, value: ir.Var, seen: set[str]
    ) -> _ThreadDataSpec | None:
        if not isinstance(value, ir.Var):
            return None
        if value.name in seen:
            return None

        seen.add(value.name)
        merged: _ThreadDataSpec | None = None
        for definition in self._lookup_definitions(value):
            candidate: _ThreadDataSpec | None = None
            if isinstance(definition, ir.Expr):
                if definition.op == "call":
                    if self._is_thread_data_ctor_call(definition):
                        candidate = self._extract_thread_data_spec(definition)
                    elif self._is_typed_group_payload_ctor_call(definition):
                        candidate = self._extract_typed_group_payload_spec(
                            definition,
                            seen=seen,
                        )
                    elif self._is_local_array_ctor_call(definition):
                        candidate = self._extract_local_array_spec(definition)
                    elif self._is_shared_array_ctor_call(definition):
                        candidate = self._extract_shared_array_spec(definition)
                elif definition.op == "cast":
                    cast_value = getattr(definition, "value", None)
                    if isinstance(cast_value, ir.Var):
                        candidate = self._resolve_array_spec_from_var(cast_value, seen)
                elif definition.op == "static_getitem":
                    for item in self._resolve_static_tuple_item_vars(definition):
                        item_spec = self._resolve_array_spec_from_var(item, set(seen))
                        if item_spec is None:
                            continue
                        merged = self._merge_thread_data_specs(merged, item_spec)
                    continue
                elif definition.op == "phi":
                    for incoming in _phi_incoming_values(definition):
                        if not isinstance(incoming, ir.Var):
                            continue
                        incoming_spec = self._resolve_array_spec_from_var(
                            incoming, set(seen)
                        )
                        if incoming_spec is None:
                            continue
                        merged = self._merge_thread_data_specs(merged, incoming_spec)
                    continue
            elif isinstance(definition, ir.Var):
                candidate = self._resolve_array_spec_from_var(definition, seen)

            if candidate is not None:
                merged = self._merge_thread_data_specs(merged, candidate)

        return merged

    def _resolve_array_extent(self, value: ir.Var):
        spec = self._resolve_array_spec_from_var(value, seen=set())
        if spec is not None:
            return spec.items_per_thread
        return None

    def _resolve_array_dtype(self, value: ir.Var):
        dtype = self._resolve_var_dtype(value)
        if dtype is not None:
            return dtype
        spec = self._resolve_array_spec_from_var(value, seen=set())
        if spec is not None:
            return spec.dtype
        return None

    def _resolve_thread_data_spec_from_var(
        self, value: ir.Var, seen: set[str]
    ) -> _ThreadDataSpec | None:
        if not isinstance(value, ir.Var):
            return None
        cached = self._thread_data_specs.get(value.name)
        if (
            cached is not None
            and cached.items_per_thread is not None
            and cached.dtype is not None
        ):
            return cached
        if value.name in seen:
            return None

        seen.add(value.name)
        merged: _ThreadDataSpec | None = cached
        for definition in self._lookup_definitions(value):
            candidate: _ThreadDataSpec | None = None
            if isinstance(definition, ir.Expr):
                if definition.op == "call":
                    if self._is_thread_data_ctor_call(definition):
                        candidate = self._extract_thread_data_spec(definition)
                    elif self._is_typed_group_payload_ctor_call(definition):
                        candidate = self._extract_typed_group_payload_spec(
                            definition,
                            seen=seen,
                        )
                    elif self._is_local_array_ctor_call(definition):
                        candidate = self._extract_local_array_spec(definition)
                elif definition.op == "cast":
                    cast_value = getattr(definition, "value", None)
                    if isinstance(cast_value, ir.Var):
                        candidate = self._resolve_thread_data_spec_from_var(
                            cast_value, seen
                        )
                elif definition.op == "static_getitem":
                    for item in self._resolve_static_tuple_item_vars(definition):
                        item_spec = self._resolve_thread_data_spec_from_var(
                            item, set(seen)
                        )
                        if item_spec is None:
                            continue
                        merged = self._merge_thread_data_specs(merged, item_spec)
                    continue
                elif definition.op == "phi":
                    for incoming in _phi_incoming_values(definition):
                        if not isinstance(incoming, ir.Var):
                            continue
                        incoming_spec = self._resolve_thread_data_spec_from_var(
                            incoming, set(seen)
                        )
                        if incoming_spec is None:
                            continue
                        merged = self._merge_thread_data_specs(merged, incoming_spec)
                    continue
            elif isinstance(definition, ir.Var):
                candidate = self._resolve_thread_data_spec_from_var(definition, seen)

            if candidate is not None:
                merged = self._merge_thread_data_specs(merged, candidate)

        if merged is not None:
            self._thread_data_specs[value.name] = merged
        return merged

    def _resolve_static_tuple_item_vars(self, definition: ir.Expr) -> list[ir.Var]:
        index = getattr(definition, "index", None)
        tuple_value = getattr(definition, "value", None)
        if not isinstance(index, int) or not isinstance(tuple_value, ir.Var):
            return []
        return self._resolve_tuple_item_vars(tuple_value, index, seen=set())

    def _resolve_tuple_item_vars(
        self, tuple_value: ir.Var, index: int, seen: set[str]
    ) -> list[ir.Var]:
        if tuple_value.name in seen:
            return []
        seen.add(tuple_value.name)
        items: list[ir.Var] = []
        for tuple_definition in self._lookup_definitions(tuple_value):
            if isinstance(tuple_definition, ir.Var):
                items.extend(
                    self._resolve_tuple_item_vars(
                        tuple_definition, index, seen=set(seen)
                    )
                )
                continue
            if not isinstance(tuple_definition, ir.Expr):
                continue
            if tuple_definition.op == "build_tuple":
                tuple_items = tuple(getattr(tuple_definition, "items", ()))
                if -len(tuple_items) <= index < len(tuple_items):
                    item = tuple_items[index]
                    if isinstance(item, ir.Var):
                        items.append(item)
                continue
            if tuple_definition.op in {"cast", "exhaust_iter"}:
                source = getattr(tuple_definition, "value", None)
                if isinstance(source, ir.Var):
                    items.extend(
                        self._resolve_tuple_item_vars(source, index, seen=set(seen))
                    )
                continue
            if tuple_definition.op == "phi":
                for incoming in _phi_incoming_values(tuple_definition):
                    if isinstance(incoming, ir.Var):
                        items.extend(
                            self._resolve_tuple_item_vars(
                                incoming,
                                index,
                                seen=set(seen),
                            )
                        )
        return items

    def _resolve_thread_data_spec(self, value: ir.Var) -> _ThreadDataSpec | None:
        if not isinstance(value, ir.Var):
            return None
        return self._resolve_thread_data_spec_from_var(value, seen=set())

    def _resolve_var_numba_type(self, value: ir.Var):
        typemap = getattr(self._state, "typemap", None)
        if isinstance(typemap, dict):
            mapped = typemap.get(value.name)
            if mapped is not None:
                return mapped

        if value.name in self._arg_type_map:
            return self._arg_type_map[value.name]

        definition = self._lookup_definition(value)
        if isinstance(definition, ir.Arg):
            arg_types = tuple(getattr(self._state, "args", ()) or ())
            if 0 <= definition.index < len(arg_types):
                return arg_types[definition.index]

        return None

    def _resolve_call_result_dtype(self, definition: ir.Expr):
        func_obj = None
        func_ref = getattr(definition, "func", None)
        if isinstance(func_ref, ir.Var):
            try:
                func_obj = self._infer_constant(func_ref)
            except _INFERENCE_EXCEPTIONS:
                func_obj = None
            if func_obj is None:
                func_def = self._lookup_definition(func_ref)
                if isinstance(func_def, (ir.Global, ir.FreeVar, ir.Const)):
                    func_obj = func_def.value
        elif isinstance(func_ref, (ir.Global, ir.FreeVar, ir.Const)):
            func_obj = func_ref.value

        if func_obj is None:
            chain = self._resolve_attribute_chain(func_ref)
            if chain is not None:
                root, attrs = chain
                obj = root
                try:
                    for attr in attrs:
                        obj = getattr(obj, attr)
                    func_obj = obj
                except _INFERENCE_EXCEPTIONS:
                    func_obj = None

        if func_obj is None:
            return None

        try:
            return np.dtype(func_obj).type
        except (TypeError, ValueError):
            return None

    def _infer_thread_data_dtype_from_writes(self, value: ir.Var):
        spec = self._resolve_thread_data_spec(value)
        if spec is None:
            return None

        alias_names = {value.name}
        changed = True
        while changed:
            changed = False
            for block in self._func_ir.blocks.values():
                for stmt in block.body:
                    if not isinstance(stmt, ir.Assign):
                        continue
                    definition = stmt.value
                    sources: tuple[ir.Var, ...] = ()
                    if isinstance(definition, ir.Var):
                        sources = (definition,)
                    elif isinstance(definition, ir.Expr) and definition.op == "cast":
                        if isinstance(definition.value, ir.Var):
                            sources = (definition.value,)
                    elif isinstance(definition, ir.Expr) and definition.op == "phi":
                        sources = tuple(
                            incoming
                            for incoming in _phi_incoming_values(definition)
                            if isinstance(incoming, ir.Var)
                        )
                    source_names = {source.name for source in sources}
                    if stmt.target.name in alias_names or source_names & alias_names:
                        additions = {stmt.target.name, *source_names} - alias_names
                        if additions:
                            alias_names.update(additions)
                            changed = True

        inferred = None
        static_setitem_cls = getattr(ir, "StaticSetItem", None)

        for block in self._func_ir.blocks.values():
            for stmt in block.body:
                if isinstance(stmt, ir.SetItem) or (
                    static_setitem_cls is not None
                    and isinstance(stmt, static_setitem_cls)
                ):
                    target = getattr(stmt, "target", None)
                    rhs = getattr(stmt, "value", None)
                else:
                    continue

                if not isinstance(target, ir.Var) or target.name not in alias_names:
                    continue
                if not isinstance(rhs, ir.Var):
                    continue

                rhs_dtype = self._resolve_var_dtype(rhs)
                if rhs_dtype is None:
                    continue
                if inferred is None:
                    inferred = rhs_dtype
                    continue
                if inferred != rhs_dtype:
                    raise CoopSinglePhaseRewriteError(
                        "Failed to infer a consistent dtype from coop.ThreadData writes."
                    )

        if inferred is not None:
            self._record_inferred_thread_data_dtype(value, inferred)
        return inferred

    def _collect_thread_data_write_roots(
        self,
        value: ir.Var,
        seen: set[str] | None = None,
    ) -> dict[str, ir.Var]:
        """Find concrete ThreadData constructors behind group payload markers."""
        if not isinstance(value, ir.Var):
            return {}
        if seen is None:
            seen = set()
        if value.name in seen:
            return {}
        seen.add(value.name)

        roots: dict[str, ir.Var] = {}
        for definition in self._lookup_definitions(value):
            sources: tuple[ir.Var, ...] = ()
            if isinstance(definition, ir.Var):
                sources = (definition,)
            elif isinstance(definition, ir.Expr):
                if definition.op == "call":
                    if self._is_thread_data_ctor_call(definition):
                        roots[value.name] = value
                        continue
                    if (
                        self._is_typed_group_payload_ctor_call(definition)
                        and definition.args
                    ):
                        prototype = definition.args[0]
                        if isinstance(prototype, ir.Var):
                            sources = (prototype,)
                elif definition.op in {"cast", "exhaust_iter"}:
                    source = getattr(definition, "value", None)
                    if isinstance(source, ir.Var):
                        sources = (source,)
                elif definition.op == "phi":
                    sources = tuple(
                        incoming
                        for incoming in _phi_incoming_values(definition)
                        if isinstance(incoming, ir.Var)
                    )
                elif definition.op == "static_getitem":
                    sources = tuple(self._resolve_static_tuple_item_vars(definition))

            for source in sources:
                roots.update(
                    self._collect_thread_data_write_roots(source, seen=set(seen))
                )
        return roots

    def _infer_thread_data_dtype_from_provenance_writes(self, value: ir.Var):
        """Infer dtype from writes without re-entering marker spec resolution."""
        inferred = None
        roots = self._collect_thread_data_write_roots(value)
        for root_name in sorted(roots):
            root = roots[root_name]
            spec = self._resolve_thread_data_spec(root)
            root_dtype = spec.dtype if spec is not None else None
            if root_dtype is None:
                root_dtype = self._resolve_var_dtype(root)
            if root_dtype is None:
                root_dtype = self._infer_thread_data_dtype_from_writes(root)
            if root_dtype is None:
                continue
            if inferred is not None and not _dtype_values_match(inferred, root_dtype):
                raise CoopSinglePhaseRewriteError(
                    "Inconsistent inferred dtype across coop.ThreadData "
                    "payload provenance."
                )
            inferred = root_dtype
        return inferred

    def _resolve_var_dtype(self, value: ir.Var, seen: set[str] | None = None):
        if seen is None:
            seen = set()
        if value.name in seen:
            return None
        seen.add(value.name)

        spec = self._resolve_thread_data_spec(value)
        if spec is not None and spec.dtype is not None:
            return spec.dtype

        var_type = self._resolve_var_numba_type(value)
        dtype = getattr(var_type, "dtype", None)
        if dtype is not None:
            return dtype
        if var_type is not None and hasattr(var_type, "bitwidth"):
            return var_type

        definition = self._lookup_definition(value)
        if isinstance(definition, ir.Const):
            try:
                return _numba_typeof(definition.value)
            except (TypeError, ValueError):
                return None
        if isinstance(definition, ir.Var):
            return self._resolve_var_dtype(definition, seen)
        if isinstance(definition, ir.Expr) and definition.op == "getattr":
            if definition.attr == "dtype" and isinstance(definition.value, ir.Var):
                return self._resolve_var_dtype(definition.value, seen)
        if isinstance(definition, ir.Expr) and definition.op in {
            "getitem",
            "static_getitem",
        }:
            base_value = getattr(definition, "value", None)
            if isinstance(base_value, ir.Var):
                return self._resolve_var_dtype(base_value, seen)
        if isinstance(definition, ir.Expr) and definition.op in {
            "binop",
            "inplace_binop",
        }:
            lhs = getattr(definition, "lhs", None)
            rhs = getattr(definition, "rhs", None)
            lhs_dtype = (
                self._resolve_var_dtype(lhs, seen) if isinstance(lhs, ir.Var) else None
            )
            rhs_dtype = (
                self._resolve_var_dtype(rhs, seen) if isinstance(rhs, ir.Var) else None
            )
            if (
                lhs_dtype is not None
                and rhs_dtype is not None
                and lhs_dtype != rhs_dtype
            ):
                return None
            return lhs_dtype if lhs_dtype is not None else rhs_dtype
        if isinstance(definition, ir.Expr) and definition.op == "unary":
            unary_value = getattr(definition, "value", None)
            if isinstance(unary_value, ir.Var):
                return self._resolve_var_dtype(unary_value, seen)
        if isinstance(definition, ir.Expr) and definition.op == "phi":
            inferred = None
            for incoming in _phi_incoming_values(definition):
                if not isinstance(incoming, ir.Var):
                    continue
                incoming_dtype = self._resolve_var_dtype(incoming, seen)
                if incoming_dtype is None:
                    continue
                if inferred is None:
                    inferred = incoming_dtype
                    continue
                if inferred != incoming_dtype:
                    return None
            if inferred is not None:
                return inferred
        if isinstance(definition, ir.Expr) and definition.op == "call":
            call_dtype = self._resolve_call_result_dtype(definition)
            if call_dtype is not None:
                return call_dtype

        return None

    def _resolve_dtype_ref(self, value_ref):
        try:
            return self._infer_constant(value_ref)
        except _INFERENCE_EXCEPTIONS:
            pass

        if isinstance(value_ref, ir.Var):
            definition = self._lookup_definition(value_ref)
            if isinstance(definition, ir.Expr) and definition.op == "getattr":
                if definition.attr == "dtype" and isinstance(definition.value, ir.Var):
                    return self._resolve_var_dtype(definition.value)
            return self._resolve_var_dtype(value_ref)

        return None

    def _resolve_factory_kwarg_value(self, name: str, value_ref):
        try:
            return self._infer_constant(value_ref)
        except _INFERENCE_EXCEPTIONS:
            pass

        if name in {"dtype", "key_dtype", "value_dtype", "keys", "values"}:
            dtype = self._resolve_dtype_ref(value_ref)
            if dtype is not None:
                return dtype

        return _UNRESOLVED

    def _resolve_call_target(self, call: ir.Expr):
        factory = self._resolve_factory_from_var(call.func)
        if factory is not None:
            return _ResolvedCallTarget(
                factory=factory,
                func_var_name=call.func.name,
                func_var_name_extra=None,
                getitem_temp_storage=None,
            )

        func_def = self._lookup_definition(call.func)
        if not (
            isinstance(func_def, ir.Expr)
            and func_def.op in {"getitem", "static_getitem"}
        ):
            return None

        factory = self._resolve_factory_from_var(func_def.value)
        if factory is None:
            return None

        getitem_temp_storage = getattr(func_def, "index", None)
        if not isinstance(getitem_temp_storage, ir.Var):
            getitem_temp_storage = getattr(func_def, "index_var", None)
        if not isinstance(getitem_temp_storage, ir.Var):
            raise CoopSinglePhaseRewriteError(
                "coop single-phase getitem syntax expects a runtime temp-storage variable: "
                f"'{factory.__name__}[temp_storage](...)'."
            )

        return _ResolvedCallTarget(
            factory=factory,
            func_var_name=call.func.name,
            func_var_name_extra=func_def.value.name,
            getitem_temp_storage=getitem_temp_storage,
        )

    @staticmethod
    def _canonical_parent_ctor_spec(
        parent_kind: str,
        factory_kwargs: dict[str, object],
        captured_vars: dict[str, str],
    ) -> _ParentCtorSpec:
        return _ParentCtorSpec(
            parent_kind=parent_kind,
            factory_kwargs_items=tuple(
                sorted((name, value) for name, value in factory_kwargs.items())
            ),
            captured_var_items=tuple(
                sorted((name, value) for name, value in captured_vars.items())
            ),
        )

    @staticmethod
    def _merge_parent_ctor_specs(
        existing: _ParentCtorSpec | None, observed: _ParentCtorSpec
    ) -> _ParentCtorSpec:
        if existing is None:
            return observed
        if existing != observed:
            raise CoopSinglePhaseRewriteError(
                "Inconsistent parent constructor metadata across merged aliases."
            )
        return existing

    def _resolve_threads_per_block_for_parent(self):
        if "threads_per_block" in self._compiletime_context:
            return self._compiletime_context["threads_per_block"]
        return self._infer_threads_per_block_from_targetoptions()

    def _can_defer_launch_dim_inference(self) -> bool:
        # Device callees compiled for inlining can reach the rewrite pass
        # before the caller launch metadata is available. Leave those calls
        # intact so the caller's inlined IR can be rewritten.
        should_defer = self._infer_threads_per_block_from_targetoptions() is None
        self._deferred_launch_dim_inference |= should_defer
        return should_defer

    def _resolve_parent_constructor_spec(self, call: ir.Expr) -> _ParentCtorSpec | None:
        factory = self._resolve_parent_factory_from_var(call.func)
        if factory is None:
            return None

        parent_kind = factory.__name__
        if parent_kind == "histogram":
            if call.vararg is not None or call.varkwarg is not None:
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase parent constructors do not support *args/**kwargs."
                )

            if len(call.args) not in {2, 3}:
                raise CoopSinglePhaseRewriteError(
                    "coop._block.histogram(...) expects two or three positional "
                    "arguments: items, histogram, and optional algorithm."
                )
            items_var = call.args[0]
            histogram_var = call.args[1]
            if not isinstance(items_var, ir.Var) or not isinstance(
                histogram_var, ir.Var
            ):
                raise CoopSinglePhaseRewriteError(
                    "coop._block.histogram(...) expects variable arguments for "
                    "items/histogram."
                )

            allowed_factory_kwargs = self._PARENT_SPECS[parent_kind][
                "allowed_factory_kwargs"
            ]
            seen_factory_kwargs: set[str] = set()
            factory_kwargs: dict[str, object] = {}
            if len(call.args) == 3:
                seen_factory_kwargs.add("algorithm")
                value = self._resolve_factory_kwarg_value("algorithm", call.args[2])
                if value is _UNRESOLVED:
                    raise CoopSinglePhaseRewriteError(
                        "Failed to evaluate coop._block.histogram factory argument "
                        "'algorithm' as a compile-time constant."
                    )
                factory_kwargs["algorithm"] = value
            for name, value_var in call.kws:
                if name not in allowed_factory_kwargs:
                    allowed = ", ".join(sorted(allowed_factory_kwargs))
                    raise CoopSinglePhaseRewriteError(
                        "Unsupported coop single-phase parent factory keyword "
                        f"'{name}' for 'histogram'. Allowed keywords are: {allowed}."
                    )
                if name in seen_factory_kwargs:
                    raise CoopSinglePhaseRewriteError(
                        "Duplicate coop single-phase parent factory keyword "
                        f"'{name}' for 'histogram'."
                    )
                seen_factory_kwargs.add(name)
                value = self._resolve_factory_kwarg_value(name, value_var)
                if value is _UNRESOLVED:
                    raise CoopSinglePhaseRewriteError(
                        "Failed to evaluate coop._block.histogram factory argument "
                        f"'{name}' as a compile-time constant."
                    )
                factory_kwargs[name] = value

            items_per_thread = self._resolve_array_extent(items_var)
            if items_per_thread is None:
                raise CoopSinglePhaseRewriteError(
                    "Failed to infer histogram items_per_thread from the items array."
                )

            bins = self._resolve_array_extent(histogram_var)
            if bins is None:
                raise CoopSinglePhaseRewriteError(
                    "Failed to infer histogram bins from the histogram array shape."
                )

            item_dtype = self._resolve_array_dtype(items_var)
            if item_dtype is None:
                raise CoopSinglePhaseRewriteError(
                    "Failed to infer histogram item dtype from the items array."
                )

            counter_dtype = self._resolve_array_dtype(histogram_var)
            if counter_dtype is None:
                raise CoopSinglePhaseRewriteError(
                    "Failed to infer histogram counter dtype from the histogram array."
                )

            common_profile_operation = factory_kwargs.pop(
                "_common_profile_operation", None
            )
            if common_profile_operation is not None:
                if common_profile_operation != "histogram":
                    raise CoopSinglePhaseRewriteError(
                        "_common_profile_operation does not match the rewritten "
                        "group operation"
                    )
                from ._common import _validate_common_histogram_dtypes

                try:
                    item_dtype, counter_dtype = _validate_common_histogram_dtypes(
                        item_dtype,
                        counter_dtype,
                    )
                except (TypeError, ValueError) as exc:
                    raise CoopSinglePhaseRewriteError(str(exc)) from exc

            threads_per_block = self._resolve_threads_per_block_for_parent()
            if threads_per_block is None:
                if self._can_defer_launch_dim_inference():
                    raise _DeferredCoopRewrite
                raise CoopSinglePhaseRewriteError(
                    "Failed to infer threads_per_block for coop._block.histogram; "
                    "provide launch-bounds metadata or launch-config-specialized "
                    "compilation context."
                )

            factory_kwargs.update(
                {
                    "item_dtype": item_dtype,
                    "counter_dtype": counter_dtype,
                    "threads_per_block": threads_per_block,
                    "items_per_thread": int(items_per_thread),
                    "bins": int(bins),
                }
            )

            captured_vars = {
                "items": items_var.name,
                "histogram": histogram_var.name,
            }
            return self._canonical_parent_ctor_spec(
                parent_kind=parent_kind,
                factory_kwargs=factory_kwargs,
                captured_vars=captured_vars,
            )

        if parent_kind in {
            "run_length",
            "_common_run_length",
            "_qualified_group_run_length",
        }:
            common_v1 = parent_kind == "_common_run_length"
            scope_name = "cuda.coop" if common_v1 else "cuda.coop.numba_mlir"
            group_first = parent_kind != "run_length"
            if call.vararg is not None or call.varkwarg is not None:
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase parent constructors do not support *args/**kwargs."
                )
            if len(call.args) not in {5, 6}:
                raise CoopSinglePhaseRewriteError(
                    "coop._block.run_length(...) expects five or six positional arguments: "
                    "run_values, run_lengths, runs_per_thread, "
                    "decoded_items_per_thread, total_decoded_size, and optional "
                    "decoded_offset_dtype."
                )

            run_values_var = call.args[0]
            run_lengths_var = call.args[1]
            runs_per_thread_ref = call.args[2]
            decoded_items_per_thread_ref = call.args[3]
            total_decoded_size_var = call.args[4]
            if not (
                isinstance(run_values_var, ir.Var)
                and isinstance(run_lengths_var, ir.Var)
                and isinstance(total_decoded_size_var, ir.Var)
            ):
                raise CoopSinglePhaseRewriteError(
                    "coop._block.run_length(...) expects array arguments as variables "
                    "(run_values, run_lengths, total_decoded_size)."
                )

            allowed_factory_kwargs = self._PARENT_SPECS[parent_kind][
                "allowed_factory_kwargs"
            ]
            seen_factory_kwargs: set[str] = set()
            factory_kwargs: dict[str, object] = {}
            temp_storage_var = None
            if len(call.args) == 6:
                seen_factory_kwargs.add("decoded_offset_dtype")
                value = self._resolve_factory_kwarg_value(
                    "decoded_offset_dtype", call.args[5]
                )
                if value is _UNRESOLVED:
                    raise CoopSinglePhaseRewriteError(
                        "Failed to evaluate coop._block.run_length factory argument "
                        "'decoded_offset_dtype' as a compile-time constant."
                    )
                factory_kwargs["decoded_offset_dtype"] = value
            for name, value_var in call.kws:
                if name not in allowed_factory_kwargs:
                    allowed = ", ".join(sorted(allowed_factory_kwargs))
                    raise CoopSinglePhaseRewriteError(
                        "Unsupported coop single-phase parent factory keyword "
                        f"'{name}' for 'run_length'. Allowed keywords are: {allowed}."
                    )
                if name in seen_factory_kwargs:
                    raise CoopSinglePhaseRewriteError(
                        "Duplicate coop single-phase parent factory keyword "
                        f"'{name}' for 'run_length'."
                    )
                seen_factory_kwargs.add(name)
                if name == "temp_storage":
                    if not isinstance(value_var, ir.Var):
                        raise CoopSinglePhaseRewriteError(
                            "coop._block.run_length temp_storage must be provided "
                            "as a variable."
                        )
                    temp_storage_var = value_var
                    continue
                value = self._resolve_factory_kwarg_value(name, value_var)
                if value is _UNRESOLVED:
                    raise CoopSinglePhaseRewriteError(
                        "Failed to evaluate coop._block.run_length factory argument "
                        f"'{name}' as a compile-time constant."
                    )
                factory_kwargs[name] = value

            try:
                runs_per_thread = normalize_positive_int(
                    "runs_per_thread",
                    self._infer_constant(runs_per_thread_ref),
                )
            except _INFERENCE_EXCEPTIONS as e:
                raise CoopSinglePhaseRewriteError(
                    "coop._block.run_length runs_per_thread must be a "
                    "compile-time positive integer."
                ) from e
            try:
                decoded_items_per_thread = normalize_positive_int(
                    "decoded_items_per_thread",
                    self._infer_constant(decoded_items_per_thread_ref),
                )
            except _INFERENCE_EXCEPTIONS as e:
                raise CoopSinglePhaseRewriteError(
                    "coop._block.run_length decoded_items_per_thread must be a "
                    "compile-time positive integer."
                ) from e

            run_values_extent = self._resolve_array_extent(run_values_var)
            if (
                run_values_extent is not None
                and int(run_values_extent) != runs_per_thread
            ):
                raise CoopSinglePhaseRewriteError(
                    "run_length run_values shape does not match runs_per_thread."
                )

            run_lengths_extent = self._resolve_array_extent(run_lengths_var)
            if (
                run_lengths_extent is not None
                and int(run_lengths_extent) != runs_per_thread
            ):
                raise CoopSinglePhaseRewriteError(
                    "run_length run_lengths shape does not match runs_per_thread."
                )

            total_decoded_size_extent = self._resolve_array_extent(
                total_decoded_size_var
            )
            if (
                total_decoded_size_extent is not None
                and int(total_decoded_size_extent) != 1
            ):
                raise CoopSinglePhaseRewriteError(
                    "run_length total_decoded_size must be a single-element array."
                )

            item_dtype = self._resolve_array_dtype(run_values_var)
            if item_dtype is None:
                raise CoopSinglePhaseRewriteError(
                    "Failed to infer run_length item dtype from run_values."
                )

            run_length_dtype = self._resolve_array_dtype(run_lengths_var)
            if run_length_dtype is None:
                raise CoopSinglePhaseRewriteError(
                    "Failed to infer run_length length dtype from run_lengths."
                )

            total_decoded_size_dtype = self._resolve_array_dtype(total_decoded_size_var)
            if total_decoded_size_dtype is None:
                raise CoopSinglePhaseRewriteError(
                    "Failed to infer run_length total_decoded_size dtype."
                )

            decoded_offset_dtype = factory_kwargs.get(
                "decoded_offset_dtype", total_decoded_size_dtype
            )
            if decoded_offset_dtype is None:
                raise CoopSinglePhaseRewriteError(
                    "Failed to infer run_length decoded_offset_dtype."
                )

            if common_v1:
                from ._common import (
                    _validate_common_run_length_decode_dtypes,
                )

                try:
                    item_dtype, run_length_dtype = (
                        _validate_common_run_length_decode_dtypes(
                            item_dtype,
                            run_length_dtype,
                        )
                    )
                except (TypeError, ValueError) as exc:
                    raise CoopSinglePhaseRewriteError(str(exc)) from exc
            static_decoded_window_offset = factory_kwargs.get(
                "_static_decoded_window_offset",
                None,
            )
            if group_first and static_decoded_window_offset is not None:
                bitwidth = getattr(run_length_dtype, "bitwidth", None)
                signed = getattr(run_length_dtype, "signed", None)
                if isinstance(bitwidth, int) and isinstance(signed, bool):
                    value_bits = bitwidth - 1 if signed else bitwidth
                    if int(static_decoded_window_offset) >= 1 << value_bits:
                        raise CoopSinglePhaseRewriteError(
                            f"{scope_name}.run_length_decode "
                            "decoded_window_offset must be representable in the "
                            "run_lengths dtype"
                        )
            if group_first and not _dtype_values_match(
                total_decoded_size_dtype,
                run_length_dtype,
            ):
                raise CoopSinglePhaseRewriteError(
                    f"{scope_name}.run_length_decode total_decoded_size dtype must "
                    "match run_lengths dtype"
                )
            if group_first and not _dtype_values_match(
                decoded_offset_dtype,
                run_length_dtype,
            ):
                raise CoopSinglePhaseRewriteError(
                    f"{scope_name}.run_length_decode decoded offset dtype must "
                    "match run_lengths dtype"
                )

            threads_per_block = self._resolve_threads_per_block_for_parent()
            if threads_per_block is None:
                if self._can_defer_launch_dim_inference():
                    raise _DeferredCoopRewrite
                raise CoopSinglePhaseRewriteError(
                    "Failed to infer threads_per_block for coop._block.run_length."
                )

            factory_kwargs.update(
                {
                    "item_dtype": item_dtype,
                    "run_length_dtype": run_length_dtype,
                    "decoded_offset_dtype": decoded_offset_dtype,
                    "total_decoded_size_dtype": total_decoded_size_dtype,
                    "threads_per_block": threads_per_block,
                    "runs_per_thread": runs_per_thread,
                    "decoded_items_per_thread": decoded_items_per_thread,
                }
            )

            captured_vars = {
                "run_values": run_values_var.name,
                "run_lengths": run_lengths_var.name,
                "total_decoded_size": total_decoded_size_var.name,
            }
            if temp_storage_var is not None:
                captured_vars["temp_storage"] = temp_storage_var.name
            return self._canonical_parent_ctor_spec(
                parent_kind=parent_kind,
                factory_kwargs=factory_kwargs,
                captured_vars=captured_vars,
            )

        return None

    def _collect_parent_ctor_keys(self, value: ir.Var, seen: set[str]) -> set[str]:
        if not isinstance(value, ir.Var):
            return set()
        if value.name in seen:
            return set()
        if value.name in self._parent_ctor_specs:
            return {value.name}

        seen.add(value.name)
        keys: set[str] = set()
        for definition in self._lookup_definitions(value):
            if isinstance(definition, ir.Expr):
                if definition.op == "call":
                    spec = self._resolve_parent_constructor_spec(definition)
                    if spec is not None:
                        self._parent_ctor_specs[value.name] = (
                            self._merge_parent_ctor_specs(
                                self._parent_ctor_specs.get(value.name),
                                spec,
                            )
                        )
                        keys.add(value.name)
                        continue
                if definition.op == "cast":
                    cast_value = getattr(definition, "value", None)
                    if isinstance(cast_value, ir.Var):
                        keys.update(self._collect_parent_ctor_keys(cast_value, seen))
                    continue
                if definition.op == "phi":
                    for incoming in _phi_incoming_values(definition):
                        if isinstance(incoming, ir.Var):
                            keys.update(self._collect_parent_ctor_keys(incoming, seen))
                    continue
            if isinstance(definition, ir.Var):
                keys.update(self._collect_parent_ctor_keys(definition, seen))

        return keys

    def _resolve_parent_ctor_key(self, value: ir.Var) -> str | None:
        if not isinstance(value, ir.Var):
            return None
        keys = self._collect_parent_ctor_keys(value, seen=set())
        if not keys:
            return None
        if len(keys) == 1:
            return next(iter(keys))

        merged_spec: _ParentCtorSpec | None = None
        for key in sorted(keys):
            spec = self._parent_ctor_specs.get(key)
            if spec is None:
                continue
            merged_spec = self._merge_parent_ctor_specs(merged_spec, spec)
        if merged_spec is None:
            return None

        self._parent_ctor_specs[value.name] = merged_spec
        return value.name

    def _var_from_name(self, name: str, loc: ir.Loc) -> ir.Var:
        assert self._block is not None
        return ir.Var(self._block.scope, name, loc)

    def _resolve_parent_child_match(self, call: ir.Expr) -> _RewriteMatch | None:
        func_def = self._lookup_definition(call.func)
        if not (isinstance(func_def, ir.Expr) and func_def.op == "getattr"):
            return None

        method_name = getattr(func_def, "attr", None)
        parent_var = getattr(func_def, "value", None)
        if method_name is None or not isinstance(parent_var, ir.Var):
            return None

        parent_key = self._resolve_parent_ctor_key(parent_var)
        if parent_key is None:
            return None

        parent_spec = self._parent_ctor_specs.get(parent_key)
        if parent_spec is None:
            return None

        parent_kind = parent_spec.parent_kind

        if call.vararg is not None or call.varkwarg is not None:
            raise CoopSinglePhaseRewriteError(
                "coop single-phase parent/child calls do not support *args/**kwargs."
            )

        parent_factory_kwargs = parent_spec.factory_kwargs()
        captured_vars = parent_spec.captured_vars()

        if parent_kind == "histogram":
            parent_items = self._var_from_name(captured_vars["items"], call.loc)
            parent_histogram = self._var_from_name(captured_vars["histogram"], call.loc)

        if parent_kind == "histogram" and method_name == "init":
            if len(call.args) > 1:
                raise CoopSinglePhaseRewriteError(
                    "coop._block.histogram(...).init(...) accepts at most one "
                    "runtime argument."
                )
            if call.kws:
                allowed = {"histogram"}
                for kw, _ in call.kws:
                    if kw not in allowed:
                        raise CoopSinglePhaseRewriteError(
                            "Unsupported keyword for histogram.init(...): "
                            f"'{kw}'. Allowed keywords are: histogram."
                        )
                if len(call.args) == 1:
                    raise CoopSinglePhaseRewriteError(
                        "histogram.init(...) cannot mix positional and keyword "
                        "runtime arguments."
                    )
            histogram_var = parent_histogram
            if call.args:
                histogram_var = call.args[0]
            elif call.kws:
                histogram_var = call.kws[0][1]
            if not isinstance(histogram_var, ir.Var):
                raise CoopSinglePhaseRewriteError(
                    "histogram.init(...) expects histogram to be provided as a variable."
                )
            histogram_dtype = self._resolve_array_dtype(histogram_var)
            if (
                histogram_dtype is not None
                and histogram_dtype != parent_factory_kwargs["counter_dtype"]
            ):
                raise CoopSinglePhaseRewriteError(
                    "histogram.init(...) histogram dtype does not match constructor histogram dtype."
                )

            from cuda.coop.numba_mlir._block._block_histogram import _histogram_init

            return _RewriteMatch(
                op_name="histogram_init",
                factory=_histogram_init,
                func_var_name=call.func.name,
                func_var_name_extra=None,
                runtime_args=(histogram_var,),
                runtime_temp_storage_var=None,
                factory_kwargs=parent_factory_kwargs,
                factory_kw_value_vars=(),
                loc=call.loc,
            )

        if parent_kind == "histogram" and method_name == "composite":
            if len(call.args) > 2:
                raise CoopSinglePhaseRewriteError(
                    "histogram.composite(...) accepts at most two runtime "
                    "arguments: items and histogram."
                )
            if call.kws:
                kw_map = {name: value for name, value in call.kws}
                allowed = {"items", "histogram"}
                invalid = [name for name in kw_map if name not in allowed]
                if invalid:
                    raise CoopSinglePhaseRewriteError(
                        "Unsupported keyword(s) for histogram.composite(...): "
                        + ", ".join(sorted(invalid))
                    )
                if len(call.args) > 0:
                    raise CoopSinglePhaseRewriteError(
                        "histogram.composite(...) cannot mix positional and keyword "
                        "runtime arguments."
                    )
                items_var = kw_map.get("items", parent_items)
                histogram_var = kw_map.get("histogram", parent_histogram)
            else:
                items_var = call.args[0] if len(call.args) >= 1 else parent_items
                histogram_var = (
                    call.args[1] if len(call.args) >= 2 else parent_histogram
                )

            if not isinstance(items_var, ir.Var) or not isinstance(
                histogram_var, ir.Var
            ):
                raise CoopSinglePhaseRewriteError(
                    "histogram.composite(...) expects variable arguments."
                )

            items_dtype = self._resolve_array_dtype(items_var)
            if (
                items_dtype is not None
                and items_dtype != parent_factory_kwargs["item_dtype"]
            ):
                raise CoopSinglePhaseRewriteError(
                    "histogram.composite(...) items dtype does not match constructor items dtype."
                )
            items_per_thread = self._resolve_array_extent(items_var)
            if (
                items_per_thread is not None
                and int(items_per_thread) != parent_factory_kwargs["items_per_thread"]
            ):
                raise CoopSinglePhaseRewriteError(
                    "histogram.composite(...) items shape does not match constructor items_per_thread."
                )

            histogram_dtype = self._resolve_array_dtype(histogram_var)
            if (
                histogram_dtype is not None
                and histogram_dtype != parent_factory_kwargs["counter_dtype"]
            ):
                raise CoopSinglePhaseRewriteError(
                    "histogram.composite(...) histogram dtype does not match constructor histogram dtype."
                )

            bins = self._resolve_array_extent(histogram_var)
            if bins is not None and int(bins) != parent_factory_kwargs["bins"]:
                raise CoopSinglePhaseRewriteError(
                    "histogram.composite(...) histogram shape does not match constructor bins."
                )

            from cuda.coop.numba_mlir._block._block_histogram import (
                _histogram_composite,
            )

            return _RewriteMatch(
                op_name="histogram_composite",
                factory=_histogram_composite,
                func_var_name=call.func.name,
                func_var_name_extra=None,
                runtime_args=(items_var, histogram_var),
                runtime_temp_storage_var=None,
                factory_kwargs=parent_factory_kwargs,
                factory_kw_value_vars=(),
                loc=call.loc,
            )

        if (
            parent_kind
            in {
                "run_length",
                "_common_run_length",
                "_qualified_group_run_length",
            }
            and method_name == "decode"
        ):
            common_v1 = parent_kind == "_common_run_length"
            group_first = parent_kind != "run_length"
            scope_name = "cuda.coop" if common_v1 else "cuda.coop.numba_mlir"
            static_decoded_window_offset = parent_factory_kwargs.pop(
                "_static_decoded_window_offset",
                None,
            )
            parent_run_values = self._var_from_name(
                captured_vars["run_values"], call.loc
            )
            parent_run_lengths = self._var_from_name(
                captured_vars["run_lengths"], call.loc
            )
            parent_total_decoded_size = self._var_from_name(
                captured_vars["total_decoded_size"], call.loc
            )
            runtime_temp_storage_var = None
            if "temp_storage" in captured_vars:
                runtime_temp_storage_var = self._var_from_name(
                    captured_vars["temp_storage"], call.loc
                )

            if len(call.args) > 3:
                raise CoopSinglePhaseRewriteError(
                    "run_length.decode(...) accepts at most three runtime "
                    "arguments: decoded_items, decoded_window_offset, "
                    "relative_offsets."
                )

            if call.kws:
                kw_map = {name: value for name, value in call.kws}
                allowed = {
                    "decoded_items",
                    "decoded_window_offset",
                    "relative_offsets",
                }
                invalid = [name for name in kw_map if name not in allowed]
                if invalid:
                    raise CoopSinglePhaseRewriteError(
                        "Unsupported keyword(s) for run_length.decode(...): "
                        + ", ".join(sorted(invalid))
                    )
                if len(call.args) > 0:
                    raise CoopSinglePhaseRewriteError(
                        "run_length.decode(...) cannot mix positional and keyword "
                        "runtime arguments."
                    )
                decoded_items_var = kw_map.get("decoded_items")
                decoded_window_offset_var = kw_map.get("decoded_window_offset")
                relative_offsets_var = kw_map.get("relative_offsets")
            else:
                decoded_items_var = call.args[0] if len(call.args) >= 1 else None
                decoded_window_offset_var = (
                    call.args[1] if len(call.args) >= 2 else None
                )
                relative_offsets_var = call.args[2] if len(call.args) >= 3 else None

            if not isinstance(decoded_items_var, ir.Var):
                raise CoopSinglePhaseRewriteError(
                    "run_length.decode(...) requires decoded_items as a variable."
                )
            if decoded_window_offset_var is not None and not isinstance(
                decoded_window_offset_var, ir.Var
            ):
                raise CoopSinglePhaseRewriteError(
                    "run_length.decode(...) decoded_window_offset must be a variable."
                )
            if (
                group_first
                and static_decoded_window_offset is None
                and decoded_window_offset_var is not None
            ):
                decoded_window_offset_dtype = self._resolve_var_dtype(
                    decoded_window_offset_var
                )
                from numba_cuda_mlir import types as numba_mlir_types

                if not isinstance(
                    decoded_window_offset_dtype, numba_mlir_types.Integer
                ):
                    raise CoopSinglePhaseRewriteError(
                        f"{scope_name}.run_length_decode decoded_window_offset "
                        "must have an integer dtype"
                    )
            if relative_offsets_var is not None and not isinstance(
                relative_offsets_var, ir.Var
            ):
                raise CoopSinglePhaseRewriteError(
                    "run_length.decode(...) relative_offsets must be a variable."
                )

            decoded_items_dtype = self._resolve_array_dtype(decoded_items_var)
            if (
                decoded_items_dtype is not None
                and decoded_items_dtype != parent_factory_kwargs["item_dtype"]
            ):
                raise CoopSinglePhaseRewriteError(
                    "run_length.decode(...) decoded_items dtype does not match constructor item dtype."
                )
            decoded_items_extent = self._resolve_array_extent(decoded_items_var)
            if (
                decoded_items_extent is not None
                and int(decoded_items_extent)
                != parent_factory_kwargs["decoded_items_per_thread"]
            ):
                raise CoopSinglePhaseRewriteError(
                    "run_length.decode(...) decoded_items shape does not match decoded_items_per_thread."
                )

            factory_kwargs = dict(parent_factory_kwargs)
            runtime_args: list[ir.Var] = [
                parent_run_values,
                parent_run_lengths,
                parent_total_decoded_size,
                decoded_items_var,
            ]

            with_relative_offsets = relative_offsets_var is not None
            with_decoded_window_offset = decoded_window_offset_var is not None

            if with_relative_offsets:
                relative_offsets_dtype = self._resolve_array_dtype(relative_offsets_var)
                if relative_offsets_dtype is None:
                    raise CoopSinglePhaseRewriteError(
                        "Failed to infer run_length.decode(...) relative_offsets dtype."
                    )
                if group_first and not _dtype_values_match(
                    relative_offsets_dtype,
                    parent_factory_kwargs["run_length_dtype"],
                ):
                    raise CoopSinglePhaseRewriteError(
                        f"{scope_name}.run_length_decode relative_offsets dtype must "
                        "match run_lengths dtype"
                    )
                relative_offsets_extent = self._resolve_array_extent(
                    relative_offsets_var
                )
                if (
                    relative_offsets_extent is not None
                    and int(relative_offsets_extent)
                    != parent_factory_kwargs["decoded_items_per_thread"]
                ):
                    raise CoopSinglePhaseRewriteError(
                        "run_length.decode(...) relative_offsets shape does not match decoded_items_per_thread."
                    )
                factory_kwargs["relative_offset_dtype"] = relative_offsets_dtype
                runtime_args.append(relative_offsets_var)

            if with_decoded_window_offset:
                runtime_args.append(decoded_window_offset_var)

            factory_kwargs["with_relative_offsets"] = with_relative_offsets
            factory_kwargs["with_decoded_window_offset"] = with_decoded_window_offset

            op_name = (
                "run_length_decode"
                f"_rel{1 if with_relative_offsets else 0}"
                f"_off{1 if with_decoded_window_offset else 0}"
            )

            from cuda.coop.numba_mlir._block._block_run_length_decode import (
                _run_length_decode,
            )

            return _RewriteMatch(
                op_name=op_name,
                factory=_run_length_decode,
                func_var_name=call.func.name,
                func_var_name_extra=None,
                runtime_args=tuple(runtime_args),
                runtime_temp_storage_var=runtime_temp_storage_var,
                factory_kwargs=factory_kwargs,
                factory_kw_value_vars=(),
                loc=call.loc,
            )

        public_parent_kind = (
            "run_length"
            if parent_kind in {"_common_run_length", "_qualified_group_run_length"}
            else parent_kind
        )
        raise CoopSinglePhaseRewriteError(
            f"coop._block.{public_parent_kind}(...).{method_name}(...) is not "
            "supported by the single-phase rewrite."
        )

    def _validate_and_split_args(
        self,
        op_name: str,
        call: ir.Expr,
        getitem_temp_storage: ir.Var | None,
    ) -> tuple[
        tuple[ir.Var, ...],
        ir.Var | None,
        dict[str, object],
        tuple[ir.Var, ...],
        tuple[tuple[int, object], ...],
    ]:
        spec = self._OP_SPECS[op_name]

        if call.vararg is not None or call.varkwarg is not None:
            raise CoopSinglePhaseRewriteError(
                "coop single-phase v1 does not support *args or **kwargs."
            )

        runtime_arg_count = len(call.args)
        if runtime_arg_count not in spec["runtime_arg_counts"]:
            expected_csv = ", ".join(str(v) for v in sorted(spec["runtime_arg_counts"]))
            raise CoopSinglePhaseRewriteError(
                f"coop single-phase '{op_name}' expects positional runtime argument "
                f"count in {{{expected_csv}}}; got {runtime_arg_count}."
            )

        runtime_args = list(call.args)
        factory_kw_value_vars: list[ir.Var] = []

        allowed_factory_kwargs = spec["allowed_factory_kwargs"]
        required_factory_kwargs = spec["required_factory_kwargs"]
        seen_factory_kwargs: set[str] = set()
        factory_kwargs: dict[str, object] = {}
        runtime_temp_storage = getitem_temp_storage
        runtime_factory_kwargs = tuple(spec.get("runtime_factory_kwargs", ()))
        runtime_only_kwargs = tuple(spec.get("runtime_only_kwargs", ()))
        runtime_factory_kw_prerequisites = dict(
            spec.get("runtime_factory_kw_prerequisites", {})
        )
        base_runtime_arg_count = min(spec["runtime_arg_counts"])
        extra_runtime_arg_count = runtime_arg_count - base_runtime_arg_count
        seen_runtime_factory_kwargs: set[str] = set()
        runtime_factory_kw_vars: dict[str, ir.Var] = {}
        runtime_factory_control_vars: dict[str, ir.Var] = {}
        seen_runtime_only_kwargs: set[str] = set()
        runtime_only_kw_vars: dict[str, ir.Var] = {}
        runtime_offset_var = None
        if runtime_factory_kwargs:
            if extra_runtime_arg_count > len(runtime_factory_kwargs):
                raise CoopSinglePhaseRewriteError(
                    f"coop single-phase '{op_name}' received too many positional "
                    "runtime arguments for its partial-tile overload."
                )
            for index, name in enumerate(
                runtime_factory_kwargs[:extra_runtime_arg_count]
            ):
                factory_kwargs[name] = True
                seen_factory_kwargs.add(name)
                seen_runtime_factory_kwargs.add(name)
                runtime_factory_control_vars[name] = runtime_args[
                    base_runtime_arg_count + index
                ]
        if runtime_only_kwargs:
            if extra_runtime_arg_count > len(runtime_only_kwargs):
                raise CoopSinglePhaseRewriteError(
                    f"coop single-phase '{op_name}' received too many positional "
                    "runtime arguments for its runtime overload."
                )
            for name in runtime_only_kwargs[:extra_runtime_arg_count]:
                seen_runtime_only_kwargs.add(name)

        for name, value_var in call.kws:
            if name == "temp_storage" and op_name in self._TEMP_STORAGE_RUNTIME_KW_OPS:
                if runtime_temp_storage is not None:
                    raise CoopSinglePhaseRewriteError(
                        f"Duplicate coop single-phase '{op_name}' runtime temp storage."
                    )
                if not isinstance(value_var, ir.Var):
                    raise CoopSinglePhaseRewriteError(
                        "coop single-phase runtime temp_storage must be a variable."
                    )
                runtime_temp_storage = value_var
                continue

            if name == "offset" and op_name in {
                "load",
                "store",
                "warp_load",
                "warp_store",
            }:
                if runtime_offset_var is not None:
                    raise CoopSinglePhaseRewriteError(
                        f"Duplicate coop single-phase '{op_name}' runtime "
                        "argument 'offset'."
                    )
                if not isinstance(value_var, ir.Var):
                    raise CoopSinglePhaseRewriteError(
                        "coop single-phase load/store runtime argument "
                        "'offset' must be a variable."
                    )
                runtime_offset_var = value_var
                continue

            if name == "block_aggregate" and op_name == "scan":
                value = self._resolve_factory_kwarg_value(name, value_var)
                if value is None:
                    continue
                if "block_aggregate" in seen_factory_kwargs:
                    raise CoopSinglePhaseRewriteError(
                        "Duplicate coop single-phase 'scan' runtime argument "
                        "'block_aggregate'."
                    )
                if value is not _UNRESOLVED or not isinstance(value_var, ir.Var):
                    raise CoopSinglePhaseRewriteError(
                        "coop single-phase 'scan' runtime argument "
                        "'block_aggregate' must be a variable."
                    )
                runtime_args.append(value_var)
                factory_kwargs["block_aggregate"] = True
                seen_factory_kwargs.add("block_aggregate")
                continue

            if name in {"block_prefix", "block_suffix"} and op_name == "shuffle":
                value = self._resolve_factory_kwarg_value(name, value_var)
                if value is None:
                    continue
                if name in seen_factory_kwargs:
                    raise CoopSinglePhaseRewriteError(
                        f"Duplicate coop single-phase 'shuffle' runtime argument "
                        f"'{name}'."
                    )
                if value is not _UNRESOLVED or not isinstance(value_var, ir.Var):
                    raise CoopSinglePhaseRewriteError(
                        "coop single-phase shuffle boundary output must be a variable."
                    )
                runtime_args.append(value_var)
                factory_kwargs[name] = True
                seen_factory_kwargs.add(name)
                continue

            if name in runtime_factory_kwargs:
                if (
                    name in seen_runtime_factory_kwargs
                    or name in runtime_factory_kw_vars
                ):
                    raise CoopSinglePhaseRewriteError(
                        f"Duplicate coop single-phase '{op_name}' runtime "
                        f"argument '{name}'."
                    )
                if not isinstance(value_var, ir.Var):
                    raise CoopSinglePhaseRewriteError(
                        "coop single-phase partial-tile runtime argument "
                        f"'{name}' must be a variable."
                    )
                runtime_factory_kw_vars[name] = value_var
                runtime_factory_control_vars[name] = value_var
                continue

            if name in runtime_only_kwargs:
                if name in seen_runtime_only_kwargs or name in runtime_only_kw_vars:
                    raise CoopSinglePhaseRewriteError(
                        f"Duplicate coop single-phase '{op_name}' runtime "
                        f"argument '{name}'."
                    )
                if not isinstance(value_var, ir.Var):
                    raise CoopSinglePhaseRewriteError(
                        "coop single-phase runtime argument "
                        f"'{name}' must be a variable."
                    )
                runtime_only_kw_vars[name] = value_var
                continue

            if name not in allowed_factory_kwargs:
                allowed = ", ".join(
                    sorted(
                        set(allowed_factory_kwargs)
                        | set(runtime_factory_kwargs)
                        | set(runtime_only_kwargs)
                    )
                )
                raise CoopSinglePhaseRewriteError(
                    f"Unsupported coop single-phase '{op_name}' factory keyword "
                    f"'{name}'. Allowed keywords are: {allowed}."
                )
            if name in seen_factory_kwargs:
                raise CoopSinglePhaseRewriteError(
                    f"Duplicate coop single-phase '{op_name}' factory keyword '{name}'."
                )
            seen_factory_kwargs.add(name)
            value = self._resolve_factory_kwarg_value(name, value_var)
            if value is _UNRESOLVED:
                raise CoopSinglePhaseRewriteError(
                    "Failed to evaluate coop single-phase factory argument "
                    f"'{name}' for '{op_name}' as a compile-time constant."
                )
            factory_kwargs[name] = value
            if isinstance(value_var, ir.Var):
                factory_kw_value_vars.append(value_var)

        for index, name in enumerate(runtime_factory_kwargs):
            value_var = runtime_factory_kw_vars.get(name)
            if value_var is None:
                continue
            prerequisite = runtime_factory_kw_prerequisites.get(name)
            if (
                prerequisite is not None
                and prerequisite not in seen_runtime_factory_kwargs
                and prerequisite not in runtime_factory_kw_vars
            ):
                raise CoopSinglePhaseRewriteError(
                    f"coop single-phase '{op_name}' runtime argument '{name}' "
                    f"requires '{prerequisite}'."
                )
            runtime_args.append(value_var)
            factory_kwargs[name] = True
            seen_factory_kwargs.add(name)
            seen_runtime_factory_kwargs.add(name)

        for name in runtime_only_kwargs:
            value_var = runtime_only_kw_vars.get(name)
            if value_var is None:
                continue
            prerequisite = runtime_factory_kw_prerequisites.get(name)
            if (
                prerequisite is not None
                and prerequisite not in seen_runtime_only_kwargs
                and prerequisite not in runtime_only_kw_vars
            ):
                raise CoopSinglePhaseRewriteError(
                    f"coop single-phase '{op_name}' runtime argument '{name}' "
                    f"requires '{prerequisite}'."
                )
            runtime_args.append(value_var)
            seen_runtime_only_kwargs.add(name)

        if runtime_offset_var is not None:
            runtime_args.append(runtime_offset_var)

        self._infer_factory_kwargs_from_thread_data(
            op_name,
            runtime_args,
            allowed_factory_kwargs,
            seen_factory_kwargs,
            factory_kwargs,
        )
        self._validate_common_merge_sort_runtime_controls(
            op_name=op_name,
            runtime_factory_control_vars=runtime_factory_control_vars,
            factory_kwargs=factory_kwargs,
        )
        self._canonicalize_dim_factory_alias(
            op_name=op_name,
            seen_factory_kwargs=seen_factory_kwargs,
            factory_kwargs=factory_kwargs,
        )
        self._infer_factory_kwargs_from_context(
            allowed_factory_kwargs,
            seen_factory_kwargs,
            factory_kwargs,
        )
        self._validate_topk_runtime_controls(
            op_name=op_name,
            runtime_args=runtime_args,
            runtime_factory_control_vars=runtime_factory_control_vars,
            factory_kwargs=factory_kwargs,
        )
        runtime_arg_constant_replacements = (
            self._radix_sort_runtime_constant_replacements(
                op_name=op_name,
                runtime_args=runtime_args,
                runtime_only_kw_vars=runtime_only_kw_vars,
                factory_kwargs=factory_kwargs,
            )
        )
        if op_name == "adjacent_difference":
            self._finalize_adjacent_difference_factory_kwargs(
                runtime_arg_count=runtime_arg_count,
                seen_factory_kwargs=seen_factory_kwargs,
                factory_kwargs=factory_kwargs,
            )
        if op_name == "shuffle":
            self._finalize_shuffle_factory_kwargs(
                runtime_arg_count=len(runtime_args),
                seen_factory_kwargs=seen_factory_kwargs,
                factory_kwargs=factory_kwargs,
            )
        if op_name == "discontinuity":
            self._finalize_discontinuity_factory_kwargs(
                runtime_arg_count=runtime_arg_count,
                seen_factory_kwargs=seen_factory_kwargs,
                factory_kwargs=factory_kwargs,
            )
            runtime_args = self._reorder_discontinuity_runtime_args(
                runtime_args, factory_kwargs
            )
        if op_name in {"radix_rank", "_common_radix_rank"}:
            self._finalize_radix_rank_factory_kwargs(
                op_name=op_name,
                runtime_arg_count=runtime_arg_count,
                seen_factory_kwargs=seen_factory_kwargs,
                factory_kwargs=factory_kwargs,
            )
        if op_name in {
            "scan",
            "exclusive_sum",
            "inclusive_sum",
            "exclusive_scan",
            "inclusive_scan",
        }:
            self._finalize_scan_factory_kwargs(
                op_name=op_name,
                runtime_arg_count=runtime_arg_count,
                factory_kwargs=factory_kwargs,
            )
        if op_name == "exchange":
            self._finalize_exchange_factory_kwargs(
                runtime_args=runtime_args,
                runtime_arg_count=runtime_arg_count,
                seen_factory_kwargs=seen_factory_kwargs,
                factory_kwargs=factory_kwargs,
            )
        if op_name == "warp_exchange":
            self._finalize_warp_exchange_factory_kwargs(
                runtime_args=runtime_args,
                runtime_arg_count=runtime_arg_count,
                seen_factory_kwargs=seen_factory_kwargs,
                factory_kwargs=factory_kwargs,
            )

        missing = required_factory_kwargs - seen_factory_kwargs
        if missing:
            if self._can_defer_launch_dim_inference():
                raise _DeferredCoopRewrite
            missing_csv = ", ".join(sorted(missing))
            raise CoopSinglePhaseRewriteError(
                f"coop single-phase '{op_name}' requires explicit factory keywords: "
                f"{missing_csv}."
            )

        if (
            runtime_temp_storage is not None
            and op_name not in self._TEMP_STORAGE_RUNTIME_KW_OPS
        ):
            raise CoopSinglePhaseRewriteError(
                f"coop single-phase '{op_name}' does not support runtime temp_storage."
            )

        return (
            tuple(runtime_args),
            runtime_temp_storage,
            factory_kwargs,
            tuple(factory_kw_value_vars),
            runtime_arg_constant_replacements,
        )

    @staticmethod
    def _store_algorithm_mutates_payload(op_name: str, algorithm: object) -> bool:
        if isinstance(algorithm, bool):
            return False
        if isinstance(algorithm, int):
            from ._enums import BlockStoreAlgorithm, WarpStoreAlgorithm

            enum_type = (
                WarpStoreAlgorithm if op_name == "warp_store" else BlockStoreAlgorithm
            )
            try:
                resolved = enum_type(algorithm)
            except ValueError:
                return False
            if op_name == "warp_store":
                return resolved is WarpStoreAlgorithm.TRANSPOSE
            return resolved in {
                BlockStoreAlgorithm.TRANSPOSE,
                BlockStoreAlgorithm.WARP_TRANSPOSE,
                BlockStoreAlgorithm.WARP_TRANSPOSE_TIMESLICED,
            }
        normalized = str(algorithm).lower()
        if op_name == "warp_store":
            return normalized in {"transpose", "::cub::warp_store_transpose"}
        return normalized in {
            "transpose",
            "warp_transpose",
            "warp_transpose_timesliced",
            "::cub::block_store_transpose",
            "::cub::block_store_warp_transpose",
            "::cub::block_store_warp_transpose_timesliced",
        }

    def _extract_group_root_match_metadata(
        self,
        *,
        op_name: str,
        runtime_args: tuple[ir.Var, ...],
        factory_kwargs: dict[str, object],
    ) -> tuple[bool, bool, bool]:
        physical_warp_tile_origin = factory_kwargs.pop(
            "_physical_warp_tile_origin", False
        )
        group_root_store = factory_kwargs.pop("_group_root_store", False)
        common_profile_operation = factory_kwargs.pop("_common_profile_operation", None)
        if not isinstance(physical_warp_tile_origin, bool):
            raise CoopSinglePhaseRewriteError(
                "_physical_warp_tile_origin must be a compile-time bool"
            )
        if not isinstance(group_root_store, bool):
            raise CoopSinglePhaseRewriteError(
                "_group_root_store must be a compile-time bool"
            )
        if common_profile_operation is not None:
            operation_families = {
                "load": frozenset({"load"}),
                "warp_load": frozenset({"load"}),
                "store": frozenset({"store"}),
                "warp_store": frozenset({"store"}),
                "exchange": frozenset({"exchange"}),
                "warp_exchange": frozenset({"exchange"}),
                "group_reduce": frozenset({"reduce", "sum"}),
                "reduce": frozenset({"reduce", "sum"}),
                "sum": frozenset({"reduce", "sum"}),
                "block_reduce_builtin": frozenset({"reduce", "sum"}),
                "warp_reduce_builtin": frozenset({"reduce", "sum"}),
                "warp_reduce": frozenset({"reduce", "sum"}),
                "warp_sum": frozenset({"reduce", "sum"}),
                "scan": frozenset(
                    {
                        "scan",
                        "exclusive_sum",
                        "inclusive_sum",
                        "exclusive_scan",
                        "inclusive_scan",
                    }
                ),
                "warp_exclusive_sum": frozenset(
                    {"scan", "exclusive_sum", "exclusive_scan"}
                ),
                "warp_inclusive_sum": frozenset(
                    {"scan", "inclusive_sum", "inclusive_scan"}
                ),
                "warp_exclusive_scan": frozenset({"scan", "exclusive_scan"}),
                "warp_inclusive_scan": frozenset({"scan", "inclusive_scan"}),
                "adjacent_difference": frozenset({"adjacent_difference"}),
                "discontinuity": frozenset({"discontinuity"}),
                "shuffle": frozenset({"shuffle"}),
            }
            if common_profile_operation not in operation_families.get(
                op_name, frozenset()
            ):
                raise CoopSinglePhaseRewriteError(
                    "_common_profile_operation does not match the rewritten "
                    "group operation"
                )
            from ._common import _validate_common_numeric_dtype

            if op_name in {"load", "warp_load", "store", "warp_store"}:
                operand_names = (
                    ("source", "output")
                    if op_name in {"load", "warp_load"}
                    else ("destination", "value")
                )
                for operand_name, operand in zip(operand_names, runtime_args):
                    operand_dtype = self._resolve_var_dtype(operand)
                    if operand_dtype is None:
                        raise CoopSinglePhaseRewriteError(
                            f"Failed to infer cuda.coop.{common_profile_operation} "
                            f"{operand_name} dtype for common V1 validation."
                        )
                    try:
                        _validate_common_numeric_dtype(
                            operand_dtype,
                            operation=common_profile_operation,
                        )
                    except (TypeError, ValueError) as exc:
                        raise CoopSinglePhaseRewriteError(str(exc)) from exc
            else:
                try:
                    _validate_common_numeric_dtype(
                        factory_kwargs.get("dtype"),
                        operation=common_profile_operation,
                    )
                except (TypeError, ValueError) as exc:
                    raise CoopSinglePhaseRewriteError(str(exc)) from exc
        root_store_scalar = False
        preserve_root_store_payload = False
        if group_root_store:
            if op_name not in {"store", "warp_store"} or len(runtime_args) < 2:
                raise CoopSinglePhaseRewriteError(
                    "_group_root_store is valid only for root store calls"
                )
            root_store_scalar = self._resolve_thread_data_spec(runtime_args[1]) is None
            preserve_root_store_payload = root_store_scalar or (
                self._store_algorithm_mutates_payload(
                    op_name,
                    factory_kwargs.get("algorithm", "direct"),
                )
            )
        return (
            physical_warp_tile_origin,
            preserve_root_store_payload,
            root_store_scalar,
        )

    def _finalize_scan_factory_kwargs(
        self,
        *,
        op_name: str,
        runtime_arg_count: int,
        factory_kwargs: dict[str, object],
    ) -> None:
        if runtime_arg_count == 2:
            return

        from cuda.coop.numba_mlir._types import StatefulFunction

        block_prefix_callback_op = factory_kwargs.get(
            "block_prefix_callback_op", factory_kwargs.get("prefix_op")
        )
        if not isinstance(block_prefix_callback_op, StatefulFunction):
            raise CoopSinglePhaseRewriteError(
                f"coop single-phase '{op_name}' third runtime argument is only "
                "valid with a stateful block_prefix_callback_op."
            )

    def _finalize_adjacent_difference_factory_kwargs(
        self,
        runtime_arg_count: int,
        seen_factory_kwargs: set[str],
        factory_kwargs: dict[str, object],
    ) -> None:
        from cuda.coop.numba_mlir._block._block_adjacent_difference import (
            BlockAdjacentDifferenceType,
        )

        adjacent_difference_type = factory_kwargs.get(
            "block_adjacent_difference_type", BlockAdjacentDifferenceType.SubtractLeft
        )
        if isinstance(adjacent_difference_type, int):
            adjacent_difference_type = BlockAdjacentDifferenceType(
                adjacent_difference_type
            )
        if not isinstance(adjacent_difference_type, BlockAdjacentDifferenceType):
            raise CoopSinglePhaseRewriteError(
                "coop single-phase 'adjacent_difference' "
                "block_adjacent_difference_type must be a "
                "BlockAdjacentDifferenceType enum value."
            )

        if adjacent_difference_type == BlockAdjacentDifferenceType.SubtractLeft:
            tile_kw = "tile_predecessor_item"
            invalid_tile_kw = "tile_successor_item"
        else:
            tile_kw = "tile_successor_item"
            invalid_tile_kw = "tile_predecessor_item"

        if invalid_tile_kw in seen_factory_kwargs:
            raise CoopSinglePhaseRewriteError(
                "coop single-phase 'adjacent_difference' received an invalid "
                f"factory kwarg '{invalid_tile_kw}' for "
                f"{adjacent_difference_type.name}."
            )

        valid_items_specified = "valid_items" in seen_factory_kwargs
        tile_item_specified = tile_kw in seen_factory_kwargs

        if runtime_arg_count == 4:
            if not valid_items_specified:
                factory_kwargs["valid_items"] = True
                seen_factory_kwargs.add("valid_items")
                valid_items_specified = True
            if not tile_item_specified:
                factory_kwargs[tile_kw] = True
                seen_factory_kwargs.add(tile_kw)
                tile_item_specified = True
        elif runtime_arg_count == 3:
            if valid_items_specified and tile_item_specified:
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase 'adjacent_difference' runtime argument count 3 "
                    "cannot satisfy both partial-tile and tile-boundary overloads."
                )
            if not valid_items_specified and not tile_item_specified:
                factory_kwargs["valid_items"] = True
                seen_factory_kwargs.add("valid_items")
                valid_items_specified = True
        elif runtime_arg_count != 2:
            raise CoopSinglePhaseRewriteError(
                "coop single-phase 'adjacent_difference' runtime argument count must "
                "be one of {2, 3, 4}."
            )

        if (
            adjacent_difference_type == BlockAdjacentDifferenceType.SubtractRight
            and valid_items_specified
            and tile_item_specified
        ):
            raise CoopSinglePhaseRewriteError(
                "coop single-phase 'adjacent_difference' cannot use "
                "tile_successor_item with SubtractRightPartialTile."
            )

        expected_count = 2 + int(valid_items_specified) + int(tile_item_specified)
        if runtime_arg_count != expected_count:
            raise CoopSinglePhaseRewriteError(
                "coop single-phase 'adjacent_difference' runtime argument count "
                f"{runtime_arg_count} is incompatible with selected overload; "
                f"expected {expected_count}."
            )

    def _finalize_shuffle_factory_kwargs(
        self,
        runtime_arg_count: int,
        seen_factory_kwargs: set[str],
        factory_kwargs: dict[str, object],
    ) -> None:
        from cuda.coop.numba_mlir._block._block_shuffle import BlockShuffleType

        shuffle_type = factory_kwargs.get("block_shuffle_type", BlockShuffleType.Up)
        if isinstance(shuffle_type, int):
            shuffle_type = BlockShuffleType(shuffle_type)
        if not isinstance(shuffle_type, BlockShuffleType):
            raise CoopSinglePhaseRewriteError(
                "coop single-phase 'shuffle' block_shuffle_type must be a "
                "BlockShuffleType enum value."
            )

        if runtime_arg_count == 1:
            if "items_per_thread" in seen_factory_kwargs:
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase scalar 'shuffle' does not accept "
                    "items_per_thread."
                )
            if (
                "block_prefix" in seen_factory_kwargs
                or "block_suffix" in seen_factory_kwargs
            ):
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase scalar 'shuffle' does not accept "
                    "block_prefix/block_suffix."
                )
            if "distance" not in seen_factory_kwargs:
                factory_kwargs["distance"] = 1
                seen_factory_kwargs.add("distance")
            return

        if runtime_arg_count not in {2, 3}:
            raise CoopSinglePhaseRewriteError(
                "coop single-phase 'shuffle' runtime argument count must be one of "
                "{1, 2, 3}."
            )

        if shuffle_type not in {BlockShuffleType.Up, BlockShuffleType.Down}:
            raise CoopSinglePhaseRewriteError(
                "coop single-phase array 'shuffle' only supports "
                "BlockShuffleType.Up/Down."
            )
        if "distance" in seen_factory_kwargs:
            raise CoopSinglePhaseRewriteError(
                "coop single-phase array 'shuffle' does not support distance."
            )

        if runtime_arg_count == 2:
            if (
                "block_prefix" in seen_factory_kwargs
                or "block_suffix" in seen_factory_kwargs
            ):
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase array 'shuffle' received block_prefix/"
                    "block_suffix without a matching runtime boundary argument."
                )
            return

        # runtime_arg_count == 3
        if shuffle_type == BlockShuffleType.Up:
            if "block_prefix" in seen_factory_kwargs:
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase array 'shuffle' with BlockShuffleType.Up "
                    "does not support block_prefix."
                )
            if "block_suffix" not in seen_factory_kwargs:
                factory_kwargs["block_suffix"] = True
                seen_factory_kwargs.add("block_suffix")
        else:
            if "block_suffix" in seen_factory_kwargs:
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase array 'shuffle' with BlockShuffleType.Down "
                    "does not support block_suffix."
                )
            if "block_prefix" not in seen_factory_kwargs:
                factory_kwargs["block_prefix"] = True
                seen_factory_kwargs.add("block_prefix")

    def _finalize_discontinuity_factory_kwargs(
        self,
        runtime_arg_count: int,
        seen_factory_kwargs: set[str],
        factory_kwargs: dict[str, object],
    ) -> None:
        from cuda.coop.numba_mlir._block._block_discontinuity import (
            BlockDiscontinuityType,
        )

        discontinuity_type = factory_kwargs.get(
            "block_discontinuity_type", BlockDiscontinuityType.HEADS
        )
        if isinstance(discontinuity_type, int):
            discontinuity_type = BlockDiscontinuityType(discontinuity_type)
        if not isinstance(discontinuity_type, BlockDiscontinuityType):
            raise CoopSinglePhaseRewriteError(
                "coop single-phase 'discontinuity' block_discontinuity_type must be a "
                "BlockDiscontinuityType enum value."
            )

        pred_specified = "tile_predecessor_item" in seen_factory_kwargs
        succ_specified = "tile_successor_item" in seen_factory_kwargs

        if discontinuity_type == BlockDiscontinuityType.HEADS:
            if succ_specified:
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase 'discontinuity' HEADS does not accept "
                    "tile_successor_item."
                )
            if runtime_arg_count == 3 and not pred_specified:
                factory_kwargs["tile_predecessor_item"] = True
                seen_factory_kwargs.add("tile_predecessor_item")
                pred_specified = True
            expected_count = 2 + int(pred_specified)
        elif discontinuity_type == BlockDiscontinuityType.TAILS:
            if pred_specified:
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase 'discontinuity' TAILS does not accept "
                    "tile_predecessor_item."
                )
            if runtime_arg_count == 3 and not succ_specified:
                factory_kwargs["tile_successor_item"] = True
                seen_factory_kwargs.add("tile_successor_item")
                succ_specified = True
            expected_count = 2 + int(succ_specified)
        else:
            if runtime_arg_count == 5:
                if not pred_specified:
                    factory_kwargs["tile_predecessor_item"] = True
                    seen_factory_kwargs.add("tile_predecessor_item")
                    pred_specified = True
                if not succ_specified:
                    factory_kwargs["tile_successor_item"] = True
                    seen_factory_kwargs.add("tile_successor_item")
                    succ_specified = True
            elif runtime_arg_count == 4:
                if pred_specified and succ_specified:
                    raise CoopSinglePhaseRewriteError(
                        "coop single-phase 'discontinuity' HEADS_AND_TAILS runtime "
                        "argument count 4 cannot satisfy both tile boundary kwargs."
                    )
                if not pred_specified and not succ_specified:
                    factory_kwargs["tile_predecessor_item"] = True
                    seen_factory_kwargs.add("tile_predecessor_item")
                    pred_specified = True
            expected_count = 3 + int(pred_specified) + int(succ_specified)

        if runtime_arg_count != expected_count:
            raise CoopSinglePhaseRewriteError(
                "coop single-phase 'discontinuity' runtime argument count "
                f"{runtime_arg_count} is incompatible with selected overload; "
                f"expected {expected_count}."
            )

    @staticmethod
    def _reorder_discontinuity_runtime_args(
        runtime_args: list[ir.Var], factory_kwargs: dict[str, object]
    ) -> list[ir.Var]:
        from cuda.coop.numba_mlir._block._block_discontinuity import (
            BlockDiscontinuityType,
        )

        discontinuity_type = factory_kwargs.get(
            "block_discontinuity_type", BlockDiscontinuityType.HEADS
        )
        if isinstance(discontinuity_type, int):
            discontinuity_type = BlockDiscontinuityType(discontinuity_type)

        if discontinuity_type in {
            BlockDiscontinuityType.HEADS,
            BlockDiscontinuityType.TAILS,
        }:
            if len(runtime_args) < 2:
                return runtime_args
            return [runtime_args[1], runtime_args[0], *runtime_args[2:]]

        if len(runtime_args) < 3:
            return runtime_args

        input_items = runtime_args[0]
        head_flags = runtime_args[1]
        tail_flags = runtime_args[2]
        has_predecessor = "tile_predecessor_item" in factory_kwargs
        has_successor = "tile_successor_item" in factory_kwargs

        if has_predecessor and has_successor:
            return [
                head_flags,
                runtime_args[3],
                tail_flags,
                runtime_args[4],
                input_items,
            ]
        if has_predecessor:
            return [head_flags, runtime_args[3], tail_flags, input_items]
        if has_successor:
            return [head_flags, tail_flags, runtime_args[3], input_items]
        return [head_flags, tail_flags, input_items]

    def _finalize_radix_rank_factory_kwargs(
        self,
        op_name: str,
        runtime_arg_count: int,
        seen_factory_kwargs: set[str],
        factory_kwargs: dict[str, object],
    ) -> None:
        if runtime_arg_count not in {2, 3}:
            raise CoopSinglePhaseRewriteError(
                "coop single-phase 'radix_rank' runtime argument count must be one "
                "of {2, 3}."
            )

        if "begin_bit" not in seen_factory_kwargs:
            factory_kwargs["begin_bit"] = 0
            seen_factory_kwargs.add("begin_bit")

        dtype = factory_kwargs.get("dtype")
        bitwidth = getattr(dtype, "bitwidth", None)
        if bitwidth is not None:
            bitwidth = int(bitwidth)

        explicit_end_bit = factory_kwargs.get("end_bit")
        try:
            end_bit = resolve_static_radix_end_bit(
                begin_bit=factory_kwargs["begin_bit"],
                end_bit=explicit_end_bit,
                bit_width=bitwidth,
                default_radix_bits=4,
                clamp_default=False,
            )
        except ValueError as exc:
            scope_name = (
                "cuda.coop"
                if op_name == "_common_radix_rank"
                else "cuda.coop.numba_mlir"
            )
            raise CoopSinglePhaseRewriteError(f"{scope_name}.radix_rank {exc}") from exc
        factory_kwargs["begin_bit"] = int(factory_kwargs["begin_bit"])
        factory_kwargs["end_bit"] = end_bit
        seen_factory_kwargs.add("end_bit")
        if "descending" in seen_factory_kwargs:
            try:
                factory_kwargs["descending"] = normalize_radix_order(
                    factory_kwargs["descending"]
                ).descending
            except ValueError as exc:
                raise CoopSinglePhaseRewriteError(
                    f"coop single-phase 'radix_rank' {exc}."
                ) from exc

    def _radix_sort_runtime_constant_replacements(
        self,
        *,
        op_name: str,
        runtime_args: list[ir.Var],
        runtime_only_kw_vars: dict[str, ir.Var],
        factory_kwargs: dict[str, object],
    ) -> tuple[tuple[int, object], ...]:
        radix_sort_operations = {
            "radix_sort_keys",
            "radix_sort_keys_descending",
            "radix_sort_pairs",
            "radix_sort_pairs_descending",
            "_common_radix_sort_keys",
            "_common_radix_sort_pairs",
        }
        if op_name not in radix_sort_operations:
            return ()

        begin_var = runtime_only_kw_vars.get("begin_bit")
        end_var = runtime_only_kw_vars.get("end_bit")
        if begin_var is None and end_var is None:
            return ()
        if begin_var is None or end_var is None:
            # The ordinary runtime-argument prerequisite diagnostic handles
            # this before the normalization hook is reached.
            return ()

        common_root = op_name in {
            "_common_radix_sort_keys",
            "_common_radix_sort_pairs",
        }
        public_operation = (
            "radix_sort_pairs" if "pairs" in op_name else "radix_sort_keys"
        )
        scope_name = "cuda.coop" if common_root else "cuda.coop.numba_mlir"
        prefix = f"{scope_name}.{public_operation}"

        begin_bit = self._resolve_factory_kwarg_value("begin_bit", begin_var)
        end_bit = self._resolve_factory_kwarg_value("end_bit", end_var)

        def static_bound(name: str, value: object) -> int | None:
            if value is _UNRESOLVED:
                return None
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise CoopSinglePhaseRewriteError(f"{prefix} {name} must be an integer")
            return int(value)

        static_begin = static_bound("begin_bit", begin_bit)
        dtype = factory_kwargs.get("dtype")
        if dtype is None:
            dtype = factory_kwargs.get("key_dtype")
        if dtype is None:
            dtype = factory_kwargs.get("keys")
        bit_width = getattr(dtype, "bitwidth", None)
        if bit_width is not None:
            bit_width = int(bit_width)

        replacements: tuple[tuple[int, object], ...] = ()
        if end_bit is None:
            if bit_width is None:
                raise CoopSinglePhaseRewriteError(
                    f"{prefix} end_bit must be provided when the key dtype bit "
                    "width cannot be inferred"
                )
            static_end = bit_width
            end_index = next(
                index
                for index, argument in enumerate(runtime_args)
                if argument is end_var or argument.name == end_var.name
            )
            replacements = ((end_index, static_end),)
        else:
            static_end = static_bound("end_bit", end_bit)

        if static_begin is not None:
            if static_begin < 0:
                raise CoopSinglePhaseRewriteError(
                    f"{prefix} begin_bit must be non-negative"
                )
            if bit_width is not None and static_begin >= bit_width:
                raise CoopSinglePhaseRewriteError(
                    f"{prefix} begin_bit must be < {bit_width}"
                )
        if static_end is not None:
            if static_end < 1:
                raise CoopSinglePhaseRewriteError(f"{prefix} end_bit must be positive")
            if bit_width is not None and static_end > bit_width:
                raise CoopSinglePhaseRewriteError(
                    f"{prefix} end_bit must be <= {bit_width}"
                )
        if (
            static_begin is not None
            and static_end is not None
            and static_end <= static_begin
        ):
            raise CoopSinglePhaseRewriteError(
                f"{prefix} end_bit must be greater than begin_bit"
            )
        return replacements

    def _validate_common_merge_sort_runtime_controls(
        self,
        *,
        op_name: str,
        runtime_factory_control_vars: dict[str, ir.Var],
        factory_kwargs: dict[str, object],
    ) -> None:
        if op_name not in {
            "_common_merge_sort_keys",
            "_common_merge_sort_pairs",
            "_common_warp_merge_sort_keys",
            "_common_warp_merge_sort_pairs",
        }:
            return

        oob_default_var = runtime_factory_control_vars.get("oob_default")
        if oob_default_var is None:
            return
        operation = "merge_sort_pairs" if "pairs" in op_name else "merge_sort_keys"
        key_dtype = factory_kwargs.get("dtype")
        if key_dtype is None:
            key_dtype = factory_kwargs.get("keys")
        if key_dtype is None:
            raise CoopSinglePhaseRewriteError(
                f"cuda.coop.{operation} could not infer the keys dtype "
                "before validating oob_default"
            )

        from ._common import _validate_common_merge_sort_oob_default

        static_value = self._resolve_factory_kwarg_value(
            "oob_default",
            oob_default_var,
        )
        try:
            if static_value is not _UNRESOLVED:
                _validate_common_merge_sort_oob_default(
                    key_dtype,
                    operation=operation,
                    static_value=static_value,
                )
                return

            runtime_dtype = self._resolve_var_dtype(oob_default_var)
            if runtime_dtype is None:
                raise CoopSinglePhaseRewriteError(
                    f"cuda.coop.{operation} could not infer the integer "
                    "dtype of oob_default"
                )
            _validate_common_merge_sort_oob_default(
                key_dtype,
                operation=operation,
                runtime_dtype=runtime_dtype,
            )
        except (TypeError, ValueError) as exc:
            raise CoopSinglePhaseRewriteError(str(exc)) from exc

    def _validate_topk_runtime_controls(
        self,
        *,
        op_name: str,
        runtime_args: list[ir.Var],
        runtime_factory_control_vars: dict[str, ir.Var],
        factory_kwargs: dict[str, object],
    ) -> None:
        operation = {
            "topk_max_keys": "topk_max_keys",
            "topk_min_keys": "topk_min_keys",
            "topk_max_pairs": "topk_max_pairs",
            "topk_min_pairs": "topk_min_pairs",
            "_common_topk_max_keys": "topk_max_keys",
            "_common_topk_min_keys": "topk_min_keys",
            "_common_topk_max_pairs": "topk_max_pairs",
            "_common_topk_min_pairs": "topk_min_pairs",
            "_qualified_group_topk_max_keys": "topk_max_keys",
            "_qualified_group_topk_min_keys": "topk_min_keys",
            "_qualified_group_topk_max_pairs": "topk_max_pairs",
            "_qualified_group_topk_min_pairs": "topk_min_pairs",
        }.get(op_name)
        if operation is None:
            return

        is_common_root = op_name.startswith("_common_")
        scope_name = "cuda.coop" if is_common_root else "cuda.coop.numba_mlir"
        prefix = f"{scope_name}.{operation}"
        k_index = 2 if operation.endswith("_pairs") else 1

        from numba_cuda_mlir import types as numba_mlir_types

        control_vars = {
            "k": runtime_args[k_index],
            **runtime_factory_control_vars,
        }
        for name, value_ref in control_vars.items():
            dtype = self._resolve_var_dtype(value_ref)
            if dtype is not None and not isinstance(dtype, numba_mlir_types.Integer):
                public_name = "valid_items" if name == "num_valid" else name
                raise CoopSinglePhaseRewriteError(
                    f"{prefix} {public_name} must have an integer dtype"
                )

        def static_int(name: str, value_ref: ir.Var | None) -> int | None:
            if value_ref is None:
                return None
            value = self._resolve_factory_kwarg_value(name, value_ref)
            if value is _UNRESOLVED:
                return None
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise CoopSinglePhaseRewriteError(
                    f"{prefix} {name} must be an int-like scalar"
                )
            return int(value)

        if is_common_root:
            static_k = static_int("k", control_vars["k"])
            if static_k is not None and static_k <= 0:
                raise CoopSinglePhaseRewriteError(f"{prefix} k must be positive")

            threads_per_block = factory_kwargs.get("threads_per_block")
            items_per_thread = factory_kwargs.get("items_per_thread")
            tile_size = None
            if threads_per_block is not None and isinstance(items_per_thread, Integral):
                dim = normalize_dim_param(threads_per_block)
                tile_size = dim.x * dim.y * dim.z * int(items_per_thread)

            valid_items_var = control_vars.get("num_valid")
            static_valid_items = (
                tile_size
                if valid_items_var is None
                else static_int("valid_items", valid_items_var)
            )
            if static_valid_items is not None and static_valid_items <= 0:
                raise CoopSinglePhaseRewriteError(
                    f"{prefix} valid_items must be positive"
                )
            if (
                static_valid_items is not None
                and tile_size is not None
                and static_valid_items > tile_size
            ):
                raise CoopSinglePhaseRewriteError(
                    f"{prefix} valid_items must be <= tile size {tile_size}"
                )
            if (
                static_k is not None
                and static_valid_items is not None
                and static_k > static_valid_items
            ):
                raise CoopSinglePhaseRewriteError(f"{prefix} k must be <= valid_items")

        dtype = factory_kwargs.get("dtype")
        if dtype is None:
            dtype = factory_kwargs.get("key_dtype")
        if dtype is None:
            dtype = factory_kwargs.get("keys")
        key_width = getattr(dtype, "bitwidth", None)
        if key_width is None:
            return
        key_width = int(key_width)

        begin_var = control_vars.get("begin_bit")
        end_var = control_vars.get("end_bit")
        static_begin = 0 if begin_var is None else static_int("begin_bit", begin_var)
        static_end = key_width if end_var is None else static_int("end_bit", end_var)
        if static_begin is not None and static_begin < 0:
            raise CoopSinglePhaseRewriteError(
                f"{prefix} begin_bit must be non-negative"
            )
        if static_begin is not None and static_begin >= key_width:
            raise CoopSinglePhaseRewriteError(
                f"{prefix} begin_bit must be < {key_width}"
            )
        if static_end is not None and static_end < 1:
            raise CoopSinglePhaseRewriteError(f"{prefix} end_bit must be positive")
        if static_end is not None and static_end > key_width:
            raise CoopSinglePhaseRewriteError(
                f"{prefix} end_bit must be <= {key_width}"
            )
        if (
            static_begin is not None
            and static_end is not None
            and static_end <= static_begin
        ):
            raise CoopSinglePhaseRewriteError(
                f"{prefix} end_bit must be greater than begin_bit"
            )

    def _finalize_exchange_factory_kwargs(
        self,
        runtime_args: list[ir.Var],
        runtime_arg_count: int,
        seen_factory_kwargs: set[str],
        factory_kwargs: dict[str, object],
    ) -> None:
        from cuda.coop.numba_mlir._block._block_exchange import BlockExchangeType

        exchange_type = factory_kwargs.get(
            "block_exchange_type", BlockExchangeType.StripedToBlocked
        )
        if isinstance(exchange_type, int):
            exchange_type = BlockExchangeType(exchange_type)
        if not isinstance(exchange_type, BlockExchangeType):
            raise CoopSinglePhaseRewriteError(
                "coop single-phase 'exchange' block_exchange_type must be a "
                "BlockExchangeType enum value."
            )

        uses_ranks = exchange_type in {
            BlockExchangeType.ScatterToBlocked,
            BlockExchangeType.ScatterToStriped,
            BlockExchangeType.ScatterToStripedGuarded,
            BlockExchangeType.ScatterToStripedFlagged,
        }
        uses_valid_flags = exchange_type == BlockExchangeType.ScatterToStripedFlagged

        if uses_valid_flags:
            expected_counts = {3, 4}
        elif uses_ranks:
            expected_counts = {2, 3}
        else:
            expected_counts = {1, 2}
        if runtime_arg_count not in expected_counts:
            expected_csv = ", ".join(str(v) for v in sorted(expected_counts))
            raise CoopSinglePhaseRewriteError(
                "coop single-phase 'exchange' runtime argument count "
                f"{runtime_arg_count} is incompatible with block_exchange_type="
                f"{exchange_type.name}; expected one of {{{expected_csv}}}."
            )

        out_of_place = runtime_arg_count in {2, 3, 4} and (
            (not uses_ranks and runtime_arg_count == 2)
            or (uses_ranks and not uses_valid_flags and runtime_arg_count == 3)
            or (uses_valid_flags and runtime_arg_count == 4)
        )
        if "use_output_items" in seen_factory_kwargs:
            requested_value_form = factory_kwargs["use_output_items"]
            if requested_value_form is not None and not isinstance(
                requested_value_form, bool
            ):
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase 'exchange' use_output_items must be a "
                    "boolean or None."
                )
            if (
                requested_value_form is not None
                and requested_value_form != out_of_place
            ):
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase 'exchange' use_output_items does not "
                    "match the runtime argument form."
                )
        factory_kwargs["use_output_items"] = out_of_place
        seen_factory_kwargs.add("use_output_items")

        ranks_idx = 2 if out_of_place else 1
        valid_flags_idx = 3 if out_of_place else 2

        if uses_ranks:
            ranks_var = runtime_args[ranks_idx]
            if not isinstance(ranks_var, ir.Var):
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase 'exchange' ranks runtime argument must be a variable."
                )
            inferred_offset_dtype = self._resolve_var_dtype(ranks_var)
            if inferred_offset_dtype is None:
                raise CoopSinglePhaseRewriteError(
                    "Failed to infer offset_dtype from exchange ranks argument."
                )
            if "offset_dtype" in seen_factory_kwargs:
                if not _dtype_values_match(
                    factory_kwargs["offset_dtype"], inferred_offset_dtype
                ):
                    raise CoopSinglePhaseRewriteError(
                        "exchange offset_dtype does not match ranks argument dtype."
                    )
            else:
                factory_kwargs["offset_dtype"] = inferred_offset_dtype
                seen_factory_kwargs.add("offset_dtype")

        if uses_valid_flags:
            valid_flags_var = runtime_args[valid_flags_idx]
            if not isinstance(valid_flags_var, ir.Var):
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase 'exchange' valid_flags runtime argument must be a variable."
                )
            inferred_valid_flag_dtype = self._resolve_var_dtype(valid_flags_var)
            if inferred_valid_flag_dtype is None:
                raise CoopSinglePhaseRewriteError(
                    "Failed to infer valid_flag_dtype from exchange valid_flags argument."
                )
            if "valid_flag_dtype" in seen_factory_kwargs:
                if not _dtype_values_match(
                    factory_kwargs["valid_flag_dtype"], inferred_valid_flag_dtype
                ):
                    raise CoopSinglePhaseRewriteError(
                        "exchange valid_flag_dtype does not match valid_flags argument dtype."
                    )
            else:
                factory_kwargs["valid_flag_dtype"] = inferred_valid_flag_dtype
                seen_factory_kwargs.add("valid_flag_dtype")

    def _finalize_warp_exchange_factory_kwargs(
        self,
        runtime_args: list[ir.Var],
        runtime_arg_count: int,
        seen_factory_kwargs: set[str],
        factory_kwargs: dict[str, object],
    ) -> None:
        from cuda.coop.numba_mlir._warp._warp_exchange import WarpExchangeType

        exchange_type = factory_kwargs.get(
            "warp_exchange_type", WarpExchangeType.StripedToBlocked
        )
        if isinstance(exchange_type, int):
            exchange_type = WarpExchangeType(exchange_type)
        if not isinstance(exchange_type, WarpExchangeType):
            raise CoopSinglePhaseRewriteError(
                "coop single-phase 'warp_exchange' warp_exchange_type must be a "
                "WarpExchangeType enum value."
            )

        uses_ranks = exchange_type == WarpExchangeType.ScatterToStriped
        expected_counts = {2, 3} if uses_ranks else {2}
        if runtime_arg_count not in expected_counts:
            expected_csv = ", ".join(str(v) for v in sorted(expected_counts))
            raise CoopSinglePhaseRewriteError(
                "coop single-phase 'warp_exchange' runtime argument count "
                f"{runtime_arg_count} is incompatible with warp_exchange_type="
                f"{exchange_type.name}; expected one of {{{expected_csv}}}."
            )

        if uses_ranks:
            inferred_use_output_items = runtime_arg_count == 3
            if "use_output_items" in seen_factory_kwargs:
                if factory_kwargs["use_output_items"] != inferred_use_output_items:
                    raise CoopSinglePhaseRewriteError(
                        "warp_exchange use_output_items does not match the "
                        "runtime argument count."
                    )
            else:
                factory_kwargs["use_output_items"] = inferred_use_output_items
                seen_factory_kwargs.add("use_output_items")

            ranks_var = runtime_args[2 if inferred_use_output_items else 1]
            if not isinstance(ranks_var, ir.Var):
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase 'warp_exchange' ranks runtime argument "
                    "must be a variable."
                )
            inferred_offset_dtype = self._resolve_var_dtype(ranks_var)
            if inferred_offset_dtype is None:
                raise CoopSinglePhaseRewriteError(
                    "Failed to infer offset_dtype from warp_exchange ranks argument."
                )
            if "offset_dtype" in seen_factory_kwargs:
                if not _dtype_values_match(
                    factory_kwargs["offset_dtype"], inferred_offset_dtype
                ):
                    raise CoopSinglePhaseRewriteError(
                        "warp_exchange offset_dtype does not match ranks argument dtype."
                    )
            else:
                factory_kwargs["offset_dtype"] = inferred_offset_dtype
                seen_factory_kwargs.add("offset_dtype")

    def _infer_factory_kwargs_from_thread_data(
        self,
        op_name: str,
        runtime_args: list[ir.Var],
        allowed_factory_kwargs: set[str],
        seen_factory_kwargs: set[str],
        factory_kwargs: dict[str, object],
    ) -> None:
        common_integer_key_operation = {
            "_common_merge_sort_keys": "merge_sort_keys",
            "_common_merge_sort_pairs": "merge_sort_pairs",
            "_common_warp_merge_sort_keys": "merge_sort_keys",
            "_common_warp_merge_sort_pairs": "merge_sort_pairs",
            "_common_radix_sort_keys": "radix_sort_keys",
            "_common_radix_sort_pairs": "radix_sort_pairs",
            "_common_radix_rank": "radix_rank",
            "_common_topk_max_keys": "topk_max_keys",
            "_common_topk_min_keys": "topk_min_keys",
            "_common_topk_max_pairs": "topk_max_pairs",
            "_common_topk_min_pairs": "topk_min_pairs",
        }.get(op_name)
        common_numeric_value_operation = {
            "_common_merge_sort_pairs": "merge_sort_pairs",
            "_common_warp_merge_sort_pairs": "merge_sort_pairs",
            "_common_radix_sort_pairs": "radix_sort_pairs",
            "_common_topk_max_pairs": "topk_max_pairs",
            "_common_topk_min_pairs": "topk_min_pairs",
        }.get(op_name)
        op_name = {
            "_common_merge_sort_keys": "merge_sort_keys",
            "_common_merge_sort_pairs": "merge_sort_pairs",
            "_common_warp_merge_sort_keys": "warp_merge_sort_keys",
            "_common_warp_merge_sort_pairs": "warp_merge_sort_pairs",
            "_common_radix_sort_keys": "radix_sort_keys",
            "_common_radix_sort_pairs": "radix_sort_pairs",
            "_common_radix_rank": "radix_rank",
            "_common_topk_max_keys": "topk_max_keys",
            "_common_topk_min_keys": "topk_min_keys",
            "_common_topk_max_pairs": "topk_max_pairs",
            "_common_topk_min_pairs": "topk_min_pairs",
            "_qualified_group_topk_max_keys": "topk_max_keys",
            "_qualified_group_topk_min_keys": "topk_min_keys",
            "_qualified_group_topk_max_pairs": "topk_max_pairs",
            "_qualified_group_topk_min_pairs": "topk_min_pairs",
        }.get(op_name, op_name)

        def validate_common_integer_key_dtype(dtype):
            if common_integer_key_operation is None or dtype is None:
                return dtype
            from ._common import _validate_common_integer_key_dtype

            try:
                return _validate_common_integer_key_dtype(
                    dtype, operation=common_integer_key_operation
                )
            except TypeError as exc:
                raise CoopSinglePhaseRewriteError(str(exc)) from exc

        def validate_common_numeric_value_dtype(dtype):
            if common_numeric_value_operation is None or dtype is None:
                return dtype
            from ._common import _validate_common_numeric_dtype

            try:
                return _validate_common_numeric_dtype(
                    dtype,
                    operation=common_numeric_value_operation,
                    parameter="value",
                )
            except TypeError as exc:
                raise CoopSinglePhaseRewriteError(str(exc)) from exc

        def _factory_value(*names: str):
            for name in names:
                value = factory_kwargs.get(name)
                if value is not None:
                    return value
            return None

        def _factory_kwarg_matches(name: str, actual, expected) -> bool:
            if name in {"dtype", "key_dtype", "value_dtype", "keys", "values"}:
                try:
                    actual = normalize_dtype_param(actual)
                    expected = normalize_dtype_param(expected)
                except ValueError:
                    pass
            return actual == expected

        def infer_kwarg(name: str, value) -> None:
            if name not in allowed_factory_kwargs or value is None:
                return
            if name in seen_factory_kwargs:
                if not _factory_kwarg_matches(name, factory_kwargs[name], value):
                    raise CoopSinglePhaseRewriteError(
                        f"coop single-phase {op_name!r} {name} does not match "
                        "the value inferred from coop.ThreadData."
                    )
                return
            factory_kwargs[name] = value
            seen_factory_kwargs.add(name)

        def candidate(index: int) -> tuple[ir.Var | None, _ThreadDataSpec | None]:
            if not (0 <= index < len(runtime_args)):
                return None, None
            candidate = runtime_args[index]
            if not isinstance(candidate, ir.Var):
                return None, None
            spec = self._resolve_thread_data_spec(candidate)
            if self._is_typed_group_payload_var(candidate) and (
                spec is None or spec.items_per_thread is None
            ):
                raise CoopSinglePhaseRewriteError(
                    f"coop single-phase {op_name!r} could not infer the static "
                    "extent of a typed group payload"
                )
            return candidate, spec

        if op_name in {"load", "store", "warp_load", "warp_store"}:
            thread_data_var, spec = candidate(1)
            if spec is None:
                if op_name in {"store", "warp_store"}:
                    inferred_dtype = None
                    if runtime_args:
                        arg0 = runtime_args[0]
                        if isinstance(arg0, ir.Var):
                            inferred_dtype = self._resolve_var_dtype(arg0)
                    if (
                        inferred_dtype is None
                        and len(runtime_args) > 1
                        and isinstance(runtime_args[1], ir.Var)
                    ):
                        inferred_dtype = self._resolve_var_dtype(runtime_args[1])
                    infer_kwarg("items_per_thread", 1)
                    infer_kwarg("dtype", inferred_dtype)
                return
            infer_kwarg("items_per_thread", spec.items_per_thread)
            inferred_dtype = spec.dtype
            if inferred_dtype is None and len(runtime_args) > 0:
                arg0 = runtime_args[0]
                if isinstance(arg0, ir.Var):
                    inferred_dtype = self._resolve_var_dtype(arg0)
            if inferred_dtype is None:
                inferred_dtype = _factory_value("dtype")
            infer_kwarg("dtype", inferred_dtype)
            if inferred_dtype is not None and thread_data_var is not None:
                self._record_inferred_thread_data_dtype(thread_data_var, inferred_dtype)
            return

        if op_name in {
            "reduce",
            "sum",
            "group_reduce",
            "block_reduce_builtin",
            "warp_reduce_builtin",
        }:
            thread_data_var, spec = candidate(0)
            if spec is not None:
                infer_kwarg("items_per_thread", spec.items_per_thread)
            inferred_dtype = spec.dtype if spec is not None else None
            if (
                inferred_dtype is None
                and spec is not None
                and thread_data_var is not None
            ):
                inferred_dtype = self._infer_thread_data_dtype_from_writes(
                    thread_data_var
                )
            if (
                inferred_dtype is None
                and runtime_args
                and isinstance(runtime_args[0], ir.Var)
            ):
                inferred_dtype = self._resolve_var_dtype(runtime_args[0])
            if inferred_dtype is None:
                inferred_dtype = _factory_value("dtype")
            infer_kwarg("dtype", inferred_dtype)
            if inferred_dtype is not None and thread_data_var is not None:
                self._record_inferred_thread_data_dtype(thread_data_var, inferred_dtype)
            return

        if op_name in {"exchange", "warp_exchange"}:
            input_var, input_spec = candidate(0)
            second_var, second_spec = candidate(1)
            if input_spec is None and second_spec is None:
                return
            output_var = second_var
            output_spec = second_spec
            rank_spec = None
            valid_flag_spec = None
            uses_ranks = False
            uses_valid_flags = False
            out_of_place = True
            if op_name == "exchange":
                from cuda.coop.numba_mlir._block._block_exchange import (
                    BlockExchangeType,
                )

                exchange_type = factory_kwargs.get(
                    "block_exchange_type", BlockExchangeType.StripedToBlocked
                )
                if isinstance(exchange_type, int):
                    exchange_type = BlockExchangeType(exchange_type)
                uses_ranks = exchange_type in {
                    BlockExchangeType.ScatterToBlocked,
                    BlockExchangeType.ScatterToStriped,
                    BlockExchangeType.ScatterToStripedGuarded,
                    BlockExchangeType.ScatterToStripedFlagged,
                }
                uses_valid_flags = (
                    exchange_type == BlockExchangeType.ScatterToStripedFlagged
                )
                out_of_place = (
                    (not uses_ranks and len(runtime_args) == 2)
                    or (uses_ranks and not uses_valid_flags and len(runtime_args) == 3)
                    or (uses_valid_flags and len(runtime_args) == 4)
                )
                if not out_of_place:
                    output_var = None
                    output_spec = None
                if uses_ranks:
                    _, rank_spec = candidate(2 if out_of_place else 1)
                if uses_valid_flags:
                    _, valid_flag_spec = candidate(3 if out_of_place else 2)
            elif op_name == "warp_exchange":
                from cuda.coop.numba_mlir._warp._warp_exchange import WarpExchangeType

                exchange_type = factory_kwargs.get(
                    "warp_exchange_type", WarpExchangeType.StripedToBlocked
                )
                if isinstance(exchange_type, int):
                    exchange_type = WarpExchangeType(exchange_type)
                uses_ranks = exchange_type == WarpExchangeType.ScatterToStriped
                out_of_place = (not uses_ranks and len(runtime_args) == 2) or (
                    uses_ranks and len(runtime_args) == 3
                )
                if not out_of_place:
                    output_var = None
                    output_spec = None
                if uses_ranks:
                    _, rank_spec = candidate(2 if out_of_place else 1)
            self._require_matching_items_per_thread(
                op_name, "input", input_spec, "output", output_spec
            )
            self._require_matching_items_per_thread(
                op_name, "input", input_spec, "ranks", rank_spec
            )
            self._require_matching_items_per_thread(
                op_name, "input", input_spec, "valid_flags", valid_flag_spec
            )
            if (
                input_spec is not None
                and output_spec is not None
                and input_spec.dtype is not None
                and output_spec.dtype is not None
                and input_spec.dtype != output_spec.dtype
            ):
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase 'exchange' requires input/output arrays "
                    "to have matching dtype."
                )
            inferred_dtype = None
            if input_spec is not None:
                inferred_dtype = input_spec.dtype
            if inferred_dtype is None and output_spec is not None:
                inferred_dtype = output_spec.dtype
            if inferred_dtype is None:
                inferred_dtype = _factory_value("dtype")
            infer_kwarg(
                "items_per_thread",
                input_spec.items_per_thread
                if input_spec is not None
                else output_spec.items_per_thread,
            )
            infer_kwarg("dtype", inferred_dtype)
            if inferred_dtype is not None:
                if input_var is not None:
                    self._record_inferred_thread_data_dtype(input_var, inferred_dtype)
                if output_var is not None:
                    self._record_inferred_thread_data_dtype(output_var, inferred_dtype)
            return

        if op_name == "adjacent_difference":
            input_var, input_spec = candidate(0)
            output_var, output_spec = candidate(1)
            self._require_matching_items_per_thread(
                op_name, "input", input_spec, "output", output_spec
            )
            infer_kwarg(
                "items_per_thread",
                (
                    input_spec.items_per_thread
                    if input_spec is not None
                    else output_spec.items_per_thread
                    if output_spec is not None
                    else None
                ),
            )
            inferred_dtype = input_spec.dtype if input_spec is not None else None
            if inferred_dtype is None and output_spec is not None:
                inferred_dtype = output_spec.dtype
            if inferred_dtype is None and input_var is not None:
                inferred_dtype = self._infer_thread_data_dtype_from_writes(input_var)
            if inferred_dtype is None and output_var is not None:
                inferred_dtype = self._infer_thread_data_dtype_from_writes(output_var)
            if (
                inferred_dtype is None
                and runtime_args
                and isinstance(runtime_args[0], ir.Var)
            ):
                inferred_dtype = self._resolve_var_dtype(runtime_args[0])
            if inferred_dtype is None:
                inferred_dtype = _factory_value("dtype")
            infer_kwarg("dtype", inferred_dtype)
            if inferred_dtype is not None and input_var is not None:
                self._record_inferred_thread_data_dtype(input_var, inferred_dtype)
            if inferred_dtype is not None and output_var is not None:
                self._record_inferred_thread_data_dtype(output_var, inferred_dtype)
            return

        if op_name == "shuffle":
            if len(runtime_args) == 1:
                arg0 = runtime_args[0]
                inferred_dtype = (
                    self._resolve_var_dtype(arg0) if isinstance(arg0, ir.Var) else None
                )
                if inferred_dtype is None:
                    inferred_dtype = _factory_value("dtype")
                infer_kwarg("dtype", inferred_dtype)
                return

            input_var, input_spec = candidate(0)
            output_var, output_spec = candidate(1)
            self._require_matching_items_per_thread(
                op_name, "input", input_spec, "output", output_spec
            )
            infer_kwarg(
                "items_per_thread",
                (
                    input_spec.items_per_thread
                    if input_spec is not None
                    else output_spec.items_per_thread
                    if output_spec is not None
                    else None
                ),
            )
            inferred_dtype = input_spec.dtype if input_spec is not None else None
            if inferred_dtype is None and output_spec is not None:
                inferred_dtype = output_spec.dtype
            if (
                inferred_dtype is None
                and runtime_args
                and isinstance(runtime_args[0], ir.Var)
            ):
                inferred_dtype = self._resolve_var_dtype(runtime_args[0])
            if inferred_dtype is None:
                inferred_dtype = _factory_value("dtype")
            infer_kwarg("dtype", inferred_dtype)
            if inferred_dtype is not None and input_var is not None:
                self._record_inferred_thread_data_dtype(input_var, inferred_dtype)
            if inferred_dtype is not None and output_var is not None:
                self._record_inferred_thread_data_dtype(output_var, inferred_dtype)
            return

        if op_name == "discontinuity":
            input_var, input_spec = candidate(0)
            first_flags_var, first_flags_spec = candidate(1)
            second_flags_var, second_flags_spec = candidate(2)
            if (
                input_spec is None
                and first_flags_spec is None
                and second_flags_spec is None
            ):
                return

            self._require_matching_items_per_thread(
                op_name, "input", input_spec, "flags", first_flags_spec
            )
            self._require_matching_items_per_thread(
                op_name, "input", input_spec, "second_flags", second_flags_spec
            )

            infer_kwarg(
                "items_per_thread",
                (
                    input_spec.items_per_thread
                    if input_spec is not None
                    else first_flags_spec.items_per_thread
                    if first_flags_spec is not None
                    else second_flags_spec.items_per_thread
                    if second_flags_spec is not None
                    else None
                ),
            )

            inferred_dtype = input_spec.dtype if input_spec is not None else None
            if inferred_dtype is None and input_var is not None:
                inferred_dtype = self._infer_thread_data_dtype_from_writes(input_var)
            if inferred_dtype is None:
                inferred_dtype = self._resolve_var_dtype(input_var)
            if inferred_dtype is None:
                inferred_dtype = _factory_value("dtype")
            infer_kwarg("dtype", inferred_dtype)

            inferred_flag_dtype = (
                first_flags_spec.dtype if first_flags_spec is not None else None
            )
            if inferred_flag_dtype is None and second_flags_spec is not None:
                inferred_flag_dtype = second_flags_spec.dtype
            if inferred_flag_dtype is not None and second_flags_spec is not None:
                if (
                    second_flags_spec.dtype is not None
                    and second_flags_spec.dtype != inferred_flag_dtype
                ):
                    raise CoopSinglePhaseRewriteError(
                        "coop single-phase 'discontinuity' requires head/tail flags "
                        "to share the same dtype."
                    )
            if inferred_flag_dtype is None:
                inferred_flag_dtype = _factory_value("flag_dtype")
            if inferred_flag_dtype is None:
                from numba_cuda_mlir import types as numba_mlir_types

                inferred_flag_dtype = numba_mlir_types.boolean
            infer_kwarg("flag_dtype", inferred_flag_dtype)

            if inferred_dtype is not None and input_var is not None:
                self._record_inferred_thread_data_dtype(input_var, inferred_dtype)
            if inferred_flag_dtype is not None and first_flags_var is not None:
                self._record_inferred_thread_data_dtype(
                    first_flags_var, inferred_flag_dtype
                )
            if inferred_flag_dtype is not None and second_flags_var is not None:
                self._record_inferred_thread_data_dtype(
                    second_flags_var, inferred_flag_dtype
                )
            return

        if op_name == "merge_sort_keys":
            thread_data_var, spec = candidate(0)
            if spec is None:
                return
            infer_kwarg("items_per_thread", spec.items_per_thread)
            inferred_dtype = spec.dtype
            if inferred_dtype is None and thread_data_var is not None:
                inferred_dtype = self._resolve_var_dtype(thread_data_var)
            if inferred_dtype is None:
                inferred_dtype = _factory_value("dtype")
            if inferred_dtype is None and thread_data_var is not None:
                inferred_dtype = self._infer_thread_data_dtype_from_provenance_writes(
                    thread_data_var
                )
            inferred_dtype = validate_common_integer_key_dtype(inferred_dtype)
            infer_kwarg("dtype", inferred_dtype)
            if inferred_dtype is not None and thread_data_var is not None:
                self._record_inferred_thread_data_dtype(thread_data_var, inferred_dtype)
            return

        if op_name == "merge_sort_pairs":
            key_var, key_spec = candidate(0)
            value_var, value_spec = candidate(1)
            self._require_matching_items_per_thread(
                op_name, "key", key_spec, "value", value_spec
            )
            infer_kwarg(
                "items_per_thread",
                (
                    key_spec.items_per_thread
                    if key_spec is not None
                    else value_spec.items_per_thread
                    if value_spec is not None
                    else None
                ),
            )
            key_dtype = key_spec.dtype if key_spec is not None else None
            value_dtype = value_spec.dtype if value_spec is not None else None
            if (
                key_dtype is None
                and len(runtime_args) > 0
                and isinstance(runtime_args[0], ir.Var)
            ):
                key_dtype = self._resolve_var_dtype(runtime_args[0])
            if (
                value_dtype is None
                and len(runtime_args) > 1
                and isinstance(runtime_args[1], ir.Var)
            ):
                value_dtype = self._resolve_var_dtype(runtime_args[1])
            if key_dtype is None:
                key_dtype = _factory_value("keys")
            if value_dtype is None:
                value_dtype = _factory_value("values")
            if key_dtype is None and key_var is not None:
                key_dtype = self._infer_thread_data_dtype_from_provenance_writes(
                    key_var
                )
            if value_dtype is None and value_var is not None:
                value_dtype = self._infer_thread_data_dtype_from_provenance_writes(
                    value_var
                )
            key_dtype = validate_common_integer_key_dtype(key_dtype)
            value_dtype = validate_common_numeric_value_dtype(value_dtype)
            infer_kwarg("keys", key_dtype)
            infer_kwarg("values", value_dtype)
            if key_dtype is not None and key_var is not None:
                self._record_inferred_thread_data_dtype(key_var, key_dtype)
            if value_dtype is not None and value_var is not None:
                self._record_inferred_thread_data_dtype(value_var, value_dtype)
            return

        if op_name in {
            "warp_reduce",
            "warp_sum",
            "warp_max",
            "warp_min",
            "warp_exclusive_sum",
            "warp_inclusive_sum",
            "warp_exclusive_scan",
            "warp_inclusive_scan",
        }:
            thread_data_var, spec = candidate(0)
            inferred_dtype = spec.dtype if spec is not None else None
            if (
                inferred_dtype is None
                and runtime_args
                and isinstance(runtime_args[0], ir.Var)
            ):
                inferred_dtype = self._resolve_var_dtype(runtime_args[0])
            if inferred_dtype is None:
                inferred_dtype = _factory_value("dtype")
            infer_kwarg("dtype", inferred_dtype)
            if inferred_dtype is not None and thread_data_var is not None:
                self._record_inferred_thread_data_dtype(thread_data_var, inferred_dtype)
            return

        if op_name == "warp_merge_sort_keys":
            thread_data_var, spec = candidate(0)
            if spec is None:
                return
            infer_kwarg("items_per_thread", spec.items_per_thread)
            inferred_dtype = spec.dtype
            if inferred_dtype is None and thread_data_var is not None:
                inferred_dtype = self._resolve_var_dtype(thread_data_var)
            if inferred_dtype is None:
                inferred_dtype = _factory_value("dtype")
            if inferred_dtype is None and thread_data_var is not None:
                inferred_dtype = self._infer_thread_data_dtype_from_provenance_writes(
                    thread_data_var
                )
            inferred_dtype = validate_common_integer_key_dtype(inferred_dtype)
            infer_kwarg("dtype", inferred_dtype)
            if inferred_dtype is not None and thread_data_var is not None:
                self._record_inferred_thread_data_dtype(thread_data_var, inferred_dtype)
            return

        if op_name in {
            "scan",
            "exclusive_sum",
            "inclusive_sum",
            "exclusive_scan",
            "inclusive_scan",
        }:
            _, input_spec = candidate(0)
            _, output_spec = candidate(1)
            self._require_matching_items_per_thread(
                op_name, "input", input_spec, "output", output_spec
            )
            for idx in (0, 1):
                thread_data_var, spec = candidate(idx)
                if spec is None:
                    continue
                infer_kwarg("items_per_thread", spec.items_per_thread)
                inferred_dtype = spec.dtype
                if inferred_dtype is None:
                    inferred_dtype = self._infer_consistent_dtype_from_runtime_args(
                        runtime_args, exclude_indices=(idx,)
                    )
                if inferred_dtype is None and thread_data_var is not None:
                    inferred_dtype = self._infer_thread_data_dtype_from_writes(
                        thread_data_var
                    )
                if inferred_dtype is None:
                    inferred_dtype = _factory_value("dtype")
                infer_kwarg("dtype", inferred_dtype)
                if inferred_dtype is not None and thread_data_var is not None:
                    self._record_inferred_thread_data_dtype(
                        thread_data_var, inferred_dtype
                    )
                if {
                    "items_per_thread",
                    "dtype",
                }.issubset(seen_factory_kwargs):
                    return
            return

        if op_name == "warp_merge_sort_pairs":
            key_var, key_spec = candidate(0)
            value_var, value_spec = candidate(1)
            self._require_matching_items_per_thread(
                op_name, "key", key_spec, "value", value_spec
            )
            infer_kwarg(
                "items_per_thread",
                (
                    key_spec.items_per_thread
                    if key_spec is not None
                    else value_spec.items_per_thread
                    if value_spec is not None
                    else None
                ),
            )
            key_dtype = key_spec.dtype if key_spec is not None else None
            value_dtype = value_spec.dtype if value_spec is not None else None
            if (
                key_dtype is None
                and len(runtime_args) > 0
                and isinstance(runtime_args[0], ir.Var)
            ):
                key_dtype = self._resolve_var_dtype(runtime_args[0])
            if (
                value_dtype is None
                and len(runtime_args) > 1
                and isinstance(runtime_args[1], ir.Var)
            ):
                value_dtype = self._resolve_var_dtype(runtime_args[1])
            if key_dtype is None:
                key_dtype = _factory_value("keys")
            if value_dtype is None:
                value_dtype = _factory_value("values")
            if key_dtype is None and key_var is not None:
                key_dtype = self._infer_thread_data_dtype_from_provenance_writes(
                    key_var
                )
            if value_dtype is None and value_var is not None:
                value_dtype = self._infer_thread_data_dtype_from_provenance_writes(
                    value_var
                )
            key_dtype = validate_common_integer_key_dtype(key_dtype)
            value_dtype = validate_common_numeric_value_dtype(value_dtype)
            infer_kwarg("keys", key_dtype)
            infer_kwarg("values", value_dtype)
            if key_dtype is not None and key_var is not None:
                self._record_inferred_thread_data_dtype(key_var, key_dtype)
            if value_dtype is not None and value_var is not None:
                self._record_inferred_thread_data_dtype(value_var, value_dtype)
            return

        if op_name in {"radix_sort_keys", "radix_sort_keys_descending"}:
            thread_data_var, spec = candidate(0)
            if spec is None:
                return
            infer_kwarg("items_per_thread", spec.items_per_thread)
            inferred_dtype = spec.dtype
            if (
                inferred_dtype is None
                and runtime_args
                and isinstance(runtime_args[0], ir.Var)
            ):
                inferred_dtype = self._resolve_var_dtype(runtime_args[0])
            if inferred_dtype is None:
                inferred_dtype = _factory_value("dtype")
            inferred_dtype = validate_common_integer_key_dtype(inferred_dtype)
            infer_kwarg("dtype", inferred_dtype)
            if inferred_dtype is not None and thread_data_var is not None:
                self._record_inferred_thread_data_dtype(thread_data_var, inferred_dtype)
            return

        if op_name in {"topk_max_keys", "topk_min_keys"}:
            thread_data_var, spec = candidate(0)
            if spec is None:
                return
            infer_kwarg("items_per_thread", spec.items_per_thread)
            inferred_dtype = spec.dtype
            if (
                inferred_dtype is None
                and runtime_args
                and isinstance(runtime_args[0], ir.Var)
            ):
                inferred_dtype = self._resolve_var_dtype(runtime_args[0])
            if inferred_dtype is None:
                inferred_dtype = _factory_value("dtype")
            inferred_dtype = validate_common_integer_key_dtype(inferred_dtype)
            infer_kwarg("dtype", inferred_dtype)
            if inferred_dtype is not None and thread_data_var is not None:
                self._record_inferred_thread_data_dtype(thread_data_var, inferred_dtype)
            return

        if op_name in {"radix_sort_pairs", "radix_sort_pairs_descending"}:
            key_var, key_spec = candidate(0)
            value_var, value_spec = candidate(1)
            self._require_matching_items_per_thread(
                op_name, "key", key_spec, "value", value_spec
            )
            infer_kwarg(
                "items_per_thread",
                (
                    key_spec.items_per_thread
                    if key_spec is not None
                    else value_spec.items_per_thread
                    if value_spec is not None
                    else None
                ),
            )
            key_dtype = key_spec.dtype if key_spec is not None else None
            value_dtype = value_spec.dtype if value_spec is not None else None
            if (
                key_dtype is None
                and len(runtime_args) > 0
                and isinstance(runtime_args[0], ir.Var)
            ):
                key_dtype = self._resolve_var_dtype(runtime_args[0])
            if (
                value_dtype is None
                and len(runtime_args) > 1
                and isinstance(runtime_args[1], ir.Var)
            ):
                value_dtype = self._resolve_var_dtype(runtime_args[1])
            if key_dtype is None:
                key_dtype = _factory_value("key_dtype")
            if key_dtype is None:
                key_dtype = _factory_value("keys")
            if value_dtype is None:
                value_dtype = _factory_value("value_dtype")
            if value_dtype is None:
                value_dtype = _factory_value("values")
            key_dtype = validate_common_integer_key_dtype(key_dtype)
            value_dtype = validate_common_numeric_value_dtype(value_dtype)
            infer_kwarg("key_dtype", key_dtype)
            infer_kwarg("value_dtype", value_dtype)
            if key_dtype is not None and key_var is not None:
                self._record_inferred_thread_data_dtype(key_var, key_dtype)
            if value_dtype is not None and value_var is not None:
                self._record_inferred_thread_data_dtype(value_var, value_dtype)
            return

        if op_name in {"topk_max_pairs", "topk_min_pairs"}:
            key_var, key_spec = candidate(0)
            value_var, value_spec = candidate(1)
            self._require_matching_items_per_thread(
                op_name, "key", key_spec, "value", value_spec
            )
            infer_kwarg(
                "items_per_thread",
                (
                    key_spec.items_per_thread
                    if key_spec is not None
                    else value_spec.items_per_thread
                    if value_spec is not None
                    else None
                ),
            )
            key_dtype = key_spec.dtype if key_spec is not None else None
            value_dtype = value_spec.dtype if value_spec is not None else None
            if (
                key_dtype is None
                and len(runtime_args) > 0
                and isinstance(runtime_args[0], ir.Var)
            ):
                key_dtype = self._resolve_var_dtype(runtime_args[0])
            if (
                value_dtype is None
                and len(runtime_args) > 1
                and isinstance(runtime_args[1], ir.Var)
            ):
                value_dtype = self._resolve_var_dtype(runtime_args[1])
            if key_dtype is None:
                key_dtype = _factory_value("key_dtype")
            if key_dtype is None:
                key_dtype = _factory_value("keys")
            if value_dtype is None:
                value_dtype = _factory_value("value_dtype")
            if value_dtype is None:
                value_dtype = _factory_value("values")
            key_dtype = validate_common_integer_key_dtype(key_dtype)
            value_dtype = validate_common_numeric_value_dtype(value_dtype)
            infer_kwarg("key_dtype", key_dtype)
            infer_kwarg("value_dtype", value_dtype)
            if key_dtype is not None and key_var is not None:
                self._record_inferred_thread_data_dtype(key_var, key_dtype)
            if value_dtype is not None and value_var is not None:
                self._record_inferred_thread_data_dtype(value_var, value_dtype)
            return

        if op_name == "radix_rank":
            import numpy as np
            from numba_cuda_mlir import types as numba_mlir_types

            def _is_signed_int32_dtype(dtype) -> bool:
                if dtype is None:
                    return False
                if dtype == numba_mlir_types.int32:
                    return True
                bitwidth = getattr(dtype, "bitwidth", None)
                signed = getattr(dtype, "signed", None)
                if bitwidth == 32 and signed is True:
                    return True
                try:
                    return np.dtype(dtype) == np.dtype(np.int32)
                except (TypeError, ValueError):
                    return False

            keys_var, keys_spec = candidate(0)
            ranks_var, ranks_spec = candidate(1)

            self._require_matching_items_per_thread(
                "radix_rank", "keys", keys_spec, "ranks", ranks_spec
            )

            infer_kwarg(
                "items_per_thread",
                (
                    keys_spec.items_per_thread
                    if keys_spec is not None
                    else ranks_spec.items_per_thread
                    if ranks_spec is not None
                    else None
                ),
            )

            inferred_dtype = keys_spec.dtype if keys_spec is not None else None
            if inferred_dtype is None and keys_var is not None:
                inferred_dtype = self._infer_thread_data_dtype_from_writes(keys_var)
            if (
                inferred_dtype is None
                and runtime_args
                and isinstance(runtime_args[0], ir.Var)
            ):
                inferred_dtype = self._resolve_var_dtype(runtime_args[0])
            if inferred_dtype is None:
                inferred_dtype = _factory_value("dtype")
            infer_kwarg("dtype", inferred_dtype)

            if keys_var is not None and inferred_dtype is not None:
                self._record_inferred_thread_data_dtype(keys_var, inferred_dtype)

            if ranks_spec is not None and ranks_var is not None:
                if ranks_spec.dtype is not None and not _is_signed_int32_dtype(
                    ranks_spec.dtype
                ):
                    raise CoopSinglePhaseRewriteError(
                        "coop single-phase 'radix_rank' requires ranks dtype int32."
                    )
                if ranks_spec.dtype is None:
                    self._record_inferred_thread_data_dtype(
                        ranks_var, numba_mlir_types.int32
                    )
            return

    def _infer_factory_kwargs_from_context(
        self,
        allowed_factory_kwargs: set[str],
        seen_factory_kwargs: set[str],
        factory_kwargs: dict[str, object],
    ) -> None:
        for name in ("threads_per_block", "items_per_thread", "threads_in_warp"):
            if (
                name in allowed_factory_kwargs
                and name not in seen_factory_kwargs
                and name in self._compiletime_context
            ):
                factory_kwargs[name] = self._compiletime_context[name]
                seen_factory_kwargs.add(name)

        if (
            "threads_per_block" in allowed_factory_kwargs
            and "threads_per_block" not in seen_factory_kwargs
        ):
            threads_per_block = self._infer_threads_per_block_from_targetoptions()
            if threads_per_block is not None:
                factory_kwargs["threads_per_block"] = threads_per_block
                seen_factory_kwargs.add("threads_per_block")

    def _infer_threads_per_block_from_targetoptions(self):
        metadata = getattr(self._state, "metadata", {}) or {}
        targetoptions = metadata.get("targetoptions", {}) or {}
        launch_config = targetoptions.get("__launch_config__")
        if isinstance(launch_config, dict):
            block = launch_config.get("block")
            if (
                isinstance(block, (tuple, list))
                and block
                and all(isinstance(v, int) and v > 0 for v in block)
            ):
                dims = tuple(int(v) for v in block)
                if len(dims) >= 3 and dims[2] != 1:
                    return dims[:3]
                if len(dims) >= 2 and dims[1] != 1:
                    return dims[:2]
                return dims[0]

        launch_bounds = targetoptions.get("launch_bounds")

        if isinstance(launch_bounds, int):
            return launch_bounds
        if (
            isinstance(launch_bounds, (tuple, list))
            and launch_bounds
            and isinstance(launch_bounds[0], int)
        ):
            return launch_bounds[0]
        return None

    def _canonicalize_dim_factory_alias(
        self,
        *,
        op_name: str,
        seen_factory_kwargs: set[str],
        factory_kwargs: dict[str, object],
    ) -> None:
        if "dim" not in seen_factory_kwargs:
            return
        if "threads_per_block" in seen_factory_kwargs:
            raise CoopSinglePhaseRewriteError(
                f"coop single-phase '{op_name}' received both 'threads_per_block' "
                "and its 'dim' alias; provide only one."
            )
        factory_kwargs["threads_per_block"] = factory_kwargs.pop("dim")
        seen_factory_kwargs.remove("dim")
        seen_factory_kwargs.add("threads_per_block")

    @staticmethod
    def _invocable_cache_key(
        op_name: str, factory_kwargs: dict[str, object]
    ) -> tuple[str, tuple[tuple[str, str, str], ...]]:
        def cache_component(name, value):
            hasher = hashlib.sha1()
            _hash_symbol_value(hasher, value)
            value_type = f"{type(value).__module__}.{type(value).__qualname__}"
            return name, value_type, hasher.hexdigest()

        return (
            op_name,
            tuple(
                sorted(
                    cache_component(name, value)
                    for name, value in factory_kwargs.items()
                )
            ),
        )

    @staticmethod
    def _validate_invocable(invocable, op_name: str):
        if not callable(invocable) or not hasattr(invocable, "files"):
            raise CoopSinglePhaseRewriteError(
                "coop single-phase factory for "
                f"'{op_name}' did not produce a coop invocable; got {type(invocable)!r}."
            )

    def _prepare_ltoir_bundle_for_matches(self, matches: list[_RewriteMatch]) -> None:
        self._prebundled_specializations = {}
        if not matches:
            return
        if self._state.metadata.get(
            "__cuda_coop_numba_mlir_materialized_specializations__"
        ):
            return

        unique_matches: dict[
            tuple[str, tuple[tuple[str, str, str], ...]], _RewriteMatch
        ] = {}
        for match in matches:
            key = self._invocable_cache_key(match.op_name, match.factory_kwargs)
            if key not in unique_matches:
                unique_matches[key] = match

        # Bundling only helps when more than one unique specialization is needed.
        if len(unique_matches) < 2:
            return

        try:
            with collect_specializations() as collected:
                for match in unique_matches.values():
                    _ = match.factory(**match.factory_kwargs)

            if len(collected) != len(unique_matches):
                return

            algorithms = []
            threads_by_algo = {}
            block_threads_by_algo = {}
            prebundled = {}
            for key, (algo, threads, block_threads) in zip(
                unique_matches.keys(), collected
            ):
                algorithms.append(algo)
                prebundled[key] = (algo, threads, block_threads)
                if threads is not None:
                    threads_by_algo[id(algo)] = int(threads)
                if block_threads is not None:
                    block_threads_by_algo[id(algo)] = block_threads

            prepare_ltoir_bundle(
                algorithms,
                bundle_name=f"cuda_coop_numba_mlir_bundle_{id(self)}_{id(self._func_ir)}",
                allow_single=False,
                threads_by_algo=threads_by_algo,
                block_threads_by_algo=block_threads_by_algo,
            )
            self._prebundled_specializations = prebundled
        except (ImportError, OSError, RuntimeError):
            self._prebundled_specializations = {}

    def _materialize_invocable(self, match: _RewriteMatch):
        key = self._invocable_cache_key(match.op_name, match.factory_kwargs)
        if key in self._invocable_cache:
            return self._invocable_cache[key], False
        compile_cache = self._state.metadata.setdefault(
            "__cuda_coop_numba_mlir_invocable_cache__", {}
        )
        if key in compile_cache:
            invocable = compile_cache[key]
            self._validate_invocable(invocable, match.op_name)
            self._invocable_cache[key] = invocable
            return invocable, False

        try:
            prebundled = self._prebundled_specializations.get(key)
            if prebundled is not None:
                specialization, threads, block_threads = prebundled
                invocable = make_invocable_from_specialization(
                    specialization, threads=threads, block_threads=block_threads
                )
            else:
                invocable = match.factory(**match.factory_kwargs)
        except Exception as e:
            raise CoopSinglePhaseRewriteError(
                "Failed to evaluate coop single-phase factory at compile time for "
                f"'{match.op_name}'."
            ) from e

        self._validate_invocable(invocable, match.op_name)
        self._invocable_cache[key] = invocable
        compile_cache[key] = invocable
        return invocable, True

    def _record_invocable_specialization(self, invocable):
        specialization = getattr(invocable, "specialization", None)
        link_key = (
            algo_coalesce_key(specialization) if specialization is not None else None
        )
        materialized_specializations = self._state.metadata.setdefault(
            "__cuda_coop_numba_mlir_materialized_specializations__", []
        )
        if link_key is not None and link_key not in materialized_specializations:
            materialized_specializations.append(link_key)

    def _get_device_shared_memory_limits(self) -> tuple[int, int]:
        max_default = _DEFAULT_STATIC_SHARED_MEMORY_BYTES
        max_optin = max_default
        try:
            limits = _query_device_shared_memory_limits()
            max_default = int(
                limits.get("max_default_shared_memory_per_block", max_default)
                or max_default
            )
            max_optin = int(
                limits.get("max_optin_shared_memory_per_block", max_default)
                or max_default
            )
            if max_optin <= 0:
                max_optin = max_default
        except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
            pass
        return max_default, max_optin

    def _ensure_temp_storage_global_plan(self) -> _TempStorageGlobalPlan:
        cached = self._temp_storage_global_plan
        if cached is not None:
            return cached

        ordered_keys = sorted(
            self._temp_storage_ctor_specs.keys(),
            key=lambda name: (self._temp_storage_ctor_order.get(name, 1 << 30), name),
        )
        offset = 0
        max_alignment = 1
        for key in ordered_keys:
            plan = self._finalize_temp_storage_plan_for_var(key)
            alignment = max(1, int(plan.alignment))
            offset = _align_up(offset, alignment)
            self._temp_storage_plans[key] = replace(plan, base_offset=offset)
            offset += int(plan.size_in_bytes)
            max_alignment = max(max_alignment, alignment)

        total_size = _align_up(offset, max_alignment)
        max_default, max_optin = self._get_device_shared_memory_limits()
        uses_dynamic_smem = total_size > max_default
        dynamic_shared_bytes = total_size if uses_dynamic_smem else 0
        if dynamic_shared_bytes > max_optin:
            raise CoopSinglePhaseRewriteError(
                "TempStorage requires "
                f"{dynamic_shared_bytes} bytes dynamic shared memory, but "
                f"device max opt-in is {max_optin} bytes."
            )
        if dynamic_shared_bytes > 0:
            set_required_dynamic_shared_memory(
                self._state,
                dynamic_shared_bytes,
            )

        plan = _TempStorageGlobalPlan(
            total_size=total_size,
            max_alignment=max_alignment,
            uses_dynamic_smem=uses_dynamic_smem,
            dynamic_shared_bytes=dynamic_shared_bytes,
            max_default_smem=max_default,
            max_optin_smem=max_optin,
        )
        self._temp_storage_global_plan = plan
        return plan

    def _emit_array_slice(
        self,
        block: ir.Block,
        *,
        source_var: ir.Var,
        target_var: ir.Var,
        start: int,
        stop: int,
        loc: ir.Loc,
    ) -> None:
        slice_ctor_global_name = _next_global_name("temp_storage_slice_ctor")
        slice_ctor_var = ir.Var(
            target_var.scope,
            f"__coop_temp_storage_slice_ctor_var_{next(_GLOBAL_NAME_COUNTER)}__",
            loc,
        )
        start_var = ir.Var(
            target_var.scope,
            f"__coop_temp_storage_slice_start_{next(_GLOBAL_NAME_COUNTER)}__",
            loc,
        )
        stop_var = ir.Var(
            target_var.scope,
            f"__coop_temp_storage_slice_stop_{next(_GLOBAL_NAME_COUNTER)}__",
            loc,
        )
        slice_obj_var = ir.Var(
            target_var.scope,
            f"__coop_temp_storage_slice_obj_{next(_GLOBAL_NAME_COUNTER)}__",
            loc,
        )
        block.append(
            ir.Assign(
                ir.Global(slice_ctor_global_name, slice, loc),
                slice_ctor_var,
                loc,
            )
        )
        block.append(ir.Assign(ir.Const(int(start), loc), start_var, loc))
        block.append(ir.Assign(ir.Const(int(stop), loc), stop_var, loc))
        block.append(
            ir.Assign(
                ir.Expr.call(slice_ctor_var, [start_var, stop_var], (), loc),
                slice_obj_var,
                loc,
            )
        )
        block.append(
            ir.Assign(
                ir.Expr.getitem(source_var, slice_obj_var, loc),
                target_var,
                loc,
            )
        )

    def _runtime_temp_storage_arg_for_call(
        self,
        block: ir.Block,
        *,
        source_var: ir.Var,
        call_assign: ir.Assign,
    ) -> tuple[ir.Var, _TempStoragePlan | None]:
        temp_storage_arg = source_var
        temp_storage_plan = self._resolve_temp_storage_plan(source_var)
        if temp_storage_plan is not None and temp_storage_plan.sharing == "exclusive":
            slice_info = temp_storage_plan.slices_by_call_id.get(id(call_assign))
            if slice_info is None:
                raise CoopSinglePhaseRewriteError(
                    f"Could not resolve TempStorage slice for call at {call_assign.loc}."
                )
            sliced_var = ir.Var(
                call_assign.target.scope,
                f"__coop_temp_storage_slice_{next(_GLOBAL_NAME_COUNTER)}__",
                call_assign.loc,
            )
            self._emit_array_slice(
                block,
                source_var=source_var,
                target_var=sliced_var,
                start=slice_info.offset,
                stop=slice_info.offset + slice_info.size_in_bytes,
                loc=call_assign.loc,
            )
            temp_storage_arg = sliced_var
        return temp_storage_arg, temp_storage_plan

    def _emit_temp_storage_auto_sync(
        self,
        block: ir.Block,
        *,
        scope: ir.Scope,
        loc: ir.Loc,
        sync_attr: str,
    ) -> None:
        sync_module_global_name = _next_global_name("temp_storage_sync_mod")
        sync_module_var = ir.Var(
            scope,
            f"__coop_sync_mod_var_{next(_GLOBAL_NAME_COUNTER)}__",
            loc,
        )
        sync_fn_var = ir.Var(
            scope,
            f"__coop_sync_fn_{next(_GLOBAL_NAME_COUNTER)}__",
            loc,
        )
        sync_result_var = ir.Var(
            scope,
            f"__coop_sync_result_{next(_GLOBAL_NAME_COUNTER)}__",
            loc,
        )
        block.append(
            ir.Assign(
                ir.Global(sync_module_global_name, _cuda_module, loc),
                sync_module_var,
                loc,
            )
        )
        block.append(
            ir.Assign(
                ir.Expr.getattr(sync_module_var, sync_attr, loc),
                sync_fn_var,
                loc,
            )
        )
        block.append(
            ir.Assign(
                ir.Expr.call(sync_fn_var, (), (), loc),
                sync_result_var,
                loc,
            )
        )

    def _emit_physical_warp_tile_offset(
        self,
        block: ir.Block,
        *,
        match: _RewriteMatch,
        user_offset: ir.Var,
        scope: ir.Scope,
        loc: ir.Loc,
    ) -> ir.Var:
        items_per_thread = match.factory_kwargs.get("items_per_thread")
        if (
            not isinstance(items_per_thread, int)
            or isinstance(items_per_thread, bool)
            or items_per_thread < 1
        ):
            raise CoopSinglePhaseRewriteError(
                "root physical-warp load/store requires an inferred positive "
                "items_per_thread"
            )
        logical_warp_threads = match.factory_kwargs.get("threads_in_warp", 32)
        if (
            not isinstance(logical_warp_threads, int)
            or isinstance(logical_warp_threads, bool)
            or logical_warp_threads < 1
            or 32 % logical_warp_threads != 0
        ):
            raise CoopSinglePhaseRewriteError(
                "root warp load/store requires threads_in_warp to be a "
                "positive divisor of 32"
            )
        block_dim = normalize_dim_param(match.factory_kwargs.get("threads_per_block"))

        def new_var(stem: str) -> ir.Var:
            return ir.Var(
                scope,
                f"__coop_warp_tile_{stem}_{next(_GLOBAL_NAME_COUNTER)}__",
                loc,
            )

        def constant(value: int, stem: str) -> ir.Var:
            result = new_var(stem)
            block.append(ir.Assign(ir.Const(value, loc), result, loc))
            return result

        def binary(function, lhs: ir.Var, rhs: ir.Var, stem: str) -> ir.Var:
            result = new_var(stem)
            block.append(
                ir.Assign(
                    ir.Expr.binop(function, lhs, rhs, loc),
                    result,
                    loc,
                )
            )
            return result

        module_var = new_var("cuda")
        block.append(
            ir.Assign(
                ir.Global(_next_global_name("warp_tile_cuda"), _cuda_module, loc),
                module_var,
                loc,
            )
        )
        thread_idx = new_var("thread_idx")
        block.append(
            ir.Assign(
                ir.Expr.getattr(module_var, "threadIdx", loc),
                thread_idx,
                loc,
            )
        )

        def component(axis: str) -> ir.Var:
            result = new_var(f"thread_idx_{axis}")
            block.append(
                ir.Assign(
                    ir.Expr.getattr(thread_idx, axis, loc),
                    result,
                    loc,
                )
            )
            return result

        linear_tid = component("x")
        if block_dim[1] > 1 or block_dim[2] > 1:
            y = component("y")
            z = component("z")
            yz = binary(
                operator.mul,
                constant(block_dim[1], "block_y"),
                z,
                "linear_yz",
            )
            yz = binary(operator.add, y, yz, "linear_y")
            yz = binary(
                operator.mul,
                constant(block_dim[0], "block_x"),
                yz,
                "linear_x_stride",
            )
            linear_tid = binary(operator.add, linear_tid, yz, "linear_tid")
        warp_id = binary(
            operator.floordiv,
            linear_tid,
            constant(logical_warp_threads, "warp_threads"),
            "warp_id",
        )
        tile_offset = binary(
            operator.mul,
            warp_id,
            constant(logical_warp_threads * items_per_thread, "warp_tile_items"),
            "implicit_offset",
        )
        return binary(operator.add, tile_offset, user_offset, "offset")

    def _emit_root_store_payload(
        self,
        block: ir.Block,
        *,
        match: _RewriteMatch,
        value: ir.Var,
        scope: ir.Scope,
        loc: ir.Loc,
    ) -> ir.Var:
        dtype = match.factory_kwargs.get("dtype")
        if dtype is None:
            raise CoopSinglePhaseRewriteError("root store requires an inferred dtype")
        items_per_thread = (
            1
            if match.root_store_scalar
            else match.factory_kwargs.get("items_per_thread")
        )
        if (
            isinstance(items_per_thread, bool)
            or not isinstance(items_per_thread, int)
            or items_per_thread < 1
        ):
            raise CoopSinglePhaseRewriteError(
                "root store requires an inferred positive items_per_thread"
            )

        def new_var(stem: str) -> ir.Var:
            return ir.Var(
                scope,
                f"__coop_root_store_{stem}_{next(_GLOBAL_NAME_COUNTER)}__",
                loc,
            )

        module_var = new_var("cuda")
        local_var = new_var("local")
        array_fn = new_var("array")
        shape_var = new_var("shape")
        dtype_var = new_var("dtype")
        payload = new_var("payload")
        block.append(
            ir.Assign(
                ir.Global(_next_global_name("root_store_cuda"), _cuda_module, loc),
                module_var,
                loc,
            )
        )
        block.append(
            ir.Assign(ir.Expr.getattr(module_var, "local", loc), local_var, loc)
        )
        block.append(ir.Assign(ir.Expr.getattr(local_var, "array", loc), array_fn, loc))
        block.append(ir.Assign(ir.Const(items_per_thread, loc), shape_var, loc))
        block.append(
            ir.Assign(
                ir.Global(_next_global_name("root_store_dtype"), dtype, loc),
                dtype_var,
                loc,
            )
        )
        block.append(
            ir.Assign(
                ir.Expr.call(array_fn, [shape_var, dtype_var], (), loc),
                payload,
                loc,
            )
        )
        for item_index in range(items_per_thread):
            index_var = new_var(f"index_{item_index}")
            block.append(ir.Assign(ir.Const(item_index, loc), index_var, loc))
            item_var = value
            if not match.root_store_scalar:
                item_var = new_var(f"item_{item_index}")
                block.append(
                    ir.Assign(ir.Expr.getitem(value, index_var, loc), item_var, loc)
                )
            block.append(ir.SetItem(payload, index_var, item_var, loc))
        return payload

    def _compute_func_temp_storage_requirements(
        self, func_ir
    ) -> dict[str, _TempStorageRequirementSummary]:
        requirements: dict[str, _TempStorageRequirementSummary] = {}
        saved_block_defs = self._block_defs
        saved_block = self._block
        self._temp_storage_ctor_specs = {}
        self._temp_storage_ctor_order = {}

        try:
            # Seed ThreadData and TempStorage constructor specs across all blocks
            # so kwarg inference can resolve dtype/items-per-thread/metadata
            # consistently while scanning.
            ctor_order = 0
            for label in sorted(func_ir.blocks):
                scan_block = func_ir.blocks[label]
                self._block = scan_block
                self._block_defs = {
                    inst.target.name: inst.value
                    for inst in scan_block.body
                    if isinstance(inst, ir.Assign)
                }
                for inst in scan_block.body:
                    if not isinstance(inst, ir.Assign):
                        continue
                    call = inst.value
                    if not isinstance(call, ir.Expr) or call.op != "call":
                        continue
                    if self._is_thread_data_ctor_call(call):
                        self._thread_data_specs[inst.target.name] = (
                            self._merge_thread_data_specs(
                                self._thread_data_specs.get(inst.target.name),
                                self._extract_thread_data_spec(call),
                            )
                        )
                        continue
                    if self._is_typed_group_payload_ctor_call(call):
                        self._thread_data_specs[inst.target.name] = (
                            self._merge_thread_data_specs(
                                self._thread_data_specs.get(inst.target.name),
                                self._extract_typed_group_payload_spec(call),
                            )
                        )
                        continue
                    if self._is_temp_storage_ctor_call(call):
                        self._temp_storage_ctor_specs[inst.target.name] = (
                            self._extract_temp_storage_ctor_spec(call)
                        )
                        self._temp_storage_ctor_order[inst.target.name] = ctor_order
                        ctor_order += 1

            all_scan_matches: list[_RewriteMatch] = []
            temp_storage_scan_entries: list[
                tuple[int, ir.Assign, _RewriteMatch, str]
            ] = []
            source_ordinal = 0
            for label in sorted(func_ir.blocks):
                scan_block = func_ir.blocks[label]
                self._block = scan_block
                self._block_defs = {
                    inst.target.name: inst.value
                    for inst in scan_block.body
                    if isinstance(inst, ir.Assign)
                }
                for inst in scan_block.body:
                    inst_source_ordinal = source_ordinal
                    source_ordinal += 1
                    if not isinstance(inst, ir.Assign):
                        continue
                    call = inst.value
                    if not isinstance(call, ir.Expr) or call.op != "call":
                        continue

                    parent_child_match = self._resolve_parent_child_match(call)
                    if parent_child_match is not None:
                        all_scan_matches.append(parent_child_match)
                        if parent_child_match.runtime_temp_storage_var is not None:
                            ctor_key = self._resolve_temp_storage_ctor_key(
                                parent_child_match.runtime_temp_storage_var
                            )
                            if ctor_key is not None:
                                temp_storage_scan_entries.append(
                                    (
                                        inst_source_ordinal,
                                        inst,
                                        parent_child_match,
                                        ctor_key,
                                    )
                                )
                        continue

                    target = self._resolve_call_target(call)
                    if target is None:
                        direct_invocable_call = (
                            self._resolve_direct_invocable_temp_storage_call(call)
                        )
                        if direct_invocable_call is None:
                            continue
                        ctor_key = self._resolve_temp_storage_ctor_key(
                            direct_invocable_call.temp_storage_var
                        )
                        if ctor_key is None:
                            continue
                        size_in_bytes = max(
                            1,
                            int(
                                getattr(
                                    direct_invocable_call.invocable,
                                    "temp_storage_bytes",
                                    0,
                                )
                                or 0
                            ),
                        )
                        alignment = max(
                            1,
                            int(
                                getattr(
                                    direct_invocable_call.invocable,
                                    "temp_storage_alignment",
                                    0,
                                )
                                or 0
                            ),
                        )
                        summary = requirements.setdefault(
                            ctor_key,
                            _TempStorageRequirementSummary(),
                        )
                        summary.max_size_in_bytes = max(
                            summary.max_size_in_bytes, size_in_bytes
                        )
                        summary.max_alignment = max(summary.max_alignment, alignment)
                        summary.uses.append(
                            _TempStorageUseRequirement(
                                call_assign=inst,
                                order=inst_source_ordinal,
                                size_in_bytes=size_in_bytes,
                                alignment=alignment,
                            )
                        )
                        continue

                    op_name = target.factory.__name__
                    (
                        runtime_args,
                        runtime_temp_storage_var,
                        factory_kwargs,
                        factory_kw_value_vars,
                        runtime_arg_constant_replacements,
                    ) = self._validate_and_split_args(
                        op_name,
                        call,
                        target.getitem_temp_storage,
                    )
                    (
                        physical_warp_tile_origin,
                        preserve_root_store_payload,
                        root_store_scalar,
                    ) = self._extract_group_root_match_metadata(
                        op_name=op_name,
                        runtime_args=runtime_args,
                        factory_kwargs=factory_kwargs,
                    )

                    scan_match = _RewriteMatch(
                        op_name=op_name,
                        factory=target.factory,
                        func_var_name=target.func_var_name,
                        func_var_name_extra=target.func_var_name_extra,
                        runtime_args=runtime_args,
                        runtime_temp_storage_var=runtime_temp_storage_var,
                        factory_kwargs=factory_kwargs,
                        factory_kw_value_vars=factory_kw_value_vars,
                        loc=inst.loc,
                        runtime_arg_constant_replacements=(
                            runtime_arg_constant_replacements
                        ),
                        physical_warp_tile_origin=physical_warp_tile_origin,
                        preserve_root_store_payload=preserve_root_store_payload,
                        root_store_scalar=root_store_scalar,
                    )
                    all_scan_matches.append(scan_match)
                    if runtime_temp_storage_var is None:
                        continue

                    ctor_key = self._resolve_temp_storage_ctor_key(
                        runtime_temp_storage_var
                    )
                    if ctor_key is None:
                        # Runtime arrays that are not coop.TempStorage placeholders
                        # are forwarded as-is and do not participate in TempStorage
                        # constructor planning.
                        continue
                    temp_storage_scan_entries.append(
                        (inst_source_ordinal, inst, scan_match, ctor_key)
                    )

            self._prepare_ltoir_bundle_for_matches(all_scan_matches)

            for source_order, inst, scan_match, ctor_key in temp_storage_scan_entries:
                invocable, _ = self._materialize_invocable(scan_match)
                size_in_bytes = max(
                    1, int(getattr(invocable, "temp_storage_bytes", 0) or 0)
                )
                alignment = max(
                    1, int(getattr(invocable, "temp_storage_alignment", 0) or 0)
                )

                summary = requirements.setdefault(
                    ctor_key,
                    _TempStorageRequirementSummary(),
                )
                summary.max_size_in_bytes = max(
                    summary.max_size_in_bytes, size_in_bytes
                )
                summary.max_alignment = max(summary.max_alignment, alignment)
                summary.uses.append(
                    _TempStorageUseRequirement(
                        call_assign=inst,
                        order=source_order,
                        size_in_bytes=size_in_bytes,
                        alignment=alignment,
                    )
                )
        finally:
            self._block_defs = saved_block_defs
            self._block = saved_block

        return requirements

    def _compute_parent_ctor_specs(self, func_ir) -> dict[str, _ParentCtorSpec]:
        specs: dict[str, _ParentCtorSpec] = {}
        saved_block_defs = self._block_defs

        try:
            for label in sorted(func_ir.blocks):
                scan_block = func_ir.blocks[label]
                self._block_defs = {
                    inst.target.name: inst.value
                    for inst in scan_block.body
                    if isinstance(inst, ir.Assign)
                }
                for inst in scan_block.body:
                    if not isinstance(inst, ir.Assign):
                        continue
                    call = inst.value
                    if not isinstance(call, ir.Expr) or call.op != "call":
                        continue
                    try:
                        spec = self._resolve_parent_constructor_spec(call)
                    except _DeferredCoopRewrite:
                        return {}
                    if spec is None:
                        continue
                    specs[inst.target.name] = self._merge_parent_ctor_specs(
                        specs.get(inst.target.name),
                        spec,
                    )
        finally:
            self._block_defs = saved_block_defs

        return specs

    def match(self, func_ir, block, typemap, calltypes):
        # Group-first calls need exact launch metadata and whole-function
        # provenance.  Leave their helper constructors untouched during the
        # legacy block-local rewrite; the post-inline planners first lower the
        # group calls and then run this rewrite over the resulting providers.
        from ._group_rewrites import has_group_markers

        if has_group_markers(func_ir):
            return False

        func_ir_identity = id(func_ir)
        if self._func_ir_identity != func_ir_identity:
            self._func_ir_identity = func_ir_identity
            self._func_ir = func_ir
            self._thread_data_specs = {}
            self._temp_storage_plans = {}
            self._temp_storage_global_plan = None
            self._temp_storage_ctor_order = {}
            self._prebundled_specializations = {}
            self._parent_ctor_specs = self._compute_parent_ctor_specs(func_ir)
            try:
                self._func_temp_storage_requirements = (
                    self._compute_func_temp_storage_requirements(func_ir)
                )
            except _DeferredCoopRewrite:
                self._func_temp_storage_requirements = {}
                return False

        self._block = block
        self._block_defs = {
            inst.target.name: inst.value
            for inst in block.body
            if isinstance(inst, ir.Assign)
        }
        self._matches = {}
        self._temp_storage_assigns = set()
        self._parent_ctor_assigns = set()
        self._direct_invocable_temp_storage_calls = {}
        self._parent_ctor_func_vars = set()
        self._temp_storage_func_vars = set()
        self._thread_data_func_vars = set()
        self._typed_group_payload_func_vars = set()
        self._dataclass_invocable_getattrs = {}
        self._dataclass_invocable_func_vars = set()
        self._local_array_literal_shape_rewrites = {}

        for inst in block.body:
            if not isinstance(inst, ir.Assign):
                continue
            if isinstance(inst.value, ir.Expr) and inst.value.op == "getattr":
                invocable = self._resolve_dataclass_invocable_getattr(inst.value)
                if invocable is not None:
                    self._dataclass_invocable_getattrs[inst] = invocable
                    self._dataclass_invocable_func_vars.add(inst.target.name)
                    continue

            call = inst.value
            if not isinstance(call, ir.Expr) or call.op != "call":
                continue

            if self._is_local_array_ctor_call(call):
                shape_ref = self._call_shape_ref(call)
                if self._resolve_dataclass_field_value_ref(shape_ref) is not None:
                    self._local_array_literal_shape_rewrites[inst] = (
                        self._extract_local_array_spec(call)
                    )
                continue

            if self._is_temp_storage_ctor_call(call):
                self._temp_storage_assigns.add(inst)
                self._temp_storage_func_vars.add(call.func.name)
                self._temp_storage_ctor_specs[inst.target.name] = (
                    self._extract_temp_storage_ctor_spec(call)
                )
                self._temp_storage_ctor_order.setdefault(
                    inst.target.name, len(self._temp_storage_ctor_order)
                )
                continue
            if self._is_thread_data_ctor_call(call):
                self._thread_data_func_vars.add(call.func.name)
                self._thread_data_specs[inst.target.name] = (
                    self._merge_thread_data_specs(
                        self._thread_data_specs.get(inst.target.name),
                        self._extract_thread_data_spec(call),
                    )
                )
                continue
            if self._is_typed_group_payload_ctor_call(call):
                self._typed_group_payload_func_vars.add(call.func.name)
                self._thread_data_specs[inst.target.name] = (
                    self._merge_thread_data_specs(
                        self._thread_data_specs.get(inst.target.name),
                        self._extract_typed_group_payload_spec(call),
                    )
                )
                continue

            try:
                parent_ctor_spec = self._resolve_parent_constructor_spec(call)
            except _DeferredCoopRewrite:
                continue
            if parent_ctor_spec is not None:
                self._parent_ctor_assigns.add(inst)
                self._parent_ctor_func_vars.add(call.func.name)
                self._parent_ctor_specs[inst.target.name] = (
                    self._merge_parent_ctor_specs(
                        self._parent_ctor_specs.get(inst.target.name),
                        parent_ctor_spec,
                    )
                )
                continue

            try:
                parent_child_match = self._resolve_parent_child_match(call)
            except _DeferredCoopRewrite:
                continue
            if parent_child_match is not None:
                self._matches[inst] = parent_child_match
                continue

            target = self._resolve_call_target(call)
            if target is None:
                direct_invocable_call = (
                    self._resolve_direct_invocable_temp_storage_call(call)
                )
                if direct_invocable_call is not None:
                    self._direct_invocable_temp_storage_calls[inst] = (
                        direct_invocable_call
                    )
                continue

            op_name = target.factory.__name__
            try:
                (
                    runtime_args,
                    runtime_temp_storage_var,
                    factory_kwargs,
                    factory_kw_value_vars,
                    runtime_arg_constant_replacements,
                ) = self._validate_and_split_args(
                    op_name,
                    call,
                    target.getitem_temp_storage,
                )
            except _DeferredCoopRewrite:
                continue
            (
                physical_warp_tile_origin,
                preserve_root_store_payload,
                root_store_scalar,
            ) = self._extract_group_root_match_metadata(
                op_name=op_name,
                runtime_args=runtime_args,
                factory_kwargs=factory_kwargs,
            )
            self._matches[inst] = _RewriteMatch(
                op_name=op_name,
                factory=target.factory,
                func_var_name=target.func_var_name,
                func_var_name_extra=target.func_var_name_extra,
                runtime_args=runtime_args,
                runtime_temp_storage_var=runtime_temp_storage_var,
                factory_kwargs=factory_kwargs,
                factory_kw_value_vars=factory_kw_value_vars,
                loc=inst.loc,
                runtime_arg_constant_replacements=(runtime_arg_constant_replacements),
                physical_warp_tile_origin=physical_warp_tile_origin,
                preserve_root_store_payload=preserve_root_store_payload,
                root_store_scalar=root_store_scalar,
            )

        if self._deferred_launch_dim_inference:
            # Do not partially erase TempStorage/ThreadData constructors while
            # leaving their launch-dependent primitive calls unresolved. The
            # whole-function planner will request launch metadata and retry the
            # complete function transactionally.
            return False

        return (
            bool(self._matches)
            or bool(self._temp_storage_assigns)
            or bool(self._parent_ctor_assigns)
            or bool(self._direct_invocable_temp_storage_calls)
            or bool(self._thread_data_func_vars)
            or bool(self._typed_group_payload_func_vars)
            or bool(self._dataclass_invocable_getattrs)
            or bool(self._local_array_literal_shape_rewrites)
        )

    def apply(self):
        assert self._block is not None

        refresh_typing_context = True
        call_invocable_globals: dict[ir.Assign, tuple[str, object]] = {}
        func_var_names_to_clear: set[str] = set()
        candidate_dead_factory_kw_vars: set[str] = set()
        temp_storage_global_plan = (
            self._ensure_temp_storage_global_plan()
            if self._temp_storage_ctor_specs
            else None
        )
        for match_inst, match in self._matches.items():
            invocable, created = self._materialize_invocable(match)
            self._record_invocable_specialization(invocable)
            refresh_typing_context |= created
            candidate_dead_factory_kw_vars.update(
                value_var.name for value_var in match.factory_kw_value_vars
            )
            global_name = _next_global_name("single_phase")
            call_invocable_globals[match_inst] = (global_name, invocable)
            func_var_names_to_clear.add(match.func_var_name)
            if match.func_var_name_extra is not None:
                func_var_names_to_clear.add(match.func_var_name_extra)

        new_block = ir.Block(self._block.scope, self._block.loc)
        for inst in self._block.body:
            dataclass_invocable = self._dataclass_invocable_getattrs.get(inst)
            if dataclass_invocable is not None:
                global_name = _next_global_name("gpu_dataclass_invocable")
                new_block.append(
                    ir.Assign(
                        ir.Global(global_name, dataclass_invocable, inst.loc),
                        inst.target,
                        inst.loc,
                    )
                )
                continue

            if (
                isinstance(inst, ir.Assign)
                and isinstance(inst.value, ir.Expr)
                and inst.value.op == "call"
                and isinstance(inst.value.func, ir.Var)
                and inst.value.func.name in self._dataclass_invocable_func_vars
            ):
                temp_storage_var = None
                rewritten_kws = []
                for name, value in inst.value.kws:
                    if name == "temp_storage":
                        if temp_storage_var is not None:
                            raise CoopSinglePhaseRewriteError(
                                "Duplicate dataclass primitive temp_storage keyword."
                            )
                        if not isinstance(value, ir.Var):
                            raise CoopSinglePhaseRewriteError(
                                "Dataclass primitive temp_storage must be a variable."
                            )
                        temp_storage_var = value
                    else:
                        rewritten_kws.append((name, value))
                if temp_storage_var is not None:
                    rewritten_call = ir.Expr.call(
                        inst.value.func,
                        [temp_storage_var, *inst.value.args],
                        tuple(rewritten_kws),
                        inst.loc,
                    )
                    new_block.append(ir.Assign(rewritten_call, inst.target, inst.loc))
                    continue

            local_array_spec = self._local_array_literal_shape_rewrites.get(inst)
            if local_array_spec is not None:
                if local_array_spec.items_per_thread is None:
                    raise CoopSinglePhaseRewriteError(
                        "Failed to infer local array shape from gpu_dataclass field."
                    )
                items_var = ir.Var(
                    inst.target.scope,
                    f"__coop_local_array_items_{next(_GLOBAL_NAME_COUNTER)}__",
                    inst.loc,
                )
                new_block.append(
                    ir.Assign(
                        ir.Const(local_array_spec.items_per_thread, inst.loc),
                        items_var,
                        inst.loc,
                    )
                )
                rewritten_args = list(inst.value.args)
                rewritten_kws = list(inst.value.kws)
                if rewritten_args:
                    rewritten_args[0] = items_var
                elif any(name == "shape" for name, _ in rewritten_kws):
                    rewritten_kws = [
                        (name, items_var if name == "shape" else value)
                        for name, value in rewritten_kws
                    ]
                new_block.append(
                    ir.Assign(
                        ir.Expr.call(
                            inst.value.func,
                            rewritten_args,
                            tuple(rewritten_kws),
                            inst.loc,
                        ),
                        inst.target,
                        inst.loc,
                    )
                )
                continue

            if isinstance(inst, ir.Assign) and inst.target.name in (
                self._thread_data_func_vars | self._typed_group_payload_func_vars
            ):
                module_global_name = _next_global_name("thread_data_mod")
                module_var = ir.Var(
                    inst.target.scope,
                    f"__coop_thread_data_local_{next(_GLOBAL_NAME_COUNTER)}__",
                    inst.loc,
                )
                local_module_var = ir.Var(
                    inst.target.scope,
                    f"__coop_thread_data_local_mod_{next(_GLOBAL_NAME_COUNTER)}__",
                    inst.loc,
                )
                new_block.append(
                    ir.Assign(
                        ir.Global(module_global_name, _cuda_module, inst.loc),
                        module_var,
                        inst.loc,
                    )
                )
                new_block.append(
                    ir.Assign(
                        ir.Expr.getattr(module_var, "local", inst.loc),
                        local_module_var,
                        inst.loc,
                    )
                )
                new_block.append(
                    ir.Assign(
                        ir.Expr.getattr(local_module_var, "array", inst.loc),
                        inst.target,
                        inst.loc,
                    )
                )
                continue

            if (
                isinstance(inst, ir.Assign)
                and inst.target.name in func_var_names_to_clear
            ):
                new_block.append(
                    ir.Assign(
                        ir.Const(None, inst.loc),
                        inst.target,
                        inst.loc,
                    )
                )
                continue

            if (
                isinstance(inst, ir.Assign)
                and inst.target.name in self._parent_ctor_func_vars
            ):
                new_block.append(
                    ir.Assign(
                        ir.Const(None, inst.loc),
                        inst.target,
                        inst.loc,
                    )
                )
                continue

            if (
                isinstance(inst, ir.Assign)
                and inst.target.name in self._temp_storage_func_vars
            ):
                new_block.append(
                    ir.Assign(
                        ir.Const(None, inst.loc),
                        inst.target,
                        inst.loc,
                    )
                )
                continue

            if isinstance(inst, ir.Assign) and inst in self._parent_ctor_assigns:
                # Parent objects are compile-time placeholders for child-method
                # rewrites; runtime value is intentionally inert.
                new_block.append(
                    ir.Assign(
                        ir.Const(None, inst.loc),
                        inst.target,
                        inst.loc,
                    )
                )
                continue

            if (
                isinstance(inst, ir.Assign)
                and isinstance(inst.value, ir.Expr)
                and inst.value.op == "call"
                and (
                    self._is_thread_data_ctor_call(inst.value)
                    or self._is_typed_group_payload_ctor_call(inst.value)
                )
            ):
                is_typed_group_payload = self._is_typed_group_payload_ctor_call(
                    inst.value
                )
                thread_data_spec = self._thread_data_specs.get(inst.target.name)
                if thread_data_spec is not None and thread_data_spec.dtype is None:
                    self._infer_thread_data_dtype_from_writes(inst.target)
                    thread_data_spec = self._thread_data_specs.get(inst.target.name)
                if thread_data_spec is None or thread_data_spec.dtype is None:
                    subject = (
                        "typed group payload"
                        if is_typed_group_payload
                        else "coop.ThreadData(...)"
                    )
                    raise CoopSinglePhaseRewriteError(
                        f"Failed to infer dtype for {subject}. "
                        "Use it with coop primitives that provide enough dtype "
                        "context (runtime args and/or primitive dtype kwargs)."
                    )
                if thread_data_spec.common_v1:
                    from ._common import _validate_common_numeric_dtype

                    try:
                        _validate_common_numeric_dtype(
                            thread_data_spec.dtype,
                            operation="ThreadData",
                        )
                    except (TypeError, ValueError) as exc:
                        raise CoopSinglePhaseRewriteError(str(exc)) from exc

                dtype_global_name = _next_global_name("thread_data_dtype")
                dtype_var = ir.Var(
                    inst.target.scope,
                    f"__coop_thread_data_dtype_var_{next(_GLOBAL_NAME_COUNTER)}__",
                    inst.loc,
                )
                new_block.append(
                    ir.Assign(
                        ir.Global(
                            dtype_global_name,
                            thread_data_spec.dtype,
                            inst.loc,
                        ),
                        dtype_var,
                        inst.loc,
                    )
                )
                rewritten_args = [] if is_typed_group_payload else list(inst.value.args)
                rewritten_kws = [] if is_typed_group_payload else list(inst.value.kws)
                # ``ThreadData`` exposes the Numba-facing ``alignas`` spelling,
                # while the lower-level local-array typer needs that third
                # argument bound as ``alignment``. Leaving a gap for its
                # ``alignment`` parameter makes ``alignas`` remain a keyword
                # after signature folding, which the compiler rejects.
                rewritten_kws = [
                    ("alignment" if name == "alignas" else name, value)
                    for name, value in rewritten_kws
                ]
                if thread_data_spec.items_per_thread is not None:
                    items_var = ir.Var(
                        inst.target.scope,
                        f"__coop_thread_data_items_{next(_GLOBAL_NAME_COUNTER)}__",
                        inst.loc,
                    )
                    new_block.append(
                        ir.Assign(
                            ir.Const(thread_data_spec.items_per_thread, inst.loc),
                            items_var,
                            inst.loc,
                        )
                    )
                    if rewritten_args:
                        rewritten_args[0] = items_var
                    elif any(
                        name in {"shape", "items_per_thread"}
                        for name, _ in rewritten_kws
                    ):
                        rewritten_kws = [
                            ("shape", items_var)
                            if name in {"shape", "items_per_thread"}
                            else (name, value)
                            for name, value in rewritten_kws
                        ]
                    elif is_typed_group_payload:
                        rewritten_args.append(items_var)
                elif is_typed_group_payload:
                    raise CoopSinglePhaseRewriteError(
                        "Failed to infer static extent for typed group payload."
                    )
                if len(rewritten_args) >= 2:
                    rewritten_args[1] = dtype_var
                elif any(name == "dtype" for name, _ in rewritten_kws):
                    rewritten_kws = [
                        (name, dtype_var if name == "dtype" else value)
                        for name, value in rewritten_kws
                    ]
                elif rewritten_args:
                    rewritten_args.append(dtype_var)
                else:
                    rewritten_kws.append(("dtype", dtype_var))
                new_block.append(
                    ir.Assign(
                        ir.Expr.call(
                            inst.value.func,
                            rewritten_args,
                            tuple(rewritten_kws),
                            inst.loc,
                        ),
                        inst.target,
                        inst.loc,
                    )
                )
                continue

            direct_invocable_call = self._direct_invocable_temp_storage_calls.get(inst)
            if direct_invocable_call is not None:
                temp_storage_arg, temp_storage_plan = (
                    self._runtime_temp_storage_arg_for_call(
                        new_block,
                        source_var=direct_invocable_call.temp_storage_var,
                        call_assign=inst,
                    )
                )
                rewritten_call = ir.Expr.call(
                    inst.value.func,
                    [temp_storage_arg, *inst.value.args],
                    direct_invocable_call.rewritten_kws,
                    inst.loc,
                )
                new_block.append(ir.Assign(rewritten_call, inst.target, inst.loc))
                if temp_storage_plan is not None and temp_storage_plan.auto_sync:
                    c_name = getattr(
                        direct_invocable_call.invocable.specialization,
                        "c_name",
                        "",
                    )
                    sync_attr = (
                        "syncwarp" if c_name.startswith("warp_") else "syncthreads"
                    )
                    self._emit_temp_storage_auto_sync(
                        new_block,
                        scope=inst.target.scope,
                        loc=inst.loc,
                        sync_attr=sync_attr,
                    )
                continue

            match = self._matches.get(inst)
            if match is None and inst not in self._temp_storage_assigns:
                new_block.append(inst)
                continue

            if inst in self._temp_storage_assigns:
                ctor_key = self._resolve_temp_storage_ctor_key(inst.target)
                if ctor_key is None:
                    raise CoopSinglePhaseRewriteError(
                        f"Missing TempStorage constructor metadata for '{inst.target.name}'."
                    )
                plan = self._finalize_temp_storage_plan_for_var(ctor_key)
                uses_dynamic_smem = bool(
                    temp_storage_global_plan is not None
                    and temp_storage_global_plan.uses_dynamic_smem
                )
                alloc_size = 0 if uses_dynamic_smem else int(plan.size_in_bytes)
                # Global base offsets only apply to the single dynamic shared-memory
                # backing allocation. Static placeholders each own their array.
                needs_slice_binding = uses_dynamic_smem
                backing_var = inst.target
                if needs_slice_binding:
                    backing_var = ir.Var(
                        inst.target.scope,
                        f"__coop_temp_storage_backing_{next(_GLOBAL_NAME_COUNTER)}__",
                        inst.loc,
                    )
                module_global_name = _next_global_name("temp_storage_mod")
                module_var = ir.Var(
                    inst.target.scope,
                    f"__coop_temp_storage_local_{next(_GLOBAL_NAME_COUNTER)}__",
                    inst.loc,
                )
                shared_module_var = ir.Var(
                    inst.target.scope,
                    f"__coop_temp_storage_shared_mod_{next(_GLOBAL_NAME_COUNTER)}__",
                    inst.loc,
                )
                array_fn_var = ir.Var(
                    inst.target.scope,
                    f"__coop_temp_storage_array_fn_{next(_GLOBAL_NAME_COUNTER)}__",
                    inst.loc,
                )
                bytes_var = ir.Var(
                    inst.target.scope,
                    f"__coop_temp_storage_bytes_{next(_GLOBAL_NAME_COUNTER)}__",
                    inst.loc,
                )
                align_var = ir.Var(
                    inst.target.scope,
                    f"__coop_temp_storage_align_{next(_GLOBAL_NAME_COUNTER)}__",
                    inst.loc,
                )
                dtype_global_name = _next_global_name("temp_storage_dtype")
                dtype_var = ir.Var(
                    inst.target.scope,
                    f"__coop_temp_storage_dtype_var_{next(_GLOBAL_NAME_COUNTER)}__",
                    inst.loc,
                )
                new_block.append(
                    ir.Assign(
                        ir.Global(module_global_name, _cuda_module, inst.loc),
                        module_var,
                        inst.loc,
                    )
                )
                new_block.append(
                    ir.Assign(
                        ir.Expr.getattr(module_var, "shared", inst.loc),
                        shared_module_var,
                        inst.loc,
                    )
                )
                new_block.append(
                    ir.Assign(
                        ir.Expr.getattr(shared_module_var, "array", inst.loc),
                        array_fn_var,
                        inst.loc,
                    )
                )
                new_block.append(
                    ir.Assign(
                        ir.Const(alloc_size, inst.loc),
                        bytes_var,
                        inst.loc,
                    )
                )
                new_block.append(
                    ir.Assign(
                        ir.Const(plan.alignment, inst.loc),
                        align_var,
                        inst.loc,
                    )
                )
                new_block.append(
                    ir.Assign(
                        ir.Global(dtype_global_name, _cuda_module.uint8, inst.loc),
                        dtype_var,
                        inst.loc,
                    )
                )
                temp_storage_call = ir.Expr.call(
                    array_fn_var,
                    [bytes_var, dtype_var, align_var],
                    (),
                    inst.loc,
                )
                new_block.append(
                    ir.Assign(
                        temp_storage_call,
                        backing_var,
                        inst.loc,
                    )
                )
                if needs_slice_binding:
                    self._emit_array_slice(
                        new_block,
                        source_var=backing_var,
                        target_var=inst.target,
                        start=plan.base_offset,
                        stop=plan.base_offset + plan.size_in_bytes,
                        loc=inst.loc,
                    )
                continue

            rewritten_runtime_args = list(match.runtime_args)
            for argument_index, value in match.runtime_arg_constant_replacements:
                replacement_var = ir.Var(
                    inst.target.scope,
                    f"__coop_runtime_constant_{next(_GLOBAL_NAME_COUNTER)}__",
                    match.loc,
                )
                new_block.append(
                    ir.Assign(ir.Const(value, match.loc), replacement_var, match.loc)
                )
                rewritten_runtime_args[argument_index] = replacement_var
            if match.preserve_root_store_payload:
                if len(rewritten_runtime_args) < 2:
                    raise CoopSinglePhaseRewriteError(
                        "root store is missing its value argument"
                    )
                rewritten_runtime_args[1] = self._emit_root_store_payload(
                    new_block,
                    match=match,
                    value=rewritten_runtime_args[1],
                    scope=inst.target.scope,
                    loc=match.loc,
                )
            if match.physical_warp_tile_origin:
                if not rewritten_runtime_args:
                    raise CoopSinglePhaseRewriteError(
                        "root physical-warp load/store is missing its offset argument"
                    )
                user_offset = rewritten_runtime_args.pop()
                rewritten_runtime_args.append(
                    self._emit_physical_warp_tile_offset(
                        new_block,
                        match=match,
                        user_offset=user_offset,
                        scope=inst.target.scope,
                        loc=match.loc,
                    )
                )
            runtime_temp_storage_plan = None
            if match.runtime_temp_storage_var is not None:
                runtime_temp_storage_arg, runtime_temp_storage_plan = (
                    self._runtime_temp_storage_arg_for_call(
                        new_block,
                        source_var=match.runtime_temp_storage_var,
                        call_assign=inst,
                    )
                )
                rewritten_runtime_args.insert(0, runtime_temp_storage_arg)
            call_func = inst.value.func
            call_invocable = call_invocable_globals.get(inst)
            if call_invocable is not None:
                global_name, invocable = call_invocable
                call_func = ir.Var(
                    inst.target.scope,
                    f"__coop_single_phase_call_{next(_GLOBAL_NAME_COUNTER)}__",
                    match.loc,
                )
                new_block.append(
                    ir.Assign(
                        ir.Global(global_name, invocable, match.loc),
                        call_func,
                        match.loc,
                    )
                )
            rewritten_call = ir.Expr.call(
                call_func,
                rewritten_runtime_args,
                (),
                match.loc,
            )
            new_block.append(ir.Assign(rewritten_call, inst.target, match.loc))
            if (
                runtime_temp_storage_plan is not None
                and runtime_temp_storage_plan.auto_sync
            ):
                sync_attr = (
                    "syncthreads" if match.op_name in self._BLOCK_OPS else "syncwarp"
                )
                self._emit_temp_storage_auto_sync(
                    new_block,
                    scope=inst.target.scope,
                    loc=inst.loc,
                    sync_attr=sync_attr,
                )

        # Rewriting removes compile-time factory kwargs from runtime calls.
        # Drop now-dead keyword value assignments (e.g. global dict `methods`)
        # so pre-inference typing does not try to type unsupported runtime objects.
        used_var_names: set[str] = set()
        for stmt in new_block.body:
            stmt_vars = list(stmt.list_vars())
            if isinstance(stmt, ir.Assign):
                stmt_vars = [var for var in stmt_vars if var.name != stmt.target.name]
            used_var_names.update(var.name for var in stmt_vars)

        if candidate_dead_factory_kw_vars:
            filtered_block = ir.Block(new_block.scope, new_block.loc)
            for stmt in new_block.body:
                if (
                    isinstance(stmt, ir.Assign)
                    and stmt.target.name in candidate_dead_factory_kw_vars
                    and stmt.target.name not in used_var_names
                ):
                    continue
                filtered_block.append(stmt)
            new_block = filtered_block

        if refresh_typing_context:
            self._state.typingctx.refresh()

        return new_block


# Register hierarchy lowering first so its root markers become ordinary
# provider factories before the generic cooperative rewrite inspects them.
from . import _group_rewrites as _group_rewrites  # noqa: E402

try:
    from numba_cuda_mlir.extending import (
        WholeFunctionPlanner,
        register_planner,
        require_launch_config,
        set_required_dynamic_shared_memory,
    )
except ImportError:
    # Older numba-cuda-mlir releases retain the block-local compatibility
    # rewrite. Group-aware releases require the whole-function planner API.
    WholeFunctionPlanner = None
else:

    @register_planner
    class CoopWholeFunctionPlanner(WholeFunctionPlanner):
        """Apply cooperative rewrites after device-function inlining."""

        def run(self) -> bool:
            rewrite = CoopSinglePhaseRewrite(self.state)
            modified = False

            def apply_matches() -> None:
                nonlocal modified
                for label in sorted(self.state.func_ir.blocks):
                    block = self.state.func_ir.blocks[label]
                    while rewrite.match(
                        self.state.func_ir,
                        block,
                        self.state.typemap,
                        self.state.calltypes,
                    ):
                        block = rewrite.apply()
                        self.state.func_ir.blocks[label] = block
                        modified = True

            apply_matches()
            if rewrite._deferred_launch_dim_inference and not self.is_device_function:
                require_launch_config(self.state)
                rewrite = CoopSinglePhaseRewrite(self.state)
                apply_matches()
            return modified
