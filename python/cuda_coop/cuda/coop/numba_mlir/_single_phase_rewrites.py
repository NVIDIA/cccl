# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Single-phase cooperative-provider lowering for Numba-CUDA-MLIR."""

from __future__ import annotations

import hashlib
import operator
import struct
from dataclasses import dataclass, field, replace
from itertools import count

import numpy as np
from numba_cuda_mlir import cuda as _cuda_module
from numba_cuda_mlir import types as _numba_types
from numba_cuda_mlir.extending import (
    WholeFunctionPlanner,
    register_planner,
    set_required_dynamic_shared_memory,
)
from numba_cuda_mlir.numba_cuda.core import errors as _numba_errors
from numba_cuda_mlir.numba_cuda.core.rewrites import Rewrite, register_rewrite
from numba_cuda_mlir.numba_cuda.typing.typeof import typeof as _numba_typeof
from numba_cuda_mlir.numbair_transforms import ir

from cuda.coop._core import root_api as _common_root_api
from cuda.coop._core.block import normalize_radix_order, resolve_static_radix_end_bit

from ._common import normalize_dim_param, normalize_dtype_param
from ._types import (
    _hash_symbol_value,
    algo_coalesce_key,
    collect_specializations,
    make_invocable_from_specialization,
    prepare_ltoir_bundle,
)

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
class _ThreadDataSpec:
    items_per_thread: object | None
    dtype: object | None
    common_v1: bool = False

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


@register_rewrite("before-inference")
class CoopSinglePhaseRewrite(Rewrite):
    """Rewrite planner-private providers into two-phase invocable calls."""

    _CUDA_ROOT_MODULES = frozenset({"cuda", "numba_cuda_mlir.cuda"})
    _TEMP_STORAGE_RUNTIME_KW_OPS = frozenset(
        {
            "_common_radix_rank",
            "_common_radix_sort_keys",
            "_common_radix_sort_pairs",
            "load",
            "merge_sort_keys",
            "merge_sort_pairs",
            "radix_rank",
            "radix_sort_keys",
            "radix_sort_keys_descending",
            "radix_sort_pairs",
            "radix_sort_pairs_descending",
            "scan",
            "store",
            "topk_max_keys",
            "topk_max_pairs",
            "topk_min_keys",
            "topk_min_pairs",
            "_common_topk_max_keys",
            "_common_topk_max_pairs",
            "_common_topk_min_keys",
            "_common_topk_min_pairs",
            "_qualified_group_topk_max_keys",
            "_qualified_group_topk_max_pairs",
            "_qualified_group_topk_min_keys",
            "_qualified_group_topk_min_pairs",
            "adjacent_difference",
            "discontinuity",
            "warp_load",
            "warp_store",
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
                "_common_profile_operation",
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
                "_common_profile_operation",
            },
            "required_factory_kwargs": {
                "dtype",
                "threads_per_block",
                "binary_op",
            },
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
                "_common_profile_operation",
            },
            "required_factory_kwargs": {
                "dtype",
                "threads_per_block",
                "binary_op",
            },
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
                "_common_profile_operation",
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
                "_common_profile_operation",
            },
            "required_factory_kwargs": {
                "dtype",
                "threads_per_block",
                "difference_op",
            },
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
                "block_aggregate",
                "algorithm",
                "methods",
                "_common_profile_operation",
            },
            "required_factory_kwargs": {"dtype", "threads_per_block"},
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
                "_common_profile_operation",
            },
            "required_factory_kwargs": {"dtype", "binary_op"},
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
                "valid_items",
                "methods",
                "_common_profile_operation",
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
                "valid_items",
                "_common_profile_operation",
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
                "_common_profile_operation",
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
                "_common_profile_operation",
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
                "_common_profile_operation",
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
                "_common_profile_operation",
            },
            "required_factory_kwargs": {"dtype", "scan_op"},
        },
        "load": {
            "namespace": "block",
            "runtime_arg_counts": {2, 3, 4},
            "runtime_factory_kwargs": ("num_valid_items", "oob_default"),
            "runtime_factory_kw_prerequisites": {"oob_default": "num_valid_items"},
            "allowed_factory_kwargs": {
                "num_valid_items",
                "offset",
                "algorithm",
                "threads_per_block",
                "items_per_thread",
                "_common_profile_operation",
                "dtype",
                "oob_default",
            },
            "required_factory_kwargs": {"threads_per_block", "dtype"},
        },
        "store": {
            "namespace": "block",
            "runtime_arg_counts": {2, 3},
            "runtime_factory_kwargs": ("num_valid_items",),
            "allowed_factory_kwargs": {
                "num_valid_items",
                "offset",
                "algorithm",
                "threads_per_block",
                "_group_root_store",
                "items_per_thread",
                "_common_profile_operation",
                "dtype",
            },
            "required_factory_kwargs": {"threads_per_block", "dtype"},
        },
        "shuffle": {
            "namespace": "block",
            "runtime_arg_counts": {1, 2, 3},
            "allowed_factory_kwargs": {
                "block_suffix",
                "threads_per_block",
                "items_per_thread",
                "_common_profile_operation",
                "distance",
                "dtype",
                "block_shuffle_type",
                "block_prefix",
                "methods",
            },
            "required_factory_kwargs": {"threads_per_block", "dtype"},
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
                "_common_profile_operation",
            },
            "required_factory_kwargs": {"dtype", "threads_per_block", "flag_op"},
        },
        "exchange": {
            "namespace": "block",
            "runtime_arg_counts": {1, 2, 3, 4},
            "allowed_factory_kwargs": {
                "use_output_items",
                "offset_dtype",
                "threads_per_block",
                "items_per_thread",
                "_common_profile_operation",
                "dtype",
                "block_exchange_type",
                "warp_time_slicing",
                "methods",
                "valid_flag_dtype",
            },
            "required_factory_kwargs": {"threads_per_block", "dtype"},
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
                "_common_profile_operation",
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
                "_common_profile_operation",
            },
            "required_factory_kwargs": {
                "keys",
                "values",
                "threads_per_block",
                "compare_op",
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
                "threads_per_block",
                "items_per_thread",
                "num_valid",
                "begin_bit",
                "end_bit",
            },
            "required_factory_kwargs": {"keys", "values", "threads_per_block"},
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
                "threads_per_block",
                "items_per_thread",
                "num_valid",
                "begin_bit",
                "end_bit",
            },
            "required_factory_kwargs": {"keys", "values", "threads_per_block"},
        },
        "warp_load": {
            "namespace": "warp",
            "runtime_arg_counts": {2, 3, 4},
            "runtime_factory_kwargs": ("num_valid_items", "oob_default"),
            "runtime_factory_kw_prerequisites": {"oob_default": "num_valid_items"},
            "allowed_factory_kwargs": {
                "offset",
                "_physical_warp_tile_origin",
                "algorithm",
                "threads_per_block",
                "items_per_thread",
                "_common_profile_operation",
                "threads_in_warp",
                "dtype",
                "methods",
            },
            "required_factory_kwargs": {"dtype"},
        },
        "warp_store": {
            "namespace": "warp",
            "runtime_arg_counts": {2, 3},
            "runtime_factory_kwargs": ("num_valid_items",),
            "allowed_factory_kwargs": {
                "offset",
                "_physical_warp_tile_origin",
                "algorithm",
                "threads_per_block",
                "_group_root_store",
                "items_per_thread",
                "_common_profile_operation",
                "threads_in_warp",
                "dtype",
                "methods",
            },
            "required_factory_kwargs": {"dtype"},
        },
        "warp_exchange": {
            "namespace": "warp",
            "runtime_arg_counts": {2, 3},
            "allowed_factory_kwargs": {
                "warp_exchange_type",
                "use_output_items",
                "offset_dtype",
                "threads_per_block",
                "items_per_thread",
                "_common_profile_operation",
                "threads_in_warp",
                "dtype",
                "methods",
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
                "_common_profile_operation",
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
                "_common_profile_operation",
            },
            "required_factory_kwargs": {
                "keys",
                "values",
                "items_per_thread",
                "compare_op",
            },
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
                "key_dtype",
                "value_dtype",
                "threads_per_block",
                "items_per_thread",
                "blocked_to_striped",
            },
            "required_factory_kwargs": {
                "key_dtype",
                "value_dtype",
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
                "key_dtype",
                "value_dtype",
                "threads_per_block",
                "items_per_thread",
                "blocked_to_striped",
            },
            "required_factory_kwargs": {
                "key_dtype",
                "value_dtype",
                "threads_per_block",
            },
        },
    }
    for _private_name, _public_name in {
        "_common_radix_rank": "radix_rank",
        "_common_radix_sort_keys": "radix_sort_keys",
        "_common_radix_sort_pairs": "radix_sort_pairs",
        "_common_topk_max_keys": "topk_max_keys",
        "_common_topk_max_pairs": "topk_max_pairs",
        "_common_topk_min_keys": "topk_min_keys",
        "_common_topk_min_pairs": "topk_min_pairs",
        "_qualified_group_topk_max_keys": "topk_max_keys",
        "_qualified_group_topk_max_pairs": "topk_max_pairs",
        "_qualified_group_topk_min_keys": "topk_min_keys",
        "_qualified_group_topk_min_pairs": "topk_min_pairs",
    }.items():
        _public_spec = _OP_SPECS[_public_name]
        _OP_SPECS[_private_name] = {
            **_public_spec,
            "allowed_factory_kwargs": set(_public_spec["allowed_factory_kwargs"]),
            "required_factory_kwargs": set(_public_spec["required_factory_kwargs"]),
        }
        if _private_name in {"_common_radix_sort_keys", "_common_radix_sort_pairs"}:
            _OP_SPECS[_private_name]["allowed_factory_kwargs"].add("descending")
        if "topk_" in _private_name:
            _OP_SPECS[_private_name]["runtime_factory_kw_prerequisites"] = {
                "end_bit": "begin_bit"
            }
    del _private_name, _public_name, _public_spec

    _BLOCK_OPS = frozenset(
        name for name, spec in _OP_SPECS.items() if spec["namespace"] == "block"
    )
    _WARP_OPS = frozenset(
        {
            "warp_exchange",
            "warp_exclusive_scan",
            "warp_exclusive_sum",
            "warp_inclusive_scan",
            "warp_inclusive_sum",
            "warp_load",
            "warp_merge_sort_keys",
            "warp_merge_sort_pairs",
            "warp_store",
        }
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
        if lhs_items is not None and rhs_items is not None and (lhs_items != rhs_items):
            raise CoopSinglePhaseRewriteError(
                f"coop single-phase '{op_name}' requires {lhs_name}/{rhs_name} arrays to have matching items_per_thread."
            )

    def __init__(self, state):
        super().__init__(state)
        self._state = state
        self._func_ir = state.func_ir
        self._block: ir.Block | None = None
        self._block_defs: dict[str, object] = {}
        self._matches: dict[ir.Assign, _RewriteMatch] = {}
        self._temp_storage_assigns: set[ir.Assign] = set()
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
        self._arg_type_map = self._build_arg_type_map()
        self._invocable_cache: dict[
            tuple[str, tuple[tuple[str, str, str], ...]], object
        ] = {}
        self._prebundled_specializations: dict[
            tuple[str, tuple[tuple[str, str, str], ...]],
            tuple[object, int | None, int | tuple[int, ...] | None],
        ] = {}

    def _infer_constant(self, value):
        if isinstance(value, ir.Var):
            definition = self._block_defs.get(value.name)
            if isinstance(definition, (ir.Const, ir.Global, ir.FreeVar)):
                return definition.value
        return self._func_ir.infer_constant(value)

    def _build_arg_type_map(self) -> dict[str, object]:
        arg_names = tuple(getattr(self._func_ir, "arg_names", ()) or ())
        arg_types = tuple(getattr(self._state, "args", ()) or ())
        if len(arg_names) != len(arg_types):
            return {}
        return dict(zip(arg_names, arg_types))

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
        return (root, attrs)

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

    def _is_supported_factory(self, obj) -> bool:
        name = getattr(obj, "__name__", None)
        module_name = getattr(obj, "__module__", "")
        if not callable(obj) or name not in self._OP_SPECS:
            return False
        expected_ns = self._OP_SPECS[name]["namespace"]
        if expected_ns == "group":
            return module_name == "cuda.coop.numba_mlir._group_provider"
        private_namespace = f"_{expected_ns}"
        return (
            module_name == f"cuda.coop.numba_mlir.{private_namespace}"
            or module_name.startswith(f"cuda.coop.numba_mlir.{private_namespace}.")
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
        return direct if self._is_supported_factory(direct) else None

    def _extract_1d_extent_literal(self, value_ref):
        try:
            value = self._infer_constant(value_ref)
        except _INFERENCE_EXCEPTIONS:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, tuple) and len(value) == 1 and isinstance(value[0], int):
            return int(value[0])
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], int):
            return int(value[0])
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
            (
                isinstance(definition, ir.Expr)
                and definition.op == "call"
                and self._is_typed_group_payload_ctor_call(definition)
                for definition in self._lookup_definitions(value)
            )
        )

    def _extract_typed_group_payload_spec(
        self, call: ir.Expr, *, seen: set[str] | None = None
    ) -> _ThreadDataSpec:
        if seen is None:
            seen = set()
        if len(call.args) not in {3, 4} or call.kws:
            raise CoopSinglePhaseRewriteError(
                "typed group payload marker requires prototype, array-kind, dtype-policy, and optional explicit-extent arguments"
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
                "typed group payload shape and dtype policy must be compile-time constants"
            ) from exc
        if not isinstance(is_array, bool):
            raise CoopSinglePhaseRewriteError(
                "typed group payload array-kind must be a compile-time bool"
            )
        from ._group_rewrites import _PAYLOAD_DTYPE_INT32, _PAYLOAD_DTYPE_LIKE

        if dtype_policy not in {_PAYLOAD_DTYPE_INT32, _PAYLOAD_DTYPE_LIKE}:
            raise CoopSinglePhaseRewriteError(
                f"unknown typed group payload dtype policy {dtype_policy!r}"
            )
        prototype_spec = self._resolve_array_spec_from_var(prototype, seen=set(seen))
        if len(call.args) == 4:
            try:
                items_per_thread = self._infer_constant(call.args[3])
            except _INFERENCE_EXCEPTIONS as exc:
                raise CoopSinglePhaseRewriteError(
                    "typed group payload explicit extent must be a compile-time positive integer"
                ) from exc
            if (
                isinstance(items_per_thread, bool)
                or not isinstance(items_per_thread, int)
                or items_per_thread < 1
            ):
                raise CoopSinglePhaseRewriteError(
                    "typed group payload explicit extent must be a compile-time positive integer"
                )
        elif is_array:
            items_per_thread = (
                prototype_spec.items_per_thread if prototype_spec is not None else None
            )
        else:
            items_per_thread = 1
        if dtype_policy == _PAYLOAD_DTYPE_INT32:
            dtype = _numba_types.int32
        else:
            dtype = prototype_spec.dtype if prototype_spec is not None else None
            if dtype is None:
                dtype = self._resolve_var_dtype(prototype)
        return _ThreadDataSpec(
            items_per_thread=items_per_thread,
            dtype=dtype,
            common_v1=prototype_spec.common_v1 if prototype_spec is not None else False,
        )

    def _extract_thread_data_spec(self, call: ir.Expr) -> _ThreadDataSpec:
        kw_map = {name: value for name, value in call.kws}
        is_common_root = self._is_common_root_member(call.func, "ThreadData")
        allowed_keywords = {"items_per_thread", "dtype"}
        if not is_common_root:
            allowed_keywords.update(("alignas", "alignment"))
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
                "coop.ThreadData accepts at most items_per_thread and dtype positional arguments."
            )
        if len(extent_refs) > 1:
            names = " and ".join((name for name, _ in extent_refs))
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
        if "alignas" in kw_map and "alignment" in kw_map:
            raise CoopSinglePhaseRewriteError(
                "cuda.coop.numba_mlir.ThreadData accepts only one of "
                "alignas or alignment"
            )
        alignment_ref = kw_map.get("alignment", kw_map.get("alignas"))
        if alignment_ref is not None:
            try:
                alignment = self._infer_constant(alignment_ref)
            except _INFERENCE_EXCEPTIONS as exc:
                raise CoopSinglePhaseRewriteError(
                    "cuda.coop.numba_mlir.ThreadData alignment must be a compile-time positive integer"
                ) from exc
            if (
                isinstance(alignment, bool)
                or not isinstance(alignment, int)
                or alignment < 1
            ):
                raise CoopSinglePhaseRewriteError(
                    "cuda.coop.numba_mlir.ThreadData alignment must be a compile-time positive integer"
                )
            if alignment & (alignment - 1):
                raise CoopSinglePhaseRewriteError(
                    "cuda.coop.numba_mlir.ThreadData alignment must be a power of 2"
                )
            if alignment % _MIN_TEMP_STORAGE_ALIGNMENT:
                raise CoopSinglePhaseRewriteError(
                    "cuda.coop.numba_mlir.ThreadData alignment must be a multiple "
                    f"of {_MIN_TEMP_STORAGE_ALIGNMENT}"
                )
        try:
            raw_items_per_thread = self._infer_constant(items_ref)
        except _INFERENCE_EXCEPTIONS as exc:
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
            if any((dtype is alias for alias in (bool, int, float, complex))):
                dtype = normalize_dtype_param(dtype)
        return _ThreadDataSpec(
            items_per_thread=items_per_thread, dtype=dtype, common_v1=is_common_root
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
            and (existing.items_per_thread != observed.items_per_thread)
        ):
            raise CoopSinglePhaseRewriteError(
                "Inconsistent items_per_thread across merged coop.ThreadData aliases."
            )
        if (
            existing.dtype is not None
            and observed.dtype is not None
            and (existing.dtype != observed.dtype)
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

    def _extract_temp_storage_ctor_spec(self, call: ir.Expr) -> _TempStorageCtorSpec:
        kw_map = {name: value for name, value in call.kws}
        parameter_names = ("size_in_bytes", "alignment", "auto_sync", "sharing")
        if len(call.args) > len(parameter_names):
            raise CoopSinglePhaseRewriteError(
                "TempStorage accepts at most size_in_bytes, alignment, auto_sync, and sharing positional arguments."
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
            if auto_sync is not None and (not isinstance(auto_sync, bool)):
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
                            self._temp_storage_ctor_specs.get(value.name), spec
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
            keys, key=lambda key: (self._temp_storage_ctor_order.get(key, 1 << 30), key)
        )
        for key in ordered_keys:
            spec = self._temp_storage_ctor_specs.get(key)
            if spec is None:
                continue
            merged_spec = self._merge_temp_storage_ctor_specs(merged_spec, spec)
        if merged_spec is None:
            return None
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
                    offset=0, size_in_bytes=entry.size_in_bytes
                )
        else:
            required_size = 0
            for entry in uses:
                required_size = _align_up(required_size, max(1, int(entry.alignment)))
                slices_by_call_id[id(entry.call_assign)] = _TempStorageSlice(
                    offset=required_size, size_in_bytes=entry.size_in_bytes
                )
                required_size += entry.size_in_bytes
        if ctor_spec.size_in_bytes is None:
            if required_size <= 0:
                raise CoopSinglePhaseRewriteError(
                    "TempStorage size_in_bytes must be specified until a "
                    "cooperative primitive provides a storage requirement."
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
                f"TempStorage size_in_bytes is smaller than required by primitive uses ({size_in_bytes} < {required_size})."
            )
        if ctor_spec.alignment is None:
            alignment = _default_temp_storage_alignment(required_alignment)
        else:
            alignment = int(ctor_spec.alignment)
        _validate_temp_storage_alignment(alignment)
        if required_alignment > 0 and alignment < required_alignment:
            raise CoopSinglePhaseRewriteError(
                f"TempStorage alignment is smaller than required by primitive uses ({alignment} < {required_alignment})."
            )
        if ctor_spec.sharing == "exclusive":
            if ctor_spec.auto_sync is True:
                raise CoopSinglePhaseRewriteError(
                    "TempStorage with sharing='exclusive' does not support auto_sync=True."
                )
            auto_sync = False
        else:
            auto_sync = True if ctor_spec.auto_sync is None else ctor_spec.auto_sync
        if not uses and (ctor_spec.sharing != "shared" or ctor_spec.auto_sync is False):
            raise CoopSinglePhaseRewriteError(
                "TempStorage non-default sharing or auto_sync requires a "
                "cooperative primitive to consume the storage descriptor."
            )
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
                            definition, seen=seen
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

    def _resolve_thread_data_spec_from_var(
        self, value: ir.Var, seen: set[str]
    ) -> _ThreadDataSpec | None:
        if not isinstance(value, ir.Var):
            return None
        cached = self._thread_data_specs.get(value.name)
        if (
            cached is not None
            and cached.items_per_thread is not None
            and (cached.dtype is not None)
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
                            definition, seen=seen
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
                                incoming, index, seen=set(seen)
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
                            (
                                incoming
                                for incoming in _phi_incoming_values(definition)
                                if isinstance(incoming, ir.Var)
                            )
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
        self, value: ir.Var, seen: set[str] | None = None
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
                        (
                            incoming
                            for incoming in _phi_incoming_values(definition)
                            if isinstance(incoming, ir.Var)
                        )
                    )
                elif definition.op == "static_getitem":
                    sources = tuple(self._resolve_static_tuple_item_vars(definition))
            for source in sources:
                roots.update(
                    self._collect_thread_data_write_roots(source, seen=set(seen))
                )
        return roots

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
                and (lhs_dtype != rhs_dtype)
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
        if isinstance(value_ref, ir.Var):
            from numba_cuda_mlir import types as numba_mlir_types

            value_type = self._arg_type_map.get(value_ref.name)
            definition = self._lookup_definition(value_ref)
            if isinstance(definition, ir.Arg):
                value_type = self._state.args[definition.index]
            if isinstance(value_type, numba_mlir_types.NoneType) or (
                isinstance(value_type, numba_mlir_types.Omitted)
                and value_type.value is None
            ):
                return None
        if name == "dtype":
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
                f"coop single-phase getitem syntax expects a runtime temp-storage variable: '{factory.__name__}[temp_storage](...)'."
            )
        return _ResolvedCallTarget(
            factory=factory,
            func_var_name=call.func.name,
            func_var_name_extra=func_def.value.name,
            getitem_temp_storage=getitem_temp_storage,
        )

    @staticmethod
    def _lossless_merge_sort_sentinel(value: object, key_dtype: object) -> object:
        from ._common import _NUMBA_MLIR_DTYPE_NAMES

        try:
            key_dtype = normalize_dtype_param(key_dtype)
        except (TypeError, ValueError):
            return value
        dtype_name = _NUMBA_MLIR_DTYPE_NAMES.get(key_dtype)
        if dtype_name is None:
            return value
        numpy_dtype = np.dtype(dtype_name)

        if key_dtype == _numba_types.boolean:
            if not isinstance(value, (bool, np.bool_)):
                raise CoopSinglePhaseRewriteError(
                    "Merge Sort oob_default must have the same bool dtype as keys"
                )
            return np.bool_(value)

        if isinstance(key_dtype, _numba_types.Integer):
            if isinstance(value, (bool, np.bool_)):
                raise CoopSinglePhaseRewriteError(
                    "Merge Sort oob_default must be an integer, not bool"
                )
            try:
                integer = operator.index(value)
            except TypeError as exc:
                raise CoopSinglePhaseRewriteError(
                    "Merge Sort oob_default must be an integer for integer keys"
                ) from exc
            limits = np.iinfo(numpy_dtype)
            if not limits.min <= integer <= limits.max:
                raise CoopSinglePhaseRewriteError(
                    f"Merge Sort oob_default={integer} is not representable "
                    f"in keys dtype {dtype_name}"
                )
            return numpy_dtype.type(integer)

        if isinstance(key_dtype, _numba_types.Float):
            if isinstance(value, (bool, np.bool_)):
                raise CoopSinglePhaseRewriteError(
                    "Merge Sort oob_default must be numeric, not bool"
                )
            if not isinstance(value, (int, float, np.integer, np.floating)):
                raise CoopSinglePhaseRewriteError(
                    "Merge Sort oob_default must be numeric for floating keys"
                )
            with np.errstate(over="ignore", invalid="ignore"):
                converted = numpy_dtype.type(value)
            original_float = float(value)
            converted_float = float(converted)
            exact = original_float == converted_float
            if exact and original_float == 0.0:
                exact = np.signbit(original_float) == np.signbit(converted_float)
            if isinstance(value, (int, np.integer)) and np.isfinite(converted_float):
                exact = exact and int(converted_float) == int(value)
            if not exact or np.isnan(original_float):
                raise CoopSinglePhaseRewriteError(
                    f"Merge Sort oob_default={value!r} is not losslessly "
                    f"representable in keys dtype {dtype_name}"
                )
            return converted

        return value

    def _validate_merge_sort_runtime_controls(
        self,
        *,
        op_name: str,
        runtime_args: list[ir.Var],
        control_vars: dict[str, ir.Var],
        factory_kwargs: dict[str, object],
    ) -> tuple[tuple[int, object], ...]:
        if op_name not in {
            "merge_sort_keys",
            "merge_sort_pairs",
            "warp_merge_sort_keys",
            "warp_merge_sort_pairs",
        }:
            return ()

        operation = "merge_sort_pairs" if "pairs" in op_name else "merge_sort_keys"
        prefix = f"cuda.coop.numba_mlir.{operation}"
        valid_items_var = control_vars.get("valid_items")
        if valid_items_var is not None:
            static_valid_items = self._resolve_factory_kwarg_value(
                "valid_items", valid_items_var
            )
            if static_valid_items is not _UNRESOLVED:
                if isinstance(static_valid_items, (bool, np.bool_)):
                    raise CoopSinglePhaseRewriteError(
                        f"{prefix} valid_items must be an integer, not bool"
                    )
                try:
                    operator.index(static_valid_items)
                except TypeError as exc:
                    raise CoopSinglePhaseRewriteError(
                        f"{prefix} valid_items must be an integer"
                    ) from exc
            else:
                valid_items_dtype = self._resolve_var_dtype(valid_items_var)
                if valid_items_dtype is None:
                    valid_items_dtype = self._resolve_var_numba_type(valid_items_var)
                if valid_items_dtype == _numba_types.boolean:
                    raise CoopSinglePhaseRewriteError(
                        f"{prefix} valid_items must be an integer, not bool"
                    )
                if valid_items_dtype is not None and not isinstance(
                    valid_items_dtype, _numba_types.Integer
                ):
                    raise CoopSinglePhaseRewriteError(
                        f"{prefix} valid_items must have an integer dtype"
                    )

        oob_default_var = control_vars.get("oob_default")
        if oob_default_var is None:
            return ()
        key_dtype = factory_kwargs.get("keys" if "pairs" in op_name else "dtype")
        if key_dtype is None:
            raise CoopSinglePhaseRewriteError(
                f"{prefix} could not infer the keys dtype before validating oob_default"
            )
        static_oob_default = self._resolve_factory_kwarg_value(
            "oob_default", oob_default_var
        )
        if static_oob_default is not _UNRESOLVED:
            converted = self._lossless_merge_sort_sentinel(
                static_oob_default,
                key_dtype,
            )
            argument_index = next(
                index
                for index, argument in enumerate(runtime_args)
                if argument is oob_default_var or argument.name == oob_default_var.name
            )
            return ((argument_index, converted),)

        oob_default_dtype = self._resolve_var_dtype(oob_default_var)
        if oob_default_dtype is None:
            oob_default_dtype = self._resolve_var_numba_type(oob_default_var)
        if oob_default_dtype is None:
            return ()
        try:
            key_dtype = normalize_dtype_param(key_dtype)
            oob_default_dtype = normalize_dtype_param(oob_default_dtype)
        except (TypeError, ValueError):
            pass
        if oob_default_dtype != key_dtype:
            raise CoopSinglePhaseRewriteError(
                f"{prefix} oob_default must have the same dtype as keys "
                f"({key_dtype}); got {oob_default_dtype}"
            )
        return ()

    def _validate_and_split_args(
        self, op_name: str, call: ir.Expr, getitem_temp_storage: ir.Var | None
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
                "coop movement calls do not support *args or **kwargs."
            )
        runtime_arg_count = len(call.args)
        if runtime_arg_count not in spec["runtime_arg_counts"]:
            expected_csv = ", ".join(
                (str(v) for v in sorted(spec["runtime_arg_counts"]))
            )
            raise CoopSinglePhaseRewriteError(
                f"coop movement '{op_name}' expects positional runtime argument count in {{{expected_csv}}}; got {runtime_arg_count}."
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
                    f"coop movement '{op_name}' received too many positional partial-tile arguments."
                )
            for index, name in enumerate(
                runtime_factory_kwargs[:extra_runtime_arg_count]
            ):
                factory_kwargs[name] = True
                seen_factory_kwargs.add(name)
                seen_runtime_factory_kwargs.add(name)
                value_var = runtime_args[base_runtime_arg_count + index]
                if isinstance(value_var, ir.Var):
                    runtime_factory_control_vars[name] = value_var
        if runtime_only_kwargs:
            if extra_runtime_arg_count > len(runtime_only_kwargs):
                raise CoopSinglePhaseRewriteError(
                    f"coop movement '{op_name}' received too many positional "
                    "runtime arguments"
                )
            for name in runtime_only_kwargs[:extra_runtime_arg_count]:
                seen_runtime_only_kwargs.add(name)
        for name, value_var in call.kws:
            if name == "temp_storage" and op_name in self._TEMP_STORAGE_RUNTIME_KW_OPS:
                if runtime_temp_storage is not None:
                    raise CoopSinglePhaseRewriteError(
                        f"Duplicate coop movement '{op_name}' runtime temp storage."
                    )
                if not isinstance(value_var, ir.Var):
                    raise CoopSinglePhaseRewriteError(
                        "coop movement temp_storage must be a variable."
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
                        f"Duplicate coop movement '{op_name}' runtime offset."
                    )
                if not isinstance(value_var, ir.Var):
                    raise CoopSinglePhaseRewriteError(
                        "coop load/store offset must be a variable."
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
                        "coop single-phase 'scan' block_aggregate must be a "
                        "runtime array variable."
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
                        f"Duplicate coop shuffle boundary output '{name}'."
                    )
                if value is not _UNRESOLVED or not isinstance(value_var, ir.Var):
                    raise CoopSinglePhaseRewriteError(
                        "coop shuffle boundary output must be a variable or None."
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
                        f"Duplicate coop movement '{op_name}' runtime argument '{name}'."
                    )
                if not isinstance(value_var, ir.Var):
                    raise CoopSinglePhaseRewriteError(
                        f"coop partial-tile argument '{name}' must be a variable."
                    )
                runtime_factory_kw_vars[name] = value_var
                continue
            if name in runtime_only_kwargs:
                if name in seen_runtime_only_kwargs or name in runtime_only_kw_vars:
                    raise CoopSinglePhaseRewriteError(
                        f"Duplicate coop movement '{op_name}' runtime "
                        f"argument '{name}'."
                    )
                if not isinstance(value_var, ir.Var):
                    raise CoopSinglePhaseRewriteError(
                        f"coop runtime argument '{name}' must be a variable."
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
                    f"Unsupported coop movement '{op_name}' factory keyword '{name}'. Allowed keywords are: {allowed}."
                )
            if name in seen_factory_kwargs:
                raise CoopSinglePhaseRewriteError(
                    f"Duplicate coop movement '{op_name}' factory keyword '{name}'."
                )
            seen_factory_kwargs.add(name)
            value = self._resolve_factory_kwarg_value(name, value_var)
            if value is _UNRESOLVED:
                raise CoopSinglePhaseRewriteError(
                    f"Failed to evaluate coop movement factory argument '{name}' for '{op_name}' as a compile-time constant."
                )
            factory_kwargs[name] = value
            if isinstance(value_var, ir.Var):
                factory_kw_value_vars.append(value_var)
        for name in runtime_factory_kwargs:
            value_var = runtime_factory_kw_vars.get(name)
            if value_var is None:
                continue
            prerequisite = runtime_factory_kw_prerequisites.get(name)
            if (
                prerequisite is not None
                and prerequisite not in seen_runtime_factory_kwargs
                and prerequisite not in runtime_factory_kw_vars
                and prerequisite not in seen_factory_kwargs
            ):
                raise CoopSinglePhaseRewriteError(
                    f"coop movement '{op_name}' runtime argument '{name}' requires '{prerequisite}'."
                )
            runtime_args.append(value_var)
            factory_kwargs[name] = True
            seen_factory_kwargs.add(name)
            seen_runtime_factory_kwargs.add(name)
            runtime_factory_control_vars[name] = value_var
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
                    f"coop movement '{op_name}' runtime argument '{name}' "
                    f"requires '{prerequisite}'."
                )
            runtime_args.append(value_var)
            seen_runtime_only_kwargs.add(name)
        if runtime_offset_var is not None:
            runtime_args.append(runtime_offset_var)
        self._validate_integer_runtime_controls(
            op_name=op_name,
            runtime_args=runtime_args,
            factory_kwargs=factory_kwargs,
        )
        self._infer_factory_kwargs_from_thread_data(
            op_name,
            runtime_args,
            allowed_factory_kwargs,
            seen_factory_kwargs,
            factory_kwargs,
        )
        merge_sort_replacements = self._validate_merge_sort_runtime_controls(
            op_name=op_name,
            runtime_args=runtime_args,
            control_vars=runtime_factory_control_vars,
            factory_kwargs=factory_kwargs,
        )
        radix_sort_replacements = self._radix_sort_runtime_constant_replacements(
            op_name=op_name,
            runtime_args=runtime_args,
            runtime_only_kw_vars=runtime_only_kw_vars,
            factory_kwargs=factory_kwargs,
        )
        runtime_arg_constant_replacements = (
            *merge_sort_replacements,
            *radix_sort_replacements,
        )
        if op_name in {"radix_rank", "_common_radix_rank"}:
            self._finalize_radix_rank_factory_kwargs(
                op_name=op_name,
                runtime_arg_count=runtime_arg_count,
                seen_factory_kwargs=seen_factory_kwargs,
                factory_kwargs=factory_kwargs,
            )
        self._validate_topk_runtime_controls(
            op_name=op_name,
            runtime_args=runtime_args,
            factory_kwargs=factory_kwargs,
        )
        if op_name == "adjacent_difference":
            self._finalize_adjacent_difference_factory_kwargs(
                runtime_arg_count=runtime_arg_count,
                seen_factory_kwargs=seen_factory_kwargs,
                factory_kwargs=factory_kwargs,
            )
        elif op_name == "discontinuity":
            self._finalize_discontinuity_factory_kwargs(
                runtime_arg_count=runtime_arg_count,
                seen_factory_kwargs=seen_factory_kwargs,
                factory_kwargs=factory_kwargs,
            )
            runtime_args = self._reorder_discontinuity_runtime_args(
                runtime_args,
                factory_kwargs,
            )
        elif op_name == "shuffle":
            self._finalize_shuffle_factory_kwargs(
                runtime_arg_count=len(runtime_args),
                seen_factory_kwargs=seen_factory_kwargs,
                factory_kwargs=factory_kwargs,
            )
        elif op_name == "exchange":
            self._finalize_exchange_factory_kwargs(
                runtime_args=runtime_args,
                runtime_arg_count=runtime_arg_count,
                seen_factory_kwargs=seen_factory_kwargs,
                factory_kwargs=factory_kwargs,
            )
        elif op_name == "warp_exchange":
            self._finalize_warp_exchange_factory_kwargs(
                runtime_args=runtime_args,
                runtime_arg_count=runtime_arg_count,
                seen_factory_kwargs=seen_factory_kwargs,
                factory_kwargs=factory_kwargs,
            )
        missing = required_factory_kwargs - seen_factory_kwargs
        if missing:
            missing_csv = ", ".join(sorted(missing))
            raise CoopSinglePhaseRewriteError(
                f"coop movement '{op_name}' requires explicit factory keywords: {missing_csv}."
            )
        if (
            runtime_temp_storage is not None
            and op_name not in self._TEMP_STORAGE_RUNTIME_KW_OPS
        ):
            raise CoopSinglePhaseRewriteError(
                f"coop movement '{op_name}' does not support runtime temp_storage."
            )
        return (
            tuple(runtime_args),
            runtime_temp_storage,
            factory_kwargs,
            tuple(factory_kw_value_vars),
            runtime_arg_constant_replacements,
        )

    def _finalize_radix_rank_factory_kwargs(
        self,
        *,
        op_name: str,
        runtime_arg_count: int,
        seen_factory_kwargs: set[str],
        factory_kwargs: dict[str, object],
    ) -> None:
        if runtime_arg_count not in {2, 3}:
            raise CoopSinglePhaseRewriteError(
                "coop single-phase 'radix_rank' runtime argument count must "
                "be one of {2, 3}."
            )
        scope_name = (
            "cuda.coop" if op_name == "_common_radix_rank" else "cuda.coop.numba_mlir"
        )

        def static_index(name: str, value: object) -> int:
            if isinstance(value, (bool, np.bool_)):
                raise CoopSinglePhaseRewriteError(
                    f"{scope_name}.radix_rank {name} must be an integer"
                )
            try:
                return operator.index(value)
            except TypeError as exc:
                raise CoopSinglePhaseRewriteError(
                    f"{scope_name}.radix_rank {name} must be an integer"
                ) from exc

        begin_bit = static_index("begin_bit", factory_kwargs.get("begin_bit", 0))
        dtype = factory_kwargs.get("dtype")
        bitwidth = getattr(dtype, "bitwidth", None)
        if bitwidth is not None:
            bitwidth = int(bitwidth)
        explicit_end_bit = factory_kwargs.get("end_bit")
        if explicit_end_bit is not None:
            explicit_end_bit = static_index("end_bit", explicit_end_bit)
        try:
            end_bit = resolve_static_radix_end_bit(
                begin_bit=begin_bit,
                end_bit=explicit_end_bit,
                bit_width=bitwidth,
                default_radix_bits=4,
                clamp_default=False,
            )
        except ValueError as exc:
            raise CoopSinglePhaseRewriteError(f"{scope_name}.radix_rank {exc}") from exc
        if end_bit - begin_bit > 8:
            raise CoopSinglePhaseRewriteError(
                f"{scope_name}.radix_rank bit width must be <= 8"
            )
        factory_kwargs["begin_bit"] = begin_bit
        factory_kwargs["end_bit"] = end_bit
        seen_factory_kwargs.update({"begin_bit", "end_bit"})
        if "descending" in seen_factory_kwargs:
            try:
                factory_kwargs["descending"] = normalize_radix_order(
                    factory_kwargs["descending"]
                ).descending
            except ValueError as exc:
                raise CoopSinglePhaseRewriteError(
                    f"{scope_name}.radix_rank descending must be a bool"
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
            "_common_radix_sort_keys",
            "_common_radix_sort_pairs",
            "radix_sort_keys",
            "radix_sort_keys_descending",
            "radix_sort_pairs",
            "radix_sort_pairs_descending",
        }
        if op_name not in radix_sort_operations:
            return ()
        begin_var = runtime_only_kw_vars.get("begin_bit")
        end_var = runtime_only_kw_vars.get("end_bit")
        if begin_var is None and end_var is None:
            return ()
        if begin_var is None or end_var is None:
            return ()

        common_root = op_name.startswith("_common_")
        public_operation = (
            "radix_sort_pairs" if "pairs" in op_name else "radix_sort_keys"
        )
        scope_name = "cuda.coop" if common_root else "cuda.coop.numba_mlir"
        prefix = f"{scope_name}.{public_operation}"

        def static_bound(name: str, value_ref: ir.Var) -> int | None:
            value = self._resolve_factory_kwarg_value(name, value_ref)
            if value is _UNRESOLVED:
                from numba_cuda_mlir import types as numba_mlir_types

                value_type = self._resolve_var_numba_type(value_ref)
                if value_type is None:
                    value_type = self._resolve_var_dtype(value_ref)
                if isinstance(value_type, numba_mlir_types.Boolean) or not isinstance(
                    value_type, numba_mlir_types.Integer
                ):
                    raise CoopSinglePhaseRewriteError(
                        f"{prefix} {name} must have an integer dtype"
                    )
                return None
            if value is None:
                return None
            if isinstance(value, (bool, np.bool_)):
                raise CoopSinglePhaseRewriteError(f"{prefix} {name} must be an integer")
            try:
                return operator.index(value)
            except TypeError as exc:
                raise CoopSinglePhaseRewriteError(
                    f"{prefix} {name} must be an integer"
                ) from exc

        static_begin = static_bound("begin_bit", begin_var)
        dtype = factory_kwargs.get("dtype", factory_kwargs.get("key_dtype"))
        bit_width = getattr(dtype, "bitwidth", None)
        if bit_width is not None:
            bit_width = int(bit_width)
        static_end = static_bound("end_bit", end_var)
        replacements: tuple[tuple[int, object], ...] = ()
        end_value = self._resolve_factory_kwarg_value("end_bit", end_var)
        if end_value is None:
            if bit_width is None:
                raise CoopSinglePhaseRewriteError(
                    f"{prefix} end_bit must be provided when the key dtype "
                    "bit width cannot be inferred"
                )
            static_end = bit_width
            end_index = next(
                index
                for index, argument in enumerate(runtime_args)
                if argument is end_var or argument.name == end_var.name
            )
            replacements = ((end_index, static_end),)
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

    def _validate_integer_runtime_controls(
        self,
        *,
        op_name: str,
        runtime_args: list[ir.Var],
        factory_kwargs: dict[str, object],
    ) -> None:
        """Reject bool and noninteger partial-tile controls before codegen."""

        parameter = None
        index = None
        if op_name in {"reduce", "sum", "block_reduce_builtin"} and factory_kwargs.get(
            "num_valid"
        ):
            parameter, index = "valid_items", 1
        elif op_name in {
            "warp_reduce",
            "warp_sum",
            "warp_reduce_builtin",
        } and factory_kwargs.get("valid_items"):
            parameter, index = "valid_items", 1
        elif op_name in {"warp_exclusive_scan", "warp_inclusive_scan"} and (
            factory_kwargs.get("valid_items")
        ):
            parameter, index = "valid_items", 1
        elif op_name == "adjacent_difference" and factory_kwargs.get("valid_items"):
            parameter, index = "valid_items", 2
        if parameter is None or index is None or index >= len(runtime_args):
            return
        value = runtime_args[index]
        if not isinstance(value, ir.Var):
            raise CoopSinglePhaseRewriteError(
                f"coop single-phase '{op_name}' {parameter} must be an integer"
            )
        from numba_cuda_mlir import types as numba_mlir_types

        value_type = self._resolve_var_numba_type(value)
        if value_type is None:
            value_type = self._resolve_var_dtype(value)
        if isinstance(value_type, numba_mlir_types.Boolean) or not isinstance(
            value_type, numba_mlir_types.Integer
        ):
            raise CoopSinglePhaseRewriteError(
                f"coop single-phase '{op_name}' {parameter} must be an "
                "integer, not bool or a noninteger scalar"
            )

    def _validate_topk_runtime_controls(
        self,
        *,
        op_name: str,
        runtime_args: list[ir.Var],
        factory_kwargs: dict[str, object],
    ) -> None:
        operation = {
            "topk_max_keys": "topk_max_keys",
            "topk_max_pairs": "topk_max_pairs",
            "topk_min_keys": "topk_min_keys",
            "topk_min_pairs": "topk_min_pairs",
            "_common_topk_max_keys": "topk_max_keys",
            "_common_topk_max_pairs": "topk_max_pairs",
            "_common_topk_min_keys": "topk_min_keys",
            "_common_topk_min_pairs": "topk_min_pairs",
            "_qualified_group_topk_max_keys": "topk_max_keys",
            "_qualified_group_topk_max_pairs": "topk_max_pairs",
            "_qualified_group_topk_min_keys": "topk_min_keys",
            "_qualified_group_topk_min_pairs": "topk_min_pairs",
        }.get(op_name)
        if operation is None:
            return

        prefix = (
            f"cuda.coop.{operation}"
            if op_name.startswith("_common_")
            else f"cuda.coop.numba_mlir.{operation}"
        )
        base_count = 3 if operation.endswith("_pairs") else 2
        controls: dict[str, ir.Var] = {"k": runtime_args[base_count - 1]}
        control_index = base_count
        for name in ("num_valid", "begin_bit", "end_bit"):
            if factory_kwargs.get(name) is not True:
                continue
            if control_index >= len(runtime_args):
                raise CoopSinglePhaseRewriteError(
                    f"{prefix} is missing runtime control {name}"
                )
            controls[name] = runtime_args[control_index]
            control_index += 1

        from numba_cuda_mlir import types as numba_mlir_types

        for name, value_var in controls.items():
            value_type = self._resolve_var_numba_type(value_var)
            if value_type is None:
                value_type = self._resolve_var_dtype(value_var)
            if isinstance(value_type, numba_mlir_types.Boolean) or (
                value_type is not None
                and not isinstance(value_type, numba_mlir_types.Integer)
            ):
                public_name = "valid_items" if name == "num_valid" else name
                raise CoopSinglePhaseRewriteError(
                    f"{prefix} {public_name} must have an integer dtype"
                )

        def static_index(name: str) -> int | None:
            value_var = controls.get(name)
            if value_var is None:
                return None
            value = self._resolve_factory_kwarg_value(name, value_var)
            if value is _UNRESOLVED:
                return None
            if isinstance(value, (bool, np.bool_)):
                raise CoopSinglePhaseRewriteError(
                    f"{prefix} {name} must be an int-like scalar"
                )
            try:
                normalized = operator.index(value)
            except TypeError as exc:
                raise CoopSinglePhaseRewriteError(
                    f"{prefix} {name} must be an int-like scalar"
                ) from exc
            if isinstance(normalized, bool):
                raise CoopSinglePhaseRewriteError(
                    f"{prefix} {name} must be an int-like scalar"
                )
            return int(normalized)

        static_k = static_index("k")
        if static_k is not None and static_k <= 0:
            raise CoopSinglePhaseRewriteError(f"{prefix} k must be positive")

        threads_per_block = factory_kwargs.get("threads_per_block")
        items_per_thread = factory_kwargs.get("items_per_thread")
        tile_size = None
        if threads_per_block is not None and isinstance(items_per_thread, int):
            dim = normalize_dim_param(threads_per_block)
            tile_size = dim.x * dim.y * dim.z * items_per_thread
        static_valid = (
            tile_size if "num_valid" not in controls else static_index("num_valid")
        )
        if static_valid is not None and (
            static_valid <= 0 or (tile_size is not None and static_valid > tile_size)
        ):
            raise CoopSinglePhaseRewriteError(
                f"{prefix} valid_items must be in [1, {tile_size}]"
            )
        if (
            static_k is not None
            and static_valid is not None
            and static_k > static_valid
        ):
            raise CoopSinglePhaseRewriteError(f"{prefix} k must be <= valid_items")

        key_dtype = factory_kwargs.get("dtype")
        if key_dtype is None:
            key_dtype = factory_kwargs.get("keys")
        key_width = getattr(key_dtype, "bitwidth", None)
        if key_width is None:
            return
        key_width = int(key_width)
        static_begin = 0 if "begin_bit" not in controls else static_index("begin_bit")
        static_end = key_width if "end_bit" not in controls else static_index("end_bit")
        if static_begin is not None and not 0 <= static_begin < key_width:
            raise CoopSinglePhaseRewriteError(
                f"{prefix} begin_bit must be in [0, {key_width})"
            )
        if static_end is not None and not 0 < static_end <= key_width:
            raise CoopSinglePhaseRewriteError(
                f"{prefix} end_bit must be in (0, {key_width}]"
            )
        if (
            static_begin is not None
            and static_end is not None
            and static_end <= static_begin
        ):
            raise CoopSinglePhaseRewriteError(f"{prefix} end_bit must exceed begin_bit")

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
                "group_reduce": frozenset({"reduce", "sum"}),
                "block_reduce_builtin": frozenset({"reduce", "sum"}),
                "reduce": frozenset({"reduce"}),
                "sum": frozenset({"reduce", "sum"}),
                "warp_reduce_builtin": frozenset({"reduce", "sum"}),
                "warp_reduce": frozenset({"reduce"}),
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
                "warp_exclusive_sum": frozenset({"scan", "exclusive_sum"}),
                "warp_inclusive_sum": frozenset({"scan", "inclusive_sum"}),
                "warp_exclusive_scan": frozenset({"scan", "exclusive_scan"}),
                "warp_inclusive_scan": frozenset({"scan", "inclusive_scan"}),
                "exchange": frozenset({"exchange"}),
                "warp_exchange": frozenset({"exchange"}),
                "adjacent_difference": frozenset({"adjacent_difference"}),
                "discontinuity": frozenset({"discontinuity"}),
                "shuffle": frozenset({"shuffle"}),
                "merge_sort_keys": frozenset({"merge_sort_keys"}),
                "merge_sort_pairs": frozenset({"merge_sort_pairs"}),
                "warp_merge_sort_keys": frozenset({"merge_sort_keys"}),
                "warp_merge_sort_pairs": frozenset({"merge_sort_pairs"}),
            }
            if common_profile_operation not in operation_families.get(
                op_name, frozenset()
            ):
                raise CoopSinglePhaseRewriteError(
                    "_common_profile_operation does not match the rewritten group operation"
                )
            from ._common import (
                _validate_common_integer_key_dtype,
                _validate_common_numeric_dtype,
            )

            if op_name in {"merge_sort_keys", "warp_merge_sort_keys"}:
                try:
                    _validate_common_integer_key_dtype(
                        factory_kwargs.get("dtype"),
                        operation=common_profile_operation,
                    )
                except (TypeError, ValueError) as exc:
                    raise CoopSinglePhaseRewriteError(str(exc)) from exc
            elif op_name in {"merge_sort_pairs", "warp_merge_sort_pairs"}:
                try:
                    _validate_common_integer_key_dtype(
                        factory_kwargs.get("keys"),
                        operation=common_profile_operation,
                    )
                    _validate_common_numeric_dtype(
                        factory_kwargs.get("values"),
                        operation=common_profile_operation,
                        parameter="values",
                    )
                except (TypeError, ValueError) as exc:
                    raise CoopSinglePhaseRewriteError(str(exc)) from exc
            elif op_name in {"load", "warp_load", "store", "warp_store"}:
                operand_names = (
                    ("source", "output")
                    if op_name in {"load", "warp_load"}
                    else ("destination", "value")
                )
                for operand_name, operand in zip(operand_names, runtime_args):
                    operand_dtype = self._resolve_var_dtype(operand)
                    if operand_dtype is None:
                        raise CoopSinglePhaseRewriteError(
                            f"Failed to infer cuda.coop.{common_profile_operation} {operand_name} dtype for common V1 validation."
                        )
                    try:
                        _validate_common_numeric_dtype(
                            operand_dtype, operation=common_profile_operation
                        )
                    except (TypeError, ValueError) as exc:
                        raise CoopSinglePhaseRewriteError(str(exc)) from exc
            else:
                try:
                    _validate_common_numeric_dtype(
                        factory_kwargs.get("dtype"), operation=common_profile_operation
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
            preserve_root_store_payload = (
                root_store_scalar
                or self._store_algorithm_mutates_payload(
                    op_name, factory_kwargs.get("algorithm", "direct")
                )
            )
        return (
            physical_warp_tile_origin,
            preserve_root_store_payload,
            root_store_scalar,
        )

    def _finalize_adjacent_difference_factory_kwargs(
        self,
        runtime_arg_count: int,
        seen_factory_kwargs: set[str],
        factory_kwargs: dict[str, object],
    ) -> None:
        from ._block._block_adjacent_difference import (
            BlockAdjacentDifferenceType,
        )

        adjacent_type = factory_kwargs.get(
            "block_adjacent_difference_type",
            BlockAdjacentDifferenceType.SubtractLeft,
        )
        try:
            adjacent_type = BlockAdjacentDifferenceType(adjacent_type)
        except (TypeError, ValueError) as exc:
            raise CoopSinglePhaseRewriteError(
                "coop single-phase 'adjacent_difference' "
                "block_adjacent_difference_type must be a "
                "BlockAdjacentDifferenceType value."
            ) from exc

        if adjacent_type is BlockAdjacentDifferenceType.SubtractLeft:
            tile_kw = "tile_predecessor_item"
            invalid_tile_kw = "tile_successor_item"
        else:
            tile_kw = "tile_successor_item"
            invalid_tile_kw = "tile_predecessor_item"
        if invalid_tile_kw in seen_factory_kwargs:
            raise CoopSinglePhaseRewriteError(
                "coop single-phase 'adjacent_difference' received invalid "
                f"'{invalid_tile_kw}' for {adjacent_type.name}."
            )

        has_valid_items = "valid_items" in seen_factory_kwargs
        has_boundary = tile_kw in seen_factory_kwargs
        if runtime_arg_count == 4:
            if not has_valid_items:
                factory_kwargs["valid_items"] = True
                seen_factory_kwargs.add("valid_items")
                has_valid_items = True
            if not has_boundary:
                factory_kwargs[tile_kw] = True
                seen_factory_kwargs.add(tile_kw)
                has_boundary = True
        elif runtime_arg_count == 3:
            if has_valid_items and has_boundary:
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase 'adjacent_difference' cannot map three "
                    "runtime arguments to both valid_items and a boundary item."
                )
            if not has_valid_items and not has_boundary:
                factory_kwargs["valid_items"] = True
                seen_factory_kwargs.add("valid_items")
                has_valid_items = True
        elif runtime_arg_count != 2:
            raise CoopSinglePhaseRewriteError(
                "coop single-phase 'adjacent_difference' expects two, three, "
                "or four runtime arguments."
            )

        if (
            adjacent_type is BlockAdjacentDifferenceType.SubtractRight
            and has_valid_items
            and has_boundary
        ):
            raise CoopSinglePhaseRewriteError(
                "coop single-phase 'adjacent_difference' cannot combine a "
                "right partial tile with tile_successor_item."
            )
        expected_count = 2 + int(has_valid_items) + int(has_boundary)
        if runtime_arg_count != expected_count:
            raise CoopSinglePhaseRewriteError(
                "coop single-phase 'adjacent_difference' runtime argument "
                f"count {runtime_arg_count} does not match {expected_count}."
            )

    def _finalize_discontinuity_factory_kwargs(
        self,
        runtime_arg_count: int,
        seen_factory_kwargs: set[str],
        factory_kwargs: dict[str, object],
    ) -> None:
        from ._block._block_discontinuity import BlockDiscontinuityType

        discontinuity_type = factory_kwargs.get(
            "block_discontinuity_type",
            BlockDiscontinuityType.HEADS,
        )
        try:
            discontinuity_type = BlockDiscontinuityType(discontinuity_type)
        except (TypeError, ValueError) as exc:
            raise CoopSinglePhaseRewriteError(
                "coop single-phase 'discontinuity' block_discontinuity_type "
                "must be a BlockDiscontinuityType value."
            ) from exc

        has_predecessor = "tile_predecessor_item" in seen_factory_kwargs
        has_successor = "tile_successor_item" in seen_factory_kwargs
        if discontinuity_type is BlockDiscontinuityType.HEADS:
            if has_successor:
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase 'discontinuity' HEADS does not accept "
                    "tile_successor_item."
                )
            if runtime_arg_count == 3 and not has_predecessor:
                factory_kwargs["tile_predecessor_item"] = True
                seen_factory_kwargs.add("tile_predecessor_item")
                has_predecessor = True
            expected_count = 2 + int(has_predecessor)
        elif discontinuity_type is BlockDiscontinuityType.TAILS:
            if has_predecessor:
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase 'discontinuity' TAILS does not accept "
                    "tile_predecessor_item."
                )
            if runtime_arg_count == 3 and not has_successor:
                factory_kwargs["tile_successor_item"] = True
                seen_factory_kwargs.add("tile_successor_item")
                has_successor = True
            expected_count = 2 + int(has_successor)
        else:
            if runtime_arg_count == 5:
                if not has_predecessor:
                    factory_kwargs["tile_predecessor_item"] = True
                    seen_factory_kwargs.add("tile_predecessor_item")
                    has_predecessor = True
                if not has_successor:
                    factory_kwargs["tile_successor_item"] = True
                    seen_factory_kwargs.add("tile_successor_item")
                    has_successor = True
            elif runtime_arg_count == 4:
                if has_predecessor and has_successor:
                    raise CoopSinglePhaseRewriteError(
                        "coop single-phase 'discontinuity' cannot map four "
                        "runtime arguments to both boundary items."
                    )
                if not has_predecessor and not has_successor:
                    factory_kwargs["tile_predecessor_item"] = True
                    seen_factory_kwargs.add("tile_predecessor_item")
                    has_predecessor = True
            expected_count = 3 + int(has_predecessor) + int(has_successor)
        if runtime_arg_count != expected_count:
            raise CoopSinglePhaseRewriteError(
                "coop single-phase 'discontinuity' runtime argument count "
                f"{runtime_arg_count} does not match {expected_count}."
            )

    @staticmethod
    def _reorder_discontinuity_runtime_args(
        runtime_args: list[ir.Var],
        factory_kwargs: dict[str, object],
    ) -> list[ir.Var]:
        from ._block._block_discontinuity import BlockDiscontinuityType

        discontinuity_type = BlockDiscontinuityType(
            factory_kwargs.get(
                "block_discontinuity_type",
                BlockDiscontinuityType.HEADS,
            )
        )
        if discontinuity_type in {
            BlockDiscontinuityType.HEADS,
            BlockDiscontinuityType.TAILS,
        }:
            if len(runtime_args) < 2:
                return runtime_args
            return [runtime_args[1], runtime_args[0], *runtime_args[2:]]
        if len(runtime_args) < 3:
            return runtime_args

        input_items, head_flags, tail_flags = runtime_args[:3]
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

    def _finalize_shuffle_factory_kwargs(
        self,
        runtime_arg_count: int,
        seen_factory_kwargs: set[str],
        factory_kwargs: dict[str, object],
    ) -> None:
        from cuda.coop.numba_mlir._block._block_shuffle import (
            BlockShuffleType,
            _normalize_shuffle_type,
        )

        try:
            shuffle_type = _normalize_shuffle_type(
                factory_kwargs.get("block_shuffle_type", BlockShuffleType.Up)
            )
        except (TypeError, ValueError) as exc:
            raise CoopSinglePhaseRewriteError(str(exc)) from exc
        if runtime_arg_count == 1:
            if "items_per_thread" in seen_factory_kwargs:
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase scalar 'shuffle' does not accept items_per_thread."
                )
            if (
                "block_prefix" in seen_factory_kwargs
                or "block_suffix" in seen_factory_kwargs
            ):
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase scalar 'shuffle' does not accept block_prefix/block_suffix."
                )
            if "distance" not in seen_factory_kwargs:
                factory_kwargs["distance"] = 1
                seen_factory_kwargs.add("distance")
            return
        if runtime_arg_count not in {2, 3}:
            raise CoopSinglePhaseRewriteError(
                "coop single-phase 'shuffle' runtime argument count must be one of {1, 2, 3}."
            )
        if shuffle_type not in {BlockShuffleType.Up, BlockShuffleType.Down}:
            raise CoopSinglePhaseRewriteError(
                "coop single-phase array 'shuffle' only supports BlockShuffleType.Up/Down."
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
                    "coop single-phase array 'shuffle' received block_prefix/block_suffix without a matching runtime boundary argument."
                )
            return
        if shuffle_type == BlockShuffleType.Up:
            if "block_prefix" in seen_factory_kwargs:
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase array 'shuffle' with BlockShuffleType.Up does not support block_prefix."
                )
            if "block_suffix" not in seen_factory_kwargs:
                factory_kwargs["block_suffix"] = True
                seen_factory_kwargs.add("block_suffix")
        else:
            if "block_suffix" in seen_factory_kwargs:
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase array 'shuffle' with BlockShuffleType.Down does not support block_suffix."
                )
            if "block_prefix" not in seen_factory_kwargs:
                factory_kwargs["block_prefix"] = True
                seen_factory_kwargs.add("block_prefix")

    def _finalize_exchange_factory_kwargs(
        self,
        runtime_args: list[ir.Var],
        runtime_arg_count: int,
        seen_factory_kwargs: set[str],
        factory_kwargs: dict[str, object],
    ) -> None:
        from cuda.coop.numba_mlir._block._block_exchange import (
            BlockExchangeType,
            _normalize_exchange_type,
        )

        try:
            exchange_type = _normalize_exchange_type(
                factory_kwargs.get(
                    "block_exchange_type", BlockExchangeType.StripedToBlocked
                )
            )
        except (TypeError, ValueError) as exc:
            raise CoopSinglePhaseRewriteError(str(exc)) from exc
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
            expected_csv = ", ".join((str(v) for v in sorted(expected_counts)))
            raise CoopSinglePhaseRewriteError(
                f"coop single-phase 'exchange' runtime argument count {runtime_arg_count} is incompatible with block_exchange_type={exchange_type.name}; expected one of {{{expected_csv}}}."
            )
        out_of_place = runtime_arg_count in {2, 3, 4} and (
            not uses_ranks
            and runtime_arg_count == 2
            or (uses_ranks and (not uses_valid_flags) and (runtime_arg_count == 3))
            or (uses_valid_flags and runtime_arg_count == 4)
        )
        if "use_output_items" in seen_factory_kwargs:
            requested_value_form = factory_kwargs["use_output_items"]
            if requested_value_form is not None and (
                not isinstance(requested_value_form, bool)
            ):
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase 'exchange' use_output_items must be a boolean or None."
                )
            if (
                requested_value_form is not None
                and requested_value_form != out_of_place
            ):
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase 'exchange' use_output_items does not match the runtime argument form."
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
        from cuda.coop.numba_mlir._warp._warp_exchange import (
            WarpExchangeType,
            _normalize_exchange_type,
        )

        try:
            exchange_type = _normalize_exchange_type(
                factory_kwargs.get(
                    "warp_exchange_type", WarpExchangeType.StripedToBlocked
                )
            )
        except (TypeError, ValueError) as exc:
            raise CoopSinglePhaseRewriteError(str(exc)) from exc
        uses_ranks = exchange_type == WarpExchangeType.ScatterToStriped
        expected_counts = {2, 3} if uses_ranks else {2}
        if runtime_arg_count not in expected_counts:
            expected_csv = ", ".join((str(v) for v in sorted(expected_counts)))
            raise CoopSinglePhaseRewriteError(
                f"coop single-phase 'warp_exchange' runtime argument count {runtime_arg_count} is incompatible with warp_exchange_type={exchange_type.name}; expected one of {{{expected_csv}}}."
            )
        if uses_ranks:
            inferred_use_output_items = runtime_arg_count == 3
            if "use_output_items" in seen_factory_kwargs:
                if factory_kwargs["use_output_items"] != inferred_use_output_items:
                    raise CoopSinglePhaseRewriteError(
                        "warp_exchange use_output_items does not match the runtime argument count."
                    )
            else:
                factory_kwargs["use_output_items"] = inferred_use_output_items
                seen_factory_kwargs.add("use_output_items")
            ranks_var = runtime_args[2 if inferred_use_output_items else 1]
            if not isinstance(ranks_var, ir.Var):
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase 'warp_exchange' ranks runtime argument must be a variable."
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
        common_operation = {
            "_common_radix_rank": "radix_rank",
            "_common_radix_sort_keys": "radix_sort_keys",
            "_common_radix_sort_pairs": "radix_sort_pairs",
        }.get(op_name)
        op_name = {
            "_common_radix_rank": "radix_rank",
            "_common_radix_sort_keys": "radix_sort_keys",
            "_common_radix_sort_pairs": "radix_sort_pairs",
        }.get(op_name, op_name)

        common_topk_operation = {
            "_common_topk_max_keys": "topk_max_keys",
            "_common_topk_max_pairs": "topk_max_pairs",
            "_common_topk_min_keys": "topk_min_keys",
            "_common_topk_min_pairs": "topk_min_pairs",
        }.get(op_name)
        op_name = {
            "_common_topk_max_keys": "topk_max_keys",
            "_common_topk_max_pairs": "topk_max_pairs",
            "_common_topk_min_keys": "topk_min_keys",
            "_common_topk_min_pairs": "topk_min_pairs",
            "_qualified_group_topk_max_keys": "topk_max_keys",
            "_qualified_group_topk_max_pairs": "topk_max_pairs",
            "_qualified_group_topk_min_keys": "topk_min_keys",
            "_qualified_group_topk_min_pairs": "topk_min_pairs",
        }.get(op_name, op_name)

        def factory_value(name: str):
            return factory_kwargs.get(name)

        def validate_integer_key_dtype(dtype):
            if dtype is None:
                return None
            from ._common import _validate_common_integer_key_dtype

            try:
                return _validate_common_integer_key_dtype(
                    dtype, operation=common_operation or op_name
                )
            except (TypeError, ValueError) as exc:
                raise CoopSinglePhaseRewriteError(str(exc)) from exc

        def validate_numeric_value_dtype(dtype):
            if dtype is None:
                return None
            from ._common import _validate_common_numeric_dtype

            try:
                return _validate_common_numeric_dtype(
                    dtype,
                    operation=common_operation or op_name,
                    parameter="value",
                )
            except (TypeError, ValueError) as exc:
                raise CoopSinglePhaseRewriteError(str(exc)) from exc

        def factory_kwarg_matches(name: str, actual, expected) -> bool:
            if name in {"dtype", "keys", "values"}:
                try:
                    actual = normalize_dtype_param(actual)
                    expected = normalize_dtype_param(expected)
                except (TypeError, ValueError):
                    pass
            return actual == expected

        def infer_kwarg(name: str, value) -> None:
            if name not in allowed_factory_kwargs or value is None:
                return
            if name in seen_factory_kwargs:
                if not factory_kwarg_matches(name, factory_kwargs[name], value):
                    raise CoopSinglePhaseRewriteError(
                        f"coop movement {op_name!r} {name} does not match the value inferred from ThreadData."
                    )
                return
            factory_kwargs[name] = value
            seen_factory_kwargs.add(name)

        def candidate(index: int) -> tuple[ir.Var | None, _ThreadDataSpec | None]:
            if not 0 <= index < len(runtime_args):
                return (None, None)
            value = runtime_args[index]
            if not isinstance(value, ir.Var):
                return (None, None)
            spec = self._resolve_thread_data_spec(value)
            if self._is_typed_group_payload_var(value) and (
                spec is None or spec.items_per_thread is None
            ):
                raise CoopSinglePhaseRewriteError(
                    f"coop movement {op_name!r} could not infer the static extent of a typed group payload"
                )
            return (value, spec)

        def validate_common_key_dtype(dtype):
            if common_topk_operation is None or dtype is None:
                return dtype
            from ._common import _validate_common_integer_key_dtype

            try:
                return _validate_common_integer_key_dtype(
                    dtype,
                    operation=common_topk_operation,
                )
            except TypeError as exc:
                raise CoopSinglePhaseRewriteError(str(exc)) from exc

        def validate_common_value_dtype(dtype):
            if common_topk_operation is None or dtype is None:
                return dtype
            from ._common import _validate_common_numeric_dtype

            try:
                return _validate_common_numeric_dtype(
                    dtype,
                    operation=common_topk_operation,
                    parameter="value",
                )
            except TypeError as exc:
                raise CoopSinglePhaseRewriteError(str(exc)) from exc

        if op_name in {"topk_max_keys", "topk_min_keys"}:
            key_var, key_spec = candidate(0)
            if key_spec is not None:
                infer_kwarg("items_per_thread", key_spec.items_per_thread)
            key_dtype = key_spec.dtype if key_spec is not None else None
            if key_dtype is None and key_var is not None:
                key_dtype = self._resolve_var_dtype(key_var)
            if key_dtype is None:
                key_dtype = factory_value("dtype")
            key_dtype = validate_common_key_dtype(key_dtype)
            infer_kwarg("dtype", key_dtype)
            if key_dtype is not None and key_var is not None:
                self._record_inferred_thread_data_dtype(key_var, key_dtype)
            return

        if op_name in {"topk_max_pairs", "topk_min_pairs"}:
            key_var, key_spec = candidate(0)
            value_var, value_spec = candidate(1)
            self._require_matching_items_per_thread(
                op_name,
                "key",
                key_spec,
                "value",
                value_spec,
            )
            extent = key_spec.items_per_thread if key_spec is not None else None
            if extent is None and value_spec is not None:
                extent = value_spec.items_per_thread
            infer_kwarg("items_per_thread", extent)
            key_dtype = key_spec.dtype if key_spec is not None else None
            value_dtype = value_spec.dtype if value_spec is not None else None
            if key_dtype is None and key_var is not None:
                key_dtype = self._resolve_var_dtype(key_var)
            if value_dtype is None and value_var is not None:
                value_dtype = self._resolve_var_dtype(value_var)
            if key_dtype is None:
                key_dtype = factory_value("keys")
            if value_dtype is None:
                value_dtype = factory_value("values")
            key_dtype = validate_common_key_dtype(key_dtype)
            value_dtype = validate_common_value_dtype(value_dtype)
            infer_kwarg("keys", key_dtype)
            infer_kwarg("values", value_dtype)
            if key_dtype is not None and key_var is not None:
                self._record_inferred_thread_data_dtype(key_var, key_dtype)
            if value_dtype is not None and value_var is not None:
                self._record_inferred_thread_data_dtype(value_var, value_dtype)
            return

        if op_name in {
            "group_reduce",
            "block_reduce_builtin",
            "reduce",
            "sum",
            "warp_reduce_builtin",
            "warp_reduce",
            "warp_sum",
        }:
            payload_var, payload_spec = candidate(0)
            if payload_spec is not None:
                infer_kwarg("items_per_thread", payload_spec.items_per_thread)
            inferred_dtype = payload_spec.dtype if payload_spec is not None else None
            if inferred_dtype is None and payload_var is not None:
                inferred_dtype = self._resolve_var_dtype(payload_var)
            if inferred_dtype is None:
                inferred_dtype = factory_value("dtype")
            infer_kwarg("dtype", inferred_dtype)
            if inferred_dtype is not None and payload_var is not None:
                self._record_inferred_thread_data_dtype(payload_var, inferred_dtype)
            return

        if op_name == "scan":
            input_var, input_spec = candidate(0)
            output_var, output_spec = candidate(1)
            self._require_matching_items_per_thread(
                op_name, "input", input_spec, "output", output_spec
            )
            input_dtype = input_spec.dtype if input_spec is not None else None
            output_dtype = output_spec.dtype if output_spec is not None else None
            if input_dtype is None and input_var is not None:
                input_dtype = self._resolve_var_dtype(input_var)
            if output_dtype is None and output_var is not None:
                output_dtype = self._resolve_var_dtype(output_var)
            if (
                input_dtype is not None
                and output_dtype is not None
                and not _dtype_values_match(input_dtype, output_dtype)
            ):
                raise CoopSinglePhaseRewriteError(
                    "coop scan requires input/output arrays to have matching dtype."
                )
            inferred_dtype = input_dtype
            if inferred_dtype is None:
                inferred_dtype = output_dtype
            if inferred_dtype is None:
                inferred_dtype = factory_value("dtype")
            extent = input_spec.items_per_thread if input_spec is not None else None
            if extent is None and output_spec is not None:
                extent = output_spec.items_per_thread
            infer_kwarg("items_per_thread", extent)
            infer_kwarg("dtype", inferred_dtype)
            for payload_var in (input_var, output_var):
                if inferred_dtype is not None and payload_var is not None:
                    self._record_inferred_thread_data_dtype(payload_var, inferred_dtype)
            if factory_kwargs.get("block_aggregate"):
                aggregate_var, aggregate_spec = candidate(2)
                if aggregate_spec is None or aggregate_spec.items_per_thread != 1:
                    raise CoopSinglePhaseRewriteError(
                        "coop scan block_aggregate must be a one-item "
                        "ThreadData or local array."
                    )
                aggregate_dtype = aggregate_spec.dtype
                if aggregate_dtype is None and aggregate_var is not None:
                    aggregate_dtype = self._resolve_var_dtype(aggregate_var)
                if (
                    inferred_dtype is not None
                    and aggregate_dtype is not None
                    and not _dtype_values_match(inferred_dtype, aggregate_dtype)
                ):
                    raise CoopSinglePhaseRewriteError(
                        "coop scan block_aggregate dtype must match the input dtype."
                    )
                if aggregate_var is not None and inferred_dtype is not None:
                    self._record_inferred_thread_data_dtype(
                        aggregate_var, inferred_dtype
                    )
            return

        if op_name == "adjacent_difference":
            input_var, input_spec = candidate(0)
            output_var, output_spec = candidate(1)
            self._require_matching_items_per_thread(
                op_name,
                "input",
                input_spec,
                "output",
                output_spec,
            )
            extent = input_spec.items_per_thread if input_spec is not None else None
            if extent is None and output_spec is not None:
                extent = output_spec.items_per_thread
            infer_kwarg("items_per_thread", extent)

            input_dtype = input_spec.dtype if input_spec is not None else None
            output_dtype = output_spec.dtype if output_spec is not None else None
            if input_dtype is None and input_var is not None:
                input_dtype = self._resolve_var_dtype(input_var)
            if output_dtype is None and output_var is not None:
                output_dtype = self._resolve_var_dtype(output_var)
            if (
                input_dtype is not None
                and output_dtype is not None
                and not _dtype_values_match(input_dtype, output_dtype)
            ):
                raise CoopSinglePhaseRewriteError(
                    "coop adjacent_difference input and output dtypes must match."
                )
            inferred_dtype = input_dtype
            if inferred_dtype is None:
                inferred_dtype = output_dtype
            if inferred_dtype is None:
                inferred_dtype = factory_value("dtype")
            infer_kwarg("dtype", inferred_dtype)
            for payload_var in (input_var, output_var):
                if inferred_dtype is not None and payload_var is not None:
                    self._record_inferred_thread_data_dtype(
                        payload_var,
                        inferred_dtype,
                    )

            boundary_index = 2 + int(bool(factory_kwargs.get("valid_items")))
            boundary_name = None
            if factory_kwargs.get("tile_predecessor_item"):
                boundary_name = "tile_predecessor_item"
            elif factory_kwargs.get("tile_successor_item"):
                boundary_name = "tile_successor_item"
            if boundary_name is not None and boundary_index < len(runtime_args):
                boundary_dtype = self._resolve_var_dtype(runtime_args[boundary_index])
                if (
                    inferred_dtype is not None
                    and boundary_dtype is not None
                    and not _dtype_values_match(inferred_dtype, boundary_dtype)
                ):
                    raise CoopSinglePhaseRewriteError(
                        "coop adjacent_difference boundary dtype must match "
                        "the input dtype."
                    )
            return

        if op_name == "discontinuity":
            from ._block._block_discontinuity import BlockDiscontinuityType

            mode = BlockDiscontinuityType(
                factory_kwargs.get(
                    "block_discontinuity_type",
                    BlockDiscontinuityType.HEADS,
                )
            )
            input_var, input_spec = candidate(0)
            head_var, head_spec = candidate(1)
            tail_var, tail_spec = (
                candidate(2)
                if mode is BlockDiscontinuityType.HEADS_AND_TAILS
                else (None, None)
            )
            self._require_matching_items_per_thread(
                op_name,
                "input",
                input_spec,
                "head flags",
                head_spec,
            )
            self._require_matching_items_per_thread(
                op_name,
                "input",
                input_spec,
                "tail flags",
                tail_spec,
            )
            extent = input_spec.items_per_thread if input_spec is not None else None
            if extent is None and head_spec is not None:
                extent = head_spec.items_per_thread
            if extent is None and tail_spec is not None:
                extent = tail_spec.items_per_thread
            infer_kwarg("items_per_thread", extent)

            inferred_dtype = input_spec.dtype if input_spec is not None else None
            if inferred_dtype is None and input_var is not None:
                inferred_dtype = self._resolve_var_dtype(input_var)
            if inferred_dtype is None:
                inferred_dtype = factory_value("dtype")
            infer_kwarg("dtype", inferred_dtype)

            from numba_cuda_mlir import types as numba_mlir_types

            flag_dtype = numba_mlir_types.int32
            infer_kwarg("flag_dtype", flag_dtype)
            for flag_name, flag_var, flag_spec in (
                ("head", head_var, head_spec),
                ("tail", tail_var, tail_spec),
            ):
                if flag_var is None:
                    continue
                actual_flag_dtype = flag_spec.dtype if flag_spec is not None else None
                if actual_flag_dtype is None:
                    actual_flag_dtype = self._resolve_var_dtype(flag_var)
                if actual_flag_dtype is not None and not _dtype_values_match(
                    actual_flag_dtype,
                    flag_dtype,
                ):
                    raise CoopSinglePhaseRewriteError(
                        f"coop discontinuity {flag_name} flags must use int32 dtype."
                    )
                self._record_inferred_thread_data_dtype(flag_var, flag_dtype)
            if inferred_dtype is not None and input_var is not None:
                self._record_inferred_thread_data_dtype(input_var, inferred_dtype)

            boundary_index = 3 if mode is BlockDiscontinuityType.HEADS_AND_TAILS else 2
            for boundary_name in (
                "tile_predecessor_item",
                "tile_successor_item",
            ):
                if not factory_kwargs.get(boundary_name):
                    continue
                if boundary_index >= len(runtime_args):
                    break
                boundary_dtype = self._resolve_var_dtype(runtime_args[boundary_index])
                if (
                    inferred_dtype is not None
                    and boundary_dtype is not None
                    and not _dtype_values_match(inferred_dtype, boundary_dtype)
                ):
                    raise CoopSinglePhaseRewriteError(
                        "coop discontinuity boundary dtype must match the input dtype."
                    )
                boundary_index += 1
            return

        if op_name in {
            "warp_exclusive_sum",
            "warp_inclusive_sum",
            "warp_exclusive_scan",
            "warp_inclusive_scan",
        }:
            value_var, value_spec = candidate(0)
            inferred_dtype = value_spec.dtype if value_spec is not None else None
            if inferred_dtype is None and value_var is not None:
                inferred_dtype = self._resolve_var_dtype(value_var)
            if inferred_dtype is None:
                inferred_dtype = factory_value("dtype")
            infer_kwarg("dtype", inferred_dtype)
            aggregate_index = None
            if factory_kwargs.get("warp_aggregate"):
                aggregate_index = (
                    2
                    if factory_kwargs.get("valid_items")
                    and op_name in {"warp_exclusive_scan", "warp_inclusive_scan"}
                    else 1
                )
            if aggregate_index is not None:
                aggregate_var, aggregate_spec = candidate(aggregate_index)
                if aggregate_spec is None or aggregate_spec.items_per_thread != 1:
                    raise CoopSinglePhaseRewriteError(
                        "coop scan warp_aggregate must be a one-item "
                        "ThreadData or local array."
                    )
                aggregate_dtype = aggregate_spec.dtype
                if aggregate_dtype is None and aggregate_var is not None:
                    aggregate_dtype = self._resolve_var_dtype(aggregate_var)
                if (
                    inferred_dtype is not None
                    and aggregate_dtype is not None
                    and not _dtype_values_match(inferred_dtype, aggregate_dtype)
                ):
                    raise CoopSinglePhaseRewriteError(
                        "coop scan warp_aggregate dtype must match the input dtype."
                    )
                if aggregate_var is not None and inferred_dtype is not None:
                    self._record_inferred_thread_data_dtype(
                        aggregate_var, inferred_dtype
                    )
            return

        if op_name in {"load", "store", "warp_load", "warp_store"}:
            payload_var, payload_spec = candidate(1)
            if payload_spec is None:
                if op_name in {"store", "warp_store"}:
                    inferred_dtype = None
                    for arg in runtime_args[:2]:
                        if isinstance(arg, ir.Var):
                            inferred_dtype = self._resolve_var_dtype(arg)
                        if inferred_dtype is not None:
                            break
                    infer_kwarg("items_per_thread", 1)
                    infer_kwarg("dtype", inferred_dtype)
                return
            infer_kwarg("items_per_thread", payload_spec.items_per_thread)
            inferred_dtype = payload_spec.dtype
            if (
                inferred_dtype is None
                and runtime_args
                and isinstance(runtime_args[0], ir.Var)
            ):
                inferred_dtype = self._resolve_var_dtype(runtime_args[0])
            if inferred_dtype is None:
                inferred_dtype = factory_value("dtype")
            infer_kwarg("dtype", inferred_dtype)
            if inferred_dtype is not None and payload_var is not None:
                self._record_inferred_thread_data_dtype(payload_var, inferred_dtype)
            return
        if op_name in {"radix_sort_keys", "radix_sort_keys_descending"}:
            keys_var, keys_spec = candidate(0)
            if keys_spec is None:
                return
            infer_kwarg("items_per_thread", keys_spec.items_per_thread)
            key_dtype = keys_spec.dtype
            if key_dtype is None and keys_var is not None:
                key_dtype = self._resolve_var_dtype(keys_var)
            if key_dtype is None:
                key_dtype = factory_value("dtype")
            key_dtype = validate_integer_key_dtype(key_dtype)
            infer_kwarg("dtype", key_dtype)
            if key_dtype is not None and keys_var is not None:
                self._record_inferred_thread_data_dtype(keys_var, key_dtype)
            return
        if op_name in {"radix_sort_pairs", "radix_sort_pairs_descending"}:
            keys_var, keys_spec = candidate(0)
            values_var, values_spec = candidate(1)
            self._require_matching_items_per_thread(
                op_name, "keys", keys_spec, "values", values_spec
            )
            extent = keys_spec.items_per_thread if keys_spec is not None else None
            if extent is None and values_spec is not None:
                extent = values_spec.items_per_thread
            infer_kwarg("items_per_thread", extent)
            key_dtype = keys_spec.dtype if keys_spec is not None else None
            value_dtype = values_spec.dtype if values_spec is not None else None
            if key_dtype is None and keys_var is not None:
                key_dtype = self._resolve_var_dtype(keys_var)
            if value_dtype is None and values_var is not None:
                value_dtype = self._resolve_var_dtype(values_var)
            if key_dtype is None:
                key_dtype = factory_value("key_dtype")
            if value_dtype is None:
                value_dtype = factory_value("value_dtype")
            key_dtype = validate_integer_key_dtype(key_dtype)
            value_dtype = validate_numeric_value_dtype(value_dtype)
            infer_kwarg("key_dtype", key_dtype)
            infer_kwarg("value_dtype", value_dtype)
            if key_dtype is not None and keys_var is not None:
                self._record_inferred_thread_data_dtype(keys_var, key_dtype)
            if value_dtype is not None and values_var is not None:
                self._record_inferred_thread_data_dtype(values_var, value_dtype)
            return
        if op_name == "radix_rank":
            from numba_cuda_mlir import types as numba_mlir_types

            def is_int32_dtype(dtype) -> bool:
                if dtype == numba_mlir_types.int32:
                    return True
                try:
                    return np.dtype(dtype) == np.dtype(np.int32)
                except (TypeError, ValueError):
                    return False

            keys_var, keys_spec = candidate(0)
            ranks_var, ranks_spec = candidate(1)
            self._require_matching_items_per_thread(
                op_name, "keys", keys_spec, "ranks", ranks_spec
            )
            extent = keys_spec.items_per_thread if keys_spec is not None else None
            if extent is None and ranks_spec is not None:
                extent = ranks_spec.items_per_thread
            infer_kwarg("items_per_thread", extent)
            key_dtype = keys_spec.dtype if keys_spec is not None else None
            if key_dtype is None and keys_var is not None:
                key_dtype = self._resolve_var_dtype(keys_var)
            if key_dtype is None:
                key_dtype = factory_value("dtype")
            key_dtype = validate_integer_key_dtype(key_dtype)
            infer_kwarg("dtype", key_dtype)
            if key_dtype is not None and keys_var is not None:
                self._record_inferred_thread_data_dtype(keys_var, key_dtype)
            if ranks_spec is not None and ranks_var is not None:
                if ranks_spec.dtype is not None and not is_int32_dtype(
                    ranks_spec.dtype
                ):
                    raise CoopSinglePhaseRewriteError(
                        "coop single-phase 'radix_rank' requires ranks dtype int32."
                    )
                self._record_inferred_thread_data_dtype(
                    ranks_var, numba_mlir_types.int32
                )
            if factory_kwargs.get("exclusive_digit_prefix"):
                prefix_var, prefix_spec = candidate(2)
                if prefix_spec is None or prefix_var is None:
                    raise CoopSinglePhaseRewriteError(
                        "radix_rank exclusive_digit_prefix must be a local array"
                    )
                if prefix_spec.dtype is not None and not is_int32_dtype(
                    prefix_spec.dtype
                ):
                    raise CoopSinglePhaseRewriteError(
                        "radix_rank exclusive_digit_prefix dtype must be int32"
                    )
                self._record_inferred_thread_data_dtype(
                    prefix_var, numba_mlir_types.int32
                )
                threads = factory_kwargs.get("threads_per_block")
                begin = factory_kwargs.get("begin_bit")
                end = factory_kwargs.get("end_bit")
                if threads is not None and begin is not None and end is not None:
                    block_dim = normalize_dim_param(threads)
                    block_threads = block_dim.x * block_dim.y * block_dim.z
                    expected = max(
                        1,
                        ((1 << (int(end) - int(begin))) + block_threads - 1)
                        // block_threads,
                    )
                    if (
                        prefix_spec.items_per_thread is not None
                        and prefix_spec.items_per_thread != expected
                    ):
                        raise CoopSinglePhaseRewriteError(
                            "radix_rank exclusive_digit_prefix must contain "
                            f"{expected} items per thread"
                        )
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
                from ._block._block_exchange import (
                    BlockExchangeType,
                    _normalize_exchange_type,
                )

                exchange_type = _normalize_exchange_type(
                    factory_kwargs.get(
                        "block_exchange_type", BlockExchangeType.StripedToBlocked
                    )
                )
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
                    not uses_ranks
                    and len(runtime_args) == 2
                    or (
                        uses_ranks
                        and (not uses_valid_flags)
                        and (len(runtime_args) == 3)
                    )
                    or (uses_valid_flags and len(runtime_args) == 4)
                )
                if uses_ranks:
                    _, rank_spec = candidate(2 if out_of_place else 1)
                if uses_valid_flags:
                    _, valid_flag_spec = candidate(3 if out_of_place else 2)
            else:
                from ._warp._warp_exchange import (
                    WarpExchangeType,
                    _normalize_exchange_type,
                )

                exchange_type = _normalize_exchange_type(
                    factory_kwargs.get(
                        "warp_exchange_type", WarpExchangeType.StripedToBlocked
                    )
                )
                uses_ranks = exchange_type == WarpExchangeType.ScatterToStriped
                out_of_place = not uses_ranks or len(runtime_args) == 3
                if uses_ranks:
                    _, rank_spec = candidate(2 if out_of_place else 1)
            if not out_of_place:
                output_var = None
                output_spec = None
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
                and (input_spec.dtype is not None)
                and (output_spec.dtype is not None)
                and (input_spec.dtype != output_spec.dtype)
            ):
                raise CoopSinglePhaseRewriteError(
                    "coop exchange requires input/output arrays to have matching dtype."
                )
            inferred_dtype = input_spec.dtype if input_spec is not None else None
            if inferred_dtype is None and output_spec is not None:
                inferred_dtype = output_spec.dtype
            if inferred_dtype is None:
                inferred_dtype = factory_value("dtype")
            extent = input_spec.items_per_thread if input_spec is not None else None
            if extent is None and output_spec is not None:
                extent = output_spec.items_per_thread
            infer_kwarg("items_per_thread", extent)
            infer_kwarg("dtype", inferred_dtype)
            if inferred_dtype is not None:
                if input_var is not None:
                    self._record_inferred_thread_data_dtype(input_var, inferred_dtype)
                if output_var is not None:
                    self._record_inferred_thread_data_dtype(output_var, inferred_dtype)
            return
        if op_name in {"merge_sort_keys", "warp_merge_sort_keys"}:
            keys_var, keys_spec = candidate(0)
            if keys_spec is None:
                return
            infer_kwarg("items_per_thread", keys_spec.items_per_thread)
            key_dtype = keys_spec.dtype
            if key_dtype is None and keys_var is not None:
                key_dtype = self._resolve_var_dtype(keys_var)
            if key_dtype is None and keys_var is not None:
                key_dtype = self._infer_thread_data_dtype_from_writes(keys_var)
            if key_dtype is None:
                key_dtype = factory_value("dtype")
            infer_kwarg("dtype", key_dtype)
            if key_dtype is not None and keys_var is not None:
                self._record_inferred_thread_data_dtype(keys_var, key_dtype)
            return
        if op_name in {"merge_sort_pairs", "warp_merge_sort_pairs"}:
            keys_var, keys_spec = candidate(0)
            values_var, values_spec = candidate(1)
            self._require_matching_items_per_thread(
                op_name,
                "keys",
                keys_spec,
                "values",
                values_spec,
            )
            extent = keys_spec.items_per_thread if keys_spec is not None else None
            if extent is None and values_spec is not None:
                extent = values_spec.items_per_thread
            infer_kwarg("items_per_thread", extent)
            key_dtype = keys_spec.dtype if keys_spec is not None else None
            value_dtype = values_spec.dtype if values_spec is not None else None
            if key_dtype is None and keys_var is not None:
                key_dtype = self._resolve_var_dtype(keys_var)
            if value_dtype is None and values_var is not None:
                value_dtype = self._resolve_var_dtype(values_var)
            if key_dtype is None and keys_var is not None:
                key_dtype = self._infer_thread_data_dtype_from_writes(keys_var)
            if value_dtype is None and values_var is not None:
                value_dtype = self._infer_thread_data_dtype_from_writes(values_var)
            if key_dtype is None:
                key_dtype = factory_value("keys")
            if value_dtype is None:
                value_dtype = factory_value("values")
            infer_kwarg("keys", key_dtype)
            infer_kwarg("values", value_dtype)
            if key_dtype is not None and keys_var is not None:
                self._record_inferred_thread_data_dtype(keys_var, key_dtype)
            if value_dtype is not None and values_var is not None:
                self._record_inferred_thread_data_dtype(values_var, value_dtype)
            return
        if op_name == "shuffle":
            if len(runtime_args) == 1:
                value = runtime_args[0]
                inferred_dtype = (
                    self._resolve_var_dtype(value)
                    if isinstance(value, ir.Var)
                    else None
                )
                infer_kwarg("dtype", inferred_dtype or factory_value("dtype"))
                return
            input_var, input_spec = candidate(0)
            output_var, output_spec = candidate(1)
            self._require_matching_items_per_thread(
                op_name, "input", input_spec, "output", output_spec
            )
            extent = input_spec.items_per_thread if input_spec is not None else None
            if extent is None and output_spec is not None:
                extent = output_spec.items_per_thread
            infer_kwarg("items_per_thread", extent)
            inferred_dtype = input_spec.dtype if input_spec is not None else None
            if inferred_dtype is None and output_spec is not None:
                inferred_dtype = output_spec.dtype
            if inferred_dtype is None and input_var is not None:
                inferred_dtype = self._resolve_var_dtype(input_var)
            infer_kwarg("dtype", inferred_dtype or factory_value("dtype"))
            if inferred_dtype is not None:
                if input_var is not None:
                    self._record_inferred_thread_data_dtype(input_var, inferred_dtype)
                if output_var is not None:
                    self._record_inferred_thread_data_dtype(output_var, inferred_dtype)

    @staticmethod
    def _invocable_cache_key(
        op_name: str, factory_kwargs: dict[str, object]
    ) -> tuple[str, tuple[tuple[str, str, str], ...]]:

        def cache_component(name, value):
            hasher = hashlib.sha1()
            _hash_symbol_value(hasher, value)
            value_type = f"{type(value).__module__}.{type(value).__qualname__}"
            return (name, value_type, hasher.hexdigest())

        return (
            op_name,
            tuple(
                sorted(
                    (
                        cache_component(name, value)
                        for name, value in factory_kwargs.items()
                    )
                )
            ),
        )

    @staticmethod
    def _validate_invocable(invocable, op_name: str):
        if not callable(invocable) or not hasattr(invocable, "files"):
            raise CoopSinglePhaseRewriteError(
                f"coop single-phase factory for '{op_name}' did not produce a coop invocable; got {type(invocable)!r}."
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
            return (self._invocable_cache[key], False)
        compile_cache = self._state.metadata.setdefault(
            "__cuda_coop_numba_mlir_invocable_cache__", {}
        )
        if key in compile_cache:
            invocable = compile_cache[key]
            self._validate_invocable(invocable, match.op_name)
            self._invocable_cache[key] = invocable
            return (invocable, False)
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
                f"Failed to evaluate coop single-phase factory at compile time for '{match.op_name}'."
            ) from e
        self._validate_invocable(invocable, match.op_name)
        self._invocable_cache[key] = invocable
        compile_cache[key] = invocable
        return (invocable, True)

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
        return (max_default, max_optin)

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
                f"TempStorage requires {dynamic_shared_bytes} bytes dynamic shared memory, but device max opt-in is {max_optin} bytes."
            )
        if dynamic_shared_bytes > 0:
            set_required_dynamic_shared_memory(self._state, dynamic_shared_bytes)
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
                ir.Global(slice_ctor_global_name, slice, loc), slice_ctor_var, loc
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
            ir.Assign(ir.Expr.getitem(source_var, slice_obj_var, loc), target_var, loc)
        )

    def _runtime_temp_storage_arg_for_call(
        self, block: ir.Block, *, source_var: ir.Var, call_assign: ir.Assign
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
        return (temp_storage_arg, temp_storage_plan)

    def _emit_temp_storage_auto_sync(
        self, block: ir.Block, *, scope: ir.Scope, loc: ir.Loc, sync_attr: str
    ) -> None:
        sync_module_global_name = _next_global_name("temp_storage_sync_mod")
        sync_module_var = ir.Var(
            scope, f"__coop_sync_mod_var_{next(_GLOBAL_NAME_COUNTER)}__", loc
        )
        sync_fn_var = ir.Var(
            scope, f"__coop_sync_fn_{next(_GLOBAL_NAME_COUNTER)}__", loc
        )
        sync_result_var = ir.Var(
            scope, f"__coop_sync_result_{next(_GLOBAL_NAME_COUNTER)}__", loc
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
                ir.Expr.getattr(sync_module_var, sync_attr, loc), sync_fn_var, loc
            )
        )
        block.append(
            ir.Assign(ir.Expr.call(sync_fn_var, (), (), loc), sync_result_var, loc)
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
                "root physical-warp load/store requires an inferred positive items_per_thread"
            )
        logical_warp_threads = match.factory_kwargs.get("threads_in_warp", 32)
        if (
            not isinstance(logical_warp_threads, int)
            or isinstance(logical_warp_threads, bool)
            or logical_warp_threads < 1
            or (32 % logical_warp_threads != 0)
        ):
            raise CoopSinglePhaseRewriteError(
                "root warp load/store requires threads_in_warp to be a positive divisor of 32"
            )
        block_dim = normalize_dim_param(match.factory_kwargs.get("threads_per_block"))

        def new_var(stem: str) -> ir.Var:
            return ir.Var(
                scope, f"__coop_warp_tile_{stem}_{next(_GLOBAL_NAME_COUNTER)}__", loc
            )

        def constant(value: int, stem: str) -> ir.Var:
            result = new_var(stem)
            block.append(ir.Assign(ir.Const(value, loc), result, loc))
            return result

        def binary(function, lhs: ir.Var, rhs: ir.Var, stem: str) -> ir.Var:
            result = new_var(stem)
            block.append(ir.Assign(ir.Expr.binop(function, lhs, rhs, loc), result, loc))
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
            ir.Assign(ir.Expr.getattr(module_var, "threadIdx", loc), thread_idx, loc)
        )

        def component(axis: str) -> ir.Var:
            result = new_var(f"thread_idx_{axis}")
            block.append(ir.Assign(ir.Expr.getattr(thread_idx, axis, loc), result, loc))
            return result

        linear_tid = component("x")
        if block_dim[1] > 1 or block_dim[2] > 1:
            y = component("y")
            z = component("z")
            yz = binary(operator.mul, constant(block_dim[1], "block_y"), z, "linear_yz")
            yz = binary(operator.add, y, yz, "linear_y")
            yz = binary(
                operator.mul, constant(block_dim[0], "block_x"), yz, "linear_x_stride"
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
                scope, f"__coop_root_store_{stem}_{next(_GLOBAL_NAME_COUNTER)}__", loc
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
                ir.Expr.call(array_fn, [shape_var, dtype_var], (), loc), payload, loc
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
                    elif self._is_typed_group_payload_ctor_call(call):
                        self._thread_data_specs[inst.target.name] = (
                            self._merge_thread_data_specs(
                                self._thread_data_specs.get(inst.target.name),
                                self._extract_typed_group_payload_spec(call),
                            )
                        )
                    elif self._is_temp_storage_ctor_call(call):
                        self._temp_storage_ctor_specs[inst.target.name] = (
                            self._extract_temp_storage_ctor_spec(call)
                        )
                        self._temp_storage_ctor_order[inst.target.name] = ctor_order
                        ctor_order += 1
            all_scan_matches: list[_RewriteMatch] = []
            storage_uses: list[tuple[int, ir.Assign, _RewriteMatch, str]] = []
            source_order = 0
            for label in sorted(func_ir.blocks):
                scan_block = func_ir.blocks[label]
                self._block = scan_block
                self._block_defs = {
                    inst.target.name: inst.value
                    for inst in scan_block.body
                    if isinstance(inst, ir.Assign)
                }
                for inst in scan_block.body:
                    current_order = source_order
                    source_order += 1
                    if not isinstance(inst, ir.Assign):
                        continue
                    call = inst.value
                    if not isinstance(call, ir.Expr) or call.op != "call":
                        continue
                    target = self._resolve_call_target(call)
                    if target is None:
                        continue
                    op_name = target.factory.__name__
                    (
                        runtime_args,
                        runtime_temp_storage_var,
                        factory_kwargs,
                        factory_kw_value_vars,
                        runtime_arg_constant_replacements,
                    ) = self._validate_and_split_args(
                        op_name, call, target.getitem_temp_storage
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
                        runtime_arg_constant_replacements=runtime_arg_constant_replacements,
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
                    if ctor_key is not None:
                        storage_uses.append((current_order, inst, scan_match, ctor_key))
            self._prepare_ltoir_bundle_for_matches(all_scan_matches)
            for source_order, inst, scan_match, ctor_key in storage_uses:
                invocable, _ = self._materialize_invocable(scan_match)
                size_in_bytes = max(
                    1, int(getattr(invocable, "temp_storage_bytes", 0) or 0)
                )
                alignment = max(
                    1, int(getattr(invocable, "temp_storage_alignment", 0) or 0)
                )
                summary = requirements.setdefault(
                    ctor_key, _TempStorageRequirementSummary()
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

    def match(self, func_ir, block, typemap, calltypes):
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
            self._func_temp_storage_requirements = (
                self._compute_func_temp_storage_requirements(func_ir)
            )
        self._block = block
        self._block_defs = {
            inst.target.name: inst.value
            for inst in block.body
            if isinstance(inst, ir.Assign)
        }
        self._matches = {}
        self._temp_storage_assigns = set()
        self._temp_storage_func_vars = set()
        self._thread_data_func_vars = set()
        self._typed_group_payload_func_vars = set()
        for inst in block.body:
            if not isinstance(inst, ir.Assign):
                continue
            call = inst.value
            if not isinstance(call, ir.Expr) or call.op != "call":
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
            target = self._resolve_call_target(call)
            if target is None:
                continue
            op_name = target.factory.__name__
            (
                runtime_args,
                runtime_temp_storage_var,
                factory_kwargs,
                factory_kw_value_vars,
                runtime_arg_constant_replacements,
            ) = self._validate_and_split_args(
                op_name, call, target.getitem_temp_storage
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
                runtime_arg_constant_replacements=runtime_arg_constant_replacements,
                physical_warp_tile_origin=physical_warp_tile_origin,
                preserve_root_store_payload=preserve_root_store_payload,
                root_store_scalar=root_store_scalar,
            )
        return (
            bool(self._matches)
            or bool(self._temp_storage_assigns)
            or bool(self._thread_data_func_vars)
            or bool(self._typed_group_payload_func_vars)
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
                (value_var.name for value_var in match.factory_kw_value_vars)
            )
            global_name = _next_global_name("single_phase")
            call_invocable_globals[match_inst] = (global_name, invocable)
            func_var_names_to_clear.add(match.func_var_name)
            if match.func_var_name_extra is not None:
                func_var_names_to_clear.add(match.func_var_name_extra)
        new_block = ir.Block(self._block.scope, self._block.loc)
        for inst in self._block.body:
            if (
                isinstance(inst, ir.Assign)
                and inst.target.name
                in self._thread_data_func_vars | self._typed_group_payload_func_vars
            ):
                module_var = ir.Var(
                    inst.target.scope,
                    f"__coop_thread_data_module_{next(_GLOBAL_NAME_COUNTER)}__",
                    inst.loc,
                )
                local_module_var = ir.Var(
                    inst.target.scope,
                    f"__coop_thread_data_local_{next(_GLOBAL_NAME_COUNTER)}__",
                    inst.loc,
                )
                new_block.append(
                    ir.Assign(
                        ir.Global(
                            _next_global_name("thread_data_module"),
                            _cuda_module,
                            inst.loc,
                        ),
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
                    ir.Assign(ir.Const(None, inst.loc), inst.target, inst.loc)
                )
                continue
            if (
                isinstance(inst, ir.Assign)
                and inst.target.name in self._temp_storage_func_vars
            ):
                new_block.append(
                    ir.Assign(ir.Const(None, inst.loc), inst.target, inst.loc)
                )
                continue
            if (
                isinstance(inst, ir.Assign)
                and isinstance(inst.value, ir.Expr)
                and (inst.value.op == "call")
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
                        f"Failed to infer dtype for {subject}. Use it with a movement operation that provides dtype context."
                    )
                if thread_data_spec.common_v1:
                    from ._common import _validate_common_numeric_dtype

                    try:
                        _validate_common_numeric_dtype(
                            thread_data_spec.dtype, operation="ThreadData"
                        )
                    except (TypeError, ValueError) as exc:
                        raise CoopSinglePhaseRewriteError(str(exc)) from exc
                dtype_var = ir.Var(
                    inst.target.scope,
                    f"__coop_thread_data_dtype_{next(_GLOBAL_NAME_COUNTER)}__",
                    inst.loc,
                )
                new_block.append(
                    ir.Assign(
                        ir.Global(
                            _next_global_name("thread_data_dtype"),
                            thread_data_spec.dtype,
                            inst.loc,
                        ),
                        dtype_var,
                        inst.loc,
                    )
                )
                rewritten_args = [] if is_typed_group_payload else list(inst.value.args)
                rewritten_kws = [] if is_typed_group_payload else list(inst.value.kws)
                rewritten_kws = [
                    (
                        "shape"
                        if name == "items_per_thread"
                        else "alignment"
                        if name == "alignas"
                        else name,
                        value,
                    )
                    for name, value in rewritten_kws
                ]
                if thread_data_spec.items_per_thread is None:
                    raise CoopSinglePhaseRewriteError(
                        "Failed to infer static extent for typed group payload."
                    )
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
                elif any((name == "shape" for name, _ in rewritten_kws)):
                    rewritten_kws = [
                        (name, items_var if name == "shape" else value)
                        for name, value in rewritten_kws
                    ]
                else:
                    rewritten_args.append(items_var)
                if len(rewritten_args) >= 2:
                    rewritten_args[1] = dtype_var
                elif any((name == "dtype" for name, _ in rewritten_kws)):
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
            match = self._matches.get(inst)
            if match is None and inst not in self._temp_storage_assigns:
                new_block.append(inst)
                continue
            if inst in self._temp_storage_assigns:
                ctor_key = self._resolve_temp_storage_ctor_key(inst.target)
                if ctor_key is None:
                    raise CoopSinglePhaseRewriteError(
                        f"Missing TempStorage metadata for '{inst.target.name}'."
                    )
                plan = self._finalize_temp_storage_plan_for_var(ctor_key)
                uses_dynamic_smem = bool(
                    temp_storage_global_plan is not None
                    and temp_storage_global_plan.uses_dynamic_smem
                )
                alloc_size = 0 if uses_dynamic_smem else int(plan.size_in_bytes)
                backing_var = inst.target
                if uses_dynamic_smem:
                    backing_var = ir.Var(
                        inst.target.scope,
                        f"__coop_temp_storage_backing_{next(_GLOBAL_NAME_COUNTER)}__",
                        inst.loc,
                    )
                module_var = ir.Var(
                    inst.target.scope,
                    f"__coop_temp_storage_module_{next(_GLOBAL_NAME_COUNTER)}__",
                    inst.loc,
                )
                shared_var = ir.Var(
                    inst.target.scope,
                    f"__coop_temp_storage_shared_{next(_GLOBAL_NAME_COUNTER)}__",
                    inst.loc,
                )
                array_fn_var = ir.Var(
                    inst.target.scope,
                    f"__coop_temp_storage_array_{next(_GLOBAL_NAME_COUNTER)}__",
                    inst.loc,
                )
                bytes_var = ir.Var(
                    inst.target.scope,
                    f"__coop_temp_storage_bytes_{next(_GLOBAL_NAME_COUNTER)}__",
                    inst.loc,
                )
                align_var = ir.Var(
                    inst.target.scope,
                    f"__coop_temp_storage_alignment_{next(_GLOBAL_NAME_COUNTER)}__",
                    inst.loc,
                )
                dtype_var = ir.Var(
                    inst.target.scope,
                    f"__coop_temp_storage_dtype_{next(_GLOBAL_NAME_COUNTER)}__",
                    inst.loc,
                )
                new_block.append(
                    ir.Assign(
                        ir.Global(
                            _next_global_name("temp_storage_module"),
                            _cuda_module,
                            inst.loc,
                        ),
                        module_var,
                        inst.loc,
                    )
                )
                new_block.append(
                    ir.Assign(
                        ir.Expr.getattr(module_var, "shared", inst.loc),
                        shared_var,
                        inst.loc,
                    )
                )
                new_block.append(
                    ir.Assign(
                        ir.Expr.getattr(shared_var, "array", inst.loc),
                        array_fn_var,
                        inst.loc,
                    )
                )
                new_block.append(
                    ir.Assign(ir.Const(alloc_size, inst.loc), bytes_var, inst.loc)
                )
                new_block.append(
                    ir.Assign(ir.Const(plan.alignment, inst.loc), align_var, inst.loc)
                )
                new_block.append(
                    ir.Assign(
                        ir.Global(
                            _next_global_name("temp_storage_dtype"),
                            _cuda_module.uint8,
                            inst.loc,
                        ),
                        dtype_var,
                        inst.loc,
                    )
                )
                temp_storage_call = ir.Expr.call(
                    array_fn_var,
                    [bytes_var, dtype_var],
                    (("alignment", align_var),),
                    inst.loc,
                )
                new_block.append(ir.Assign(temp_storage_call, backing_var, inst.loc))
                if uses_dynamic_smem:
                    self._emit_array_slice(
                        new_block,
                        source_var=backing_var,
                        target_var=inst.target,
                        start=plan.base_offset,
                        stop=plan.base_offset + plan.size_in_bytes,
                        loc=inst.loc,
                    )
                continue
            assert match is not None
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
            new_block.append(
                ir.Assign(
                    ir.Expr.call(call_func, rewritten_runtime_args, (), match.loc),
                    inst.target,
                    match.loc,
                )
            )
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
        used_var_names: set[str] = set()
        for stmt in new_block.body:
            stmt_vars = list(stmt.list_vars())
            if isinstance(stmt, ir.Assign):
                stmt_vars = [var for var in stmt_vars if var.name != stmt.target.name]
            used_var_names.update((var.name for var in stmt_vars))
        if candidate_dead_factory_kw_vars:
            filtered_block = ir.Block(new_block.scope, new_block.loc)
            for stmt in new_block.body:
                if (
                    isinstance(stmt, ir.Assign)
                    and stmt.target.name in candidate_dead_factory_kw_vars
                    and (stmt.target.name not in used_var_names)
                ):
                    continue
                filtered_block.append(stmt)
            new_block = filtered_block
        if refresh_typing_context:
            self._state.typingctx.refresh()
        return new_block


from . import _group_rewrites as _group_rewrites  # noqa: E402


@register_planner
class CoopWholeFunctionPlanner(WholeFunctionPlanner):
    """Apply cooperative-provider rewrites after device-function inlining."""

    def run(self) -> bool:
        rewrite = CoopSinglePhaseRewrite(self.state)
        modified = False
        for label in sorted(self.state.func_ir.blocks):
            block = self.state.func_ir.blocks[label]
            while rewrite.match(
                self.state.func_ir, block, self.state.typemap, self.state.calltypes
            ):
                block = rewrite.apply()
                self.state.func_ir.blocks[label] = block
                modified = True
        return modified


__all__ = [
    "CoopSinglePhaseRewrite",
    "CoopSinglePhaseRewriteError",
    "CoopWholeFunctionPlanner",
]
