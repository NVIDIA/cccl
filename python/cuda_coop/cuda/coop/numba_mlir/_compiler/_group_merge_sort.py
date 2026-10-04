# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Plan MergeSort calls while preserving the caller's input payloads.

CUB sorts keys and optional values in place. These rewrites allocate and
copy result payloads first, then sort those copies and return them through
IR aliases. Shared-core planning selects the block or warp implementation
and its storage contract. The rewrite keeps comparison and shape inputs
separate from runtime partial-tile counts and sentinels.
"""

import math
from dataclasses import replace

import numba_cuda_mlir.numba_cuda.types as numba_types
import numpy as np

from cuda.coop._core import (
    BindingKind,
    GroupLoweringTarget,
    GroupMergeSortSemantics,
    StorageOwnership,
    SynchronizationScope,
    make_block_merge_sort_semantics,
    make_group_primitive_call,
    plan_group_primitive,
)

from ._group_planner_support import _PAYLOAD_DTYPE_LIKE, GroupRewriteError, ir
from ._operations import (
    GroupResultSource,
    RewriteOperationSpecification,
    register_group_primitive,
    register_rewrite_operation,
)
from ._parameters import (
    _validate_common_numeric_dtype,
    _validate_runtime_integer_dtype,
    coerce_static_scalar,
)
from ._rewrite_merge_sort import infer_merge_sort_payload


def _payload(context, operation, name, value, is_common_root):
    """Recover a fixed array extent and numeric dtype for keys or values.

    Merge-sort planning calls this separately for keys and associated
    values before selecting an overload. Extent here is the number of
    elements held by one thread, not the total tile size.

    Common calls require ThreadData; qualified calls also accept local
    arrays. Use an existing dtype or infer it from writes, then record it on
    the payload so the result allocation inherits the same type. Return the
    extent and dtype; pairwise extent matching is checked by the caller.

    Parameters
    ----------
    context : GroupPlanningContext
        Access to launch dimensions, constant controls, payload
        facts, and IR builders for this group-planning attempt.
    operation : str
        Canonical public operation name, used in diagnostics and
        generated temporary names.
    name : str
        Public payload argument name, such as keys or values, for
        diagnostics.
    value : ir.Var
        Per-thread array whose item count and element dtype are
        required.
    is_common_root : bool
        Whether the call came through the common ``cuda.coop`` API
        and must satisfy its narrower operand and selector rules.
    """

    if not context.is_array(operation, value):
        raise TypeError(
            f"{operation} {name} must be a fixed-size ThreadData or local array"
        )
    if is_common_root and not context.is_thread_data(operation, name, value):
        raise TypeError(
            f"cuda.coop.{operation} {name} requires ThreadData; use "
            f"cuda.coop.numba_mlir for local arrays"
        )
    extent = context.array_extent(value)
    if extent is None:
        raise GroupRewriteError(f"{operation} could not infer {name} extent")
    dtype = context.dtype(value)
    if dtype is None:
        dtype = context.payload_write_dtype(value)
    if dtype is None:
        raise GroupRewriteError(f"{operation} could not infer {name} dtype")
    dtype = _validate_common_numeric_dtype(
        dtype, operation=operation, parameter=name
    )
    context.record_thread_data_dtype(value, dtype)
    return extent, dtype


def _cast(context, statements, inst, value, dtype, name):
    """Append a typed scalar conversion for a partial-tile operand.

    Counts must reach the checked C++ wrapper as int64; sentinels must have
    the key dtype. Materialize a constant or existing IR value, append its
    conversion call, and return the new variable for the provider arguments.

    Parameters
    ----------
    context : GroupPlanningContext
        Access to launch dimensions, constant controls, payload
        facts, and IR builders for this group-planning attempt.
    statements : list of IR statements
        Pending replacement statements, appended to in execution
        order. The function's blocks are unchanged until the owning
        planner installs this list.
    inst : ir.Assign
        Original public call assignment. Its target, scope, and
        source location identify the replacement result and
        generated temporaries.
    value : object
        Constant scalar or existing IR variable to convert in the
        generated kernel.
    dtype : numba type
        Target scalar type required by the provider ABI.
    name : str
        Operand label used to distinguish generated temporary names.
    """

    kwargs = {"scope": inst.target.scope, "loc": inst.loc}
    cast = context.value_var(
        statements, stem=f"merge_sort_{name}_type", value=dtype, **kwargs
    )
    value = context.value_var(
        statements, stem=f"merge_sort_{name}", value=value, **kwargs
    )
    result = context.new_var(
        inst.target.scope, inst.loc, f"merge_sort_{name}_cast"
    )
    statements.append(
        ir.Assign(ir.Expr.call(cast, [value], (), inst.loc), result, inst.loc)
    )
    return result


def _coerce_static_sentinel(value, dtype, *, operation):
    """Convert a static key sentinel while allowing infinite float bounds.

    The shared scalar coercion rejects nonfinite values, but infinity can be
    a useful sorting bound. Validate a matching zero first so this special
    case keeps the usual dtype checks, including exact NumPy scalar dtypes.
    Then restore the infinity in the validated type. Other values use the
    ordinary representability checks.
    """

    if (type(value) is float or isinstance(value, np.floating)) and math.isinf(
        value
    ):
        # Validate the scalar dtype before preserving an infinite sorting bound.
        zero = coerce_static_scalar(
            type(value)(0), dtype, operation=operation, parameter="oob_default"
        )
        return type(zero)(value)
    return coerce_static_scalar(
        value, dtype, operation=operation, parameter="oob_default"
    )


def _lower_merge_sort(
    context, inst, *, operation, group, bound, is_common_root
):
    """Build a supported Merge Sort call over fresh copies of the inputs.

    Infer payload dtypes and matching extents. Require constant comparison
    choices and validate the paired valid-count and sentinel controls. The
    shared semantics record whether the tile is partial; the actual count
    binding stays in the group plan, and the sentinel remains a call operand.

    Ask core planning to select the CUB block or warp implementation. Apply
    any caller-owned block storage descriptor to that plan, then choose the
    registered factory whose namespace and partial-tile form match it.

    Allocate and copy keys and optional values before the provider call, so
    CUB's in-place sort cannot modify the public inputs. Convert a partial
    count to int64 and its sentinel to the key dtype, even when the values
    are static. Return pending IR statements with the plan attached; the
    public result aliases one copied payload or the pair of copied payloads.
    """

    from .._lowering import _merge_sort

    arguments = bound.arguments
    pairs = operation == "merge_sort_pairs"
    names = ("keys", "values") if pairs else ("keys",)
    payloads = [
        _payload(context, operation, name, arguments[name], is_common_root)
        for name in names
    ]
    extent, key_dtype = payloads[0]
    if pairs and payloads[1][0] != extent:
        raise ValueError(
            "Merge Sort keys and values must have "
            "matching items_per_thread extents"
        )
    value_dtype = payloads[1][1] if pairs else None
    descending = context.constant(arguments["descending"])
    compare_raw = arguments.get("compare_op")
    compare_op = (
        None if context.is_none(compare_raw) else context.constant(compare_raw)
    )
    compare_operator = _merge_sort.comparison_operator(descending, compare_op)
    valid_raw = arguments["valid_items"]
    sentinel_raw = arguments["oob_default"]
    valid = context.planning_binding(valid_raw)
    partial = valid.kind is not BindingKind.OMITTED
    if partial == context.is_none(sentinel_raw):
        raise ValueError(
            "Merge Sort valid_items and oob_default must be provided together"
        )
    if valid.kind is BindingKind.RUNTIME:
        _validate_runtime_integer_dtype(
            context.dtype(valid_raw),
            operation=operation,
            parameter="valid_items",
        )
    sentinel = None
    if partial:
        binding = context.planning_binding(sentinel_raw)
        if binding.kind is BindingKind.STATIC:
            sentinel = _coerce_static_sentinel(
                binding.value, key_dtype, operation=operation
            )
        else:
            sentinel_dtype = _validate_common_numeric_dtype(
                context.dtype(sentinel_raw),
                operation=operation,
                parameter="oob_default",
            )
            if sentinel_dtype != key_dtype:
                raise TypeError(
                    f"Merge Sort oob_default dtype {sentinel_dtype} does not "
                    f"match keys dtype {key_dtype}"
                )
            sentinel = sentinel_raw
    from .._lowering._core import NumbaMlirCoreAdapter

    adapter = NumbaMlirCoreAdapter()
    semantics = GroupMergeSortSemantics(
        make_block_merge_sort_semantics(
            key_dtype=adapter.core_dtype(key_dtype),
            value_dtype=adapter.core_dtype(value_dtype),
            items_per_thread=extent,
            compare_operator=compare_operator,
            valid_items=0 if partial else None,
            oob_default=0 if partial else None,
        ),
        valid_items=valid,
    )
    plan = plan_group_primitive(
        make_group_primitive_call(group, semantics), context.launch
    ).require_supported()
    participation = plan.participation
    assert participation is not None
    assert plan.temp_storage is not None
    assert plan.synchronization is not None
    topology = plan.topology
    assert topology is not None
    temp_storage = arguments["temp_storage"]
    if not context.is_none(temp_storage):
        if plan.target is not GroupLoweringTarget.CUB_BLOCK:
            raise ValueError(
                "Merge Sort temp_storage applies only to block groups"
            )
        descriptor = context.temp_storage(temp_storage)
        if descriptor is None:
            raise GroupRewriteError(
                "Merge Sort temp_storage must resolve to a compile-time "
                "TempStorage descriptor"
            )
        size, alignment, auto_sync, sharing = descriptor
        plan = replace(
            plan,
            temp_storage=replace(
                plan.temp_storage,
                ownership=StorageOwnership.CALLER,
                exact_layout_required=True,
                sharing=sharing,
                requested_size_in_bytes=size,
                requested_alignment=alignment,
                auto_sync=auto_sync,
            ),
            synchronization=replace(
                plan.synchronization,
                storage_reuse_barrier=SynchronizationScope.BLOCK
                if auto_sync
                else SynchronizationScope.NONE,
            ),
        )
    namespace = (
        "block" if plan.target is GroupLoweringTarget.CUB_BLOCK else "warp"
    )
    assert plan.provenance is not None
    expected_class = (
        "cub::BlockMergeSort" if namespace == "block" else "cub::WarpMergeSort"
    )
    if plan.provenance.cpp_class != expected_class:
        raise GroupRewriteError(
            "Merge Sort received unknown provider provenance"
        )
    factory = getattr(
        _merge_sort,
        f"{namespace}_{operation}" + ("_partial" if partial else ""),
    )
    kwargs = {
        "key_dtype": key_dtype,
        "items_per_thread": extent,
        "threads_per_block": participation.exact_block_dim,
        "descending": descending,
        "compare_op": compare_raw,
    }
    if pairs:
        kwargs["value_dtype"] = value_dtype
    if namespace == "warp":
        kwargs["threads_in_warp"] = topology.logical_width
    if not context.is_none(temp_storage):
        kwargs["temp_storage"] = temp_storage
    statements = []
    results = []
    for name in names:
        value = context.value_var(
            statements,
            scope=inst.target.scope,
            loc=inst.loc,
            stem=f"merge_sort_{name}",
            value=arguments[name],
        )
        result = context.typed_payload_like(
            statements,
            scope=inst.target.scope,
            loc=inst.loc,
            stem=f"merge_sort_{name}_result",
            prototype=value,
            is_array=True,
            dtype_policy=_PAYLOAD_DTYPE_LIKE,
            items_per_thread=extent,
        )
        context.copy_array_payload(
            statements,
            operation=operation,
            source=value,
            destination=result,
            scope=inst.target.scope,
            loc=inst.loc,
            known_items_per_thread=extent,
        )
        results.append(result)
    runtime_args = list(results)
    if partial:
        count = valid.value if valid.kind is BindingKind.STATIC else valid_raw
        runtime_args.extend(
            (
                _cast(
                    context,
                    statements,
                    inst,
                    count,
                    numba_types.int64,
                    "valid_items",
                ),
                _cast(
                    context,
                    statements,
                    inst,
                    sentinel,
                    key_dtype,
                    "oob_default",
                ),
            )
        )
    statements.extend(
        context.rewrite_call(
            inst,
            lowering_plan=plan,
            factory=factory,
            args=runtime_args,
            kwargs=kwargs,
            return_alias=tuple(results) if pairs else results[0],
        )
    )
    return statements


for _name, _results in (
    ("merge_sort_keys", (GroupResultSource("keys", "keys"),)),
    (
        "merge_sort_pairs",
        (
            GroupResultSource("keys", "keys"),
            GroupResultSource("values", "values"),
        ),
    ),
):
    register_group_primitive(_name, lower=_lower_merge_sort, results=_results)
    for _partial in (False, True):
        register_rewrite_operation(
            _name + ("_partial" if _partial else ""),
            RewriteOperationSpecification(
                factory_namespaces=frozenset({"block", "warp"}),
                dtype_factory_kwargs=frozenset({"key_dtype", "value_dtype"}),
                runtime_arg_counts=frozenset(
                    {len(_results) + (2 if _partial else 0)}
                ),
                runtime_factory_kwargs=(),
                runtime_factory_kw_prerequisites=(),
                allowed_factory_kwargs=frozenset(
                    {
                        "key_dtype",
                        "value_dtype",
                        "items_per_thread",
                        "threads_per_block",
                        "threads_in_warp",
                        "descending",
                        "compare_op",
                    }
                ),
                required_factory_kwargs=frozenset(
                    {"key_dtype", "items_per_thread", "threads_per_block"}
                ),
                accepts_temp_storage=True,
                scalar_binding_kwargs=frozenset(),
                runtime_offset_kwarg=None,
                infer_payload=infer_merge_sort_payload,
            ),
        )
del _name, _results, _partial

__all__: tuple[str, ...] = ()
