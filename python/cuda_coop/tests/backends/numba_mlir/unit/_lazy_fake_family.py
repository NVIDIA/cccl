# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Test-only family used to prove additive Numba compiler registration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cuda.coop._core import (
    ArgumentKind,
    CudaxCallDescription,
    GroupLoweringPlan,
    GroupLoweringTarget,
    GroupOperandKind,
    ImplementationProvenance,
    LogicalResultContract,
    ParameterClassification,
    ParameterRole,
    ResultContract,
    ResultOwnership,
    ResultVisibility,
    StorageOwnership,
    SynchronizationScope,
    make_group_primitive_call,
    plan_group_primitive,
)
from cuda.coop._core.group import _dispatch
from cuda.coop._core.group._contracts import _contracts
from cuda.coop.numba_mlir._compiler._operations import (
    GroupResultSource,
    RewriteOperationSpec,
    StorageABI,
    register_factory,
    register_group_primitive,
    register_rewrite_operation,
)

SCALAR_OPERATION = "_test_lazy_family_scalar"
ARRAY_OPERATION = "_test_lazy_family_array"
PAIR_OPERATION = "_test_lazy_family_pair"
OPERATIONS = (SCALAR_OPERATION, ARRAY_OPERATION, PAIR_OPERATION)

_GROUP_KIND_BY_OPERATION = {
    SCALAR_OPERATION: "thread",
    ARRAY_OPERATION: "block",
    PAIR_OPERATION: "warp",
}
_EXECUTION_SCOPE_BY_OPERATION = {
    SCALAR_OPERATION: SynchronizationScope.NONE,
    ARRAY_OPERATION: SynchronizationScope.BLOCK,
    PAIR_OPERATION: SynchronizationScope.WARP,
}
_STORAGE_ABI_BY_OPERATION = {
    SCALAR_OPERATION: StorageABI.NONE,
    ARRAY_OPERATION: StorageABI.LEADING_POINTER,
    PAIR_OPERATION: StorageABI.LEADING_POINTER,
}
_RESULT_SOURCES_BY_OPERATION = {
    SCALAR_OPERATION: (GroupResultSource("value", None),),
    ARRAY_OPERATION: (GroupResultSource("values", "values"),),
    PAIR_OPERATION: (
        GroupResultSource("key", None),
        GroupResultSource("values", "values"),
    ),
}


@dataclass(frozen=True)
class LazyFamilySemantics:
    """Minimal semantics spanning scalar, array, and mixed results."""

    operation: str
    result_dtypes: tuple[Any, ...]
    array_extent: int

    def __post_init__(self) -> None:
        if self.operation not in OPERATIONS:
            raise ValueError(f"unsupported lazy-family operation {self.operation!r}")
        expected_results = 2 if self.operation == PAIR_OPERATION else 1
        if len(self.result_dtypes) != expected_results:
            raise ValueError("result_dtypes does not match the operation results")
        if (
            not isinstance(self.array_extent, int)
            or isinstance(self.array_extent, bool)
            or self.array_extent < 1
        ):
            raise ValueError("array_extent must be a positive integer")

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            "lazy-fake-family",
            self.operation,
            self.result_dtypes,
            self.array_extent,
        )

    @property
    def result_visibility(self) -> ResultVisibility:
        return ResultVisibility.PER_MEMBER

    @property
    def returns_value(self) -> bool:
        return True


def _classifications(
    semantics: LazyFamilySemantics,
) -> tuple[ParameterClassification, ...]:
    names = {
        SCALAR_OPERATION: ("value",),
        ARRAY_OPERATION: ("values",),
        PAIR_OPERATION: ("key", "values"),
    }[semantics.operation]
    return tuple(
        ParameterClassification(name, ArgumentKind.RUNTIME, ParameterRole.INPUT)
        for name in names
    )


def _result_contract(semantics: LazyFamilySemantics) -> ResultContract:
    if semantics.operation == SCALAR_OPERATION:
        layouts = (("scalar", GroupOperandKind.SCALAR, 1),)
    elif semantics.operation == ARRAY_OPERATION:
        layouts = (("array", GroupOperandKind.ARRAY, semantics.array_extent),)
    else:
        layouts = (
            ("key", GroupOperandKind.SCALAR, 1),
            ("values", GroupOperandKind.ARRAY, semantics.array_extent),
        )
    return ResultContract(
        tuple(
            LogicalResultContract(
                name=name,
                dtype=dtype,
                visibility=ResultVisibility.PER_MEMBER,
                ownership=ResultOwnership.EACH_MEMBER,
                operand_kind=operand_kind,
                items_per_member=items_per_member,
            )
            for (name, operand_kind, items_per_member), dtype in zip(
                layouts, semantics.result_dtypes
            )
        )
    )


def _plan(
    call,
    resolved_group,
    launch,
    semantics: LazyFamilySemantics,
) -> GroupLoweringPlan:
    expected_group_kind = _GROUP_KIND_BY_OPERATION[semantics.operation]
    if resolved_group.kind != expected_group_kind:
        raise ValueError(
            f"{semantics.operation} requires group kind {expected_group_kind!r}"
        )
    storage_abi = _STORAGE_ABI_BY_OPERATION[semantics.operation]
    storage_ownership = (
        StorageOwnership.NONE
        if storage_abi is StorageABI.NONE
        else StorageOwnership.IMPLEMENTATION
    )
    result = _result_contract(semantics)
    topology, participation, synchronization, temp_storage = _contracts(
        resolved_group,
        launch,
        result=result,
        storage_ownership=storage_ownership,
        cpp_type=None if storage_abi is StorageABI.NONE else "FakeStorage",
        auto_sync=storage_abi is not StorageABI.NONE,
    )
    target = {
        "thread": GroupLoweringTarget.CUDAX_GROUP,
        "block": GroupLoweringTarget.CUB_BLOCK,
        "warp": GroupLoweringTarget.CUB_WARP,
    }[resolved_group.kind]
    return GroupLoweringPlan(
        target=target,
        call=call,
        resolved_group=resolved_group,
        implementation=CudaxCallDescription(
            primitive=semantics.operation,
            header="test/lazy_fake_family.cuh",
            namespace="test::lazy_family",
        ),
        topology=topology,
        participation=participation,
        result=result,
        synchronization=synchronization,
        temp_storage=temp_storage,
        provenance=ImplementationProvenance(
            library="test",
            header="test/lazy_fake_family.cuh",
            cpp_class="test::lazy_family",
            method="invoke",
        ),
    )


_dispatch._register_group_operation_family(
    LazyFamilySemantics,
    classifications=_classifications,
    planner=_plan,
    group_kinds=frozenset({"thread", "warp", "block"}),
    unsupported_group_message="lazy fake family requires thread, warp, or block",
)


@dataclass(frozen=True)
class _FakeInvocable:
    operation: str
    storage_abi: StorageABI
    execution_scope: SynchronizationScope
    synchronization_scope: SynchronizationScope
    temp_storage_bytes: int
    temp_storage_alignment: int
    files: tuple[str, ...] = ("lazy-fake-family-test.ltoir",)
    specialization: None = None

    def __call__(self, *args: Any) -> None:
        del args


INVOCABLES = {
    operation: _FakeInvocable(
        operation=operation,
        storage_abi=_STORAGE_ABI_BY_OPERATION[operation],
        execution_scope=_EXECUTION_SCOPE_BY_OPERATION[operation],
        synchronization_scope=_EXECUTION_SCOPE_BY_OPERATION[operation],
        temp_storage_bytes=(0 if operation == SCALAR_OPERATION else 24),
        temp_storage_alignment=(1 if operation == SCALAR_OPERATION else 8),
    )
    for operation in OPERATIONS
}
FACTORY_CALLS: list[tuple[str, Any]] = []
PLANNING_EVENTS: list[tuple[str, str, bool, GroupLoweringPlan]] = []


def _make_provider(operation: str):
    def provider(*runtime_args: Any, value_type: Any):
        if runtime_args:
            raise AssertionError("provider factories receive only specialization args")
        FACTORY_CALLS.append((operation, value_type))
        return INVOCABLES[operation]

    provider.__name__ = f"{operation}_provider"
    return register_factory(
        provider,
        operation=operation,
        namespace="lazy_test_namespace",
        storage_abi=_STORAGE_ABI_BY_OPERATION[operation],
        execution_scope=_EXECUTION_SCOPE_BY_OPERATION[operation],
        synchronization_scope=_EXECUTION_SCOPE_BY_OPERATION[operation],
    )


PROVIDERS = {operation: _make_provider(operation) for operation in OPERATIONS}


def _infer_payload_at(index: int):
    def infer_payload(context, inference) -> None:
        value = inference.runtime_args[index]
        _, payload = inference.candidate(index)
        dtype = payload.dtype if payload is not None else context.dtype(value)
        inference.infer_kwarg("value_type", dtype)

    return infer_payload


def _lower(
    context,
    inst,
    *,
    operation: str,
    group,
    bound,
    is_common_root: bool,
):
    sources = _RESULT_SOURCES_BY_OPERATION[operation]
    result_dtypes = tuple(
        context.dtype(bound.arguments[source.dtype_parameter]) for source in sources
    )
    if any(dtype is None for dtype in result_dtypes):
        raise TypeError("lazy fake family requires statically known result dtypes")
    array_sources = [
        source.array_parameter
        for source in sources
        if source.array_parameter is not None
    ]
    array_extent = (
        1
        if not array_sources
        else context.array_extent(bound.arguments[array_sources[0]])
    )
    if array_extent is None:
        raise TypeError("lazy fake family requires a static array extent")
    semantics = LazyFamilySemantics(
        operation=operation,
        result_dtypes=result_dtypes,
        array_extent=array_extent,
    )
    plan = plan_group_primitive(
        make_group_primitive_call(group, semantics),
        context.launch,
    ).require_supported()
    PLANNING_EVENTS.append((operation, group.kind, is_common_root, plan))

    runtime_names = {
        SCALAR_OPERATION: ("value",),
        ARRAY_OPERATION: ("values",),
        PAIR_OPERATION: ("key", "values"),
    }[operation]
    runtime_args = [bound.arguments[name] for name in runtime_names]
    if operation == SCALAR_OPERATION:
        return_alias = bound.arguments["value"]
        dtype_source = bound.arguments["value"]
    elif operation == ARRAY_OPERATION:
        return_alias = bound.arguments["values"]
        dtype_source = bound.arguments["values"]
    else:
        return_alias = (bound.arguments["key"], bound.arguments["values"])
        dtype_source = bound.arguments["values"]
    return context.rewrite_call(
        inst,
        lowering_plan=plan,
        factory=PROVIDERS[operation],
        args=runtime_args,
        kwargs={"value_type": context.dtype(dtype_source)},
        return_alias=return_alias,
    )


for _operation in OPERATIONS:
    register_group_primitive(
        _operation,
        lower=_lower,
        results=_RESULT_SOURCES_BY_OPERATION[_operation],
    )
    _runtime_arg_count = 2 if _operation == PAIR_OPERATION else 1
    _payload_index = 1 if _operation == PAIR_OPERATION else 0
    register_rewrite_operation(
        _operation,
        RewriteOperationSpec(
            factory_namespaces=frozenset({"lazy_test_namespace"}),
            dtype_factory_kwargs=frozenset({"value_type"}),
            runtime_arg_counts=frozenset({_runtime_arg_count}),
            runtime_factory_kwargs=(),
            runtime_factory_kw_prerequisites=(),
            allowed_factory_kwargs=frozenset({"value_type"}),
            required_factory_kwargs=frozenset({"value_type"}),
            accepts_temp_storage=False,
            scalar_binding_kwargs=frozenset(),
            runtime_offset_kwarg=None,
            infer_payload=_infer_payload_at(_payload_index),
        ),
    )
del _operation, _payload_index, _runtime_arg_count
