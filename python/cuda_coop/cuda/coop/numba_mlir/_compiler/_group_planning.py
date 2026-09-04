# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Declared whole-function planning interface for primitive families."""

from __future__ import annotations

from numbers import Integral
from typing import Any

from numba_cuda_mlir import types

import cuda.coop._core.api as _portable_api
from cuda.coop._core import (
    ArgumentBinding,
    GroupLoweringPlan,
    StorageOwnership,
    SynchronizationScope,
)

from .._temp_storage import TempStorage
from .._thread_data import ThreadData
from ._group_planner_support import (
    GroupRewriteError,
    _cuda_module,
    _typed_group_payload_like,
    ir,
)
from ._operations import (
    _GROUP_LOWERING_PLAN_KWARG,
    StorageABI,
    factory_operation,
)
from ._parameters import (
    _python_scalar_dtype,
    _scalar_cast_dtype,
    _scalar_operator_result_dtype,
    normalize_dtype_param,
)


class GroupPlanningContext:
    """Stable cross-family view of one whole-function planner."""

    __slots__ = ("__planner",)

    def __init__(self, planner: Any) -> None:
        self.__planner = planner

    @property
    def launch(self) -> Any:
        return self.__planner.launch

    def _definition(self, value: Any) -> Any:
        return self.__planner._definition(value)

    def _all_definitions(self, value: ir.Var) -> tuple[Any, ...]:
        return self.__planner._all_definitions(value)

    def _callable(self, value: Any) -> Any:
        return self.__planner._callable(value)

    def constant(self, value: Any) -> Any:
        return self.__planner._constant(value)

    def try_constant(self, value: Any) -> tuple[bool, Any]:
        return self.__planner._try_constant(value)

    def try_static_scalar(self, value: Any) -> tuple[bool, Any]:
        return self.__planner._try_static_scalar(value)

    def try_static_scalar_provenance(self, value: Any) -> tuple[bool, Any]:
        return self.__planner._try_static_scalar_provenance(value)

    def bind(self, function: Any, call: ir.Expr) -> Any:
        return self.__planner._bind(function, call)

    def validate_common_selector(
        self,
        operation: str,
        parameter: str,
        value: Any,
        allowed: frozenset[str],
        *,
        allow_none: bool = False,
    ) -> Any:
        return self.__planner._validate_common_selector(
            operation,
            parameter,
            value,
            allowed,
            allow_none=allow_none,
        )

    def is_none(self, value: Any) -> bool:
        return self.__planner._is_none(value)

    def is_array(self, operation: str, value: Any) -> bool:
        return self.__planner._array_operand_state(operation, value)

    def is_thread_data(self, operation: str, parameter: str, value: Any) -> bool:
        return self.__planner._thread_data_operand_state(
            operation,
            parameter,
            value,
        )

    def array_extent(self, value: Any) -> int | None:
        return self.__planner._array_extent(value)

    def new_var(self, scope: Any, loc: ir.Loc, stem: str) -> ir.Var:
        return self.__planner._new_var(scope, loc, stem)

    def value_var(
        self,
        statements: list[Any],
        *,
        scope: Any,
        loc: ir.Loc,
        stem: str,
        value: Any,
    ) -> ir.Var:
        return self.__planner._value_var(
            statements,
            scope=scope,
            loc=loc,
            stem=stem,
            value=value,
        )

    @staticmethod
    def _validate_provider_contract(
        lowering_plan: GroupLoweringPlan,
        factory: Any,
        *,
        runtime_temp_storage_supplied: bool | None = None,
    ) -> None:
        if not isinstance(lowering_plan, GroupLoweringPlan):
            raise TypeError("lowering_plan must be a GroupLoweringPlan")
        if lowering_plan.unsupported is not None:
            raise GroupRewriteError(
                "cuda.coop.numba_mlir cannot select a provider for an "
                "unsupported lowering plan"
            )
        metadata = factory_operation(factory)
        if metadata is None:
            raise GroupRewriteError(
                "cuda.coop.numba_mlir lowering plan selected an unregistered "
                "provider factory"
            )
        topology = lowering_plan.topology
        participation = lowering_plan.participation
        synchronization = lowering_plan.synchronization
        storage = lowering_plan.temp_storage
        if (
            topology is None
            or participation is None
            or synchronization is None
            or storage is None
        ):
            raise GroupRewriteError(
                "cuda.coop.numba_mlir provider selection requires complete "
                "group topology, participation, synchronization, and storage "
                "contracts"
            )
        storage_bearing = storage.ownership is not StorageOwnership.NONE
        if topology.execution_scope is SynchronizationScope.GROUP and (
            storage_bearing
            or synchronization.storage_reuse_barrier is not SynchronizationScope.NONE
            or metadata.storage_abi is not StorageABI.NONE
            or metadata.execution_scope is not SynchronizationScope.GROUP
            or metadata.synchronization_scope is not SynchronizationScope.NONE
        ):
            raise GroupRewriteError(
                "cuda.coop.numba_mlir provider execution scope 'group' is "
                "supported only for storage-free providers with no emitted "
                "synchronization"
            )
        if storage_bearing:
            if storage.address_space != "shared":
                raise GroupRewriteError(
                    "cuda.coop.numba_mlir storage-bearing providers require "
                    "shared-address-space TempStorage"
                )
            if (
                storage.instances != topology.instances
                or storage.instance_index != topology.instance_index
            ):
                raise GroupRewriteError(
                    "cuda.coop.numba_mlir TempStorage layout disagrees with "
                    "its group topology"
                )
            exact_block_dim = participation.exact_block_dim
            if exact_block_dim is None:
                raise GroupRewriteError(
                    "cuda.coop.numba_mlir storage-bearing providers require "
                    "exact block dimensions"
                )
            block_threads = exact_block_dim[0] * exact_block_dim[1] * exact_block_dim[2]
            if topology.logical_width * topology.instances != block_threads:
                raise GroupRewriteError(
                    "cuda.coop.numba_mlir group topology does not cover the "
                    "exact block dimensions"
                )
        caller_owned = storage.ownership is StorageOwnership.CALLER
        if (
            storage_bearing
            and runtime_temp_storage_supplied is not None
            and (caller_owned != runtime_temp_storage_supplied)
        ):
            raise GroupRewriteError(
                "cuda.coop.numba_mlir TempStorage ownership disagrees with "
                "the provider call arguments"
            )
        if caller_owned and (
            topology.execution_scope is not SynchronizationScope.BLOCK
            or topology.instances != 1
        ):
            if topology.execution_scope is SynchronizationScope.WARP:
                raise GroupRewriteError(
                    "cuda.coop.numba_mlir caller-owned TempStorage is not "
                    "supported for warp-scoped cooperative primitives; omit "
                    "temp_storage so the implementation can provide one "
                    "aligned slice per group instance"
                )
            raise GroupRewriteError(
                "cuda.coop.numba_mlir caller-owned TempStorage is supported "
                "only for single-instance block-scoped cooperative primitives"
            )
        expected_reuse_barrier = (
            topology.execution_scope
            if storage_bearing and storage.auto_sync
            else SynchronizationScope.NONE
        )
        if synchronization.storage_reuse_barrier is not expected_reuse_barrier:
            raise GroupRewriteError(
                "cuda.coop.numba_mlir TempStorage automatic synchronization "
                "disagrees with the planned storage-reuse barrier"
            )
        expected = {
            "storage_abi": (
                StorageABI.LEADING_POINTER if storage_bearing else StorageABI.NONE
            ),
            "execution_scope": topology.execution_scope,
            "synchronization_scope": synchronization.storage_reuse_barrier,
        }
        mismatches = [
            f"{name}={getattr(metadata, name).value!r} (plan {planned.value!r})"
            for name, planned in expected.items()
            if name != "synchronization_scope"
            if getattr(metadata, name) is not planned
        ]
        planned_synchronization = expected["synchronization_scope"]
        allowed_synchronization = {planned_synchronization}
        # The provider's convenience ``_alloc`` wrapper owns its declared
        # reuse barrier. Pointer rewrites bypass that wrapper, and the compiler
        # rewrite emits the descriptor-selected barrier only when auto_sync is
        # enabled.
        if (
            planned_synchronization is SynchronizationScope.NONE
            and caller_owned
            and not storage.auto_sync
        ):
            allowed_synchronization.add(expected["execution_scope"])
        if metadata.synchronization_scope not in allowed_synchronization:
            mismatches.append(
                "synchronization_scope="
                f"{metadata.synchronization_scope.value!r} "
                f"(plan {planned_synchronization.value!r})"
            )
        if mismatches:
            details = ", ".join(mismatches)
            raise GroupRewriteError(
                f"cuda.coop.numba_mlir provider {metadata.operation!r} "
                f"metadata disagrees with its lowering plan: {details}"
            )

    def rewrite_call(
        self,
        inst: ir.Assign,
        *,
        lowering_plan: GroupLoweringPlan,
        factory: Any,
        args: list[Any],
        kwargs: dict[str, Any],
        return_alias: ir.Var | tuple[ir.Var, ...] | None = None,
        common_root_operation: str | None = None,
    ) -> list[Any]:
        self._validate_provider_contract(
            lowering_plan,
            factory,
            runtime_temp_storage_supplied="temp_storage" in kwargs,
        )
        if _GROUP_LOWERING_PLAN_KWARG in kwargs:
            raise GroupRewriteError(
                "cuda.coop.numba_mlir family lowering used a reserved provider keyword"
            )
        kwargs = {
            **kwargs,
            _GROUP_LOWERING_PLAN_KWARG: lowering_plan,
        }
        return self.__planner._rewritten_call(
            inst,
            factory=factory,
            args=args,
            kwargs=kwargs,
            return_alias=return_alias,
            common_root_operation=common_root_operation,
        )

    def copy_array_payload(self, *args: Any, **kwargs: Any) -> None:
        self.__planner._copy_array_payload(*args, **kwargs)

    def typed_payload_like(self, *args: Any, **kwargs: Any) -> ir.Var:
        return self.__planner._typed_payload_like(*args, **kwargs)

    def box_group_operand(self, *args: Any, **kwargs: Any) -> tuple[ir.Var, bool]:
        return self.__planner._boxed_group_operand(*args, **kwargs)

    def result_value(self, *args: Any, **kwargs: Any) -> ir.Var:
        return self.__planner._result_value(*args, **kwargs)

    def planning_binding(self, value: Any) -> ArgumentBinding:
        resolved, constant = self.try_static_scalar(value)
        if not resolved:
            return ArgumentBinding.runtime()
        if constant is None:
            return ArgumentBinding.omitted()
        return ArgumentBinding.static(constant)

    @staticmethod
    def _dtype_from_numba_type(value: Any) -> Any | None:
        if isinstance(value, types.Array):
            value = value.dtype
        elif not isinstance(value, types.Type):
            return None
        return normalize_dtype_param(value)

    @staticmethod
    def _one_dtype(candidates: set[Any], *, message: str) -> Any | None:
        if len(candidates) > 1:
            raise GroupRewriteError(message)
        return next(iter(candidates), None)

    @classmethod
    def _complete_dtype(
        cls,
        candidates: Any,
        *,
        message: str,
    ) -> Any | None:
        resolved = list(candidates)
        if not resolved or any(dtype is None for dtype in resolved):
            return None
        return cls._one_dtype(set(resolved), message=message)

    def _result_dtype(
        self,
        definition: ir.Expr,
        *,
        index: int | None,
        seen: set[str],
    ) -> Any | None:
        resolved = self.__planner._result_source(definition, index)
        if resolved is None:
            return None
        result, bound = resolved
        if result.dtype_parameter is None:
            return None
        return self.dtype(bound.arguments[result.dtype_parameter], seen=seen)

    def _tuple_dtype(
        self,
        value: Any,
        index: int,
        *,
        seen: set[str],
    ) -> Any | None:
        if not isinstance(value, ir.Var):
            return None
        seen_key = f"{value.name}[{index}]"
        if seen_key in seen:
            return None
        seen.add(seen_key)
        return self._complete_dtype(
            (
                self._tuple_dtype_definition(
                    definition,
                    index,
                    seen=set(seen),
                )
                for definition in self._all_definitions(value)
            ),
            message=("cuda.coop.numba_mlir tuple projections have inconsistent dtypes"),
        )

    def _tuple_dtype_definition(
        self,
        definition: Any,
        index: int,
        *,
        seen: set[str],
    ) -> Any | None:
        if isinstance(definition, ir.Var):
            return self._tuple_dtype(definition, index, seen=seen)
        if not isinstance(definition, ir.Expr):
            return None
        if definition.op in {"cast", "exhaust_iter"}:
            return self._tuple_dtype(definition.value, index, seen=seen)
        if definition.op == "phi":
            return self._complete_dtype(
                (
                    self._tuple_dtype(incoming, index, seen=set(seen))
                    for incoming in getattr(definition, "incoming_values", ())
                ),
                message=(
                    "cuda.coop.numba_mlir loop-carried tuple payloads have "
                    "inconsistent dtypes"
                ),
            )
        if definition.op == "build_tuple":
            items = tuple(getattr(definition, "items", ()))
            if not -len(items) <= index < len(items):
                return None
            return self.dtype(items[index], seen=seen)
        if definition.op == "call":
            return self._result_dtype(definition, index=index, seen=seen)
        return None

    def _dtype_definition(self, definition: Any, *, seen: set[str]) -> Any | None:
        if isinstance(definition, ir.Var):
            return self.dtype(definition, seen=seen)
        if isinstance(definition, ir.Arg):
            if not 0 <= definition.index < len(self.__planner.state.args):
                return None
            return self._dtype_from_numba_type(
                self.__planner.state.args[definition.index]
            )
        if isinstance(definition, (ir.Global, ir.FreeVar, ir.Const)):
            return _python_scalar_dtype(definition.value)
        if not isinstance(definition, ir.Expr):
            return None
        if definition.op in {"cast", "exhaust_iter"}:
            return self.dtype(definition.value, seen=seen)
        if definition.op == "phi":
            return self._complete_dtype(
                (
                    self.dtype(incoming, seen=set(seen))
                    for incoming in getattr(definition, "incoming_values", ())
                ),
                message=(
                    "cuda.coop.numba_mlir payload aliases have inconsistent dtypes"
                ),
            )
        if definition.op in {"getitem", "static_getitem"}:
            index = getattr(definition, "index", None)
            if isinstance(index, ir.Var):
                resolved, index = self.try_constant(index)
                if not resolved:
                    return self.dtype(definition.value, seen=seen)
            if isinstance(index, Integral) and not isinstance(index, bool):
                tuple_dtype = self._tuple_dtype(
                    definition.value,
                    int(index),
                    seen=set(seen),
                )
                if tuple_dtype is not None:
                    return tuple_dtype
            return self.dtype(definition.value, seen=seen)
        if definition.op in {"binop", "inplace_binop"}:
            return _scalar_operator_result_dtype(
                getattr(definition, "fn", None),
                self.dtype(getattr(definition, "lhs", None), seen=set(seen)),
                self.dtype(getattr(definition, "rhs", None), seen=set(seen)),
            )
        if definition.op == "unary":
            return _scalar_operator_result_dtype(
                getattr(definition, "fn", None),
                self.dtype(getattr(definition, "value", None), seen=set(seen)),
            )
        if definition.op == "getattr":
            chain = self._attribute_chain(definition.value)
            if chain is not None:
                root, attributes = chain
                if root is _cuda_module and (*attributes, definition.attr) in {
                    (index, component)
                    for index in ("blockDim", "blockIdx", "gridDim", "threadIdx")
                    for component in ("x", "y", "z")
                }:
                    return types.int32
            return None
        if definition.op != "call":
            return None
        function = self._callable(definition.func)
        if function in {ThreadData, _portable_api.ThreadData}:
            bound = self.bind(function, definition)
            resolved, dtype = self.try_constant(bound.arguments["dtype"])
            if resolved and dtype is not None:
                return normalize_dtype_param(dtype)
            return None
        if function is _cuda_module.local.array:
            if len(definition.args) >= 2:
                resolved, dtype = self.try_constant(definition.args[1])
                if resolved:
                    return normalize_dtype_param(dtype)
            dtype_ref = dict(definition.kws).get("dtype")
            if dtype_ref is not None:
                resolved, dtype = self.try_constant(dtype_ref)
                if resolved:
                    return normalize_dtype_param(dtype)
            return None
        if function is _typed_group_payload_like and definition.args:
            return self.dtype(definition.args[0], seen=seen)
        result_dtype = self._result_dtype(definition, index=None, seen=seen)
        if result_dtype is not None:
            return result_dtype
        cast_dtype = _scalar_cast_dtype(function)
        if cast_dtype is None:
            return None
        if len(definition.args) == 1:
            inferred = _scalar_operator_result_dtype(
                function,
                self.dtype(definition.args[0], seen=set(seen)),
            )
            if inferred is not None:
                return inferred
        return cast_dtype

    def _attribute_chain(self, value: Any) -> tuple[Any, tuple[str, ...]] | None:
        attributes: list[str] = []
        current = self._definition(value)
        while isinstance(current, ir.Expr) and current.op == "getattr":
            attributes.append(current.attr)
            current = self._definition(current.value)
        if not isinstance(current, (ir.Global, ir.FreeVar, ir.Const)):
            return None
        attributes.reverse()
        return current.value, tuple(attributes)

    def dtype(self, value: Any, *, seen: set[str] | None = None) -> Any | None:
        if not isinstance(value, ir.Var):
            return self._dtype_from_numba_type(value)
        if seen is None:
            seen = set()
        if value.name in seen:
            return None
        seen.add(value.name)
        return self._complete_dtype(
            (
                self._dtype_definition(
                    definition,
                    seen=set(seen),
                )
                for definition in self._all_definitions(value)
            ),
            message="cuda.coop.numba_mlir payload aliases have inconsistent dtypes",
        )

    def payload_write_dtype(self, payload: Any) -> Any | None:
        """Infer an untyped payload from values written through its aliases."""

        if not isinstance(payload, ir.Var):
            return None
        alias_names = {payload.name}
        changed = True
        while changed:
            changed = False
            for block in self.__planner.func_ir.blocks.values():
                for statement in block.body:
                    if not isinstance(statement, ir.Assign):
                        continue
                    definition = statement.value
                    sources: tuple[ir.Var, ...] = ()
                    if isinstance(definition, ir.Var):
                        sources = (definition,)
                    elif isinstance(definition, ir.Expr) and definition.op in {
                        "cast",
                        "exhaust_iter",
                    }:
                        if isinstance(definition.value, ir.Var):
                            sources = (definition.value,)
                    elif isinstance(definition, ir.Expr) and definition.op == "phi":
                        sources = tuple(
                            incoming
                            for incoming in getattr(definition, "incoming_values", ())
                            if isinstance(incoming, ir.Var)
                        )
                    source_names = {source.name for source in sources}
                    if (
                        statement.target.name in alias_names
                        or source_names & alias_names
                    ):
                        additions = {statement.target.name, *source_names} - alias_names
                        if additions:
                            alias_names.update(additions)
                            changed = True

        inferred = None
        static_setitem_cls = getattr(ir, "StaticSetItem", None)
        for block in self.__planner.func_ir.blocks.values():
            for statement in block.body:
                if isinstance(statement, ir.SetItem) or (
                    static_setitem_cls is not None
                    and isinstance(statement, static_setitem_cls)
                ):
                    target = getattr(statement, "target", None)
                    value = getattr(statement, "value", None)
                else:
                    continue
                if not isinstance(target, ir.Var) or target.name not in alias_names:
                    continue
                if not isinstance(value, ir.Var):
                    continue
                value_dtype = self.dtype(value)
                if value_dtype is None:
                    continue
                if inferred is None:
                    inferred = value_dtype
                elif inferred != value_dtype:
                    raise TypeError(
                        "cuda.coop.numba_mlir could not infer one consistent "
                        "dtype from payload writes"
                    )
        return inferred

    def _temp_storage_definition(
        self,
        definition: Any,
        *,
        seen: set[str],
    ) -> tuple[int | None, int | None, bool, str] | None:
        if isinstance(definition, ir.Var):
            return self.temp_storage(definition, seen=seen)
        if not isinstance(definition, ir.Expr):
            return None
        if definition.op in {"cast", "exhaust_iter"}:
            return self.temp_storage(definition.value, seen=seen)
        if definition.op == "phi":
            incoming_values = tuple(getattr(definition, "incoming_values", ()))
            resolved = tuple(
                self.temp_storage(incoming, seen=set(seen))
                for incoming in incoming_values
            )
            candidates = {
                descriptor for descriptor in resolved if descriptor is not None
            }
            if candidates and any(descriptor is None for descriptor in resolved):
                raise GroupRewriteError(
                    "cuda.coop.numba_mlir TempStorage aliases have "
                    "inconsistent contracts"
                )
            if len(candidates) > 1:
                raise GroupRewriteError(
                    "cuda.coop.numba_mlir TempStorage aliases have "
                    "inconsistent contracts"
                )
            return next(iter(candidates), None)
        if definition.op != "call":
            return None
        function = self._callable(definition.func)
        if function not in {TempStorage, _portable_api.TempStorage}:
            return None
        bound = self.bind(function, definition)
        values = {name: self.constant(value) for name, value in bound.arguments.items()}
        descriptor = TempStorage(**values)
        return (
            descriptor.size_in_bytes,
            descriptor.alignment,
            descriptor.auto_sync,
            descriptor.sharing,
        )

    def temp_storage(
        self,
        value: Any,
        *,
        seen: set[str] | None = None,
    ) -> tuple[int | None, int | None, bool, str] | None:
        if not isinstance(value, ir.Var):
            return None
        if seen is None:
            seen = set()
        if value.name in seen:
            return None
        seen.add(value.name)
        candidates = {
            descriptor
            for definition in self._all_definitions(value)
            if (
                descriptor := self._temp_storage_definition(
                    definition,
                    seen=set(seen),
                )
            )
            is not None
        }
        if len(candidates) > 1:
            raise GroupRewriteError(
                "cuda.coop.numba_mlir TempStorage aliases have inconsistent contracts"
            )
        return next(iter(candidates), None)


__all__ = ["GroupPlanningContext"]
