# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from collections.abc import Callable
from numbers import Integral
from typing import TYPE_CHECKING, Any

import numba_cuda_mlir.numba_cuda.types as _numba_types
from numba_cuda_mlir import cuda as _cuda_module
from numba_cuda_mlir.cuda.local import array as _cuda_local_array

import cuda.coop._core.api as _portable_api
from cuda.coop._core import (
    ArgumentBinding,
    GroupLoweringPlan,
    StorageOwnership,
    SynchronizationScope,
)

from .._temp_storage import TempStorage
from .._thread_data import ThreadData
from ._descriptor_provenance import (
    descriptor_definitions,
    payload_write_dtypes,
    temp_storage_constructor,
)
from ._group_planner_support import GroupRewriteError, ir
from ._operations import (
    _GROUP_LOWERING_PLAN_KWARG,
    StorageABI,
    factory_operation,
)
from ._parameters import (
    _python_scalar_dtype,
    normalize_dtype_param,
)
from ._scalar_provenance import (
    cuda_index_dtype,
    scalar_call_dtype,
    scalar_expression_dtype,
)

if TYPE_CHECKING:
    from ._group_planner import _GroupCallPlanner


class GroupPlanningContext:
    """Stable cross-family view of one whole-function planner."""

    __slots__ = ("__planner", "__thread_data_dtypes")

    def __init__(self, planner: _GroupCallPlanner) -> None:
        self.__planner = planner
        self.__thread_data_dtypes: dict[int, Any] = {}

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
        self.__planner._reject_literal_unroll_value(
            value, "a compile-time argument"
        )
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

    def is_thread_data(
        self, operation: str, parameter: str, value: Any
    ) -> bool:
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
        factory: Callable[..., Any],
        *,
        runtime_temp_storage_supplied: bool | None = None,
    ) -> None:
        """Check the selected provider against planned storage and execution.

        This is the boundary between compiler-neutral planning and the private
        provider call. Require complete topology, participation,
        synchronization, and storage facts, then compare them with the
        registered factory's ABI and scopes. Storage-bearing plans must cover
        the exact block with shared storage slices matching the group instances.
        Caller-owned storage is supported only for one block-scoped instance.

        Normally the provider's declared reuse barrier must match the plan.
        Caller-owned storage with ``auto_sync=False`` also permits the
        provider's execution-scope barrier declaration: the pointer rewrite
        bypasses its allocating wrapper and controls synchronization itself.
        This exception does not apply to implementation-owned storage. No IR is
        mutated here.

        Parameters
        ----------
        lowering_plan : GroupLoweringPlan
            Supported plan whose storage and execution requirements are checked.
        factory : callable
            Selected host-side provider factory registered with operation
            metadata. It is not invoked here.
        runtime_temp_storage_supplied : bool or None, optional
            Whether the proposed provider call supplies ``temp_storage``. For a
            storage-bearing plan, a boolean must agree with caller ownership.
            ``None`` skips this argument-presence check only.

        Returns
        -------
        None
            The provider metadata and supported storage contracts agree.

        Raises
        ------
        TypeError
            ``lowering_plan`` is not a ``GroupLoweringPlan``.
        GroupRewriteError
            The plan is unsupported or incomplete, the provider is unregistered,
            or its ABI, scopes, storage ownership, or layout are incompatible.
        """

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
        if topology.execution_scope is SynchronizationScope.GROUP:
            raise GroupRewriteError(
                "cuda.coop.numba_mlir provider execution scope 'group' has no "
                "storage or synchronization emitter"
            )
        storage_bearing = storage.ownership is not StorageOwnership.NONE
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
            block_threads = (
                exact_block_dim[0] * exact_block_dim[1] * exact_block_dim[2]
            )
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
                StorageABI.LEADING_POINTER
                if storage_bearing
                else StorageABI.NONE
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
        factory: Callable[..., Any],
        args: list[Any],
        kwargs: dict[str, Any],
        common_root_operation: str | None = None,
    ) -> list[Any]:
        """Build a provider call carrying the validated group-lowering plan.

        Check the provider ABI and storage contract before embedding the plan in
        its reserved keyword argument. The later provider rewrite consumes this
        metadata, so it does not have to reconstruct the public group semantics.
        The returned assignments materialize non-IR arguments and invoke the
        factory with the original result target. The caller installs them into
        the function; this method does not replace the original instruction.

        Parameters
        ----------
        inst : ir.Assign
            Original public call assignment; supplies the result target, scope,
            and source location for generated statements.
        lowering_plan : GroupLoweringPlan
            Supported semantic plan to validate and attach to the provider call.
        factory : callable
            Registered host-side provider factory selected by the operation
            family. Embedded as the generated call target, not invoked here.
        args : list of object
            Positional provider arguments, as existing IR variables or host
            values.
        kwargs : dict of str to object
            Provider keyword arguments. Copied before plan metadata is added;
            presence of ``temp_storage`` is checked against planned ownership.
        common_root_operation : str or None, optional
            Common API operation name to retain for downstream validation. When
            present, supplies the private marker unless ``kwargs`` already has
            it.

        Returns
        -------
        list of object
            Ordered argument-materialization and call assignments replacing
            ``inst``.

        Raises
        ------
        GroupRewriteError
            Provider-contract validation fails or ``kwargs`` uses the reserved
            lowering-plan keyword.
        """

        self._validate_provider_contract(
            lowering_plan,
            factory,
            runtime_temp_storage_supplied="temp_storage" in kwargs,
        )
        if _GROUP_LOWERING_PLAN_KWARG in kwargs:
            raise GroupRewriteError(
                "cuda.coop.numba_mlir family lowering "
                "used a reserved provider keyword"
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
            common_root_operation=common_root_operation,
        )

    def planning_binding(self, value: Any) -> ArgumentBinding:
        """Classify a scalar control from its explicit static provenance.

        Use explicit static provenance rather than general constant inference. A
        runtime expression remains a runtime binding even if another compiler
        analysis could fold it. The original runtime operand is retained by the
        operation family, not inside the returned binding.

        Parameters
        ----------
        value : ir.Var or object
            Optional scalar control such as ``valid_items``, ``offset``, or a
            load default, represented by an IR variable or an already-static
            value.

        Returns
        -------
        ArgumentBinding
            ``OMITTED`` for statically known ``None``, ``STATIC`` with the
            resolved value otherwise, or ``RUNTIME`` when static provenance is
            not established. Numeric validity and operation-specific constraints
            are checked later.
        """

        resolved, constant = self.try_static_scalar(value)
        if not resolved:
            return ArgumentBinding.runtime()
        if constant is None:
            return ArgumentBinding.omitted()
        return ArgumentBinding.static(constant)

    @staticmethod
    def _dtype_from_numba_type(value: Any) -> Any | None:
        if isinstance(value, _numba_types.Array):
            value = value.dtype
        elif not isinstance(value, _numba_types.Type):
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

    def record_thread_data_dtype(
        self, value: Any, dtype: _numba_types.Type
    ) -> None:
        """Record a producer's element dtype at the payload's constructor sites.

        Group planning precedes the provider rewrite that materializes payloads.
        A load into untyped ``ThreadData`` therefore records its inferred dtype
        here so subsequent group calls can recover it. Follow descriptor
        aliases, casts, phi inputs, and constant tuple projections to
        constructor calls, keying the cache by call-expression identity so
        aliases share the fact. Explicit constructor dtypes and earlier inferred
        dtypes must agree.

        Only recognized constructors reached by this traversal are updated;
        unresolved tuple projections and other leaves contribute no cache entry.
        This updates the planning context, not constructor arguments in the IR.

        Parameters
        ----------
        value : ir.Var
            Producer's output payload, possibly reached through supported
            aliases or tuple projections.
        dtype : numba_types.Type
            Normalized element dtype inferred by the producer.

        Returns
        -------
        None
            Any recognized constructor sites now carry the inferred dtype.

        Raises
        ------
        GroupRewriteError
            A reached constructor already has a different explicit or inferred
            dtype. Cache entries recorded before the conflict are not rolled
            back.
        """

        def payload_definitions(current, seen):
            for _, definition in descriptor_definitions(
                current, self._all_definitions, seen=seen
            ):
                if not isinstance(definition, ir.Expr) or definition.op not in {
                    "getitem",
                    "static_getitem",
                }:
                    yield definition
                    continue
                index = definition.index
                if isinstance(index, ir.Var):
                    resolved, index = self.try_constant(index)
                    if not resolved:
                        continue
                if not isinstance(index, Integral) or isinstance(index, bool):
                    continue
                index = int(index)
                next_seen = {*seen, current.name}
                for packed in payload_definitions(definition.value, next_seen):
                    if (
                        isinstance(packed, ir.Expr)
                        and packed.op == "build_tuple"
                        and -len(packed.items) <= index < len(packed.items)
                    ):
                        yield from payload_definitions(
                            packed.items[index], next_seen
                        )

        for definition in payload_definitions(value, set()):
            if (
                isinstance(definition, ir.Expr)
                and definition.op == "call"
                and self._callable(definition.func)
                in {ThreadData, _portable_api.ThreadData}
            ):
                previous = self._dtype_definition(definition, seen=set())
                if previous is not None and previous != dtype:
                    raise GroupRewriteError(
                        "cuda.coop.numba_mlir ThreadData aliases have "
                        "inconsistent dtypes"
                    )
                self.__thread_data_dtypes[id(definition)] = dtype

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
            message=(
                "cuda.coop.numba_mlir tuple "
                "projections have inconsistent dtypes"
            ),
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
        return None

    def _dtype_definition(
        self, definition: Any, *, seen: set[str]
    ) -> Any | None:
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
                    "cuda.coop.numba_mlir payload "
                    "aliases have inconsistent dtypes"
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
        if definition.op in {"binop", "inplace_binop", "unary"}:
            return scalar_expression_dtype(
                definition, lambda value: self.dtype(value, seen=set(seen))
            )
        if definition.op == "getattr":
            return cuda_index_dtype(
                definition, self._attribute_chain, _cuda_module
            )
        if definition.op != "call":
            return None
        function = self._callable(definition.func)
        if function in {ThreadData, _portable_api.ThreadData}:
            bound = self.bind(function, definition)
            resolved, dtype = self.try_constant(bound.arguments["dtype"])
            if resolved and dtype is not None:
                return normalize_dtype_param(dtype)
            return self.__thread_data_dtypes.get(id(definition))
        if function is _cuda_local_array:
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
        return scalar_call_dtype(
            function,
            definition.args,
            lambda value: self.dtype(value, seen=set(seen)),
        )

    def _attribute_chain(
        self, value: Any
    ) -> tuple[Any, tuple[str, ...]] | None:
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
        """Infer a normalized dtype from facts available during group planning.

        Use argument types, scalar constants and operators, supported CUDA index
        attributes, local-array constructors, and ``ThreadData`` declarations or
        recorded producer dtypes. Follow aliases, casts, phi inputs, and tuple
        projections; array indexing contributes the source element dtype. This
        is a limited pre-typing analysis, not full Numba type inference.

        Every reaching definition must produce a dtype before agreement is
        checked. Unknown definitions and cycles return ``None`` even when
        another path has a known type; fully known but inconsistent paths are
        rejected.

        Parameters
        ----------
        value : ir.Var or Numba type or object
            IR value to inspect, or a compiler type to normalize directly. Array
            types contribute their element dtype. Other non-variable values have
            no inferred dtype through this entry point.
        seen : set of str, optional
            Recursion-path variable names and tuple-projection keys. The current
            name is added in place; definitions are visited with separate
            copies.

        Returns
        -------
        object or None
            Normalized compiler dtype when every relevant path is known and
            agrees, or ``None`` when inference is incomplete.

        Raises
        ------
        GroupRewriteError
            Fully resolved aliases or tuple projections disagree on the dtype.
        """

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
            message=(
                "cuda.coop.numba_mlir payload aliases have inconsistent dtypes"
            ),
        )

    def payload_write_dtype(self, payload: Any) -> Any | None:
        """Infer an untyped payload from values written through its aliases."""

        inferred = None
        for value_dtype in payload_write_dtypes(
            self.__planner.func_ir, payload, self.dtype
        ):
            if inferred is None:
                inferred = value_dtype
            elif inferred != value_dtype:
                raise TypeError(
                    "cuda.coop.numba_mlir could not infer one consistent "
                    "dtype from payload writes"
                )
        return inferred

    def temp_storage(
        self,
        value: Any,
        *,
        seen: set[str] | None = None,
    ) -> tuple[int | None, int | None, bool, str] | None:
        """Recover one storage contract from reaching descriptor definitions.

        Parse each recognized constructor with the planning constant resolver,
        then require its normalized contract to agree with the others. A
        concrete non-descriptor path, including a ``None`` initializer,
        invalidates a value that also reaches a descriptor. Backedges contribute
        no new leaf.

        Equivalent constructors may merge when automatic synchronization is
        used. With ``auto_sync=False``, all aliases must reach exactly one call
        expression: merging separately constructed regions would lose the origin
        needed to reason about caller-managed synchronization. This checks
        provenance and constructor options, not backing storage or capacity.

        Parameters
        ----------
        value : ir.Var or object
            Value expected to name a storage descriptor. Non-variables return
            ``None`` without parsing.
        seen : set of str, optional
            Recursion-path names passed to ``descriptor_definitions``. The
            supplied set is not mutated.

        Returns
        -------
        tuple or None
            ``(size_in_bytes, alignment, auto_sync, sharing)`` for the unique
            contract, or ``None`` if no recognized constructor is reached. The
            first two fields may be ``None`` to defer size or alignment
            selection.

        Raises
        ------
        GroupRewriteError
            Contracts conflict, descriptor and non-descriptor paths mix,
            multiple constructor sites use manual synchronization, or call
            syntax is invalid.
        ForceLiteralArg
            A constructor option requires literal argument specialization.
        """

        if not isinstance(value, ir.Var):
            return None
        candidates = set()
        sites: set[int] = set()
        non_descriptor = False
        for _, definition in descriptor_definitions(
            value, self._all_definitions, seen=seen
        ):
            if not (
                isinstance(definition, ir.Expr)
                and definition.op == "call"
                and self._callable(definition.func)
                in {TempStorage, _portable_api.TempStorage}
            ):
                non_descriptor = True
                continue
            descriptor = temp_storage_constructor(
                definition,
                lambda argument, *, name: self.constant(argument),
                syntax_error=GroupRewriteError,
            )
            candidates.add(
                (
                    descriptor.size_in_bytes,
                    descriptor.alignment,
                    descriptor.auto_sync,
                    descriptor.sharing,
                )
            )
            sites.add(id(definition))
        if len(candidates) > 1:
            raise GroupRewriteError(
                "cuda.coop.numba_mlir TempStorage "
                "aliases have inconsistent contracts"
            )
        descriptor = next(iter(candidates), None)
        if descriptor is not None and non_descriptor:
            raise GroupRewriteError(
                "cuda.coop.numba_mlir TempStorage variables must be bound to "
                f"a TempStorage descriptor on every path; {value.name!r} is "
                "also bound to a non-descriptor value such as None. Remove "
                "the None initializer or construct "
                "the descriptor unconditionally."
            )
        if descriptor is not None and descriptor[2] is False and len(sites) > 1:
            raise GroupRewriteError(
                "cuda.coop.numba_mlir TempStorage with auto_sync=False must "
                f"be constructed at exactly one site; {value.name!r} reaches "
                f"{len(sites)} constructor sites. The compiler cannot verify "
                "caller synchronization when it merges these regions. "
                "Construct the descriptor once or set auto_sync=True."
            )
        return descriptor


__all__ = ["GroupPlanningContext"]
