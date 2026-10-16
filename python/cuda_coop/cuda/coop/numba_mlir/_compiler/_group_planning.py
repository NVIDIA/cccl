# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Provide operation planners with group, payload, and storage facts.

Each supported operation needs some of the same information: the participating
threads, a payload's element type and item count, compile-time scalar
controls, and the options of a ``TempStorage`` descriptor.
``GroupPlanningContext`` exposes those queries to operation-specific code
without direct access to all of ``_GroupCallPlanner``'s bookkeeping.

This context belongs to the group-resolution phase of the single
whole-function planner, before provider calls are materialized. It follows IR
assignments to recover facts that ordinary type inference has not established
yet and records element types inferred by earlier ``load()`` operations for
later calls. Once an operation has chosen a provider, it also checks that the
provider's storage and synchronization declarations agree with the group plan
and builds the call IR that carries that plan to the rewriting phase.
"""

from __future__ import annotations

from collections.abc import Callable
from numbers import Integral
from typing import TYPE_CHECKING, Any

import numba_cuda_mlir.numba_cuda.types as _numba_types
from numba_cuda_mlir import cuda as _cuda_module
from numba_cuda_mlir.cuda.local import array as _cuda_local_array

import cuda.coop._core.api as _common_api
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
from ._group_planner_support import (
    GroupRewriteError,
    _typed_group_payload_like,
    ir,
)
from ._operations import (
    _GROUP_LOWERING_PLAN_KWARG,
    StorageABI,
    expected_storage_reuse_barrier,
    factory_operation,
    provider_synchronization_matches,
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
    """Share one planner's analysis with operation implementations.

    An operation planner needs launch dimensions, payload types and extents,
    and static scalar controls before ordinary Numba typing runs. This
    context exposes those queries and builds provider-call replacements.
    The owning planner retains the IR and decides when to install them.

    The context retains inferred ThreadData dtypes at constructor sites so
    a later operation can use the type established by an earlier load.
    Aliases of the same constructor share this fact. Loop dtype candidates
    have a shorter lifetime: they exist for one query, seed a type cycle,
    and must pass a second query in which all definitions resolve and agree.
    See ``_loop_dtype`` for that two-stage analysis.

    Parameters
    ----------
    planner : _GroupCallPlanner
        Active planner supplying IR definitions, argument types, and launch
        facts. Create one context for each planner attempt.
    """

    __slots__ = (
        "__loop_dtypes",
        "__planner",
        "__seed_loop_dtypes",
        "__thread_data_dtypes",
    )

    def __init__(self, planner: _GroupCallPlanner) -> None:
        self.__planner = planner
        self.__thread_data_dtypes: dict[int, Any] = {}
        self.__loop_dtypes: dict[str, Any] = {}
        self.__seed_loop_dtypes = False

    @property
    def launch(self) -> Any:
        """Return launch dimensions and their sources for this attempt."""

        return self.__planner.launch

    def _definition(self, value: Any) -> Any:
        return self.__planner._definition(value)

    def _all_definitions(self, value: ir.Var) -> tuple[Any, ...]:
        return self.__planner._all_definitions(value)

    def _callable(self, value: Any) -> Any:
        return self.__planner._callable(value)

    def constant(self, value: Any) -> Any:
        """Require a constant control without literal-unroll dependencies.

        The planner may request literal specialization for a kernel argument.
        Use ``try_static_scalar`` when a runtime scalar is also permitted.
        """

        self.__planner._reject_literal_unroll_value(
            value, "a compile-time argument"
        )
        return self.__planner._constant(value)

    def try_constant(self, value: Any) -> tuple[bool, Any]:
        """Probe for a constant without requesting another compiler attempt.

        Return ``(resolved, value)``. Numba constant inference may resolve it;
        use ``try_static_scalar`` when the source must be explicitly static.
        """

        return self.__planner._try_constant(value)

    def try_static_scalar(self, value: Any) -> tuple[bool, Any]:
        """Recover an explicitly static scalar and keep its known width.

        Return ``(resolved, value)`` without evaluating runtime expressions or
        requesting specialization. A false result means a runtime operand.
        """

        return self.__planner._try_static_scalar(value)

    def try_static_scalar_provenance(self, value: Any) -> tuple[bool, Any]:
        """Recover a static scalar together with its recorded dtype.

        Return ``(resolved, provenance)``. The record distinguishes an untyped
        Python literal from a value that already has a compiler dtype.
        """

        return self.__planner._try_static_scalar_provenance(value)

    def bind(self, function: Any, call: ir.Expr) -> Any:
        """Bind IR arguments and defaults to the public signature."""

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
        """Apply the common API's selector rules after its wrapper is removed.

        Resolve the selector as a constant, normalize its spelling, and check
        ``allowed``. ``allow_none`` controls whether omission is valid.
        """

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
        """Check for a supported per-thread array payload.

        Accept ThreadData, CUDA local-array constructors, and array results
        from earlier group calls or generated result markers. Check the common
        API's ThreadData-only rule separately with ``is_thread_data``.
        An unresolved cycle raises a diagnostic. A false result still needs
        scalar validation by the caller.
        """

        return self.__planner._array_operand_state(operation, value)

    def is_thread_data(
        self, operation: str, parameter: str, value: Any
    ) -> bool:
        """Check the payload-origin restriction required by the common API.

        Follow aliases, generated result markers, and earlier group results
        to ThreadData constructors. An unresolved cycle raises a diagnostic
        that names the operation and parameter.
        """

        return self.__planner._thread_data_operand_state(
            operation,
            parameter,
            value,
        )

    def array_extent(self, value: Any) -> int | None:
        """Recover the unique known per-thread item count, or ``None``.

        Known counts must agree. Unknown paths contribute no count, so the
        caller must validate the payload's form separately.
        """

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
        registered factory's ABI and scopes.

        Some algorithms need temporary shared memory to exchange values
        between threads. A block ``load()`` operation with
        ``algorithm="transpose"`` uses one scratch region for the block; a
        warp ``load()`` operation with that algorithm needs a separate region
        for each participating physical or logical warp. For such plans,
        require exact block dimensions and one shared-memory slice per
        complete group instance. Non-exhaustive logical warps leave trailing
        lanes outside their groups. Direct algorithms for
        ``load()`` and ``store()`` need no scratch and skip those checks. An
        explicit ``temp_storage`` argument is supported only for a single
        block-scoped instance.

        Normally the provider's declared reuse barrier must match the plan.
        Caller-owned storage with ``auto_sync=False`` also permits the
        provider's execution-scope barrier declaration: the pointer rewrite
        bypasses its allocating wrapper and controls synchronization itself.
        This exception does not apply to implementation-owned storage. No IR
        is mutated here.

        A provider with GROUP execution scope is accepted only when the plan
        and factory require no TempStorage operand and no emitted reuse
        barrier. The helper can manage native synchronization and internal
        memory itself; this check does not establish that its implementation
        uses no memory.

        Parameters
        ----------
        lowering_plan : GroupLoweringPlan
            Supported plan with storage and execution requirements to check.
        factory : callable
            Selected host-side provider factory registered with operation
            metadata. It is not invoked here.
        runtime_temp_storage_supplied : bool or None, optional
            Whether the proposed provider call supplies ``temp_storage``. When
            the plan needs scratch, this flag must agree with caller
            ownership. ``None`` skips this argument-presence check only.

        Returns
        -------
        None
            The provider metadata and supported storage contracts agree.

        Raises
        ------
        TypeError
            ``lowering_plan`` is not a ``GroupLoweringPlan``.
        GroupRewriteError
            The plan is unsupported or incomplete, the provider is
            unregistered, or its ABI, scopes, ownership, or layout disagree.
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
        storage_bearing = storage.ownership is not StorageOwnership.NONE
        if topology.execution_scope is SynchronizationScope.GROUP and (
            storage_bearing
            or synchronization.storage_reuse_barrier
            is not SynchronizationScope.NONE
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
            block_threads = (
                exact_block_dim[0] * exact_block_dim[1] * exact_block_dim[2]
            )
            participating_threads = block_threads
            if (
                participation.group_kind == "threads_within_warp"
                and not participation.complete_parent_partition
                and 1 <= topology.logical_width <= 32
            ):
                participating_threads = (
                    (block_threads // 32)
                    * (32 // topology.logical_width)
                    * topology.logical_width
                )
            if (
                topology.logical_width * topology.instances
                != participating_threads
            ):
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
        expected_reuse_barrier = expected_storage_reuse_barrier(
            topology, storage
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
        if not provider_synchronization_matches(
            metadata, topology, synchronization, storage
        ):
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
        return_alias: ir.Var | tuple[ir.Var, ...] | None = None,
        common_root_operation: str | None = None,
    ) -> list[Any]:
        """Build a provider call carrying the validated group-lowering plan.

        Check the provider ABI and storage contract before embedding the plan
        in its reserved keyword argument. The later provider rewrite consumes
        this metadata instead of reconstructing the public group semantics.
        The assignments materialize non-IR arguments, invoke the factory,
        and assign the public result or requested payload alias. The caller
        installs them; this method does not replace the original instruction.

        Parameters
        ----------
        inst : ir.Assign
            Original public call assignment; supplies the result target,
            scope, and source location for generated statements.
        lowering_plan : GroupLoweringPlan
            Supported plan to validate and attach to the provider call.
        factory : callable
            Registered host-side provider factory selected by the operation
            family. Embedded as the generated call target, not invoked here.
        args : list of object
            Positional provider arguments, as existing IR variables or host
            values.
        kwargs : dict of str to object
            Provider keyword arguments. Copied before plan metadata is added;
            presence of ``temp_storage`` is checked against planned ownership.
        return_alias : ir.Var or tuple of ir.Var or None, optional
            Existing payload or payload tuple to assign to the public result
            after the provider call. ``None`` uses the provider's result.
        common_root_operation : str or None, optional
            Common API operation name to retain for downstream validation.
            When present, supplies the private marker unless ``kwargs``
            already has it.

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
            return_alias=return_alias,
            common_root_operation=common_root_operation,
        )

    def copy_array_payload(self, *args: Any, **kwargs: Any) -> None:
        """Append a fixed-size copy to preserve a provider input.

        Forward the source, destination, and pending statement list to the
        planner. The destination must already exist; unknown source extent is
        an error because the planner emits one copy statement pair per item.
        """

        self.__planner._copy_array_payload(*args, **kwargs)

    def typed_payload_like(self, *args: Any, **kwargs: Any) -> ir.Var:
        """Append a result-payload marker and return its IR variable.

        The marker carries a prototype and shape/type policy until the
        provider rewrite can create a local array. This only appends pending
        statements; the owning planner decides when to install them.
        """

        return self.__planner._typed_payload_like(*args, **kwargs)

    def box_group_operand(
        self, *args: Any, **kwargs: Any
    ) -> tuple[ir.Var, bool]:
        return self.__planner._boxed_group_operand(*args, **kwargs)

    def result_value(self, *args: Any, **kwargs: Any) -> ir.Var:
        return self.__planner._result_value(*args, **kwargs)

    def planning_binding(self, value: Any) -> ArgumentBinding:
        """Classify a scalar control from its explicit static provenance.

        Use explicit static provenance rather than general constant inference.
        A runtime expression remains a runtime binding even if another
        compiler analysis could fold it. The original runtime operand is
        retained by the operation family, not inside the returned binding.

        Parameters
        ----------
        value : ir.Var or object
            Optional scalar control such as ``valid_items``, ``offset``, or a
            ``load()`` fill value, represented by an IR variable or a value
            already known during compilation.

        Returns
        -------
        ArgumentBinding
            ``OMITTED`` for statically known ``None``, ``STATIC`` with the
            resolved value otherwise, or ``RUNTIME`` when static provenance is
            not established. Numeric validity and operation-specific
            constraints are checked later.
        """

        resolved, constant = self.try_static_scalar(value)
        if not resolved:
            return ArgumentBinding.runtime()
        if constant is None:
            return ArgumentBinding.omitted()
        return ArgumentBinding.static(constant)

    @staticmethod
    def _dtype_from_numba_type(value: object) -> _numba_types.Type | None:
        """Get an element dtype from a compiler type before IR typing runs.

        ``dtype`` uses this for an already known Numba type;
        ``_dtype_definition`` uses it for kernel argument types. An array
        contributes its element dtype, another Numba type is normalized
        directly, and an object that is not a Numba type returns ``None``.
        """

        if isinstance(value, _numba_types.Array):
            value = value.dtype
        elif not isinstance(value, _numba_types.Type):
            return None
        return normalize_dtype_param(value)

    @staticmethod
    def _one_dtype(candidates: set[Any], *, message: str) -> Any | None:
        """Return the sole known dtype collected by ``_complete_dtype``.

        ``candidates`` contains distinct, resolved dtypes. An empty set
        returns ``None``; more than one dtype raises ``GroupRewriteError``
        with the caller's ``message`` so it identifies the failing query.
        """

        if len(candidates) > 1:
            raise GroupRewriteError(message)
        return next(iter(candidates), None)

    def _complete_dtype(
        self,
        candidates: Any,
        *,
        message: str,
    ) -> Any | None:
        """Require agreement among the dtype candidates for one query.

        Variable, tuple, and phi queries call this with an iterable of
        normalized dtypes or ``None`` entries for unknown sources.
        During loop discovery, omit unknown candidates so a known entry value
        can seed a cycle. The later strict query requires every candidate to
        be known. Return ``None`` if information is incomplete. After that
        check, raise ``GroupRewriteError`` with ``message`` when the remaining
        dtypes disagree.
        """

        resolved = list(candidates)
        if self.__seed_loop_dtypes:
            # Discovery can use a known loop-entry type before the backedge
            # resolves. The final pass requires every candidate to resolve.
            resolved = [dtype for dtype in resolved if dtype is not None]
        if not resolved or any(dtype is None for dtype in resolved):
            return None
        return self._one_dtype(set(resolved), message=message)

    def _loop_dtype(self, value: ir.Var) -> Any | None:
        """Resolve a loop's type cycle without accepting an unknown producer.

        Group planning runs before ordinary type inference. A value
        initialized from a typed array may then feed a computation whose
        result becomes the next iteration's input. Following that input
        recursively reaches the same variable before its dtype is known.
        Returning ``None`` at every such backedge would discard the useful
        type supplied by the array.

        First propagate candidate dtypes from known definitions until they
        stop changing. During this discovery pass only, a join may omit
        unresolved inputs; a recursive backedge can use its variable's
        candidate. Then repeat the query with strict joins: every recorded
        definition must resolve and agree, including the computation on the
        backedge. An opaque helper remains unknown, and a conflicting dtype is
        still an error.

        Candidates live only for this query and are cleared on failure too.
        They are neither IR annotations nor permanent facts for later calls.
        This solves supported type-preserving cycles; it does not implement
        general type promotion or infer arbitrary device-helper return types.

        Parameters
        ----------
        value : ir.Var
            Variable passed to ``dtype`` without an existing recursion path.
            Its recorded assignments may include a loop's initial value and
            values produced on later iterations.

        Returns
        -------
        numba_types.Type or None
            Dtype verified on all paths, or ``None`` if any path is unknown.

        Raises
        ------
        GroupRewriteError
            Known recorded definitions require different dtypes.
        """
        self.__seed_loop_dtypes = True
        try:
            while True:
                previous = self.__loop_dtypes.copy()
                self.dtype(value, seen=set())
                if self.__loop_dtypes == previous:
                    break
            self.__seed_loop_dtypes = False
            return self.dtype(value, seen=set())
        finally:
            self.__seed_loop_dtypes = False
            self.__loop_dtypes.clear()

    def _result_dtype(
        self,
        definition: ir.Expr,
        *,
        index: int | None,
        seen: set[str],
    ) -> Any | None:
        """Infer a registered result's explicit or argument-derived dtype.

        The context's dtype traversal calls this before provider rewriting.
        It follows the public operation's result policy so a chain of group
        calls can be planned even though Numba has not typed their results
        yet.

        ``index`` selects a tuple result or is ``None`` for a direct result.
        Use a non-None dtype keyword first, then a fixed dtype, then the
        policy's source argument. Fixed int32 policies keep flag and rank
        results independent of the input payload's dtype. Traverse arguments
        with the caller's recursion path. Return ``None`` if no policy or
        dtype source applies.

        Parameters
        ----------
        definition : ir.Expr
            Potential public operation call whose result is used by
            another group operation.
        index : int or None
            Position in a tuple of public results, or ``None`` for a
            direct result.
        seen : set of str
            IR variable names already visited on this recursive path.
            Reusing the path prevents a loop-carried value from causing
            unbounded traversal.
        """

        resolved = self.__planner._result_source(definition, index)
        if resolved is None:
            return None
        result, bound = resolved
        if result.dtype_keyword is not None:
            dtype = self.constant(bound.arguments[result.dtype_keyword])
            if dtype is not None:
                return normalize_dtype_param(dtype)
        if result.fixed_dtype is not None:
            return result.fixed_dtype
        if result.dtype_parameter is None:
            return None
        return self.dtype(bound.arguments[result.dtype_parameter], seen=seen)

    def record_thread_data_dtype(
        self, value: Any, dtype: _numba_types.Type
    ) -> None:
        """Record a producer's dtype at the payload's constructor sites.

        Group planning precedes the provider rewrite that materializes
        payloads. A load into untyped ``ThreadData`` therefore records its
        inferred dtype here so subsequent group calls can recover it. Follow
        descriptor aliases, casts, phi inputs, and constant tuple projections
        to constructor calls, keying the cache by call-expression identity so
        aliases share the fact. Explicit constructor dtypes and earlier
        inferred dtypes must agree.

        Only recognized constructors reached by this traversal are updated;
        unresolved tuple projections and other leaves contribute no cache
        entry. This updates the planning context, not constructor arguments in
        the IR.

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
                in {ThreadData, _common_api.ThreadData}
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
        """Infer one tuple element across recorded container definitions.

        ``_dtype_definition`` calls this for a constant tuple index so a
        payload packed into a tuple retains its element dtype when unpacked.
        Track ``variable[index]`` separately from the container variable. This
        allows one projection to recurse without hiding a different element.
        Return ``None`` for an unresolved projection or cycle. Candidate
        agreement uses ``_complete_dtype``, including its temporary
        loop-discovery rule.

        Parameters
        ----------
        value : ir.Var or object
            Variable holding the tuple; other objects return ``None``.
        index : int
            Tuple position, with Python's negative-index convention. This is
            not an index into the payload array stored at that position.
        seen : set of str
            Recursion-path keys. Add the current projection in place, then
            give each recorded definition its own copy.
        """

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
        """Follow one tuple definition to the selected element's dtype.

        ``_tuple_dtype`` calls this for each recorded source of a tuple.
        Aliases, casts, iterator unpacking, and phi inputs retain the index. A
        built tuple delegates its selected element to ``dtype``. Unknown forms
        and out-of-range indices contribute no dtype; known conflicts at a phi
        join raise ``GroupRewriteError``.

        Registered calls with multiple results use the selected result's
        dtype policy: a non-None dtype keyword, then a fixed dtype such as
        int32 flags, then the source argument. This makes tuple-returned
        payloads available to later group planning before provider rewriting.

        Parameters
        ----------
        definition : object
            Assignment source, usually an IR variable or expression.
        index : int
            Position within the tuple, accepting Python negative indices.
        seen : set of str
            Variable and tuple-projection keys already on this recursion path.
            Each phi input receives a separate copy.

        Returns
        -------
        numba_types.Type or None
            Selected element's normalized dtype, or ``None`` if unknown.
        """

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

    def _dtype_definition(
        self, definition: Any, *, seen: set[str]
    ) -> Any | None:
        """Extract dtype evidence from one supported IR definition.

        Use argument types, constants, payload constructors, selected scalar
        operators, casts, and CUDA index attributes. Follow aliases and joins
        through the context's dtype queries. A tuple projection can have its
        own element type. Array indexing uses the source's element type.

        ThreadData constructors use an explicit dtype or a dtype recorded by a
        producer. An unrelated call stays unknown unless the scalar-cast
        helper recognizes it. This limited analysis supplies provider
        selection before ordinary Numba typing; it does not execute kernel
        expressions.

        ``dtype`` calls this for each recorded source of its variable.
        ``definition`` is that assignment source; ``seen`` contains variable
        and tuple-projection keys already on the recursion path. Return a
        normalized dtype or ``None`` when the source supplies no known dtype.

        Generated payload markers either inherit their prototype's dtype or
        select fixed int32 output. Registered results use their declared dtype
        policy, including fixed types or source-argument dtypes, before
        scalar-call inference.
        """

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
        if function in {ThreadData, _common_api.ThreadData}:
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
        if function is _typed_group_payload_like and definition.args:
            if (
                len(definition.args) >= 3
                and self.constant(definition.args[2]) == "int32"
            ):
                return _numba_types.int32
            return self.dtype(definition.args[0], seen=seen)
        result_dtype = self._result_dtype(definition, index=None, seen=seen)
        if result_dtype is not None:
            return result_dtype
        return scalar_call_dtype(
            function,
            definition.args,
            lambda value: self.dtype(value, seen=set(seen)),
        )

    def _attribute_chain(
        self, value: Any
    ) -> tuple[Any, tuple[str, ...]] | None:
        """Recover an attribute path and its constant root without calling it.

        CUDA index recognition compares the root object's identity and the
        attribute names. Return ``None`` when the root cannot be recovered.
        """

        attributes: list[str] = []
        current = self._definition(value)
        while isinstance(current, ir.Expr) and current.op == "getattr":
            attributes.append(current.attr)
            current = self._definition(current.value)
        if not isinstance(current, (ir.Global, ir.FreeVar, ir.Const)):
            return None
        attributes.reverse()
        return current.value, tuple(attributes)

    def dtype(
        self, value: object, *, seen: set[str] | None = None
    ) -> _numba_types.Type | None:
        """Infer a normalized dtype from facts known during group planning.

        Use argument types, scalar constants and operators, supported CUDA
        index attributes, local-array constructors, and ``ThreadData``
        declarations or recorded producer dtypes. Result policies and payload
        markers supply returned payload types. Follow aliases, casts,
        phi inputs, and tuple projections. Array indexing contributes the
        source element dtype. This analysis runs before full Numba typing.

        Loop backedges can use a candidate established by a known incoming
        definition, provided every recorded definition then resolves to the
        same type. Unknown definitions and unseeded cycles return ``None``;
        fully known but inconsistent paths are rejected.

        Parameters
        ----------
        value : object
            IR variable to inspect, or a Numba compiler type to normalize
            directly. Array types contribute their element dtype. Other inputs
            are accepted as probes and return ``None``; a raw Python scalar is
            not treated as an IR constant by this entry point.
        seen : set of str, optional
            Recursion-path variable names and tuple-projection keys. The
            current name is added in place; definitions are visited with
            separate copies.

        Returns
        -------
        numba_types.Type or None
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
            inferred = self.dtype(value, seen=set())
            return inferred if inferred is not None else self._loop_dtype(value)
        if value.name in seen:
            # Only the loop-resolution query supplies a candidate here.
            return self.__loop_dtypes.get(value.name)
        seen.add(value.name)
        inferred = self._complete_dtype(
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
        if self.__seed_loop_dtypes and inferred is not None:
            self.__loop_dtypes[value.name] = inferred
        return inferred

    def payload_write_dtype(self, payload: Any) -> Any | None:
        """Infer a payload dtype from values assigned through its aliases.

        The Store planner calls this when an untyped ``ThreadData`` payload
        has no declared or producer-inferred dtype. ``payload`` is its IR
        variable, possibly reached through aliases.

        Inspect the known types of element writes across the function. All
        known types must agree or ``TypeError`` is raised. Unknown writes
        contribute no evidence; ``None`` means no known write dtype was found.
        The scan does not prove that every element is initialized or that all
        paths write.
        """

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
        """Recover one storage contract from recorded descriptor definitions.

        The Load/Store planner uses this to validate an explicit storage
        operand before choosing and recording its provider's storage plan.
        Parse each recognized constructor with the planning constant resolver,
        then require its normalized contract to agree with the others. A
        concrete non-descriptor path, including a ``None`` initializer,
        invalidates a value that also reaches a descriptor. Backedges
        contribute no new leaf.

        Equivalent constructors may merge when automatic synchronization is
        used. With ``auto_sync=False``, all aliases must reach exactly one
        call expression: merging separately constructed regions would lose the
        origin needed to reason about caller-managed synchronization. This
        checks provenance and options, not backing storage or capacity.

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
                in {TempStorage, _common_api.TempStorage}
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
                "cuda.coop.numba_mlir TempStorage aliases have "
                "inconsistent contracts"
            )
        descriptor = next(iter(candidates), None)
        if descriptor is not None and non_descriptor:
            raise GroupRewriteError(
                "cuda.coop.numba_mlir TempStorage variables must be bound to "
                f"a TempStorage descriptor on every path; {value.name!r} is "
                "also bound to a non-descriptor value such as None. "
                "Remove the None initializer or construct the descriptor "
                "unconditionally."
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
