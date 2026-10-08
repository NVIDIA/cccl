# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Replace kernel group descriptions with calls to chosen implementations.

A kernel can describe its participating threads with ``this_block()`` and
``group_by()``, then pass that group to operations such as ``load()``. Those
Python descriptions cannot execute on the device. This module recovers them
from the kernel's IR, resolves their sizes against the configured launch, and
asks the operation's registered planning code to choose an implementation.
These compile-time ``ThreadGroup`` and ``ThreadHierarchy`` values are called
group descriptors here. ``ThreadData`` supplies the per-thread payload and
is lowered separately to a local array.

``CoopWholeFunctionPlanner`` invokes this work after device-helper inlining
and before type inference. ``has_group_markers`` first determines whether
group resolution is needed; ``_GroupPlanning`` then requests launch facts and
runs ``_GroupCallPlanner``. The latter builds replacements, checks that group
descriptors have no remaining runtime uses, and substitutes calls to private
provider factories. A provider factory is a host callable that specializes the
selected implementation for types, item counts, and launch dimensions.

The same whole-function planner repairs the changed IR before its next phase
materializes those provider calls and allocates payloads and scratch storage.
This module does not register a separate Numba pass. Its analysis is limited
to supported constructors and expressions; ordinary Numba type inference
handles the rest of the kernel afterward.
"""

import inspect
from enum import Enum
from numbers import Integral
from typing import TYPE_CHECKING, Any, cast

import numba_cuda_mlir.numba_cuda.types as _numba_types
from numba_cuda_mlir.cuda.local import array as _cuda_local_array
from numba_cuda_mlir.extending import require_launch_config
from numba_cuda_mlir.numba_cuda.core.errors import ForceLiteralArg

import cuda.coop._core.api as _common_api
import cuda.coop._core.api._dispatch as _common_dispatch
from cuda.coop._core import (
    LaunchFactOrigin,
    LaunchFacts,
    ThreadGroup,
    ThreadHierarchy,
    normalize_thread_level,
    resolve_thread_group,
)

from .._temp_storage import TempStorage
from .._thread_data import ThreadData
from ._group_errors import (
    CyclicArrayProvenanceError,
    EscapingGroupDescriptorError,
    InconsistentArrayExtentError,
    InconsistentLoopPayloadExtentError,
    InconsistentLoopTupleExtentError,
    InconsistentTupleExtentError,
    InvalidGroupSelectorError,
    NonConstantGroupArgumentError,
    NonConstantThreadGroupError,
    UnknownResultExtentError,
)
from ._group_planner_support import (
    _COMMON_GROUP_CONSTRUCTORS,
    _GROUP_CONSTRUCTORS,
    _GROUP_METHODS,
    _NAME_COUNTER,
    _PAYLOAD_DTYPE_LIKE,
    GroupRewriteError,
    _group_operation_name,
    _is_common_root_operation,
    _typed_group_payload_like,
    ir,
)
from ._group_planning import GroupPlanningContext
from ._operations import group_primitive
from ._scalar_provenance import (
    try_resolve_static_scalar,
    try_resolve_static_scalar_provenance,
)

if TYPE_CHECKING:
    from ._planner import CoopWholeFunctionPlanner


class _GroupCallPlanner:
    """Resolve group descriptions before ordinary type inference.

    A group such as ``this_block()`` describes which threads cooperate; it
    is not an object that the device should construct. For one compiler
    attempt, this planner recovers those descriptions and payload facts
    from IR, asks each operation's registered planner to choose a
    provider, then replaces the public calls with calls that the next
    rewriting phase understands.

    Analysis and mutation are separate. Helpers collect descriptor
    assignments and replacement statements while the original assignment
    definitions remain available. ``run()`` installs the replacements only
    after every operation and remaining descriptor use has passed
    validation. Create a fresh planner for each attempt; its caches and
    pending replacements belong to that IR.

    Parameters
    ----------
    state : compiler state
        Numba state containing ``func_ir`` and argument types in ``args``.
        Device helpers must already be inlined into this function.
    launch_config : dict of str to object
        Normalized configuration returned by ``require_launch_config``.
        Its exact dimensions let group resolution choose a supported
        topology.

    Attributes
    ----------
    func_ir : ir.FunctionIR
        Function being analyzed and, after successful planning, rewritten.
    launch : LaunchFacts
        Shared-core representation of the configured launch, including
        where its dimensions came from.
    descriptor_assigns : set of ir.Assign
        Group and hierarchy construction, alias, and cast assignments
        whose compile-time meaning has been consumed. Their targets become
        ``None``.
    dead_func_names : set of str
        Callable-variable names consumed by descriptor or operation
        rewriting. Their old assignments also become ``None`` so Numba
        need not type the removed Python callables.
    replacements : dict of ir.Assign to list
        Original operation assignments mapped to pending provider-call IR.
        Building this map does not yet replace the function's block
        bodies.
    context : GroupPlanningContext
        Restricted analysis interface passed to operation planners. It
        exposes group, payload, scalar, and storage facts and retains
        inferred payload dtypes for later calls in this planning attempt.
    """

    def __init__(self, state, launch_config: dict[str, Any]) -> None:
        self.state = state
        self.func_ir = state.func_ir
        self.launch_config = launch_config
        self.launch = self._make_launch_facts(launch_config)
        # Pending edits refer to the original IR until all calls validate.
        self.dead_func_names: set[str] = set()
        self.descriptor_assigns: set[ir.Assign] = set()
        self.replacements: dict[ir.Assign, list[Any]] = {}
        # Host-side thread groups and hierarchies, keyed by IR variable name.
        self._group_cache: dict[str, ThreadGroup] = {}
        self._hierarchy_cache: dict[str, ThreadHierarchy] = {}
        self._compile_context = None
        self._group_method_invocables: dict[tuple[Any, ...], Any] = {}
        self.context = GroupPlanningContext(self)

    def _provider_compile_context(self):
        """Resolve and reuse one provider compile context for this planner.

        All generated group-method helpers in the attempt share the same
        header and toolkit context. Cache it lazily so plans with no such
        helper do not need to resolve compiler inputs here.
        """

        if self._compile_context is None:
            from ._nvrtc import resolve_compile_context

            self._compile_context = resolve_compile_context()
        return self._compile_context

    @staticmethod
    def _make_launch_facts(config: dict[str, Any]) -> LaunchFacts:
        """Convert the configured launch to shared group-planning facts.

        The shared resolver consumes ``LaunchFacts`` rather than Numba
        metadata. Record the configured block, grid, and optional cluster
        dimensions with origins identifying the backend launch configuration.
        Supplying a cluster is also the evidence for ``cluster_launch``. This
        launch interface does not establish cooperative-grid launch support,
        so that fact stays false.

        Parameters
        ----------
        config : dict of str to object
            Normalized launch configuration supplied by the compiler
            dispatcher. Other entries, such as dynamic shared-memory bytes,
            are not topology facts and do not enter this record.

        Returns
        -------
        LaunchFacts
            Dimensions and their origins for resolving a requested thread
            group.
        """

        block = config.get("block")
        grid = config.get("grid")
        cluster = config.get("cluster")
        cluster_launch = cluster is not None
        origins = [
            LaunchFactOrigin(
                fact="exact_block_dim",
                source="numba_cuda_mlir_launch_config",
                verified=True,
            ),
            LaunchFactOrigin(
                fact="exact_grid_dim",
                source="numba_cuda_mlir_launch_config",
                verified=True,
            ),
            LaunchFactOrigin(
                fact="cluster_launch",
                source="numba_cuda_mlir_launch_config",
                verified=True,
            ),
        ]
        if cluster is not None:
            origins.append(
                LaunchFactOrigin(
                    fact="exact_cluster_dim",
                    source="numba_cuda_mlir_launch_config",
                    verified=True,
                )
            )
        return LaunchFacts(
            exact_block_dim=block,
            exact_grid_dim=grid,
            exact_cluster_dim=cluster,
            cluster_launch=cluster_launch,
            cooperative_launch=False,
            provenance=tuple(origins),
        )

    def _definition(self, value: Any) -> Any:
        """Look up the expression that supplies a variable's value.

        A definition is the right-hand side of an assignment, such as the
        ``ThreadData(...)`` call assigned to a payload variable. Call and
        descriptor recognition use this helper when they need a single source.
        For an IR variable ``value``, return Numba's recorded definition, or
        ``None`` if it is missing or ambiguous. Other inputs pass through
        unchanged. Use ``_all_definitions`` when several assignments may supply
        the value after a branch or loop.
        """

        if not isinstance(value, ir.Var):
            return value
        try:
            return self.func_ir.get_definition(value)
        except KeyError:
            return None

    def _all_definitions(self, value: ir.Var) -> tuple[Any, ...]:
        """Find every recorded assignment that may supply a variable's value.

        An assignment is a reaching definition for a use if execution can get
        from that assignment to the use without replacing the value. For
        example, two branches can assign different ``ThreadData`` objects to
        the same payload variable. Array analysis must check both sources
        before choosing a provider. This helper conservatively collects all
        recorded sources for the name; it does not prove that each assignment
        can reach the particular use being analyzed.

        ``value`` is an IR variable; its ``name`` indexes Numba's definition
        table. Return the recorded right-hand-side expressions as a tuple,
        falling back to ``_definition`` if the table has no entry. An empty
        tuple means no definition was found, so callers have no source to
        classify.
        """

        definitions = getattr(self.func_ir, "_definitions", {}).get(
            value.name, ()
        )
        if definitions:
            return tuple(definitions)
        definition = self._definition(value)
        return () if definition is None else (definition,)

    def _callable(self, value: Any) -> Any:
        """Recover the Python object behind a call target or attribute chain.

        Resolve globals, captured values, constants, and their attributes
        without calling the resulting object. Operation recognition compares
        the returned object's identity with registered constructors and
        operations. Return ``None`` for an unresolved base or failed attribute
        lookup. Despite this helper's name, the recovered object is not
        required to be callable.
        """

        current = self._definition(value)
        attrs: list[str] = []
        while isinstance(current, ir.Expr) and current.op == "getattr":
            attrs.append(current.attr)
            current = self._definition(current.value)
        if isinstance(current, (ir.Global, ir.FreeVar, ir.Const)):
            obj = current.value
        elif callable(current):
            obj = current
        else:
            return None
        try:
            for attr in reversed(attrs):
                obj = getattr(obj, attr)
        except (AttributeError, ImportError):
            return None
        return obj

    def _reject_literal_unroll_value(self, value: Any, parameter: str) -> None:
        """Reject constant controls that depend on literal unrolling.

        Group planning needs shapes and selectors before the literal-unroll
        pass has expanded its iterations. Trace all reaching definitions and
        expression operands for a recognized ``literal_unroll`` call instead
        of trying to resolve an iteration value prematurely. Unrelated
        literal-unroll loops are allowed. Cycles terminate the search without
        establishing a dependency.

        Parameters
        ----------
        value : ir.Var or object
            Argument whose IR dependencies are inspected. Non-variables have
            no dependencies to inspect.
        parameter : str
            Description of the shape, selector, or other compile-time control
            used in the diagnostic.

        Returns
        -------
        None
            The value has no detected dependency on a recognized unroll call.

        Raises
        ------
        GroupRewriteError
            A dependency reaches the standard or vendored Numba literal-unroll
            marker; use constant controls or an ordinary loop with a fixed
            cooperative shape.
        """

        def depends_on_unroll(current, seen):
            if not isinstance(current, ir.Var) or current.name in seen:
                return False
            seen = {*seen, current.name}
            for definition in self._all_definitions(current):
                if isinstance(definition, ir.Var):
                    if depends_on_unroll(definition, seen):
                        return True
                elif isinstance(definition, ir.Expr):
                    if definition.op == "call":
                        function = self._callable(definition.func)
                        if getattr(
                            function, "__name__", None
                        ) == "literal_unroll" and getattr(
                            function, "__module__", None
                        ) in {
                            "numba.misc.special",
                            "numba_cuda_mlir.numba_cuda.misc.special",
                        }:
                            return True
                    if any(
                        depends_on_unroll(source, seen)
                        for source in definition.list_vars()
                    ):
                        return True
            return False

        if depends_on_unroll(value, set()):
            raise GroupRewriteError(
                "cuda.coop.numba_mlir does not support literal_unroll values "
                f"that determine {parameter}. Write separate "
                "cooperative calls with explicit constant shapes/selectors, "
                "or use an ordinary loop with a fixed cooperative shape."
            )

    def _reject_literal_unroll_constructors(self) -> None:
        """Check constructor controls before building any group replacements.

        Inspect recognized group, hierarchy, payload, and storage
        constructors. Their arguments must be usable before Numba expands
        ``literal_unroll``; ``_reject_literal_unroll_value`` supplies the
        diagnostic for a detected dependency. Ordinary loops and unrelated
        unrolls remain available.
        """

        constructors = {
            *_GROUP_CONSTRUCTORS,
            ThreadHierarchy,
            ThreadData,
            _common_api.ThreadData,
            TempStorage,
            _common_api.TempStorage,
        }
        for block in self.func_ir.blocks.values():
            for inst in block.body:
                call = getattr(inst, "value", None)
                if not isinstance(call, ir.Expr) or call.op != "call":
                    continue
                function = self._callable(call.func)
                if function not in constructors:
                    continue
                for argument in (*call.args, *(value for _, value in call.kws)):
                    self._reject_literal_unroll_value(
                        argument, f"{function.__name__} arguments"
                    )

    def _constant(self, value: Any) -> Any:
        """Resolve a required compile-time argument or request specialization.

        Try direct argument and constant definitions first, then reconstruct
        hierarchy or group descriptors, and finally use Numba constant
        inference. A non-literal kernel argument requests dispatcher
        specialization rather than being treated as an unsupported value.
        Descriptor reconstruction may populate the planner's caches. This is
        the required-constant path; callers classifying optional runtime
        controls use ``_try_static_scalar`` instead.

        Parameters
        ----------
        value : ir.Var or object
            IR variable to resolve, or an already-resolved value returned
            unchanged.

        Returns
        -------
        object
            Compile-time value, including ``None`` or a reconstructed
            descriptor.

        Raises
        ------
        ForceLiteralArg
            A function argument must be recompiled with a literal type.
        NonConstantGroupArgumentError
            The value cannot be resolved by descriptor or constant inference.
        GroupRewriteError
            A recognized descriptor has unsupported call syntax or arguments.
        """

        if not isinstance(value, ir.Var):
            return value
        definition = self._definition(value)
        if isinstance(definition, ir.Arg):
            position = definition.index
            argtype = self.state.args[position]
            if not isinstance(argtype, _numba_types.Literal):
                raise ForceLiteralArg({position})
            return argtype.literal_value
        if isinstance(definition, (ir.Global, ir.FreeVar, ir.Const)):
            return definition.value
        hierarchy = self._hierarchy(value)
        if hierarchy is not None:
            return hierarchy
        group = self._group(value)
        if group is not None:
            return group
        try:
            return self.func_ir.infer_constant(value)
        except Exception as exc:
            raise NonConstantGroupArgumentError(value.name) from exc

    def _try_constant(self, value: Any) -> tuple[bool, Any]:
        """Probe for a constant without requesting dispatcher specialization.

        Literal arguments and arguments known to be ``None`` resolve directly.
        Other inputs use ``_constant``; its specialization requests and group
        resolution failures become an unresolved result. This broad constant
        probe may use Numba constant inference. Use ``_try_static_scalar``
        when an operation needs evidence that a scalar was explicitly supplied
        as a compile-time value.

        Parameters
        ----------
        value : ir.Var or object
            Argument to inspect, usually an IR variable or an already-resolved
            Python default. Unknown runtime values leave this probe unresolved.

        Returns
        -------
        resolved : bool
            Whether a constant was recovered without another compiler attempt.
        value : object
            Recovered value, or ``None`` on failure. Check ``resolved`` to
            distinguish failure from a known ``None``.
        """
        if isinstance(value, ir.Var):
            definition = self._definition(value)
            if isinstance(definition, ir.Arg):
                argtype = self.state.args[definition.index]
                if isinstance(argtype, _numba_types.Literal):
                    return (True, argtype.literal_value)
                if isinstance(argtype, _numba_types.NoneType) or (
                    isinstance(argtype, _numba_types.Omitted)
                    and argtype.value is None
                ):
                    return (True, None)
                return (False, None)
        try:
            return (True, self._constant(value))
        except (ForceLiteralArg, GroupRewriteError):
            return (False, None)

    def _try_static_scalar(
        self,
        value: Any,
    ) -> tuple[bool, Any]:
        """Resolve an explicitly static scalar with its known numeric width.

        Delegate to the shared provenance walker with this function's recorded
        definitions and argument types. Constants and literal arguments must
        agree across all paths; runtime expressions are not evaluated to turn
        them into constants. The result is ``(resolved, value)``; an
        unresolved value is ``(False, None)``. No specialization is requested.
        """

        return try_resolve_static_scalar(
            value,
            definitions=self._all_definitions,
            argument_type=lambda index: (
                self.state.args[index]
                if 0 <= index < len(self.state.args)
                else None
            ),
        )

    def _try_static_scalar_provenance(self, value: Any) -> tuple[bool, Any]:
        """Resolve a static scalar and retain its dtype provenance.

        Use the same conservative traversal as ``_try_static_scalar``,
        retaining the provenance record so callers can distinguish an untyped
        Python literal from an explicitly typed value. Return ``(True,
        provenance)`` when every path agrees, otherwise ``(False, None)``. A
        known ``None`` is contained in a provenance record and is distinct
        from failure.
        """

        return try_resolve_static_scalar_provenance(
            value,
            definitions=self._all_definitions,
            argument_type=lambda index: (
                self.state.args[index]
                if 0 <= index < len(self.state.args)
                else None
            ),
        )

    def _bind(self, function: Any, call: ir.Expr) -> inspect.BoundArguments:
        """Bind IR arguments to the API signature and apply its defaults.

        Explicit arguments remain IR values; defaults are ordinary Python
        values. This lets operation planners use parameter names independently
        of the caller's positional or keyword spelling. Argument unpacking is
        rejected because its contents are not available to this binding step.

        Parameters
        ----------
        function : callable
            Recognized constructor or operation whose signature defines the
            call.
        call : ir.Expr
            Call expression to validate and bind.

        Returns
        -------
        inspect.BoundArguments
            Named arguments with Python defaults applied, ready for planning.

        Raises
        ------
        GroupRewriteError
            The call uses ``*args`` or ``**kwargs``.
        TypeError
            Arguments do not match the signature. The diagnostic includes the
            function name and the IR call's source location.
        """
        if call.vararg is not None or call.varkwarg is not None:
            raise GroupRewriteError(
                "cuda.coop.numba_mlir group calls do not support *args/**kwargs"
            )
        try:
            bound = inspect.signature(function).bind(
                *call.args, **dict(call.kws)
            )
        except TypeError as exc:
            raise TypeError(
                f"{function.__name__}() {exc}\n{call.loc.strformat()}"
            ) from None
        bound.apply_defaults()
        return bound

    def _validate_common_selector(
        self,
        operation: str,
        parameter: str,
        value: Any,
        allowed: frozenset[str],
        *,
        allow_none: bool = False,
    ) -> Any:
        """Validate selector tokens for a rewritten common-API call.

        Rewriting recognizes a public operation by identity and does not
        execute its host wrapper. Repeat the selector validation here so
        compiled calls retain that API's accepted strings and diagnostics. The
        value must be constant and independent of pending literal unrolling.

        Parameters
        ----------
        operation : str
            Public operation name used in diagnostics.
        parameter : str
            Selector parameter name used in diagnostics.
        value : object
            IR argument or an already-resolved default.
        allowed : frozenset of str
            Accepted normalized selector tokens.
        allow_none : bool, optional
            Whether ``None`` is an accepted omission value.

        Returns
        -------
        str or None
            Token with whitespace stripped, lowercased, and hyphens replaced
            by underscores, or an explicitly allowed ``None``.

        Raises
        ------
        TypeError
            The value is not a string, or is an enum instead of a string
            token.
        InvalidGroupSelectorError
            The normalized token is not supported.
        ForceLiteralArg
            Resolving the selector requires a literal function argument.
        GroupRewriteError
            The selector cannot be resolved or depends on literal unrolling.
        """
        self._reject_literal_unroll_value(value, f"{operation} {parameter}")
        token = self._constant(value)
        if token is None and allow_none:
            return None
        if not isinstance(token, str) or isinstance(token, Enum):
            raise TypeError(
                f"cuda.coop.{operation} {parameter} must be a string"
            )
        token = token.strip().lower().replace("-", "_")
        if token not in allowed:
            choices = ", ".join(sorted(allowed))
            raise InvalidGroupSelectorError(operation, parameter, choices)
        return token

    def _hierarchy(self, value: Any) -> ThreadHierarchy | None:
        """Reconstruct a hierarchy descriptor for host-side planning.

        Accept an existing ``ThreadHierarchy``, or follow an IR variable
        through aliases, casts, constants, and the recognized constructor.
        Validate a constructor's signature and cache the resulting host
        descriptor by variable name. Return ``None`` for an unrecognized
        definition; descriptor arguments requiring a group or hierarchy are
        diagnosed by their caller.
        """

        if isinstance(value, ThreadHierarchy):
            return value
        if not isinstance(value, ir.Var):
            return None
        cached = self._hierarchy_cache.get(value.name)
        if cached is not None:
            return cached
        definition = self._definition(value)
        if isinstance(definition, ir.Var):
            return self._hierarchy(definition)
        if isinstance(definition, ir.Expr) and definition.op == "cast":
            return self._hierarchy(definition.value)
        if isinstance(definition, (ir.Global, ir.FreeVar, ir.Const)):
            if isinstance(definition.value, ThreadHierarchy):
                return definition.value
            return None
        if not isinstance(definition, ir.Expr) or definition.op != "call":
            return None
        function = self._callable(definition.func)
        if function is not ThreadHierarchy:
            return None
        self._bind(ThreadHierarchy, definition)
        hierarchy = ThreadHierarchy()
        self._hierarchy_cache[value.name] = hierarchy
        return hierarchy

    def _group(self, value: object) -> ThreadGroup | None:
        """Reconstruct a group descriptor from a supported IR definition.

        Follow aliases and casts, accept existing host descriptors, and
        interpret registered constructors and ``group_by`` calls on recognized
        parents. Constructor identity matters; matching a callable's name is
        insufficient. Arguments are resolved through ``_constant``, which may
        request literal specialization. Common-API constructors retain
        ``common_root`` provenance so later validation applies the common API
        contract.

        Cache newly constructed descriptors by variable name for this planner.
        This describes the requested group; launch-dependent resolution
        belongs to ``_resolve_group``. Unlike marker detection, this routine
        requires a single resolvable definition and does not merge phi inputs.

        Parameters
        ----------
        value : object
            Existing ``ThreadGroup`` or IR variable expected to name one.
            Other inputs are allowed as probes and return ``None``; the
            caller decides whether an unrecognized group is an error.

        Returns
        -------
        ThreadGroup or None
            Reconstructed or cached descriptor, or ``None`` if the definition
            is not a recognized group expression.

        Raises
        ------
        ForceLiteralArg
            A constructor or subgroup argument needs literal specialization.
        TypeError
            A recognized constructor call does not match its Python signature.
        GroupRewriteError
            A group constructor or subgroup call uses unsupported syntax,
            or its arguments cannot be resolved as compile-time values.
        """

        if isinstance(value, ThreadGroup):
            return value
        if not isinstance(value, ir.Var):
            return None
        cached = self._group_cache.get(value.name)
        if cached is not None:
            return cached
        definition = self._definition(value)
        if isinstance(definition, ir.Var):
            return self._group(definition)
        if isinstance(definition, (ir.Global, ir.FreeVar, ir.Const)):
            if isinstance(definition.value, ThreadGroup):
                return definition.value
            return None
        if isinstance(definition, ir.Expr) and definition.op == "cast":
            return self._group(definition.value)
        if not isinstance(definition, ir.Expr) or definition.op != "call":
            return None
        function = self._callable(definition.func)
        if function in _GROUP_CONSTRUCTORS:
            bound = self._bind(function, definition)
            args = []
            kwargs = {}
            parameters = inspect.signature(function).parameters
            for name, parameter in parameters.items():
                argument = self._constant(bound.arguments[name])
                if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
                    args.append(argument)
                elif parameter.kind is inspect.Parameter.KEYWORD_ONLY:
                    kwargs[name] = argument
            group = _GROUP_CONSTRUCTORS[function](*args, **kwargs)
            if function in _COMMON_GROUP_CONSTRUCTORS:
                assert group.hierarchy is not None
                group = group.with_hierarchy(
                    group.hierarchy, source="common_root"
                )
            self._group_cache[value.name] = group
            return group
        function_definition = self._definition(definition.func)
        if (
            isinstance(function_definition, ir.Expr)
            and function_definition.op == "getattr"
            and (function_definition.attr == "group_by")
        ):
            parent = self._group(function_definition.value)
            if parent is None:
                return None
            if definition.vararg is not None or definition.varkwarg is not None:
                raise GroupRewriteError(
                    "ThreadGroup.group_by does not support *args/**kwargs"
                )
            raw_args = tuple(definition.args)
            raw_kwargs = dict(definition.kws)
            unknown_kwargs = set(raw_kwargs) - {"count", "exhaustive"}
            if unknown_kwargs:
                names = ", ".join(sorted(unknown_kwargs))
                raise GroupRewriteError(
                    f"ThreadGroup.group_by got unexpected keyword(s): {names}"
                )
            if len(raw_args) > 1:
                raise GroupRewriteError(
                    "ThreadGroup.group_by accepts one positional count argument"
                )
            if raw_args and "count" in raw_kwargs:
                raise GroupRewriteError(
                    "ThreadGroup.group_by received count more than once"
                )
            if raw_args:
                count_arg = raw_args[0]
            elif "count" in raw_kwargs:
                count_arg = raw_kwargs["count"]
            else:
                raise GroupRewriteError("ThreadGroup.group_by requires count")
            self._reject_literal_unroll_value(count_arg, "group_by count")
            exhaustive_arg = raw_kwargs.get("exhaustive", True)
            self._reject_literal_unroll_value(
                exhaustive_arg, "group_by exhaustive"
            )
            count_value = self._constant(count_arg)
            exhaustive = self._constant(exhaustive_arg)
            group = parent.group_by(count_value, exhaustive=exhaustive)
            self._group_cache[value.name] = group
            return group
        return None

    def _resolve_group(
        self,
        group: ThreadGroup,
        *,
        feature: str,
        through_level: str | None = None,
    ) -> ThreadGroup:
        """Resolve a group request against the configured launch.

        Descriptor reconstruction records what the kernel requested.
        Resolution fills the launch-dependent hierarchy and verifies that the
        requested composition can be represented. Preserve common-API origin
        information so operation validation can still enforce that contract
        afterward.

        Parameters
        ----------
        group : ThreadGroup
            Reconstructed or supplied group description.
        feature : str
            Operation or group-method name prepended to unsupported
            diagnostics.
        through_level : str, optional
            Hierarchy level through which resolution is required, for example
            when a query needs enclosing group counts.

        Returns
        -------
        ThreadGroup
            Supported group with launch-dependent topology resolved.

        Raises
        ------
        NotImplementedError
            Shared group resolution reports an unsupported request. The
            message identifies the Numba operation that needed the group.
        """

        resolution = resolve_thread_group(
            group, self.launch, through_level=through_level
        )
        try:
            resolved = resolution.require_supported()
        except NotImplementedError as exc:
            raise NotImplementedError(
                f"cuda.coop.numba_mlir.{feature} {exc}"
            ) from exc
        if group.source == "common_root":
            assert resolved.hierarchy is not None
            resolved = resolved.with_hierarchy(
                resolved.hierarchy, source="common_root"
            )
        return resolved

    def _is_none(self, value: Any) -> bool:
        """Check whether an optional operation argument is explicitly omitted.

        Operation planners call this through ``GroupPlanningContext.is_none``
        for arguments such as ``temp_storage``. ``value`` is an IR operand or
        Python default. Return true only when constant resolution establishes
        ``None``; an unknown runtime value is not treated as omission. This
        probe does not request another compiler specialization.
        """

        resolved, constant = self._try_constant(value)
        return resolved and constant is None

    @staticmethod
    def _merge_array_states(states: tuple[bool | None, ...]) -> bool | None:
        """Combine the array checks for all possible sources of an operand.

        The array walkers call this when a branch or loop gives a variable more
        than one possible value. An operand can be treated as an array only if
        no concrete source contradicts that choice. A loop that keeps a payload
        created before the loop is valid: revisiting the loop variable adds no
        new evidence, but must not discard the known constructor either.

        Parameters
        ----------
        states : tuple of bool or None
            Results for the alternative sources. ``True`` means a supported
            array constructor was found; ``False`` means that source is
            unsupported; ``None`` means traversal revisited a variable on the
            current recursion path, before finding a constructor there.

        Returns
        -------
        bool or None
            ``False`` if any source is unsupported or there are no definitions.
            Otherwise ``True`` if a constructor was found, or ``None`` if every
            source only led back into the same cycle. Thus ``(True, None)`` is
            accepted, while ``(True, False)`` is rejected.
        """

        if not states or any(state is False for state in states):
            return False
        if any(state is True for state in states):
            return True
        return None

    def _result_source(self, definition: Any, index: int | None = None):
        """Find the argument policy for a registered group call's result.

        For example, planning ``store(group, output, exchange(group,
        values))`` needs the Exchange result's element type and per-thread
        item count before either public call has been replaced.

        Bind public arguments before resolving results so a static selector
        can choose the result layout. Return the selected
        ``GroupResultSource`` and bound arguments, or ``None`` for an
        unrelated call or invalid result selection. Without ``index``, only
        a single-result operation qualifies. With ``index``, only a
        multiple-result operation qualifies: indexing a single array result
        selects an element, not a tuple result.

        Dtype, array-origin, and extent queries share this policy. It lets
        later group calls consume the result before provider rewriting and
        ordinary typing.

        Parameters
        ----------
        definition : object
            Right-hand side of an IR assignment being queried. Only a
            registered public group-operation call provides a result
            policy.
        index : int or None, optional
            Tuple-result position, including supported negative indices.
            ``None`` requests the policy for a directly returned value;
            it does not select an array element.
        """

        if not isinstance(definition, ir.Expr) or definition.op != "call":
            return None
        operation = _group_operation_name(self._callable(definition.func))
        registration = None if operation is None else group_primitive(operation)
        if registration is None:
            return None
        bound = self._bind(self._callable(definition.func), definition)
        results = (
            registration.results
            if registration.result_resolver is None
            else registration.result_resolver(self.context, bound)
        )
        if index is None:
            if len(results) != 1:
                return None
            result = results[0]
        else:
            if len(results) == 1:
                # A single-result primitive returns its value directly, so an
                # integer subscript selects an element of that value (a
                # scalar), not a tuple item.
                return None
            if not -len(results) <= index < len(results):
                return None
            result = results[index]
        return result, bound

    def _is_array_tuple_item(
        self,
        value: Any,
        index: int,
        *,
        seen: set[str],
        thread_data_only: bool = False,
    ) -> bool | None:
        """Check whether a selected tuple element holds an array payload.

        An operation can receive a payload taken from a tuple, for example
        ``payloads[0]``. ``_is_array_definition`` calls this helper for such
        indexing expressions. Follow every possible source of the tuple so a
        branch cannot hide an unsupported value in the selected position.
        Track the tuple and index together: revisiting ``payloads[0]`` is a
        cycle, while inspecting another element is a separate question.

        Parameters
        ----------
        value : ir.Var or object
            IR variable containing the tuple. Other values return ``False``.
        index : int
            Tuple position to inspect, with Python's zero-based and negative
            indexing rules. This is an element index, not a byte offset.
        seen : set of str
            Variable names and ``"name[index]"`` keys already visited on this
            recursion path. The selected key is added in place; each possible
            definition receives a copy so one branch cannot hide another.
        thread_data_only : bool, optional
            Accept only ``ThreadData`` constructors when true. Otherwise,
            recognized CUDA local-array constructors are also accepted.

        Returns
        -------
        bool or None
            ``True`` when at least one supported constructor is found and no
            source is rejected; ``False`` for a missing or unsupported source;
            ``None`` if traversal only revisits variables in a cycle. The
            caller combines this result with the operand's other sources.
        """

        if not isinstance(value, ir.Var):
            return False
        seen_key = f"{value.name}[{index}]"
        if seen_key in seen:
            return None
        seen.add(seen_key)
        return self._merge_array_states(
            tuple(
                self._is_array_tuple_item_definition(
                    definition,
                    index,
                    seen=set(seen),
                    thread_data_only=thread_data_only,
                )
                for definition in self._all_definitions(value)
            )
        )

    def _is_array_tuple_item_definition(
        self,
        definition: Any,
        index: int,
        *,
        seen: set[str],
        thread_data_only: bool,
    ) -> bool | None:
        """Check one tuple source for the selected array payload.

        ``_is_array_tuple_item`` calls this once per recorded tuple definition.
        Follow tuple copies, casts, and the IR used to unpack an iterator; each
        preserves which position the caller selected. A phi expression chooses
        between values arriving from different branches or loop iterations, so
        its inputs must all be checked. When a ``build_tuple`` expression is
        reached, classify the selected element with ``_is_array_value``.

        For a registered operation with multiple results, an extent resolver
        establishes an array result without an array-shaped input. Otherwise,
        follow the selected result's array-source argument. A single-result
        array call is not a tuple producer; indexing it selects a scalar.

        Parameters
        ----------
        definition : object
            One right-hand-side IR value or expression that supplies the tuple.
        index : int
            Zero-based tuple position; negative indices count from the end.
        seen : set of str
            Variable names and tuple-position keys visited on this recursion
            path. Independent branches receive copies to avoid skipping work.
        thread_data_only : bool
            Require ``ThreadData`` when following an input array source.
            False also permits CUDA local arrays. A registered extent resolver
            establishes an array result without either input constructor.

        Returns
        -------
        bool or None
            The selected element's array classification. ``False`` means an
            unsupported source or invalid index. ``None`` means a recursive
            path supplied no array evidence; ``True`` means a supported
            array source was found without a conflicting source.
        """

        if isinstance(definition, ir.Var):
            return self._is_array_tuple_item(
                definition, index, seen=seen, thread_data_only=thread_data_only
            )
        if not isinstance(definition, ir.Expr):
            return False
        if definition.op in {"cast", "exhaust_iter"}:
            return self._is_array_tuple_item(
                definition.value,
                index,
                seen=seen,
                thread_data_only=thread_data_only,
            )
        if definition.op == "phi":
            incoming_values = getattr(definition, "incoming_values", ())
            return self._merge_array_states(
                tuple(
                    self._is_array_tuple_item(
                        incoming,
                        index,
                        seen=set(seen),
                        thread_data_only=thread_data_only,
                    )
                    for incoming in incoming_values
                )
            )
        if definition.op == "build_tuple":
            items = tuple(getattr(definition, "items", ()))
            if not -len(items) <= index < len(items):
                return False
            return self._is_array_value(
                items[index], seen=set(seen), thread_data_only=thread_data_only
            )
        if definition.op != "call":
            return False
        resolved = self._result_source(definition, index)
        if resolved is None:
            return False
        result, bound = resolved
        if result.extent_resolver is not None:
            return True
        if result.array_parameter is None:
            return False
        return self._is_array_value(
            bound.arguments[result.array_parameter],
            seen=seen,
            thread_data_only=thread_data_only,
        )

    def _is_array_value(
        self,
        value: object,
        *,
        seen: set[str] | None = None,
        thread_data_only: bool = False,
    ) -> bool | None:
        """Determine whether a value comes from a supported per-thread array.

        A ``store()`` operation can consume a scalar or a per-thread array,
        whereas ``load()`` needs an array destination. Their validation uses
        this result to select the appropriate shape and argument rules. A
        variable may instead name a scalar, a kernel argument, an unrelated
        object, or a loop-carried value whose origin is not yet known. A false
        result only rules out a recognized array origin; further validation
        must establish whether the value is a supported scalar.

        Trace aliases, casts, phi inputs, and constant tuple projections to
        ``ThreadData`` or qualified local-array constructors. The three-state
        result lets a loop backedge coexist with a known constructor: a cycle
        alone is unresolved, a constructor plus cycles is accepted, and any
        concrete unrecognized path rejects the value. This classifies the
        payload form only; dtype and extent are inferred separately.

        Parameters
        ----------
        value : object
            IR variable to classify, or any other value to probe.
            Non-variables return ``False``.
        seen : set of str, optional
            Recursion-path variable names and tuple-projection keys. The
            current name is added in place; branches receive separate copies.
        thread_data_only : bool, optional
            If true, accept only common or qualified ``ThreadData``
            constructors. Otherwise, also accept the recognized CUDA
            local-array constructor.

        Returns
        -------
        bool or None
            ``True`` if at least one constructor is found and no path is
            rejected; ``False`` for an unsupported definition, missing
            definitions, or a non-variable; ``None`` when only cyclic paths
            remain. The operand validation wrappers turn that last state into
            a provenance diagnostic.
        """

        if not isinstance(value, ir.Var):
            return False
        if seen is None:
            seen = set()
        if value.name in seen:
            return None
        seen.add(value.name)
        return self._merge_array_states(
            tuple(
                self._is_array_definition(
                    definition,
                    seen=set(seen),
                    thread_data_only=thread_data_only,
                )
                for definition in self._all_definitions(value)
            )
        )

    def _is_array_definition(
        self, definition: Any, *, seen: set[str], thread_data_only: bool
    ) -> bool | None:
        """Check whether one assignment supplies a supported array payload.

        ``_is_array_value`` calls this for each possible source of an operation
        operand. Tracing the assignment back to its constructor tells operation
        planning whether it has a per-thread array or must apply scalar rules.
        For a value selected from a tuple, resolve the index before examining
        that tuple element. Branch and loop merges inspect all alternatives;
        an unsupported source cannot be hidden by another valid source.

        A generated result marker follows its prototype's origin. For a
        registered single-result operation, an extent resolver establishes an
        array result without an array-shaped input. Otherwise, follow the
        declared array-source argument and retain the ThreadData restriction.

        Parameters
        ----------
        definition : object
            A recorded right-hand-side IR value or expression, such as a
            constructor call, variable alias, cast, phi, or tuple lookup.
        seen : set of str
            Variable names and tuple-position keys already visited on the
            current recursion path. Each branch receives its own copy.
        thread_data_only : bool
            Require common or qualified ``ThreadData`` when tracing an input
            array. False also accepts CUDA local arrays. An extent resolver
            establishes an array result without an input array.

        Returns
        -------
        bool or None
            ``True`` for a supported array source or alternatives that agree;
            ``False`` for an unsupported source or unresolved tuple index;
            ``None`` when following aliases or loop inputs only leads back to
            a variable already being examined. Operand validation reports that
            unresolved cycle if no other path supplies an array source.
        """

        if isinstance(definition, ir.Var):
            return self._is_array_value(
                definition, seen=seen, thread_data_only=thread_data_only
            )
        if not isinstance(definition, ir.Expr):
            return False
        if definition.op == "cast":
            return self._is_array_value(
                definition.value, seen=seen, thread_data_only=thread_data_only
            )
        if definition.op == "phi":
            incoming_values = getattr(definition, "incoming_values", ())
            return self._merge_array_states(
                tuple(
                    self._is_array_value(
                        incoming,
                        seen=set(seen),
                        thread_data_only=thread_data_only,
                    )
                    for incoming in incoming_values
                )
            )
        if definition.op in {"getitem", "static_getitem"}:
            index = getattr(definition, "index", None)
            if isinstance(index, ir.Var):
                resolved, index = self._try_constant(index)
                if not resolved:
                    return False
            if isinstance(index, Integral) and (not isinstance(index, bool)):
                return self._is_array_tuple_item(
                    definition.value,
                    int(index),
                    seen=set(seen),
                    thread_data_only=thread_data_only,
                )
            return False
        if definition.op != "call":
            return False
        function = self._callable(definition.func)
        if function in {ThreadData, _common_api.ThreadData}:
            return True
        if function is _typed_group_payload_like:
            return self._is_array_value(
                definition.args[0], seen=seen, thread_data_only=thread_data_only
            )
        if function is _cuda_local_array:
            return not thread_data_only
        resolved = self._result_source(definition)
        if resolved is None:
            return False
        result, bound = resolved
        if result.extent_resolver is not None:
            return True
        if result.array_parameter is None:
            return False
        return self._is_array_value(
            bound.arguments[result.array_parameter],
            seen=seen,
            thread_data_only=thread_data_only,
        )

    @staticmethod
    def _new_var(scope: Any, loc: ir.Loc, stem: str) -> ir.Var:
        """Name a temporary needed by a generated provider call.

        ``_rewritten_call`` and operation planners use these variables for the
        selected callable and its arguments. The shared counter avoids name
        collisions when several operations are rewritten in the same function.

        Parameters
        ----------
        scope : ir.Scope
            Lexical scope of the original operation's result variable.
        loc : ir.Loc
            Source location to retain for compiler diagnostics.
        stem : str
            Readable name fragment, such as ``"factory"`` or ``"arg0"``,
            identifying the temporary's purpose in an IR dump.

        Returns
        -------
        ir.Var
            A fresh variable. Its defining assignment is created by the caller.
        """

        return ir.Var(
            scope, f"__cuda_coop_group_{stem}_{next(_NAME_COUNTER)}__", loc
        )

    def _value_var(
        self,
        statements: list[Any],
        *,
        scope: Any,
        loc: ir.Loc,
        stem: str,
        value: Any,
    ) -> ir.Var:
        """Supply an IR variable for one argument of a generated provider call.

        Numba call expressions refer to variables, but planning also produces
        Python values such as an item count or a provider's metadata object.
        ``_rewritten_call`` and operation planners use this helper to bind those
        values before emitting a call. Existing variables need no assignment.
        For a Python value, append a constant or global assignment to the
        caller's pending statement list. ``run()`` installs that list only after
        all cooperative operations pass validation.

        Parameters
        ----------
        statements : list of IR statements
            Pending replacement statements, in execution order. A new
            assignment is appended here when ``value`` needs a variable.
        scope : ir.Scope
            Scope for a new variable, taken from the original call's result.
        loc : ir.Loc
            Original source location for the variable and assignment.
        stem : str
            Name fragment describing the argument, such as ``"arg0"`` or a
            keyword name. A counter makes the complete name distinct.
        value : object
            Existing IR variable or Python value to pass. ``None``, booleans,
            integers, floats, strings, and tuples use ``ir.Const``; other
            objects use ``ir.Global`` so later compiler phases can recover
            the object.

        Returns
        -------
        ir.Var
            The unchanged input variable or the newly assigned temporary.
            No function block is modified by this helper.
        """

        if isinstance(value, ir.Var):
            return value
        result = self._new_var(scope, loc, stem)
        if value is None or isinstance(value, (bool, int, float, str, tuple)):
            rhs = ir.Const(value, loc)
        else:
            rhs = ir.Global(result.name, value, loc)
        statements.append(ir.Assign(rhs, result, loc))
        return result

    def _rewritten_call(
        self,
        inst: ir.Assign,
        *,
        factory: Any,
        args: list[Any],
        kwargs: dict[str, Any],
        return_alias: ir.Var | tuple[ir.Var, ...] | None = None,
        common_root_operation: str | None = None,
    ) -> list[Any]:
        """Build replacement IR calling the selected provider.

        Bind the host factory and any non-variable arguments to fresh
        temporaries, then emit a call at the original source location.
        Existing argument variables are reused, and the replacement preserves
        the original result target. Nothing is installed in a block until
        ``run()`` accepts the whole rewrite.

        Parameters
        ----------
        inst : ir.Assign
            Original operation assignment whose target, scope, and location
            are retained by the replacement.
        factory : callable
            Selected private provider callable for the next rewriting phase.
        args : list of object
            Positional IR variables or Python values supplied to the provider.
        kwargs : dict of str to object
            Named IR variables or Python values supplied to the provider.
        return_alias : ir.Var or tuple of ir.Var, optional
            Public result value or values to assign to the original target
            after the provider runs. When supplied, discard the provider
            return value and preserve the public result ownership contract.
        common_root_operation : str, optional
            Public operation identity forwarded to provider validation. It
            keeps common-API rules available after the public wrapper has been
            removed.

        Returns
        -------
        list
            Ordered assignments defining the callable and arguments and making
            the replacement call, followed by any requested result aliases.
        """

        statements: list[Any] = []
        scope = inst.target.scope
        loc = inst.loc
        if common_root_operation is not None:
            kwargs = dict(kwargs)
            kwargs.setdefault("_common_root_operation", common_root_operation)
        function_var = self._new_var(scope, loc, "factory")
        statements.append(
            ir.Assign(
                ir.Global(function_var.name, factory, loc), function_var, loc
            )
        )
        rewritten_args = [
            self._value_var(
                statements, scope=scope, loc=loc, stem=f"arg{idx}", value=value
            )
            for idx, value in enumerate(args)
        ]
        rewritten_kwargs = tuple(
            (
                (
                    name,
                    self._value_var(
                        statements, scope=scope, loc=loc, stem=name, value=value
                    ),
                )
                for name, value in kwargs.items()
            )
        )
        call_target = (
            inst.target
            if return_alias is None
            else self._new_var(scope, loc, "ignored_result")
        )
        statements.append(
            ir.Assign(
                ir.Expr.call(
                    function_var, rewritten_args, rewritten_kwargs, loc
                ),
                call_target,
                loc,
            )
        )
        if isinstance(return_alias, tuple):
            statements.append(
                ir.Assign(
                    ir.Expr.build_tuple(list(return_alias), loc),
                    inst.target,
                    loc,
                )
            )
        elif return_alias is not None:
            statements.append(ir.Assign(return_alias, inst.target, loc))
        return statements

    def _array_operand_state(self, operation: str, value: Any) -> bool:
        """Tell an operation planner whether its operand is a per-thread array.

        ``GroupPlanningContext.is_array`` calls this to validate an operand.
        For example, Store needs to choose between a scalar value and an array
        of values per thread, while Load requires an array destination. Trace
        the operand to its constructor before making that choice. If traversal
        only finds a cycle, report that the payload's source could not be
        established rather than treating it as a scalar.

        Parameters
        ----------
        operation : str
            Public operation name, such as ``"load"`` or ``"store"``, used
            to identify the call if source tracing fails.
        value : object
            Operand's IR variable, or another value to probe. Recognized
            sources are ``ThreadData`` and CUDA local-array constructors.

        Returns
        -------
        bool
            Whether the operand has a supported array source on all concrete
            paths. ``False`` leaves scalar or invalid-value checks to the
            operation planner; it does not establish that the value is scalar.

        Raises
        ------
        CyclicArrayProvenanceError
            All traced sources only revisit variables already being examined,
            so none establishes how the payload was created.
        """

        state = self._is_array_value(value)
        if state is None:
            raise CyclicArrayProvenanceError(operation)
        return state

    def _thread_data_operand_state(
        self, operation: str, parameter: str, value: Any
    ) -> bool:
        """Check the payload constructor required by a common-API parameter.

        ``GroupPlanningContext.is_thread_data`` calls this when an operation
        needs a ``ThreadData`` payload. For example, common ``load`` requires
        ``ThreadData`` for its output even though the qualified backend also
        accepts CUDA local arrays. Follow copies and control-flow alternatives
        so that the same rule applies when the constructor is elsewhere in the
        kernel.

        Parameters
        ----------
        operation : str
            Public operation name used in a source-tracing diagnostic.
        parameter : str
            Public parameter name, such as ``"output"`` or ``"value"``,
            identifying which operand could not be traced.
        value : object
            IR operand to trace to a common or qualified ``ThreadData`` call.

        Returns
        -------
        bool
            ``True`` for a recognized ``ThreadData`` source on all concrete
            paths. ``False`` lets the caller report its parameter-specific
            error for an unsupported source.

        Raises
        ------
        GroupRewriteError
            The search only finds cyclic references and cannot establish the
            payload's constructor.
        """

        state = self._is_array_value(value, thread_data_only=True)
        if state is None:
            raise GroupRewriteError(
                f"cuda.coop.{operation} could not "
                f"resolve {parameter} payload provenance"
            )
        return state

    def _array_extent(
        self, value: Any, *, seen: set[str] | None = None
    ) -> int | None:
        """Recover how many elements one thread holds in an array payload.

        This element count is the payload's *extent*. For example,
        ``ThreadData(items_per_thread=4)`` has extent four, independent of its
        element dtype or the number of threads in the block. Operation planners
        call ``GroupPlanningContext.array_extent`` to obtain this count before
        specializing a CUB provider whose item count is a template argument.

        Follow the assignments that can supply ``value`` through copies, casts,
        branch or loop merges, and constant tuple indices. Read the count from
        constructors, generated payload markers, or registered result policies.
        All known counts must agree: one provider cannot use different array
        sizes on different paths through the kernel. Paths with no known count
        are ignored here; the separate array-kind check must establish that the
        operand has a supported array source.

        Parameters
        ----------
        value : ir.Var or object
            Payload variable whose element count is needed. Non-variables
            provide no count and return ``None``.
        seen : set of str, optional
            Variable names and tuple-position keys already visited on this
            recursion path. Add the current name in place and copy the set
            when following alternative assignments to avoid infinite loops.

        Returns
        -------
        int or None
            The single known number of elements per thread, or ``None`` if no
            count is found. Constructor counts are integral and exclude
            booleans; result-policy callbacks supply their own counts. The
            operation's later shape validation checks positivity.

        Raises
        ------
        GroupRewriteError
            Known counts disagree, or a constructor's dimension depends on
            literal unrolling that has not run yet.
        ForceLiteralArg
            A constructor dimension needs a kernel argument specialized to a
            compile-time value.
        """

        if not isinstance(value, ir.Var):
            return None
        if seen is None:
            seen = set()
        if value.name in seen:
            return None
        seen.add(value.name)
        extents: set[int] = set()
        for definition in self._all_definitions(value):
            extent = self._array_extent_definition(definition, seen=set(seen))
            if extent is not None:
                extents.add(extent)
        if len(extents) > 1:
            raise InconsistentArrayExtentError()
        return next(iter(extents), None)

    def _array_extent_tuple_item(
        self, value: Any, index: int, *, seen: set[str]
    ) -> int | None:
        """Find the per-thread element count of a payload selected from a tuple.

        ``_array_extent_definition`` calls this for indexing such as
        ``payloads[0]``. The tuple may have been copied or assigned on several
        branches, so inspect each possible assignment before selecting one
        count for provider specialization. The tuple length is unrelated to
        the payload's extent: a tuple of two four-element payloads has length
        two, while either selected payload has extent four.

        Parameters
        ----------
        value : ir.Var or object
            IR variable containing the tuple. Other values have no known
            payload count and return ``None``.
        index : int
            Payload position within the tuple, using Python's zero-based and
            negative indexing rules.
        seen : set of str
            Variable names and tuple-position keys visited on this recursion
            path. Add ``"name[index]"`` in place and copy the set for each
            alternative assignment so loops terminate without hiding branches.

        Returns
        -------
        int or None
            The unique known element count per thread for the selected payload.
            Unknown sources and repeated visits contribute no count; ``None``
            means none was found. Array-kind validation is a separate check.

        Raises
        ------
        InconsistentTupleExtentError
            Possible definitions of this tuple give the selected payload
            different known element counts.
        """

        if not isinstance(value, ir.Var):
            return None
        seen_key = f"{value.name}[{index}]"
        if seen_key in seen:
            return None
        seen.add(seen_key)
        extents = {
            extent
            for definition in self._all_definitions(value)
            if (
                extent := self._array_extent_tuple_item_definition(
                    definition, index, seen=set(seen)
                )
            )
            is not None
        }
        if len(extents) > 1:
            raise InconsistentTupleExtentError()
        return next(iter(extents), None)

    def _array_extent_tuple_item_definition(
        self, definition: Any, index: int, *, seen: set[str]
    ) -> int | None:
        """Read one tuple assignment for the selected payload's element count.

        ``_array_extent_tuple_item`` calls this for each possible tuple source.
        A *reaching definition* is an assignment whose value can arrive at the
        operation along a branch or loop path. The caller conservatively checks
        all recorded sources, without proving each reaches this use. Trace back
        to the payload constructor because CUB provider specialization needs a
        fixed number of elements per thread. That number is the payload's
        *extent*, not the tuple length or its size in bytes.

        Copies, casts, and iterator-unpacking IR preserve the selected position.
        A phi expression combines values from different control-flow paths,
        so known counts on its inputs must agree. At a tuple construction,
        ``_array_extent`` examines the selected payload itself.

        Registered multiple-result calls use the selected result's extent
        resolver when present. Otherwise, follow its array-source argument
        or use one scalar item when the policy has no array source.

        Parameters
        ----------
        definition : object
            One right-hand-side IR value or expression that supplies the tuple.
        index : int
            Payload position in the tuple; negative indices count from the end.
        seen : set of str
            Variable names and tuple-position keys already visited while
            tracing this path. Separate phi inputs receive copies of the set.

        Returns
        -------
        int or None
            Known number of elements per thread in the selected payload.
            ``None`` means the source or index is unsupported, its count is
            unknown, or following it only revisits an existing recursion path.
            The caller checks payload kind separately.

        Raises
        ------
        InconsistentLoopTupleExtentError
            A branch or loop merge supplies different known counts for the
            selected tuple position.
        """

        if isinstance(definition, ir.Var):
            return self._array_extent_tuple_item(definition, index, seen=seen)
        if not isinstance(definition, ir.Expr):
            return None
        if definition.op in {"cast", "exhaust_iter"}:
            return self._array_extent_tuple_item(
                definition.value, index, seen=seen
            )
        if definition.op == "phi":
            extents = {
                extent
                for incoming in getattr(definition, "incoming_values", ())
                if (
                    extent := self._array_extent_tuple_item(
                        incoming, index, seen=set(seen)
                    )
                )
                is not None
            }
            if len(extents) > 1:
                raise InconsistentLoopTupleExtentError()
            return next(iter(extents), None)
        if definition.op == "build_tuple":
            items = tuple(getattr(definition, "items", ()))
            if not -len(items) <= index < len(items):
                return None
            return self._array_extent(items[index], seen=set(seen))
        if definition.op != "call":
            return None
        resolved = self._result_source(definition, index)
        if resolved is None:
            return None
        result, bound = resolved
        if result.extent_resolver is not None:
            return result.extent_resolver(self.context, bound)
        if result.array_parameter is None:
            return 1
        return self._array_extent(
            bound.arguments[result.array_parameter],
            seen=seen,
        )

    def _array_extent_definition(
        self, definition: Any, *, seen: set[str]
    ) -> int | None:
        """Read one payload assignment to recover its per-thread element count.

        ``_array_extent`` calls this for every assignment that might supply an
        operand. Provider selection needs a fixed element count, even when
        the operation receives a copied value or one selected from a tuple.
        Follow intermediate expressions to a constructor or registered result
        policy. Constructors supply ``items_per_thread`` or an integer
        ``shape`` argument.

        Generated result markers use an explicit extent when present,
        otherwise inherit array extent or use one scalar item. Registered
        direct results use their extent resolver when present. Otherwise,
        follow their array-source argument or use one scalar item when the
        policy has no array source.

        Parameters
        ----------
        definition : object
            One right-hand-side IR value or expression that supplies a payload:
            a constructor call, alias, cast, phi, or tuple lookup.
        seen : set of str
            Variable names and tuple-position keys already visited on this
            recursion path. Copies let each phi input be examined independently.

        Returns
        -------
        int or None
            Known number of elements per thread. Constructor and generated
            marker counts are integers excluding booleans.
            ``None`` means no usable count was found. An unsupported source,
            nonconstant tuple index, or
            noninteger local-array shape supplies no count. This does not
            establish that every source is an array or that the count is
            positive.

        Raises
        ------
        InconsistentLoopPayloadExtentError
            A phi's inputs have different known element counts, so one provider
            specialization cannot describe every path.
        GroupRewriteError
            A constructor dimension depends on an unexpanded literal-unroll
            value. Errors from nested tuple analysis also propagate.
        ForceLiteralArg
            Resolving a constructor dimension requires the dispatcher to retry
            compilation with a literal kernel argument.
        """

        if isinstance(definition, ir.Var):
            return self._array_extent(definition, seen=seen)
        if not isinstance(definition, ir.Expr):
            return None
        if definition.op == "cast":
            return self._array_extent(definition.value, seen=seen)
        if definition.op == "phi":
            extents = {
                extent
                for incoming in getattr(definition, "incoming_values", ())
                if (extent := self._array_extent(incoming, seen=set(seen)))
                is not None
            }
            if len(extents) > 1:
                raise InconsistentLoopPayloadExtentError()
            return next(iter(extents), None)
        if definition.op in {"getitem", "static_getitem"}:
            index = getattr(definition, "index", None)
            if isinstance(index, ir.Var):
                resolved, index = self._try_constant(index)
                if not resolved:
                    return None
            if isinstance(index, Integral) and (not isinstance(index, bool)):
                return self._array_extent_tuple_item(
                    definition.value, int(index), seen=set(seen)
                )
            return None
        if definition.op != "call":
            return None
        function = self._callable(definition.func)
        if function is _typed_group_payload_like:
            try:
                is_array = self._constant(definition.args[1])
            except (GroupRewriteError, IndexError):
                return None
            if len(definition.args) >= 4:
                try:
                    extent = self._constant(definition.args[3])
                except GroupRewriteError:
                    return None
                if isinstance(extent, Integral) and (
                    not isinstance(extent, bool)
                ):
                    return int(extent)
                return None
            if is_array is False:
                return 1
            return self._array_extent(definition.args[0], seen=seen)
        if function in {ThreadData, _common_api.ThreadData}:
            bound = self._bind(function, definition)
            extent_argument = bound.arguments["items_per_thread"]
            self._reject_literal_unroll_value(extent_argument, "payload extent")
            try:
                extent = self._constant(extent_argument)
            except GroupRewriteError:
                return None
            if isinstance(extent, Integral) and (not isinstance(extent, bool)):
                return int(extent)
            return None
        if function is _cuda_local_array:
            shape_ref = (
                definition.args[0]
                if definition.args
                else dict(definition.kws).get("shape")
            )
            if shape_ref is None:
                return None
            self._reject_literal_unroll_value(shape_ref, "payload extent")
            try:
                extent = self._constant(shape_ref)
            except GroupRewriteError:
                return None
            if isinstance(extent, Integral) and (not isinstance(extent, bool)):
                return int(extent)
            return None
        resolved = self._result_source(definition)
        if resolved is None:
            return None
        result, bound = resolved
        if result.extent_resolver is not None:
            return result.extent_resolver(self.context, bound)
        if result.array_parameter is None:
            return 1
        return self._array_extent(
            bound.arguments[result.array_parameter], seen=seen
        )

    def _copy_array_payload(
        self,
        statements: list[Any],
        *,
        operation: str,
        source: ir.Var,
        destination: ir.Var,
        scope: Any,
        loc: ir.Loc,
        known_items_per_thread: int | None = None,
    ) -> None:
        """Append an unrolled copy between two fixed-size local payloads.

        Use ``known_items_per_thread`` when supplied; otherwise infer the
        source extent. Emit one item read and write per index into
        ``statements``. The caller must already have defined ``destination``
        in earlier pending statements. It usually names a result marker that
        the provider rewrite allocates later. The copy keeps caller-owned
        input intact when a native provider may modify it.

        Raise ``UnknownResultExtentError`` if no static extent is available.
        The statements remain pending until the owning planner installs them.

        Parameters
        ----------
        statements : list of IR statements
            Pending replacement statements, appended to in execution
            order. The function's blocks are unchanged until the owning
            planner installs this list.
        operation : str
            Canonical public operation name, used in diagnostics and
            generated temporary names.
        source : ir.Var
            Input payload whose elements must be preserved.
        destination : ir.Var
            Previously defined writable payload with enough slots for
            the copy.
        scope : ir.Scope
            Scope in which to create temporary IR variables.
        loc : ir.Loc
            Source location attached to generated statements and
            diagnostics.
        known_items_per_thread : int or None, optional
            Validated number of elements to copy per thread. ``None``
            asks the planner to infer this count from the source.
        """

        extent = (
            known_items_per_thread
            if known_items_per_thread is not None
            else self._array_extent(source)
        )
        if extent is None:
            raise UnknownResultExtentError(operation)
        for item_index in range(extent):
            index = self._value_var(
                statements,
                scope=scope,
                loc=loc,
                stem=f"{operation}_copy_index_{item_index}",
                value=item_index,
            )
            item = self._new_var(
                scope, loc, f"{operation}_copy_item_{item_index}"
            )
            statements.append(
                ir.Assign(ir.Expr.getitem(source, index, loc), item, loc)
            )
            statements.append(ir.SetItem(destination, index, item, loc))

    def _typed_payload_like(
        self,
        statements: list[Any],
        *,
        scope: Any,
        loc: ir.Loc,
        stem: str,
        prototype: ir.Var,
        is_array: bool,
        dtype_policy: str,
        items_per_thread: Any = None,
    ) -> ir.Var:
        """Append a fresh result marker and return its IR variable.

        Operation-family planners call this while constructing their
        replacement IR. Deferring allocation lets later payload inference
        establish the element type before a concrete local array is created.

        The marker retains ``prototype`` for dtype inference and ``is_array``
        for extent selection. An explicit ``items_per_thread`` overrides the
        inherited extent. ``dtype_policy`` identifies how the later rewrite
        should obtain the element type.

        Materialize the marker callable and constant controls in
        ``statements``. No local array is allocated here. The provider rewrite
        resolves the marker after payload facts are available and emits the
        allocation.

        Parameters
        ----------
        statements : list of IR statements
            Pending replacement statements, appended to in execution
            order. The function's blocks are unchanged until the owning
            planner installs this list.
        scope : ir.Scope
            Scope in which to create temporary IR variables.
        loc : ir.Loc
            Source location attached to generated statements and
            diagnostics.
        stem : str
            Readable prefix for fresh temporary names.
        prototype : ir.Var
            Existing scalar or array supplying type evidence for the new
            payload.
        is_array : bool
            Whether to inherit the prototype's per-thread array item
            count; false selects one item unless an explicit count is
            supplied.
        dtype_policy : str
            Registered rule for deriving the element dtype from the
            prototype or selecting a fixed result type.
        items_per_thread : int or ir.Var or None, optional
            Explicit compile-time item count, as a value or an IR
            variable that resolves to one. ``None`` uses the prototype-
            based count.
        """

        function_var = self._new_var(scope, loc, f"{stem}_payload_factory")
        statements.append(
            ir.Assign(
                ir.Global(function_var.name, _typed_group_payload_like, loc),
                function_var,
                loc,
            )
        )
        is_array_var = self._value_var(
            statements,
            scope=scope,
            loc=loc,
            stem=f"{stem}_is_array",
            value=is_array,
        )
        dtype_policy_var = self._value_var(
            statements,
            scope=scope,
            loc=loc,
            stem=f"{stem}_dtype_policy",
            value=dtype_policy,
        )
        args = [prototype, is_array_var, dtype_policy_var]
        if items_per_thread is not None:
            args.append(
                self._value_var(
                    statements,
                    scope=scope,
                    loc=loc,
                    stem=f"{stem}_items_per_thread",
                    value=items_per_thread,
                )
            )
        payload = self._new_var(scope, loc, f"{stem}_payload")
        statements.append(
            ir.Assign(ir.Expr.call(function_var, args, (), loc), payload, loc)
        )
        return payload

    def _boxed_group_operand(
        self,
        statements: list[Any],
        *,
        operation: str,
        value: ir.Var,
        scope: Any,
        loc: ir.Loc,
    ) -> tuple[ir.Var, bool]:
        """Represent a scalar as a one-item array for an array-only provider.

        Array-only families call this during group planning to reuse the
        same provider for scalar input. The Boolean result records the
        public operand form, so later result construction can restore that
        form without mistaking a one-item array for a scalar.

        Return an existing array unchanged. For a scalar, append a payload
        allocation marker and a write to element zero. Return the payload and
        a flag describing the original operand's array form. A family can use
        that flag to restore a scalar result after the provider call.

        Parameters
        ----------
        statements : list of IR statements
            Pending replacement statements, appended to in execution
            order. The function's blocks are unchanged until the owning
            planner installs this list.
        operation : str
            Canonical public operation name, used in diagnostics and
            generated temporary names.
        value : ir.Var
            Public operand, either a supported per-thread array or a
            scalar.
        scope : ir.Scope
            Scope in which to create temporary IR variables.
        loc : ir.Loc
            Source location attached to generated statements and
            diagnostics.
        """

        is_array = self._array_operand_state(operation, value)
        if is_array:
            return value, True
        payload = self._typed_payload_like(
            statements,
            scope=scope,
            loc=loc,
            stem=f"{operation}_input",
            prototype=value,
            is_array=False,
            dtype_policy=_PAYLOAD_DTYPE_LIKE,
        )
        index = self._value_var(
            statements,
            scope=scope,
            loc=loc,
            stem=f"{operation}_input_index",
            value=0,
        )
        statements.append(ir.SetItem(payload, index, value, loc))
        return payload, False

    def _result_value(
        self,
        statements: list[Any],
        *,
        payload: ir.Var,
        is_array: bool,
        scope: Any,
        loc: ir.Loc,
        stem: str,
    ) -> ir.Var:
        """Recover the public result shape from an internal array payload.

        A family calls this after appending the provider call to its
        replacement statements. This restores the public result form when
        the provider itself always writes an array.

        Return an array payload unchanged. For a scalar result, append a read
        of element zero and return its variable. The caller must supply a
        one-item payload for that case; this helper does not check its extent.

        Parameters
        ----------
        statements : list of IR statements
            Pending replacement statements, appended to in execution
            order. The function's blocks are unchanged until the owning
            planner installs this list.
        payload : ir.Var
            Internal provider result array, already defined by earlier
            pending statements.
        is_array : bool
            Whether the public call should return the whole array. False
            extracts element zero as a scalar.
        scope : ir.Scope
            Scope in which to create temporary IR variables.
        loc : ir.Loc
            Source location attached to generated statements and
            diagnostics.
        stem : str
            Readable prefix for the index and scalar-result temporary
            names.
        """

        if is_array:
            return payload
        index = self._value_var(
            statements,
            scope=scope,
            loc=loc,
            stem=f"{stem}_index",
            value=0,
        )
        result = self._new_var(scope, loc, f"{stem}_scalar")
        statements.append(
            ir.Assign(ir.Expr.getitem(payload, index, loc), result, loc)
        )
        return result

    def _lower_root_operation(
        self, inst: ir.Assign, call: ir.Expr, function: Any, operation: str
    ) -> None:
        """Validate a public call and queue its registered provider rewrite.

        Bind the public signature, reconstruct and resolve its group, and
        apply common-API group and argument rules when the call came from that
        API. Then delegate operation-specific choices to its registration
        using ``GroupPlanningContext``. Record the returned statements and
        mark the old callable variable for removal; the original IR remains
        available while other calls are planned.

        Parameters
        ----------
        inst : ir.Assign
            Assignment to replace after the complete function passes
            validation.
        call : ir.Expr
            Public operation call stored in ``inst``.
        function : callable
            Recognized public callable whose signature and API origin apply.
        operation : str
            Canonical name used to look up the registered operation planner.

        Raises
        ------
        GroupRewriteError
            Call arguments or group reconstruction are invalid, no planner is
            registered, or operation-specific validation rejects the request.
        TypeError
            Arguments do not match the public signature or accepted parameter
            types.
        ForceLiteralArg
            Planning needs a function argument specialized as a literal.
        NotImplementedError
            Group resolution or the operation planner finds no supported
            lowering.
        """

        bound = self._bind(function, call)
        if bound.arguments.get("kwargs"):
            names = ", ".join(sorted(bound.arguments["kwargs"]))
            raise GroupRewriteError(
                f"cuda.coop.numba_mlir.{operation} "
                f"got unexpected keyword(s): {names}"
            )
        group = self._group(bound.arguments["group"])
        if group is None:
            raise NonConstantThreadGroupError(operation)
        is_common_root = _is_common_root_operation(function, operation)
        if is_common_root:
            _common_dispatch._validate_common_operation_group(operation, group)
        group = self._resolve_group(group, feature=operation)
        registration = group_primitive(operation)
        if registration is None:
            raise GroupRewriteError(
                f"cuda.coop.numba_mlir operation {operation!r} has no planner"
            )
        if (
            is_common_root
            and registration.validate_common_arguments is not None
        ):
            registration.validate_common_arguments(
                self.context, operation, bound
            )
        replacement = registration.lower(
            self.context,
            inst,
            operation=operation,
            group=group,
            bound=bound,
            is_common_root=is_common_root,
        )
        self.dead_func_names.add(call.func.name)
        self.replacements[inst] = replacement

    def _group_method(self, call: ir.Expr) -> tuple[str, ThreadGroup] | None:
        """Recognize a supported method on a resolvable group descriptor.

        ``run`` uses this probe to distinguish executable queries such as
        ``group.rank()`` from descriptor construction before it chooses a
        replacement device helper.

        Return the method name and group, or ``None`` for unrelated calls.
        ``group_by`` is excluded because it constructs another descriptor;
        this path handles methods that must become executable device helpers.

        Parameters
        ----------
        call : ir.Expr
            Call expression inspected by ``run`` while the original
            group descriptors and their aliases are still present.
        """

        definition = self._definition(call.func)
        if (
            not isinstance(definition, ir.Expr)
            or definition.op != "getattr"
            or definition.attr not in _GROUP_METHODS
            or definition.attr == "group_by"
        ):
            return None
        group = self._group(definition.value)
        if group is None:
            return None
        return definition.attr, group

    def _lower_group_method(
        self, inst: ir.Assign, call: ir.Expr, *, method: str, group: ThreadGroup
    ) -> None:
        """Plan a group method and stage its no-argument device helper call.

        ``run`` calls this after recognizing a query or synchronization
        method. Unlike an ordinary Python object method, the receiver
        describes compile-time thread membership and must disappear before
        device typing.

        Validate argument shape and resolve dtype/level controls as constants.
        Resolve the group through the requested hierarchy level and reject
        mapped-group queries above the immediate parent, mapped-warp
        synchronization, and grid synchronization. Grid sync needs a
        cooperative launch, which this launch descriptor cannot request.

        Reuse an invocable keyed by group semantics, operation, dtype, and
        level within this planner attempt. Its C++ helper embeds the
        descriptor, so the replacement passes no runtime group object. Record
        the old callee as dead and stage the replacement; the normal planner
        run installs it later.

        Parameters
        ----------
        inst : ir.Assign
            Original public call assignment. Its target, scope, and
            source location identify the replacement result and
            generated temporaries.
        call : ir.Expr
            Method-call expression stored in ``inst.value``, retaining
            the public positional and keyword arguments.
        method : str
            Supported method name returned by ``_group_method``,
            including typed query variants such as ``rank_as``.
        group : ThreadGroup
            Descriptor recovered from the method receiver. Its hierarchy
            is resolved against this compilation's launch before code
            generation.
        """

        if call.vararg is not None or call.varkwarg is not None:
            raise GroupRewriteError(
                f"ThreadGroup.{method} does not support splats"
            )
        kwargs = dict(call.kws)
        dtype = None
        level = "thread"
        if method in {"rank", "count"}:
            if len(call.args) > 1 or any(name != "level" for name in kwargs):
                raise GroupRewriteError(
                    f"invalid ThreadGroup.{method} arguments"
                )
            if call.args and "level" in kwargs:
                raise GroupRewriteError(
                    f"ThreadGroup.{method} received level more than once"
                )
            if call.args:
                level = self._constant(call.args[0])
            elif "level" in kwargs:
                level = self._constant(kwargs["level"])
            operation = method
        elif method in {"rank_as", "count_as"}:
            if len(call.args) > 2 or any(
                name not in {"dtype", "level"} for name in kwargs
            ):
                raise GroupRewriteError(
                    f"invalid ThreadGroup.{method} arguments"
                )
            if call.args and "dtype" in kwargs:
                raise GroupRewriteError(
                    f"ThreadGroup.{method} received dtype more than once"
                )
            if len(call.args) > 1 and "level" in kwargs:
                raise GroupRewriteError(
                    f"ThreadGroup.{method} received level more than once"
                )
            if call.args:
                dtype = self._constant(call.args[0])
            elif "dtype" in kwargs:
                dtype = self._constant(kwargs["dtype"])
            if len(call.args) > 1:
                level = self._constant(call.args[1])
            elif "level" in kwargs:
                level = self._constant(kwargs["level"])
            operation = method.removesuffix("_as")
        else:
            if call.args or kwargs:
                raise GroupRewriteError(
                    f"ThreadGroup.{method} accepts no arguments"
                )
            operation = method

        if operation in {"rank", "count"}:
            level = normalize_thread_level(
                level,
                scope="cuda.coop.numba_mlir",
                feature=f"ThreadGroup.{operation}",
            )
            if group.mapping is not None:
                level_order = {
                    "thread": 0,
                    "warp": 1,
                    "block": 2,
                    "cluster": 3,
                    "grid": 4,
                }
                if level_order[level] > level_order[group.mapping.parent]:
                    raise NotImplementedError(
                        "cuda.coop.numba_mlir mapped ThreadGroup queries above "
                        "the immediate parent require "
                        "recursive group composition"
                    )
            group = self._resolve_group(
                group, feature=f"ThreadGroup.{operation}", through_level=level
            )
        else:
            group = self._resolve_group(
                group, feature=f"ThreadGroup.{operation}"
            )
        if group.kind == "warps_within_block" and operation in {
            "sync",
            "sync_aligned",
        }:
            raise NotImplementedError(
                "cuda.coop.numba_mlir mapped-Warp synchronization requires "
                "planner-owned barrier lifetime"
            )
        if group.kind == "grid" and operation in {"sync", "sync_aligned"}:
            raise NotImplementedError(
                "cuda.coop.numba_mlir grid synchronization requires a verified "
                "cooperative launch, which the "
                "current launch descriptor cannot "
                "request"
            )

        from .._lowering._thread_group import (
            _normalize_query_dtype,
            make_group_method_invocable,
        )

        if operation in {"rank", "count"}:
            dtype = _normalize_query_dtype(group, level, dtype)

        key = (group.semantic_key, operation, dtype, level)
        invocable = self._group_method_invocables.get(key)
        if invocable is None:
            invocable = make_group_method_invocable(
                group=group,
                operation=operation,
                dtype=dtype,
                level=level,
                compile_context=self._provider_compile_context(),
            )
            self._group_method_invocables[key] = invocable
        self.dead_func_names.add(call.func.name)
        self.replacements[inst] = self._rewritten_call(
            inst,
            factory=invocable,
            args=[],
            kwargs={},
        )

    def _mark_descriptor_calls(self) -> None:
        """Mark thread-group and hierarchy assignments for removal.

        Recognize hierarchy and group constructors by callable identity, and
        ``group_by`` only when its receiver resolves to a group. Then
        repeatedly follow aliases and casts until every reachable descriptor
        assignment is marked. Record consumed callable names too. This is
        bookkeeping only: ``_validate_group_descriptor_references`` must
        approve removal before blocks change.
        """

        for block in self.func_ir.blocks.values():
            for inst in block.body:
                if not isinstance(inst, ir.Assign):
                    continue
                call = inst.value
                if not isinstance(call, ir.Expr) or call.op != "call":
                    continue
                function = self._callable(call.func)
                if (
                    function is ThreadHierarchy
                    or function in _GROUP_CONSTRUCTORS
                ):
                    self.descriptor_assigns.add(inst)
                    self.dead_func_names.add(call.func.name)
                    continue
                definition = self._definition(call.func)
                if (
                    isinstance(definition, ir.Expr)
                    and definition.op == "getattr"
                    and (definition.attr == "group_by")
                    and (self._group(definition.value) is not None)
                ):
                    self.descriptor_assigns.add(inst)
                    self.dead_func_names.add(call.func.name)
        descriptor_names = {
            inst.target.name for inst in self.descriptor_assigns
        }
        changed = True
        while changed:
            changed = False
            for block in self.func_ir.blocks.values():
                for inst in block.body:
                    if not isinstance(inst, ir.Assign):
                        continue
                    source = inst.value
                    if isinstance(source, ir.Expr) and source.op == "cast":
                        source = source.value
                    if (
                        isinstance(source, ir.Var)
                        and source.name in descriptor_names
                        and (inst.target.name not in descriptor_names)
                    ):
                        self.descriptor_assigns.add(inst)
                        descriptor_names.add(inst.target.name)
                        changed = True

    def _validate_group_descriptor_references(self) -> None:
        """Reject runtime references to thread groups or hierarchies.

        Here, a descriptor is a ``ThreadGroup`` from ``this_block()``,
        ``this_warp()``, or another group constructor or ``group_by()`` call,
        or a ``ThreadHierarchy`` value. These describe which threads cooperate.
        ``ThreadData`` provides the per-thread payload array and is handled
        separately.

        A group or hierarchy value may feed another group/hierarchy assignment,
        a primitive call with a queued replacement, or a method lookup marked
        for removal. Returning that value or passing it to an unrelated call
        would require a device object after its construction has been erased.
        Such uses raise ``EscapingGroupDescriptorError`` naming the variables.
        Run this after collecting replacements, before rewriting block bodies.
        """

        descriptor_names = {
            inst.target.name for inst in self.descriptor_assigns
        }
        if not descriptor_names:
            return
        for block in self.func_ir.blocks.values():
            for inst in block.body:
                used_names = {
                    value.name for value in inst.list_vars()
                } & descriptor_names
                if isinstance(inst, ir.Assign):
                    used_names.discard(inst.target.name)
                if not used_names:
                    continue
                if inst in self.descriptor_assigns or inst in self.replacements:
                    continue
                if isinstance(inst, ir.Assign):
                    value = inst.value
                    if (
                        isinstance(value, ir.Expr)
                        and value.op == "getattr"
                        and isinstance(value.value, ir.Var)
                        and (value.value.name in used_names)
                        and (inst.target.name in self.dead_func_names)
                    ):
                        continue
                names = ", ".join(sorted(used_names))
                raise EscapingGroupDescriptorError(names)

    def run(self) -> bool:
        """Plan group operations and replace their descriptors in the IR.

        ``_GroupPlanning._resolve_groups`` creates this planner and calls
        ``run()`` during ``CoopWholeFunctionPlanner.run``. Numba has already
        inlined device helpers, making their cooperative calls visible in the
        kernel, and ordinary type inference has not started. The caller has
        obtained the configured launch dimensions so this phase can resolve
        thread groups and choose provider specializations.

        First reject unsupported literal-unroll dependencies and identify
        assignments that construct ``ThreadHierarchy`` or groups, including
        ``group_by()`` calls and descriptor aliases or casts. These describe
        compile-time choices and are marked for removal once their uses have
        been consumed. Ask each registered operation family to build
        replacement statements, then check the remaining descriptor uses
        before replacing any block bodies. Returning a descriptor or passing
        it to an unrelated runtime call, for example, prevents its removal and
        raises an error.

        Accepted descriptor assignments and obsolete callable assignments
        become ``None`` assignments, preserving their targets; public
        operations become private provider calls carrying their lowering
        plans. This removes the group markers so the same whole-function
        planner can proceed to provider rewriting. Planning updates this
        instance's caches, dtype facts, and replacement bookkeeping; block
        bodies change only after all calls and descriptor uses pass.

        Recognized rank, count, membership, and synchronization methods also
        become provider calls. Their helpers embed the resolved group and
        compile-time query controls before the descriptor is erased.

        Returns
        -------
        bool
            ``True`` when block bodies were rewritten, including
            descriptor-only cleanup. ``False`` when no descriptor, call, or
            callable needs replacing. ``_resolve_groups`` forwards this result
            to ``CoopWholeFunctionPlanner.run``: a true result triggers IR
            repair before provider rewriting, which runs in either case. The
            outer planner reports changes from either phase to Numba, which
            repairs the final IR and proceeds to the next registered planner.
            This boolean does not request another run of group planning.

        Raises
        ------
        GroupRewriteError
            A group call is invalid or a descriptor escapes its compile-time
            uses.
        ForceLiteralArg
            Planning requires specialization of a function argument.
        NotImplementedError
            The requested group or operation has no supported lowering.
        """

        self._reject_literal_unroll_constructors()
        self._mark_descriptor_calls()
        for block in self.func_ir.blocks.values():
            for inst in block.body:
                if not isinstance(inst, ir.Assign):
                    continue
                call = inst.value
                if not isinstance(call, ir.Expr) or call.op != "call":
                    continue
                function = self._callable(call.func)
                operation = _group_operation_name(function)
                if operation is not None:
                    self._lower_root_operation(inst, call, function, operation)
                    continue
                method = self._group_method(call)
                if method is not None:
                    method_name, group = method
                    self._lower_group_method(
                        inst,
                        call,
                        method=method_name,
                        group=group,
                    )
        self._validate_group_descriptor_references()
        if not (
            self.descriptor_assigns or self.replacements or self.dead_func_names
        ):
            return False
        for block in self.func_ir.blocks.values():
            rewritten: list[Any] = []
            for inst in block.body:
                replacement = self.replacements.get(inst)
                if replacement is not None:
                    rewritten.extend(replacement)
                    continue
                if isinstance(inst, ir.Assign) and (
                    inst in self.descriptor_assigns
                    or inst.target.name in self.dead_func_names
                ):
                    rewritten.append(
                        ir.Assign(
                            ir.Const(None, inst.loc), inst.target, inst.loc
                        )
                    )
                    continue
                rewritten.append(inst)
            block.body = rewritten
        return True


def has_group_markers(func_ir: ir.FunctionIR) -> bool:
    """Return whether the current function IR still needs group planning.

    One recognized call anywhere in the function is enough: a group
    constructor such as ``this_block()``, ``ThreadHierarchy()``, a registered
    public group operation such as ``load()`` or ``store()``, or a supported
    method on a recognized group descriptor. Partitioning, hierarchy queries,
    and synchronization need planning even when no primitive remains.

    Trace method receivers through aliases, casts, control-flow merges, and
    earlier subgroup calls to distinguish group descriptors from unrelated
    objects with the same method name.

    Inspect only the supplied IR. Calls inside device helpers become visible
    here after inlining; this scan does not visit their bodies. ``ThreadData``
    and ``TempStorage`` constructors alone do not count as group markers.

    The single whole-function planner uses a positive result in its group
    resolution phase to request launch metadata and resolve the calls.
    Successful resolution removes group descriptors and replaces public
    operations with private provider calls, which do not count as group
    markers. The following ``CoopSinglePhaseRewrite`` helper also checks for
    unresolved markers before materializing those provider calls.

    Detection does not validate the calls or resolve their launch dimensions;
    those checks belong to the group planner.

    Parameters
    ----------
    func_ir : ir.FunctionIR
        Current function IR with its recorded-definition lookup available.
        Scanned without modifying its blocks or resolving launch metadata.

    Returns
    -------
    bool
        Whether at least one recognized group-planning marker remains.
    """
    # Detection needs definition lookup only, before launch facts exist.
    analyzer = object.__new__(_GroupCallPlanner)
    analyzer.func_ir = func_ir
    analyzer._group_cache = {}
    analyzer._hierarchy_cache = {}

    def is_group_descriptor(value: Any, seen: set[str]) -> bool:
        if isinstance(value, ThreadGroup):
            return True
        if not isinstance(value, ir.Var) or value.name in seen:
            return False
        seen = {*seen, value.name}
        for definition in analyzer._all_definitions(value):
            if isinstance(definition, ir.Var) and is_group_descriptor(
                definition, seen
            ):
                return True
            if isinstance(definition, (ir.Global, ir.FreeVar, ir.Const)):
                if isinstance(definition.value, ThreadGroup):
                    return True
                continue
            if not isinstance(definition, ir.Expr):
                continue
            if definition.op == "cast" and is_group_descriptor(
                definition.value, seen
            ):
                return True
            if definition.op == "phi" and any(
                is_group_descriptor(incoming, seen)
                for incoming in getattr(definition, "incoming_values", ())
            ):
                return True
            if definition.op != "call":
                continue
            function = analyzer._callable(definition.func)
            if function in _GROUP_CONSTRUCTORS:
                return True
            function_definition = analyzer._definition(definition.func)
            if (
                isinstance(function_definition, ir.Expr)
                and function_definition.op == "getattr"
                and function_definition.attr == "group_by"
                and is_group_descriptor(function_definition.value, seen)
            ):
                return True
        return False

    for block in func_ir.blocks.values():
        for inst in block.body:
            if not isinstance(inst, ir.Assign):
                continue
            value = inst.value
            if not isinstance(value, ir.Expr) or value.op != "call":
                continue
            function = analyzer._callable(value.func)
            if (
                function is ThreadHierarchy
                or function in _GROUP_CONSTRUCTORS
                or _group_operation_name(function) is not None
            ):
                return True
            function_definition = analyzer._definition(value.func)
            if (
                isinstance(function_definition, ir.Expr)
                and function_definition.op == "getattr"
                and function_definition.attr in _GROUP_METHODS
                and is_group_descriptor(function_definition.value, set())
            ):
                return True
    return False


class _GroupPlanning:
    """Add the group-resolution phase to ``CoopWholeFunctionPlanner``.

    This is a mixin: the whole-function planner inherits its methods rather
    than creating a separate ``_GroupPlanning`` object. The owning planner
    supplies ``state`` and ``is_device_function`` through Numba's
    ``WholeFunctionPlanner`` base class. Its ``run()`` method calls
    ``_resolve_groups()`` after device helpers have been inlined and before
    provider-call rewriting starts.

    The method first checks whether any group calls remain, then requests
    launch dimensions and delegates to a fresh ``_GroupCallPlanner``. Calls
    in a standalone device function must instead be inlined into a kernel,
    whose configured launch supplies those dimensions. The returned boolean
    tells the owner whether to rebuild IR analysis before rewriting provider
    calls.

    Group resolution also checks explicit launch bounds against the exact
    block size. When neither launch bounds nor a register limit is supplied,
    it records that thread count as a launch bound for this compilation only;
    the dispatcher's user options remain unchanged.
    """

    def _resolve_groups(self) -> bool:
        """Resolve group calls using launch metadata and report IR changes.

        Request launch facts only when group markers remain in the function.
        Device helpers must first be inlined into a kernel so their groups
        can be resolved against that kernel's configured launch.

        ``require_launch_config()`` explicitly declares that generated code
        depends on launch values. Numba-CUDA-MLIR already has those values in
        its dispatcher, but does not automatically expose them as compiler
        specialization inputs. This opt-in lets kernels that do not need
        launch facts retain generic dispatch, avoiding additional launch
        normalization and specialization lookups. The helper runs during
        compilation; the later launch cost comes from selecting a matching
        specialization.

        The runtime helper reads the options shared by compiler passes in
        ``state.metadata["targetoptions"]``. If ``"__launch_config__"``
        already holds a dictionary containing ``grid`` and ``block``, it
        returns that dictionary. Otherwise, it gets the per-attempt tracker
        from ``state.metadata`` using the runtime's
        ``_LAUNCH_CONFIG_TRACKER_METADATA_KEY`` (``"launch_config_tracker"``).
        This slot holds a temporary coordination object. The tracker lazily
        normalizes the available ``grid``, ``block``, ``sharedmem``, and
        ``cluster`` values and marks them as required. The helper caches the
        result in ``targetoptions["__launch_config__"]`` for later passes.

        Compilation continues in the same attempt. The dispatcher observes
        the tracker's required flag and keys the compiled result by launch
        values so later launches reuse a matching specialization. The runtime
        removes the temporary tracker from the final compilation metadata.
        Without an active configuration or tracker, the helper raises an
        error requiring compilation through a configured kernel launch.

        Pass the returned configuration to ``_GroupCallPlanner`` to resolve
        the group topology and replace public operations with provider calls.
        Check explicit ``launch_bounds`` against the exact block size. When
        neither ``launch_bounds`` nor ``max_registers`` is set, set
        ``launch_bounds`` to the exact thread count, ``x * y * z``. Without a
        bound, code generation can give each thread more registers than a
        large block can supply, and the launch fails. An explicit register
        limit already chooses that tradeoff, so it turns inference off.
        Inferred bounds affect this compiled result only. The dispatcher keeps
        the user options and can infer a bound for each later specialization.

        Returns
        -------
        bool
            Whether group descriptors or public operations were replaced in
            ``state.func_ir``. A function without group markers returns
            ``False`` without requesting launch metadata.

        Raises
        ------
        GroupRewriteError
            Cooperative calls remain in a standalone device function, or group
            planning finds invalid calls or escaping descriptors, or the exact
            block exceeds explicit launch bounds.
        ForceLiteralArg
            A group or operation needs a compile-time argument value. The
            dispatcher consumes this compiler signal and retries with the
            requested literal specialization.
        RuntimeError
            Numba has no configured launch metadata for this compilation.
            Public compiler entry points translate the compiler's internal
            signal to a diagnostic requiring a configured kernel launch.
        NotImplementedError
            The requested group or operation has no supported lowering.
        """
        planner = cast("CoopWholeFunctionPlanner", self)
        if not has_group_markers(planner.state.func_ir):
            return False
        if planner.is_device_function:
            function_name = planner.state.func_ir.func_id.func_qualname
            raise GroupRewriteError(
                "cuda.coop.numba_mlir cooperative calls in device function "
                f"{function_name!r} must be inlined into a kernel. "
                "Standalone primitive helpers and primitives inside "
                "standalone callbacks are unsupported; use inline='always' "
                "for a kernel helper or move the cooperative calls "
                "into the kernel."
            )
        launch_config = require_launch_config(planner.state)
        group_planner = _GroupCallPlanner(planner.state, launch_config)
        changed = group_planner.run()
        assert isinstance(group_planner.launch.exact_block_dim, tuple)
        x, y, z = group_planner.launch.exact_block_dim
        threads = x * y * z
        # Configured compiles own these options; never write inferred bounds
        # into the dispatcher's persistent user options. An explicit register
        # limit keeps its original compiler/resource tradeoff.
        options = planner.state.metadata["targetoptions"]
        bounds = options.get("launch_bounds")
        if bounds is not None:
            maximum = bounds[0] if isinstance(bounds, tuple) else bounds
            if threads > maximum:
                raise GroupRewriteError(
                    f"cuda.coop exact launch block {(x, y, z)!r} has {threads} "
                    f"threads, exceeding explicit launch_bounds={bounds!r}."
                )
        elif options.get("max_registers") is None:
            options["launch_bounds"] = threads
        return changed


__all__ = [
    "GroupRewriteError",
    "_GroupCallPlanner",
    "_GroupPlanning",
    "has_group_markers",
]
