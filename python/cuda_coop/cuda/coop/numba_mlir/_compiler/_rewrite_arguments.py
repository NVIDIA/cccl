# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Separate provider specialization inputs from device-call operands.

After group resolution selects a provider, argument validation applies its
registered grammar, infers missing payload and launch facts, and distinguishes
static scalar bindings from runtime controls. The result supplies host-side
factory keywords and device-side IR operands to the remaining rewrite helpers;
this module does not replace the call or allocate its storage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from cuda.coop._core import ArgumentBinding, GroupLoweringPlan

from ._group_rewriting import GroupRewriteContext
from ._operations import _GROUP_LOWERING_PLAN_KWARG, rewrite_operation
from ._rewrite_support import (
    _UNRESOLVED,
    CoopSinglePhaseRewriteError,
    _DeferredCoopRewrite,
    ir,
)

if TYPE_CHECKING:
    from ._rewrite import CoopSinglePhaseRewrite


class _ArgumentRewrite:
    """Split provider calls using their registered argument rules.

    This mixin gathers static factory values and ordered runtime operands.
    ``CoopSinglePhaseRewrite`` supplies provenance, payload, and launch
    inference through the other mixins.
    """

    def _validate_and_split_args(
        self, op_name: str, call: ir.Expr, getitem_temp_storage: ir.Var | None
    ) -> tuple[
        tuple[ir.Var, ...],
        ir.Var | None,
        dict[str, object],
        tuple[ir.Var, ...],
    ]:
        """Partition provider arguments into static and runtime inputs.

        The operation registry defines positional arity, factory keywords,
        scalar bindings, and optional runtime controls. Static scalar controls
        become ``ArgumentBinding.static`` values; unresolved controls stay in
        the runtime argument list in registry order. A resolved ``None`` omits
        an optional control, whereas ``_UNRESOLVED`` preserves its runtime
        operand. Temporary storage is returned separately for later
        ABI-specific insertion.

        Pass the call's lowering plan to payload inference so Load/Store can
        reuse checked dtype and item-count facts while validating explicit
        factory inputs. Infer missing inputs from operands and exact launch
        metadata, validate runtime controls, and normalize the ``dim`` alias.
        The private plan is also carried in the factory dictionary for the
        caller to remove before invoking the provider. Record the variables
        that supplied compile-time inputs so ``finish_rewrite`` can remove
        their unused assignments after all blocks have been rewritten. The
        call expression itself is not rewritten here.

        Parameters
        ----------
        op_name : str
            Registered operation whose argument contract governs the call.
        call : ir.Expr
            Untyped IR call with positional and keyword operands.
        getitem_temp_storage : ir.Var or None
            Storage operand discovered in a subscripted callee. Argument
            splitting recognizes it, but descriptor-use validation decides
            whether that syntax is permitted.

        Returns
        -------
        runtime_args : tuple of ir.Var
            Provider operands, excluding the leading storage pointer.
        runtime_temp_storage : ir.Var or None
            Caller scratch operand, or None for implementation-owned storage.
        factory_kwargs : dict
            Resolved Python specialization values and ``ArgumentBinding``
            descriptors, plus the private lowering plan until the caller
            extracts it.
        factory_kw_value_vars : tuple of ir.Var
            Original IR variables whose assignments may become unused after
            rewriting. These are cleanup candidates, never factory inputs.
            They do not correspond one-to-one with ``factory_kwargs``:
            inferred keywords have no source variable, while omitted controls
            and lowering metadata can leave a variable without a keyword.
            Runtime controls keep their variables in ``runtime_args``.

        Raises
        ------
        CoopSinglePhaseRewriteError
            Arguments are invalid, factory values cannot be resolved, or
            required inputs remain unavailable when deferral is disabled.
        _DeferredCoopRewrite
            Internal signal that exact launch metadata is still needed.
            ``CoopSinglePhaseRewrite.prepare_calls_and_storage`` catches it and
            preserves the call while ``_CallRewriting._rewrite_calls`` obtains
            the kernel launch shape and retries within
            ``CoopWholeFunctionPlanner``.
        """

        rewrite = cast("CoopSinglePhaseRewrite", self)
        specification = rewrite_operation(op_name)
        if specification is None:
            raise CoopSinglePhaseRewriteError(
                f"unsupported Numba-CUDA-MLIR operation {op_name!r}"
            )
        if call.vararg is not None or call.varkwarg is not None:
            raise CoopSinglePhaseRewriteError(
                "cooperative group calls do not support *args or **kwargs."
            )
        runtime_arg_count = len(call.args)
        if runtime_arg_count not in specification.runtime_arg_counts:
            expected_csv = ", ".join(
                str(v) for v in sorted(specification.runtime_arg_counts)
            )
            raise CoopSinglePhaseRewriteError(
                f"cooperative group operation {op_name!r} expects a "
                f"positional runtime argument count in {{{expected_csv}}}; "
                f"got {runtime_arg_count}."
            )
        base_runtime_arg_count = min(specification.runtime_arg_counts)
        runtime_args = list(call.args[:base_runtime_arg_count])
        factory_kw_value_vars: list[ir.Var] = []
        allowed_factory_kwargs = set(specification.allowed_factory_kwargs)
        required_factory_kwargs = specification.required_factory_kwargs
        seen_factory_kwargs: set[str] = set()
        factory_kwargs: dict[str, object] = {}
        runtime_temp_storage = getitem_temp_storage
        runtime_factory_kwargs = specification.runtime_factory_kwargs
        runtime_factory_kw_prerequisites = dict(
            specification.runtime_factory_kw_prerequisites
        )
        scalar_binding_kwargs = specification.scalar_binding_kwargs
        extra_runtime_arg_count = runtime_arg_count - base_runtime_arg_count
        positional_runtime_factory_kwargs = set(
            runtime_factory_kwargs[:extra_runtime_arg_count]
        )
        seen_runtime_factory_kwargs: set[str] = set()
        runtime_factory_kw_vars: dict[str, ir.Var] = {}
        runtime_offset_var = None
        lowering_plan = None
        seen_lowering_plan = False
        if runtime_factory_kwargs:
            if extra_runtime_arg_count > len(runtime_factory_kwargs):
                raise CoopSinglePhaseRewriteError(
                    f"cooperative group operation {op_name!r} received too "
                    f"many positional runtime arguments."
                )
            for index, name in enumerate(
                runtime_factory_kwargs[:extra_runtime_arg_count]
            ):
                value_var = call.args[base_runtime_arg_count + index]
                if name in scalar_binding_kwargs:
                    value = rewrite._resolve_static_scalar_value(value_var)
                    if value is not _UNRESOLVED:
                        if value is not None:
                            factory_kwargs[name] = (
                                value
                                if isinstance(value, ArgumentBinding)
                                else ArgumentBinding.static(value)
                            )
                            seen_factory_kwargs.add(name)
                        if isinstance(value_var, ir.Var):
                            factory_kw_value_vars.append(value_var)
                        continue
                runtime_args.append(value_var)
                factory_kwargs[name] = (
                    ArgumentBinding.runtime()
                    if name in scalar_binding_kwargs
                    else True
                )
                seen_factory_kwargs.add(name)
                seen_runtime_factory_kwargs.add(name)
        for name, value_var in call.kws:
            if name == _GROUP_LOWERING_PLAN_KWARG:
                if seen_lowering_plan:
                    raise CoopSinglePhaseRewriteError(
                        "cooperative group provider marker received "
                        "duplicate lowering-plan metadata."
                    )
                seen_lowering_plan = True
                lowering_plan = rewrite._resolve_factory_kwarg_value(
                    op_name, name, value_var
                )
                if not isinstance(lowering_plan, GroupLoweringPlan):
                    raise CoopSinglePhaseRewriteError(
                        "cooperative group provider marker carries invalid "
                        "lowering-plan metadata."
                    )
                if isinstance(value_var, ir.Var):
                    factory_kw_value_vars.append(value_var)
                continue
            if name == "temp_storage" and specification.accepts_temp_storage:
                if runtime_temp_storage is not None:
                    raise CoopSinglePhaseRewriteError(
                        f"cooperative group operation {op_name!r} received "
                        f"duplicate temp_storage arguments."
                    )
                if not isinstance(value_var, ir.Var):
                    raise CoopSinglePhaseRewriteError(
                        "cooperative group temp_storage must be a variable."
                    )
                runtime_temp_storage = value_var
                continue
            if name == specification.runtime_offset_kwarg:
                if (
                    runtime_offset_var is not None
                    or name in seen_factory_kwargs
                ):
                    raise CoopSinglePhaseRewriteError(
                        f"cooperative group operation {op_name!r} received a "
                        f"duplicate runtime argument {name!r}."
                    )
                if not isinstance(value_var, ir.Var):
                    raise CoopSinglePhaseRewriteError(
                        f"cooperative group runtime argument {name!r} must "
                        f"be a variable."
                    )
                value = rewrite._resolve_static_scalar_value(value_var)
                if value is not _UNRESOLVED:
                    if value is not None:
                        factory_kwargs[name] = (
                            value
                            if isinstance(value, ArgumentBinding)
                            else ArgumentBinding.static(value)
                        )
                        seen_factory_kwargs.add(name)
                    factory_kw_value_vars.append(value_var)
                    continue
                runtime_offset_var = value_var
                continue
            if name in runtime_factory_kwargs:
                if (
                    name in seen_runtime_factory_kwargs
                    or name in runtime_factory_kw_vars
                    or name in seen_factory_kwargs
                    or name in positional_runtime_factory_kwargs
                ):
                    raise CoopSinglePhaseRewriteError(
                        f"cooperative group operation {op_name!r} received a "
                        f"duplicate runtime argument {name!r}."
                    )
                if not isinstance(value_var, ir.Var):
                    raise CoopSinglePhaseRewriteError(
                        f"cooperative group runtime argument {name!r} must "
                        f"be a variable."
                    )
                if name in scalar_binding_kwargs:
                    value = rewrite._resolve_static_scalar_value(value_var)
                    if value is not _UNRESOLVED:
                        if value is not None:
                            factory_kwargs[name] = (
                                value
                                if isinstance(value, ArgumentBinding)
                                else ArgumentBinding.static(value)
                            )
                            seen_factory_kwargs.add(name)
                        factory_kw_value_vars.append(value_var)
                        continue
                runtime_factory_kw_vars[name] = value_var
                continue
            if name not in allowed_factory_kwargs:
                allowed = ", ".join(
                    sorted(
                        set(allowed_factory_kwargs)
                        | set(runtime_factory_kwargs)
                    )
                )
                raise CoopSinglePhaseRewriteError(
                    f"cooperative group operation {op_name!r} does not "
                    f"support factory keyword {name!r}. Allowed keywords "
                    f"are: {allowed}."
                )
            if name in seen_factory_kwargs:
                raise CoopSinglePhaseRewriteError(
                    f"cooperative group operation {op_name!r} received a "
                    f"duplicate factory keyword {name!r}."
                )
            seen_factory_kwargs.add(name)
            value = rewrite._resolve_factory_kwarg_value(
                op_name, name, value_var
            )
            if value is _UNRESOLVED:
                raise CoopSinglePhaseRewriteError(
                    f"Failed to evaluate cooperative group operation "
                    f"{op_name!r} factory argument {name!r} as a "
                    f"compile-time constant."
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
                    f"cooperative group operation {op_name!r} runtime "
                    f"argument {name!r} requires {prerequisite!r}."
                )
            runtime_args.append(value_var)
            factory_kwargs[name] = (
                ArgumentBinding.runtime()
                if name in scalar_binding_kwargs
                else True
            )
            seen_factory_kwargs.add(name)
            seen_runtime_factory_kwargs.add(name)
        if runtime_offset_var is not None:
            runtime_args.append(runtime_offset_var)
        rewrite._infer_factory_kwargs_from_payload(
            op_name,
            runtime_args,
            allowed_factory_kwargs,
            seen_factory_kwargs,
            factory_kwargs,
            lowering_plan=lowering_plan,
        )
        if specification.validate_runtime_controls is not None:
            specification.validate_runtime_controls(
                GroupRewriteContext(rewrite),
                op_name=op_name,
                runtime_args=runtime_args,
                factory_kwargs=factory_kwargs,
            )
        rewrite._canonicalize_dim_factory_alias(
            op_name=op_name,
            seen_factory_kwargs=seen_factory_kwargs,
            factory_kwargs=factory_kwargs,
        )
        rewrite._infer_threads_per_block_from_context(
            op_name=op_name,
            allowed_factory_kwargs=allowed_factory_kwargs,
            seen_factory_kwargs=seen_factory_kwargs,
            factory_kwargs=factory_kwargs,
        )
        missing = required_factory_kwargs - seen_factory_kwargs
        if missing:
            if "threads_per_block" in missing:
                if rewrite._can_defer_launch_dim_inference():
                    raise _DeferredCoopRewrite
                if not rewrite._allow_launch_dim_deferral:
                    other_missing = sorted(missing - {"threads_per_block"})
                    other_missing_message = (
                        " Also missing required factory keywords: "
                        f"{', '.join(other_missing)}."
                        if other_missing
                        else ""
                    )
                    raise CoopSinglePhaseRewriteError(
                        f"coop operation '{op_name}' could not infer an "
                        f"exact positive threads_per_block value because "
                        f"{rewrite._launch_dim_inference_failure_detail()}. "
                        f"Use a compile-time constant launch shape or pass "
                        f"explicit threads_per_block.{other_missing_message}"
                    )
            missing_csv = ", ".join(sorted(missing))
            raise CoopSinglePhaseRewriteError(
                f"coop operation '{op_name}' requires explicit factory "
                f"keywords: {missing_csv}."
            )
        if (
            runtime_temp_storage is not None
            and not specification.accepts_temp_storage
        ):
            raise CoopSinglePhaseRewriteError(
                f"cooperative group operation {op_name!r} does not support "
                f"runtime temp_storage."
            )
        if lowering_plan is not None:
            factory_kwargs[_GROUP_LOWERING_PLAN_KWARG] = lowering_plan
        return (
            tuple(runtime_args),
            runtime_temp_storage,
            factory_kwargs,
            tuple(factory_kw_value_vars),
        )


__all__ = ["_ArgumentRewrite"]
