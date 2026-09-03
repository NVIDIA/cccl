# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generic provider argument binding and runtime splitting.

This mixin is composed by CoopSinglePhaseRewrite. Registration and pass
ordering remain in the rewrite orchestrator.
"""

from cuda.coop._core import ArgumentBinding

from ._operations import rewrite_operation
from ._rewrite_support import (
    _UNRESOLVED,
    CoopSinglePhaseRewriteError,
    _DeferredCoopRewrite,
    ir,
)


class _ArgumentRewrite:
    def _validate_and_split_args(
        self, op_name: str, call: ir.Expr, getitem_temp_storage: ir.Var | None
    ) -> tuple[
        tuple[ir.Var, ...],
        ir.Var | None,
        dict[str, object],
        tuple[ir.Var, ...],
    ]:
        spec = rewrite_operation(op_name)
        if spec is None:
            raise CoopSinglePhaseRewriteError(
                f"unsupported Numba-CUDA-MLIR operation {op_name!r}"
            )
        if call.vararg is not None or call.varkwarg is not None:
            raise CoopSinglePhaseRewriteError(
                "coop movement calls do not support *args or **kwargs."
            )
        runtime_arg_count = len(call.args)
        if runtime_arg_count not in spec.runtime_arg_counts:
            expected_csv = ", ".join((str(v) for v in sorted(spec.runtime_arg_counts)))
            raise CoopSinglePhaseRewriteError(
                f"coop movement '{op_name}' expects positional runtime argument count in {{{expected_csv}}}; got {runtime_arg_count}."
            )
        runtime_args = list(call.args)
        factory_kw_value_vars: list[ir.Var] = []
        allowed_factory_kwargs = set(spec.allowed_factory_kwargs)
        required_factory_kwargs = spec.required_factory_kwargs
        seen_factory_kwargs: set[str] = set()
        factory_kwargs: dict[str, object] = {}
        runtime_temp_storage = getitem_temp_storage
        runtime_factory_kwargs = spec.runtime_factory_kwargs
        runtime_factory_kw_prerequisites = dict(spec.runtime_factory_kw_prerequisites)
        scalar_binding_kwargs = spec.scalar_binding_kwargs
        base_runtime_arg_count = min(spec.runtime_arg_counts)
        extra_runtime_arg_count = runtime_arg_count - base_runtime_arg_count
        seen_runtime_factory_kwargs: set[str] = set()
        runtime_factory_kw_vars: dict[str, ir.Var] = {}
        runtime_offset_var = None
        if runtime_factory_kwargs:
            if extra_runtime_arg_count > len(runtime_factory_kwargs):
                raise CoopSinglePhaseRewriteError(
                    f"coop movement '{op_name}' received too many positional partial-tile arguments."
                )
            for index, name in enumerate(
                runtime_factory_kwargs[:extra_runtime_arg_count]
            ):
                factory_kwargs[name] = (
                    ArgumentBinding.runtime() if name in scalar_binding_kwargs else True
                )
                seen_factory_kwargs.add(name)
                seen_runtime_factory_kwargs.add(name)
        for name, value_var in call.kws:
            if name == "temp_storage" and spec.runtime_temp_storage:
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
            if name == spec.runtime_offset_kwarg:
                if runtime_offset_var is not None or name in seen_factory_kwargs:
                    raise CoopSinglePhaseRewriteError(
                        f"Duplicate coop movement '{op_name}' runtime offset."
                    )
                if not isinstance(value_var, ir.Var):
                    raise CoopSinglePhaseRewriteError(
                        f"coop movement {name} must be a variable."
                    )
                value = self._resolve_factory_kwarg_value(name, value_var)
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
                ):
                    raise CoopSinglePhaseRewriteError(
                        f"Duplicate coop movement '{op_name}' runtime argument '{name}'."
                    )
                if not isinstance(value_var, ir.Var):
                    raise CoopSinglePhaseRewriteError(
                        f"coop partial-tile argument '{name}' must be a variable."
                    )
                if name in scalar_binding_kwargs:
                    value = self._resolve_factory_kwarg_value(name, value_var)
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
                    sorted(set(allowed_factory_kwargs) | set(runtime_factory_kwargs))
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
            factory_kwargs[name] = (
                ArgumentBinding.runtime() if name in scalar_binding_kwargs else True
            )
            seen_factory_kwargs.add(name)
            seen_runtime_factory_kwargs.add(name)
        if runtime_offset_var is not None:
            runtime_args.append(runtime_offset_var)
        self._infer_factory_kwargs_from_thread_data(
            op_name,
            runtime_args,
            allowed_factory_kwargs,
            seen_factory_kwargs,
            factory_kwargs,
        )
        if spec.validate_runtime_controls is not None:
            spec.validate_runtime_controls(
                self,
                op_name=op_name,
                runtime_args=runtime_args,
                factory_kwargs=factory_kwargs,
            )
        self._canonicalize_dim_factory_alias(
            op_name=op_name,
            seen_factory_kwargs=seen_factory_kwargs,
            factory_kwargs=factory_kwargs,
        )
        self._infer_threads_per_block_from_context(
            op_name=op_name,
            allowed_factory_kwargs=allowed_factory_kwargs,
            seen_factory_kwargs=seen_factory_kwargs,
            factory_kwargs=factory_kwargs,
        )
        missing = required_factory_kwargs - seen_factory_kwargs
        if missing:
            if "threads_per_block" in missing:
                if self._can_defer_launch_dim_inference():
                    raise _DeferredCoopRewrite
                if not self._allow_launch_dim_deferral:
                    other_missing = sorted(missing - {"threads_per_block"})
                    other_missing_message = (
                        " Also missing required factory keywords: "
                        f"{', '.join(other_missing)}."
                        if other_missing
                        else ""
                    )
                    raise CoopSinglePhaseRewriteError(
                        f"coop operation '{op_name}' could not infer an exact "
                        "positive threads_per_block value because "
                        f"{self._launch_dim_inference_failure_detail()}. Use a "
                        "compile-time constant launch shape or pass explicit "
                        f"threads_per_block.{other_missing_message}"
                    )
            missing_csv = ", ".join(sorted(missing))
            raise CoopSinglePhaseRewriteError(
                f"coop operation '{op_name}' requires explicit factory keywords: {missing_csv}."
            )
        if runtime_temp_storage is not None and not spec.runtime_temp_storage:
            raise CoopSinglePhaseRewriteError(
                f"coop movement '{op_name}' does not support runtime temp_storage."
            )
        return (
            tuple(runtime_args),
            runtime_temp_storage,
            factory_kwargs,
            tuple(factory_kw_value_vars),
        )


__all__ = ["_ArgumentRewrite"]
