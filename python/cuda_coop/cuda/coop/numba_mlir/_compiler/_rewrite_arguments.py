# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generic provider argument binding and runtime splitting.

This mixin is composed by CoopSinglePhaseRewrite. Registration and pass
ordering remain in the rewrite orchestrator.
"""

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
        if op_name == "scan":
            self._finalize_scan_factory_kwargs(
                runtime_arg_count=runtime_arg_count,
                factory_kwargs=factory_kwargs,
            )
        elif op_name in {"radix_rank", "_common_radix_rank"}:
            self._finalize_radix_rank_factory_kwargs(
                op_name=op_name,
                runtime_arg_count=runtime_arg_count,
                seen_factory_kwargs=seen_factory_kwargs,
                factory_kwargs=factory_kwargs,
            )
            self._validate_radix_rank_exclusive_digit_prefix_extent(
                op_name=op_name,
                control_vars=runtime_factory_control_vars,
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


__all__ = ["_ArgumentRewrite"]
