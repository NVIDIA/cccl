# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from functools import cached_property

from numba.core import types

import cuda.coop._rewrite as _core

CoopNode = _core.CoopNode
CoopNodeMixin = _core.CoopNodeMixin
Disposition = _core.Disposition
ir = _core.ir


# =============================================================================
# Warp Reduce/Sum
# =============================================================================
@dataclass
class CoopWarpReduceNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.warp.reduce"
    disposition = Disposition.ONE_SHOT

    def refine_match(self, rewriter):
        launch_config = rewriter.launch_config
        if launch_config is None:
            return False

        runtime_args = []
        runtime_arg_types = []
        runtime_arg_names = []

        expr = self.expr
        expr_args = list(expr.args)
        src = expr_args.pop(0)
        if src is None:
            raise RuntimeError("coop.warp.reduce requires a src argument")

        src_ty = self.typemap[src.name]
        if isinstance(src_ty, types.Array):
            raise RuntimeError("coop.warp.reduce requires a scalar input")
        if not isinstance(src_ty, types.Number):
            raise RuntimeError("coop.warp.reduce requires a numeric input")

        runtime_args.append(src)
        runtime_arg_types.append(src_ty)
        runtime_arg_names.append("src")

        binary_op = self.get_arg_value_safe("binary_op")
        if binary_op is None:
            raise RuntimeError("coop.warp.reduce requires binary_op to be provided")

        threads_in_warp = self.get_arg_value_safe("threads_in_warp")
        threads_in_warp_arg = self.bound.arguments.get("threads_in_warp")
        if threads_in_warp is None and threads_in_warp_arg is not None:
            raise RuntimeError("threads_in_warp must be a compile-time constant")
        if threads_in_warp is None:
            threads_in_warp = 32
        if not isinstance(threads_in_warp, int) or threads_in_warp < 1:
            raise RuntimeError("threads_in_warp must be a positive integer")

        methods = getattr(src_ty, "methods", None)
        if methods is not None and not methods:
            methods = None

        valid_items = self.get_arg_value_safe("valid_items")
        if valid_items is None:
            valid_items = self.bound.arguments.get("valid_items", None)
        valid_items_var = None
        valid_items_type = None
        if valid_items is not None:
            if isinstance(valid_items, ir.Var):
                valid_items_var = valid_items
                valid_items_type = self.typemap[valid_items.name]
            elif isinstance(valid_items, ir.Const):
                valid_items_value = valid_items.value
            else:
                valid_items_value = valid_items

            if valid_items_var is None:
                scope = self.instr.target.scope
                const_name = f"$warp_reduce_valid_items_{self.unique_id}"
                const_var = ir.Var(scope, const_name, self.expr.loc)
                if const_name in self.typemap:
                    raise RuntimeError(
                        f"Variable {const_name} already exists in typemap."
                    )
                const_assign = ir.Assign(
                    value=ir.Const(int(valid_items_value), self.expr.loc),
                    target=const_var,
                    loc=self.expr.loc,
                )
                self.typemap[const_name] = types.int32
                self.valid_items_assign = const_assign
                valid_items_var = const_var
                valid_items_type = types.int32

            runtime_args.append(valid_items_var)
            runtime_arg_types.append(valid_items_type or types.int32)
            runtime_arg_names.append("valid_items")

        temp_storage = self.bound.arguments.get("temp_storage")
        temp_storage_info = None
        if temp_storage is not None:
            if not isinstance(temp_storage, ir.Var):
                raise RuntimeError(
                    "coop.warp.reduce temp_storage must be provided as a variable"
                )
            (_, _, temp_storage_info) = rewriter.bind_temp_storage_runtime_arg(
                node=self,
                temp_storage=temp_storage,
                runtime_args=runtime_args,
                runtime_arg_types=runtime_arg_types,
                runtime_arg_names=runtime_arg_names,
                insert_pos=0,
            )

        self.impl_kwds = {
            "dtype": src_ty,
            "binary_op": binary_op,
            "threads_in_warp": threads_in_warp,
            "valid_items": valid_items,
            "methods": methods,
            "unique_id": self.unique_id,
            "temp_storage": temp_storage,
            "node": self,
        }

        self.return_type = src_ty
        self.runtime_args = runtime_args
        self.runtime_arg_types = runtime_arg_types
        self.runtime_arg_names = runtime_arg_names
        self.temp_storage = temp_storage
        self.temp_storage_info = temp_storage_info

        if (
            self.is_two_phase
            and self.two_phase_instance is not None
            and (valid_items is not None or temp_storage is not None)
        ):
            instance = self.two_phase_instance
            needs_valid_items = (
                valid_items is not None
                and getattr(instance, "valid_items", None) is None
            )
            needs_temp_storage = (
                temp_storage is not None
                and getattr(instance, "temp_storage", None) is None
            )
            if needs_valid_items or needs_temp_storage:
                self.instance = self.instantiate_impl(
                    dtype=src_ty,
                    binary_op=binary_op,
                    threads_in_warp=threads_in_warp,
                    valid_items=valid_items,
                    methods=methods,
                    unique_id=self.unique_id,
                    temp_storage=temp_storage,
                    node=self,
                )

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        instrs = [rd.g_assign]
        valid_items_assign = getattr(self, "valid_items_assign", None)
        if valid_items_assign is not None:
            instrs.append(valid_items_assign)
        instrs.append(rd.new_assign)
        if self.temp_storage_info is not None and self.temp_storage_info.auto_sync:
            instrs.extend(
                rewriter.emit_syncthreads_call(self.instr.target.scope, self.expr.loc)
            )
        return tuple(instrs)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()


@dataclass
class CoopWarpSumNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.warp.sum"
    disposition = Disposition.ONE_SHOT

    def refine_match(self, rewriter):
        launch_config = rewriter.launch_config
        if launch_config is None:
            return False

        runtime_args = []
        runtime_arg_types = []
        runtime_arg_names = []

        expr = self.expr
        expr_args = list(expr.args)
        src = expr_args.pop(0)
        if src is None:
            raise RuntimeError("coop.warp.sum requires a src argument")

        src_ty = self.typemap[src.name]
        if isinstance(src_ty, types.Array):
            raise RuntimeError("coop.warp.sum requires a scalar input")
        if not isinstance(src_ty, types.Number):
            raise RuntimeError("coop.warp.sum requires a numeric input")

        runtime_args.append(src)
        runtime_arg_types.append(src_ty)
        runtime_arg_names.append("src")

        threads_in_warp = self.get_arg_value_safe("threads_in_warp")
        threads_in_warp_arg = self.bound.arguments.get("threads_in_warp")
        if threads_in_warp is None and threads_in_warp_arg is not None:
            raise RuntimeError("threads_in_warp must be a compile-time constant")
        if threads_in_warp is None:
            threads_in_warp = 32
        if not isinstance(threads_in_warp, int) or threads_in_warp < 1:
            raise RuntimeError("threads_in_warp must be a positive integer")

        valid_items = self.get_arg_value_safe("valid_items")
        if valid_items is None:
            valid_items = self.bound.arguments.get("valid_items", None)
        valid_items_var = None
        valid_items_type = None
        if valid_items is not None:
            if isinstance(valid_items, ir.Var):
                valid_items_var = valid_items
                valid_items_type = self.typemap[valid_items.name]
            elif isinstance(valid_items, ir.Const):
                valid_items_value = valid_items.value
            else:
                valid_items_value = valid_items

            if valid_items_var is None:
                scope = self.instr.target.scope
                const_name = f"$warp_sum_valid_items_{self.unique_id}"
                const_var = ir.Var(scope, const_name, self.expr.loc)
                if const_name in self.typemap:
                    raise RuntimeError(
                        f"Variable {const_name} already exists in typemap."
                    )
                const_assign = ir.Assign(
                    value=ir.Const(int(valid_items_value), self.expr.loc),
                    target=const_var,
                    loc=self.expr.loc,
                )
                self.typemap[const_name] = types.int32
                self.valid_items_assign = const_assign
                valid_items_var = const_var
                valid_items_type = types.int32

            runtime_args.append(valid_items_var)
            runtime_arg_types.append(valid_items_type or types.int32)
            runtime_arg_names.append("valid_items")

        temp_storage = self.bound.arguments.get("temp_storage")
        temp_storage_info = None
        if temp_storage is not None:
            if not isinstance(temp_storage, ir.Var):
                raise RuntimeError(
                    "coop.warp.sum temp_storage must be provided as a variable"
                )
            (_, _, temp_storage_info) = rewriter.bind_temp_storage_runtime_arg(
                node=self,
                temp_storage=temp_storage,
                runtime_args=runtime_args,
                runtime_arg_types=runtime_arg_types,
                runtime_arg_names=runtime_arg_names,
                insert_pos=0,
            )

        self.impl_kwds = {
            "dtype": src_ty,
            "threads_in_warp": threads_in_warp,
            "valid_items": valid_items,
            "unique_id": self.unique_id,
            "temp_storage": temp_storage,
        }

        self.return_type = src_ty
        self.runtime_args = runtime_args
        self.runtime_arg_types = runtime_arg_types
        self.runtime_arg_names = runtime_arg_names
        self.temp_storage = temp_storage
        self.temp_storage_info = temp_storage_info

        if (
            self.is_two_phase
            and self.two_phase_instance is not None
            and (valid_items is not None or temp_storage is not None)
        ):
            instance = self.two_phase_instance
            needs_valid_items = (
                valid_items is not None
                and getattr(instance, "valid_items", None) is None
            )
            needs_temp_storage = (
                temp_storage is not None
                and getattr(instance, "temp_storage", None) is None
            )
            if needs_valid_items or needs_temp_storage:
                self.instance = self.instantiate_impl(
                    dtype=src_ty,
                    threads_in_warp=threads_in_warp,
                    valid_items=valid_items,
                    unique_id=self.unique_id,
                    temp_storage=temp_storage,
                )

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        instrs = [rd.g_assign]
        valid_items_assign = getattr(self, "valid_items_assign", None)
        if valid_items_assign is not None:
            instrs.append(valid_items_assign)
        instrs.append(rd.new_assign)
        if self.temp_storage_info is not None and self.temp_storage_info.auto_sync:
            instrs.extend(
                rewriter.emit_syncthreads_call(self.instr.target.scope, self.expr.loc)
            )
        return tuple(instrs)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()
