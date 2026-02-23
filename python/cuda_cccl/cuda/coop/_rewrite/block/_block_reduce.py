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
# Block reduce/sum
# =============================================================================
@dataclass
class CoopBlockReduceNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.block.reduce"
    disposition = Disposition.ONE_SHOT

    def refine_match(self, rewriter):
        launch_config = rewriter.launch_config
        if launch_config is None:
            return False

        self.threads_per_block = launch_config.blockdim

        runtime_args = []
        runtime_arg_types = []
        runtime_arg_names = []

        expr = self.expr
        expr_args = list(expr.args)

        src = expr_args.pop(0)
        if src is None:
            raise RuntimeError("coop.block.reduce requires a src argument")

        src_ty = self.typemap[src.name]
        try:
            from ..._decls import ThreadDataType
        except Exception:
            ThreadDataType = None

        src_is_thread = ThreadDataType is not None and isinstance(
            src_ty, ThreadDataType
        )
        src_is_array = isinstance(src_ty, types.Array) or src_is_thread
        src_info = rewriter.get_thread_data_info(src) if src_is_thread else None
        if src_is_thread:
            dtype = src_info.dtype
        elif src_is_array:
            dtype = src_ty.dtype
        else:
            dtype = src_ty

        methods = getattr(dtype, "methods", None)
        if methods is not None and not methods:
            methods = None

        runtime_args.append(src)
        runtime_arg_types.append(
            types.Array(dtype, 1, "C") if src_is_thread else src_ty
        )
        runtime_arg_names.append("src")

        items_per_thread = self.get_arg_value_safe("items_per_thread")
        if src_is_thread:
            src_items_per_thread = src_info.items_per_thread
            if items_per_thread is None:
                items_per_thread = src_items_per_thread
            elif items_per_thread != src_items_per_thread:
                raise RuntimeError(
                    "coop.block.reduce items_per_thread must match the array "
                    f"shape ({src_items_per_thread}); got {items_per_thread}"
                )
        if items_per_thread is None:
            items_per_thread = 1
        if items_per_thread < 1:
            raise RuntimeError("items_per_thread must be >= 1")
        if items_per_thread > 1 and not src_is_array:
            raise RuntimeError(
                "coop.block.reduce requires array inputs when items_per_thread > 1"
            )

        binary_op = self.get_arg_value_safe("binary_op")
        if binary_op is None:
            raise RuntimeError("coop.block.reduce requires binary_op to be provided")

        bound = self.bound.arguments
        num_valid = bound.get("num_valid")
        num_valid_var = None
        num_valid_type = None
        if num_valid is not None:
            if src_is_array:
                raise RuntimeError(
                    "coop.block.reduce does not support num_valid for array inputs"
                )
            if isinstance(num_valid, ir.Var):
                num_valid_var = num_valid
                num_valid_type = types.int32
            elif isinstance(num_valid, ir.Const):
                num_valid_value = num_valid.value
            else:
                num_valid_value = num_valid

            if num_valid_var is None:
                scope = self.instr.target.scope
                const_name = f"$block_reduce_num_valid_{self.unique_id}"
                const_var = ir.Var(scope, const_name, expr.loc)
                if const_name in self.typemap:
                    raise RuntimeError(
                        f"Variable {const_name} already exists in typemap."
                    )
                const_assign = ir.Assign(
                    value=ir.Const(int(num_valid_value), expr.loc),
                    target=const_var,
                    loc=expr.loc,
                )
                self.typemap[const_name] = types.int32
                self.num_valid_assign = const_assign
                num_valid_var = const_var
                num_valid_type = types.int32

            runtime_args.append(num_valid_var)
            runtime_arg_types.append(num_valid_type or types.int32)
            runtime_arg_names.append("num_valid")

        algorithm = self.get_arg_value_safe("algorithm")
        if algorithm is None:
            algorithm = "warp_reductions"

        temp_storage = bound.get("temp_storage")
        temp_storage_info = None
        if temp_storage is not None:
            if not isinstance(temp_storage, ir.Var):
                raise RuntimeError(
                    "coop.block.reduce temp_storage must be provided as a variable"
                )
            (temp_storage, _, temp_storage_info) = (
                rewriter.bind_temp_storage_runtime_arg(
                    node=self,
                    temp_storage=temp_storage,
                    runtime_args=runtime_args,
                    runtime_arg_types=runtime_arg_types,
                    runtime_arg_names=runtime_arg_names,
                    insert_pos=0,
                )
            )

        self.dtype = dtype
        self.items_per_thread = items_per_thread
        self.algorithm = algorithm
        self.temp_storage = temp_storage
        self.temp_storage_info = temp_storage_info
        self.use_array_inputs = src_is_array
        self.methods = methods
        self.num_valid = num_valid

        self.impl_kwds = {
            "dtype": dtype,
            "threads_per_block": self.threads_per_block,
            "binary_op": binary_op,
            "items_per_thread": items_per_thread,
            "algorithm": algorithm,
            "methods": methods,
            "unique_id": self.unique_id,
            "temp_storage": temp_storage,
            "num_valid": num_valid,
            "use_array_inputs": src_is_array,
            "node": self,
        }

        self.return_type = dtype
        self.runtime_args = runtime_args
        self.runtime_arg_types = runtime_arg_types
        self.runtime_arg_names = runtime_arg_names

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        instrs = [rd.g_assign]
        num_valid_assign = getattr(self, "num_valid_assign", None)
        if num_valid_assign is not None:
            instrs.append(num_valid_assign)
        instrs.append(rd.new_assign)
        if self.temp_storage_info is not None and self.temp_storage_info.auto_sync:
            instrs.extend(
                rewriter.emit_syncthreads_call(self.instr.target.scope, self.expr.loc)
            )
        return instrs

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()


@dataclass
class CoopBlockSumNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.block.sum"
    disposition = Disposition.ONE_SHOT

    def refine_match(self, rewriter):
        launch_config = rewriter.launch_config
        if launch_config is None:
            return False

        self.threads_per_block = launch_config.blockdim

        runtime_args = []
        runtime_arg_types = []
        runtime_arg_names = []

        expr = self.expr
        expr_args = list(expr.args)

        src = expr_args.pop(0)
        if src is None:
            raise RuntimeError("coop.block.sum requires a src argument")

        src_ty = self.typemap[src.name]
        try:
            from ..._decls import ThreadDataType
        except Exception:
            ThreadDataType = None

        src_is_thread = ThreadDataType is not None and isinstance(
            src_ty, ThreadDataType
        )
        src_is_array = isinstance(src_ty, types.Array) or src_is_thread
        src_info = rewriter.get_thread_data_info(src) if src_is_thread else None
        if src_is_thread:
            dtype = src_info.dtype
        elif src_is_array:
            dtype = src_ty.dtype
        else:
            dtype = src_ty

        methods = getattr(dtype, "methods", None)
        if methods is not None and not methods:
            methods = None

        runtime_args.append(src)
        runtime_arg_types.append(
            types.Array(dtype, 1, "C") if src_is_thread else src_ty
        )
        runtime_arg_names.append("src")

        items_per_thread = self.get_arg_value_safe("items_per_thread")
        if src_is_thread:
            src_items_per_thread = src_info.items_per_thread
            if items_per_thread is None:
                items_per_thread = src_items_per_thread
            elif items_per_thread != src_items_per_thread:
                raise RuntimeError(
                    "coop.block.sum items_per_thread must match the array "
                    f"shape ({src_items_per_thread}); got {items_per_thread}"
                )
        if items_per_thread is None:
            items_per_thread = 1
        if items_per_thread < 1:
            raise RuntimeError("items_per_thread must be >= 1")
        if items_per_thread > 1 and not src_is_array:
            raise RuntimeError(
                "coop.block.sum requires array inputs when items_per_thread > 1"
            )

        bound = self.bound.arguments
        num_valid = bound.get("num_valid")
        num_valid_var = None
        num_valid_type = None
        if num_valid is not None:
            if src_is_array:
                raise RuntimeError(
                    "coop.block.sum does not support num_valid for array inputs"
                )
            if isinstance(num_valid, ir.Var):
                num_valid_var = num_valid
                num_valid_type = types.int32
            elif isinstance(num_valid, ir.Const):
                num_valid_value = num_valid.value
            else:
                num_valid_value = num_valid

            if num_valid_var is None:
                scope = self.instr.target.scope
                const_name = f"$block_sum_num_valid_{self.unique_id}"
                const_var = ir.Var(scope, const_name, expr.loc)
                if const_name in self.typemap:
                    raise RuntimeError(
                        f"Variable {const_name} already exists in typemap."
                    )
                const_assign = ir.Assign(
                    value=ir.Const(int(num_valid_value), expr.loc),
                    target=const_var,
                    loc=expr.loc,
                )
                self.typemap[const_name] = types.int32
                self.num_valid_assign = const_assign
                num_valid_var = const_var
                num_valid_type = types.int32

            runtime_args.append(num_valid_var)
            runtime_arg_types.append(num_valid_type or types.int32)
            runtime_arg_names.append("num_valid")

        algorithm = self.get_arg_value_safe("algorithm")
        if algorithm is None:
            algorithm = "warp_reductions"

        temp_storage = bound.get("temp_storage")
        temp_storage_info = None
        if temp_storage is not None:
            if not isinstance(temp_storage, ir.Var):
                raise RuntimeError(
                    "coop.block.sum temp_storage must be provided as a variable"
                )
            (temp_storage, _, temp_storage_info) = (
                rewriter.bind_temp_storage_runtime_arg(
                    node=self,
                    temp_storage=temp_storage,
                    runtime_args=runtime_args,
                    runtime_arg_types=runtime_arg_types,
                    runtime_arg_names=runtime_arg_names,
                    insert_pos=0,
                )
            )

        self.dtype = dtype
        self.items_per_thread = items_per_thread
        self.algorithm = algorithm
        self.temp_storage = temp_storage
        self.temp_storage_info = temp_storage_info
        self.use_array_inputs = src_is_array
        self.methods = methods
        self.num_valid = num_valid

        self.impl_kwds = {
            "dtype": dtype,
            "threads_per_block": self.threads_per_block,
            "items_per_thread": items_per_thread,
            "algorithm": algorithm,
            "methods": methods,
            "unique_id": self.unique_id,
            "temp_storage": temp_storage,
            "num_valid": num_valid,
            "use_array_inputs": src_is_array,
            "node": self,
        }

        self.return_type = dtype
        self.runtime_args = runtime_args
        self.runtime_arg_types = runtime_arg_types
        self.runtime_arg_names = runtime_arg_names

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        instrs = [rd.g_assign]
        num_valid_assign = getattr(self, "num_valid_assign", None)
        if num_valid_assign is not None:
            instrs.append(num_valid_assign)
        instrs.append(rd.new_assign)
        if self.temp_storage_info is not None and self.temp_storage_info.auto_sync:
            instrs.extend(
                rewriter.emit_syncthreads_call(self.instr.target.scope, self.expr.loc)
            )
        return instrs

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()
