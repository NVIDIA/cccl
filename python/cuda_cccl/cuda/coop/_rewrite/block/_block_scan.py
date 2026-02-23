# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
from functools import cached_property

from numba.core import types

import cuda.coop._rewrite as _core

ArrayCallDefinition = _core.ArrayCallDefinition
CoopNode = _core.CoopNode
CoopNodeMixin = _core.CoopNodeMixin
Disposition = _core.Disposition
ir = _core.ir


# =============================================================================
# Block scan
# =============================================================================
class CoopBlockScanNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.block.scan"
    disposition = Disposition.ONE_SHOT
    threads_per_block: int = None
    codegen_return_type2: types.Type = None
    codegen_return_type3: types.Type = types.void

    def refine_match(self, rewriter):
        launch_config = rewriter.launch_config
        if launch_config is None:
            return False

        self.threads_per_block = launch_config.blockdim
        instance = self.two_phase_instance if self.is_two_phase else None
        if instance is not None:
            self.instance = instance

        runtime_args = []
        runtime_arg_types = []
        runtime_arg_names = []

        expr = self.expr
        items_per_thread = None
        algorithm = None
        initial_value = None
        mode = None
        scan_op = None
        block_prefix_callback_op = None
        block_aggregate = None
        temp_storage = None

        bound = self.bound.arguments

        src = bound.get("src")
        dst = bound.get("dst")
        block_aggregate = bound.get("block_aggregate")
        if block_aggregate is not None and isinstance(block_aggregate, types.NoneType):
            block_aggregate = None

        assert src is not None, src

        src_ty = self.typemap[src.name]
        dst_ty = self.typemap[dst.name] if dst is not None else None

        try:
            from ..._decls import ThreadDataType
        except Exception:
            ThreadDataType = None

        src_is_thread = ThreadDataType is not None and isinstance(
            src_ty, ThreadDataType
        )
        dst_is_thread = ThreadDataType is not None and isinstance(
            dst_ty, ThreadDataType
        )

        src_is_array = isinstance(src_ty, types.Array) or src_is_thread
        dst_is_array = dst is not None and (
            isinstance(dst_ty, types.Array) or dst_is_thread
        )
        src_is_scalar = isinstance(src_ty, types.Number)
        dst_is_scalar = dst is not None and isinstance(dst_ty, types.Number)

        if dst is None:
            if not src_is_scalar:
                raise RuntimeError(
                    "coop.block.scan requires array dst when src is an array"
                )
            use_array_inputs = False
            dtype = src_ty
        else:
            if src_is_scalar or dst_is_scalar:
                raise RuntimeError(
                    "coop.block.scan scalar inputs must omit dst in single-phase"
                )
            if src_is_array != dst_is_array:
                raise RuntimeError(
                    "coop.block.scan requires src and dst to be both arrays"
                )
            use_array_inputs = src_is_array and dst_is_array

            if src_is_thread:
                dtype = rewriter.get_thread_data_info(src).dtype
            else:
                dtype = src_ty.dtype

            if dst_is_thread:
                dst_dtype = rewriter.get_thread_data_info(dst).dtype
            else:
                dst_dtype = dst_ty.dtype

            if dst_dtype != dtype:
                raise RuntimeError(
                    "coop.block.scan requires src and dst to have the same dtype"
                )

        methods = getattr(dtype, "methods", None)
        if methods is not None and not methods:
            methods = None

        if dst is not None:
            runtime_args.append(src)
            runtime_arg_types.append(src_ty)
            runtime_arg_names.append("src")
            runtime_args.append(dst)
            runtime_arg_types.append(dst_ty)
            runtime_arg_names.append("dst")
        else:
            runtime_args.append(src)
            runtime_arg_types.append(src_ty)
            runtime_arg_names.append("src")

        if ThreadDataType is not None and use_array_inputs:
            array_ty = types.Array(dtype, 1, "C")
            if src_is_thread:
                runtime_arg_types[0] = array_ty
            if dst_is_thread:
                runtime_arg_types[1] = array_ty

        items_per_thread = self.get_arg_value_safe("items_per_thread")
        if isinstance(items_per_thread, types.IntegerLiteral):
            items_per_thread = items_per_thread.literal_value
        if use_array_inputs:

            def _infer_items_per_thread(var, is_thread):
                if is_thread:
                    return rewriter.get_thread_data_info(var).items_per_thread
                root = rewriter.get_root_def(var)
                leaf = root.leaf_constructor_call
                if not isinstance(leaf, ArrayCallDefinition):
                    raise RuntimeError(
                        f"Expected array constructor call for {var!r}, got {leaf!r}"
                    )
                return leaf.shape

            src_items = _infer_items_per_thread(src, src_is_thread)
            dst_items = _infer_items_per_thread(dst, dst_is_thread)
            if src_items != dst_items:
                raise RuntimeError(
                    "coop.block.scan requires src and dst to have the same "
                    "items_per_thread"
                )
            if isinstance(src_items, types.IntegerLiteral):
                src_items = src_items.literal_value
            if isinstance(dst_items, types.IntegerLiteral):
                dst_items = dst_items.literal_value
            if items_per_thread is None:
                items_per_thread = src_items
            elif items_per_thread != src_items:
                raise RuntimeError(
                    "coop.block.scan items_per_thread must match the "
                    f"array shape ({src_items}); got {items_per_thread}"
                )

        else:
            if items_per_thread is None:
                items_per_thread = 1
            elif items_per_thread != 1:
                raise RuntimeError(
                    "coop.block.scan requires items_per_thread == 1 for scalar inputs"
                )

        if instance is not None:
            instance_items = getattr(instance, "items_per_thread", None)
            if instance_items is not None:
                if items_per_thread is None:
                    items_per_thread = instance_items
                elif items_per_thread != instance_items:
                    raise RuntimeError(
                        "coop.block.scan items_per_thread must match the "
                        f"two-phase instance ({instance_items}); got "
                        f"{items_per_thread}"
                    )

        mode = self.get_arg_value_safe("mode")
        if mode is None and instance is not None:
            mode = getattr(instance, "mode", None)
        if mode is None:
            mode = "exclusive"

        scan_op = self.get_arg_value_safe("scan_op")
        if scan_op is None and instance is not None:
            scan_op = getattr(instance, "scan_op", None)
        if scan_op is None:
            scan_op = "+"

        forced_mode = getattr(self.template, "forced_mode", None)
        if forced_mode is not None:
            mode = forced_mode

        forced_scan_op = getattr(self.template, "forced_scan_op", None)
        if forced_scan_op is not None:
            scan_op = forced_scan_op

        initial_value = bound.get("initial_value")
        initial_value_var = None
        initial_value_value = None
        initial_value_type = None
        instance_initial_value = None
        initial_value_is_none_type = isinstance(initial_value, types.NoneType)
        if initial_value_is_none_type or initial_value is None:
            initial_value = None
        if initial_value is not None:
            if isinstance(initial_value, ir.Var):
                initial_value_var = initial_value
                initial_value_type = self.typemap[initial_value.name]
            elif isinstance(initial_value, ir.Const):
                initial_value_value = initial_value.value
            else:
                initial_value_value = initial_value
        elif not initial_value_is_none_type and instance is not None:
            instance_initial_value = getattr(instance, "initial_value", None)
            if instance_initial_value is not None:
                initial_value_value = instance_initial_value

        from ..._scan_op import ScanOp
        from ...block._block_scan import _validate_initial_value

        scan_op_obj = scan_op if isinstance(scan_op, ScanOp) else ScanOp(scan_op)

        block_prefix_callback_op = bound.get("block_prefix_callback_op")
        if block_prefix_callback_op is not None:
            if not isinstance(block_prefix_callback_op, ir.Var):
                raise RuntimeError(
                    f"Expected a variable for block_prefix_callback_op, "
                    f"got {block_prefix_callback_op!r}"
                )

            block_prefix_callback_op_var = block_prefix_callback_op
            prefix_state_ty = self.typemap[block_prefix_callback_op.name]
            runtime_prefix_ty = prefix_state_ty
            prefix_op_root_def = rewriter.get_root_def(block_prefix_callback_op)
            if prefix_op_root_def is None:
                raise RuntimeError(
                    "Expected a root definition for "
                    "{block_prefix_callback_op!r}, got None"
                )
            instance = prefix_op_root_def.instance
            if instance is None:
                raise RuntimeError(
                    f"Expected an instance for {block_prefix_callback_op!r}, got None"
                )
            if instance.__class__.__name__ == "module":
                # Assume we've got the array-style invocation.
                call_def = prefix_op_root_def.leaf_constructor_call
                if not isinstance(call_def, ArrayCallDefinition):
                    raise RuntimeError(
                        f"Expected a leaf array call definition for "
                        f"{block_prefix_callback_op!r}, got {call_def!r}"
                    )
                assert isinstance(prefix_state_ty, types.Array)
                runtime_prefix_ty = prefix_state_ty
                prefix_state_ty = prefix_state_ty.dtype
                modulename = prefix_state_ty.__module__
                module = sys.modules[modulename]
                instance = getattr(module, prefix_state_ty.name)

            op = instance

            from ..._types import StatefulFunction

            callback_name = f"block_scan_{self.unique_id}_callback"
            if callback_name in self.typemap:
                raise RuntimeError(
                    f"Callback name {callback_name} already exists in typemap."
                )
            self.typemap[callback_name] = runtime_prefix_ty

            block_prefix_callback_op = StatefulFunction(
                op,
                prefix_state_ty,
                name=callback_name,
            )
            runtime_args.append(block_prefix_callback_op_var)
            runtime_arg_types.append(runtime_prefix_ty)
            runtime_arg_names.append("block_prefix_callback_op")
        if block_aggregate is not None:
            if block_prefix_callback_op is not None:
                raise RuntimeError(
                    "coop.block.scan does not support block_aggregate when "
                    "block_prefix_callback_op is provided"
                )
            if dst is None:
                raise RuntimeError(
                    "coop.block.scan block_aggregate requires a dst array when using scalar inputs"
                )
            if isinstance(block_aggregate, ir.Var):
                block_aggregate_ty = self.typemap[block_aggregate.name]
                if not isinstance(block_aggregate_ty, types.Array):
                    raise RuntimeError(
                        "coop.block.scan block_aggregate must be a device array"
                    )
                expected_dtype = dtype
                if block_aggregate_ty.dtype != expected_dtype:
                    raise RuntimeError(
                        "coop.block.scan requires block_aggregate to have the same "
                        "dtype as the input"
                    )
            else:
                raise RuntimeError(
                    "coop.block.scan block_aggregate must be provided as a variable"
                )

        if scan_op_obj.is_sum:
            explicit_initial = initial_value_var is not None
            if explicit_initial and not initial_value_is_none_type:
                raise RuntimeError(
                    "initial_value is not supported for inclusive and exclusive sums"
                )
        else:
            if initial_value_var is None and initial_value_value is None:
                try:
                    initial_value_value = _validate_initial_value(
                        None,
                        dtype,
                        items_per_thread,
                        mode,
                        scan_op_obj,
                        block_prefix_callback_op,
                    )
                except ValueError as e:
                    raise RuntimeError(str(e)) from e

        include_initial_value = (
            not scan_op_obj.is_sum
            and block_prefix_callback_op is None
            and (initial_value_var is not None or initial_value_value is not None)
            and (use_array_inputs or mode == "exclusive")
        )
        if include_initial_value:
            if initial_value_var is not None:
                runtime_args.append(initial_value_var)
                runtime_arg_types.append(initial_value_type)
            else:
                from numba.np.numpy_support import as_dtype

                const_value = initial_value_value
                try:
                    const_value = as_dtype(dtype).type(initial_value_value)
                except Exception:
                    pass
                scope = self.instr.target.scope
                const_name = f"$block_scan_init_{self.unique_id}"
                const_var = ir.Var(scope, const_name, expr.loc)
                if const_name in self.typemap:
                    raise RuntimeError(
                        f"Variable {const_name} already exists in typemap."
                    )
                const_assign = ir.Assign(
                    value=ir.Const(const_value, expr.loc),
                    target=const_var,
                    loc=expr.loc,
                )
                if isinstance(dtype, types.Integer):
                    self.typemap[const_name] = types.IntegerLiteral(int(const_value))
                elif isinstance(dtype, types.Boolean):
                    self.typemap[const_name] = types.BooleanLiteral(bool(const_value))
                else:
                    self.typemap[const_name] = dtype
                self.initial_value_assign = const_assign
                runtime_args.append(const_var)
                runtime_arg_types.append(dtype)
            runtime_arg_names.append("initial_value")

        if block_aggregate is not None:
            runtime_args.append(block_aggregate)
            runtime_arg_types.append(block_aggregate_ty)
            runtime_arg_names.append("block_aggregate")

        if scan_op_obj.is_sum:
            initial_value_for_impl = None
        else:
            initial_value_for_impl = (
                initial_value_var
                if initial_value_var is not None
                else initial_value_value
            )

        algorithm = self.get_arg_value_safe("algorithm")
        if algorithm is None and instance is not None:
            algorithm = getattr(instance, "algorithm_id", None)
        temp_storage = bound.get("temp_storage")
        temp_storage_info = None
        if temp_storage is not None:
            if not isinstance(temp_storage, ir.Var):
                raise RuntimeError(
                    "coop.block.scan temp_storage must be provided as a variable"
                )
            (_, _, temp_storage_info) = rewriter.bind_temp_storage_runtime_arg(
                node=self,
                temp_storage=temp_storage,
                runtime_args=runtime_args,
                runtime_arg_types=runtime_arg_types,
                runtime_arg_names=runtime_arg_names,
                insert_pos=0,
            )

        self.runtime_args = runtime_args
        self.runtime_arg_types = runtime_arg_types
        self.runtime_arg_names = runtime_arg_names

        self.src = src
        self.dst = dst
        if use_array_inputs:
            self.dtype = dtype
        else:
            self.dtype = src_ty
        self.items_per_thread = items_per_thread
        self.mode = mode
        self.scan_op = scan_op
        self.initial_value = initial_value_for_impl
        self.block_prefix_callback_op = block_prefix_callback_op
        self.block_aggregate = block_aggregate
        self.algorithm = algorithm
        self.temp_storage = temp_storage
        self.temp_storage_info = temp_storage_info
        self.use_array_inputs = use_array_inputs
        self.methods = methods

        self.impl_kwds = {
            "dtype": self.dtype,
            "threads_per_block": self.threads_per_block,
            "items_per_thread": items_per_thread,
            "initial_value": initial_value_for_impl,
            "mode": mode,
            "scan_op": scan_op,
            "block_prefix_callback_op": block_prefix_callback_op,
            "block_aggregate": block_aggregate,
            "algorithm": algorithm,
            "unique_id": self.unique_id,
            "temp_storage": temp_storage,
            "use_array_inputs": use_array_inputs,
            "methods": methods,
        }

        if not use_array_inputs:
            self.return_type = dtype
        else:
            self.return_type = types.void

        if (
            self.is_two_phase
            and self.two_phase_instance is not None
            and block_aggregate is not None
        ):
            instance = self.two_phase_instance
            if getattr(instance, "block_aggregate", None) is None:
                self.instance = self.instantiate_impl(**self.impl_kwds)

        if self.is_two_phase and self.two_phase_instance is not None:
            instance = self.two_phase_instance
            instance_use_array = getattr(instance, "use_array_inputs", None)
            if (
                instance_use_array is not None
                and instance_use_array != use_array_inputs
            ):
                self.instance = self.instantiate_impl(**self.impl_kwds)

        if not use_array_inputs:
            if self.is_two_phase and self.two_phase_instance is not None:
                instance = self.two_phase_instance
                if getattr(instance, "return_type", None) != self.return_type:
                    self.instance = self.instantiate_impl(**self.impl_kwds)

        return

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        return (rd.g_assign, rd.new_assign)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()
