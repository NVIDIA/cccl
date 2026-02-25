# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from functools import cached_property

from numba.core import types

import cuda.coop._rewrite as _core

ArrayCallDefinition = _core.ArrayCallDefinition
CoopNode = _core.CoopNode
CoopNodeMixin = _core.CoopNodeMixin
Disposition = _core.Disposition
ThreadDataCallDefinition = _core.ThreadDataCallDefinition
get_thread_data_type = _core.get_thread_data_type
ir = _core.ir


# =============================================================================
# Warp Load / Store
# =============================================================================
class CoopWarpLoadStoreNode(CoopNode):
    threads_in_warp = None
    disposition = Disposition.ONE_SHOT

    def refine_match(self, rewriter):
        expr = self.expr
        dtype = None
        items_per_thread = None
        algorithm_id = None
        expr_args = self.expr_args = list(expr.args)
        if self.is_load:
            src = expr_args.pop(0)
            dst = expr_args.pop(0)
            runtime_args = [src, dst]
            runtime_arg_types = [
                self.typemap[src.name],
                self.typemap[dst.name],
            ]
            runtime_arg_names = ["src", "dst"]
            items_per_thread_array_var = dst
        else:
            dst = expr_args.pop(0)
            src = expr_args.pop(0)
            runtime_args = [dst, src]
            runtime_arg_types = [
                self.typemap[dst.name],
                self.typemap[src.name],
            ]
            runtime_arg_names = ["dst", "src"]
            items_per_thread_array_var = src

        thread_data_type = get_thread_data_type()
        src_ty = self.typemap[src.name]
        dst_ty = self.typemap[dst.name]
        src_is_thread = isinstance(src_ty, thread_data_type)
        dst_is_thread = isinstance(dst_ty, thread_data_type)

        array_root = rewriter.get_root_def(items_per_thread_array_var)
        array_leaf = array_root.leaf_constructor_call
        if isinstance(array_leaf, ArrayCallDefinition):
            items_per_thread = array_leaf.shape
        elif isinstance(array_leaf, ThreadDataCallDefinition):
            items_per_thread = rewriter.get_thread_data_info(
                items_per_thread_array_var
            ).items_per_thread
        else:
            raise RuntimeError(
                "Expected leaf constructor call to be an ArrayCallDefinition or "
                f"ThreadDataCallDefinition, but got {array_leaf!r} for "
                f"{items_per_thread_array_var!r}"
            )
        if isinstance(items_per_thread, types.IntegerLiteral):
            items_per_thread = items_per_thread.literal_value
        if not isinstance(items_per_thread, int):
            raise RuntimeError(
                f"Expected items_per_thread to be an int, got {items_per_thread!r}"
            )

        items_per_thread_kwarg = self.get_arg_value_safe("items_per_thread")
        if (
            items_per_thread_kwarg is not None
            and items_per_thread_kwarg != items_per_thread
        ):
            raise RuntimeError(
                f"Expected items_per_thread to be {items_per_thread}, "
                f"but got {items_per_thread_kwarg} for {self!r}"
            )

        if src_is_thread and dst_is_thread:
            raise RuntimeError(
                "coop.warp.load/store requires at least one device array to infer "
                "dtype when using ThreadData"
            )

        if src_is_thread:
            if not isinstance(dst_ty, types.Array):
                raise RuntimeError(
                    "coop.warp.store requires destination array when source is "
                    "ThreadData"
                )
            dtype = dst_ty.dtype
            thread_info = rewriter.get_thread_data_info(src)
            if thread_info.dtype != dtype:
                raise RuntimeError(
                    "ThreadData dtype does not match destination array dtype"
                )
        elif dst_is_thread:
            if not isinstance(src_ty, types.Array):
                raise RuntimeError(
                    "coop.warp.load requires source array when destination is "
                    "ThreadData"
                )
            dtype = src_ty.dtype
            thread_info = rewriter.get_thread_data_info(dst)
            if thread_info.dtype != dtype:
                raise RuntimeError("ThreadData dtype does not match source array dtype")
        else:
            if not isinstance(src_ty, types.Array) or not isinstance(
                dst_ty, types.Array
            ):
                raise RuntimeError(
                    "coop.warp.load/store requires array inputs in single-phase"
                )
            if src_ty.dtype != dst_ty.dtype:
                raise RuntimeError(
                    "coop.warp.load/store requires src and dst to have the same dtype"
                )
            dtype = src_ty.dtype

        array_ty = types.Array(dtype, 1, "C")
        if src_is_thread:
            runtime_arg_types[0] = array_ty
        if dst_is_thread:
            runtime_arg_types[1] = array_ty

        methods = getattr(dtype, "methods", None)
        if methods is not None and not methods:
            methods = None

        threads_in_warp = self.get_arg_value_safe("threads_in_warp")
        threads_in_warp_arg = self.bound.arguments.get("threads_in_warp")
        if threads_in_warp is None and threads_in_warp_arg is not None:
            raise RuntimeError("threads_in_warp must be a compile-time constant")
        if threads_in_warp is None:
            threads_in_warp = 32
        if not isinstance(threads_in_warp, int) or threads_in_warp < 1:
            raise RuntimeError("threads_in_warp must be a positive integer")
        self.threads_in_warp = threads_in_warp

        algorithm_id = self.get_arg_value_safe("algorithm")
        if algorithm_id is None:
            algorithm_var = self.bound.arguments.get("algorithm")
            if isinstance(algorithm_var, ir.Var):
                algorithm_ty = self.typemap.get(algorithm_var.name)
                if isinstance(algorithm_ty, types.EnumMember):
                    literal_value = getattr(algorithm_ty, "literal_value", None)
                    if literal_value is None:
                        literal_value = algorithm_ty.value
                    algorithm_id = algorithm_ty.instance_class(literal_value)
        if algorithm_id is None:
            try:
                from cuda.coop._enums import WarpLoadAlgorithm, WarpStoreAlgorithm
            except Exception:
                WarpLoadAlgorithm = None
                WarpStoreAlgorithm = None
            if self.is_load and WarpLoadAlgorithm is not None:
                algorithm_id = WarpLoadAlgorithm.DIRECT
            elif not self.is_load and WarpStoreAlgorithm is not None:
                algorithm_id = WarpStoreAlgorithm.DIRECT

        num_valid_items = self.get_arg_value_safe("num_valid_items")
        num_valid_items_var = None
        num_valid_items_value = None
        num_valid_items_type = None
        if num_valid_items is None:
            num_valid_items = self.bound.arguments.get("num_valid_items", None)
        if num_valid_items is not None:
            if isinstance(num_valid_items, ir.Var):
                num_valid_items_var = num_valid_items
                num_valid_items_type = self.typemap[num_valid_items.name]
            elif isinstance(num_valid_items, ir.Const):
                num_valid_items_value = num_valid_items.value
            else:
                num_valid_items_value = num_valid_items

            if num_valid_items_var is None:
                scope = self.instr.target.scope
                const_name = f"$warp_load_num_valid_{self.unique_id}"
                const_var = ir.Var(scope, const_name, expr.loc)
                if const_name in self.typemap:
                    raise RuntimeError(
                        f"Variable {const_name} already exists in typemap."
                    )
                const_assign = ir.Assign(
                    value=ir.Const(int(num_valid_items_value), expr.loc),
                    target=const_var,
                    loc=expr.loc,
                )
                self.typemap[const_name] = types.int32
                self.num_valid_assign = const_assign
                num_valid_items_var = const_var
                num_valid_items_type = types.int32

            runtime_args.append(num_valid_items_var)
            runtime_arg_types.append(num_valid_items_type or types.int32)
            runtime_arg_names.append("num_valid_items")

        oob_default = self.get_arg_value_safe("oob_default")
        oob_default_var = None
        oob_default_value = None
        oob_default_type = None
        if oob_default is None:
            oob_default = self.bound.arguments.get("oob_default", None)
        if oob_default is not None:
            if not self.is_load:
                raise RuntimeError("oob_default is only valid for coop.warp.load")
            if num_valid_items is None:
                raise RuntimeError(
                    "coop.warp.load requires num_valid_items when using oob_default"
                )
            if isinstance(oob_default, ir.Var):
                oob_default_var = oob_default
                oob_default_type = self.typemap[oob_default.name]
            elif isinstance(oob_default, ir.Const):
                oob_default_value = oob_default.value
            else:
                oob_default_value = oob_default

            if oob_default_var is None:
                from numba.np.numpy_support import as_dtype

                const_value = oob_default_value
                try:
                    const_value = as_dtype(dtype).type(oob_default_value)
                except Exception:
                    pass
                scope = self.instr.target.scope
                const_name = f"$warp_load_oob_default_{self.unique_id}"
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
                self.oob_default_assign = const_assign
                oob_default_var = const_var
                oob_default_type = dtype

            runtime_args.append(oob_default_var)
            runtime_arg_types.append(oob_default_type or dtype)
            runtime_arg_names.append("oob_default")

        temp_storage = self.bound.arguments.get("temp_storage")
        temp_storage_info = None
        if temp_storage is not None:
            if not isinstance(temp_storage, ir.Var):
                raise RuntimeError(
                    "coop.warp.load/store temp_storage must be provided as a variable"
                )
            (_, _, temp_storage_info) = rewriter.bind_temp_storage_runtime_arg(
                node=self,
                temp_storage=temp_storage,
                runtime_args=runtime_args,
                runtime_arg_types=runtime_arg_types,
                runtime_arg_names=runtime_arg_names,
                insert_pos=0,
            )

        self.dtype = dtype
        self.items_per_thread = items_per_thread
        self.algorithm_id = algorithm_id
        self.num_valid_items = num_valid_items
        self.src = src
        self.dst = dst
        self.temp_storage = temp_storage
        self.temp_storage_info = temp_storage_info
        self.runtime_args = runtime_args
        self.runtime_arg_types = runtime_arg_types
        self.runtime_arg_names = runtime_arg_names

        impl_kwds = {
            "dtype": dtype,
            "items_per_thread": items_per_thread,
            "threads_in_warp": threads_in_warp,
            "algorithm": algorithm_id,
            "num_valid_items": num_valid_items,
            "methods": methods,
            "unique_id": self.unique_id,
            "temp_storage": temp_storage,
            "node": self,
        }
        if self.is_load:
            impl_kwds["oob_default"] = oob_default

        self.impl_kwds = impl_kwds
        self.return_type = types.void

        if self.is_two_phase and self.two_phase_instance is not None:
            instance = self.two_phase_instance
            needs_temp_storage = (
                temp_storage is not None
                and getattr(instance, "temp_storage", None) is None
            )
            if needs_temp_storage:
                self.instance = self.instantiate_impl(**self.impl_kwds)

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        instrs = [rd.g_assign]
        num_valid_assign = getattr(self, "num_valid_assign", None)
        if num_valid_assign is not None:
            instrs.append(num_valid_assign)
        oob_default_assign = getattr(self, "oob_default_assign", None)
        if oob_default_assign is not None:
            instrs.append(oob_default_assign)
        instrs.append(rd.new_assign)
        return tuple(instrs)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()


@dataclass
class CoopWarpLoadNode(CoopWarpLoadStoreNode, CoopNodeMixin):
    primitive_name = "coop.warp.load"


@dataclass
class CoopWarpStoreNode(CoopWarpLoadStoreNode, CoopNodeMixin):
    primitive_name = "coop.warp.store"
