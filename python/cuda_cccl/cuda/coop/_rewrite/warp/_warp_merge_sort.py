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
# Warp Merge Sort
# =============================================================================
@dataclass
class CoopWarpMergeSortNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.warp.merge_sort_keys"
    disposition = Disposition.ONE_SHOT

    def refine_match(self, rewriter):
        runtime_args = []
        runtime_arg_types = []
        runtime_arg_names = []

        bound = self.bound.arguments
        keys = bound.get("keys")
        values = bound.get("values")
        if keys is None or not isinstance(keys, ir.Var):
            raise RuntimeError("coop.warp.merge_sort_keys requires keys")

        keys_ty = self.typemap[keys.name]
        thread_data_type = get_thread_data_type()
        keys_is_thread = isinstance(keys_ty, thread_data_type)
        if not keys_is_thread and not isinstance(keys_ty, types.Array):
            raise RuntimeError(
                "coop.warp.merge_sort_keys requires keys to be an array or ThreadData"
            )

        keys_root = rewriter.get_root_def(keys)
        keys_leaf = keys_root.leaf_constructor_call
        if keys_is_thread:
            if not isinstance(keys_leaf, ThreadDataCallDefinition):
                raise RuntimeError(
                    "Expected keys constructor call to be a ThreadDataCallDefinition, "
                    f"but got {keys_leaf!r} for {keys!r}"
                )
            keys_info = rewriter.get_thread_data_info(keys)
            items_per_thread = keys_info.items_per_thread
            dtype = keys_info.dtype
        else:
            if not isinstance(keys_leaf, ArrayCallDefinition):
                raise RuntimeError(
                    "coop.warp.merge_sort_keys requires keys to be a local array"
                )
            items_per_thread = keys_leaf.shape
            if isinstance(items_per_thread, types.IntegerLiteral):
                items_per_thread = items_per_thread.literal_value
            if not isinstance(items_per_thread, int):
                raise RuntimeError(
                    f"Expected items_per_thread to be an int, got {items_per_thread!r}"
                )
            dtype = keys_ty.dtype

        items_per_thread_kwarg = self.get_arg_value_safe("items_per_thread")
        if items_per_thread_kwarg is None:
            items_per_thread_kwarg = items_per_thread
        if items_per_thread_kwarg != items_per_thread:
            raise RuntimeError(
                "coop.warp.merge_sort_keys items_per_thread must match the "
                f"keys array shape ({items_per_thread}); got {items_per_thread_kwarg}"
            )
        if items_per_thread < 1:
            raise RuntimeError("items_per_thread must be >= 1")

        primitive_name = getattr(self, "primitive_name", "coop.warp.merge_sort_keys")
        compare_op = self.get_arg_value_safe("compare_op")
        if compare_op is None and values is not None:
            compare_op = self.get_arg_value_safe("values")
            if primitive_name.endswith("merge_sort_pairs") and compare_op is values:
                compare_op = None
        if compare_op is None:
            raise RuntimeError("coop.warp.merge_sort_keys requires compare_op")

        value_dtype = None
        values_ty = None
        values_is_thread = False
        if values is not None and compare_op is not values:
            if not isinstance(values, ir.Var):
                raise RuntimeError(
                    "coop.warp.merge_sort_keys values must be a variable"
                )
            values_ty = self.typemap[values.name]
            values_is_thread = isinstance(values_ty, thread_data_type)
            if not values_is_thread and not isinstance(values_ty, types.Array):
                raise RuntimeError(
                    "coop.warp.merge_sort_keys requires values to be an array or "
                    "ThreadData"
                )
            values_root = rewriter.get_root_def(values)
            values_leaf = values_root.leaf_constructor_call
            if values_is_thread:
                if not isinstance(values_leaf, ThreadDataCallDefinition):
                    raise RuntimeError(
                        "Expected values constructor call to be a "
                        "ThreadDataCallDefinition, but got "
                        f"{values_leaf!r} for {values!r}"
                    )
                values_info = rewriter.get_thread_data_info(values)
                values_items = values_info.items_per_thread
                value_dtype = values_info.dtype
            else:
                if not isinstance(values_leaf, ArrayCallDefinition):
                    raise RuntimeError(
                        "Expected values constructor call to be an "
                        f"ArrayCallDefinition, but got {values_leaf!r} for {values!r}"
                    )
                values_items = values_leaf.shape
                value_dtype = values_ty.dtype
            if values_items != items_per_thread:
                raise RuntimeError(
                    "coop.warp.merge_sort_keys requires keys and values to have the "
                    f"same items_per_thread; got {values_items} vs {items_per_thread}"
                )

        if compare_op is values:
            values = None
            values_ty = None

        threads_in_warp = self.get_arg_value_safe("threads_in_warp")
        threads_in_warp_arg = self.bound.arguments.get("threads_in_warp")
        if threads_in_warp is None and threads_in_warp_arg is not None:
            raise RuntimeError("threads_in_warp must be a compile-time constant")
        if threads_in_warp is None:
            threads_in_warp = 32
        if not isinstance(threads_in_warp, int) or threads_in_warp < 1:
            raise RuntimeError("threads_in_warp must be a positive integer")

        methods = getattr(dtype, "methods", None)
        if methods is not None and not methods:
            methods = None

        if compare_op is values:
            values = None
            values_ty = None

        temp_storage = bound.get("temp_storage")
        temp_storage_info = None
        if temp_storage is not None:
            if not isinstance(temp_storage, ir.Var):
                raise RuntimeError(
                    "coop.warp.merge_sort_keys temp_storage must be provided as a variable"
                )
            (_, _, temp_storage_info) = rewriter.bind_temp_storage_runtime_arg(
                node=self,
                temp_storage=temp_storage,
                runtime_args=runtime_args,
                runtime_arg_types=runtime_arg_types,
                runtime_arg_names=runtime_arg_names,
                insert_pos=0,
            )

        runtime_keys_ty = types.Array(dtype, 1, "C") if keys_is_thread else keys_ty
        runtime_values_ty = None
        if values is not None:
            runtime_values_ty = (
                types.Array(value_dtype, 1, "C") if values_is_thread else values_ty
            )

        runtime_args.append(keys)
        runtime_arg_types.append(runtime_keys_ty)
        runtime_arg_names.append("keys")
        if values is not None:
            runtime_args.append(values)
            runtime_arg_types.append(runtime_values_ty)
            runtime_arg_names.append("values")

        alias_pairs = primitive_name.endswith("merge_sort_pairs")
        self.impl_kwds = {
            "dtype": dtype,
            "items_per_thread": items_per_thread,
            "compare_op": compare_op,
            "value_dtype": value_dtype,
            "threads_in_warp": threads_in_warp,
            "methods": methods,
            "unique_id": self.unique_id,
            "temp_storage": temp_storage,
            "node": self,
        }
        if alias_pairs:
            self.impl_kwds = {
                "keys": dtype,
                "values": value_dtype,
                "items_per_thread": items_per_thread,
                "compare_op": compare_op,
                "threads_in_warp": threads_in_warp,
                "methods": methods,
                "unique_id": self.unique_id,
                "temp_storage": temp_storage,
                "node": self,
            }
            if value_dtype is None and values_ty is not None:
                self.impl_kwds["values"] = values_ty.dtype
        elif value_dtype is None and values_ty is not None:
            self.impl_kwds["value_dtype"] = values_ty.dtype

        self.return_type = types.void
        self.runtime_args = runtime_args
        self.runtime_arg_types = runtime_arg_types
        self.runtime_arg_names = runtime_arg_names
        self.temp_storage = temp_storage
        self.temp_storage_info = temp_storage_info

        if self.is_two_phase and self.two_phase_instance is not None:
            instance = self.two_phase_instance
            instance_value_dtype = getattr(instance, "value_dtype", None)
            if value_dtype is not None and instance_value_dtype is None:
                self.instance = self.instantiate_impl(**self.impl_kwds)
            elif value_dtype is not None and instance_value_dtype is not None:
                if value_dtype != instance_value_dtype:
                    self.instance = self.instantiate_impl(**self.impl_kwds)
            needs_temp_storage = (
                temp_storage is not None
                and getattr(instance, "temp_storage", None) is None
            )
            if needs_temp_storage:
                self.instance = self.instantiate_impl(**self.impl_kwds)

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        instrs = [rd.g_assign, rd.new_assign]
        if self.temp_storage_info is not None and self.temp_storage_info.auto_sync:
            instrs.extend(
                rewriter.emit_syncthreads_call(self.instr.target.scope, self.expr.loc)
            )
        return tuple(instrs)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()


@dataclass
class CoopWarpMergeSortPairsNode(CoopWarpMergeSortNode, CoopNodeMixin):
    primitive_name = "coop.warp.merge_sort_pairs"
