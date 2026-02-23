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
# Merge Sort
# =============================================================================
@dataclass
class CoopBlockMergeSortNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.block.merge_sort_keys"
    disposition = Disposition.ONE_SHOT

    def refine_match(self, rewriter):
        launch_config = rewriter.launch_config
        if launch_config is None:
            return False

        self.threads_per_block = launch_config.blockdim

        runtime_args = []
        runtime_arg_types = []
        runtime_arg_names = []

        bound = self.bound.arguments
        keys = bound.get("keys")
        values = bound.get("values")
        valid_items = bound.get("valid_items")
        oob_default = bound.get("oob_default")
        if keys is None:
            raise RuntimeError("coop.block.merge_sort_keys requires keys")
        if not isinstance(keys, ir.Var):
            raise RuntimeError("coop.block.merge_sort_keys keys must be a variable")

        keys_ty = self.typemap[keys.name]
        if not isinstance(keys_ty, types.Array):
            raise RuntimeError(
                "coop.block.merge_sort_keys requires keys to be an array"
            )

        thread_data_type = get_thread_data_type()
        keys_is_thread = isinstance(keys_ty, thread_data_type)

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
                    "Expected keys constructor call to be an ArrayCallDefinition, "
                    f"but got {keys_leaf!r} for {keys!r}"
                )
            items_per_thread = keys_leaf.shape
            if isinstance(items_per_thread, types.IntegerLiteral):
                items_per_thread = items_per_thread.literal_value
            if not isinstance(items_per_thread, int):
                raise RuntimeError(
                    f"Expected items_per_thread to be an int, got {items_per_thread!r}"
                )
            dtype = keys_ty.dtype

        primitive_name = getattr(self, "primitive_name", "coop.block.merge_sort_keys")
        compare_op = self.get_arg_value_safe("compare_op")
        if compare_op is None and values is not None:
            compare_op = self.get_arg_value_safe("values")
            if primitive_name.endswith("merge_sort_pairs") and compare_op is values:
                compare_op = None
        if compare_op is None:
            raise RuntimeError("coop.block.merge_sort_keys requires compare_op")

        value_dtype = None
        values_ty = None
        values_is_thread = False
        if values is not None and compare_op is not values:
            if not isinstance(values, ir.Var):
                raise RuntimeError(
                    "coop.block.merge_sort_keys values must be a variable"
                )
            values_ty = self.typemap[values.name]
            values_is_thread = isinstance(values_ty, thread_data_type)
            if not isinstance(values_ty, types.Array):
                raise RuntimeError(
                    "coop.block.merge_sort_keys requires values to be an array"
                )
            values_root = rewriter.get_root_def(values)
            values_leaf = values_root.leaf_constructor_call
            if values_is_thread:
                if not isinstance(values_leaf, ThreadDataCallDefinition):
                    raise RuntimeError(
                        "Expected values constructor call to be a ThreadDataCallDefinition, "
                        f"but got {values_leaf!r} for {values!r}"
                    )
                values_info = rewriter.get_thread_data_info(values)
                values_items = values_info.items_per_thread
                value_dtype = values_info.dtype
            else:
                if not isinstance(values_leaf, ArrayCallDefinition):
                    raise RuntimeError(
                        "Expected values constructor call to be an ArrayCallDefinition, "
                        f"but got {values_leaf!r} for {values!r}"
                    )
                values_items = values_leaf.shape
                value_dtype = values_ty.dtype
            if values_items != items_per_thread:
                raise RuntimeError(
                    "coop.block.merge_sort_keys requires keys and values to have "
                    f"the same items_per_thread; got {values_items} vs {items_per_thread}"
                )
        if compare_op is values:
            values = None
            values_ty = None
            value_dtype = None

        items_per_thread_kwarg = self.get_arg_value_safe("items_per_thread")
        if items_per_thread_kwarg is None:
            items_per_thread_kwarg = items_per_thread
        if items_per_thread_kwarg != items_per_thread:
            raise RuntimeError(
                "coop.block.merge_sort_keys items_per_thread must match the "
                f"keys array shape ({items_per_thread}); got {items_per_thread_kwarg}"
            )
        if items_per_thread < 1:
            raise RuntimeError("items_per_thread must be >= 1")

        if keys_is_thread:
            keys_ty = types.Array(dtype, 1, "C")
        if values is not None and values_is_thread:
            values_ty = types.Array(value_dtype, 1, "C")
        methods = getattr(dtype, "methods", None)
        if methods is not None and not methods:
            methods = None
        if (valid_items is None) != (oob_default is None):
            raise RuntimeError(
                "coop.block.merge_sort_keys requires valid_items and oob_default together"
            )

        valid_items_var = None
        if valid_items is not None:
            if isinstance(valid_items, ir.Var):
                valid_items_var = valid_items
            elif isinstance(valid_items, ir.Const):
                valid_items_value = valid_items.value
            else:
                valid_items_value = valid_items

            if valid_items_var is None:
                scope = self.instr.target.scope
                const_name = f"$block_merge_sort_valid_items_{self.unique_id}"
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

            oob_default_var = None
            if isinstance(oob_default, ir.Var):
                oob_default_var = oob_default
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
                const_name = f"$block_merge_sort_oob_{self.unique_id}"
                const_var = ir.Var(scope, const_name, self.expr.loc)
                if const_name in self.typemap:
                    raise RuntimeError(
                        f"Variable {const_name} already exists in typemap."
                    )
                const_assign = ir.Assign(
                    value=ir.Const(const_value, self.expr.loc),
                    target=const_var,
                    loc=self.expr.loc,
                )
                self.typemap[const_name] = dtype
                self.oob_default_assign = const_assign
                oob_default_var = const_var

        temp_storage = bound.get("temp_storage")
        temp_storage_info = None
        if temp_storage is not None:
            if not isinstance(temp_storage, ir.Var):
                raise RuntimeError(
                    "coop.block.merge_sort_keys temp_storage must be provided as a variable"
                )
            (_, _, temp_storage_info) = rewriter.bind_temp_storage_runtime_arg(
                node=self,
                temp_storage=temp_storage,
                runtime_args=runtime_args,
                runtime_arg_types=runtime_arg_types,
                runtime_arg_names=runtime_arg_names,
                insert_pos=0,
            )

        runtime_args.append(keys)
        runtime_arg_types.append(keys_ty)
        runtime_arg_names.append("keys")
        if values is not None:
            runtime_args.append(values)
            runtime_arg_types.append(values_ty)
            runtime_arg_names.append("values")
        if valid_items is not None:
            runtime_args.append(valid_items_var)
            runtime_arg_types.append(types.int32)
            runtime_arg_names.append("valid_items")
            runtime_args.append(oob_default_var)
            runtime_arg_types.append(dtype)
            runtime_arg_names.append("oob_default")

        alias_pairs = primitive_name.endswith("merge_sort_pairs")
        self.impl_kwds = {
            "dtype": dtype,
            "threads_per_block": self.threads_per_block,
            "items_per_thread": items_per_thread,
            "compare_op": compare_op,
            "value_dtype": value_dtype,
            "valid_items": valid_items,
            "oob_default": oob_default,
            "methods": methods,
            "unique_id": self.unique_id,
            "temp_storage": temp_storage,
            "node": self,
        }
        if alias_pairs:
            self.impl_kwds = {
                "keys": dtype,
                "values": value_dtype,
                "threads_per_block": self.threads_per_block,
                "items_per_thread": items_per_thread,
                "compare_op": compare_op,
                "valid_items": valid_items,
                "oob_default": oob_default,
                "methods": methods,
                "unique_id": self.unique_id,
                "temp_storage": temp_storage,
                "node": self,
            }
        if alias_pairs and value_dtype is None and values_ty is not None:
            self.impl_kwds["value_dtype"] = values_ty.dtype

        self.return_type = types.void
        self.runtime_args = runtime_args
        self.runtime_arg_types = runtime_arg_types
        self.runtime_arg_names = runtime_arg_names
        self.temp_storage = temp_storage
        self.temp_storage_info = temp_storage_info

        if self.is_two_phase and self.two_phase_instance is not None:
            instance = self.two_phase_instance
            needs_rebuild = False
            instance_value_dtype = getattr(instance, "value_dtype", None)
            if value_dtype is not None and instance_value_dtype is None:
                needs_rebuild = True
            if value_dtype is not None and instance_value_dtype is not None:
                if value_dtype != instance_value_dtype:
                    needs_rebuild = True
            if (
                valid_items is not None
                and getattr(instance, "valid_items", None) is None
            ):
                needs_rebuild = True
            if needs_rebuild:
                self.instance = self.instantiate_impl(**self.impl_kwds)
            needs_temp_storage = (
                temp_storage is not None
                and getattr(instance, "temp_storage", None) is None
            )
            if needs_temp_storage:
                self.instance = self.instantiate_impl(**self.impl_kwds)

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        instrs = [rd.g_assign]
        valid_items_assign = getattr(self, "valid_items_assign", None)
        if valid_items_assign is not None:
            instrs.append(valid_items_assign)
        oob_default_assign = getattr(self, "oob_default_assign", None)
        if oob_default_assign is not None:
            instrs.append(oob_default_assign)
        instrs.append(rd.new_assign)
        return tuple(instrs)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()


@dataclass
class CoopBlockMergeSortPairsNode(CoopBlockMergeSortNode, CoopNodeMixin):
    primitive_name = "coop.block.merge_sort_pairs"
