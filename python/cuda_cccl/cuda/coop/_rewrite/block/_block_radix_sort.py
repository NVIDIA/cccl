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
# Radix Sort
# =============================================================================
@dataclass
class CoopBlockRadixSortNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.block.radix_sort_keys"
    disposition = Disposition.ONE_SHOT

    def refine_match(self, rewriter):
        return self._refine_block_radix_sort(rewriter, descending=False)

    def _refine_block_radix_sort(self, rewriter, descending: bool):
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
        decomposer = bound.get("decomposer")
        blocked_to_striped = None
        if keys is None:
            raise RuntimeError("coop.block.radix_sort_keys requires keys")
        if not isinstance(keys, ir.Var):
            raise RuntimeError("coop.block.radix_sort_keys keys must be a variable")

        keys_ty = self.typemap[keys.name]
        if not isinstance(keys_ty, types.Array):
            raise RuntimeError(
                "coop.block.radix_sort_keys requires keys to be an array"
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

        value_dtype = None
        values_ty = None
        values_is_thread = False
        if values is not None:
            if not isinstance(values, ir.Var):
                raise RuntimeError(
                    "coop.block.radix_sort_keys values must be a variable"
                )
            values_ty = self.typemap[values.name]
            values_is_thread = isinstance(values_ty, thread_data_type)
            if not isinstance(values_ty, types.Array):
                raise RuntimeError(
                    "coop.block.radix_sort_keys requires values to be an array"
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
                    "coop.block.radix_sort_keys requires keys and values to have the "
                    f"same items_per_thread; got {values_items} vs {items_per_thread}"
                )

        items_per_thread_kwarg = self.get_arg_value_safe("items_per_thread")
        if items_per_thread_kwarg is None:
            items_per_thread_kwarg = items_per_thread
        if items_per_thread_kwarg != items_per_thread:
            raise RuntimeError(
                "coop.block.radix_sort_keys items_per_thread must match the "
                f"keys array shape ({items_per_thread}); got {items_per_thread_kwarg}"
            )
        if items_per_thread < 1:
            raise RuntimeError("items_per_thread must be >= 1")

        begin_bit = bound.get("begin_bit")
        end_bit = bound.get("end_bit")
        if (begin_bit is None) != (end_bit is None):
            raise RuntimeError(
                "coop.block.radix_sort_keys requires both begin_bit and end_bit"
            )

        begin_bit_var = None
        end_bit_var = None
        if begin_bit is not None:
            if isinstance(begin_bit, ir.Var):
                begin_bit_var = begin_bit
            elif isinstance(begin_bit, ir.Const):
                begin_bit_value = begin_bit.value
            else:
                begin_bit_value = begin_bit

            if begin_bit_var is None:
                scope = self.instr.target.scope
                const_name = f"$block_radix_sort_begin_bit_{self.unique_id}"
                const_var = ir.Var(scope, const_name, self.expr.loc)
                if const_name in self.typemap:
                    raise RuntimeError(
                        f"Variable {const_name} already exists in typemap."
                    )
                const_assign = ir.Assign(
                    value=ir.Const(int(begin_bit_value), self.expr.loc),
                    target=const_var,
                    loc=self.expr.loc,
                )
                self.typemap[const_name] = types.int32
                self.begin_bit_assign = const_assign
                begin_bit_var = const_var

            if isinstance(end_bit, ir.Var):
                end_bit_var = end_bit
            elif isinstance(end_bit, ir.Const):
                end_bit_value = end_bit.value
            else:
                end_bit_value = end_bit

            if end_bit_var is None:
                scope = self.instr.target.scope
                const_name = f"$block_radix_sort_end_bit_{self.unique_id}"
                const_var = ir.Var(scope, const_name, self.expr.loc)
                if const_name in self.typemap:
                    raise RuntimeError(
                        f"Variable {const_name} already exists in typemap."
                    )
                const_assign = ir.Assign(
                    value=ir.Const(int(end_bit_value), self.expr.loc),
                    target=const_var,
                    loc=self.expr.loc,
                )
                self.typemap[const_name] = types.int32
                self.end_bit_assign = const_assign
                end_bit_var = const_var

        blocked_to_striped = self.get_arg_value_safe("blocked_to_striped")
        blocked_arg = self.bound.arguments.get("blocked_to_striped")
        if blocked_to_striped is None and blocked_arg is not None:
            raise RuntimeError("blocked_to_striped must be a compile-time constant")
        if blocked_to_striped is None:
            blocked_to_striped = False
        if not isinstance(blocked_to_striped, bool):
            raise RuntimeError("blocked_to_striped must be a boolean")

        decomposer_value = self.get_arg_value_safe("decomposer")
        if decomposer_value is None and decomposer is not None:
            raise RuntimeError("decomposer must be a compile-time constant")
        decomposer_obj = None
        decomposer_ret_dtype = None
        if decomposer_value is not None:
            from ..._types import Decomposer

            if isinstance(decomposer_value, Decomposer):
                decomposer_obj = decomposer_value
                decomposer_ret_dtype = decomposer_value.ret_dtype
            else:
                decomposer_obj = decomposer_value
                decomposer_ret_dtype = getattr(
                    decomposer_value,
                    "ret_dtype",
                    getattr(decomposer_value, "return_dtype", None),
                )
            if decomposer_ret_dtype is None:
                raise RuntimeError(
                    "decomposer requires a return dtype; use coop.Decomposer(op, ret_dtype)"
                )

        temp_storage = bound.get("temp_storage")
        temp_storage_info = None
        if temp_storage is not None:
            if not isinstance(temp_storage, ir.Var):
                raise RuntimeError(
                    "coop.block.radix_sort_keys temp_storage must be provided as a variable"
                )
            (_, _, temp_storage_info) = rewriter.bind_temp_storage_runtime_arg(
                node=self,
                temp_storage=temp_storage,
                runtime_args=runtime_args,
                runtime_arg_types=runtime_arg_types,
                runtime_arg_names=runtime_arg_names,
                insert_pos=0,
            )

        if keys_is_thread:
            keys_ty = types.Array(dtype, 1, "C")
        if values is not None and values_is_thread:
            values_ty = types.Array(value_dtype, 1, "C")

        runtime_args.append(keys)
        runtime_arg_types.append(keys_ty)
        runtime_arg_names.append("keys")
        if values is not None:
            runtime_args.append(values)
            runtime_arg_types.append(values_ty)
            runtime_arg_names.append("values")

        if begin_bit_var is not None:
            runtime_args.extend([begin_bit_var, end_bit_var])
            runtime_arg_types.extend([types.int32, types.int32])
            runtime_arg_names.extend(["begin_bit", "end_bit"])

        # If keys came from ThreadData, dtype has already been inferred.
        methods = getattr(dtype, "methods", None)
        if methods is not None and not methods:
            methods = None

        self.impl_kwds = {
            "dtype": dtype,
            "threads_per_block": self.threads_per_block,
            "items_per_thread": items_per_thread,
            "value_dtype": value_dtype,
            "begin_bit": begin_bit,
            "end_bit": end_bit,
            "decomposer": decomposer_obj,
            "blocked_to_striped": blocked_to_striped,
            "unique_id": self.unique_id,
            "temp_storage": temp_storage,
            "node": self,
        }

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
                decomposer_obj is not None
                and getattr(instance, "decomposer", None) is None
            ):
                needs_rebuild = True
            if (
                decomposer_obj is not None
                and getattr(instance, "decomposer", None) is not None
            ):
                if decomposer_obj != getattr(instance, "decomposer"):
                    needs_rebuild = True
            if blocked_to_striped and not getattr(
                instance, "blocked_to_striped", False
            ):
                needs_rebuild = True
            if (
                temp_storage is not None
                and getattr(instance, "temp_storage", None) is None
            ):
                needs_rebuild = True
            if needs_rebuild:
                self.instance = self.instantiate_impl(**self.impl_kwds)

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        instrs = [rd.g_assign]
        begin_bit_assign = getattr(self, "begin_bit_assign", None)
        if begin_bit_assign is not None:
            instrs.append(begin_bit_assign)
        end_bit_assign = getattr(self, "end_bit_assign", None)
        if end_bit_assign is not None:
            instrs.append(end_bit_assign)
        instrs.append(rd.new_assign)
        return instrs

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()


@dataclass
class CoopBlockRadixSortDescendingNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.block.radix_sort_keys_descending"
    disposition = Disposition.ONE_SHOT

    def refine_match(self, rewriter):
        return CoopBlockRadixSortNode._refine_block_radix_sort(
            self, rewriter, descending=True
        )

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        instrs = [rd.g_assign]
        scalar_alloc = getattr(self, "scalar_output_alloc", None)
        if scalar_alloc is not None:
            instrs.insert(0, scalar_alloc)
        instrs.append(rd.new_assign)
        return tuple(instrs)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()
