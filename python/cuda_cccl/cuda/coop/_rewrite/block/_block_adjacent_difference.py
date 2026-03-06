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
get_thread_data_type = _core.get_thread_data_type
ir = _core.ir


# =============================================================================
# Adjacent Difference
# =============================================================================
@dataclass
class CoopBlockAdjacentDifferenceNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.block.adjacent_difference"
    disposition = Disposition.ONE_SHOT

    def refine_match(self, rewriter):
        self.threads_per_block = self.resolve_threads_per_block()
        instance = self.two_phase_instance if self.is_two_phase else None
        if instance is not None:
            self.instance = instance

        runtime_args = []
        runtime_arg_types = []
        runtime_arg_names = []

        bound = self.bound.arguments
        items = bound.get("items")
        output_items = bound.get("output_items")

        if items is None or output_items is None:
            raise RuntimeError(
                "coop.block.adjacent_difference requires items and output_items"
            )

        if not isinstance(items, ir.Var):
            raise RuntimeError(
                "coop.block.adjacent_difference items must be a variable"
            )
        if not isinstance(output_items, ir.Var):
            raise RuntimeError(
                "coop.block.adjacent_difference output_items must be a variable"
            )

        items_ty = self.typemap[items.name]
        output_items_ty = self.typemap[output_items.name]
        thread_data_type = get_thread_data_type()
        items_is_thread = isinstance(items_ty, thread_data_type)
        output_is_thread = isinstance(output_items_ty, thread_data_type)

        if not items_is_thread and not isinstance(items_ty, types.Array):
            raise RuntimeError(
                "coop.block.adjacent_difference requires items to be an array or "
                "ThreadData"
            )
        if not output_is_thread and not isinstance(output_items_ty, types.Array):
            raise RuntimeError(
                "coop.block.adjacent_difference requires output_items to be an array "
                "or ThreadData"
            )

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

        items_per_thread = _infer_items_per_thread(items, items_is_thread)
        output_items_per_thread = _infer_items_per_thread(
            output_items, output_is_thread
        )
        if items_per_thread != output_items_per_thread:
            raise RuntimeError(
                "coop.block.adjacent_difference requires items and output_items to "
                "have the same items_per_thread"
            )
        if isinstance(items_per_thread, types.IntegerLiteral):
            items_per_thread = items_per_thread.literal_value
        if not isinstance(items_per_thread, int):
            raise RuntimeError(
                f"Expected items_per_thread to be an int, got {items_per_thread!r}"
            )

        items_per_thread_kwarg = self.get_arg_value_safe("items_per_thread")
        if items_per_thread_kwarg is not None:
            if items_per_thread_kwarg != items_per_thread:
                raise RuntimeError(
                    "coop.block.adjacent_difference items_per_thread must match the "
                    f"array shape ({items_per_thread}); got {items_per_thread_kwarg}"
                )

        if items_is_thread:
            item_dtype = rewriter.get_thread_data_info(items).dtype
        else:
            item_dtype = items_ty.dtype

        if output_is_thread:
            output_dtype = rewriter.get_thread_data_info(output_items).dtype
        else:
            output_dtype = output_items_ty.dtype

        if output_dtype != item_dtype:
            raise RuntimeError(
                "coop.block.adjacent_difference requires output_items to have the "
                "same dtype as items"
            )

        methods = getattr(item_dtype, "methods", None)
        if methods is not None and not methods:
            methods = None

        block_adjacent_difference_type = self.get_arg_value_safe(
            "block_adjacent_difference_type"
        )
        if block_adjacent_difference_type is None:
            from cuda.coop.block._block_adjacent_difference import (
                BlockAdjacentDifferenceType,
            )

            block_adjacent_difference_type = BlockAdjacentDifferenceType.SubtractLeft
        else:
            from cuda.coop.block._block_adjacent_difference import (
                BlockAdjacentDifferenceType,
            )

            if isinstance(block_adjacent_difference_type, types.EnumMember):
                literal_value = getattr(
                    block_adjacent_difference_type, "literal_value", None
                )
                if literal_value is None:
                    literal_value = block_adjacent_difference_type.value
                block_adjacent_difference_type = (
                    block_adjacent_difference_type.instance_class(literal_value)
                )
            if isinstance(block_adjacent_difference_type, int):
                block_adjacent_difference_type = BlockAdjacentDifferenceType(
                    block_adjacent_difference_type
                )
            if block_adjacent_difference_type not in BlockAdjacentDifferenceType:
                raise RuntimeError(
                    "coop.block.adjacent_difference requires "
                    "block_adjacent_difference_type to be a "
                    "BlockAdjacentDifferenceType enum value"
                )

        difference_op = self.get_arg_value_safe("difference_op")
        if difference_op is None:
            raise RuntimeError(
                "coop.block.adjacent_difference requires difference_op to be set"
            )

        valid_items = bound.get("valid_items")
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
                const_name = f"$block_adjacent_difference_valid_{self.unique_id}"
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

        tile_predecessor_item = bound.get("tile_predecessor_item")
        tile_successor_item = bound.get("tile_successor_item")
        if tile_predecessor_item is not None and tile_successor_item is not None:
            raise RuntimeError(
                "coop.block.adjacent_difference accepts only one of "
                "tile_predecessor_item or tile_successor_item"
            )
        if (
            block_adjacent_difference_type == BlockAdjacentDifferenceType.SubtractLeft
            and tile_successor_item is not None
        ):
            raise RuntimeError(
                "coop.block.adjacent_difference does not accept tile_successor_item "
                "for SubtractLeft"
            )
        if (
            block_adjacent_difference_type == BlockAdjacentDifferenceType.SubtractRight
            and tile_predecessor_item is not None
        ):
            raise RuntimeError(
                "coop.block.adjacent_difference does not accept tile_predecessor_item "
                "for SubtractRight"
            )

        tile_item_var = None
        tile_item_type = None
        tile_item_name = None
        if tile_predecessor_item is not None:
            if not isinstance(tile_predecessor_item, ir.Var):
                raise RuntimeError(
                    "tile_predecessor_item must be provided as a variable"
                )
            tile_item_var = tile_predecessor_item
            tile_item_type = self.typemap[tile_item_var.name]
            tile_item_name = "tile_predecessor_item"
        if tile_successor_item is not None:
            if not isinstance(tile_successor_item, ir.Var):
                raise RuntimeError("tile_successor_item must be provided as a variable")
            tile_item_var = tile_successor_item
            tile_item_type = self.typemap[tile_item_var.name]
            tile_item_name = "tile_successor_item"

        if tile_item_var is not None and tile_item_type != item_dtype:
            raise RuntimeError(
                "tile_*_item dtype must match items dtype for "
                "coop.block.adjacent_difference"
            )

        temp_storage = bound.get("temp_storage")
        temp_storage_info = None
        if temp_storage is not None:
            if not isinstance(temp_storage, ir.Var):
                raise RuntimeError(
                    "coop.block.adjacent_difference temp_storage must be provided "
                    "as a variable"
                )
            (_, _, temp_storage_info) = rewriter.bind_temp_storage_runtime_arg(
                node=self,
                temp_storage=temp_storage,
                runtime_args=runtime_args,
                runtime_arg_types=runtime_arg_types,
                runtime_arg_names=runtime_arg_names,
                insert_pos=0,
            )

        array_items_ty = types.Array(item_dtype, 1, "C")
        if items_is_thread:
            items_ty = array_items_ty
        if output_is_thread:
            output_items_ty = array_items_ty

        runtime_args.append(items)
        runtime_arg_types.append(items_ty)
        runtime_arg_names.append("items")

        runtime_args.append(output_items)
        runtime_arg_types.append(output_items_ty)
        runtime_arg_names.append("output_items")

        if valid_items_var is not None:
            runtime_args.append(valid_items_var)
            runtime_arg_types.append(types.int32)
            runtime_arg_names.append("valid_items")

        if tile_item_var is not None:
            runtime_args.append(tile_item_var)
            runtime_arg_types.append(tile_item_type)
            runtime_arg_names.append(tile_item_name)

        self.items = items
        self.output_items = output_items
        self.item_dtype = item_dtype
        self.items_per_thread = items_per_thread
        self.difference_op = difference_op
        self.block_adjacent_difference_type = block_adjacent_difference_type
        self.valid_items = valid_items
        self.tile_predecessor_item = tile_predecessor_item
        self.tile_successor_item = tile_successor_item
        self.temp_storage = temp_storage
        self.temp_storage_info = temp_storage_info
        self.methods = methods

        self.impl_kwds = {
            "block_adjacent_difference_type": block_adjacent_difference_type,
            "dtype": item_dtype,
            "threads_per_block": self.threads_per_block,
            "items_per_thread": items_per_thread,
            "difference_op": difference_op,
            "methods": methods,
            "valid_items": valid_items,
            "tile_predecessor_item": tile_predecessor_item,
            "tile_successor_item": tile_successor_item,
            "unique_id": self.unique_id,
            "temp_storage": temp_storage,
            "node": self,
        }

        self.runtime_args = runtime_args
        self.runtime_arg_types = runtime_arg_types
        self.runtime_arg_names = runtime_arg_names

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        instrs = []
        valid_items_assign = getattr(self, "valid_items_assign", None)
        if valid_items_assign is not None:
            instrs.append(valid_items_assign)
        instrs.extend([rd.g_assign, rd.new_assign])
        if self.temp_storage_info is not None and self.temp_storage_info.auto_sync:
            instrs.extend(
                rewriter.emit_syncthreads_call(self.instr.target.scope, self.expr.loc)
            )
        return tuple(instrs)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()
