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
# Discontinuity
# =============================================================================
@dataclass
class CoopBlockDiscontinuityNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.block.discontinuity"
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
        head_flags = bound.get("head_flags")
        tail_flags = bound.get("tail_flags")

        if items is None or head_flags is None:
            raise RuntimeError("coop.block.discontinuity requires items and head_flags")

        if not isinstance(items, ir.Var):
            raise RuntimeError("coop.block.discontinuity items must be a variable")
        if not isinstance(head_flags, ir.Var):
            raise RuntimeError("coop.block.discontinuity head_flags must be a variable")
        if tail_flags is not None and not isinstance(tail_flags, ir.Var):
            raise RuntimeError("coop.block.discontinuity tail_flags must be a variable")

        items_ty = self.typemap[items.name]
        head_flags_ty = self.typemap[head_flags.name]
        tail_flags_ty = (
            self.typemap[tail_flags.name] if tail_flags is not None else None
        )
        thread_data_type = get_thread_data_type()
        items_is_thread = isinstance(items_ty, thread_data_type)
        head_is_thread = isinstance(head_flags_ty, thread_data_type)
        tail_is_thread = tail_flags_ty is not None and isinstance(
            tail_flags_ty, thread_data_type
        )

        if not items_is_thread and not isinstance(items_ty, types.Array):
            raise RuntimeError(
                "coop.block.discontinuity requires items to be an array or ThreadData"
            )
        if not head_is_thread and not isinstance(head_flags_ty, types.Array):
            raise RuntimeError(
                "coop.block.discontinuity requires head_flags to be an array or ThreadData"
            )
        if tail_flags is not None:
            if not tail_is_thread and not isinstance(tail_flags_ty, types.Array):
                raise RuntimeError(
                    "coop.block.discontinuity requires tail_flags to be an array or "
                    "ThreadData"
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
        if isinstance(items_per_thread, types.IntegerLiteral):
            items_per_thread = items_per_thread.literal_value
        if not isinstance(items_per_thread, int):
            raise RuntimeError(
                f"Expected items_per_thread to be an int, got {items_per_thread!r}"
            )
        head_items = _infer_items_per_thread(head_flags, head_is_thread)
        if items_per_thread != head_items:
            raise RuntimeError(
                "coop.block.discontinuity requires items and head_flags to have "
                "the same items_per_thread"
            )
        if tail_flags is not None:
            tail_items = _infer_items_per_thread(tail_flags, tail_is_thread)
            if tail_items != items_per_thread:
                raise RuntimeError(
                    "coop.block.discontinuity requires tail_flags to have the same "
                    "items_per_thread as items"
                )

        items_per_thread_kwarg = self.get_arg_value_safe("items_per_thread")
        if items_per_thread_kwarg is not None:
            if items_per_thread_kwarg != items_per_thread:
                raise RuntimeError(
                    "coop.block.discontinuity items_per_thread must match the "
                    f"array shape ({items_per_thread}); got {items_per_thread_kwarg}"
                )

        if items_is_thread:
            item_dtype = rewriter.get_thread_data_info(items).dtype
        else:
            item_dtype = items_ty.dtype

        if head_is_thread:
            flag_dtype = rewriter.get_thread_data_info(head_flags).dtype
        else:
            flag_dtype = head_flags_ty.dtype

        if tail_flags is not None:
            if tail_is_thread:
                tail_dtype = rewriter.get_thread_data_info(tail_flags).dtype
            else:
                tail_dtype = tail_flags_ty.dtype
            if tail_dtype != flag_dtype:
                raise RuntimeError(
                    "coop.block.discontinuity requires head_flags and tail_flags to "
                    "have the same dtype"
                )

        methods = getattr(item_dtype, "methods", None)
        if methods is not None and not methods:
            methods = None

        block_discontinuity_type = self.get_arg_value_safe("block_discontinuity_type")
        if block_discontinuity_type is None:
            from cuda.coop.block._block_discontinuity import BlockDiscontinuityType

            block_discontinuity_type = BlockDiscontinuityType.HEADS
        else:
            from cuda.coop.block._block_discontinuity import BlockDiscontinuityType

            if isinstance(block_discontinuity_type, types.EnumMember):
                literal_value = getattr(block_discontinuity_type, "literal_value", None)
                if literal_value is None:
                    literal_value = block_discontinuity_type.value
                block_discontinuity_type = block_discontinuity_type.instance_class(
                    literal_value
                )
            if isinstance(block_discontinuity_type, int):
                block_discontinuity_type = BlockDiscontinuityType(
                    block_discontinuity_type
                )
            if block_discontinuity_type not in BlockDiscontinuityType:
                raise RuntimeError(
                    "coop.block.discontinuity requires block_discontinuity_type to "
                    "be a BlockDiscontinuityType enum value"
                )

        if (
            block_discontinuity_type == BlockDiscontinuityType.HEADS_AND_TAILS
            and tail_flags is None
        ):
            raise RuntimeError(
                "coop.block.discontinuity requires tail_flags for HEADS_AND_TAILS"
            )

        flag_op = self.get_arg_value_safe("flag_op")
        if flag_op is None:
            raise RuntimeError("coop.block.discontinuity requires flag_op to be set")

        tile_predecessor_item = bound.get("tile_predecessor_item")
        tile_successor_item = bound.get("tile_successor_item")
        tile_predecessor_var = None
        tile_successor_var = None
        if (
            tile_predecessor_item is not None
            and block_discontinuity_type == BlockDiscontinuityType.TAILS
        ):
            raise RuntimeError(
                "coop.block.discontinuity does not accept tile_predecessor_item for TAILS"
            )
        if (
            tile_successor_item is not None
            and block_discontinuity_type == BlockDiscontinuityType.HEADS
        ):
            raise RuntimeError(
                "coop.block.discontinuity does not accept tile_successor_item for HEADS"
            )

        if tile_predecessor_item is not None:
            if isinstance(tile_predecessor_item, ir.Var):
                tile_predecessor_var = tile_predecessor_item
            elif isinstance(tile_predecessor_item, ir.Const):
                tile_predecessor_value = tile_predecessor_item.value
            else:
                tile_predecessor_value = tile_predecessor_item
            if tile_predecessor_var is None:
                scope = self.instr.target.scope
                const_name = f"$block_disc_tile_predecessor_{self.unique_id}"
                const_var = ir.Var(scope, const_name, self.expr.loc)
                if const_name in self.typemap:
                    raise RuntimeError(
                        f"Variable {const_name} already exists in typemap."
                    )
                const_assign = ir.Assign(
                    value=ir.Const(tile_predecessor_value, self.expr.loc),
                    target=const_var,
                    loc=self.expr.loc,
                )
                self.typemap[const_name] = item_dtype
                self.tile_predecessor_assign = const_assign
                tile_predecessor_var = const_var

        if tile_successor_item is not None:
            if isinstance(tile_successor_item, ir.Var):
                tile_successor_var = tile_successor_item
            elif isinstance(tile_successor_item, ir.Const):
                tile_successor_value = tile_successor_item.value
            else:
                tile_successor_value = tile_successor_item
            if tile_successor_var is None:
                scope = self.instr.target.scope
                const_name = f"$block_disc_tile_successor_{self.unique_id}"
                const_var = ir.Var(scope, const_name, self.expr.loc)
                if const_name in self.typemap:
                    raise RuntimeError(
                        f"Variable {const_name} already exists in typemap."
                    )
                const_assign = ir.Assign(
                    value=ir.Const(tile_successor_value, self.expr.loc),
                    target=const_var,
                    loc=self.expr.loc,
                )
                self.typemap[const_name] = item_dtype
                self.tile_successor_assign = const_assign
                tile_successor_var = const_var

        temp_storage = bound.get("temp_storage")
        temp_storage_info = None
        if temp_storage is not None:
            if not isinstance(temp_storage, ir.Var):
                raise RuntimeError(
                    "coop.block.discontinuity temp_storage must be provided as a "
                    "variable"
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
        array_flags_ty = types.Array(flag_dtype, 1, "C")

        if items_is_thread:
            items_ty = array_items_ty
        if head_is_thread:
            head_flags_ty = array_flags_ty
        if tail_flags is not None and tail_is_thread:
            tail_flags_ty = array_flags_ty

        if block_discontinuity_type == BlockDiscontinuityType.HEADS:
            runtime_args.extend([head_flags, items])
            runtime_arg_types.extend([head_flags_ty, items_ty])
            runtime_arg_names.extend(["head_flags", "items"])
            if tile_predecessor_var is not None:
                runtime_args.append(tile_predecessor_var)
                runtime_arg_types.append(item_dtype)
                runtime_arg_names.append("tile_predecessor_item")
        elif block_discontinuity_type == BlockDiscontinuityType.TAILS:
            runtime_args.extend([head_flags, items])
            runtime_arg_types.extend([head_flags_ty, items_ty])
            runtime_arg_names.extend(["tail_flags", "items"])
            if tile_successor_var is not None:
                runtime_args.append(tile_successor_var)
                runtime_arg_types.append(item_dtype)
                runtime_arg_names.append("tile_successor_item")
        else:
            if tile_predecessor_var is not None and tile_successor_var is not None:
                runtime_args.extend(
                    [
                        head_flags,
                        tile_predecessor_var,
                        tail_flags,
                        tile_successor_var,
                        items,
                    ]
                )
                runtime_arg_types.extend(
                    [head_flags_ty, item_dtype, tail_flags_ty, item_dtype, items_ty]
                )
                runtime_arg_names.extend(
                    [
                        "head_flags",
                        "tile_predecessor_item",
                        "tail_flags",
                        "tile_successor_item",
                        "items",
                    ]
                )
            elif tile_predecessor_var is not None:
                runtime_args.extend(
                    [head_flags, tile_predecessor_var, tail_flags, items]
                )
                runtime_arg_types.extend(
                    [head_flags_ty, item_dtype, tail_flags_ty, items_ty]
                )
                runtime_arg_names.extend(
                    ["head_flags", "tile_predecessor_item", "tail_flags", "items"]
                )
            elif tile_successor_var is not None:
                runtime_args.extend([head_flags, tail_flags, tile_successor_var, items])
                runtime_arg_types.extend(
                    [head_flags_ty, tail_flags_ty, item_dtype, items_ty]
                )
                runtime_arg_names.extend(
                    ["head_flags", "tail_flags", "tile_successor_item", "items"]
                )
            else:
                runtime_args.extend([head_flags, tail_flags, items])
                runtime_arg_types.extend([head_flags_ty, tail_flags_ty, items_ty])
                runtime_arg_names.extend(["head_flags", "tail_flags", "items"])

        self.items = items
        self.head_flags = head_flags
        self.tail_flags = tail_flags
        self.item_dtype = item_dtype
        self.flag_dtype = flag_dtype
        self.items_per_thread = items_per_thread
        self.flag_op = flag_op
        self.block_discontinuity_type = block_discontinuity_type
        self.tile_predecessor_item = tile_predecessor_item
        self.tile_successor_item = tile_successor_item
        self.temp_storage = temp_storage
        self.temp_storage_info = temp_storage_info
        self.methods = methods

        self.impl_kwds = {
            "block_discontinuity_type": block_discontinuity_type,
            "dtype": item_dtype,
            "threads_per_block": self.threads_per_block,
            "items_per_thread": items_per_thread,
            "flag_op": flag_op,
            "flag_dtype": flag_dtype,
            "methods": methods,
            "unique_id": self.unique_id,
            "temp_storage": temp_storage,
            "tile_predecessor_item": tile_predecessor_item,
            "tile_successor_item": tile_successor_item,
            "node": self,
        }

        self.runtime_args = runtime_args
        self.runtime_arg_types = runtime_arg_types
        self.runtime_arg_names = runtime_arg_names

        if self.is_two_phase and self.two_phase_instance is not None:
            instance = self.two_phase_instance
            needs_pred = (
                tile_predecessor_item is not None
                and getattr(instance, "tile_predecessor_item", None) is None
            )
            needs_succ = (
                tile_successor_item is not None
                and getattr(instance, "tile_successor_item", None) is None
            )
            if needs_pred or needs_succ:
                self.instance = self.instantiate_impl(**self.impl_kwds)

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        instrs = []
        tile_predecessor_assign = getattr(self, "tile_predecessor_assign", None)
        if tile_predecessor_assign is not None:
            instrs.append(tile_predecessor_assign)
        tile_successor_assign = getattr(self, "tile_successor_assign", None)
        if tile_successor_assign is not None:
            instrs.append(tile_successor_assign)
        instrs.extend([rd.g_assign, rd.new_assign])
        if self.temp_storage_info is not None and self.temp_storage_info.auto_sync:
            instrs.extend(
                rewriter.emit_syncthreads_call(self.instr.target.scope, self.expr.loc)
            )
        return tuple(instrs)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()
