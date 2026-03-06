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
# Shuffle
# =============================================================================
@dataclass
class CoopBlockShuffleNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.block.shuffle"
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
        block_prefix = bound.get("block_prefix")
        block_suffix = bound.get("block_suffix")

        if items is None:
            raise RuntimeError("coop.block.shuffle requires items")
        if not isinstance(items, ir.Var):
            raise RuntimeError("coop.block.shuffle items must be a variable")

        items_ty = self.typemap[items.name]
        thread_data_type = get_thread_data_type()
        items_is_thread = isinstance(items_ty, thread_data_type)
        items_is_array = items_is_thread or isinstance(items_ty, types.Array)
        items_is_scalar = isinstance(items_ty, types.Number)

        if items_is_array:
            if output_items is None:
                raise RuntimeError(
                    "coop.block.shuffle requires output_items for Up/Down shuffles"
                )
            if not isinstance(output_items, ir.Var):
                raise RuntimeError("coop.block.shuffle output_items must be a variable")
            output_items_ty = self.typemap[output_items.name]
            output_is_thread = isinstance(output_items_ty, thread_data_type)
            if not output_is_thread and not isinstance(output_items_ty, types.Array):
                raise RuntimeError(
                    "coop.block.shuffle output_items must be an array or ThreadData"
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
                    "coop.block.shuffle requires items and output_items to have the "
                    "same items_per_thread"
                )
            if not isinstance(items_per_thread, int):
                raise RuntimeError(
                    f"Expected items_per_thread to be an int, got {items_per_thread!r}"
                )

            items_per_thread_kwarg = self.get_arg_value_safe("items_per_thread")
            if items_per_thread_kwarg is not None:
                if items_per_thread_kwarg != items_per_thread:
                    raise RuntimeError(
                        "coop.block.shuffle items_per_thread must match the "
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
                    "coop.block.shuffle requires output_items to have the same "
                    "dtype as items"
                )

            methods = getattr(item_dtype, "methods", None)
            if methods is not None and not methods:
                methods = None

            block_shuffle_type = self.get_arg_value_safe("block_shuffle_type")
            if block_shuffle_type is None:
                from cuda.coop.block._block_shuffle import BlockShuffleType

                block_shuffle_type = BlockShuffleType.Up
            else:
                from cuda.coop.block._block_shuffle import BlockShuffleType

                if isinstance(block_shuffle_type, types.EnumMember):
                    literal_value = getattr(block_shuffle_type, "literal_value", None)
                    if literal_value is None:
                        literal_value = block_shuffle_type.value
                    block_shuffle_type = block_shuffle_type.instance_class(
                        literal_value
                    )
                if isinstance(block_shuffle_type, int):
                    block_shuffle_type = BlockShuffleType(block_shuffle_type)
                if block_shuffle_type not in BlockShuffleType:
                    raise RuntimeError(
                        "coop.block.shuffle requires block_shuffle_type to be a "
                        "BlockShuffleType enum value"
                    )

            if block_shuffle_type not in (
                BlockShuffleType.Up,
                BlockShuffleType.Down,
            ):
                raise RuntimeError(
                    "coop.block.shuffle requires Up or Down for array shuffles"
                )

            distance = self.get_arg_value_safe("distance")
            if distance is not None:
                raise RuntimeError(
                    "coop.block.shuffle does not accept distance for Up/Down"
                )

            if block_prefix is not None and block_suffix is not None:
                raise RuntimeError(
                    "coop.block.shuffle does not allow block_prefix and "
                    "block_suffix together"
                )

            if block_shuffle_type == BlockShuffleType.Up and block_prefix is not None:
                raise RuntimeError(
                    "coop.block.shuffle does not allow block_prefix for Up shuffles"
                )
            if block_shuffle_type == BlockShuffleType.Down and block_suffix is not None:
                raise RuntimeError(
                    "coop.block.shuffle does not allow block_suffix for Down shuffles"
                )

            block_prefix_ty = None
            block_suffix_ty = None
            if block_prefix is not None:
                if not isinstance(block_prefix, ir.Var):
                    raise RuntimeError(
                        "coop.block.shuffle block_prefix must be a variable"
                    )
                block_prefix_ty = self.typemap[block_prefix.name]
                if not isinstance(block_prefix_ty, types.Array):
                    raise RuntimeError(
                        "coop.block.shuffle block_prefix must be a device array"
                    )
                if block_prefix_ty.dtype != item_dtype:
                    raise RuntimeError(
                        "coop.block.shuffle requires block_prefix to have the same "
                        "dtype as items"
                    )

            if block_suffix is not None:
                if not isinstance(block_suffix, ir.Var):
                    raise RuntimeError(
                        "coop.block.shuffle block_suffix must be a variable"
                    )
                block_suffix_ty = self.typemap[block_suffix.name]
                if not isinstance(block_suffix_ty, types.Array):
                    raise RuntimeError(
                        "coop.block.shuffle block_suffix must be a device array"
                    )
                if block_suffix_ty.dtype != item_dtype:
                    raise RuntimeError(
                        "coop.block.shuffle requires block_suffix to have the same "
                        "dtype as items"
                    )

            temp_storage = bound.get("temp_storage")
            temp_storage_info = None
            if temp_storage is not None:
                if not isinstance(temp_storage, ir.Var):
                    raise RuntimeError(
                        "coop.block.shuffle temp_storage must be provided as a variable"
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

            runtime_args.extend([items, output_items])
            runtime_arg_types.extend([items_ty, output_items_ty])
            runtime_arg_names.extend(["items", "output_items"])

            if block_prefix is not None:
                runtime_args.append(block_prefix)
                runtime_arg_types.append(block_prefix_ty)
                runtime_arg_names.append("block_prefix")
            if block_suffix is not None:
                runtime_args.append(block_suffix)
                runtime_arg_types.append(block_suffix_ty)
                runtime_arg_names.append("block_suffix")

            self.return_type = types.void
            self.items_per_thread = items_per_thread
            self.block_shuffle_type = block_shuffle_type
            self.distance = None
            self.temp_storage_info = temp_storage_info
            self.temp_storage = temp_storage
            self.item_dtype = item_dtype
            self.methods = methods

            self.impl_kwds = {
                "block_shuffle_type": block_shuffle_type,
                "dtype": item_dtype,
                "threads_per_block": self.threads_per_block,
                "items_per_thread": items_per_thread,
                "distance": None,
                "block_prefix": block_prefix,
                "block_suffix": block_suffix,
                "methods": methods,
                "unique_id": self.unique_id,
                "temp_storage": temp_storage,
                "node": self,
            }

            self.runtime_args = runtime_args
            self.runtime_arg_types = runtime_arg_types
            self.runtime_arg_names = runtime_arg_names

            if self.is_two_phase and self.two_phase_instance is not None:
                instance = self.two_phase_instance
                needs_rebuild = False
                if (
                    block_prefix is not None
                    and getattr(instance, "block_prefix", None) is None
                ):
                    needs_rebuild = True
                if (
                    block_suffix is not None
                    and getattr(instance, "block_suffix", None) is None
                ):
                    needs_rebuild = True
                if needs_rebuild:
                    self.instance = self.instantiate_impl(**self.impl_kwds)
            return

        if not items_is_scalar:
            raise RuntimeError(
                "coop.block.shuffle requires items to be a scalar or array"
            )

        block_shuffle_type = self.get_arg_value_safe("block_shuffle_type")
        if block_shuffle_type is None:
            from cuda.coop.block._block_shuffle import BlockShuffleType

            block_shuffle_type = BlockShuffleType.Offset
        else:
            from cuda.coop.block._block_shuffle import BlockShuffleType

            if isinstance(block_shuffle_type, types.EnumMember):
                literal_value = getattr(block_shuffle_type, "literal_value", None)
                if literal_value is None:
                    literal_value = block_shuffle_type.value
                block_shuffle_type = block_shuffle_type.instance_class(literal_value)
            if isinstance(block_shuffle_type, int):
                block_shuffle_type = BlockShuffleType(block_shuffle_type)
            if block_shuffle_type not in BlockShuffleType:
                raise RuntimeError(
                    "coop.block.shuffle requires block_shuffle_type to be a "
                    "BlockShuffleType enum value"
                )

        if block_shuffle_type not in (
            BlockShuffleType.Offset,
            BlockShuffleType.Rotate,
            BlockShuffleType.Up,
            BlockShuffleType.Down,
        ):
            raise RuntimeError(
                "coop.block.shuffle requires a valid BlockShuffleType for scalar shuffles"
            )

        if output_items is not None:
            raise RuntimeError(
                "coop.block.shuffle does not accept output_items for scalar shuffles"
            )
        if block_prefix is not None or block_suffix is not None:
            raise RuntimeError(
                "coop.block.shuffle does not accept block_prefix/block_suffix for "
                "scalar shuffles"
            )

        items_per_thread_kwarg = self.get_arg_value_safe("items_per_thread")
        if items_per_thread_kwarg is not None:
            raise RuntimeError(
                "coop.block.shuffle does not accept items_per_thread for scalar shuffles"
            )

        distance = bound.get("distance")
        distance_var = None
        distance_value = None
        distance_literal = None
        if distance is not None:
            if isinstance(distance, ir.Var):
                distance_var = distance
                distance_var_ty = self.typemap.get(distance.name)
                if isinstance(distance_var_ty, types.IntegerLiteral):
                    distance_literal = int(distance_var_ty.literal_value)
                else:
                    distance_value = self.get_arg_value_safe("distance")
                    if isinstance(distance_value, int):
                        distance_var = None
            elif isinstance(distance, ir.Const):
                distance_value = distance.value
            else:
                distance_value = distance
        else:
            distance_value = 1

        if block_shuffle_type in (BlockShuffleType.Up, BlockShuffleType.Down):
            if distance_var is not None and distance_literal is None:
                raise RuntimeError(
                    "coop.block.shuffle requires distance to be a compile-time constant for Up/Down"
                )
            if distance_var is None and distance_literal is None:
                if distance_value is None or not isinstance(distance_value, int):
                    raise RuntimeError(
                        "coop.block.shuffle requires distance to be a compile-time constant for Up/Down"
                    )

        if distance_literal is not None:
            distance_value = distance_literal
        impl_shuffle_type = block_shuffle_type
        if block_shuffle_type in (BlockShuffleType.Up, BlockShuffleType.Down):
            distance_value = int(distance_value)
            if distance_value < 0:
                raise RuntimeError(
                    "coop.block.shuffle requires distance >= 0 for Up/Down"
                )
            impl_shuffle_type = BlockShuffleType.Offset
            if block_shuffle_type == BlockShuffleType.Up:
                distance_value = -distance_value

        if distance_var is None or distance_literal is not None:
            scope = self.instr.target.scope
            const_name = f"$block_shuffle_distance_{self.unique_id}"
            const_var = ir.Var(scope, const_name, self.expr.loc)
            if const_name in self.typemap:
                raise RuntimeError(f"Variable {const_name} already exists in typemap.")
            const_assign = ir.Assign(
                value=ir.Const(int(distance_value), self.expr.loc),
                target=const_var,
                loc=self.expr.loc,
            )
            self.typemap[const_name] = types.IntegerLiteral(int(distance_value))
            self.distance_assign = const_assign
            distance_var = const_var

        temp_storage = bound.get("temp_storage")
        temp_storage_info = None
        if temp_storage is not None:
            if not isinstance(temp_storage, ir.Var):
                raise RuntimeError(
                    "coop.block.shuffle temp_storage must be provided as a variable"
                )
            (_, _, temp_storage_info) = rewriter.bind_temp_storage_runtime_arg(
                node=self,
                temp_storage=temp_storage,
                runtime_args=runtime_args,
                runtime_arg_types=runtime_arg_types,
                runtime_arg_names=runtime_arg_names,
                insert_pos=0,
            )

        runtime_args.append(items)
        runtime_arg_types.append(items_ty)
        runtime_arg_names.append("input_item")
        runtime_args.append(distance_var)
        runtime_arg_types.append(types.int32)
        runtime_arg_names.append("distance")

        self.return_type = items_ty
        self.items_per_thread = None
        self.block_shuffle_type = impl_shuffle_type
        self.distance = distance
        self.temp_storage_info = temp_storage_info
        self.temp_storage = temp_storage
        self.item_dtype = items_ty
        self.methods = None

        self.impl_kwds = {
            "block_shuffle_type": impl_shuffle_type,
            "dtype": items_ty,
            "threads_per_block": self.threads_per_block,
            "items_per_thread": None,
            "distance": distance,
            "methods": None,
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
        distance_assign = getattr(self, "distance_assign", None)
        if distance_assign is not None:
            instrs.append(distance_assign)
        instrs.extend([rd.g_assign, rd.new_assign])
        if self.temp_storage_info is not None and self.temp_storage_info.auto_sync:
            instrs.extend(
                rewriter.emit_syncthreads_call(self.instr.target.scope, self.expr.loc)
            )
        return tuple(instrs)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()
