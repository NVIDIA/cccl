# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from functools import cached_property

from numba.core import types

import cuda.coop._rewrite as _core

ArrayCallDefinition = _core.ArrayCallDefinition
CoopNode = _core.CoopNode
CoopNodeMixin = _core.CoopNodeMixin
Disposition = _core.Disposition
ir = _core.ir


# =============================================================================
# Block exchange
# =============================================================================
class CoopBlockExchangeNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.block.exchange"
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
        if items is None:
            raise RuntimeError("coop.block.exchange requires an items argument")
        if not isinstance(items, ir.Var):
            raise RuntimeError("coop.block.exchange items must be a variable")

        output_items = bound.get("output_items")
        ranks = bound.get("ranks")
        valid_flags = bound.get("valid_flags")

        items_ty = self.typemap[items.name]
        try:
            from ..._decls import ThreadDataType
        except Exception:
            ThreadDataType = None

        items_is_thread = ThreadDataType is not None and isinstance(
            items_ty, ThreadDataType
        )
        if not items_is_thread and not isinstance(items_ty, types.Array):
            raise RuntimeError(
                "coop.block.exchange requires items to be an array or ThreadData"
            )

        output_items_ty = None
        output_is_thread = False
        if output_items is not None:
            if not isinstance(output_items, ir.Var):
                raise RuntimeError(
                    "coop.block.exchange output_items must be a variable"
                )
            output_items_ty = self.typemap[output_items.name]
            output_is_thread = ThreadDataType is not None and isinstance(
                output_items_ty, ThreadDataType
            )
            if not output_is_thread and not isinstance(output_items_ty, types.Array):
                raise RuntimeError(
                    "coop.block.exchange requires output_items to be an array "
                    "or ThreadData"
                )

        items_root = rewriter.get_root_def(items)
        items_leaf = items_root.leaf_constructor_call
        if items_is_thread:
            items_per_thread = rewriter.get_thread_data_info(items).items_per_thread
        elif not isinstance(items_leaf, ArrayCallDefinition):
            raise RuntimeError(
                f"Expected items constructor call to be an ArrayCallDefinition, "
                f"but got {items_leaf!r} for {items!r}"
            )
        else:
            items_per_thread = items_leaf.shape
        if isinstance(items_per_thread, types.IntegerLiteral):
            items_per_thread = items_per_thread.literal_value
        assert isinstance(items_per_thread, int), (
            f"Expected items_per_thread to be an int, got {items_per_thread!r}"
        )

        if output_items is not None:
            if output_is_thread:
                output_info = rewriter.get_thread_data_info(output_items)
                if output_info.items_per_thread != items_per_thread:
                    raise RuntimeError(
                        "coop.block.exchange requires items and output_items to "
                        "have the same items_per_thread"
                    )
            else:
                output_root = rewriter.get_root_def(output_items)
                output_leaf = output_root.leaf_constructor_call
                if not isinstance(output_leaf, ArrayCallDefinition):
                    raise RuntimeError(
                        "Expected output_items constructor call to be an "
                        f"ArrayCallDefinition, but got {output_leaf!r} for "
                        f"{output_items!r}"
                    )
                if output_leaf.shape != items_per_thread:
                    raise RuntimeError(
                        "coop.block.exchange requires items and output_items to "
                        "have the same items_per_thread"
                    )

        if items_is_thread:
            dtype = rewriter.get_thread_data_info(items).dtype
        else:
            dtype = items_ty.dtype

        if output_items is not None:
            if output_is_thread:
                output_info = rewriter.get_thread_data_info(output_items)
                if output_info.dtype != dtype:
                    raise RuntimeError(
                        "coop.block.exchange requires items and output_items to "
                        "have the same dtype"
                    )
            else:
                if output_items_ty.dtype != dtype:
                    raise RuntimeError(
                        "coop.block.exchange requires items and output_items to "
                        "have the same dtype"
                    )
        methods = getattr(dtype, "methods", None)
        if methods is not None and not methods:
            methods = None
        if methods is not None and items_per_thread > 1:
            raise RuntimeError(
                "coop.block.exchange only supports user-defined types when "
                "items_per_thread == 1"
            )

        block_exchange_type = self.get_arg_value_safe("block_exchange_type")
        if block_exchange_type is None:
            from cuda.coop.block._block_exchange import BlockExchangeType

            block_exchange_type = BlockExchangeType.StripedToBlocked
        else:
            from cuda.coop.block._block_exchange import BlockExchangeType

            if isinstance(block_exchange_type, types.EnumMember):
                literal_value = getattr(block_exchange_type, "literal_value", None)
                if literal_value is None:
                    literal_value = block_exchange_type.value
                block_exchange_type = block_exchange_type.instance_class(literal_value)
            if isinstance(block_exchange_type, int):
                block_exchange_type = BlockExchangeType(block_exchange_type)
            if block_exchange_type not in BlockExchangeType:
                raise RuntimeError(
                    "coop.block.exchange requires block_exchange_type to be a "
                    "BlockExchangeType enum value"
                )

        uses_ranks = block_exchange_type in (
            BlockExchangeType.ScatterToBlocked,
            BlockExchangeType.ScatterToStriped,
            BlockExchangeType.ScatterToStripedGuarded,
            BlockExchangeType.ScatterToStripedFlagged,
        )
        uses_valid_flags = (
            block_exchange_type == BlockExchangeType.ScatterToStripedFlagged
        )

        ranks_ty = None
        ranks_is_thread = False
        ranks_dtype = None
        if uses_ranks:
            if ranks is None:
                raise RuntimeError(
                    "coop.block.exchange requires ranks for scatter exchanges"
                )
            if not isinstance(ranks, ir.Var):
                raise RuntimeError("coop.block.exchange ranks must be a variable")
            ranks_ty = self.typemap[ranks.name]
            ranks_is_thread = ThreadDataType is not None and isinstance(
                ranks_ty, ThreadDataType
            )
            if not ranks_is_thread and not isinstance(ranks_ty, types.Array):
                raise RuntimeError("coop.block.exchange requires ranks to be an array")
            if ranks_is_thread:
                ranks_info = rewriter.get_thread_data_info(ranks)
                ranks_items_per_thread = ranks_info.items_per_thread
                ranks_dtype = ranks_info.dtype
            else:
                ranks_root = rewriter.get_root_def(ranks)
                ranks_leaf = ranks_root.leaf_constructor_call
                if not isinstance(ranks_leaf, ArrayCallDefinition):
                    raise RuntimeError(
                        "Expected ranks constructor call to be an ArrayCallDefinition, "
                        f"but got {ranks_leaf!r} for {ranks!r}"
                    )
                ranks_items_per_thread = ranks_leaf.shape
                ranks_dtype = ranks_ty.dtype
            if not isinstance(ranks_dtype, types.Integer):
                raise RuntimeError(
                    "coop.block.exchange requires ranks to be an integer array"
                )
            if ranks_items_per_thread != items_per_thread:
                raise RuntimeError(
                    "coop.block.exchange requires ranks to have the same "
                    "items_per_thread as items"
                )
        elif ranks is not None:
            raise RuntimeError(
                "coop.block.exchange ranks are only valid for scatter exchanges"
            )

        valid_flags_ty = None
        valid_flags_is_thread = False
        valid_flags_dtype = None
        if uses_valid_flags:
            if valid_flags is None:
                raise RuntimeError(
                    "coop.block.exchange requires valid_flags for "
                    "ScatterToStripedFlagged"
                )
            if not isinstance(valid_flags, ir.Var):
                raise RuntimeError("coop.block.exchange valid_flags must be a variable")
            valid_flags_ty = self.typemap[valid_flags.name]
            valid_flags_is_thread = ThreadDataType is not None and isinstance(
                valid_flags_ty, ThreadDataType
            )
            if not valid_flags_is_thread and not isinstance(
                valid_flags_ty, types.Array
            ):
                raise RuntimeError(
                    "coop.block.exchange requires valid_flags to be an array"
                )
            if valid_flags_is_thread:
                valid_flags_info = rewriter.get_thread_data_info(valid_flags)
                valid_flags_items_per_thread = valid_flags_info.items_per_thread
                valid_flags_dtype = valid_flags_info.dtype
            else:
                valid_flags_root = rewriter.get_root_def(valid_flags)
                valid_flags_leaf = valid_flags_root.leaf_constructor_call
                if not isinstance(valid_flags_leaf, ArrayCallDefinition):
                    raise RuntimeError(
                        "Expected valid_flags constructor call to be an "
                        f"ArrayCallDefinition, but got {valid_flags_leaf!r} for "
                        f"{valid_flags!r}"
                    )
                valid_flags_items_per_thread = valid_flags_leaf.shape
                valid_flags_dtype = valid_flags_ty.dtype
            if not isinstance(valid_flags_dtype, (types.Integer, types.Boolean)):
                raise RuntimeError(
                    "coop.block.exchange requires valid_flags to be a boolean "
                    "or integer array"
                )
            if valid_flags_items_per_thread != items_per_thread:
                raise RuntimeError(
                    "coop.block.exchange requires valid_flags to have the same "
                    "items_per_thread as items"
                )
        elif valid_flags is not None:
            raise RuntimeError(
                "coop.block.exchange valid_flags are only valid for "
                "ScatterToStripedFlagged"
            )

        items_per_thread_kwarg = self.get_arg_value_safe("items_per_thread")
        if items_per_thread_kwarg is not None:
            if items_per_thread_kwarg != items_per_thread:
                raise RuntimeError(
                    "coop.block.exchange items_per_thread must match the "
                    f"items array shape ({items_per_thread}); got "
                    f"{items_per_thread_kwarg}"
                )

        warp_time_slicing = self.get_arg_value_safe("warp_time_slicing")
        if warp_time_slicing is None:
            warp_time_slicing = False
        if not isinstance(warp_time_slicing, bool):
            raise RuntimeError(
                "coop.block.exchange requires warp_time_slicing to be a boolean"
            )

        temp_storage = bound.get("temp_storage")
        temp_storage_info = None
        if temp_storage is not None:
            if not isinstance(temp_storage, ir.Var):
                raise RuntimeError(
                    "coop.block.exchange temp_storage must be provided as a variable"
                )
            (_, _, temp_storage_info) = rewriter.bind_temp_storage_runtime_arg(
                node=self,
                temp_storage=temp_storage,
                runtime_args=runtime_args,
                runtime_arg_types=runtime_arg_types,
                runtime_arg_names=runtime_arg_names,
                insert_pos=0,
            )

        if ThreadDataType is not None:
            array_ty = types.Array(dtype, 1, "C")
            if items_is_thread:
                items_ty = array_ty
            if output_items is not None and output_is_thread:
                output_items_ty = array_ty
            if uses_ranks and ranks_is_thread:
                ranks_ty = types.Array(ranks_dtype, 1, "C")
            if uses_valid_flags and valid_flags_is_thread:
                valid_flags_ty = types.Array(valid_flags_dtype, 1, "C")

        if output_items is None:
            runtime_args.append(items)
            runtime_arg_types.append(items_ty)
            runtime_arg_names.append("input_items")
        else:
            runtime_args.append(items)
            runtime_arg_types.append(items_ty)
            runtime_arg_names.append("input_items")
            runtime_args.append(output_items)
            runtime_arg_types.append(output_items_ty)
            runtime_arg_names.append("output_items")
        if uses_ranks:
            runtime_args.append(ranks)
            runtime_arg_types.append(ranks_ty)
            runtime_arg_names.append("ranks")
        if uses_valid_flags:
            runtime_args.append(valid_flags)
            runtime_arg_types.append(valid_flags_ty)
            runtime_arg_names.append("valid_flags")

        self.impl_kwds = {
            "block_exchange_type": block_exchange_type,
            "dtype": dtype,
            "threads_per_block": self.threads_per_block,
            "items_per_thread": items_per_thread,
            "warp_time_slicing": warp_time_slicing,
            "methods": methods,
            "unique_id": self.unique_id,
            "temp_storage": temp_storage,
            "use_output_items": output_items is not None,
            "offset_dtype": ranks_dtype if uses_ranks else None,
            "valid_flag_dtype": valid_flags_dtype if uses_valid_flags else None,
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
            needs_temp_storage = (
                temp_storage is not None
                and getattr(instance, "temp_storage", None) is None
            )
            if needs_temp_storage:
                self.instance = self.instantiate_impl(**self.impl_kwds)

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        instrs = []
        swap_assign = getattr(self, "swap_assign", None)
        swap_target = getattr(self, "swap_target", None)
        if swap_assign is not None:
            instrs.append(swap_assign)
        instrs.append(rd.g_assign)
        if swap_target is not None:
            rd.new_call.args = (swap_target,) + rd.new_call.args[1:]
        instrs.append(rd.new_assign)
        temp_storage_info = getattr(self, "temp_storage_info", None)
        if temp_storage_info is not None and temp_storage_info.auto_sync:
            instrs.extend(
                rewriter.emit_syncthreads_call(self.instr.target.scope, self.expr.loc)
            )
        return tuple(instrs)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()
