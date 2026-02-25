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
# Radix Rank
# =============================================================================
@dataclass
class CoopBlockRadixRankNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.block.radix_rank"
    disposition = Disposition.ONE_SHOT

    def refine_match(self, rewriter):
        self.threads_per_block = self.resolve_threads_per_block()

        runtime_args = []
        runtime_arg_types = []
        runtime_arg_names = []

        bound = self.bound.arguments
        items = bound.get("items")
        ranks = bound.get("ranks")
        exclusive_digit_prefix = bound.get("exclusive_digit_prefix")
        if items is None or ranks is None:
            raise RuntimeError(
                "coop.block.radix_rank requires items and ranks arguments"
            )
        if not isinstance(items, ir.Var):
            raise RuntimeError("coop.block.radix_rank items must be a variable")
        if not isinstance(ranks, ir.Var):
            raise RuntimeError("coop.block.radix_rank ranks must be a variable")

        items_ty = self.typemap[items.name]
        ranks_ty = self.typemap[ranks.name]
        thread_data_type = get_thread_data_type()
        items_is_thread = isinstance(items_ty, thread_data_type)
        ranks_is_thread = isinstance(ranks_ty, thread_data_type)

        if not items_is_thread and not isinstance(items_ty, types.Array):
            raise RuntimeError(
                "coop.block.radix_rank requires items to be an array or ThreadData"
            )
        if not ranks_is_thread and not isinstance(ranks_ty, types.Array):
            raise RuntimeError(
                "coop.block.radix_rank requires ranks to be an array or ThreadData"
            )
        if exclusive_digit_prefix is not None and not isinstance(
            exclusive_digit_prefix, ir.Var
        ):
            raise RuntimeError(
                "coop.block.radix_rank exclusive_digit_prefix must be a variable"
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
        ranks_items_per_thread = _infer_items_per_thread(ranks, ranks_is_thread)
        if items_per_thread != ranks_items_per_thread:
            raise RuntimeError(
                "coop.block.radix_rank requires items and ranks to have the same "
                "items_per_thread"
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
                    "coop.block.radix_rank items_per_thread must match the array "
                    f"shape ({items_per_thread}); got {items_per_thread_kwarg}"
                )

        if items_is_thread:
            item_dtype = rewriter.get_thread_data_info(items).dtype
        else:
            item_dtype = items_ty.dtype

        if ranks_is_thread:
            ranks_dtype = rewriter.get_thread_data_info(ranks).dtype
        else:
            ranks_dtype = ranks_ty.dtype

        if not isinstance(item_dtype, types.Integer) or item_dtype.signed:
            raise RuntimeError("coop.block.radix_rank requires unsigned integer items")
        if not isinstance(ranks_dtype, types.Integer) or ranks_dtype.bitwidth != 32:
            raise RuntimeError("coop.block.radix_rank requires int32 ranks arrays")

        begin_bit = int(self.get_arg_value("begin_bit"))
        end_bit = int(self.get_arg_value("end_bit"))
        if end_bit <= begin_bit:
            raise RuntimeError("coop.block.radix_rank requires end_bit > begin_bit")

        block_threads = (
            self.threads_per_block[0]
            * self.threads_per_block[1]
            * self.threads_per_block[2]
        )
        radix_bits = end_bit - begin_bit
        radix_digits = 1 << radix_bits
        bins_per_thread = max(1, (radix_digits + block_threads - 1) // block_threads)

        descending = bound.get("descending")
        if descending is None:
            descending_value = False
        else:
            descending_value = self.get_arg_value_safe("descending")
            if descending_value is None:
                raise RuntimeError(
                    "coop.block.radix_rank requires descending to be a compile-time "
                    "boolean"
                )
            if not isinstance(descending_value, bool):
                raise RuntimeError(
                    "coop.block.radix_rank requires descending to be a boolean"
                )

        temp_storage = bound.get("temp_storage")
        temp_storage_info = None
        if temp_storage is not None:
            if not isinstance(temp_storage, ir.Var):
                raise RuntimeError(
                    "coop.block.radix_rank temp_storage must be provided as a variable"
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
        array_ranks_ty = types.Array(ranks_dtype, 1, "C")

        runtime_args.extend([items, ranks])
        runtime_arg_types.extend(
            [
                array_items_ty if items_is_thread else items_ty,
                array_ranks_ty if ranks_is_thread else ranks_ty,
            ]
        )
        runtime_arg_names.extend(["items", "ranks"])

        if exclusive_digit_prefix is not None:
            prefix_ty = self.typemap[exclusive_digit_prefix.name]
            prefix_is_thread = isinstance(prefix_ty, thread_data_type)
            if not prefix_is_thread and not isinstance(prefix_ty, types.Array):
                raise RuntimeError(
                    "coop.block.radix_rank requires exclusive_digit_prefix to be an "
                    "array or ThreadData"
                )

            prefix_items_per_thread = _infer_items_per_thread(
                exclusive_digit_prefix, prefix_is_thread
            )
            if prefix_items_per_thread != bins_per_thread:
                raise RuntimeError(
                    "coop.block.radix_rank exclusive_digit_prefix must have "
                    f"{bins_per_thread} items per thread; got {prefix_items_per_thread}"
                )

            if prefix_is_thread:
                prefix_dtype = rewriter.get_thread_data_info(
                    exclusive_digit_prefix
                ).dtype
            else:
                prefix_dtype = prefix_ty.dtype
            if (
                not isinstance(prefix_dtype, types.Integer)
                or prefix_dtype.bitwidth != 32
            ):
                raise RuntimeError(
                    "coop.block.radix_rank requires exclusive_digit_prefix to be an "
                    "int32 array"
                )

            array_prefix_ty = types.Array(prefix_dtype, 1, "C")
            runtime_args.append(exclusive_digit_prefix)
            runtime_arg_types.append(array_prefix_ty if prefix_is_thread else prefix_ty)
            runtime_arg_names.append("exclusive_digit_prefix")

        self.impl_kwds = {
            "dtype": item_dtype,
            "threads_per_block": self.threads_per_block,
            "items_per_thread": items_per_thread,
            "begin_bit": begin_bit,
            "end_bit": end_bit,
            "descending": descending_value,
            "exclusive_digit_prefix": exclusive_digit_prefix,
            "unique_id": self.unique_id,
            "temp_storage": temp_storage,
            "node": self,
        }

        self.return_type = types.void
        self.runtime_args = runtime_args
        self.runtime_arg_types = runtime_arg_types
        self.runtime_arg_names = runtime_arg_names
        self.temp_storage_info = temp_storage_info
        self.temp_storage = temp_storage

        if self.is_two_phase and self.two_phase_instance is not None:
            instance = self.two_phase_instance
            needs_prefix = (
                exclusive_digit_prefix is not None
                and getattr(instance, "exclusive_digit_prefix", None) is None
            )
            if needs_prefix:
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
