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
# Warp Exchange
# =============================================================================
@dataclass
class CoopWarpExchangeNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.warp.exchange"
    disposition = Disposition.ONE_SHOT

    def refine_match(self, rewriter):
        runtime_args = []
        runtime_arg_types = []
        runtime_arg_names = []

        bound = self.bound.arguments
        items = bound.get("items")
        if items is None or not isinstance(items, ir.Var):
            raise RuntimeError("coop.warp.exchange requires items to be a variable")

        output_items = bound.get("output_items")
        ranks = bound.get("ranks")

        items_ty = self.typemap[items.name]
        thread_data_type = get_thread_data_type()
        items_is_thread = isinstance(items_ty, thread_data_type)
        if not items_is_thread and not isinstance(items_ty, types.Array):
            raise RuntimeError(
                "coop.warp.exchange requires items to be an array or ThreadData"
            )

        output_items_ty = None
        output_is_thread = False
        if output_items is not None:
            if not isinstance(output_items, ir.Var):
                raise RuntimeError("coop.warp.exchange output_items must be a variable")
            output_items_ty = self.typemap[output_items.name]
            output_is_thread = isinstance(output_items_ty, thread_data_type)
            if not output_is_thread and not isinstance(output_items_ty, types.Array):
                raise RuntimeError(
                    "coop.warp.exchange requires output_items to be an array or "
                    "ThreadData"
                )

        def _infer_items_per_thread(var, is_thread, name):
            if is_thread:
                return rewriter.get_thread_data_info(var).items_per_thread
            root = rewriter.get_root_def(var)
            leaf = root.leaf_constructor_call
            if not isinstance(leaf, ArrayCallDefinition):
                raise RuntimeError(
                    "coop.warp.exchange requires "
                    f"{name} to be a local array or ThreadData"
                )
            return leaf.shape

        items_per_thread = _infer_items_per_thread(items, items_is_thread, "items")
        if isinstance(items_per_thread, types.IntegerLiteral):
            items_per_thread = items_per_thread.literal_value
        if not isinstance(items_per_thread, int):
            raise RuntimeError(
                f"Expected items_per_thread to be an int, got {items_per_thread!r}"
            )

        if output_items is not None:
            output_items_per_thread = _infer_items_per_thread(
                output_items, output_is_thread, "output_items"
            )
            if output_items_per_thread != items_per_thread:
                raise RuntimeError(
                    "coop.warp.exchange requires items and output_items to have "
                    "the same items_per_thread"
                )

        items_per_thread_kwarg = self.get_arg_value_safe("items_per_thread")
        if (
            items_per_thread_kwarg is not None
            and items_per_thread_kwarg != items_per_thread
        ):
            raise RuntimeError(
                f"coop.warp.exchange items_per_thread must match array shape "
                f"({items_per_thread}); got {items_per_thread_kwarg}"
            )

        if items_is_thread:
            dtype = rewriter.get_thread_data_info(items).dtype
        else:
            dtype = items_ty.dtype

        if output_items_ty is not None:
            if output_is_thread:
                output_dtype = rewriter.get_thread_data_info(output_items).dtype
            else:
                output_dtype = output_items_ty.dtype
            if output_dtype != dtype:
                raise RuntimeError(
                    "coop.warp.exchange requires items and output_items to have "
                    "the same dtype"
                )

        methods = getattr(dtype, "methods", None)
        if methods is not None and not methods:
            methods = None

        warp_exchange_type = self.get_arg_value_safe("warp_exchange_type")
        if warp_exchange_type is None:
            from cuda.coop.warp._warp_exchange import WarpExchangeType

            warp_exchange_type = WarpExchangeType.StripedToBlocked
        else:
            from cuda.coop.warp._warp_exchange import WarpExchangeType

            if isinstance(warp_exchange_type, types.EnumMember):
                literal_value = getattr(warp_exchange_type, "literal_value", None)
                if literal_value is None:
                    literal_value = warp_exchange_type.value
                warp_exchange_type = warp_exchange_type.instance_class(literal_value)
            if isinstance(warp_exchange_type, int):
                warp_exchange_type = WarpExchangeType(warp_exchange_type)
            if warp_exchange_type not in WarpExchangeType:
                raise RuntimeError(
                    "coop.warp.exchange requires warp_exchange_type to be a "
                    "WarpExchangeType enum value"
                )

        # Keep BlockedToStriped behavior; handled by coop intrinsic.

        threads_in_warp = self.get_arg_value_safe("threads_in_warp")
        threads_in_warp_arg = self.bound.arguments.get("threads_in_warp")
        if threads_in_warp is None and threads_in_warp_arg is not None:
            raise RuntimeError("threads_in_warp must be a compile-time constant")
        if threads_in_warp is None:
            threads_in_warp = 32
        if not isinstance(threads_in_warp, int) or threads_in_warp < 1:
            raise RuntimeError("threads_in_warp must be a positive integer")

        uses_ranks = warp_exchange_type == WarpExchangeType.ScatterToStriped

        ranks_ty = None
        ranks_is_thread = False
        ranks_dtype = None
        offset_dtype = None
        if uses_ranks:
            if ranks is None:
                raise RuntimeError(
                    "coop.warp.exchange requires ranks for ScatterToStriped"
                )
            if not isinstance(ranks, ir.Var):
                raise RuntimeError("coop.warp.exchange ranks must be a variable")
            ranks_ty = self.typemap[ranks.name]
            ranks_is_thread = isinstance(ranks_ty, thread_data_type)
            if not ranks_is_thread and not isinstance(ranks_ty, types.Array):
                raise RuntimeError(
                    "coop.warp.exchange requires ranks to be an array or ThreadData"
                )
            if ranks_is_thread:
                ranks_dtype = rewriter.get_thread_data_info(ranks).dtype
            else:
                ranks_dtype = ranks_ty.dtype
            if not isinstance(ranks_dtype, types.Integer):
                raise RuntimeError(
                    "coop.warp.exchange requires ranks to be an integer array"
                )
            ranks_items_per_thread = _infer_items_per_thread(
                ranks, ranks_is_thread, "ranks"
            )
            if ranks_items_per_thread != items_per_thread:
                raise RuntimeError(
                    "coop.warp.exchange requires ranks to have the same "
                    "items_per_thread as items"
                )
            offset_dtype = ranks_dtype
        elif ranks is not None:
            raise RuntimeError(
                "coop.warp.exchange ranks are only valid for ScatterToStriped"
            )

        offset_dtype_arg = self.get_arg_value_safe("offset_dtype")
        if offset_dtype_arg is not None:
            if not uses_ranks:
                raise RuntimeError(
                    "coop.warp.exchange offset_dtype is only valid for ScatterToStriped"
                )
            offset_dtype = offset_dtype_arg

        temp_storage = bound.get("temp_storage")
        temp_storage_info = None
        if temp_storage is not None:
            if not isinstance(temp_storage, ir.Var):
                raise RuntimeError(
                    "coop.warp.exchange temp_storage must be provided as a variable"
                )
            (_, _, temp_storage_info) = rewriter.bind_temp_storage_runtime_arg(
                node=self,
                temp_storage=temp_storage,
                runtime_args=runtime_args,
                runtime_arg_types=runtime_arg_types,
                runtime_arg_names=runtime_arg_names,
                insert_pos=0,
            )

        array_items_ty = types.Array(dtype, 1, "C")
        runtime_items_ty = array_items_ty if items_is_thread else items_ty
        runtime_output_items_ty = (
            array_items_ty if output_is_thread else output_items_ty
        )

        if output_items is None:
            runtime_args.append(items)
            runtime_arg_types.append(runtime_items_ty)
            runtime_arg_names.append("input_items")
        else:
            runtime_args.append(items)
            runtime_arg_types.append(runtime_items_ty)
            runtime_arg_names.append("input_items")
            runtime_args.append(output_items)
            runtime_arg_types.append(runtime_output_items_ty)
            runtime_arg_names.append("output_items")

        if uses_ranks:
            runtime_ranks_ty = (
                types.Array(ranks_dtype, 1, "C") if ranks_is_thread else ranks_ty
            )
            runtime_args.append(ranks)
            runtime_arg_types.append(runtime_ranks_ty)
            runtime_arg_names.append("ranks")

        self.impl_kwds = {
            "dtype": dtype,
            "items_per_thread": items_per_thread,
            "threads_in_warp": threads_in_warp,
            "warp_exchange_type": warp_exchange_type,
            "offset_dtype": offset_dtype,
            "methods": methods,
            "unique_id": self.unique_id,
            "temp_storage": temp_storage,
            "node": self,
        }

        if (
            warp_exchange_type == WarpExchangeType.BlockedToStriped
            and output_items is not None
        ):
            scope = self.instr.target.scope
            temp_name = f"$warp_exchange_swap_{self.unique_id}"
            temp_var = ir.Var(scope, temp_name, self.expr.loc)
            if temp_name in self.typemap:
                raise RuntimeError(f"Variable {temp_name} already exists in typemap.")
            self.typemap[temp_name] = runtime_items_ty
            self.swap_assign = ir.Assign(
                value=items, target=temp_var, loc=self.expr.loc
            )
            self.swap_target = temp_var

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
        instrs = [rd.g_assign]
        valid_items_assign = getattr(self, "valid_items_assign", None)
        if valid_items_assign is not None:
            instrs.append(valid_items_assign)
        instrs.append(rd.new_assign)
        return tuple(instrs)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()
